#!/usr/bin/env python3
"""Re-score a fixed OPD context bank with HF teacher/student models."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported context-bank input: {path}")


def _write_table(df: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix.lower()
    if suffix == ".csv":
        df.to_csv(output, index=False)
    elif suffix == ".jsonl":
        with output.open("w", encoding="utf-8") as f:
            for row in df.to_dict(orient="records"):
                f.write(json.dumps(row, ensure_ascii=True, allow_nan=True) + "\n")
    elif suffix == ".parquet":
        df.to_parquet(output, index=False)
    else:
        raise ValueError("Output must end with .csv, .jsonl, or .parquet")


def _loads_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value:
        return json.loads(value)
    return []


def _dtype(name: str):
    return {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }[name]


def _load_model(path: str, device: str, dtype_name: str, trust_remote_code: bool):
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=_dtype(dtype_name),
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    return model


def _topk_metrics(logps_s: torch.Tensor, logps_t: torch.Tensor, token_id: int, k: int) -> dict[str, Any]:
    probs_s = logps_s.exp()
    probs_t = logps_t.exp()
    top_s_logp, top_s_ids = torch.topk(logps_s, k=k)
    top_t_logp, top_t_ids = torch.topk(logps_t, k=k)
    s_ids = [int(x) for x in top_s_ids.tolist()]
    t_ids = [int(x) for x in top_t_ids.tolist()]
    set_s = set(s_ids)
    set_t = set(t_ids)
    union_ids = sorted(set_s | set_t)
    union = torch.tensor(union_ids, device=logps_s.device, dtype=torch.long)

    ps_u = probs_s[union]
    pt_u = probs_t[union]
    ps_u = ps_u / ps_u.sum().clamp_min(1e-12)
    pt_u = pt_u / pt_u.sum().clamp_min(1e-12)
    klf_union = float((pt_u * (pt_u.clamp_min(1e-12).log() - ps_u.clamp_min(1e-12).log())).sum().item())
    klr_union = float((ps_u * (ps_u.clamp_min(1e-12).log() - pt_u.clamp_min(1e-12).log())).sum().item())
    cbc = float(torch.sqrt(ps_u * pt_u).sum().item())
    teacher_top1 = t_ids[0] if t_ids else None
    student_top1 = s_ids[0] if s_ids else None

    return {
        "student_top1_id": student_top1,
        "teacher_top1_id": teacher_top1,
        "student_top1_prob": float(probs_s[student_top1].item()) if student_top1 is not None else 0.0,
        "teacher_top1_prob": float(probs_t[teacher_top1].item()) if teacher_top1 is not None else 0.0,
        "Cmass_true": float(probs_t[top_s_ids].sum().item()),
        "Coverlap": len(set_s & set_t) / max(k, 1),
        "CBC": cbc,
        "KLf_union": klf_union,
        "KLr_union": klr_union,
        "reach_t1": int(teacher_top1 in set_s) if teacher_top1 is not None else 0,
        "target_in_student_topk": int(token_id in set_s),
        "target_in_teacher_topk": int(token_id in set_t),
        "target_student_rank": s_ids.index(token_id) + 1 if token_id in set_s else None,
        "target_teacher_rank": t_ids.index(token_id) + 1 if token_id in set_t else None,
        "student_top_ids": json.dumps(s_ids),
        "student_top_logps": json.dumps([float(x) for x in top_s_logp.tolist()]),
        "teacher_top_ids": json.dumps(t_ids),
        "teacher_top_logps": json.dumps([float(x) for x in top_t_logp.tolist()]),
    }


def _score_sample(
    row: pd.Series,
    sample_ordinal: int,
    student,
    teacher,
    student_device: str,
    teacher_device: str,
    topk: int,
    max_response_tokens: int | None,
) -> list[dict[str, Any]]:
    tokens = [int(x) for x in _loads_list(row["tokens"])]
    prompt_len = int(row["prompt_len"])
    resp_len = int(row["resp_len"])
    if max_response_tokens is not None:
        resp_len = min(resp_len, max_response_tokens)
    if resp_len <= 0:
        return []

    input_s = torch.tensor([tokens], dtype=torch.long, device=student_device)
    input_t = input_s.to(teacher_device)
    with torch.no_grad():
        logits_s = student(input_ids=input_s).logits[0].float().cpu()
        logits_t = teacher(input_ids=input_t).logits[0].float().cpu()

    out = []
    for tok_pos in range(resp_len):
        seq_pos = prompt_len + tok_pos
        if seq_pos <= 0 or seq_pos >= len(tokens):
            continue
        pred_pos = seq_pos - 1
        token_id = int(tokens[seq_pos])
        logps_s = torch.log_softmax(logits_s[pred_pos], dim=-1)
        logps_t = torch.log_softmax(logits_t[pred_pos], dim=-1)
        probs_s = logps_s.exp()
        probs_t = logps_t.exp()
        hs = float(-(probs_s * logps_s).sum().item())
        ht = float(-(probs_t * logps_t).sum().item())
        klr = float((probs_s * (logps_s - logps_t)).sum().item())
        klf = float((probs_t * (logps_t - logps_s)).sum().item())
        metrics = _topk_metrics(logps_s, logps_t, token_id, topk)
        out.append(
            {
                "sample_ordinal": sample_ordinal,
                "sample_index": row.get("sample_index"),
                "group_index": row.get("group_index"),
                "tok_pos": tok_pos,
                "pos_norm": tok_pos / max(resp_len - 1, 1),
                "prompt_len": prompt_len,
                "resp_len": int(row["resp_len"]),
                "total_len": int(row["total_len"]),
                "truncated": int(row.get("truncated", 0)),
                "token_id": token_id,
                "student_logp_sampled": float(logps_s[token_id].item()),
                "teacher_logp_sampled": float(logps_t[token_id].item()),
                "Hs_full": hs,
                "Ht_full": ht,
                "KLr_full": klr,
                "KLf_full": klf,
                **metrics,
            }
        )
    return out


def _merge_gain(metrics: pd.DataFrame, baseline_path: str | None) -> pd.DataFrame:
    if baseline_path is None:
        return metrics
    base = _read_table(Path(baseline_path))
    keys = ["sample_ordinal", "tok_pos"]
    if "sample_index" in metrics.columns and "sample_index" in base.columns:
        keys = ["sample_index", "tok_pos"]
    keep = keys + [col for col in ["KLf_full", "KLr_full", "Hs_full", "Ht_full"] if col in base.columns]
    base = base[keep].rename(
        columns={
            "KLf_full": "KLf_full_base",
            "KLr_full": "KLr_full_base",
            "Hs_full": "Hs_full_base",
            "Ht_full": "Ht_full_base",
        }
    )
    merged = metrics.merge(base, on=keys, how="left")
    eps = 1e-12
    if "KLf_full_base" in merged.columns:
        merged["G_KLf"] = (merged["KLf_full_base"] - merged["KLf_full"]) / (merged["KLf_full_base"] + eps)
    if "KLr_full_base" in merged.columns:
        merged["G_KLr"] = (merged["KLr_full_base"] - merged["KLr_full"]) / (merged["KLr_full_base"] + eps)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-bank", required=True)
    parser.add_argument("--student", required=True, help="HF student model path.")
    parser.add_argument("--teacher", required=True, help="HF teacher model path.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline-metrics", default=None, help="Optional step-0 metrics table for gain computation.")
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-response-tokens", type=int, default=None)
    parser.add_argument("--student-device", default="cuda:0")
    parser.add_argument("--teacher-device", default="cuda:1")
    parser.add_argument("--dtype", choices=["bfloat16", "bf16", "float16", "fp16", "float32", "fp32"], default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--allow-tokenizer-mismatch", action="store_true")
    args = parser.parse_args()

    print(f"loading_tokenizers student={args.student} teacher={args.teacher}", flush=True)
    tok_s = AutoTokenizer.from_pretrained(args.student, trust_remote_code=args.trust_remote_code)
    tok_t = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=args.trust_remote_code)
    print("tokenizers_loaded", flush=True)
    if not args.allow_tokenizer_mismatch and tok_s.get_vocab() != tok_t.get_vocab():
        raise ValueError("Tokenizer vocab mismatch. Use same-family pairs or pass --allow-tokenizer-mismatch knowingly.")

    print(f"loading_context_bank path={args.context_bank}", flush=True)
    contexts = _read_table(Path(args.context_bank))
    if args.max_samples is not None:
        contexts = contexts.head(args.max_samples)
    print(f"context_bank_loaded samples={len(contexts)}", flush=True)

    print(f"loading_student_model device={args.student_device} dtype={args.dtype}", flush=True)
    student = _load_model(args.student, args.student_device, args.dtype, args.trust_remote_code)
    print("student_model_loaded", flush=True)
    print(f"loading_teacher_model device={args.teacher_device} dtype={args.dtype}", flush=True)
    teacher = _load_model(args.teacher, args.teacher_device, args.dtype, args.trust_remote_code)
    print("teacher_model_loaded", flush=True)

    print("scoring_start", flush=True)
    rows = []
    for ordinal, row in contexts.reset_index(drop=True).iterrows():
        rows.extend(
            _score_sample(
                row=row,
                sample_ordinal=ordinal,
                student=student,
                teacher=teacher,
                student_device=args.student_device,
                teacher_device=args.teacher_device,
                topk=args.topk,
                max_response_tokens=args.max_response_tokens,
            )
        )
        if (ordinal + 1) % 10 == 0:
            print(f"scored_samples={ordinal + 1} rows={len(rows)}", flush=True)

    metrics = pd.DataFrame(rows)
    metrics = _merge_gain(metrics, args.baseline_metrics)
    _write_table(metrics, Path(args.output))
    print(f"samples={len(contexts)}", flush=True)
    print(f"tokens={len(metrics)}", flush=True)
    print(f"saved={args.output}", flush=True)
    if "G_KLf" in metrics.columns:
        finite = metrics["G_KLf"].replace([math.inf, -math.inf], math.nan).dropna()
        print(f"mean_G_KLf={float(finite.mean()) if len(finite) else float('nan'):.6f}", flush=True)


if __name__ == "__main__":
    main()

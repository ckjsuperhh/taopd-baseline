import csv
import json
import math
import random
from pathlib import Path
from typing import Any


_ROLLOUT_COUNTER = 0
_CSV_FIELDS = [
    "schema_version",
    "pair_id",
    "teacher_name",
    "student_name",
    "seed",
    "rollout_id",
    "sample_index",
    "group_index",
    "sample_ordinal",
    "tok_pos",
    "pos_norm",
    "prompt_len",
    "resp_len",
    "total_len",
    "truncated",
    "token_id",
    "student_logp_sampled",
    "teacher_logp_sampled",
    "sampled_reverse_kl",
    "sampled_teacher_adv",
    "student_top1_id",
    "teacher_top1_id",
    "student_top1_prob",
    "teacher_top1_prob",
    "student_topk_mass",
    "teacher_topk_mass",
    "Hs_topk",
    "Ht_topk",
    "Hs_topk_norm",
    "Ht_topk_norm",
    "KLf_union",
    "KLr_union",
    "Cmass",
    "Cmass_topk",
    "Cmass_true",
    "Cmass_exact",
    "Coverlap",
    "CBC",
    "Cprefix_ema",
    "reach_t1",
    "target_in_student_topk",
    "target_in_teacher_topk",
    "target_student_rank",
    "target_teacher_rank",
    "loss_mask_original",
    "H_norm",
    "D_norm",
    "C_norm",
    "DC_norm",
    "Dlearn",
    "Dincompat",
    "tip_score",
    "ca_softor_score",
    "split_ca_score",
    "quadrant",
    "budget_method",
    "budget_ratio",
    "budget_score",
    "budget_keep",
    "student_top_ids",
    "student_top_logps",
    "teacher_top_ids",
    "teacher_top_logps",
]


def topk_enabled(args) -> bool:
    return int(getattr(args, "opd_topk_metrics_k", 0) or 0) > 0


def maybe_add_topk_request(args, payload: dict[str, Any]) -> None:
    k = int(getattr(args, "opd_topk_metrics_k", 0) or 0)
    if k <= 0:
        return
    payload["top_logprobs_num"] = k
    payload["return_text_in_logprobs"] = False


def maybe_add_teacher_token_ids_request(args, payload: dict[str, Any], sample) -> None:
    if not bool(getattr(args, "opd_exact_cmass", False)):
        return

    response_length = int(getattr(sample, "response_length", 0) or 0)
    if response_length <= 0:
        return

    student_top = ((getattr(sample, "metadata", None) or {}).get("student_top_logprobs") or [])[-response_length:]
    token_ids = []
    seen = set()
    for pos_items in student_top:
        for token_id, _logp in _parse_top_items(pos_items):
            if token_id not in seen:
                seen.add(token_id)
                token_ids.append(token_id)

    if not token_ids:
        return

    max_union = int(getattr(args, "opd_exact_cmass_max_union", 4096) or 4096)
    if len(token_ids) > max_union:
        overflow = getattr(args, "opd_exact_cmass_overflow", "fallback")
        if overflow == "error":
            raise ValueError(
                f"--opd-exact-cmass union size {len(token_ids)} exceeds "
                f"--opd-exact-cmass-max-union={max_union}"
            )
        if overflow == "fallback":
            return
        token_ids = token_ids[:max_union]

    payload["token_ids_logprob"] = token_ids


def process_tip_compat_metrics(args, samples, teacher_log_probs, raw_rewards) -> None:
    if not topk_enabled(args):
        return

    k = int(getattr(args, "opd_topk_metrics_k", 0) or 0)
    rows = []
    sample_rows: dict[int, list[dict[str, Any]]] = {}

    for sample_ordinal, (sample, t_log_probs, raw_reward) in enumerate(
        zip(samples, teacher_log_probs, raw_rewards, strict=False)
    ):
        response_length = int(sample.response_length or 0)
        if response_length <= 0:
            continue

        prompt_len = len(sample.tokens) - response_length
        student_top = (sample.metadata or {}).get("student_top_logprobs") or []
        student_top = student_top[-response_length:]
        teacher_top = _extract_teacher_response_topk(raw_reward, response_length)
        teacher_token_ids_logprobs = _extract_teacher_response_token_ids_logprobs(raw_reward, response_length)
        if len(student_top) != response_length or len(teacher_top) != response_length:
            if getattr(args, "opd_budget_mask", "none") not in ("none", "full"):
                raise ValueError(
                    "TIP compatibility budget mask requires both student and teacher top-k logprobs. "
                    f"Got student={len(student_top)}, teacher={len(teacher_top)}, response={response_length}."
                )

        original_loss_mask = sample.loss_mask if sample.loss_mask is not None else [1] * response_length
        sample_rows[sample_ordinal] = []
        prefix_ema = None
        alpha = float(getattr(args, "opd_compat_prefix_ema_alpha", 0.9))

        for tok_pos in range(response_length):
            token_id = sample.tokens[prompt_len + tok_pos]
            s_items = _parse_top_items(student_top[tok_pos] if tok_pos < len(student_top) else None)
            t_items = _parse_top_items(teacher_top[tok_pos] if tok_pos < len(teacher_top) else None)
            t_requested_items = _parse_top_items(
                teacher_token_ids_logprobs[tok_pos] if tok_pos < len(teacher_token_ids_logprobs) else None
            )
            metrics = _compute_topk_metrics(s_items, t_items, token_id, k)
            _add_exact_cmass(metrics, s_items, t_requested_items)
            prefix_ema = metrics["Cmass"] if prefix_ema is None else alpha * prefix_ema + (1.0 - alpha) * metrics["Cmass"]

            s_logp = _safe_float(sample.rollout_log_probs[tok_pos]) if sample.rollout_log_probs else float("nan")
            t_logp = _safe_float(t_log_probs[tok_pos]) if tok_pos < len(t_log_probs) else float("nan")
            row = {
                "schema_version": 1,
                "pair_id": getattr(args, "opd_token_bank_pair_id", ""),
                "teacher_name": getattr(args, "opd_teacher_name", None)
                or getattr(args, "opd_teacher_load", None)
                or getattr(args, "rm_url", ""),
                "student_name": getattr(args, "opd_student_name", None) or getattr(args, "hf_checkpoint", ""),
                "seed": getattr(args, "seed", None),
                "rollout_id": None,
                "sample_index": sample.index,
                "group_index": sample.group_index,
                "sample_ordinal": sample_ordinal,
                "tok_pos": tok_pos,
                "pos_norm": tok_pos / max(response_length - 1, 1),
                "prompt_len": prompt_len,
                "resp_len": response_length,
                "total_len": len(sample.tokens),
                "truncated": int(getattr(sample.status, "value", sample.status) == "truncated"),
                "token_id": token_id,
                "student_logp_sampled": s_logp,
                "teacher_logp_sampled": t_logp,
                "sampled_reverse_kl": s_logp - t_logp if math.isfinite(s_logp) and math.isfinite(t_logp) else float("nan"),
                "sampled_teacher_adv": t_logp - s_logp if math.isfinite(s_logp) and math.isfinite(t_logp) else float("nan"),
                "loss_mask_original": int(original_loss_mask[tok_pos]) if tok_pos < len(original_loss_mask) else 1,
                **metrics,
                "Cprefix_ema": prefix_ema,
                "budget_method": getattr(args, "opd_budget_mask", "none"),
                "budget_ratio": float(getattr(args, "opd_budget_ratio", 1.0)),
                "budget_score": 0.0,
                "budget_keep": int(original_loss_mask[tok_pos]) if tok_pos < len(original_loss_mask) else 1,
            }
            rows.append(row)
            sample_rows[sample_ordinal].append(row)

    if not rows:
        return

    _add_normalized_scores(args, rows)
    _apply_budget_mask(args, rows, sample_rows, samples)
    _export_rows(args, rows)


def _extract_teacher_response_topk(raw_reward: dict[str, Any], response_length: int) -> list[Any]:
    top = ((raw_reward or {}).get("meta_info") or {}).get("input_top_logprobs") or []
    if top and top[0] is None:
        top = top[1:]
    return top[-response_length:]


def _extract_teacher_response_token_ids_logprobs(raw_reward: dict[str, Any], response_length: int) -> list[Any]:
    values = ((raw_reward or {}).get("meta_info") or {}).get("input_token_ids_logprobs") or []
    if values and values[0] is None:
        values = values[1:]
    return values[-response_length:]


def _parse_top_items(items: Any) -> list[tuple[int, float]]:
    if not items:
        return []
    parsed = []
    for item in items:
        if not item:
            continue
        logp = _safe_float(item[0])
        token_id = int(item[1])
        if math.isfinite(logp):
            parsed.append((token_id, logp))
    return parsed


def _add_exact_cmass(
    metrics: dict[str, Any],
    student_items: list[tuple[int, float]],
    teacher_requested_items: list[tuple[int, float]],
) -> None:
    metrics["Cmass_topk"] = metrics["Cmass"]
    metrics["Cmass_true"] = ""
    metrics["Cmass_exact"] = 0
    if not student_items or not teacher_requested_items:
        return
    teacher_probs = {token: math.exp(logp) for token, logp in teacher_requested_items}
    if not teacher_probs:
        return
    cmass_true = _clamp_prob_mass(sum(teacher_probs.get(token, 0.0) for token, _ in student_items))
    metrics["Cmass_true"] = cmass_true
    metrics["Cmass"] = cmass_true
    metrics["Cmass_exact"] = 1


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _clamp_prob_mass(x: float) -> float:
    if not math.isfinite(x):
        return x
    return min(1.0, max(0.0, x))


def _compute_topk_metrics(
    student_items: list[tuple[int, float]],
    teacher_items: list[tuple[int, float]],
    token_id: int,
    k: int,
) -> dict[str, Any]:
    eps = 1e-12
    s_probs = {token: math.exp(logp) for token, logp in student_items}
    t_probs = {token: math.exp(logp) for token, logp in teacher_items}
    s_ids = [token for token, _ in student_items]
    t_ids = [token for token, _ in teacher_items]
    inter = set(s_ids).intersection(t_ids)
    union = set(s_ids).union(t_ids)

    s_mass = _clamp_prob_mass(sum(s_probs.values()))
    t_mass = _clamp_prob_mass(sum(t_probs.values()))
    hs, hs_norm = _entropy(s_probs.values())
    ht, ht_norm = _entropy(t_probs.values())
    klf, klr, bc = _union_geometry(s_probs, t_probs, union, eps)

    s_top1_id = s_ids[0] if s_ids else None
    t_top1_id = t_ids[0] if t_ids else None
    target_s_rank = _rank_of(s_ids, token_id)
    target_t_rank = _rank_of(t_ids, token_id)

    return {
        "student_top1_id": s_top1_id,
        "teacher_top1_id": t_top1_id,
        "student_top1_prob": s_probs.get(s_top1_id, 0.0) if s_top1_id is not None else 0.0,
        "teacher_top1_prob": t_probs.get(t_top1_id, 0.0) if t_top1_id is not None else 0.0,
        "student_topk_mass": s_mass,
        "teacher_topk_mass": t_mass,
        "Hs_topk": hs,
        "Ht_topk": ht,
        "Hs_topk_norm": hs_norm,
        "Ht_topk_norm": ht_norm,
        "KLf_union": klf,
        "KLr_union": klr,
        "Cmass": _clamp_prob_mass(sum(t_probs.get(token, 0.0) for token in s_ids)),
        "Cmass_topk": _clamp_prob_mass(sum(t_probs.get(token, 0.0) for token in s_ids)),
        "Cmass_true": "",
        "Cmass_exact": 0,
        "Coverlap": len(inter) / max(k, 1),
        "CBC": bc,
        "reach_t1": int(t_top1_id in set(s_ids)) if t_top1_id is not None else 0,
        "target_in_student_topk": int(token_id in set(s_ids)),
        "target_in_teacher_topk": int(token_id in set(t_ids)),
        "target_student_rank": target_s_rank,
        "target_teacher_rank": target_t_rank,
        "student_top_ids": json.dumps(s_ids),
        "student_top_logps": json.dumps([logp for _, logp in student_items]),
        "teacher_top_ids": json.dumps(t_ids),
        "teacher_top_logps": json.dumps([logp for _, logp in teacher_items]),
    }


def _entropy(probs_iter) -> tuple[float, float]:
    probs = [p for p in probs_iter if p > 0]
    mass = sum(probs)
    if mass <= 0:
        return 0.0, 0.0
    norm = [p / mass for p in probs]
    h = -sum(p * math.log(max(p, 1e-12)) for p in norm)
    return h, h / math.log(max(len(norm), 2))


def _union_geometry(s_probs: dict[int, float], t_probs: dict[int, float], union: set[int], eps: float):
    if not union:
        return 0.0, 0.0, 0.0
    s_mass = sum(s_probs.get(token, 0.0) for token in union)
    t_mass = sum(t_probs.get(token, 0.0) for token in union)
    if s_mass <= 0 or t_mass <= 0:
        return 0.0, 0.0, 0.0

    klf = 0.0
    klr = 0.0
    bc = 0.0
    for token in union:
        ps = max(s_probs.get(token, 0.0) / s_mass, eps)
        pt = max(t_probs.get(token, 0.0) / t_mass, eps)
        klf += pt * (math.log(pt) - math.log(ps))
        klr += ps * (math.log(ps) - math.log(pt))
        bc += math.sqrt(ps * pt)
    return klf, klr, bc


def _rank_of(ids: list[int], token_id: int) -> int | None:
    try:
        return ids.index(token_id) + 1
    except ValueError:
        return None


def _add_normalized_scores(args, rows: list[dict[str, Any]]) -> None:
    compat_key = {
        "mass": "Cmass",
        "set": "Coverlap",
        "bc": "CBC",
        "prefix_ema": "Cprefix_ema",
    }.get(getattr(args, "opd_compat_proxy", "mass"), "Cmass")

    h_vals = [row["Hs_topk_norm"] for row in rows]
    d_vals = [row["KLf_union"] for row in rows]
    c_vals = [row[compat_key] for row in rows]
    dc_vals = [d * c for d, c in zip(d_vals, c_vals, strict=False)]

    h_norm = _normalize(h_vals, args)
    d_norm = _normalize(d_vals, args)
    c_norm = _normalize(c_vals, args)
    dc_norm = _normalize(dc_vals, args)

    h_mid = _median(h_norm)
    d_mid = _median(d_norm)
    for row, hn, dn, cn, dcn in zip(rows, h_norm, d_norm, c_norm, dc_norm, strict=False):
        row["H_norm"] = hn
        row["D_norm"] = dn
        row["C_norm"] = cn
        row["DC_norm"] = dcn
        row["Dlearn"] = dn * cn
        row["Dincompat"] = dn * (1.0 - cn)
        row["tip_score"] = hn + dn - hn * dn
        row["ca_softor_score"] = hn + dcn - hn * dcn
        row["split_ca_score"] = (1.0 - hn) * dcn
        if hn >= h_mid and dn >= d_mid:
            row["quadrant"] = "Q1_highH_highD"
        elif hn >= h_mid and dn < d_mid:
            row["quadrant"] = "Q2_highH_lowD"
        elif hn < h_mid and dn >= d_mid:
            row["quadrant"] = "Q3_lowH_highD"
        else:
            row["quadrant"] = "Q4_lowH_lowD"


def _normalize(values: list[float], args) -> list[float]:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return [0.0 for _ in values]
    mode = getattr(args, "opd_metric_normalization", "batch_quantile")
    if mode == "none":
        lo, hi = 0.0, max(max(finite), 1e-12)
    elif mode == "batch_minmax":
        lo, hi = min(finite), max(finite)
    else:
        lo = _quantile(finite, float(getattr(args, "opd_metric_q_low", 0.05)))
        hi = _quantile(finite, float(getattr(args, "opd_metric_q_high", 0.95)))
    denom = hi - lo
    if abs(denom) < 1e-12:
        return [0.0 for _ in values]
    return [min(1.0, max(0.0, (v - lo) / denom)) if math.isfinite(v) else 0.0 for v in values]


def _quantile(values: list[float], q: float) -> float:
    xs = sorted(values)
    if not xs:
        return 0.0
    pos = min(max(q, 0.0), 1.0) * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def _median(values: list[float]) -> float:
    return _quantile(values, 0.5)


def _apply_budget_mask(args, rows, sample_rows, samples) -> None:
    method = getattr(args, "opd_budget_mask", "none")
    ratio = float(getattr(args, "opd_budget_ratio", 1.0))
    if method in ("none", "full") or ratio >= 1.0:
        return
    if ratio <= 0:
        raise ValueError(f"--opd-budget-ratio must be > 0 when masking is enabled, got {ratio}")

    valid = [i for i, row in enumerate(rows) if int(row["loss_mask_original"]) == 1]
    if not valid:
        return
    budget = max(1, int(math.ceil(len(valid) * ratio)))

    selected: set[int]
    if method == "split_budget_ca":
        gamma = float(getattr(args, "opd_budget_gamma", 0.5))
        n_h = max(0, min(budget, int(round(budget * gamma))))
        n_dc = budget - n_h
        selected = set(_top_indices(rows, valid, "H_norm", n_h))
        selected.update(_top_indices(rows, valid, "split_ca_score", n_dc, exclude=selected))
        if len(selected) < budget:
            selected.update(_top_indices(rows, valid, "ca_softor_score", budget - len(selected), exclude=selected))
    elif method == "random":
        rng = random.Random(int(getattr(args, "opd_budget_mask_seed", 42)) + _current_rollout_id())
        shuffled = valid[:]
        rng.shuffle(shuffled)
        selected = set(shuffled[:budget])
    else:
        score_key = {
            "entropy": "H_norm",
            "divergence": "D_norm",
            "tip": "tip_score",
            "compatibility": "C_norm",
            "ca_softor": "ca_softor_score",
            "q3": "split_ca_score",
            "q3_highc": "C_norm",
            "q3_lowc": "C_norm",
            "dlearn_high": "Dlearn",
            "teachability": "Dlearn",
            "teachability_high": "Dlearn",
            "q3_teachability_high": "Dlearn",
            "q3_teachability_low": "Dlearn",
            "q3_dlearn_high": "Dlearn",
            "q3_dlearn_low": "Dlearn",
            "q3_dincompat_high": "Dincompat",
            "dincompat_high": "Dincompat",
        }.get(method)
        if score_key is None:
            raise ValueError(f"Unknown --opd-budget-mask={method}")
        candidates = valid
        reverse = True
        if method in (
            "q3",
            "q3_highc",
            "q3_lowc",
            "q3_teachability_high",
            "q3_teachability_low",
            "q3_dlearn_high",
            "q3_dlearn_low",
            "q3_dincompat_high",
        ):
            candidates = [i for i in valid if rows[i]["quadrant"] == "Q3_lowH_highD"]
        if method in ("q3_lowc", "q3_teachability_low", "q3_dlearn_low"):
            reverse = False
        selected = set(_top_indices(rows, candidates, score_key, min(budget, len(candidates)), reverse=reverse))

    _ensure_min_keep_per_sample(args, rows, sample_rows, selected)

    for i, row in enumerate(rows):
        keep = int(i in selected and int(row["loss_mask_original"]) == 1)
        row["budget_keep"] = keep
        row["budget_score"] = _score_for_method(method, row)

    for sample_ordinal, per_sample_rows in sample_rows.items():
        mask = [int(row["budget_keep"]) for row in per_sample_rows]
        samples[sample_ordinal].loss_mask = mask


def _top_indices(rows, candidates, score_key, n, *, reverse=True, exclude=None):
    if n <= 0:
        return []
    exclude = exclude or set()
    xs = [i for i in candidates if i not in exclude]
    xs.sort(key=lambda i: (rows[i].get(score_key, 0.0), -rows[i]["tok_pos"]), reverse=reverse)
    return xs[:n]


def _ensure_min_keep_per_sample(args, rows, sample_rows, selected: set[int]) -> None:
    min_keep = int(getattr(args, "opd_budget_min_keep_per_sample", 1) or 0)
    if min_keep <= 0:
        return
    index_by_row_id = {id(row): i for i, row in enumerate(rows)}
    for per_sample_rows in sample_rows.values():
        valid = [index_by_row_id[id(row)] for row in per_sample_rows if int(row["loss_mask_original"]) == 1]
        if not valid:
            continue
        kept = [i for i in valid if i in selected]
        if len(kept) >= min_keep:
            continue
        selected.update(_top_indices(rows, valid, "ca_softor_score", min_keep - len(kept), exclude=selected))


def _score_for_method(method: str, row: dict[str, Any]) -> float:
    return {
        "entropy": row.get("H_norm", 0.0),
        "divergence": row.get("D_norm", 0.0),
        "tip": row.get("tip_score", 0.0),
        "compatibility": row.get("C_norm", 0.0),
        "ca_softor": row.get("ca_softor_score", 0.0),
        "split_budget_ca": max(row.get("H_norm", 0.0), row.get("split_ca_score", 0.0)),
        "q3": row.get("split_ca_score", 0.0),
        "q3_highc": row.get("C_norm", 0.0),
        "q3_lowc": 1.0 - row.get("C_norm", 0.0),
        "dlearn_high": row.get("Dlearn", 0.0),
        "teachability": row.get("Dlearn", 0.0),
        "teachability_high": row.get("Dlearn", 0.0),
        "q3_teachability_high": row.get("Dlearn", 0.0),
        "q3_teachability_low": 1.0 - row.get("Dlearn", 0.0),
        "q3_dlearn_high": row.get("Dlearn", 0.0),
        "q3_dlearn_low": 1.0 - row.get("Dlearn", 0.0),
        "q3_dincompat_high": row.get("Dincompat", 0.0),
        "dincompat_high": row.get("Dincompat", 0.0),
        "random": 0.0,
    }.get(method, 0.0)


def _current_rollout_id() -> int:
    return _ROLLOUT_COUNTER


def _next_rollout_id() -> int:
    global _ROLLOUT_COUNTER
    rollout_id = _ROLLOUT_COUNTER
    _ROLLOUT_COUNTER += 1
    return rollout_id


def _export_rows(args, rows) -> None:
    export_dir = getattr(args, "opd_token_bank_dir", None)
    if not export_dir:
        return
    rollout_id = _next_rollout_id()
    for row in rows:
        row["rollout_id"] = rollout_id

    path = Path(export_dir)
    path.mkdir(parents=True, exist_ok=True)
    fmt = getattr(args, "opd_token_bank_format", "csv")
    if not bool(getattr(args, "opd_token_bank_raw_topk", False)):
        for row in rows:
            row["student_top_ids"] = ""
            row["student_top_logps"] = ""
            row["teacher_top_ids"] = ""
            row["teacher_top_logps"] = ""

    if fmt == "jsonl":
        out = path / f"rollout_{rollout_id:06d}.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=True, allow_nan=True) + "\n")
    else:
        out = path / f"rollout_{rollout_id:06d}.csv"
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    _append_summary(path, rows, rollout_id)
    _write_config(path, args)


def _append_summary(path: Path, rows, rollout_id: int) -> None:
    valid = [row for row in rows if int(row["loss_mask_original"]) == 1]
    kept = [row for row in valid if int(row["budget_keep"]) == 1]
    summary = {
        "rollout_id": rollout_id,
        "num_tokens": len(rows),
        "num_valid_tokens": len(valid),
        "num_kept_tokens": len(kept),
        "keep_ratio": len(kept) / max(len(valid), 1),
        "mean_H_norm": _mean(row["H_norm"] for row in valid),
        "mean_D_norm": _mean(row["D_norm"] for row in valid),
        "mean_C_norm": _mean(row["C_norm"] for row in valid),
        "mean_tip_score": _mean(row["tip_score"] for row in valid),
        "mean_ca_softor_score": _mean(row["ca_softor_score"] for row in valid),
        "mean_reach_t1": _mean(row["reach_t1"] for row in valid),
        "q3_tokens": sum(row["quadrant"] == "Q3_lowH_highD" for row in valid),
        "budget_method": rows[0].get("budget_method", "none") if rows else "none",
        "budget_ratio": rows[0].get("budget_ratio", 1.0) if rows else 1.0,
    }
    out = path / "summary.csv"
    exists = out.exists()
    with out.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(summary)


def _mean(values) -> float:
    xs = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return sum(xs) / len(xs) if xs else 0.0


def _write_config(path: Path, args) -> None:
    out = path / "config.json"
    if out.exists():
        return
    cfg = {
        "schema_version": 1,
        "opd_topk_metrics_k": getattr(args, "opd_topk_metrics_k", None),
        "opd_budget_mask": getattr(args, "opd_budget_mask", None),
        "opd_budget_ratio": getattr(args, "opd_budget_ratio", None),
        "opd_budget_gamma": getattr(args, "opd_budget_gamma", None),
        "opd_compat_proxy": getattr(args, "opd_compat_proxy", None),
        "opd_metric_normalization": getattr(args, "opd_metric_normalization", None),
        "opd_exact_cmass": getattr(args, "opd_exact_cmass", None),
        "opd_exact_cmass_max_union": getattr(args, "opd_exact_cmass_max_union", None),
        "opd_exact_cmass_overflow": getattr(args, "opd_exact_cmass_overflow", None),
        "notes": (
            "Cmass and KL metrics are computed on returned top-k supports. "
            "When opd_exact_cmass is enabled, Cmass is replaced by true teacher mass on student top-k support. "
            "Cmass_topk always keeps the teacher-top-k lower bound."
        ),
    }
    out.write_text(json.dumps(cfg, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

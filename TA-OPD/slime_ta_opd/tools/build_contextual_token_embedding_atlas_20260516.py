import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM


MODEL_PATH = "/path/to/models/Qwen3/1.7B/Qwen_Qwen3-1.7B"
RUN_DIR = Path("/path/to/outputs/slime_opd/qwen3_1_7b_dapo_diag_k8_exact_20260510_082406")
METRICS_PATH = RUN_DIR / "fixed_context/eval_latest/fixed_context_metrics.parquet"
DEBUG_DIR = RUN_DIR / "debug_rollout_data"
OUT_DIR = Path("/path/to/outputs/slime_opd/storyline_20260513/contextual_token_embedding_atlas_20260516")


def add_selector_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Dlearn"] = df["KLf_full"] * df["Cmass_true"]
    df["Dincompat"] = df["KLf_full"] * (1.0 - df["Cmass_true"])
    df["q3_flag"] = (
        (df["Hs_full"] <= df["Hs_full"].median())
        & (df["KLf_full"] >= df["KLf_full"].median())
    ).astype(int)

    top_specs = {
        "dlearn_top3_flag": ("Dlearn", 0.97),
        "dincompat_top3_flag": ("Dincompat", 0.97),
        "divergence_top3_flag": ("KLf_full", 0.97),
        "entropy_top3_flag": ("Hs_full", 0.97),
        "compat_top3_flag": ("Cmass_true", 0.97),
        "gain_top10_flag": ("G_KLf", 0.90),
    }
    for flag, (col, q) in top_specs.items():
        df[flag] = (df[col] >= df[col].quantile(q)).astype(int)

    ent_hi = df["Hs_full"] >= df["Hs_full"].median()
    tip_region = ent_hi | (df["q3_flag"] == 1)
    df["tip_region_flag"] = tip_region.astype(int)
    tip_cut = df.loc[tip_region, "KLf_full"].quantile(0.97)
    df["tip_proxy_top3_flag"] = (tip_region & (df["KLf_full"] >= tip_cut)).astype(int)
    return df


def load_metrics() -> pd.DataFrame:
    print(f"loading metrics: {METRICS_PATH}", flush=True)
    df = pd.read_parquet(METRICS_PATH)
    columns = [
        "sample_ordinal",
        "group_index",
        "tok_pos",
        "pos_norm",
        "prompt_len",
        "resp_len",
        "total_len",
        "token_id",
        "Hs_full",
        "KLf_full",
        "Cmass_true",
        "Coverlap",
        "CBC",
        "G_KLf",
        "KLr_full",
        "Ht_full",
    ]
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing metric columns: {missing}")
    df = df[columns].copy()
    for col in ["Hs_full", "KLf_full", "Cmass_true", "Coverlap", "CBC", "G_KLf", "KLr_full", "Ht_full"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Hs_full", "KLf_full", "Cmass_true", "G_KLf"])
    return add_selector_flags(df)


def load_sample_tokens() -> dict[int, list[int]]:
    sample_tokens = {}
    for path in sorted(DEBUG_DIR.glob("*.pt"), key=lambda p: int(p.stem)):
        obj = torch.load(path, map_location="cpu")
        rollout_id = int(obj.get("rollout_id", int(path.stem)))
        samples = obj["samples"]
        for local_i, sample in enumerate(samples):
            ordinal = rollout_id * len(samples) + local_i
            sample_tokens[ordinal] = [int(x) for x in sample["tokens"]]
    print(f"loaded {len(sample_tokens)} debug samples", flush=True)
    return sample_tokens


def extract_prediction_states(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    print(f"loading model: {MODEL_PATH}", flush=True)
    torch.set_grad_enabled(False)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()
    sample_tokens = load_sample_tokens()

    hidden_blocks = []
    meta_blocks = []
    start = time.time()
    for n, (ordinal, group) in enumerate(df.groupby("sample_ordinal", sort=True), start=1):
        ordinal = int(ordinal)
        tokens = sample_tokens.get(ordinal)
        if tokens is None:
            continue
        input_ids = torch.tensor([tokens], dtype=torch.long, device="cuda")
        with torch.no_grad():
            out = model.model(
                input_ids=input_ids,
                output_hidden_states=False,
                use_cache=False,
                return_dict=True,
            )
            hidden = out.last_hidden_state[0].detach().float().cpu().numpy()

        group = group.sort_values("tok_pos")
        abs_pos = (
            group["prompt_len"].to_numpy(dtype=np.int64)
            + group["tok_pos"].to_numpy(dtype=np.int64)
            - 1
        )
        valid = (abs_pos >= 0) & (abs_pos < hidden.shape[0])
        if not valid.any():
            continue
        hidden_blocks.append(hidden[abs_pos[valid]].astype(np.float32))
        meta = group.loc[valid].copy()
        meta["state_abs_pos"] = abs_pos[valid]
        meta_blocks.append(meta)
        if n % 20 == 0:
            rows = sum(block.shape[0] for block in hidden_blocks)
            print(f"processed {n} samples; {rows} state rows; elapsed {time.time() - start:.1f}s", flush=True)

    if not hidden_blocks:
        raise RuntimeError("No hidden states were extracted.")
    return pd.concat(meta_blocks, ignore_index=True), np.vstack(hidden_blocks).astype(np.float32)


def add_pca(meta: pd.DataFrame, hidden: np.ndarray) -> tuple[pd.DataFrame, list[float]]:
    print(f"hidden matrix: {hidden.shape}", flush=True)
    normalized = hidden / (np.linalg.norm(hidden, axis=1, keepdims=True) + 1e-8)
    centered = normalized - normalized.mean(axis=0, keepdims=True)
    covariance = (centered.T @ centered) / max(1, centered.shape[0] - 1)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order[:8]]
    projected = centered @ vectors
    meta = meta.copy()
    for idx in range(8):
        meta[f"ctx_pc{idx + 1}"] = projected[:, idx]
    meta["ctx_pc_radius"] = np.sqrt(meta["ctx_pc1"] ** 2 + meta["ctx_pc2"] ** 2)
    meta["ctx_pc_angle"] = np.arctan2(meta["ctx_pc2"], meta["ctx_pc1"])
    explained = (values[:8] / values.sum()).astype(float).tolist()
    return meta, explained


def make_scatter_sample(meta: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(20260516)
    extreme = (
        meta["dlearn_top3_flag"].eq(1)
        | meta["dincompat_top3_flag"].eq(1)
        | meta["tip_proxy_top3_flag"].eq(1)
        | meta["gain_top10_flag"].eq(1)
        | meta["q3_flag"].eq(1)
    )
    base_idx = np.where(~extreme.to_numpy())[0]
    extreme_idx = np.where(extreme.to_numpy())[0]
    if len(base_idx) > 30000:
        base_idx = rng.choice(base_idx, size=30000, replace=False)
    selected = np.unique(np.concatenate([base_idx, extreme_idx]))
    return meta.iloc[selected].copy().reset_index(drop=True)


def make_grid(meta: pd.DataFrame) -> pd.DataFrame:
    x = meta["ctx_pc1"].to_numpy()
    y = meta["ctx_pc2"].to_numpy()
    x_bins = np.unique(np.quantile(x, np.linspace(0.01, 0.99, 61)))
    y_bins = np.unique(np.quantile(y, np.linspace(0.01, 0.99, 61)))
    x_idx = np.digitize(x, x_bins) - 1
    y_idx = np.digitize(y, y_bins) - 1
    valid = (x_idx >= 0) & (x_idx < len(x_bins) - 1) & (y_idx >= 0) & (y_idx < len(y_bins) - 1)

    records = []
    summary_cols = [
        "Dlearn",
        "Dincompat",
        "KLf_full",
        "Hs_full",
        "Cmass_true",
        "G_KLf",
        "q3_flag",
        "dlearn_top3_flag",
        "dincompat_top3_flag",
        "tip_proxy_top3_flag",
    ]
    for ix in range(len(x_bins) - 1):
        x_mask = valid & (x_idx == ix)
        if not x_mask.any():
            continue
        for iy in range(len(y_bins) - 1):
            mask = x_mask & (y_idx == iy)
            count = int(mask.sum())
            if count < 3:
                continue
            rec = {
                "x_bin": ix,
                "y_bin": iy,
                "ctx_pc1_mid": float((x_bins[ix] + x_bins[ix + 1]) / 2),
                "ctx_pc2_mid": float((y_bins[iy] + y_bins[iy + 1]) / 2),
                "count": count,
            }
            for col in summary_cols:
                rec[f"mean_{col}"] = float(meta.loc[mask, col].mean())
            records.append(rec)
    return pd.DataFrame(records)


def make_prompt_summary(meta: pd.DataFrame) -> pd.DataFrame:
    return (
        meta.groupby("sample_ordinal")
        .agg(
            ctx_pc1_mean=("ctx_pc1", "mean"),
            ctx_pc2_mean=("ctx_pc2", "mean"),
            ctx_pc3_mean=("ctx_pc3", "mean"),
            Dlearn_mean=("Dlearn", "mean"),
            Dincompat_mean=("Dincompat", "mean"),
            KLf_mean=("KLf_full", "mean"),
            Hs_mean=("Hs_full", "mean"),
            Cmass_mean=("Cmass_true", "mean"),
            G_mean=("G_KLf", "mean"),
            q3_rate=("q3_flag", "mean"),
            dlearn_top3_rate=("dlearn_top3_flag", "mean"),
            tip_proxy_top3_rate=("tip_proxy_top3_flag", "mean"),
            n_tokens=("token_id", "size"),
        )
        .reset_index()
    )


def make_selector_centroids(meta: pd.DataFrame) -> pd.DataFrame:
    records = []
    selectors = [
        ("dlearn_top3_flag", "Dlearn top 3%"),
        ("dincompat_top3_flag", "Dincompat top 3%"),
        ("divergence_top3_flag", "Divergence top 3%"),
        ("entropy_top3_flag", "Entropy top 3%"),
        ("tip_proxy_top3_flag", "TIP proxy top 3%"),
        ("q3_flag", "Q3 region"),
    ]
    for flag, label in selectors:
        rows = meta[meta[flag] == 1]
        if rows.empty:
            continue
        records.append(
            {
                "selector": label,
                "n": int(len(rows)),
                "pc1": float(rows["ctx_pc1"].mean()),
                "pc2": float(rows["ctx_pc2"].mean()),
                "pc3": float(rows["ctx_pc3"].mean()),
                "mean_gain": float(rows["G_KLf"].mean()),
                "mean_dlearn": float(rows["Dlearn"].mean()),
                "mean_dincompat": float(rows["Dincompat"].mean()),
                "mean_compat": float(rows["Cmass_true"].mean()),
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = load_metrics()
    meta, hidden = extract_prediction_states(metrics)
    meta, explained = add_pca(meta, hidden)
    atlas = make_scatter_sample(meta)
    grid = make_grid(meta)
    prompt = make_prompt_summary(meta)
    centers = make_selector_centroids(meta)

    atlas.to_csv(OUT_DIR / "contextual_token_state_atlas_sample.csv", index=False)
    meta.to_csv(OUT_DIR / "contextual_token_state_atlas_full.csv", index=False)
    grid.to_csv(OUT_DIR / "contextual_token_state_pca_grid.csv", index=False)
    prompt.to_csv(OUT_DIR / "contextual_prompt_state_summary.csv", index=False)
    centers.to_csv(OUT_DIR / "contextual_selector_centroids.csv", index=False)
    with open(OUT_DIR / "contextual_atlas_manifest.json", "w") as handle:
        json.dump(
            {
                "model_path": MODEL_PATH,
                "run_dir": str(RUN_DIR),
                "metrics_path": str(METRICS_PATH),
                "debug_dir": str(DEBUG_DIR),
                "n_tokens_full": int(len(meta)),
                "n_tokens_sample": int(len(atlas)),
                "hidden_dim": int(hidden.shape[1]),
                "pca_explained_variance_first8": explained,
                "state_definition": "causal prediction state: hidden position prompt_len+tok_pos-1 predicts the response token at prompt_len+tok_pos",
                "created_at": "2026-05-16",
            },
            handle,
            indent=2,
        )
    print(f"wrote atlas to {OUT_DIR}", flush=True)
    print(f"full rows={len(meta)} sample rows={len(atlas)} explained[:3]={explained[:3]}", flush=True)


if __name__ == "__main__":
    main()

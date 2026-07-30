#!/usr/bin/env python3
# 聚合下游评测结果 -> TA-OPD Table 3 风格 (mean±std over 5 eval seeds)
# 纯标准库实现；评测产物在 outputs/downstream_eval/<model>/seed<X>/results_*.json
#
# 用法:
#   python3 aggregate_table3.py [EVAL_ROOT] [SEED_START] [SEED_END]
# 默认 EVAL_ROOT=$DK/outputs/downstream_eval, seeds 0..4
import json, csv, glob, statistics, pathlib, sys, os

DK = "/inspire/hdd/project/multi-agent/zhangweinan-24046/dk"
EVAL_ROOT = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(DK) / "outputs" / "downstream_eval"
SEED_START = int(sys.argv[2]) if len(sys.argv) > 2 else 0
SEED_END = int(sys.argv[3]) if len(sys.argv) > 3 else 4
SEEDS = list(range(SEED_START, SEED_END + 1))

# (display, lm-eval task name, candidate metric keys) ; GPQA 暂缺 (无 HF token)
BENCH = [
    ("AIME24",   "aime24",          ["exact_match", "acc"]),
    ("AIME25",   "aime25",          ["exact_match", "acc"]),
    ("GPQA-D",   "gpqa_diamond",    ["acc_norm", "exact_match", "acc"]),
    ("HumanEval","humaneval",       ["pass@1", "humaneval_pass@1"]),
    ("IFEval",   "ifeval",          ["prompt_level_strict_acc", "inst_level_strict_acc", "exact_match"]),
    ("MATH-500", "hendrycks_math500",["exact_match", "acc"]),
]

# 模型顺序 + 各自 budget 标注 (来自 指令_下游评测_ta_opd.txt)
MODELS = [
    ("base",        "base"),
    ("pure_opd",    "10%"),
    ("ta_opd",      "0.5%"),
    ("entropy",     "1%"),
    ("tip",         "1%"),
    ("ta_opd_entropy","1%"),
]

# 部分 benchmark 的 lm-eval 指标以 0~1 比例返回，需要 ×100 转百分比
PCT_TASKS = {"aime24", "aime25", "gpqa_diamond", "humaneval", "ifeval", "hendrycks_math500"}


def grab(task, res):
    # lm-eval 0.4.x 指标键会带聚合后缀, 如 "exact_match,none" / "pass@1,create_test"
    # 用 "cand," 前缀匹配, 同时排除 "cand_stderr,..." (以 _stderr 接续, 不会误中)
    for c in ("exact_match", "acc", "acc_norm", "pass@1", "humaneval_pass@1",
              "prompt_level_strict_acc", "inst_level_strict_acc"):
        if c in res:
            return c, res[c]
        for k in res:
            if k.startswith(c + ","):
                return k, res[k]
    return None, None


def to_pct(task, val):
    if val is None:
        return None
    try:
        v = float(val)
    except (TypeError, ValueError):
        return None
    if task in PCT_TASKS and v <= 1.5:   # 0~1 比例 -> 百分比
        v = v * 100.0
    return v


rows = {}
for mname, budget in MODELS:
    rows[mname] = {"budget": budget, "vals": {t[1]: [] for t in BENCH}}
    for s in SEEDS:
        files = sorted((EVAL_ROOT / mname / f"seed{s}").rglob("results_*.json"))
        if not files:
            continue
        try:
            data = json.loads(files[-1].read_text())
        except Exception:
            continue
        results = data.get("results", {})
        for _, task, _ in BENCH:
            sub = results.get(task, {})
            if isinstance(sub, dict):
                _, val = grab(task, sub)
            else:
                val = sub
            v = to_pct(task, val)
            if v is not None:
                rows[mname]["vals"][task].append(v)


def meanstd(vals):
    vs = [v for v in vals if v is not None]
    if not vs:
        return "", "", 0
    mn = statistics.mean(vs)
    sd = statistics.pstdev(vs) if len(vs) > 1 else 0.0
    return mn, sd, len(vs)


lines = [
    "# TA-OPD 下游评测摘要 (mean±std, eval seeds %d-%d)" % (SEED_START, SEED_END),
    "",
    "| Model | Budget | Avg | AIME24 | AIME25 | GPQA-D | HumanEval | IFEval | MATH-500 |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
]
csv_rows = [["model", "budget", "avg", "aime24", "aime25", "gpqa_diamond",
             "humaneval", "ifeval", "math_500"]]

for mname, budget in MODELS:
    cells = [mname, budget]
    avg_vals = []
    for disp, task, _ in BENCH:
        mn, sd, n = meanstd(rows[mname]["vals"][task])
        if n:
            cell = f"{mn:.2f}±{sd:.2f}" if sd else f"{mn:.2f}"
            cells.append(cell)
            avg_vals.append(mn)
        else:
            cells.append("n/a")
    avg = f"{statistics.mean(avg_vals):.2f}" if avg_vals else ""
    cells.insert(2, avg)
    lines.append("| " + " | ".join(cells) + " |")
    csv_cells = [mname, budget, avg] + [
        (f"{meanstd(rows[mname]['vals'][t[1]])[0]:.2f}" if meanstd(rows[mname]['vals'][t[1]])[2] else "n/a")
        for t in BENCH
    ]
    csv_rows.append(csv_cells)

print("\n".join(lines))
out_md = EVAL_ROOT / "summary_table3.md"
out_md.write_text("\n".join(lines) + "\n")
with open(EVAL_ROOT / "summary_table3.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(csv_rows[0])
    w.writerows(csv_rows[1:])
print(f"\n[OK] wrote {out_md} and summary_table3.csv")

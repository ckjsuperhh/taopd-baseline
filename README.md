# TA-OPD Baseline 复现（Qwen3-4B → Qwen3-1.7B-Base）

TA-OPD（Teachability-Aware On-Policy Distillation）基线复现工作仓库。
完整实现/方法说明见上游项目仓库 [`KounianhuaDu/CacheOPD`](https://github.com/KounianhuaDu/CacheOPD) 的 `Baselines/TA-OPD/`，
本仓库保存复现过程的指令脚本、训练/评测产物与下游评测结果。

## ⚠️ 当前状态：结果明显有问题，正在重修（will re-run）

**已确认的问题：**
1. **评测协议对 base 模型是坏的，导致数字虚假偏低。** 学生模型是 Qwen3-1.7B-**Base**，
   原下游评测是 **0-shot、无 system prompt / chat template**（见 `指令_下游评测_ta_opd.txt`）。
   数学题（MATH-500、AIME24/25）要求生成 `\boxed{}` 答案，而 base 模型 0-shot 几乎不输出该格式，
   于是 MATH-500≈0.4%、AIME≈0（这是**评测抽取/协议问题，不是 TA-OPD 方法或模型能力问题**）。
   IFEval（指令遵循）也因同样原因偏低。HumanEval（代码）0-shot 可正常跑，所以 ~51% 还算合理。
2. **训练 budget 非统一、且与论文不可比。** 本地仅 `pure_opd` 训到 10%，其余方法最高只到 0.5–1%
   （论文 Table 3 统一 10%，并扫 30/50%）。目前只完成 4B→1.7B 一组（论文共 4 组）。
3. **GPQA-Diamond 缺列**：gated 数据集，本机无 `HF_TOKEN`，暂未评（标 `n/a`）。

> 结论：现在**不能**用这些数字判断“蒸馏是否优于 base / TA-OPD 是否有效”。
> 它们只证明整条 pipeline（环境→训练→评测→聚合）已跑通。

## 重修计划
- **评测修正**：数学基准改 few-shot（`--num_fewshot 4`，带 `\boxed{}` 示例），让 base 模型输出可抽取答案。
  已更新 `指令_下游评测_ta_opd.txt`（few-shot 调用失败自动回退 0-shot），并同步更新 `aggregate_table3.py`
  以合并拆分后的多次评测结果。全量重跑前用 `FORCE_RERUN=1 bash 指令_下游评测_ta_opd.txt`。
- **训练补全**：视评测修正后的结果，将各方法 budget 补到 10%/30%/50%、并补齐其余 teacher→student 组，再下结论。
- **GPQA**：待提供 `HF_TOKEN` 后单独补跑。

## 目录说明
- `TA-OPD/` — TA-OPD 实现（slime 修改树、训练/诊断/评测脚本、工具）。
- `指令_*.txt` — 分步执行脚本（环境准备 / 权重转换 / 下游评测），远程机 `git pull` 后 `bash 指令_*.txt` 运行。
- `repro*.py` / `repro_base*_submit.sh` — 训练任务提交与复现辅助脚本。
- `results/downstream_eval/` — 下游 6 基准评测结果与聚合表（详见该目录 `README.md`）。
- `实验进度总结与评测补全指南.md` / `实验工作流程指导_TA-OPD.md` — 进度与流程文档。

## 复现结果（当前为待重修版本，仅供参考）
| Method | Budget | AIME24 | AIME25 | HumanEval | IFEval | MATH-500 | GPQA-D |
|---|---|---:|---:|---:|---:|---:|---:|
| Base | — | 0.00 | 3.33 | 51.83 | 22.18 | 0.40 | n/a |
| Full OPD | 10% | 0.00 | 0.00 | 41.46 | 17.56 | 0.00 | n/a |
| TA-OPD | 0.5% | 0.00 | 0.00 | 40.24 | 15.82 | 0.00 | n/a |
| Entropy-only | 1% | 3.33 | 6.67 | 40.24 | 16.52 | 0.00 | n/a |
| TIP-style | 1% | 3.33 | 3.33 | 41.46 | 16.52 | 0.00 | n/a |
| TA-OPD + Entropy | 1% | 3.33 | 6.67 | 38.41 | 16.27 | 0.00 | n/a |

> 上表为旧 0-shot 协议产物，**MATH-500 / AIME / IFEval 偏低属已知评测协议问题**，请勿据此下结论。
> 修正后的结果将更新到 `KounianhuaDu/CacheOPD` 的 `Baselines/TA-OPD/eval_results/`。

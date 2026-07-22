# TA-OPD 复现实验 · 基础工作流程指导（草稿 / AI 辅助生成）

> ⚠️ 本文件由 AI 根据论文 `ta_opd.pdf` 与仓库 `wyy-code/TA-OPD` 提炼生成的**起步草稿**，
> 请按 `经验规则/AI使用规则.md` 要求**自己审阅、补全并认领**内容后再作为正式文档使用。
> 所有 `/path/to/...` 占位符都需要替换成你机器的真实路径。
>
> 参考来源：
> - 论文：`taopd-baseline/ta_opd.pdf`（已提取文本 `ta_opd.txt` 备查）
> - 仓库：`taopd-baseline/TA-OPD/`（已克隆）
> - 实验室规范：`经验规则/AI使用规则.md`、`经验规则/经验总结_EOPD评测部署.md`

---

## 0-N. 落地执行跟踪（本实验实际进度 / 机器分配 / 踩坑）

> 本节是**实际落地记录**，与上面 AI 草稿不同，是经过本机调试后确认的事实。调试细节见 git log。

### 机器分配（关键决策）
- **apex-llm**（6×A6000 / sm86 / 无外网出口 / `/home` 独立挂载曾 100% 满）：只跑 `指令_环境准备_ta_opd.txt`，把 `ta_opd` conda 环境 + torch2.5.1+cu124 + flash-attn + apex + sglang 装好。
- **8×4090 虚拟机**（有 Qwen3-4B teacher、DAPO 数据、`/inspire/...` 路径）：跑 `指令_权重转换_ta_opd.txt` → `指令_smoke_4b_to_1p7b.txt` → `指令_下游评测_ta_opd.txt`。
- 原因：apex-llm 缺 Qwen3-4B、缺 dapo 数据、磁盘紧张，且本身是 A6000 而非 4090；冒烟/评测必须在有模型与数据的 4090 上跑。

### 当前进度（截至 2026-07-22）
- [x] `指令_环境准备_ta_opd.txt` 写定并经 6 轮迭代修复（见下），已 commit+push（最新 `ee9f8f1`）。
- [~] apex-llm 环境构建：已装好 conda `ta_opd` + nvcc 12.4 + torch 2.5.1+cu124 + **flash-attn 2.7.4.post1（[OK]）**；sglang / Megatron-LM clone+patch / apex 编译待续跑（脚本幂等，重跑自动跳过已完成步骤）。
- [ ] 4090 上权重转换、冒烟、评测：未开始（需 4090 机器亲自跑）。

### 踩坑记录（必读，避免重蹈）
1. **torch 与 nvcc 必须同版本**：torch 2.5.1+cu124 必须配 nvcc 12.4；错配（如 nvcc 12.9 / torch 2.13 cu13）会让 apex/flash-attn 编出 cu13 ABI，导入报 `undefined symbol: _ZN3c104cuda29c10_cuda_check_implementation...`。
2. **requirements.txt 偷升 torch**：`TA-OPD/requirements.txt` 第 4 行是裸 `torch`，PHASE5 的 `pip install -r` 会升到 2.13.0(cu13)。已用 `PIP_CONSTRAINT=torch==2.5.1+cu124` + `PIP_EXTRA_INDEX_URL=.../cu124` 锁死（cu124 本地版本号只在该索引有，PyPI 没有）。
3. **flash-attn 跨设备硬链接失败**：预编译 wheel 先下到 TMPDIR 再 `os.link()` 进 pip 缓存；若 TMPDIR 在 `/tmp`（独立挂载）而缓存在 `/home`（另一设备），报 `[Errno 18] Invalid cross-device link`。已把 `TMPDIR` 指到 `OUTPUT_ROOT/tmp`（同设备）修复。
4. **apex-llm 无外网**：git clone / pip 均超时。已用 SSH 反向隧道 `ssh -R 7897:127.0.0.1:7897` 把本机（WSL）代理 `http://127.0.0.1:7897` 暴露给 apex-llm，脚本里 `export http_proxy/https_proxy=http://127.0.0.1:7897`。
5. **apex-llm 磁盘**：`/home`（`/dev/sdb`）曾 100% 满（仅 25G 可用）。删共享/硬链接缓存（conda clean、`.cache`）**不释放**空间；必须删**独有** conda 环境（已删 `eopd` → 释放 19G 到 44G）。`ckpts/EOPD`（87G）保留（用户：后期 ckpt 不删、早期可删、原始数据集可删）。
6. **apex-llm ≠ 4090**：6×A6000(sm86)，conda 不在 PATH（`/home/kejiechen/miniconda3`）。脚本已参数化：`FORCE_CONDA_NVCC=1`（conda-forge 装 cuda-nvcc=12.4）、`TORCH_CUDA_ARCH_LIST=8.6`、`CONDA_BIN=...`；这些只影响 apex-llm，不动 4090 默认（sm89）。

### 指令脚本清单
| 脚本 | 机器 | 作用 | 状态 |
|---|---|---|---|
| `指令_环境准备_ta_opd.txt` | apex-llm | 装 ta_opd 环境 + nvcc + torch + flash-attn + apex + sglang | 已推送，apex 上构建中 |
| `指令_权重转换_ta_opd.txt` | 4090 | 下模型、HF→torch_dist、DAPO→parquet | 待 4090 跑 |
| `指令_smoke_4b_to_1p7b.txt` | 4090 | 4B→1.7B 冒烟（pure_opd + ta_opd） | 待 4090 跑 |
| `指令_下游评测_ta_opd.txt` | 4090 | 6 基准评测（AIME24/25、GPQA、HumanEval、IFEval、MATH500） | 待 4090 跑 |

### 路径 / 入口注意
- 4090 脚本写死 `/inspire/hdd/project/multi-agent/zhangweinan-24046/dk/...`；apex-llm 跑环境脚本时通过 `CODE_ROOT=/home/kejiechen/taopd-baseline` 覆盖。
- 冒烟真实入口：`slime_ta_opd/examples/on_policy_distillation/run-qwen3-1.7B-sampled-opd-sglang.sh`（与本文 §4 写的 `run-qwen3-4B-sampled-opd-sglang.sh` 不一致，以冒烟脚本里的为准）。
- 权重转换里 DAPO 数据集名 `diagnoseailab/dapo-math-17k` 是占位，需替换为你实际拿到的 DAPO 数据。

---

## 0. 一句话理解论文

**TA-OPD = Teachability-Aware On-Policy Distillation**。
它在「sampled-token on-policy distillation（OPD）」的标准 rollout + teacher 打分流程上，
额外加一个 **hard token mask**：只对学生当前**可学（learnable）** 的 token 施加直接 KL 监督，
其余 response token 仍留在上下文里但被 mask 掉其直接 OPD loss。

核心结论（Table 3）：在 10% 监督 token 预算下，TA-OPD 在四个 teacher→student 组合里平均分数
都优于 / 持平 Full OPD。**注意：这是「监督 token 高效」，不是计算剪枝，不要宣称墙钟时间同比节省。**

---

## 1. 实验栈与环境

| 项 | 内容 |
|---|---|
| 基础设施 | `THUDM/slime` + `NVIDIA/Megatron-LM` |
| 推荐入口 | `TA-OPD/slime_ta_opd/`（完整修改版 slime 树，含训练入口 `train.py` 与全部 launch 脚本） |
| 训练期依赖 | SGLang（teacher server + student rollout）、Ray、PyTorch、Megatron-LM、Transformer Engine、slime |
| 论文硬件 | 64 × NVIDIA H800（本地需大幅缩到 smoke 规模） |
| 评测工具 | EvalScope / `lm-eval`（论文用 EvalScope 跑 6 个基准） |

环境约定（来自脚本）：教师/rollout 用含 SGLang+Ray+Torch 的环境（脚本里叫 `verl`），
Megatron 权重转换用另一个环境（脚本里叫 `opsd`，该环境**没有** Transformer Engine，
转换时需 `--no-rope-fusion --transformer-impl local` 等开关）。通过 `PYTHONPATH` 把
`MEGATRON_LM_DIR`、`SLIME_DIR`、站点包串起来。

---

## 2. 模型、数据、评测

### 2.1 Teacher → Student 组合（论文 Table 3）
- Qwen3-4B → **Qwen3-1.7B**
- Qwen3-8B（GRPO 微调）→ **Qwen3-4B**
- Qwen3-14B → **Qwen3-4B**
- DeepSeek-R1-Distill-Qwen-14B → **Qwen2.5-3B**（跨 backbone）

### 2.2 训练数据
- 论文：prompts 从 **DAPO**（Yu et al., 2026）采样，需整理成 `train.parquet`（字段 `prompt`，带 chat template）。
- 入口变量：`PROMPT_DATA`。

### 2.3 评测（6 个基准，EvalScope，每个 checkpoint 跑 5 个 eval seed，报 mean±std）
AIME24、AIME25、GPQA-Diamond、HumanEval、IFEval、MATH-500。

> ⚠️ **与 EOPD 经验不同**：EOPD 经验用的是 6 个数学基准
> （MATH500/AMC23/Minerva/OlympiadBench/AIME24/25）；TA-OPD 是上面这套含代码/事实/指令遵循的 6 基准。
> 不能直接套用 EOPD 的 `eval_six_benchmarks.sh`，需要单独准备 EvalScope 任务。

### 2.4 关键超参（贯穿全文）
- support 统计量 **K = 16**（除非特别说明）。
- **budget** = 被 KL 监督的 response-token 比例（不是墙钟时间），论文扫 5% / 10% / 30% / 50%。
- 优化器：`--lr 1e-6 --lr-decay-style constant --weight-decay 0.1`。
- OPD：`--opd-kl-coef 1.0 --advantage-estimator grpo --use-opd --opd-type sglang`。
- 训练规模（full 运行）：约 **300 轮 rollout**（fixed-context 表给出 300 / K16 / 57600）。

---

## 3. 权重格式转换

OPD 训练吃 **torch_dist（Megatron）** 格式，评测吃 **HF** 格式，所以两头都要转。

| 方向 | 工具 | 参考脚本 |
|---|---|---|
| HF → torch_dist（student 初始化） | `tools/convert_hf_to_torch_dist.py` | `slime_ta_opd/convert_qwen3_1_7b_to_torch_dist.sh`、`convert_qwen3_4b_to_torch_dist.sh` |
| torch_dist → HF（训练产物评测用） | `tools/convert_torch_dist_to_hf*.py` | `TA-OPD/tools/convert_torch_dist_to_hf.py` 等 |

用法：复制对应脚本，改 `HF_MODEL` / `SAVE_DIR`（转换脚本用环境变量覆盖即可）。
注意转换环境无 TE，保留 `--no-rope-fusion --transformer-impl local` 等。

---

## 4. 训练流程（核心）

**入口脚本**：`slime_ta_opd/run-qwen3-4B-sampled-opd-sglang.sh`

内部流程：
1. 在 1 张 GPU 上启 **teacher SGLang server**（`--tp 1`）。
2. 启 **Ray head**（actor + rollout 占用的若干 GPU）。
3. `ray job submit → python3 train.py`，拼装：
   - 模型 / 检查点 / rollout / 优化器 / GRPO / SGLang / TIP 兼容 等参数组。
4. 轮询 Ray job 状态直到 `SUCCEEDED`；产出 `latest_checkpointed_iteration.txt`。

**多方法 sweep 入口**：`slime_ta_opd/run_teachability_opd_method_suite_20260515.sh`
→ 调用 `scripts/train/run_opd_budget_ratio_sweep.sh`（真正发起各次训练）。

### 4.1 方法名 → mask 映射（论文与实现对照）
| 论文名 | mask | 含义 |
|---|---|---|
| **TA-OPD** | `dlearn_high` | 按 `Dlearn = D_norm · C_norm` 选 top token |
| Full OPD | `full` | 全部有效 response token 都作直接 OPD 目标 |
| Entropy-only | `entropy` | 按归一化学生熵选 |
| TA-OPD + Entropy | `ca_softor` | `H + Dlearn − H·Dlearn`（软或） |
| Split Entropy+TA | `split_budget_ca` | 用 `OPD_BUDGET_GAMMA` 切分预算 |
| TIP-style | `tip` | `H + 原始 divergence` 软或 |
| High divergence | `divergence` | 仅原始归一化 divergence |
| Random | `random` | 同预算随机对照 |

### 4.2 主命令（复刻论文方法套）
```bash
cd TA-OPD/slime_ta_opd
export SLIME_DIR=$PWD
export MEGATRON_LM_DIR=/path/to/Megatron-LM
export OUTPUT_ROOT=/path/to/outputs/slime_opd
export TEACHER_MODEL=/path/to/teacher-hf
export STUDENT_HF=/path/to/student-hf
export STUDENT_TORCH_DIST=/path/to/student-torch-dist
export PROMPT_DATA=/path/to/data/train.parquet
export OPD_TOPK_METRICS_K=16

# 方法套：method:ratio:idx（pure_opd 的 ratio 强制 1.0）
export METHOD_LIST="pure_opd:1.0:10 teachability:0.10:20 entropy:0.10:30 teachability_entropy:0.10:40 tip:0.10:50"
bash run_teachability_opd_method_suite_20260515.sh
```

### 4.3 Smoke 自检（先打通 pipeline，再放大规模）
```bash
export NUM_ROLLOUT=50
export METHOD_LIST="pure_opd:1.0:10 teachability:0.10:20 tip:0.10:30"
bash run_teachability_opd_method_suite_20260515.sh
```
> 默认 `NUM_ROLLOUT=2 / ROLLOUT_BATCH_SIZE=4 / N_SAMPLES_PER_PROMPT=2` 是最小冒烟值；
> `run_opd_budget_ratio_sweep.sh` 内用 `NUM_ROLLOUT=50`。**完整跑论文要用 ~300 轮**。

---

## 5. 评测

- 开发期轻量 wrapper：`TA-OPD/scripts/eval/`（gsm8k / math_hard / aime 的 smoke 检查）。
- **大规模评测请用你的生产评测栈（EvalScope）**，论文用 EvalScope 跑上述 6 基准、每 checkpoint 5 seed。
- 评测脚本示例：`slime_ta_opd/run_downstream_math_hard_20260515.sh`，用 `lm-eval` 跑 EvalScope 任务，
  多 GPU 并行（`run_one gpu label model`），已有结果则 skip。
- 训练产物需先 **torch_dist → HF**（见 §3），再把 `student_hf` 目录喂给评测。

---

## 6. 诊断（论文 Section 4，可选但推荐）

fixed-context diagnostic：冻结学生生成的上下文，在训练前后对同一批位置重新打分，
测量「同上下文下 teacher–student KL 是否下降」，以此判断某个 token 级信号是否真的可学。

流程与工具：
1. `tools/export_fixed_context_bank.py` — 导出固定上下文 bank
2. `tools/eval_fixed_context_bank.py` — 训练前后分别打分
3. `tools/analyze_fixed_context_gain.py` — 算 token 级 gain 与 support 分解
4. 入口：`scripts/diagnostics/run_*_fixed_context_300.sh`（如 `run_4b_to_1p7b_heldout_fixed_context_300.sh`）

---

## 7. 与实验室经验规则的衔接（必须照做）

把 `经验规则/` 的规范落到本实验：

- **协作规范（最重要）**：所有启动 / 诊断 / 修复命令写成 `指令_*.txt` 提交 git，
  远程机器 `git pull` 后 `bash 指令_*.txt` 执行；改脚本后 `commit + push` 再让远程 `pull`，
  不要只在聊天里贴改完的代码。
- **进度可视**：长耗时脚本（生成、评测、转换）加 `tqdm` 进度条，避免干等焦虑。
- **提速**：小模型用**数据并行**（多独立实例 / 多卡）而非张量并行；单实例速度不变，加速来自并发。
- **环境**：用指定 conda 的 python，跑前预检依赖（vllm / math_verify / transformers / pandas / sglang / ray 等）。
- **产物与重跑**：保留生成 / 中间产物（如 token_bank、merged parquet），失败只重跑那一步。
- **复现性（AI使用规则）**：文件自己撰写、AI 仅辅助；用 Git 管理且每次 commit 不超函数级；
  用**伪随机**保证数据可复现；运行 log 用 **Git LFS** 保存便于检查。
- **诊断姿势**：看 `nvidia-smi` / `pgrep -af <脚本>` / `ps -o etime,time,pcpu` 判断是否在推进；
  抓完整报错用 `timeout ... > /tmp/run.txt 2>&1; echo EXIT=$?; cat /tmp/run.txt`。

---

## 8. 落地第一步（建议顺序）

1. **定规模**：选 teacher→student 对 + 可用 GPU 数（论文 64×H800；本地先 smoke）。
2. **备料**：下载模型权重、整理 DAPO 训练 `train.parquet`、准备 EvalScope 6 基准任务。
3. **转权重**：student HF → torch_dist（§3）。
4. **打通 pipeline**：`NUM_ROLLOUT=50`，先跑 `pure_opd` + `teachability` 验证端到端能出 checkpoint 并能评测。
5. **放大**：pipeline 通了再上论文预算（budget 5/10/30/50%、~300 轮）与方法套。
6. **对照**：用 Table 3 / Table 4 的 mean±std 核对（注意那里是 6 基准含代码/事实/指令，不是 EOPD 的 6 数学基准）。

---

## 附：仓库关键文件速查
- `TA-OPD/README.md`、`TA-OPD/docs/experiment_workflows.md`、`TA-OPD/docs/method_entrypoints.md`
- `TA-OPD/slime_ta_opd/run-qwen3-4B-sampled-opd-sglang.sh` — 单跑训练入口
- `TA-OPD/slime_ta_opd/run_teachability_opd_method_suite_20260515.sh` — 方法套入口
- `TA-OPD/scripts/train/run_opd_budget_ratio_sweep.sh` — 真正的 sweep 发起
- `TA-OPD/slime_ta_opd/convert_qwen3_*_to_torch_dist.sh` — HF→torch_dist
- `TA-OPD/tools/convert_torch_dist_to_hf*.py` — torch_dist→HF
- `TA-OPD/scripts/eval/`、`TA-OPD/scripts/diagnostics/` — 评测与诊断入口
- 核心实现：`slime_ta_opd/slime/rollout/on_policy_distillation.py`、`tip_compat.py`、`slime/ray/rollout.py`

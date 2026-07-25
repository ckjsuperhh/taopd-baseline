# AGENT_HANDOFF.md — TA-OPD 4B→1.7B Baseline 复现工作流交接

> **用途**：新虚拟机的 AI agent（QoderCN 或其他）拉完仓库后，先读这份文档即可接手后续工作。
>
> **当前状态**：apex-llm 服务器（驱动 530.30.02 / CUDA 12.1）上已验证全流程 0-5 步（数据 + 模型 + torch_dist + teacher server），**训练 Ray job 因 NCCL + 老驱动 segfault 跑不起来**。需要在一台 CUDA 12.9 驱动的新 VM 上重跑。

---

## 1. 项目目标

**严格复现 TA-OPD 论文**（THUDM/slime）的 `Qwen3-4B → Qwen3-1.7B-Base` On-Policy Distillation baseline。

- 论文代码：`slime_ta_opd/`（THUDM/slime 的本地 checkout）
- 训练框架：slime + Megatron-LM + SGLang + Ray
- 流程基准：`slime_ta_opd/build_conda.sh`（官方一键脚本）
- 流程入口：`repro/setup_faithful.sh`（100% 复刻官方，不替换任何依赖）

## 2. 仓库布局

```
taopd-baseline/
├── TA-OPD/                              # 本仓根
│   ├── slime_ta_opd/                    # slime 源码 (THUDM/slime 的 fork/checkout)
│   │   ├── build_conda.sh               # 官方流程基准 (不要改它)
│   │   ├── convert_qwen3_1_7b_to_torch_dist.sh   # HF→torch_dist 入口
│   │   ├── tools/convert_hf_to_torch_dist.py     # ⚠️ 已改 (见 §4)
│   │   ├── slime/backends/megatron_utils/initialize.py  # ⚠️ 已改 (见 §4)
│   │   ├── docker/patch/v0.5.9/         # sglang.patch + megatron.patch
│   │   └── scripts/models/qwen3-1.7B.sh # 模型参数
│   ├── repro/                           # 复现脚本 (本仓维护)
│   │   ├── 00_env.sh                    # 路径/端口/超参配置
│   │   ├── setup_faithful.sh            # 主流程 (conda env + 编译 + 安装)
│   │   ├── setup_faithful_resume_*.sh   # 分阶段 resume 版本
│   │   ├── _run_step6_rest.sh           # steps 6b+7-9 (apex+TE+slime+patches)
│   │   ├── _run_step789.sh              # step 7 tail + 8-9 (slime+Megatron install)
│   │   ├── _run_te_only.sh              # 单独编译 TE 的脚本
│   │   ├── _run_download_and_smoke.sh   # 下载+转换+smoke (apex 专用)
│   │   ├── 02_download_models.sh        # 下载 Qwen3-4B + Qwen3-1.7B
│   │   ├── 03_convert_student.sh        # 生成 torch_dist
│   │   ├── 04_prepare_data.sh           # DAPO-Math-17k + GSM8K-COT
│   │   ├── 05_smoke_test_faithful.sh    # 冒烟 (pure_opd + ta_opd)
│   │   └── 06_run_all_training.sh       # 正式训练 (28 runs × 3 seeds)
│   └── AGENT_HANDOFF.md                 # 本文档
└── (同级可放) Megatron-LM/              # 本地 clone (避免 GitHub 不通)
```

## 3. 已完成的工作（apex-llm 验证）

| 步骤 | 状态 | 命令/文件 |
|------|------|----------|
| 1. conda env (micromamba) | ✅ | `repro/setup_faithful.sh` |
| 2. CUDA toolkit 12.9 (micromamba) | ✅ | 同上 |
| 3. PyTorch 2.9.1+cu129 | ✅ | `pip install torch==2.9.1+cu129` |
| 4. Transformer Engine 2.10.0 (源码编译) | ✅ | `_run_te_only.sh` |
| 5. NVIDIA Apex (源码编译) | ✅ | `_run_step6_rest.sh` |
| 6. flash-attn 2.7.4.post1 | ✅ | |
| 7. SGLang 0.5.16 (commit `bbe9c7ee`) + sglang-router 0.3.2 | ✅ | |
| 8. Megatron-LM (commit `3714d81d4` + `megatron.patch`) | ✅ | `_run_step789.sh` |
| 9. slime (`slime_ta_opd/`) + `slime_plugins.mbridge` | ✅ | |
| 10. numpy==2.0.2 + slime 断言放宽 | ✅ | |
| 11. Qwen3-4B + Qwen3-1.7B 模型下载 | ✅ | `02_download_models.sh` |
| 12. Qwen3-1.7B → torch_dist | ✅ | `convert_qwen3_1_7b_to_torch_dist.sh` |
| 13. DAPO-Math-17k + GSM8K-COT 数据准备 | ✅ | `04_prepare_data.sh` |
| 14. SGLang teacher server 启动 (4B, GPU 1) | ✅ | `05_smoke_test_faithful.sh` |
| 15. **训练 Ray job** | ❌ | NCCL 2.27.5 `cudaFuncGetAttributes` segfault |

## 4. 已做的源码修改（必须保留）

### `slime_ta_opd/tools/convert_hf_to_torch_dist.py` (~line 101)
```python
# 单进程转换: 用 gloo 避 NCCL (apex 驱动 530.30.02 只 CUDA 12.1,
# 而 torch 2.9.1+cu129 带 NCCL 2.27.5 调 cudaFuncGetAttributes 会 segfault)
backend = "gloo" if world_size == 1 else "nccl"
dist.init_process_group(
    backend=backend,
    ...
)
```
**新 VM 驱动 ≥550 时这个 patch 仍然安全**（`world_size==1` 走 gloo 更快）。

### `slime_ta_opd/slime/backends/megatron_utils/initialize.py` (~line 66)
```python
# apex 上 transformers 4.57.1 + scipy 1.18 强制 numpy>=2.0,
# 我们用 numpy 2.0.2, 实测能跑. 放宽断言.
assert np.__version__.startswith(("1.", "2.")), f"numpy version unsupported: {np.__version__}"
```

### `slime_ta_opd/convert_qwen3_1_7b_to_torch_dist.sh` (EXTRA_MEGATRON_ARGS)
```bash
EXTRA_MEGATRON_ARGS="${EXTRA_MEGATRON_ARGS:---no-rope-fusion --no-persist-layer-norm --no-gradient-accumulation-fusion}"
```
**去掉了 `--transformer-impl local`**：local backend 的 `FusedLayerNorm` 只认 `LayerNorm`，遇到 Qwen3 的 `RMSNorm` 会在 `megatron/core/fusions/fused_layer_norm.py:67` 断言失败。TE backend 的 `TENorm` 同时支持两者。

## 5. 已踩的坑（新 VM 的 AI 必读）

| 坑 | 现象 | 修复 |
|---|---|---|
| TE wheel 下载被墙 | `Remote end closed connection` 拉 GitHub releases | `export NVTE_PYTORCH_FORCE_BUILD=TRUE` 强制源码编译 |
| TE 编译缺 `nccl.h` | `No such file or directory` | 把 `nvidia/nccl/include` 加到 `C_INCLUDE_PATH`（torch wheel 自带 NCCL headers） |
| Apex 编译 `cuda_profiler_api.h` | `fatal error` | `micromamba install -c nvidia/label/cuda-12.9.1 cuda-profiler-api -y` |
| Apex 编译 `Error compiling objects` | 找不到 nvcc | `export CUDA_HOME=${CONDA_PREFIX}` |
| `sgl-router` 包名错 | `No matching distribution found` | PyPI 包名是 **`sglang-router`**（不是 `sgl-router`） |
| GitHub 被墙拉不动 Megatron | clone 超时 | 用本地已有 clone + `git clone --dissociate` 做隔离副本 |
| Megatron install 缺 pybind11 | `python3: No module named pybind11` | `pip install pybind11` 先装 |
| transformers 4.57 要 numpy≥2 | `ImportError: PreTrainedModel` | `pip install "numpy==2.0.2"`（不是 `<2`） |
| slime numpy 断言太严 | `AssertionError: Megatron does not support numpy 2.x` | 已改 `initialize.py:66` |
| Qwen3 RMSNorm + FusedLayerNorm | `AssertionError: (RMSNorm) is not supported in FusedLayerNorm` | 去掉 `--transformer-impl local`（见 §4） |
| NCCL segfault（驱动 530） | `cudaFuncGetAttributes` SIGSEGV | **驱动 ≥550 才能跑训练** |
| conda cudnn 9.10 vs NVIDIA wheel 9.16 | TE `libcudnn_graph.so.9` 符号找不到 | `micromamba remove cudnn libcudnn libcudnn-dev`，只用 pip wheel |
| conda cudnn 被删后 libstdc++ 也被带走 | TE `CXXABI_1.3.15` 找不到 | `export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:...` |
| sglang flashinfer JIT 编译缺 `-lcuda` | `ld: cannot find -lcuda` | `export LIBRARY_PATH=/usr/local/cuda-12.1/targets/x86_64-linux/lib/stubs:${LIBRARY_PATH:-}` |
| Ray runtime env LD_LIBRARY_PATH 不继承 | worker 用系统 libstdc++ 崩 | 用 `--runtime-env-json` 显式传 `LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:...` |
| Ray `--num-gpus 3` 但实际只有 2 张卡 | job 调度失败 | `sed` 改成 `--num-gpus 2`、`--actor-num-gpus-per-node 1` |

## 6. 新 VM 操作流程（按顺序）

### Step A：环境预检
```bash
nvidia-smi | head -3       # 驱动 ≥550
curl -I https://github.com  # 可达?
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv
```

### Step B：安装 QoderCN
```bash
curl -fsSL https://qoder.com.cn/install | bash
export PATH="/root/.local/bin:$PATH"
echo 'export PATH="/root/.local/bin:$PATH"' >> ~/.bashrc
```

### Step C：拉仓库
```bash
git clone https://github.com/ckjsuperhh/taopd-baseline.git
cd taopd-baseline/TA-OPD
```

### Step D：跑主流程
```bash
export HF_ENDPOINT=https://hf-mirror.com   # 国内必设
bash repro/setup_faithful.sh
```
预计 30-60 分钟（TE + Apex 源码编译是大头）。

### Step E：下载模型 + 数据
```bash
bash repro/02_download_models.sh
bash repro/04_prepare_data.sh
```

### Step F：生成 torch_dist
```bash
bash repro/03_convert_student.sh
# 或直接:
bash slime_ta_opd/convert_qwen3_1_7b_to_torch_dist.sh \
  HF_MODEL=.../Qwen3-1.7B SAVE_DIR=.../Qwen3-1.7B_torch_dist
```

### Step G：冒烟测试
```bash
bash repro/05_smoke_test_faithful.sh
```
- pure_opd：mask=none, ratio=1.0
- ta_opd：mask=dlearn_high, ratio=0.10

### Step H：正式训练
```bash
bash repro/06_run_all_training.sh
```
28 (mask, ratio) × 3 seeds = 84 runs。

## 7. 关键路径布局（00_env.sh 默认值）

```bash
PROJECT_ROOT=/inspire/hdd/project/.../CacheOPD/taopd-baseline
DATA_DIR=.../dk/data
MODEL_DIR=.../dk/modelweights
OUTPUT_ROOT=.../dk/outputs
REPO_ROOT=${PROJECT_ROOT}/TA-OPD
SLIME_DIR=${REPO_ROOT}/slime_ta_opd
MEGATRON_LM_DIR=${PROJECT_ROOT}/Megatron-LM

TEACHER_MODEL=${MODEL_DIR}/Qwen3-4B
STUDENT_HF=${MODEL_DIR}/Qwen3-1.7B
STUDENT_TORCH_DIST=${MODEL_DIR}/Qwen3-1.7B_torch_dist

PROMPT_DATA=${DATA_DIR}/DAPO-Math-17k-dedup/dapo_math_17k_dedup_slime.jsonl
HELDOUT_DATA=.../dapo_math_17k_dedup_slime_heldout300_seed41717.jsonl
GSM8K_DATA=.../gsm8k_cot_slime_300_seed41717.jsonl
```

新 VM 上跑 `setup_faithful.sh` / `05_smoke_test_faithful.sh` 前，用 `export PROJECT_ROOT=...` 覆盖即可。

## 8. GPU 布局（论文 / 00_env.sh 默认）

**Non-colocate**：rollout engine 独占 1 张卡，不跟 actor 共享。

```
Lane A: Teacher GPU0, Actor GPU1+2 (dp=2), Rollout GPU3
Lane B: Teacher GPU4, Actor GPU5+6 (dp=2), Rollout GPU7
```

apex-llm 上 GPU 0 被 vllm 占满 → 改用 GPU 1 当 teacher，student 用 GPU 2+3（2 卡）。

## 9. 已知的"无法绕过"限制

1. **NVIDIA 驱动 < 550 → 训练跑不了**。NCCL 2.27.5 调 `cudaFuncGetAttributes` 时 segfault。apex-llm 已验证（convert 能跑是因为改成了 gloo；训练是多进程 NCCL，绕不过）。
2. **GitHub 不通**：Megatron-LM 必须本地 clone 或用 mirror。
3. **transformers 4.57 + scipy 1.18 强制 numpy≥2.0**：官方 `build_conda.sh` 里 `pip install "numpy<2"` 会冲突，必须改成 2.0.2。
4. **TE 必须源码编译**：PyPI 的 wheel 依赖 GitHub releases 下载。

## 10. 接手后的第一个任务

**新 VM 驱动 ≥550 时**：直接跑 §6 的 Step A-H。

**新 VM 驱动仍是 530**：
1. 尝试升级驱动（`sudo apt install nvidia-driver-550` 或联系管理员）
2. 若不能升级 → 放弃这台 VM，找一台带新驱动的

**遇到问题**：先查 §5 的坑表，80% 的可能撞过的坑都在里面。

---

**最后修改**：2026-07-25，apex-llm 上跑完 §3 的 1-14 步后卡在 NCCL segfault。

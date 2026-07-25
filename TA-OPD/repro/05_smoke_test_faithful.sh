#!/usr/bin/env bash
# Smoke test 在 faithful env 里跑 (完全按论文流程, 不改任何依赖)
#
# 和 repro/05_smoke_test.sh 的区别:
#   1. 激活 ta_opd_faithful env (不是 ta_opd / ta_opd_modern)
#   2. 不 patch Qwen3 (sglang 0.5.10 原生支持)
#   3. sglang 0.5.10 用 --nccl-port (不用 --dist-init-addr)
#   4. sglang 0.5.10 不识别 --attention-backend triton, 删掉
#   5. Megatron 已经 apply v0.5.9/megatron.patch
#
# 用法:
#   bash repro/05_smoke_test_faithful.sh
#
# 前置条件:
#   - bash repro/setup_faithful.sh 跑完
#   - Qwen3-4B + Qwen3-1.7B 模型已下载到 ${MODEL_DIR}
#   - Qwen3-1.7B_torch_dist 已生成 (用 repro/03_convert_student.sh 或等价)
#   - PROMPT_DATA 存在
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"

# ── 激活 faithful env ────────────────────────────────────────────────────
FAITHFUL_ENV="${FAITHFUL_ENV:-ta_opd_faithful}"
MICROMAMBA_ROOT="${MICROMAMBA_ROOT:-${HOME}/micromamba}"
if [[ -x "${MICROMAMBA_ROOT}/bin/micromamba" ]]; then
  eval "$("${MICROMAMBA_ROOT}/bin/micromamba" shell hook --shell bash)"
  micromamba activate "${FAITHFUL_ENV}"
else
  # 兜底: conda
  CONDA_SH="${CONDA_SH:-$(conda info --base 2>/dev/null || echo "${HOME}/miniconda3")/etc/profile.d/conda.sh}"
  source "${CONDA_SH}"
  conda activate "${FAITHFUL_ENV}"
fi
echo "✅ activated: ${FAITHFUL_ENV}"

# ── 前置检查: 必需的资源 ──────────────────────────────────────────────────
echo ""
echo "=== 前置资源检查 ==="
MISSING=0
for f in "${TEACHER_MODEL}/config.json" "${STUDENT_HF}/config.json" "${STUDENT_TORCH_DIST}/latest_checkpointed_iteration.txt" "${PROMPT_DATA}"; do
  if [[ -e "$f" ]]; then
    echo "  ✅ $f"
  else
    echo "  ❌ $f (缺失)"
    MISSING=1
  fi
done
if [[ "${MISSING}" -eq 1 ]]; then
  echo ""
  echo "请先准备上述资源。快速办法:"
  echo "  bash repro/setup_faithful.sh        # 已跑过"
  echo "  bash repro/02_download_models.sh    # 下载 Qwen3-4B + Qwen3-1.7B"
  echo "  bash repro/03_convert_student.sh    # 生成 torch_dist"
  echo "  bash repro/04_prepare_data.sh       # 下载 DAPO-Math-17k"
  exit 1
fi

# ── 验证 sglang 版本 (必须 >= 0.5, 否则 --nccl-port 不存在) ─────────────
SGLANG_VERSION="$(python3 -c 'import sglang; print(getattr(sglang, "__version__", "0.0.0"))')"
echo "✅ sglang version: ${SGLANG_VERSION}"
case "${SGLANG_VERSION}" in
  0.5.*|0.6.*) ;;
  *) echo "❌ sglang 版本不对 (期望 0.5.x 或 0.6.x, 实际 ${SGLANG_VERSION})"; exit 1 ;;
esac

# ── 环境变量 (论文训练流程需要) ───────────────────────────────────────────
export PYTHONPATH="$(get_pythonpath):${PYTHONPATH:-}"
TORCH_CUDA_LIB="$(get_torch_cuda_lib)"
CONDA_LIB="$(get_conda_lib)"
export LD_LIBRARY_PATH="${CONDA_LIB}:${TORCH_CUDA_LIB}:${LD_LIBRARY_PATH:-}"
export PYTHONBUFFERED=16
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_CUMEM_ENABLE=0
export MASTER_ADDR=127.0.0.1

SMOKE_OUTPUT="${OUTPUT_ROOT}/smoke_test_faithful"
LOG_DIR="${SMOKE_OUTPUT}/logs"
mkdir -p "${LOG_DIR}"

cd "${SLIME_DIR}"
source "${SLIME_DIR}/scripts/models/qwen3-1.7B.sh"

echo ""
echo "========================================="
echo " Step 5 (faithful): Smoke test (2 quick runs)"
echo "   - pure_opd (mask=none, ratio=1.0)"
echo "   - ta_opd   (mask=dlearn_high, ratio=0.10)"
echo "========================================="

# ── Helper: run a single smoke test ──────────────────────────────────────
run_smoke() {
  local name=$1 mask=$2 ratio=$3
  local save_dir="${SMOKE_OUTPUT}/${name}"
  local teacher_gpu=0 student_gpus="1,2,3"
  local teacher_port=13141 nccl_port=23141
  local ray_port=26379 dash_port=8265
  local teacher_pid=""

  echo "─── Smoke: ${name} (mask=${mask}, ratio=${ratio}) ───"

  if [[ -f "${save_dir}/latest_checkpointed_iteration.txt" ]]; then
    echo "  Already completed (iter=$(cat "${save_dir}/latest_checkpointed_iteration.txt")). Skipping."
    return 0
  fi

  cleanup_smoke() {
    set +e
    [[ -n "${teacher_pid}" ]] && kill "${teacher_pid}" 2>/dev/null
    pkill -f "sglang.launch_server.*--port ${teacher_port}" 2>/dev/null || true
    ray stop --force 2>/dev/null || true
  }
  trap cleanup_smoke EXIT

  ulimit -n 100000 2>/dev/null || true
  ray stop --force 2>/dev/null || true

  mkdir -p "${save_dir}" "${LOG_DIR}"

  echo "  Starting teacher SGLang on GPU ${teacher_gpu}..."
  # 严格 sglang 0.5.10 参数: --nccl-port, 不带 --attention-backend triton
  CUDA_VISIBLE_DEVICES="${teacher_gpu}" python3 -m sglang.launch_server \
    --model-path "${TEACHER_MODEL}" --host 0.0.0.0 --port "${teacher_port}" \
    --nccl-port "${nccl_port}" --tp 1 --chunked-prefill-size 4096 \
    --mem-fraction-static "${TEACHER_MEM_FRACTION}" --cuda-graph-max-bs "${TEACHER_CUDA_GRAPH_MAX_BS}" \
    > "${LOG_DIR}/${name}_teacher.log" 2>&1 &
  teacher_pid=$!

  for _ in $(seq 1 180); do
    if ! kill -0 "${teacher_pid}" 2>/dev/null; then
      echo "  ERROR: teacher exited early" >&2
      tail -50 "${LOG_DIR}/${name}_teacher.log" >&2
      return 1
    fi
    curl -sf "http://127.0.0.1:${teacher_port}/health_generate" >/dev/null && break
    sleep 5
  done
  echo "  Teacher ready."

  echo "  Starting Ray cluster (3 GPUs: 2 actor + 1 rollout)..."
  CUDA_VISIBLE_DEVICES="${student_gpus}" ray start --head \
    --node-ip-address 127.0.0.1 --port="${ray_port}" --num-gpus 3 \
    --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port="${dash_port}" \
    --object-manager-port=28076 --node-manager-port=28077 \
    --dashboard-agent-listen-port=28078 --dashboard-agent-grpc-port=28079 \
    --metrics-export-port=28080 --temp-dir="/tmp/slime_ray_${ray_port}"

  local run_name="smoke_${name}"
  local job_id="${run_name}"

  local tip_args=()
  if [[ "${mask}" != "none" && "${mask}" != "full" ]]; then
    tip_args+=(
      --opd-topk-metrics-k 16
      --opd-token-bank-dir "${save_dir}/token_bank"
      --opd-token-bank-format csv
      --opd-token-bank-pair-id qwen3_4b_to_qwen3_1p7b
      --opd-budget-mask "${mask}"
      --opd-budget-ratio "${ratio}"
      --opd-budget-mask-seed 42
      --opd-budget-gamma 0.5
      --opd-compat-proxy mass
      --opd-metric-normalization batch_quantile
    )
  fi

  echo "  Submitting training job (non-colocate: actor=2GPU, rollout=1GPU)..."
  CUDA_VISIBLE_DEVICES="${student_gpus}" ray job submit \
    --address="http://127.0.0.1:${dash_port}" \
    --submission-id "${job_id}" --no-wait \
    --runtime-env-json="{\"env_vars\":{\"PYTHONPATH\":\"${PYTHONPATH}\",\"LD_LIBRARY_PATH\":\"${LD_LIBRARY_PATH}\",\"CUDA_DEVICE_MAX_CONNECTIONS\":\"1\",\"NCCL_CUMEM_ENABLE\":\"0\",\"PYTORCH_CUDA_ALLOC_CONF\":\"expandable_segments:True\"}}" \
    -- python3 train.py \
    --actor-num-nodes 1 --actor-num-gpus-per-node 2 --rollout-num-gpus 1 \
    --num-gpus-per-node "${NUM_GPUS_PER_NODE}" \
    --seed 1234 \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "${STUDENT_HF}" --ref-load "${STUDENT_TORCH_DIST}" \
    --load "${STUDENT_TORCH_DIST}" --save "${save_dir}" \
    --save-interval 1 --start-rollout-id 0 \
    --prompt-data "${PROMPT_DATA}" --input-key prompt --apply-chat-template \
    --rollout-shuffle --rollout-seed 42 --num-rollout 4 \
    --rollout-batch-size 4 --n-samples-per-prompt 2 \
    --rollout-max-response-len 1024 --rollout-temperature 1.0 \
    --global-batch-size 8 --balance-data \
    --optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1 \
    --adam-beta1 0.9 --adam-beta2 0.98 \
    --advantage-estimator grpo --use-opd --opd-type sglang --opd-kl-coef 1.0 \
    --use-kl-loss --kl-loss-coef 0.00 --kl-loss-type low_var_kl \
    --entropy-coef 0.00 --eps-clip 0.2 --eps-clip-high 0.28 \
    "${tip_args[@]}" \
    --qkv-format bshd --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 \
    --context-parallel-size 1 --expert-model-parallel-size 1 --expert-tensor-parallel-size 1 \
    --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
    --micro-batch-size 1 \
    --rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static "${ROLLOUT_MEM_FRACTION}" \
    --sglang-cuda-graph-max-bs "${SGLANG_CUDA_GRAPH_MAX_BS}" --sglang-enable-metrics \
    --attention-dropout 0.0 --hidden-dropout 0.0 \
    --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 \
    --attention-backend flash --no-rope-fusion --transformer-impl local \
    --no-masked-softmax-fusion --no-persist-layer-norm --no-gradient-accumulation-fusion \
    --megatron-to-hf-mode raw --no-save-optim \
    --custom-rm-path slime.rollout.on_policy_distillation.reward_func \
    --custom-reward-post-process-path slime.rollout.on_policy_distillation.post_process_rewards \
    --rm-url "http://127.0.0.1:${teacher_port}/generate" \
    > "${LOG_DIR}/${name}_submit.log" 2>&1

  echo "  Waiting for Ray job to complete..."
  while true; do
    local status_out
    status_out="$(ray job status --address="http://127.0.0.1:${dash_port}" "${job_id}" 2>&1 || true)"
    if echo "${status_out}" | grep -q "Status for job.*: SUCCEEDED"; then
      echo "  SUCCEEDED"
      break
    elif echo "${status_out}" | grep -q "Status for job.*: FAILED"; then
      echo "  FAILED" >&2
      ray job logs --address="http://127.0.0.1:${dash_port}" "${job_id}" 2>/dev/null | tail -100 >&2 || true
      return 1
    elif echo "${status_out}" | grep -q "Status for job.*: STOPPED"; then
      echo "  STOPPED" >&2
      return 1
    fi
    sleep 10
  done

  cleanup_smoke
  trap - EXIT
  echo "  Smoke test '${name}' PASSED"
}

# ── Run smoke tests ──────────────────────────────────────────────────────
run_smoke "pure_opd" "none" "1.0"
run_smoke "ta_opd" "dlearn_high" "0.10"

# ── Verify ────────────────────────────────────────────────────────────────
echo ""
echo "=== Smoke test verification ==="
PASS=0; FAIL=0
for name in pure_opd ta_opd; do
  f="${SMOKE_OUTPUT}/${name}/latest_checkpointed_iteration.txt"
  if [[ -f "$f" ]]; then
    echo "  PASS  ${name}: latest_iteration=$(cat "$f")"
    PASS=$((PASS + 1))
  else
    echo "  FAIL  ${name}: no checkpoint"
    FAIL=$((FAIL + 1))
  fi
done

echo ""
if [[ "${FAIL}" -eq 0 ]]; then
  echo "========================================="
  echo " All smoke tests PASSED! (${PASS}/${PASS})"
  echo "========================================="
else
  echo "========================================="
  echo " ${FAIL} smoke test(s) FAILED!"
  echo "========================================="
  exit 1
fi

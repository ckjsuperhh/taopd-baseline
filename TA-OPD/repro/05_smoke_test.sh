#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"
activate_env

echo "========================================="
echo " Step 5: Smoke test (2 quick runs)"
echo "========================================="

export PYTHONPATH="$(get_pythonpath):${PYTHONPATH:-}"
TORCH_CUDA_LIB="$(get_torch_cuda_lib)"
export LD_LIBRARY_PATH="${TORCH_CUDA_LIB}:${LD_LIBRARY_PATH:-}"
export PYTHONBUFFERED=16
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_CUMEM_ENABLE=0
export MASTER_ADDR=127.0.0.1

SMOKE_OUTPUT="${OUTPUT_ROOT}/smoke_test"
LOG_DIR="${SMOKE_OUTPUT}/logs"
mkdir -p "${LOG_DIR}"

cd "${SLIME_DIR}"
source "${SLIME_DIR}/scripts/models/qwen3-1.7B.sh"

# ── Helper: run a single smoke test ──────────────────────────────────────
run_smoke() {
  local name=$1 mask=$2 ratio=$3
  local save_dir="${SMOKE_OUTPUT}/${name}"
  local teacher_gpu=0 student_gpus="1,2"
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

  ray stop --force 2>/dev/null || true

  mkdir -p "${save_dir}" "${LOG_DIR}"

  echo "  Starting teacher SGLang on GPU ${teacher_gpu}..."
  CUDA_VISIBLE_DEVICES="${teacher_gpu}" python3 -m sglang.launch_server \
    --model-path "${TEACHER_MODEL}" --host 0.0.0.0 --port "${teacher_port}" \
    --nccl-port "${nccl_port}" --tp 1 --chunked-prefill-size 4096 \
    --mem-fraction-static 0.55 --cuda-graph-max-bs 8 \
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

  echo "  Starting Ray cluster..."
  CUDA_VISIBLE_DEVICES="${student_gpus}" ray start --head \
    --node-ip-address 127.0.0.1 --port="${ray_port}" --num-gpus 2 \
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

  echo "  Submitting training job..."
  CUDA_VISIBLE_DEVICES="${student_gpus}" ray job submit \
    --address="http://127.0.0.1:${dash_port}" \
    --submission-id "${job_id}" --no-wait \
    --runtime-env-json="{\"env_vars\":{\"PYTHONPATH\":\"${PYTHONPATH}\",\"LD_LIBRARY_PATH\":\"${LD_LIBRARY_PATH}\",\"CUDA_DEVICE_MAX_CONNECTIONS\":\"1\",\"NCCL_CUMEM_ENABLE\":\"0\"}}" \
    -- python3 train.py \
    --actor-num-nodes 1 --actor-num-gpus-per-node 2 --rollout-num-gpus 1 \
    --num-gpus-per-node 2 --colocate \
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
    --rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static 0.45 \
    --sglang-cuda-graph-max-bs 8 --sglang-enable-metrics \
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
    if echo "${status_out}" | grep -qi "succeeded"; then
      echo "  SUCCEEDED"
      break
    elif echo "${status_out}" | grep -qi "failed"; then
      echo "  FAILED" >&2
      ray job logs --address="http://127.0.0.1:${dash_port}" "${job_id}" 2>/dev/null | tail -100 >&2 || true
      return 1
    elif echo "${status_out}" | grep -qi "stopped"; then
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

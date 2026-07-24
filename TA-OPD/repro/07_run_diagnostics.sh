#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"
activate_env

echo "========================================="
echo " Step 7: Diagnostic runs (heldout + gsm8k)"
echo "========================================="

export PYTHONPATH="$(get_pythonpath):${PYTHONPATH:-}"
TORCH_CUDA_LIB="$(get_torch_cuda_lib)"
export LD_LIBRARY_PATH="${TORCH_CUDA_LIB}:${LD_LIBRARY_PATH:-}"
export PYTHONBUFFERED=16
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_CUMEM_ENABLE=0
export MASTER_ADDR=127.0.0.1

cd "${SLIME_DIR}"
source "${SLIME_DIR}/scripts/models/qwen3-1.7B.sh"

_wait_ray_job() {
  local job_id=$1 dash_port=$2
  while true; do
    local out
    out="$(ray job status --address="http://127.0.0.1:${dash_port}" "${job_id}" 2>&1 || true)"
    if echo "${out}" | grep -qi "succeeded"; then return 0; fi
    if echo "${out}" | grep -qi "failed\|stopped"; then
      ray job logs --address="http://127.0.0.1:${dash_port}" "${job_id}" 2>/dev/null | tail -100 >&2 || true
      return 1
    fi
    sleep 10
  done
}

# ══════════════════════════════════════════════════════════════════════════
# Run a single diagnostic
# ══════════════════════════════════════════════════════════════════════════
run_diagnostic() {
  local name=$1 prompt_data=$2 pair_id=$3 diag_seed=$4
  local save_dir="${DIAG_OUTPUT_ROOT}/${name}"
  local diag_log="${OUTPUT_ROOT}/logs/diag_${name}.log"
  mkdir -p "${save_dir}" "${OUTPUT_ROOT}/logs"

  if [[ -f "${save_dir}/latest_checkpointed_iteration.txt" ]] && \
     [[ -d "${save_dir}/fixed_context" ]]; then
    echo "  Diagnostic '${name}' already completed. Skipping."
    return 0
  fi

  local teacher_gpu=0 student_gpus="1,2" eval_gpus="0,1"
  compute_ports 0 200
  local teacher_pid=""

  _diag_cleanup() {
    set +e
    [[ -n "${teacher_pid}" ]] && kill "${teacher_pid}" 2>/dev/null || true
    pkill -f "sglang.launch_server.*--port ${TEACHER_PORT}" 2>/dev/null || true
    ray stop --force 2>/dev/null || true
    trap - EXIT
  }

  echo "─── Diagnostic: ${name} (seed=${diag_seed}) ───"

  ray stop --force 2>/dev/null || true

  echo "  Starting teacher..."
  CUDA_VISIBLE_DEVICES="${teacher_gpu}" python3 -m sglang.launch_server \
    --model-path "${TEACHER_MODEL}" --host 0.0.0.0 --port "${TEACHER_PORT}" \
    --nccl-port "${TEACHER_NCCL_PORT}" --tp 1 --chunked-prefill-size 4096 \
    --mem-fraction-static "${DIAG_TEACHER_MEM_FRACTION}" \
    --cuda-graph-max-bs "${DIAG_TEACHER_CUDA_GRAPH_MAX_BS}" \
    > "${diag_log}.teacher" 2>&1 &
  teacher_pid=$!

  for _ in $(seq 1 180); do
    if ! kill -0 "${teacher_pid}" 2>/dev/null; then
      echo "  ERROR: teacher exited" >&2; _diag_cleanup; return 1
    fi
    curl -sf "http://127.0.0.1:${TEACHER_PORT}/health_generate" >/dev/null && break
    sleep 5
  done

  echo "  Starting Ray..."
  CUDA_VISIBLE_DEVICES="${student_gpus}" ray start --head \
    --node-ip-address 127.0.0.1 --port="${RAY_PORT}" --num-gpus 2 \
    --disable-usage-stats --dashboard-host=0.0.0.0 \
    --dashboard-port="${RAY_DASHBOARD_PORT}" \
    --object-manager-port="${RAY_OBJECT_PORT}" --node-manager-port="${RAY_NODE_PORT}" \
    --dashboard-agent-listen-port="${RAY_AGENT_LISTEN}" \
    --dashboard-agent-grpc-port="${RAY_AGENT_GRPC}" \
    --metrics-export-port="${RAY_METRICS_PORT}" \
    --temp-dir="/tmp/slime_ray_${RAY_PORT}"

  local job_id="diag_${name}"

  echo "  Training (${DIAG_NUM_ROLLOUT} rollouts, max_len=${DIAG_ROLLOUT_MAX_RESPONSE_LEN})..."
  CUDA_VISIBLE_DEVICES="${student_gpus}" ray job submit \
    --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" \
    --submission-id "${job_id}" --no-wait \
    --runtime-env-json="{\"env_vars\":{\"PYTHONPATH\":\"${PYTHONPATH}\",\"LD_LIBRARY_PATH\":\"${LD_LIBRARY_PATH}\",\"CUDA_DEVICE_MAX_CONNECTIONS\":\"1\",\"NCCL_CUMEM_ENABLE\":\"0\"}}" \
    -- python3 train.py \
    --actor-num-nodes 1 --actor-num-gpus-per-node 2 --rollout-num-gpus 1 \
    --num-gpus-per-node 2 --colocate \
    --seed "${diag_seed}" \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "${STUDENT_HF}" --ref-load "${STUDENT_TORCH_DIST}" \
    --load "${STUDENT_TORCH_DIST}" --save "${save_dir}" \
    --save-interval "${DIAG_SAVE_INTERVAL}" --start-rollout-id 0 \
    --prompt-data "${prompt_data}" --input-key prompt --apply-chat-template \
    --rollout-shuffle --rollout-seed "${diag_seed}" \
    --num-rollout "${DIAG_NUM_ROLLOUT}" \
    --rollout-batch-size "${DIAG_ROLLOUT_BATCH_SIZE}" \
    --n-samples-per-prompt "${DIAG_N_SAMPLES_PER_PROMPT}" \
    --rollout-max-response-len "${DIAG_ROLLOUT_MAX_RESPONSE_LEN}" \
    --rollout-temperature 1.0 \
    --global-batch-size "${DIAG_GLOBAL_BATCH_SIZE}" --balance-data \
    --optimizer adam --lr "${LR}" --lr-decay-style "${LR_DECAY_STYLE}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --adam-beta1 "${ADAM_BETA1}" --adam-beta2 "${ADAM_BETA2}" \
    --advantage-estimator "${ADVANTAGE_ESTIMATOR}" --use-opd --opd-type sglang \
    --opd-kl-coef "${OPD_KL_COEF}" --use-kl-loss \
    --kl-loss-coef "${KL_LOSS_COEF}" --kl-loss-type "${KL_LOSS_TYPE}" \
    --entropy-coef "${ENTROPY_COEF}" \
    --eps-clip "${EPS_CLIP}" --eps-clip-high "${EPS_CLIP_HIGH}" \
    --opd-topk-metrics-k "${DIAG_OPD_TOPK_METRICS_K}" \
    --opd-token-bank-dir "${save_dir}/token_bank" \
    --opd-token-bank-format "${OPD_TOKEN_BANK_FORMAT}" \
    --opd-token-bank-pair-id "${pair_id}" \
    --opd-token-bank-raw-topk \
    --opd-exact-cmass \
    --opd-exact-cmass-max-union "${DIAG_OPD_EXACT_CMASS_MAX_UNION}" \
    --opd-exact-cmass-overflow fallback \
    --opd-teacher-name "${TEACHER_MODEL}" --opd-student-name "${STUDENT_HF}" \
    --opd-compat-proxy "${OPD_COMPAT_PROXY}" \
    --opd-metric-normalization "${OPD_METRIC_NORMALIZATION}" \
    --qkv-format bshd --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 \
    --context-parallel-size 1 --expert-model-parallel-size 1 --expert-tensor-parallel-size 1 \
    --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
    --micro-batch-size "${DIAG_MICRO_BATCH_SIZE}" \
    --rollout-num-gpus-per-engine 1 \
    --sglang-mem-fraction-static "${DIAG_ROLLOUT_MEM_FRACTION}" \
    --sglang-cuda-graph-max-bs "${DIAG_SGLANG_CUDA_GRAPH_MAX_BS}" --sglang-enable-metrics \
    --attention-dropout 0.0 --hidden-dropout 0.0 \
    --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 \
    --attention-backend flash --no-rope-fusion --transformer-impl local \
    --no-masked-softmax-fusion --no-persist-layer-norm --no-gradient-accumulation-fusion \
    --megatron-to-hf-mode raw --no-save-optim \
    --save-debug-rollout-data "${save_dir}/debug_rollout_data/{rollout_id}.pt" \
    --custom-rm-path slime.rollout.on_policy_distillation.reward_func \
    --custom-reward-post-process-path slime.rollout.on_policy_distillation.post_process_rewards \
    --rm-url "http://127.0.0.1:${TEACHER_PORT}/generate" \
    > "${diag_log}" 2>&1

  echo "  Waiting for training..."
  if ! _wait_ray_job "${job_id}" "${RAY_DASHBOARD_PORT}"; then
    echo "  Training FAILED" >&2; _diag_cleanup; return 1
  fi

  # Stop training infrastructure before eval
  kill "${teacher_pid}" 2>/dev/null || true
  pkill -f "sglang.launch_server.*--port ${TEACHER_PORT}" 2>/dev/null || true
  ray stop --force 2>/dev/null || true
  trap - EXIT

  # ── Post-training eval pipeline (4 steps) ──
  local fc_dir="${save_dir}/fixed_context"
  mkdir -p "${fc_dir}"

  echo "  Step 1/4: Exporting context bank..."
  python3 "${TOOLS_DIR}/export_fixed_context_bank.py" \
    --debug-rollout-data "${save_dir}/debug_rollout_data/*.pt" \
    --output "${fc_dir}/context_bank.parquet" \
    --max-samples 300

  echo "  Step 2/4: Scoring theta0 (baseline)..."
  CUDA_VISIBLE_DEVICES="${eval_gpus}" python3 "${TOOLS_DIR}/eval_fixed_context_bank.py" \
    --context-bank "${fc_dir}/context_bank.parquet" \
    --student "${STUDENT_HF}" --teacher "${TEACHER_MODEL}" \
    --output "${fc_dir}/theta0_metrics.parquet" \
    --student-device cuda:0 --teacher-device cuda:1 \
    --dtype bfloat16 --topk 16 --max-samples 300 \
    --max-response-tokens 192 --trust-remote-code

  echo "  Step 3/4: Scoring trained checkpoint..."
  CUDA_VISIBLE_DEVICES="${eval_gpus}" python3 "${TOOLS_DIR}/eval_fixed_context_from_megatron.py" \
    --checkpoint-root "${save_dir}" \
    --iteration latest \
    --origin-hf-dir "${STUDENT_HF}" \
    --teacher-hf-dir "${TEACHER_MODEL}" \
    --context-bank "${fc_dir}/context_bank.parquet" \
    --baseline-metrics "${fc_dir}/theta0_metrics.parquet" \
    --output-dir "${fc_dir}" \
    --student-device cuda:0 --teacher-device cuda:1 \
    --dtype bfloat16 --topk 16 --force-convert

  echo "  Step 4/4: Gain analysis (bootstrap 1000)..."
  python3 "${TOOLS_DIR}/analyze_fixed_context_gain.py" \
    --input "${fc_dir}/fixed_context_metrics.parquet" \
    --output-dir "${fc_dir}/gain" \
    --bootstrap 1000 --seed "${BOOTSTRAP_SEED}" --use-quadrant

  echo "  Diagnostic '${name}' COMPLETE"
}

# ══════════════════════════════════════════════════════════════════════════
# Main execution
# ══════════════════════════════════════════════════════════════════════════
mkdir -p "${DIAG_OUTPUT_ROOT}"

echo "[1/2] Running heldout-300 diagnostic..."
run_diagnostic "heldout_4b_to_1p7b_300" \
  "${HELDOUT_DATA}" \
  "qwen3_4b_to_qwen3_1p7b_heldout300" \
  "${DIAG_HELDOUT_SEED}"

echo ""
echo "[2/2] Running GSM8K-300 diagnostic..."
run_diagnostic "gsm8k_4b_to_1p7b_300" \
  "${GSM8K_DATA}" \
  "qwen3_4b_to_qwen3_1p7b_gsm8k300" \
  "${DIAG_GSM8K_SEED}"

echo ""
echo "========================================="
echo " Diagnostics complete!"
echo " Context bank: ${COMMON_CONTEXT}"
echo " Baseline:     ${BASELINE_METRICS}"
echo "========================================="

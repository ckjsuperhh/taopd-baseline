#!/usr/bin/env bash
set -eo pipefail  # 不能用 -u：conda 内部 deactivate 脚本有 unbound variable
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"
activate_env

echo "========================================="
echo " Step 6: Main training sweep (84 runs)"
echo " 28 (mask,ratio) × 3 seeds × 2 lanes"
echo "========================================="

export PYTHONPATH="$(get_pythonpath):${PYTHONPATH:-}"
TORCH_CUDA_LIB="$(get_torch_cuda_lib)"
CONDA_LIB="$(get_conda_lib)"
export LD_LIBRARY_PATH="${CONDA_LIB}:${TORCH_CUDA_LIB}:${LD_LIBRARY_PATH:-}"
export PYTHONBUFFERED=16
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_CUMEM_ENABLE=0
export MASTER_ADDR=127.0.0.1

SWEEP_ROOT="${OUTPUT_ROOT}/main_sweep"
LOG_DIR="${SWEEP_ROOT}/logs"
mkdir -p "${LOG_DIR}"

cd "${SLIME_DIR}"
source "${SLIME_DIR}/scripts/models/qwen3-1.7B.sh"

# ══════════════════════════════════════════════════════════════════════════
# Phase 0: Generate context bank & baseline metrics (if missing)
# ══════════════════════════════════════════════════════════════════════════
init_context_bank() {
  if [[ -f "${COMMON_CONTEXT}" ]] && [[ -f "${BASELINE_METRICS}" ]]; then
    echo "[Phase 0] Context bank and baseline metrics already exist. Skipping init."
    return 0
  fi

  echo "[Phase 0] Generating context bank via mini diagnostic run..."
  local init_save="${DIAG_OUTPUT_ROOT}/init_context_bank"
  local init_log="${LOG_DIR}/init_context_bank.log"
  mkdir -p "${init_save}"

  local teacher_gpu=${LANE_A_TEACHER_GPU}
  local student_gpus="${LANE_A_STUDENT_GPUS}"
  compute_ports ${LANE_A_PORT_BASE} 0

  # Cleanup function
  local teacher_pid=""
  _init_cleanup() {
    set +e
    [[ -n "${teacher_pid}" ]] && kill "${teacher_pid}" 2>/dev/null || true
    pkill -f "sglang.launch_server.*--port ${TEACHER_PORT}" 2>/dev/null || true
    ray stop --force 2>/dev/null || true
    trap - EXIT
  }

  ulimit -n 100000 2>/dev/null || true
  ray stop --force 2>/dev/null || true

  echo "  Starting teacher..."
  CUDA_VISIBLE_DEVICES="${teacher_gpu}" python3 -m sglang.launch_server \
    --model-path "${TEACHER_MODEL}" --host 0.0.0.0 --port "${TEACHER_PORT}" \
    --nccl-port "${TEACHER_NCCL_PORT}" --tp 1 --chunked-prefill-size 4096 \
    --mem-fraction-static "${DIAG_TEACHER_MEM_FRACTION}" \
    --cuda-graph-max-bs "${DIAG_TEACHER_CUDA_GRAPH_MAX_BS}" \
    --disable-piecewise-cuda-graph \
    > "${init_log}.teacher" 2>&1 &
  teacher_pid=$!

  for _ in $(seq 1 180); do
    if ! kill -0 "${teacher_pid}" 2>/dev/null; then
      echo "  ERROR: teacher exited" >&2; tail -50 "${init_log}.teacher" >&2
      _init_cleanup; return 1
    fi
    curl -sf "http://127.0.0.1:${TEACHER_PORT}/health_generate" >/dev/null && break
    sleep 5
  done

  echo "  Starting Ray..."
  CUDA_VISIBLE_DEVICES="${student_gpus}" ray start --head \
    --node-ip-address 127.0.0.1 --port="${RAY_PORT}" --num-gpus 3 \
    --disable-usage-stats --dashboard-host=0.0.0.0 \
    --dashboard-port="${RAY_DASHBOARD_PORT}" \
    --object-manager-port="${RAY_OBJECT_PORT}" --node-manager-port="${RAY_NODE_PORT}" \
    --dashboard-agent-listen-port="${RAY_AGENT_LISTEN}" \
    --dashboard-agent-grpc-port="${RAY_AGENT_GRPC}" \
    --metrics-export-port="${RAY_METRICS_PORT}" \
    --temp-dir="/tmp/slime_ray_${RAY_PORT}"

  echo "  Running mini diagnostic (20 rollouts, debug data)..."
  CUDA_VISIBLE_DEVICES="${student_gpus}" ray job submit \
    --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" \
    --submission-id "init_context_bank" --no-wait \
    --runtime-env-json="{\"env_vars\":{\"PYTHONPATH\":\"${PYTHONPATH}\",\"LD_LIBRARY_PATH\":\"${LD_LIBRARY_PATH}\",\"CUDA_DEVICE_MAX_CONNECTIONS\":\"1\",\"NCCL_CUMEM_ENABLE\":\"0\",\"PYTORCH_CUDA_ALLOC_CONF\":\"expandable_segments:True\"}}" \
    -- python3 train.py \
    --actor-num-nodes 1 --actor-num-gpus-per-node 2 --rollout-num-gpus 1 \
    --num-gpus-per-node "${NUM_GPUS_PER_NODE}" \
    --seed "${DIAG_HELDOUT_SEED}" \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "${STUDENT_HF}" --ref-load "${STUDENT_TORCH_DIST}" \
    --load "${STUDENT_TORCH_DIST}" --save "${init_save}" \
    --save-interval 20 --start-rollout-id 0 \
    --prompt-data "${HELDOUT_DATA}" --input-key prompt --apply-chat-template \
    --rollout-shuffle --rollout-seed "${DIAG_HELDOUT_SEED}" \
    --num-rollout 20 --rollout-batch-size 15 --n-samples-per-prompt 1 \
    --rollout-max-response-len 256 --rollout-temperature 1.0 \
    --global-batch-size 15 --balance-data \
    --optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1 \
    --adam-beta1 0.9 --adam-beta2 0.98 \
    --advantage-estimator grpo --use-opd --opd-type sglang --opd-kl-coef 1.0 \
    --use-kl-loss --kl-loss-coef 0.00 --kl-loss-type low_var_kl \
    --entropy-coef 0.00 --eps-clip 0.2 --eps-clip-high 0.28 \
    --opd-topk-metrics-k 16 \
    --qkv-format bshd --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 \
    --context-parallel-size 1 --expert-model-parallel-size 1 --expert-tensor-parallel-size 1 \
    --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
    --micro-batch-size 1 \
    --rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static "${DIAG_ROLLOUT_MEM_FRACTION}" \
    --sglang-cuda-graph-max-bs "${DIAG_SGLANG_CUDA_GRAPH_MAX_BS}" --sglang-enable-metrics \
    --attention-dropout 0.0 --hidden-dropout 0.0 \
    --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 \
    --attention-backend flash --no-rope-fusion --transformer-impl local \
    --no-masked-softmax-fusion --no-persist-layer-norm --no-gradient-accumulation-fusion \
    --megatron-to-hf-mode raw --no-save-optim \
    --save-debug-rollout-data "${init_save}/debug_rollout_data/{rollout_id}.pt" \
    --custom-rm-path slime.rollout.on_policy_distillation.reward_func \
    --custom-reward-post-process-path slime.rollout.on_policy_distillation.post_process_rewards \
    --rm-url "http://127.0.0.1:${TEACHER_PORT}/generate"

  echo "  Waiting for mini diagnostic..."
  _wait_ray_job "init_context_bank" "${RAY_DASHBOARD_PORT}" "${init_log}"

  echo "  Exporting context bank..."
  mkdir -p "$(dirname "${COMMON_CONTEXT}")"
  python3 "${TOOLS_DIR}/export_fixed_context_bank.py" \
    --debug-rollout-data "${init_save}/debug_rollout_data/*.pt" \
    --output "${COMMON_CONTEXT}" \
    --max-samples 300

  echo "  Scoring theta0 (baseline metrics)..."
  CUDA_VISIBLE_DEVICES="${LANE_A_EVAL_GPUS}" python3 "${TOOLS_DIR}/eval_fixed_context_bank.py" \
    --context-bank "${COMMON_CONTEXT}" \
    --student "${STUDENT_HF}" --teacher "${TEACHER_MODEL}" \
    --output "${BASELINE_METRICS}" \
    --student-device cuda:0 --teacher-device cuda:1 \
    --dtype bfloat16 --topk 16 --max-samples 300 \
    --max-response-tokens 192 --trust-remote-code

  _init_cleanup
  echo "[Phase 0] Context bank and baseline metrics generated."
}

# ══════════════════════════════════════════════════════════════════════════
# Helper: wait for Ray job
# ══════════════════════════════════════════════════════════════════════════
_wait_ray_job() {
  local job_id=$1 dash_port=$2 log_file=$3
  while true; do
    local out
    out="$(ray job status --address="http://127.0.0.1:${dash_port}" "${job_id}" 2>&1 || true)"
    if echo "${out}" | grep -qi "succeeded"; then return 0; fi
    if echo "${out}" | grep -qi "failed\|stopped"; then
      echo "  Ray job ${job_id} FAILED" >&2
      ray job logs --address="http://127.0.0.1:${dash_port}" "${job_id}" 2>/dev/null | tail -100 >&2 || true
      return 1
    fi
    sleep 10
  done
}

# ══════════════════════════════════════════════════════════════════════════
# Core: run a single (mask, ratio, seed) training + eval
# ══════════════════════════════════════════════════════════════════════════
run_single() {
  local mask=$1 ratio=$2 seed_label=$3 seed=$4 rollout_seed=$5 mask_seed=$6
  local teacher_gpu=$7 student_gpus=$8 eval_gpus=$9 port_base=${10} idx=${11}

  local ratio_tag="${ratio/./}"
  local tag="k16_ratio${ratio_tag}_${seed_label}_${DATE_TAG}"
  local run_dir_name="${mask}_max64_${seed_label}"
  local save_dir="${SWEEP_ROOT}/${tag}/${run_dir_name}"
  local train_log="${LOG_DIR}/${tag}_${run_dir_name}.log"

  if [[ -f "${save_dir}/latest_checkpointed_iteration.txt" ]]; then
    echo "  SKIP ${mask}:${ratio}:${seed_label} — already done"
    return 0
  fi

  compute_ports "${port_base}" "${idx}"

  echo "  START ${mask}:${ratio}:${seed_label} → ${save_dir}"
  mkdir -p "${save_dir}"

  local teacher_pid=""
  _run_cleanup() {
    set +e
    [[ -n "${teacher_pid}" ]] && kill "${teacher_pid}" 2>/dev/null || true
    pkill -f "sglang.launch_server.*--port ${TEACHER_PORT}" 2>/dev/null || true
    pkill -f "/tmp/slime_ray_${RAY_PORT}" 2>/dev/null || true
    trap - EXIT
  }

  ulimit -n 100000 2>/dev/null || true
  pkill -f "/tmp/slime_ray_${RAY_PORT}" 2>/dev/null || true
  pkill -f "sglang.launch_server.*--port ${TEACHER_PORT}" 2>/dev/null || true

  # Start teacher
  CUDA_VISIBLE_DEVICES="${teacher_gpu}" python3 -m sglang.launch_server \
    --model-path "${TEACHER_MODEL}" --host 0.0.0.0 --port "${TEACHER_PORT}" \
    --nccl-port "${TEACHER_NCCL_PORT}" --tp 1 --chunked-prefill-size 4096 \
    --mem-fraction-static "${TEACHER_MEM_FRACTION}" \
    --cuda-graph-max-bs "${TEACHER_CUDA_GRAPH_MAX_BS}" \
    --disable-piecewise-cuda-graph \
    > "${train_log}.teacher" 2>&1 &
  teacher_pid=$!

  for _ in $(seq 1 180); do
    if ! kill -0 "${teacher_pid}" 2>/dev/null; then
      echo "  ERROR: teacher exited" >&2
      _run_cleanup; return 1
    fi
    curl -sf "http://127.0.0.1:${TEACHER_PORT}/health_generate" >/dev/null && break
    sleep 5
  done

  # Start Ray
  local ray_gpu_count
  IFS=',' read -ra _gpu_arr <<< "${student_gpus}"
  ray_gpu_count=${#_gpu_arr[@]}

  CUDA_VISIBLE_DEVICES="${student_gpus}" ray start --head \
    --node-ip-address 127.0.0.1 --port="${RAY_PORT}" \
    --num-gpus "${ray_gpu_count}" --disable-usage-stats \
    --dashboard-host=0.0.0.0 --dashboard-port="${RAY_DASHBOARD_PORT}" \
    --object-manager-port="${RAY_OBJECT_PORT}" --node-manager-port="${RAY_NODE_PORT}" \
    --dashboard-agent-listen-port="${RAY_AGENT_LISTEN}" \
    --dashboard-agent-grpc-port="${RAY_AGENT_GRPC}" \
    --metrics-export-port="${RAY_METRICS_PORT}" \
    --temp-dir="/tmp/slime_ray_${RAY_PORT}"

  local job_id="${tag}_${run_dir_name}"

  # Build budget mask args
  local tip_args=()
  if [[ "${mask}" != "full" ]]; then
    tip_args+=(
      --opd-budget-mask "${mask}"
      --opd-budget-ratio "${ratio}"
      --opd-budget-mask-seed "${mask_seed}"
    )
  fi

  CUDA_VISIBLE_DEVICES="${student_gpus}" ray job submit \
    --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" \
    --submission-id "${job_id}" --no-wait \
    --runtime-env-json="{\"env_vars\":{\"PYTHONPATH\":\"${PYTHONPATH}\",\"LD_LIBRARY_PATH\":\"${LD_LIBRARY_PATH}\",\"CUDA_DEVICE_MAX_CONNECTIONS\":\"1\",\"NCCL_CUMEM_ENABLE\":\"0\",\"PYTORCH_CUDA_ALLOC_CONF\":\"expandable_segments:True\"}}" \
    -- python3 train.py \
    --actor-num-nodes 1 --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}" \
    --rollout-num-gpus "${ROLLOUT_NUM_GPUS}" \
    --num-gpus-per-node "${NUM_GPUS_PER_NODE}" \
    --seed "${seed}" \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "${STUDENT_HF}" --ref-load "${STUDENT_TORCH_DIST}" \
    --load "${STUDENT_TORCH_DIST}" --save "${save_dir}" \
    --save-interval "${MAIN_SAVE_INTERVAL}" --start-rollout-id 0 \
    --prompt-data "${PROMPT_DATA}" --input-key prompt --apply-chat-template \
    --rollout-shuffle --rollout-seed "${rollout_seed}" \
    --num-rollout "${MAIN_NUM_ROLLOUT}" \
    --rollout-batch-size "${MAIN_ROLLOUT_BATCH_SIZE}" \
    --n-samples-per-prompt "${MAIN_N_SAMPLES_PER_PROMPT}" \
    --rollout-max-response-len "${MAIN_ROLLOUT_MAX_RESPONSE_LEN}" \
    --rollout-temperature "${MAIN_ROLLOUT_TEMPERATURE}" \
    --global-batch-size "${MAIN_GLOBAL_BATCH_SIZE}" --balance-data \
    --optimizer adam --lr "${LR}" --lr-decay-style "${LR_DECAY_STYLE}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --adam-beta1 "${ADAM_BETA1}" --adam-beta2 "${ADAM_BETA2}" \
    --advantage-estimator "${ADVANTAGE_ESTIMATOR}" --use-opd --opd-type sglang \
    --opd-kl-coef "${OPD_KL_COEF}" --use-kl-loss \
    --kl-loss-coef "${KL_LOSS_COEF}" --kl-loss-type "${KL_LOSS_TYPE}" \
    --entropy-coef "${ENTROPY_COEF}" \
    --eps-clip "${EPS_CLIP}" --eps-clip-high "${EPS_CLIP_HIGH}" \
    --opd-topk-metrics-k "${MAIN_OPD_TOPK_METRICS_K}" \
    --opd-token-bank-dir "${save_dir}/token_bank" \
    --opd-token-bank-format "${OPD_TOKEN_BANK_FORMAT}" \
    --opd-token-bank-pair-id qwen3_4b_to_qwen3_1p7b \
    --opd-teacher-name "${TEACHER_MODEL}" --opd-student-name "${STUDENT_HF}" \
    --opd-budget-gamma "${OPD_BUDGET_GAMMA}" \
    --opd-compat-proxy "${OPD_COMPAT_PROXY}" \
    --opd-metric-normalization "${OPD_METRIC_NORMALIZATION}" \
    "${tip_args[@]}" \
    --qkv-format bshd --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 \
    --context-parallel-size 1 --expert-model-parallel-size 1 --expert-tensor-parallel-size 1 \
    --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
    --micro-batch-size "${MAIN_MICRO_BATCH_SIZE}" \
    --rollout-num-gpus-per-engine 1 \
    --sglang-mem-fraction-static "${ROLLOUT_MEM_FRACTION}" \
    --sglang-cuda-graph-max-bs "${SGLANG_CUDA_GRAPH_MAX_BS}" --sglang-enable-metrics \
    --attention-dropout 0.0 --hidden-dropout 0.0 \
    --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 \
    --attention-backend flash --no-rope-fusion --transformer-impl local \
    --no-masked-softmax-fusion --no-persist-layer-norm --no-gradient-accumulation-fusion \
    --megatron-to-hf-mode raw --no-save-optim \
    --custom-rm-path slime.rollout.on_policy_distillation.reward_func \
    --custom-reward-post-process-path slime.rollout.on_policy_distillation.post_process_rewards \
    --rm-url "http://127.0.0.1:${TEACHER_PORT}/generate" \
    > "${train_log}" 2>&1

  echo "  Training submitted, waiting..."
  if ! _wait_ray_job "${job_id}" "${RAY_DASHBOARD_PORT}" "${train_log}"; then
    echo "  FAILED ${mask}:${ratio}:${seed_label}" >&2
    _run_cleanup; return 1
  fi

  echo "  Merging token bank..."
  python3 "${TOOLS_DIR}/merge_token_bank.py" "${save_dir}/token_bank" \
    --output "${save_dir}/token_bank/merged.parquet" \
    --drop-raw-topk 2>&1 || echo "  WARNING: token bank merge failed"

  # Fixed-context eval (if context bank exists)
  if [[ -f "${COMMON_CONTEXT}" ]] && [[ -f "${BASELINE_METRICS}" ]]; then
    local eval_dir="${SWEEP_ROOT}/${tag}/${run_dir_name}/fixed_context"
    echo "  Running fixed-context eval..."
    CUDA_VISIBLE_DEVICES="${eval_gpus}" python3 "${TOOLS_DIR}/eval_fixed_context_from_megatron.py" \
      --checkpoint-root "${save_dir}" \
      --iteration latest \
      --origin-hf-dir "${STUDENT_HF}" \
      --teacher-hf-dir "${TEACHER_MODEL}" \
      --context-bank "${COMMON_CONTEXT}" \
      --baseline-metrics "${BASELINE_METRICS}" \
      --output-dir "${eval_dir}" \
      --student-device cuda:0 --teacher-device cuda:1 \
      --dtype bfloat16 --topk 16 --force-convert \
      >> "${train_log}" 2>&1 || echo "  WARNING: eval failed"

    echo "  Running gain analysis..."
    python3 "${TOOLS_DIR}/analyze_fixed_context_gain.py" \
      --input "${eval_dir}/fixed_context_metrics.parquet" \
      --output-dir "${eval_dir}/gain" \
      --bootstrap 1000 --seed "${BOOTSTRAP_SEED}" --use-quadrant \
      >> "${train_log}" 2>&1 || echo "  WARNING: gain analysis failed"
  else
    echo "  Context bank not available; skipping eval. Run 07_run_diagnostics.sh first."
  fi

  echo "  DONE ${mask}:${ratio}:${seed_label}"
  _run_cleanup
}

# ══════════════════════════════════════════════════════════════════════════
# Main execution
# ══════════════════════════════════════════════════════════════════════════

# Phase 0: init context bank
init_context_bank

# Seeds array
SEEDS=(
  "seed1:${SEED1_SEED}:${SEED1_ROLLOUT_SEED}:${SEED1_MASK_SEED}"
  "seed2:${SEED2_SEED}:${SEED2_ROLLOUT_SEED}:${SEED2_MASK_SEED}"
  "seed3:${SEED3_SEED}:${SEED3_ROLLOUT_SEED}:${SEED3_MASK_SEED}"
)

TOTAL=$((${#ALL_RUNS[@]} * ${#SEEDS[@]}))
DONE=0; FAILED=0; idx=0

echo ""
echo "[Phase 1] Starting main sweep: ${TOTAL} runs, 2 lanes (non-colocate)"
echo "  Lane A: Teacher GPU${LANE_A_TEACHER_GPU}, Actor+Rollout GPU ${LANE_A_STUDENT_GPUS}"
echo "  Lane B: Teacher GPU${LANE_B_TEACHER_GPU}, Actor+Rollout GPU ${LANE_B_STUDENT_GPUS}"
echo ""

# Run in pairs: Lane A gets even-indexed seeds, Lane B gets odd
# Strategy: for each (mask,ratio), run seed1+seed2 in parallel, then seed3
for run_spec in "${ALL_RUNS[@]}"; do
  IFS=: read -r mask ratio <<< "${run_spec}"
  idx=$((idx + 1))

  # Pair 1: seed1 (Lane A) + seed2 (Lane B) in parallel
  (
    IFS=: read -r sl s rs ms <<< "${SEEDS[0]}"
    run_single "${mask}" "${ratio}" "${sl}" "${s}" "${rs}" "${ms}" \
      "${LANE_A_TEACHER_GPU}" "${LANE_A_STUDENT_GPUS}" "${LANE_A_EVAL_GPUS}" \
      "${LANE_A_PORT_BASE}" "${idx}"
  ) &
  PID_A=$!

  (
    IFS=: read -r sl s rs ms <<< "${SEEDS[1]}"
    run_single "${mask}" "${ratio}" "${sl}" "${s}" "${rs}" "${ms}" \
      "${LANE_B_TEACHER_GPU}" "${LANE_B_STUDENT_GPUS}" "${LANE_B_EVAL_GPUS}" \
      "${LANE_B_PORT_BASE}" "${idx}"
  ) &
  PID_B=$!

  wait ${PID_A} || FAILED=$((FAILED + 1))
  wait ${PID_B} || FAILED=$((FAILED + 1))
  DONE=$((DONE + 2))

  # seed3 (Lane A)
  (
    IFS=: read -r sl s rs ms <<< "${SEEDS[2]}"
    run_single "${mask}" "${ratio}" "${sl}" "${s}" "${rs}" "${ms}" \
      "${LANE_A_TEACHER_GPU}" "${LANE_A_STUDENT_GPUS}" "${LANE_A_EVAL_GPUS}" \
      "${LANE_A_PORT_BASE}" "$((idx + 100))"
  ) &
  wait $! || FAILED=$((FAILED + 1))
  DONE=$((DONE + 1))

  echo "=== Progress: ${DONE}/${TOTAL} done, ${FAILED} failed ==="
done

echo ""
echo "========================================="
echo " Main sweep complete!"
echo " Total: ${TOTAL}, Done: ${DONE}, Failed: ${FAILED}"
echo "========================================="

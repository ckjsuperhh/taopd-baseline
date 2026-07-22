#!/usr/bin/env bash
set -euo pipefail

# Sampled-token OPD baseline:
#   teacher: Qwen3-4B served by SGLang on one GPU
#   student rollout/inference: Qwen3-4B SGLang engine on one GPU
#   student training: Qwen3-4B Megatron on two GPUs
#
# Usage:
#   cd /path/to/slime-main
#   bash examples/on_policy_distillation/run-qwen3-4B-sampled-opd-sglang.sh
#
# Useful overrides:
#   NUM_ROLLOUT=100 ROLLOUT_MAX_RESPONSE_LEN=4096 bash ...
#   OPD_TOPK_METRICS_K=16 OPD_TOKEN_BANK_DIR=/path/to/token_bank bash ...
#   OPD_TOPK_METRICS_K=16 OPD_BUDGET_MASK=ca_softor OPD_BUDGET_RATIO=0.5 bash ...

SLIME_DIR="${SLIME_DIR:-/path/to/slime-main}"
MEGATRON_LM_DIR="${MEGATRON_LM_DIR:-/path/to/Megatron-LM}"
CONDA_SH="${CONDA_SH:-/path/to/miniconda3/etc/profile.d/conda.sh}"
# The server's opsd env is enough for Megatron conversion, but does not include SGLang.
# The existing verl env has SGLang/Ray/Torch and can import slime + Megatron via PYTHONPATH.
CONDA_ENV="${CONDA_ENV:-verl}"
FALLBACK_SITE_PACKAGES="${FALLBACK_SITE_PACKAGES:-/path/to/site-packages}"

TEACHER_MODEL="${TEACHER_MODEL:-/path/to/models/Qwen3/4B}"
STUDENT_HF="${STUDENT_HF:-/path/to/models/Qwen3/4B}"
STUDENT_TORCH_DIST="${STUDENT_TORCH_DIST:-/path/to/models/Qwen3/4B/Qwen3-4B_torch_dist}"

DATA_DIR="${DATA_DIR:-/path/to/data/grpo_processed}"
PROMPT_DATA="${PROMPT_DATA:-${DATA_DIR}/train.parquet}"

RUN_NAME="${RUN_NAME:-qwen3_4b_student_qwen3_4b_teacher_sampled_opd_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/path/to/outputs/slime_opd}"
SAVE_DIR="${SAVE_DIR:-${OUTPUT_ROOT}/${RUN_NAME}}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/logs}"

# Physical GPU layout. Ray only sees RAY_GPUS; the teacher server only sees TEACHER_GPU.
TEACHER_GPU="${TEACHER_GPU:-7}"
RAY_GPUS="${RAY_GPUS:-0,1,2}"
ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-2}"
ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-1}"

TEACHER_IP="${TEACHER_IP:-127.0.0.1}"
TEACHER_PORT="${TEACHER_PORT:-13141}"
TEACHER_NCCL_PORT="${TEACHER_NCCL_PORT:-23141}"
RAY_PORT="${RAY_PORT:-26379}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
RAY_OBJECT_MANAGER_PORT="${RAY_OBJECT_MANAGER_PORT:-28076}"
RAY_NODE_MANAGER_PORT="${RAY_NODE_MANAGER_PORT:-28077}"
RAY_DASHBOARD_AGENT_LISTEN_PORT="${RAY_DASHBOARD_AGENT_LISTEN_PORT:-28078}"
RAY_DASHBOARD_AGENT_GRPC_PORT="${RAY_DASHBOARD_AGENT_GRPC_PORT:-28079}"
RAY_METRICS_EXPORT_PORT="${RAY_METRICS_EXPORT_PORT:-28080}"
RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/slime_ray_${RAY_PORT}}"

# Smoke-run defaults. Increase these for a real baseline sweep.
NUM_ROLLOUT="${NUM_ROLLOUT:-2}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-4}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-2}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-1024}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1}"

TEACHER_MEM_FRACTION="${TEACHER_MEM_FRACTION:-0.55}"
TEACHER_CUDA_GRAPH_MAX_BS="${TEACHER_CUDA_GRAPH_MAX_BS:-8}"
ROLLOUT_MEM_FRACTION="${ROLLOUT_MEM_FRACTION:-0.45}"
SGLANG_CUDA_GRAPH_MAX_BS="${SGLANG_CUDA_GRAPH_MAX_BS:-8}"

# TIP compatibility / token-bank instrumentation. Defaults keep the exact sampled-token OPD baseline.
OPD_TOPK_METRICS_K="${OPD_TOPK_METRICS_K:-0}"
OPD_TOKEN_BANK_DIR="${OPD_TOKEN_BANK_DIR:-}"
OPD_TOKEN_BANK_FORMAT="${OPD_TOKEN_BANK_FORMAT:-csv}"
OPD_TOKEN_BANK_RAW_TOPK="${OPD_TOKEN_BANK_RAW_TOPK:-0}"
OPD_TOKEN_BANK_PAIR_ID="${OPD_TOKEN_BANK_PAIR_ID:-qwen3_teacher_to_qwen3_4b}"
OPD_EXACT_CMASS="${OPD_EXACT_CMASS:-0}"
OPD_EXACT_CMASS_MAX_UNION="${OPD_EXACT_CMASS_MAX_UNION:-4096}"
OPD_EXACT_CMASS_OVERFLOW="${OPD_EXACT_CMASS_OVERFLOW:-fallback}"
OPD_BUDGET_MASK="${OPD_BUDGET_MASK:-none}"
OPD_BUDGET_RATIO="${OPD_BUDGET_RATIO:-1.0}"
OPD_BUDGET_GAMMA="${OPD_BUDGET_GAMMA:-0.5}"
OPD_COMPAT_PROXY="${OPD_COMPAT_PROXY:-mass}"
OPD_METRIC_NORMALIZATION="${OPD_METRIC_NORMALIZATION:-batch_quantile}"
SAVE_DEBUG_ROLLOUT_DATA="${SAVE_DEBUG_ROLLOUT_DATA:-0}"
DEBUG_ROLLOUT_ONLY="${DEBUG_ROLLOUT_ONLY:-0}"

export PYTHONBUFFERED=16
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"

TEACHER_PID=""

cleanup() {
  set +e
  if [[ -n "${TEACHER_PID}" ]] && kill -0 "${TEACHER_PID}" >/dev/null 2>&1; then
    kill "${TEACHER_PID}" >/dev/null 2>&1
    wait "${TEACHER_PID}" >/dev/null 2>&1
  fi
  pkill -f "sglang.launch_server.*--port ${TEACHER_PORT}" >/dev/null 2>&1
  ray stop --force >/dev/null 2>&1
}
trap cleanup EXIT

require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "ERROR: missing directory: $1" >&2
    exit 1
  fi
}

if [[ -f "${CONDA_SH}" ]]; then
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
fi

ACTIVE_SITE_PACKAGES="$(python3 - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"
export PYTHONPATH="${MEGATRON_LM_DIR}:${SLIME_DIR}:${ACTIVE_SITE_PACKAGES}:${FALLBACK_SITE_PACKAGES}:${PYTHONPATH:-}"

require_dir "${SLIME_DIR}"
require_dir "${MEGATRON_LM_DIR}"
require_dir "${TEACHER_MODEL}"
require_dir "${STUDENT_HF}"
require_dir "${STUDENT_TORCH_DIST}"

if [[ ! -f "${STUDENT_TORCH_DIST}/latest_checkpointed_iteration.txt" ]]; then
  echo "ERROR: invalid Megatron checkpoint: ${STUDENT_TORCH_DIST}" >&2
  exit 1
fi

if [[ ! -f "${PROMPT_DATA}" ]]; then
  echo "ERROR: missing prompt data: ${PROMPT_DATA}" >&2
  exit 1
fi

mkdir -p "${SAVE_DIR}" "${LOG_DIR}" "${RAY_TEMP_DIR}"
cd "${SLIME_DIR}"

IFS=',' read -r -a RAY_GPU_LIST <<< "${RAY_GPUS}"
RAY_NUM_GPUS="${#RAY_GPU_LIST[@]}"
REQUIRED_RAY_GPUS=$((ACTOR_NUM_GPUS_PER_NODE + ROLLOUT_NUM_GPUS))
if [[ "${RAY_NUM_GPUS}" -lt "${REQUIRED_RAY_GPUS}" ]]; then
  echo "ERROR: RAY_GPUS has ${RAY_NUM_GPUS} GPU(s), but actor+rollout needs ${REQUIRED_RAY_GPUS}." >&2
  exit 1
fi

source "${SLIME_DIR}/scripts/models/qwen3-4B.sh"

TEACHER_LOG="${LOG_DIR}/${RUN_NAME}_teacher_sglang.log"
echo "=== sampled-token OPD baseline ==="
echo "SLIME_DIR          = ${SLIME_DIR}"
echo "MEGATRON_LM_DIR    = ${MEGATRON_LM_DIR}"
echo "TEACHER_MODEL      = ${TEACHER_MODEL}"
echo "STUDENT_HF         = ${STUDENT_HF}"
echo "STUDENT_TORCH_DIST = ${STUDENT_TORCH_DIST}"
echo "PROMPT_DATA        = ${PROMPT_DATA}"
echo "SAVE_DIR           = ${SAVE_DIR}"
echo "TEACHER_GPU        = ${TEACHER_GPU}"
echo "RAY_GPUS           = ${RAY_GPUS}"
echo "TEACHER_LOG        = ${TEACHER_LOG}"
echo "TEACHER_NCCL_PORT  = ${TEACHER_NCCL_PORT}"
echo "RAY_PORT           = ${RAY_PORT}"
echo "RAY_DASHBOARD_PORT = ${RAY_DASHBOARD_PORT}"
echo "RAY_TEMP_DIR       = ${RAY_TEMP_DIR}"
echo "OPD_TOPK_METRICS_K = ${OPD_TOPK_METRICS_K}"
echo "OPD_BUDGET_MASK    = ${OPD_BUDGET_MASK}"
echo "OPD_PAIR_ID        = ${OPD_TOKEN_BANK_PAIR_ID}"
echo "OPD_EXACT_CMASS    = ${OPD_EXACT_CMASS}"
if [[ -n "${OPD_TOKEN_BANK_DIR}" ]]; then
  echo "OPD_TOKEN_BANK_DIR = ${OPD_TOKEN_BANK_DIR}"
fi
echo "SAVE_DEBUG_ROLLOUT = ${SAVE_DEBUG_ROLLOUT_DATA}"
echo "DEBUG_ROLLOUT_ONLY = ${DEBUG_ROLLOUT_ONLY}"
echo

ray stop --force >/dev/null 2>&1 || true

CUDA_VISIBLE_DEVICES="${TEACHER_GPU}" python3 -m sglang.launch_server \
  --model-path "${TEACHER_MODEL}" \
  --host 0.0.0.0 \
  --port "${TEACHER_PORT}" \
  --nccl-port "${TEACHER_NCCL_PORT}" \
  --tp 1 \
  --chunked-prefill-size 4096 \
  --mem-fraction-static "${TEACHER_MEM_FRACTION}" \
  --cuda-graph-max-bs "${TEACHER_CUDA_GRAPH_MAX_BS}" \
  > "${TEACHER_LOG}" 2>&1 &
TEACHER_PID=$!

echo "Starting teacher SGLang server on GPU ${TEACHER_GPU} (pid=${TEACHER_PID})..."
for _ in $(seq 1 180); do
  if ! kill -0 "${TEACHER_PID}" >/dev/null 2>&1; then
    echo "ERROR: teacher SGLang server exited early. Last log lines:" >&2
    tail -n 80 "${TEACHER_LOG}" >&2
    exit 1
  fi
  if curl -sf "http://${TEACHER_IP}:${TEACHER_PORT}/health_generate" >/dev/null; then
    break
  fi
  echo "Waiting for teacher server..."
  tail -n 10 "${TEACHER_LOG}" || true
  sleep 5
done

curl -sf "http://${TEACHER_IP}:${TEACHER_PORT}/get_model_info" || true
echo
echo "Teacher server is ready at http://${TEACHER_IP}:${TEACHER_PORT}."

CKPT_ARGS=(
  --hf-checkpoint "${STUDENT_HF}"
  --ref-load "${STUDENT_TORCH_DIST}"
  --load "${STUDENT_TORCH_DIST}"
  --save "${SAVE_DIR}"
  --save-interval "${SAVE_INTERVAL}"
  --start-rollout-id 0
)

ROLLOUT_ARGS=(
  --prompt-data "${PROMPT_DATA}"
  --input-key prompt
  --apply-chat-template
  --rollout-shuffle
  --num-rollout "${NUM_ROLLOUT}"
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
  --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
  --rollout-temperature "${ROLLOUT_TEMPERATURE}"
  --global-batch-size "${GLOBAL_BATCH_SIZE}"
  --balance-data
)

RM_ARGS=(
  --custom-rm-path slime.rollout.on_policy_distillation.reward_func
  --custom-reward-post-process-path slime.rollout.on_policy_distillation.post_process_rewards
  --rm-url "http://${TEACHER_IP}:${TEACHER_PORT}/generate"
)

PERF_ARGS=(
  --qkv-format bshd
  --tensor-model-parallel-size 1
  --pipeline-model-parallel-size 1
  --context-parallel-size 1
  --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --micro-batch-size "${MICRO_BATCH_SIZE}"
)

GRPO_ARGS=(
  --advantage-estimator grpo
  --use-opd
  --opd-type sglang
  --opd-kl-coef 1.0
  --use-kl-loss
  --kl-loss-coef 0.00
  --kl-loss-type low_var_kl
  --entropy-coef 0.00
  --eps-clip 0.2
  --eps-clip-high 0.28
)

TIP_COMPAT_ARGS=()
if [[ "${OPD_TOPK_METRICS_K}" != "0" || -n "${OPD_TOKEN_BANK_DIR}" || "${OPD_BUDGET_MASK}" != "none" ]]; then
  if [[ "${OPD_TOPK_METRICS_K}" == "0" ]]; then
    OPD_TOPK_METRICS_K=16
  fi
  if [[ -z "${OPD_TOKEN_BANK_DIR}" ]]; then
    OPD_TOKEN_BANK_DIR="${SAVE_DIR}/token_bank"
  fi
  TIP_COMPAT_ARGS+=(
    --opd-topk-metrics-k "${OPD_TOPK_METRICS_K}"
    --opd-token-bank-dir "${OPD_TOKEN_BANK_DIR}"
    --opd-token-bank-format "${OPD_TOKEN_BANK_FORMAT}"
    --opd-token-bank-pair-id "${OPD_TOKEN_BANK_PAIR_ID}"
    --opd-teacher-name "${TEACHER_MODEL}"
    --opd-student-name "${STUDENT_HF}"
    --opd-budget-mask "${OPD_BUDGET_MASK}"
    --opd-budget-ratio "${OPD_BUDGET_RATIO}"
    --opd-budget-gamma "${OPD_BUDGET_GAMMA}"
    --opd-compat-proxy "${OPD_COMPAT_PROXY}"
    --opd-metric-normalization "${OPD_METRIC_NORMALIZATION}"
  )
  if [[ "${OPD_TOKEN_BANK_RAW_TOPK}" == "1" ]]; then
    TIP_COMPAT_ARGS+=(--opd-token-bank-raw-topk)
  fi
  if [[ "${OPD_EXACT_CMASS}" == "1" ]]; then
    TIP_COMPAT_ARGS+=(
      --opd-exact-cmass
      --opd-exact-cmass-max-union "${OPD_EXACT_CMASS_MAX_UNION}"
      --opd-exact-cmass-overflow "${OPD_EXACT_CMASS_OVERFLOW}"
    )
  fi
fi

DEBUG_ARGS=()
if [[ "${SAVE_DEBUG_ROLLOUT_DATA}" == "1" ]]; then
  DEBUG_ARGS+=(
    --save-debug-rollout-data "${SAVE_DIR}/debug_rollout_data/{rollout_id}.pt"
  )
fi
if [[ "${DEBUG_ROLLOUT_ONLY}" == "1" ]]; then
  DEBUG_ARGS+=(
    --debug-rollout-only
  )
fi

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr 1e-6
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
)

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine 1
  --sglang-mem-fraction-static "${ROLLOUT_MEM_FRACTION}"
  --sglang-cuda-graph-max-bs "${SGLANG_CUDA_GRAPH_MAX_BS}"
  --sglang-enable-metrics
)

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
  --no-rope-fusion
  --transformer-impl local
  --no-masked-softmax-fusion
  --no-persist-layer-norm
  --no-gradient-accumulation-fusion
  --megatron-to-hf-mode raw
)

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
CUDA_VISIBLE_DEVICES="${RAY_GPUS}" ray start \
  --head \
  --node-ip-address "${MASTER_ADDR}" \
  --port="${RAY_PORT}" \
  --num-gpus "${RAY_NUM_GPUS}" \
  --disable-usage-stats \
  --dashboard-host=0.0.0.0 \
  --dashboard-port="${RAY_DASHBOARD_PORT}" \
  --object-manager-port="${RAY_OBJECT_MANAGER_PORT}" \
  --node-manager-port="${RAY_NODE_MANAGER_PORT}" \
  --dashboard-agent-listen-port="${RAY_DASHBOARD_AGENT_LISTEN_PORT}" \
  --dashboard-agent-grpc-port="${RAY_DASHBOARD_AGENT_GRPC_PORT}" \
  --metrics-export-port="${RAY_METRICS_EXPORT_PORT}" \
  --temp-dir="${RAY_TEMP_DIR}"

RAY_JOB_ID="${RAY_JOB_ID:-${RUN_NAME}}"
CUDA_VISIBLE_DEVICES="${RAY_GPUS}" ray job submit \
  --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" \
  --submission-id "${RAY_JOB_ID}" \
  --no-wait \
  --runtime-env-json="{\"env_vars\":{\"PYTHONPATH\":\"${PYTHONPATH}\",\"CUDA_DEVICE_MAX_CONNECTIONS\":\"${CUDA_DEVICE_MAX_CONNECTIONS}\",\"NCCL_CUMEM_ENABLE\":\"${NCCL_CUMEM_ENABLE}\"}}" \
  -- python3 train.py \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}" \
  --rollout-num-gpus "${ROLLOUT_NUM_GPUS}" \
  --num-gpus-per-node "${RAY_NUM_GPUS}" \
  "${MODEL_ARGS[@]}" \
  "${CKPT_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${GRPO_ARGS[@]}" \
  "${TIP_COMPAT_ARGS[@]}" \
  "${DEBUG_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${SGLANG_ARGS[@]}" \
  "${MISC_ARGS[@]}" \
  "${RM_ARGS[@]}"

echo "Submitted Ray job ${RAY_JOB_ID}; waiting for completion..."
while true; do
  STATUS_OUTPUT="$(ray job status --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" "${RAY_JOB_ID}" 2>&1 || true)"
  echo "${STATUS_OUTPUT}"
  JOB_STATUS="$(printf '%s\n' "${STATUS_OUTPUT}" | awk -F: '/Status for job/ {gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2}' | awk '{print $1}')"
  case "${JOB_STATUS}" in
    SUCCEEDED)
      break
      ;;
    FAILED|STOPPED)
      echo "ERROR: Ray job ${RAY_JOB_ID} ended with status ${JOB_STATUS}." >&2
      echo "Last Ray job logs:" >&2
      ray job logs --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" "${RAY_JOB_ID}" 2>/dev/null | tail -n 200 >&2 || true
      exit 1
      ;;
    RUNNING|PENDING)
      sleep 10
      ;;
    *)
      echo "ERROR: could not parse Ray job status for ${RAY_JOB_ID}." >&2
      exit 1
      ;;
  esac
done

echo
echo "=== sampled-token OPD run finished ==="
echo "SAVE_DIR=${SAVE_DIR}"
if [[ -f "${SAVE_DIR}/latest_checkpointed_iteration.txt" ]]; then
  echo "latest_checkpointed_iteration=$(cat "${SAVE_DIR}/latest_checkpointed_iteration.txt")"
fi

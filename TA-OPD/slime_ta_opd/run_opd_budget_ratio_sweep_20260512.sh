#!/usr/bin/env bash
set -euo pipefail

SLIME_DIR="${SLIME_DIR:-/path/to/slime-main}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/path/to/outputs/slime_opd}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/logs}"
TAG="${TAG:-budget_ratio_sweep_k16_seed1_20260512}"
SEED_LABEL="${SEED_LABEL:-seed1}"

PROMPT_DATA="${PROMPT_DATA:-/path/to/slime-main/data/DAPO-Math-17k-dedup/dapo_math_17k_dedup_slime.jsonl}"
STUDENT_HF="${STUDENT_HF:-/path/to/models/Qwen3/1.7B/Qwen_Qwen3-1.7B}"
TEACHER_MODEL="${TEACHER_MODEL:-/path/to/models/Qwen3/4B}"
COMMON_CONTEXT="${COMMON_CONTEXT:-/path/to/outputs/slime_opd/qwen3_1_7b_dapo_diag_k16_exact_20260510_084415/fixed_context/context_bank.parquet}"
BASELINE_METRICS="${BASELINE_METRICS:-/path/to/outputs/slime_opd/qwen3_1_7b_dapo_diag_k16_exact_20260510_084415/fixed_context/theta0_metrics.parquet}"

TEACHER_GPU="${TEACHER_GPU:-4}"
RAY_GPUS="${RAY_GPUS:-5}"
EVAL_GPUS="${EVAL_GPUS:-4,5}"
ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-1}"
ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-1}"
COLOCATE="${COLOCATE:-1}"
ALLOW_GLOBAL_RAY_STOP="${ALLOW_GLOBAL_RAY_STOP:-1}"
SKIP_EVAL="${SKIP_EVAL:-0}"

mkdir -p "${LOG_DIR}"
PIPELINE_LOG="${LOG_DIR}/${TAG}.log"

cd "${SLIME_DIR}"

source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate verl

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${PIPELINE_LOG}"
}

log_gpu() {
  log "GPU snapshot"
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader | tee -a "${PIPELINE_LOG}"
}

run_one() {
  local mask="$1"
  local ratio="$2"
  local idx="$3"
  local ratio_tag="${ratio/./}"
  local run_name="qwen3_1_7b_dapo_budget_k16_ratio${ratio_tag}_${mask}_max64_${SEED_LABEL}_${TAG}"
  local save_dir="${OUTPUT_ROOT}/${run_name}"
  local eval_dir="${OUTPUT_ROOT}/${TAG}/${mask}_ratio${ratio_tag}"
  local train_log="${LOG_DIR}/${run_name}_driver.log"
  local teacher_port="$((14141 + idx))"
  local teacher_nccl_port="$((24141 + idx))"
  local ray_port="$((26379 + idx))"
  local dash_port="$((8365 + idx))"
  local object_port="$((29076 + idx))"
  local node_port="$((29077 + idx))"
  local agent_listen_port="$((29078 + idx))"
  local agent_grpc_port="$((29079 + idx))"
  local metrics_port="$((29080 + idx))"

  log "START mask=${mask} ratio=${ratio} run=${run_name}"
  log_gpu

  RUN_NAME="${run_name}" \
  SAVE_DIR="${save_dir}" \
  PROMPT_DATA="${PROMPT_DATA}" \
  TEACHER_GPU="${TEACHER_GPU}" \
  RAY_GPUS="${RAY_GPUS}" \
  ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE}" \
  ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS}" \
  COLOCATE="${COLOCATE}" \
  ALLOW_GLOBAL_RAY_STOP="${ALLOW_GLOBAL_RAY_STOP}" \
  TEACHER_PORT="${teacher_port}" \
  TEACHER_NCCL_PORT="${teacher_nccl_port}" \
  RAY_PORT="${ray_port}" \
  RAY_DASHBOARD_PORT="${dash_port}" \
  RAY_OBJECT_MANAGER_PORT="${object_port}" \
  RAY_NODE_MANAGER_PORT="${node_port}" \
  RAY_DASHBOARD_AGENT_LISTEN_PORT="${agent_listen_port}" \
  RAY_DASHBOARD_AGENT_GRPC_PORT="${agent_grpc_port}" \
  RAY_METRICS_EXPORT_PORT="${metrics_port}" \
  NUM_ROLLOUT=50 \
  ROLLOUT_BATCH_SIZE=4 \
  N_SAMPLES_PER_PROMPT=2 \
  GLOBAL_BATCH_SIZE=8 \
  MICRO_BATCH_SIZE=1 \
  SAVE_INTERVAL=10 \
  ROLLOUT_MAX_RESPONSE_LEN=64 \
  OPD_TOPK_METRICS_K=16 \
  OPD_BUDGET_MASK="${mask}" \
  OPD_BUDGET_RATIO="${ratio}" \
  OPD_TOKEN_BANK_FORMAT=csv \
  bash examples/on_policy_distillation/run-qwen3-1.7B-sampled-opd-sglang.sh \
    > "${train_log}" 2>&1

  log "TRAIN_DONE mask=${mask} ratio=${ratio}"
  python3 tools/merge_token_bank.py "${save_dir}/token_bank" \
    --output "${save_dir}/token_bank/merged.parquet" \
    --drop-raw-topk \
    2>&1 | tee -a "${PIPELINE_LOG}"

  if [[ "${SKIP_EVAL}" == "1" ]]; then
    log "SKIP_EVAL mask=${mask} ratio=${ratio} save=${save_dir}"
    log "DONE mask=${mask} ratio=${ratio} save=${save_dir} eval=${eval_dir} skipped_eval=1"
    log_gpu
    return
  fi

  log "EVAL_START mask=${mask} ratio=${ratio}"
  CUDA_VISIBLE_DEVICES="${EVAL_GPUS}" python3 tools/eval_fixed_context_from_megatron.py \
    --checkpoint-root "${save_dir}" \
    --iteration latest \
    --origin-hf-dir "${STUDENT_HF}" \
    --teacher-hf-dir "${TEACHER_MODEL}" \
    --context-bank "${COMMON_CONTEXT}" \
    --baseline-metrics "${BASELINE_METRICS}" \
    --output-dir "${eval_dir}" \
    --student-device cuda:0 \
    --teacher-device cuda:1 \
    --dtype bfloat16 \
    --topk 16 \
    --force-convert \
    2>&1 | tee -a "${PIPELINE_LOG}"

  python3 tools/analyze_fixed_context_gain.py \
    --input "${eval_dir}/fixed_context_metrics.parquet" \
    --output-dir "${eval_dir}/gain" \
    --bootstrap 1000 \
    --seed 20260512 \
    --use-quadrant \
    2>&1 | tee -a "${PIPELINE_LOG}"

  log "DONE mask=${mask} ratio=${ratio} save=${save_dir} eval=${eval_dir}"
  log_gpu
}

log "PIPELINE_START tag=${TAG}"
log "seed_label=${SEED_LABEL}"
log "teacher_gpu=${TEACHER_GPU} ray_gpus=${RAY_GPUS} eval_gpus=${EVAL_GPUS}"
log "actor_gpus=${ACTOR_NUM_GPUS_PER_NODE} rollout_gpus=${ROLLOUT_NUM_GPUS} colocate=${COLOCATE} allow_global_ray_stop=${ALLOW_GLOBAL_RAY_STOP}"
log "skip_eval=${SKIP_EVAL}"
log "prompt_data=${PROMPT_DATA}"
log_gpu

RUN_LIST="${RUN_LIST:-dlearn_high:0.01:10 dlearn_high:0.05:20 dlearn_high:0.10:30 q3_highc:0.01:40 q3_highc:0.05:50}"
for item in ${RUN_LIST}; do
  IFS=: read -r mask ratio idx <<< "${item}"
  run_one "${mask}" "${ratio}" "${idx}"
done

log "PIPELINE_DONE tag=${TAG}"

#!/usr/bin/env bash
set -eo pipefail  # 不能用 -u：conda 内部 deactivate 脚本有 unbound variable
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh" 2>/dev/null || true

echo "============================================================"
echo " TA-OPD 4B→1.7B Full Reproduction Pipeline"
echo " Qwen3-4B (teacher) → Qwen3-1.7B (student)"
echo ""
echo " Steps 1-5, 7-9: 前台运行（可见输出）"
echo " Step 6 (84 次训练): 后台 tmux/nohup 静默运行"
echo " Started: $(date '+%F %T')"
echo "============================================================"
echo ""

# Allow starting from a specific step: bash run_all.sh [start_step]
START_STEP="${1:-1}"

STEPS=(
  "01_setup_env.sh:Environment setup (conda, torch, deps, patches)"
  "02_download_models.sh:Download Qwen3-4B + Qwen3-1.7B"
  "03_convert_student.sh:Convert student to torch_dist"
  "04_prepare_data.sh:Prepare DAPO-Math-17k + heldout + GSM8K"
  "05_smoke_test.sh:Smoke test (2 quick runs)"
  "06_run_all_training.sh:Main training sweep (84 runs, ~2-3 days)"
  "07_run_diagnostics.sh:Diagnostic runs (heldout + gsm8k)"
  "08_downstream_eval.sh:Downstream evaluation (GSM8K, AIME)"
  "09_aggregate_results.sh:Aggregate results and generate tables"
)

echo "Pipeline steps:"
for i in "${!STEPS[@]}"; do
  IFS=: read -r script desc <<< "${STEPS[$i]}"
  step_num=$((i + 1))
  if [[ "${step_num}" -lt "${START_STEP}" ]]; then
    echo "  [SKIP] Step ${step_num}: ${desc}"
  elif [[ "${step_num}" -eq 6 ]]; then
    echo "  [BG  ] Step ${step_num}: ${desc}  ← 后台运行"
  else
    echo "  [    ] Step ${step_num}: ${desc}"
  fi
done
echo ""

for i in "${!STEPS[@]}"; do
  IFS=: read -r script desc <<< "${STEPS[$i]}"
  step_num=$((i + 1))

  if [[ "${step_num}" -lt "${START_STEP}" ]]; then
    continue
  fi

  # ── Step 6: 训练扫描 → 后台 tmux/nohup 运行 ──────────────────────────
  if [[ "${step_num}" -eq 6 ]]; then
    TRAIN_LOG="${OUTPUT_ROOT}/logs/training_$(date +%Y%m%d_%H%M%S).log"
    mkdir -p "$(dirname "${TRAIN_LOG}")"
    TMUX_SESSION="ta_opd_train"

    echo "============================================================"
    echo " Step 6: ${desc} → 后台运行"
    echo "============================================================"
    echo ""

    if [[ -n "${TMUX:-}" ]] || command -v tmux &>/dev/null; then
      if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
        echo "  tmux session '${TMUX_SESSION}' already exists."
        echo "  Attach:  tmux attach -t ${TMUX_SESSION}"
        echo "  Skipping step 6 launch."
      else
        echo "  Launching in tmux session: ${TMUX_SESSION}"
        tmux new-session -d -s "${TMUX_SESSION}" \
          "bash '${SCRIPT_DIR}/${script}' 2>&1 | tee '${TRAIN_LOG}'"
        echo "  tmux session started."
      fi
    else
      echo "  tmux not found, using nohup."
      nohup bash "${SCRIPT_DIR}/${script}" > "${TRAIN_LOG}" 2>&1 &
      echo "  PID: $!"
    fi

    echo ""
    echo "  训练日志: ${TRAIN_LOG}"
    echo ""
    echo "  监控方式:"
    echo "    tmux attach -t ${TMUX_SESSION}   # 进入 tmux 查看"
    echo "    tail -f ${TRAIN_LOG}             # 跟踪日志"
    echo ""
    echo "  训练完成后运行后续步骤:"
    echo "    bash repro/run_all.sh 7"
    echo ""
    echo "============================================================"
    echo " Steps 1-5 全部完成 ✅"
    echo " Step 6 已在后台启动"
    echo " 训练完成后执行: bash repro/run_all.sh 7"
    echo "============================================================"
    exit 0
  fi

  # ── 其他步骤: 前台运行 ──────────────────────────────────────────────────
  echo "============================================================"
  echo " Step ${step_num}: ${desc}"
  echo " Script: ${script}"
  echo " Started: $(date '+%F %T')"
  echo "============================================================"
  echo ""

  if bash "${SCRIPT_DIR}/${script}"; then
    echo ""
    echo ">>> Step ${step_num} PASSED ($(date '+%F %T'))"
    echo ""
  else
    echo ""
    echo ">>> Step ${step_num} FAILED ($(date '+%F %T'))"
    echo ">>> To resume from this step:"
    echo ">>>   bash repro/run_all.sh ${step_num}"
    echo ""
    exit 1
  fi
done

echo "============================================================"
echo " ALL STEPS COMPLETE!"
echo " $(date '+%F %T')"
echo ""
echo " Results: ${OUTPUT_ROOT}/results"
echo "============================================================"

#!/usr/bin/env bash
set -euo pipefail

# Paper-facing OPD method entrypoint.
#
# This wrapper keeps the underlying teachability implementation unchanged and
# maps stable method names used in the paper/experiment plan to the existing
# slime budget-mask selectors.
#
# Method aliases:
#   pure_opd                 -> full OPD, no loss-mask pruning
#   teachability             -> Dlearn-high = normalized divergence * compatibility
#   entropy                  -> high student entropy selector
#   teachability_entropy     -> soft-OR(H, Dlearn)
#   teachability_entropy_split -> split budget between entropy and teachability
#   tip                      -> soft-OR(H, raw divergence)
#   q3_highc                 -> TIP-Q3 restricted high compatibility
#   divergence               -> raw high divergence
#   random                   -> random same-budget control

SLIME_DIR="${SLIME_DIR:-/path/to/slime-main}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/path/to/outputs/slime_opd}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/logs}"
TAG="${TAG:-teachability_method_suite_k16_seed1_20260515}"
SEED_LABEL="${SEED_LABEL:-seed1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Comma/space separated entries in method:ratio:idx format.
# Ratio is ignored for pure_opd and forced to 1.0.
METHOD_LIST="${METHOD_LIST:-pure_opd:1.0:10 teachability:0.10:20 entropy:0.10:30 teachability_entropy:0.10:40 tip:0.10:50}"

mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}/${TAG}"
MAPPING_CSV="${OUTPUT_ROOT}/${TAG}/method_mapping.csv"
RUN_LIST_BUILT=()

method_to_mask() {
  case "$1" in
    pure_opd|full_opd|full|none)
      echo "full"
      ;;
    teachability|teachability_high|dlearn|dlearn_high)
      echo "dlearn_high"
      ;;
    entropy|high_entropy)
      echo "entropy"
      ;;
    teachability_entropy|entropy_teachability|ta_softor|dlearn_entropy)
      echo "ca_softor"
      ;;
    teachability_entropy_split|entropy_teachability_split|ta_split)
      echo "split_budget_ca"
      ;;
    tip|tip_softor)
      echo "tip"
      ;;
    q3_highc|q3_highC)
      echo "q3_highc"
      ;;
    q3)
      echo "q3"
      ;;
    divergence|high_d|raw_d)
      echo "divergence"
      ;;
    random)
      echo "random"
      ;;
    compatibility|high_c)
      echo "compatibility"
      ;;
    *)
      echo "ERROR: unknown method '$1'" >&2
      return 1
      ;;
  esac
}

echo "method,mask,ratio,idx,notes" > "${MAPPING_CSV}"
for item in ${METHOD_LIST//,/ }; do
  IFS=: read -r method ratio idx <<< "${item}"
  if [[ -z "${method:-}" || -z "${ratio:-}" || -z "${idx:-}" ]]; then
    echo "Bad METHOD_LIST item: ${item}. Expected method:ratio:idx" >&2
    exit 1
  fi
  mask="$(method_to_mask "${method}")"
  notes=""
  if [[ "${mask}" == "full" ]]; then
    ratio="1.0"
    notes="full OPD; no budget loss mask; top-k metrics may still be logged"
  elif [[ "${mask}" == "ca_softor" ]]; then
    notes="soft-OR of normalized entropy and Dlearn"
  elif [[ "${mask}" == "split_budget_ca" ]]; then
    notes="budget split by OPD_BUDGET_GAMMA between entropy and teachability"
  fi
  echo "${method},${mask},${ratio},${idx},${notes}" >> "${MAPPING_CSV}"
  RUN_LIST_BUILT+=("${mask}:${ratio}:${idx}")
done

export TAG
export SEED_LABEL
export RUN_LIST="${RUN_LIST_BUILT[*]}"
export OPD_TOPK_METRICS_K="${OPD_TOPK_METRICS_K:-16}"
export OPD_TOKEN_BANK_FORMAT="${OPD_TOKEN_BANK_FORMAT:-csv}"

echo "Launching OPD method suite"
echo "TAG=${TAG}"
echo "SEED_LABEL=${SEED_LABEL}"
echo "METHOD_LIST=${METHOD_LIST}"
echo "RUN_LIST=${RUN_LIST}"
echo "MAPPING_CSV=${MAPPING_CSV}"

cd "${SLIME_DIR}"
bash "${SCRIPT_DIR}/run_opd_budget_ratio_sweep.sh"

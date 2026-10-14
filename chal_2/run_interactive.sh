#!/usr/bin/env bash
# =============================================================================
# Interactive GPU run -- the one to use while debugging.
#
#   bash run_interactive.sh                 # learning mode: ~1 h, verbose
#   bash run_interactive.sh --full           # full settings (~14 h)
#   bash run_interactive.sh --stage diffuse  # one stage, on the GPU
#
# If you are not already on a GPU node this re-launches itself through srun, so
# it works straight from the login node. Output streams live, which is the
# point: you watch RFdiffusion denoise against the inter-protomer groove,
# ProteinMPNN pick histidines, and ESMFold accept or reject them.
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

MODE="learn"
STAGE="all"
PASSTHRU=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --full)   MODE="full"; shift ;;
    --stage)  STAGE="$2"; shift 2 ;;
    --time)   WALL="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) PASSTHRU+=("$1"); shift ;;
  esac
done
WALL="${WALL:-$([[ "$MODE" == full ]] && echo 16:00:00 || echo 03:00:00)}"

# --- Re-launch through srun if we are not on a GPU node ----------------------
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "not on a compute node; requesting an interactive V100 (--time $WALL)"
  exec srun --partition=k2-gpu-v100 --gres=gpu:v100:1 \
            --cpus-per-task=8 --mem=64G --time="$WALL" --pty \
            bash "$0" $([[ "$MODE" == full ]] && echo --full) \
            --stage "$STAGE" "${PASSTHRU[@]+"${PASSTHRU[@]}"}"
fi

[[ -f env.sh ]] || { echo "ERROR: run 'bash setup_env.sh' first" >&2; exit 1; }
# shellcheck disable=SC1091
source env.sh

if command -v module &>/dev/null; then
  module load libs/nvidia-cuda/11.8.0 2>/dev/null || module load cuda/11.8 2>/dev/null || true
  module load apps/anaconda3/2024.06/bin 2>/dev/null || module load anaconda3 2>/dev/null || true
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

echo "=============================================================="
echo " interactive run  mode=$MODE  stage=$STAGE"
echo " node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "=============================================================="

COMMON=()
[[ "$MODE" == "learn" ]] && COMMON+=(--learn)
COMMON+=("${PASSTHRU[@]+"${PASSTHRU[@]}"}")

# Stage fold needs the other env; everything else uses rfdiff.
pick_py () { [[ "$1" == "fold" ]] && echo "$TNF_FOLD_PY" || echo "$TNF_RFDIFF_PY"; }

echo ">>> preflight selftest"
"$TNF_RFDIFF_PY" -u pipeline.py --selftest "${COMMON[@]+"${COMMON[@]}"}"

if [[ "$STAGE" == "all" ]]; then
  for s in prepare diffuse mpnn fold score submit; do
    echo; echo ">>> stage $s  ($(date -u +%H:%M:%S))"
    "$(pick_py "$s")" -u pipeline.py --stage "$s" "${COMMON[@]+"${COMMON[@]}"}"
  done
else
  "$(pick_py "$STAGE")" -u pipeline.py --stage "$STAGE" "${COMMON[@]+"${COMMON[@]}"}"
fi

echo
echo "done. results/submission.csv:"
head -5 results/submission.csv 2>/dev/null || echo "  (not written yet)"

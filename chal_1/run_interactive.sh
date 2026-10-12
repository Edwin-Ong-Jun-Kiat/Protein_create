#!/usr/bin/env bash
# =============================================================================
# Interactive GPU run -- the one to use while learning.
#
#   bash run_interactive.sh              # learning mode: ~30-60 min, verbose
#   bash run_interactive.sh --full       # full competition settings (8-12 h)
#   bash run_interactive.sh --stage fold # just one stage, on the GPU
#
# If you are not already on a GPU node this re-launches itself through srun,
# so you can run it straight from the login node. Output streams live, which
# is the whole point: you watch RFdiffusion denoise, watch ProteinMPNN pick
# sequences, and watch ESMFold accept or reject them.
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# --- Parse our own flags; everything else is forwarded to pipeline.py -------
MODE="learn"
STAGE="all"
PASSTHRU=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --full)  MODE="full"; shift ;;
    --learn) MODE="learn"; shift ;;
    --stage) STAGE="$2"; shift 2 ;;
    *)       PASSTHRU+=("$1"); shift ;;
  esac
done

# --- Get ourselves onto a GPU if we are not on one -------------------------
have_gpu() { command -v nvidia-smi &>/dev/null && nvidia-smi -L &>/dev/null; }

# Kelvin-2 uses a SEPARATE partition for interactive work (k2-gpu-interactive)
# from the batch GPU partition (k2-gpu-v100, used by submit.sbatch).
# Override any of these from the shell if the cluster policy changes:
#   EGFR_PART=... EGFR_GRES=... EGFR_TIME=... EGFR_CPUS=... EGFR_MEMPERCPU=...
# Set EGFR_GRES="" to omit --gres entirely (some interactive partitions attach
# a GPU automatically and reject an explicit request).
PART="${EGFR_PART:-k2-gpu-interactive}"
GRES="${EGFR_GRES-gpu:1}"          # note: ${VAR-default}, so EGFR_GRES="" omits it
WALL="${EGFR_TIME:-04:00:00}"
CPUS="${EGFR_CPUS:-8}"
MEMPC="${EGFR_MEMPERCPU:-8G}"      # 8 cpus x 8G = 64G, enough for a V100 run

srun_cmd=(srun -p "$PART" -N 1 -n 1 -c "$CPUS" --mem-per-cpu="$MEMPC"
          --time="$WALL")
[[ -n "$GRES" ]] && srun_cmd+=(--gres="$GRES")

# Already inside a SLURM allocation? That is either an sbatch job or an
# `srun --pty bash` session. Either way an allocation cannot gain a GPU after
# the fact, so a nested srun would not help.
if [[ -n "${SLURM_JOB_ID:-}" && -z "${EGFR_IN_SRUN:-}" ]]; then
  if have_gpu; then
    echo "Inside SLURM job $SLURM_JOB_ID and a GPU is visible -- running here."
    export EGFR_IN_SRUN=1          # do not nest another srun below
  else
    cat >&2 <<EOF

ERROR: you are inside SLURM job $SLURM_JOB_ID, but no GPU is visible, and an
       allocation cannot acquire one after it has started.

If you got here with srun, ask for a GPU explicitly:
    exit
    srun -p $PART --gres=gpu:1 -N 1 -n 1 -c $CPUS --mem-per-cpu=$MEMPC \\
         --time=$WALL --pty bash
    cd $ROOT && bash run_interactive.sh

If you got here with sbatch, use the batch entrypoint instead -- it carries
the right #SBATCH headers (partition k2-gpu-v100, --gres=gpu:v100:1):
    sbatch submit.sbatch
    sbatch --export=ALL,LEARN=1 submit.sbatch

Check what the partitions offer with:  sinfo -o "%20P %10G %6D %N"
EOF
    exit 1
  fi
fi

if ! have_gpu; then
  if [[ -n "${EGFR_IN_SRUN:-}" ]]; then
    echo "ERROR: srun gave us a node with no visible GPU." >&2
    echo "Partition '$PART' may not attach a GPU with --gres='$GRES'." >&2
    echo "Inspect with:  sinfo -o \"%20P %10G %6D %N\"" >&2
    echo "Then retry, e.g.:  EGFR_GRES=gpu:v100:1 bash $0" >&2
    exit 1
  fi
  if command -v srun &>/dev/null; then
    echo "No GPU here -- requesting an interactive GPU on '$PART'..."
    echo "  ${srun_cmd[*]} --pty bash"
    echo "(if this queues for a while, check with: squeue -u $USER)"
    export EGFR_IN_SRUN=1
    exec "${srun_cmd[@]}" --pty bash "$0" "--$MODE" --stage "$STAGE" \
         "${PASSTHRU[@]+"${PASSTHRU[@]}"}"
  fi
  echo "WARNING: no GPU and no srun. Stages diffuse/mpnn/fold will fail." >&2
  echo "Only 'prepare' and '--selftest' can run here." >&2
fi

# --- Environment ------------------------------------------------------------
[[ -f "$ROOT/env.sh" ]] || { echo "Run 'bash setup_env.sh' first." >&2; exit 1; }
# shellcheck disable=SC1091
source "$ROOT/env.sh"
if command -v module &>/dev/null; then
  module load libs/nvidia-cuda/11.8.0 2>/dev/null || module load cuda/11.8 2>/dev/null || true
  module load apps/anaconda3/2024.06/bin 2>/dev/null || module load anaconda3 2>/dev/null || true
fi
# No `conda activate` -- we call each env's interpreter by absolute path, so a
# half-initialised conda cannot put pip and python in different environments.
for v in EGFR_RFDIFF_PY EGFR_FOLD_PY; do
  [[ -x "${!v:-}" ]] || { echo "ERROR: $v is not an executable interpreter (${!v:-unset}). Re-run setup_env.sh." >&2; exit 1; }
done

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
mkdir -p logs work results

FLAGS=()
[[ "$MODE" == "learn" ]] && FLAGS+=(--learn)
FLAGS+=("${PASSTHRU[@]+"${PASSTHRU[@]}"}")

echo "=============================================================="
echo " EGFR binder design -- interactive"
echo " node   : $(hostname)"
echo " mode   : $MODE"
echo " stage  : $STAGE"
echo " flags  : ${FLAGS[*]:-none}"
echo "=============================================================="
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader 2>/dev/null \
  || echo "(no GPU visible)"
echo

# Stage -> interpreter. Only 'fold' needs the ESMFold env.
py_for () {
  if [[ "$1" == "fold" ]]; then echo "$EGFR_FOLD_PY"; else echo "$EGFR_RFDIFF_PY"; fi
}

run_one () {
  local st="$1" py; py="$(py_for "$st")"
  echo
  echo ">>> $st  [$(basename "$(dirname "$(dirname "$py")")")]  $(date -u +%H:%M:%S)"
  # -u: unbuffered, so RFdiffusion's per-step progress appears as it happens.
  "$py" -u "$ROOT/pipeline.py" --stage "$st" "${FLAGS[@]+"${FLAGS[@]}"}"
}

START=$(date +%s)
if [[ "$STAGE" == "all" ]]; then
  for st in prepare diffuse mpnn fold score submit; do run_one "$st"; done
else
  run_one "$STAGE"
fi
ELAPSED=$(( $(date +%s) - START ))

echo
echo "=============================================================="
printf " done in %dh %dm %ds\n" $((ELAPSED/3600)) $((ELAPSED%3600/60)) $((ELAPSED%60))
echo "=============================================================="

if [[ -f "$ROOT/results/submission.csv" ]]; then
  echo
  echo "Worth looking at now:"
  echo "  results/design_report.csv   per-design metrics (S_pH, His count, pLDDT)"
  echo "  results/design_details.json per-histidine pKa estimates and distances"
  echo "  work/target_meta.json       the epitope patches it chose, and why"
  echo "  work/backbones/*.pdb        RFdiffusion output -- open in PyMOL/ChimeraX"
  echo "  work/esmfold/*.pdb          ESMFold models of the designed sequences"
  echo
  echo "To see a design against the target in PyMOL:"
  echo "  pymol work/target_domIII.pdb work/backbones/<name>.pdb"
fi

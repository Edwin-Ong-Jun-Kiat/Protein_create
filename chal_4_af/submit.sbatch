#!/usr/bin/env bash
# =============================================================================
# SLURM job: AfDesign binder hallucination on one Kelvin-2 V100.
#
#   sbatch submit.sbatch                        8 designs, 6 h cap
#   sbatch --export=ALL,PROBE=1 submit.sbatch   4 timed steps, 20 min cap
#   sbatch --export=ALL,FULL=1 submit.sbatch    24 designs, 24 h cap
#
# This is challenge 2's submit.sbatch with the stage loop replaced by a single
# python call. There is no stage machinery here on purpose: AfDesign is one
# optimisation loop, not a seven-stage pipeline, and each design is independent
# and written to the CSV as it completes, so a wall-clock kill loses at most
# one design rather than the run.
#
# Configured for Kelvin-2 (QUB): partition k2-gpu-v100, CUDA 11.8, anaconda3.
# =============================================================================

#SBATCH --job-name=tfr1-af
#SBATCH --output=logs/af_%j.out
#SBATCH --error=logs/af_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=k2-gpu-v100
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=eongjunkiat01@qub.ac.uk

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT="$SLURM_SUBMIT_DIR"
else
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "$ROOT"
mkdir -p logs work results

echo "=============================================================="
echo " TfR1 apical binder -- AlphaFold track (challenge 04)"
echo " job      : ${SLURM_JOB_ID:-interactive}"
echo " node     : $(hostname)"
echo " started  : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo " root     : $ROOT"
echo "=============================================================="

# --- Environment -------------------------------------------------------------
if [[ ! -f "$ROOT/env.sh" ]]; then
  echo "ERROR: env.sh not found. Run 'bash install_af.sh --build' on a LOGIN" >&2
  echo "node first -- compute nodes have no outbound network and cannot" >&2
  echo "download the wheels or the 5.3 GB of AF2 parameters." >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$ROOT/env.sh"

if command -v module &>/dev/null; then
  module load libs/nvidia-cuda/11.8.0 2>/dev/null || module load cuda/11.8 2>/dev/null || true
  module load apps/anaconda3/2024.06/bin 2>/dev/null || module load anaconda3 2>/dev/null || true
fi

# No `conda activate` -- the interpreter is called by absolute path, so a
# half-initialised conda cannot silently split pip and python across envs.
[[ -x "${AF_PY:-}" ]] || { echo "ERROR: AF_PY is not executable (${AF_PY:-unset})" >&2; exit 1; }

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo
echo "--- GPU ---"
# The driver version is the one number that decides whether the installed jax
# CUDA wheel can run at all, so it is printed before anything imports jax.
# cuda12 wheels need >= 525.60; cuda11 wheels need >= 450.
if command -v nvidia-smi &>/dev/null; then
  nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap \
             --format=csv,noheader 2>/dev/null | sed 's/^/  /' \
    || nvidia-smi | head -12
  DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
  echo "  installed jax wheels: CUDA ${AF_CUDA:-unknown}"
  if [[ -n "${DRV:-}" && "${AF_CUDA:-}" == "12" && "${DRV%%.*}" -lt 525 ]]; then
    echo "  ERROR: driver $DRV is too old for the cuda12 wheels." >&2
    echo "  Rebuild with: AF_CUDA=11 bash install_af.sh --build" >&2
    exit 1
  fi
else
  echo "  nvidia-smi unavailable -- that is itself a red flag on a gpu node"
fi
echo

# --- Tunables ---------------------------------------------------------------
#   PROBE=1   4 timed steps and exit. Run this FIRST, every time the env
#             changes. It is 20 minutes against a multi-hour design run.
#   FULL=1    24 designs instead of 8.
#   BINDER_LEN, N_DESIGNS, HIS_BIAS, RECYCLES override individually.
#
# Why probe-first is not optional: the MI300A's ESMFold job spent 3 h 20 min
# folding fewer than 10 of 36 sequences before anyone checked GPU utilisation
# (it was 0%). A cheap timing check at the top of a new backend is the whole
# lesson of this project so far.
ARGS=(--target "${AF_TARGET:-$ROOT/work/target_af_k382.pdb}"
      --data-dir "${AF_PARAMS:?AF_PARAMS unset -- re-run install_af.sh}"
      --binder-len "${BINDER_LEN:-55}"
      --recycles "${RECYCLES:-1}"
      --his-bias "${HIS_BIAS:-1.5}")

if [[ "${PROBE:-0}" == "1" ]]; then
  echo ">>> PROBE: 4 timed steps, no designs written"
  ARGS+=(--probe)
elif [[ "${FULL:-0}" == "1" ]]; then
  echo ">>> FULL: ${N_DESIGNS:-24} designs"
  ARGS+=(--n "${N_DESIGNS:-24}")
else
  echo ">>> standard: ${N_DESIGNS:-8} designs"
  ARGS+=(--n "${N_DESIGNS:-8}")
fi
[[ -n "${SEED:-}" ]] && ARGS+=(--seed "$SEED")
[[ -n "${HOTSPOTS:-}" ]] && ARGS+=(--hotspots "$HOTSPOTS")

echo "  $AF_PY af_design.py ${ARGS[*]}"
echo

"$AF_PY" -u "$ROOT/af_design.py" "${ARGS[@]}"

echo
echo "=============================================================="
echo " finished : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
if [[ "${PROBE:-0}" != "1" ]]; then
  echo " sequences  : $ROOT/results/af_designs.csv"
  echo " structures : $ROOT/results/af_pdb/"
  echo "=============================================================="
  column -s, -t "$ROOT/results/af_designs.csv" 2>/dev/null | cut -c1-200 | head -12 \
    || head -5 "$ROOT/results/af_designs.csv" || true
else
  echo "=============================================================="
fi

# =============================================================================
# WHAT THIS DOES NOT DO
#
# It does not compute S = log10(Kd(6.0)/Kd(7.4)). AfDesign optimises AF2
# confidence, which is pH-blind; the His bias only makes a switch POSSIBLE.
# Scoring is the main track's job, and feeding these sequences through it is
# the comparison this whole track exists for:
#
#   cp results/af_designs.csv ../chal_4/work/af_candidates.csv
#   cd ../chal_4 && source env.sh && "$PHB_PY" pipeline.py --stage score
#
# It also does not filter with AF2-initial-guess (dl_binder_design), which is
# the best-validated binder filter and pins jax 0.4.20 + CUDA -- i.e. exactly
# this environment. That was impossible on ROCm and is the obvious next step
# here. BindCraft's scoring additionally wants PyRosetta, which needs an
# academic licence; AfDesign itself does not.
# =============================================================================

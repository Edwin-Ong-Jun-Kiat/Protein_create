#!/usr/bin/env bash
# =============================================================================
# SLURM job: TNF-alpha pH-selective binder design (single V100, all stages)
#
#   sbatch submit.sbatch                        full run
#   sbatch --export=ALL,LEARN=1 submit.sbatch   ~1 h smoke test
#
# Stages switch conda env where required:
#   prepare, diffuse, mpnn   -> $TNF_RFDIFF_PY
#   fold                     -> $TNF_FOLD_PY
#   score, submit            -> $TNF_RFDIFF_PY
#
# Both interpreters come from env.sh, written by setup_env.sh, and both live on
# scratch. The envs are SHARED with challenge 1 -- nothing is rebuilt here.
#
# Every stage checkpoints to work/.done_<stage>, so if the job hits the wall
# clock you can just `sbatch submit.sbatch` again and it resumes. Stage diffuse
# additionally skips batches whose PDBs already exist.
#
# Configured for Kelvin-2 (QUB): partition k2-gpu-v100, CUDA 11.8, anaconda3.
# =============================================================================

#SBATCH --job-name=tnf-binder
#SBATCH --output=logs/tnf_%j.out
#SBATCH --error=logs/tnf_%j.err
#SBATCH --time=24:00:00
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
echo " TNF-alpha pH-selective binder design (challenge 02)"
echo " job      : ${SLURM_JOB_ID:-interactive}"
echo " node     : $(hostname)"
echo " started  : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo " root     : $ROOT"
echo "=============================================================="

# --- Environment -------------------------------------------------------------
if [[ ! -f "$ROOT/env.sh" ]]; then
  echo "ERROR: env.sh not found. Run 'bash setup_env.sh' on a login node first." >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$ROOT/env.sh"

if command -v module &>/dev/null; then
  module load libs/nvidia-cuda/11.8.0 2>/dev/null || module load cuda/11.8 2>/dev/null || true
  module load apps/anaconda3/2024.06/bin 2>/dev/null || module load anaconda3 2>/dev/null || true
fi

# No `conda activate` -- each env's interpreter is called by absolute path, so
# a half-initialised conda cannot silently split pip and python across envs.
for v in TNF_RFDIFF_PY TNF_FOLD_PY; do
  if [[ ! -x "${!v:-}" ]]; then
    echo "ERROR: $v is not an executable interpreter (${!v:-unset})." >&2
    echo "Re-run 'bash setup_env.sh' on a login node." >&2
    exit 1
  fi
done

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
# Reduces fragmentation-driven OOM. Matters more here than in challenge 1: the
# target is two protomers, so RFdiffusion is reasoning over ~140 target
# residues plus the binder rather than ~90.
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

echo
echo "--- GPU ---"
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap \
           --format=csv,noheader || echo "nvidia-smi unavailable"
echo

# --- Tunables ---------------------------------------------------------------
#   LEARN=1          small, fast, permissive (~1 h). Overrides the counts below.
#   SKIP_SELFTEST=1  go straight to the GPU without the CPU preflight.
#   SKIP_ESMFOLD=1   skip stage fold (no pLDDT/RMSD filter; much weaker)
#
# Track 1 = 40 designs. For Tracks 2/3 set N_SUBMIT=20.
if [[ "${LEARN:-0}" == "1" ]]; then
  COMMON=(--learn)
  echo "LEARN=1 -> learning mode (20 backbones, 1 groove, permissive filters)"
else
  N_BACKBONES="${N_BACKBONES:-150}"
  SEQS_PER_BB="${SEQS_PER_BB:-8}"
  N_SUBMIT="${N_SUBMIT:-40}"
  N_GROOVES="${N_GROOVES:-2}"
  COMMON=(--n-backbones "$N_BACKBONES"
          --seqs-per-backbone "$SEQS_PER_BB"
          --n-submit "$N_SUBMIT"
          --n-grooves "$N_GROOVES")
fi
if [[ "${SKIP_ESMFOLD:-0}" == "1" ]]; then
  COMMON+=(--skip-esmfold)
  echo "SKIP_ESMFOLD=1 -> no pLDDT or RMSD filter; designs are much less trustworthy"
fi

run_stage () {  # run_stage <interpreter> <stage>
  local py="$1" stage="$2"
  echo
  echo ">>> stage '$stage'  [$(basename "$(dirname "$(dirname "$py")")")]  $(date -u +%H:%M:%S)"
  "$py" -u "$ROOT/pipeline.py" --stage "$stage" "${COMMON[@]}"
  echo "<<< stage '$stage' done  $(date -u +%H:%M:%S)"
}

# --- Preflight: fail on CPU in 60 s rather than 6 h into diffusion ----------
# Challenge 1's job 10073878 died in stage prepare on a JSON encoding error
# after a clean selftest, because the selftest did not exercise the encoder.
# This one does, and it also traces the cropped-target hotspot renumbering back
# to the hotspots that were selected -- an off-by-one there would aim the whole
# diffusion stage at the wrong face of the protein and look perfectly healthy.
if [[ "${SKIP_SELFTEST:-0}" == "1" ]]; then
  echo ">>> preflight selftest SKIPPED (SKIP_SELFTEST=1)"
else
  echo ">>> preflight selftest"
  "$TNF_RFDIFF_PY" -u "$ROOT/pipeline.py" --selftest "${COMMON[@]}"
fi

# --- Pipeline ---------------------------------------------------------------
run_stage "$TNF_RFDIFF_PY" prepare
run_stage "$TNF_RFDIFF_PY" diffuse
run_stage "$TNF_RFDIFF_PY" mpnn
run_stage "$TNF_FOLD_PY"   fold
run_stage "$TNF_RFDIFF_PY" score
run_stage "$TNF_RFDIFF_PY" submit

echo
echo "=============================================================="
echo " finished : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo " submission : $ROOT/results/submission.csv"
echo " metrics    : $ROOT/results/design_report.csv"
echo "=============================================================="
head -5 "$ROOT/results/submission.csv" || true

# =============================================================================
# OPTIONAL: shard backbone generation across several GPUs.
#
# Diffusion is ~90% of the wall time and shards cleanly by batch. Save this as
# diffuse_array.sbatch and run it between 'prepare' and 'mpnn':
#
#   #SBATCH --array=0-3
#   #SBATCH --partition=k2-gpu-v100
#   #SBATCH --gres=gpu:v100:1
#   #SBATCH --time=08:00:00
#   source env.sh
#   "$TNF_RFDIFF_PY" pipeline.py --stage diffuse \
#       --shard "$SLURM_ARRAY_TASK_ID" --nshards "$SLURM_ARRAY_TASK_COUNT" \
#       --n-backbones 400
#
# Then resume the rest on one GPU (--force on diffuse would redo it, so don't):
#   "$TNF_RFDIFF_PY" pipeline.py --stage mpnn
#   "$TNF_FOLD_PY"   pipeline.py --stage fold
#   "$TNF_RFDIFF_PY" pipeline.py --stage score
#   "$TNF_RFDIFF_PY" pipeline.py --stage submit
# =============================================================================

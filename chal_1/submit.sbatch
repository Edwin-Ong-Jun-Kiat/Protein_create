#!/usr/bin/env bash
# =============================================================================
# SLURM job: EGFR pH-selective binder design (single V100, all stages)
#
#   sbatch submit.sbatch
#
# Runs stages in order, switching conda env where required:
#   prepare, diffuse, mpnn   -> $EGFR_RFDIFF_ENV
#   fold                     -> $EGFR_FOLD_ENV
#   score, submit            -> $EGFR_RFDIFF_ENV
# (both env paths are set by setup_env.sh in env.sh; they live on scratch)
#
# Stages checkpoint to work/.done_<stage>, so if the job hits the wall clock
# you can simply `sbatch submit.sbatch` again and it resumes.
#
# Configured for Kelvin-2 (QUB): partition k2-gpu-v100, CUDA 11.8, anaconda3.
# =============================================================================

#SBATCH --job-name=egfr-binder
#SBATCH --output=logs/egfr_%j.out
#SBATCH --error=logs/egfr_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=k2-gpu-v100
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=eongjunkiat01@qub.ac.uk

# If k2-gpu-v100 rejects a 24 h walltime, lower --time and just resubmit when
# it expires: every stage checkpoints to work/.done_<stage>, and stage diffuse
# skips backbone groups whose PDBs already exist, so a resubmit resumes rather
# than restarting. Learning mode needs about 1 h.

set -euo pipefail

# --- Resolve project root (works whether submitted from here or elsewhere) ---
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT="$SLURM_SUBMIT_DIR"
else
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "$ROOT"
mkdir -p logs work results

echo "=============================================================="
echo " EGFR pH-selective binder design"
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

# Kelvin-2 modules (names recorded by setup_env.sh in EGFR_MODULES)
if command -v module &>/dev/null; then
  module load libs/nvidia-cuda/11.8.0 2>/dev/null || module load cuda/11.8 2>/dev/null || true
  module load apps/anaconda3/2024.06/bin 2>/dev/null || module load anaconda3 2>/dev/null || true
fi

# No `conda activate` -- each env's interpreter is called by absolute path, so
# a half-initialised conda cannot silently split pip and python across envs.
for v in EGFR_RFDIFF_PY EGFR_FOLD_PY; do
  if [[ ! -x "${!v:-}" ]]; then
    echo "ERROR: $v is not an executable interpreter (${!v:-unset})." >&2
    echo "Re-run 'bash setup_env.sh' on a login node." >&2
    exit 1
  fi
done

# Keep thread counts sane: oversubscription is a common cause of slow
# RFdiffusion runs on shared HPC nodes.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
# Reduces fragmentation-driven OOM on 16 GB V100s.
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

echo
echo "--- GPU ---"
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap \
           --format=csv,noheader || echo "nvidia-smi unavailable"
echo

# --- Tunables ---------------------------------------------------------------
#   LEARN=1        small, fast, permissive run (~30-60 min). Overrides the
#                  three counts below; see pipeline.py --learn.
#   SKIP_SELFTEST=1  go straight to the GPU without the CPU preflight.
#
#   sbatch --export=ALL,LEARN=1 submit.sbatch
#
# Track 1 = 40 designs. For Tracks 2/3 set N_SUBMIT=20.
if [[ "${LEARN:-0}" == "1" ]]; then
  COMMON=(--learn)
  echo "LEARN=1 -> learning mode (small, fast, permissive)"
else
  N_BACKBONES="${N_BACKBONES:-150}"
  SEQS_PER_BB="${SEQS_PER_BB:-8}"
  N_SUBMIT="${N_SUBMIT:-40}"
  COMMON=(--n-backbones "$N_BACKBONES"
          --seqs-per-backbone "$SEQS_PER_BB"
          --n-submit "$N_SUBMIT")
fi

# Interpreters live on scratch, not in $HOME -- paths come from env.sh.
run_stage () {  # run_stage <interpreter> <stage>
  local py="$1" stage="$2"
  echo
  echo ">>> stage '$stage'  [$(basename "$(dirname "$(dirname "$py")")")]  $(date -u +%H:%M:%S)"
  "$py" -u "$ROOT/pipeline.py" --stage "$stage" "${COMMON[@]}"
  echo "<<< stage '$stage' done  $(date -u +%H:%M:%S)"
}

# --- Preflight: fail fast on CPU rather than 6 h into diffusion -------------
# Skip with SKIP_SELFTEST=1 if you have already validated the setup.
if [[ "${SKIP_SELFTEST:-0}" == "1" ]]; then
  echo ">>> preflight selftest SKIPPED (SKIP_SELFTEST=1)"
else
  echo ">>> preflight selftest"
  "$EGFR_RFDIFF_PY" -u "$ROOT/pipeline.py" --selftest "${COMMON[@]}"
fi

# --- Pipeline ---------------------------------------------------------------
run_stage "$EGFR_RFDIFF_PY" prepare
run_stage "$EGFR_RFDIFF_PY" diffuse
run_stage "$EGFR_RFDIFF_PY" mpnn
run_stage "$EGFR_FOLD_PY"   fold
run_stage "$EGFR_RFDIFF_PY" score
run_stage "$EGFR_RFDIFF_PY" submit

echo
echo "=============================================================="
echo " finished : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo " submission : $ROOT/results/submission.csv"
echo " metrics    : $ROOT/results/design_report.csv"
echo "=============================================================="
head -5 "$ROOT/results/submission.csv" || true

# =============================================================================
# OPTIONAL: parallelise backbone generation across several GPUs.
#
# Diffusion is ~90% of the wall time and shards cleanly. Save the block below
# as diffuse_array.sbatch and run it between 'prepare' and 'mpnn':
#
#   #SBATCH --array=0-3
#   #SBATCH --partition=k2-gpu-v100
#   #SBATCH --gres=gpu:v100:1
#   #SBATCH --time=08:00:00
#   source env.sh
#   "$EGFR_RFDIFF_PY" pipeline.py --stage diffuse \
#       --shard "$SLURM_ARRAY_TASK_ID" --nshards "$SLURM_ARRAY_TASK_COUNT" \
#       --n-backbones 400
#
# Then resume the rest on one GPU:
#   "$EGFR_RFDIFF_PY" pipeline.py --stage mpnn
#   "$EGFR_FOLD_PY"   pipeline.py --stage fold
# =============================================================================

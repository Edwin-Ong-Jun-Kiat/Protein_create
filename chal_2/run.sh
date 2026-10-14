#!/usr/bin/env bash
# =============================================================================
# One command to go from a freshly uploaded directory to a queued job.
#
#   bash run.sh                 setup -> selftest -> submit a ~1 h test run
#   bash run.sh --full          setup -> selftest -> submit the full run
#   bash run.sh --interactive   setup -> selftest -> grab a GPU and run live
#   bash run.sh --check         setup -> selftest only, submit nothing
#
# Run this on a Kelvin-2 LOGIN node. It does, in order:
#
#   1. bash setup_env.sh      finds the shared conda envs and tools, or builds
#                             them if they are missing, and writes env.sh
#   2. source env.sh
#   3. pipeline.py --selftest ~60 s on CPU. Checks the physics, the JSON
#                             encoder, the sequence gates, and that the cropped
#                             target's hotspot renumbering still maps back to
#                             the residues that were selected
#   4. sbatch submit.sbatch   queues the job, prints the job id and how to watch
#
# It stops at the first failure. A failed selftest never reaches sbatch, which
# is the whole point: challenge 1's job 10073878 burned a GPU allocation to
# discover a JSON encoding bug that a CPU preflight catches in under a minute.
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

c_i=$'\033[1;36m'; c_o=$'\033[1;32m'; c_e=$'\033[1;31m'; c_x=$'\033[0m'
step() { echo; echo "${c_i}=== $* ${c_x}"; }
ok()   { echo "${c_o}[  ok]${c_x} $*"; }
die()  { echo "${c_e}[ err]${c_x} $*" >&2; exit 1; }

MODE="test"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --full)        MODE="full" ;;
    --interactive) MODE="interactive" ;;
    --check)       MODE="check" ;;
    -h|--help)     sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
  shift
done

echo "=============================================================="
echo " TNF-alpha pH-selective binder design -- challenge 02"
echo " mode : $MODE"
echo " host : $(hostname)"
echo " root : $ROOT"
echo "=============================================================="

# --- 1. environment ---------------------------------------------------------
step "1/4  environment"
if [[ -f env.sh ]]; then
  ok "env.sh already present; re-verifying"
fi
bash setup_env.sh || die "setup_env.sh failed -- read its output above"

[[ -f env.sh ]] || die "setup_env.sh did not write env.sh"
# shellcheck disable=SC1091
source env.sh
[[ -x "${TNF_RFDIFF_PY:-}" ]] || die "TNF_RFDIFF_PY is not executable: ${TNF_RFDIFF_PY:-unset}"
[[ -x "${TNF_FOLD_PY:-}" ]]   || die "TNF_FOLD_PY is not executable: ${TNF_FOLD_PY:-unset}"
ok "rfdiff: $TNF_RFDIFF_PY"
ok "fold  : $TNF_FOLD_PY"

# --- 2. preflight -----------------------------------------------------------
step "2/4  preflight selftest (CPU, ~60 s)"
"$TNF_RFDIFF_PY" -u pipeline.py --selftest --force \
  || die "selftest FAILED -- fix the problems listed above before submitting.
Nothing has been queued."
ok "selftest passed"

if [[ "$MODE" == "check" ]]; then
  step "3/4  --check: stopping here, nothing submitted"
  echo "  to queue the test run:  bash run.sh"
  echo "  to queue the full run:  bash run.sh --full"
  exit 0
fi

# --- 3. interactive path ----------------------------------------------------
if [[ "$MODE" == "interactive" ]]; then
  step "3/4  interactive GPU session"
  echo "  requesting a V100; output streams live. Ctrl-C to abandon."
  exec bash run_interactive.sh
fi

# --- 4. submit --------------------------------------------------------------
step "3/4  submitting to SLURM"
if [[ "$MODE" == "full" ]]; then
  echo "  full run: 150 backbones, 2 grooves, 24 h wall clock"
  echo "  diffusion is ~90% of this; expect 12-16 h"
  JOB=$(sbatch --parsable submit.sbatch)
else
  echo "  TEST run (LEARN=1): 20 backbones, 1 groove, permissive filters"
  echo "  exercises every stage and both conda envs in about an hour"
  JOB=$(sbatch --parsable --export=ALL,LEARN=1 submit.sbatch)
fi
ok "queued as job $JOB"

step "4/4  watching it"
cat <<EOF
  squeue -u \$USER -j $JOB              # is it running yet
  tail -f logs/tnf_${JOB}.out           # live output
  scancel $JOB                          # give up

  When it finishes:
    column -s, -t results/submission.csv   | head -20
    column -s, -t results/design_report.csv | head -5
    less results/METHODS.md                 # methodology, auto-generated

  Results back to your Mac (run this LOCALLY, not here):
    bash push.sh --pull

  If it dies on the wall clock, just resubmit -- every stage checkpoints to
  work/.done_<stage> and stage diffuse skips batches it already finished:
    sbatch $([[ "$MODE" == full ]] && echo "submit.sbatch" || echo "--export=ALL,LEARN=1 submit.sbatch")
EOF

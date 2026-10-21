#!/usr/bin/env bash
# =============================================================================
# One command from a freshly uploaded directory to a queued job.
#
#   bash run.sh --probe    verify the env -> submit a 20 min GPU timing probe
#   bash run.sh            verify the env -> submit 8 designs (6 h cap)
#   bash run.sh --full     verify the env -> submit 24 designs (24 h cap)
#   bash run.sh --check    verify the env only, submit nothing
#   bash run.sh --build    build the store first (LOGIN node, needs network)
#
# Run this on a Kelvin-2 LOGIN node. It stops at the first failure, and a
# failed check never reaches sbatch -- which is the whole point.
#
# RUN --probe FIRST. It costs 20 minutes and answers the only question that
# matters after the MI300A: is AF2 on the GPU, and what is one step?
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

c_i=$'\033[1;36m'; c_o=$'\033[1;32m'; c_e=$'\033[1;31m'; c_x=$'\033[0m'
step() { echo; echo "${c_i}=== $* ${c_x}"; }
ok()   { echo "${c_o}[  ok]${c_x} $*"; }
die()  { echo "${c_e}[ err]${c_x} $*" >&2; exit 1; }

MODE="standard"; BUILD=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --probe)   MODE="probe" ;;
    --full)    MODE="full" ;;
    --check)   MODE="check" ;;
    --build)   BUILD="--build" ;;
    -h|--help) sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
  shift
done

echo "=============================================================="
echo " TfR1 apical binder -- AlphaFold track (challenge 04)"
echo " mode : $MODE   host : $(hostname)"
echo "=============================================================="

step "1/4  environment"
bash install_af.sh $BUILD || die "install_af.sh failed -- read its output above"
[[ -f env.sh ]] || die "install_af.sh did not write env.sh"
# shellcheck disable=SC1091
source env.sh
[[ -x "${AF_PY:-}" ]] || die "AF_PY is not executable: ${AF_PY:-unset}"
ok "interpreter: $AF_PY"
ok "params: $AF_PARAMS/params"
ok "target: $AF_TARGET"

step "2/4  CPU preflight (~60 s)"
# Everything here is checkable without a GPU: do the imports resolve, does the
# target parse, are the hotspots actually in it? Challenge 1's job 10073878
# burned a GPU allocation to discover a JSON encoding bug that a CPU preflight
# catches in under a minute.
"$AF_PY" - <<'PY' || die "preflight FAILED -- nothing has been queued"
import os, sys
from pathlib import Path

import jax
print(f"  jax {jax.__version__}")
assert jax.__version__ == "0.4.20", f"jax drifted to {jax.__version__}"

from colabdesign import mk_afdesign_model          # noqa: F401
from colabdesign.af.alphafold.common import residue_constants
print(f"  colabdesign imports, His index {residue_constants.restypes.index('H')}")

target = Path(os.environ["AF_TARGET"])
assert target.exists(), f"missing target {target}"
nums, chains = [], set()
for ln in target.read_text().splitlines():
    if ln.startswith("ATOM") and ln[12:16].strip() == "CA":
        chains.add(ln[21]); nums.append(int(ln[22:26]))
print(f"  target {target.name}: {len(nums)} residues, chain(s) {sorted(chains)}, "
      f"{min(nums)}-{max(nums)}")
assert len(chains) == 1, f"AfDesign wants ONE target chain, found {sorted(chains)}"

# The hotspot list and the construct must agree. Disagreeing is not a subtle
# failure -- it aborts with "ERROR: positions 382 and chain A not found" -- but
# it aborts AFTER the queue wait, which is the expensive part. This cost me a
# job once already, keeping A382 while using a 150-280 construct.
hot = os.environ.get("HOTSPOTS", "A382,A193,A189,A381,A158,A380")
present, missing = [], []
for h in hot.split(","):
    h = h.strip()
    (present if (h[0] in chains and int(h[1:]) in set(nums)) else missing).append(h)
print(f"  hotspots present: {present}")
if missing:
    print(f"\n  ERROR: hotspots {missing} are NOT in this construct.")
    print(f"  Either widen the construct or submit with")
    print(f"      HOTSPOTS='{','.join(present)}' bash run.sh")
    sys.exit(1)

params = Path(os.environ["AF_PARAMS"]) / "params"
n = len(list(params.glob("params_*.npz")))
assert n >= 5, f"only {n} AF2 parameter files in {params}; expected 15"
print(f"  AF2 parameters: {n} files")
PY
ok "preflight passed"

if [[ "$MODE" == "check" ]]; then
  step "3/4  --check: stopping here, nothing submitted"
  echo "  queue the timing probe:  bash run.sh --probe"
  echo "  queue 8 designs:         bash run.sh"
  exit 0
fi

step "3/4  submitting to SLURM"
case "$MODE" in
  probe)
    echo "  PROBE: 4 timed steps, 20 min cap, nothing written"
    JOB=$(sbatch --parsable --time=00:20:00 --export=ALL,PROBE=1 submit.sbatch)
    ;;
  full)
    echo "  FULL: 24 designs, 160 steps each, 24 h cap"
    JOB=$(sbatch --parsable --time=24:00:00 --export=ALL,FULL=1 submit.sbatch)
    ;;
  *)
    echo "  standard: 8 designs, 160 steps each, 6 h cap"
    JOB=$(sbatch --parsable submit.sbatch)
    ;;
esac
ok "queued as job $JOB"

step "4/4  watching it"
cat <<EOF
  squeue -u \$USER -j $JOB           # is it running yet
  tail -f logs/af_${JOB}.out         # live output
  scancel $JOB                       # give up

  The first ten lines of the log tell you whether it is on the GPU. If they
  say every device is 'cpu', cancel immediately -- it would otherwise spend
  ~227 s per step, or roughly 10 h for a single design.

EOF
if [[ "$MODE" == "probe" ]]; then
cat <<EOF
  When the probe finishes, read the two numbers it separates:
    step 1        XLA compiling the AF2 graph (~250 s, amortised by the
                  JAX_COMPILATION_CACHE_DIR set in env.sh)
    steady state  the real per-step cost. On a V100 expect low single-digit
                  seconds. Anything over 60 s means it is not really on the
                  GPU, whatever nvidia-smi printed.

  Then: bash run.sh          # 8 designs
EOF
else
cat <<EOF
  When it finishes:
    column -s, -t results/af_designs.csv | cut -c1-200
    cat results/af_summary.json

  Results back to your Mac (run LOCALLY, not here):
    bash push.sh --pull

  Designs are appended to the CSV as each one completes, so a wall-clock kill
  costs you the design in flight and nothing else. Resubmitting starts fresh
  rather than resuming -- pass SEED= to continue a different part of the seed
  range instead of repeating work:
    SEED=8 N_DESIGNS=8 sbatch --export=ALL submit.sbatch
EOF
fi

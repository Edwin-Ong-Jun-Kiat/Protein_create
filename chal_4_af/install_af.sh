#!/usr/bin/env bash
# =============================================================================
# AlphaFold2 / ColabDesign installer for Kelvin-2 (QUB), NVIDIA V100.
#
#   bash install_af.sh           verify an existing store, (re)write env.sh
#   bash install_af.sh --build   build it -- login node, needs network
#
# RUN --build ON A LOGIN NODE. Kelvin-2 compute nodes have no outbound network,
# so every wheel, checkout and the 5.3 GB of AF2 parameters must be staged here
# first. Takes 40-70 minutes and about 12 GB of scratch.
#
# WHY THIS EXISTS
# ---------------
# The same stack on the MI300A is dead, not slow. AF2's graph needs an SVD; on
# GPU jax lowers that to the vendor solver, and jax-rocm7-plugin 0.6.0 does not
# register a handler for it:
#
#   XlaRuntimeError: UNIMPLEMENTED: No registered implementation for custom
#   call to hipsolver_gesvd_ffi for platform ROCM
#
# No flag fixes a missing kernel. CPU works there at 227 s/optimisation step,
# which is ~9.5 h for one 150-step design. cuSOLVER implements gesvd, so on a
# V100 the blocker simply is not there.
#
# WHY THESE VERSIONS
# ------------------
#   python  3.10        ColabDesign's own floor; 3.9 trips on newer jax typing
#   jax     0.4.20      dl_binder_design's pinned version, and inside
#                       BindCraft's jax>=0.4,<=0.6.0 range. Known-good on
#                       sm_70. Newer jax has raised its compute-capability
#                       floor more than once, and V100 is the oldest card this
#                       stack still targets -- not the place to find out.
#   numpy   <2          this whole stack predates the NumPy 2 ABI break
#   CUDA    probed      see "THE CUDA 11 vs 12 DECISION" below
#
# EVERY pip CALL GOES THROUGH A CONSTRAINTS FILE. This is not neatness. On the
# MI300A I installed chex/flax/optax in a second, unpinned pip call; the
# resolver quietly pulled jax 0.4 -> 0.12 and numpy 1.x -> 2.5, and the only
# symptom was, three steps later,
#   AttributeError: module 'jax_rocm7_plugin._triton' has no attribute 'registrations'
# A constraints file applies to transitive dependencies too, which a pin on the
# command line does not.
#
# NO conda activate ANYWHERE. In a non-interactive shell its shell-function
# integration is unreliable, and when it fails it can return 0 WITHOUT changing
# PATH -- so pip installs into one interpreter while python is another, and the
# only symptom is a ModuleNotFoundError much later. Every install below names
# the target interpreter by absolute path.
#
# NO PyRosetta. It needs an academic licence and only BindCraft's *scoring*
# uses it; AfDesign itself does not. See README.md.
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
mkdir -p "$ROOT/work" "$ROOT/results" "$ROOT/logs"

c_i=$'\033[1;36m'; c_o=$'\033[1;32m'; c_w=$'\033[1;33m'; c_e=$'\033[1;31m'; c_x=$'\033[0m'
log()  { printf '\n%s==> %s%s\n' "$c_i" "$*" "$c_x"; }
ok()   { echo "${c_o}[  ok]${c_x} $*"; }
warn() { echo "${c_w}[warn]${c_x} $*"; }
die()  { echo "${c_e}[ err]${c_x} $*" >&2; exit 1; }

BUILD=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --build)   BUILD=1 ;;
    -h|--help) sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
  shift
done

echo "=============================================================="
echo " Challenge 04 (TfR1) -- AlphaFold track, Kelvin-2 / V100"
echo " host: $(hostname)   user: ${USER:-unknown}"
echo "=============================================================="

# ---------------------------------------------------------------------------
# 0. Refuse to --build without a network, which means a compute node
# ---------------------------------------------------------------------------
# Discovering this after a queue wait is worse than refusing now.
if [[ $BUILD -eq 1 ]]; then
  curl -sS -m 20 -o /dev/null https://pypi.org/simple/ 2>/dev/null \
    || die "no outbound network -- this looks like a compute node.
Run 'bash install_af.sh --build' on a Kelvin-2 LOGIN node."
fi

# ---------------------------------------------------------------------------
# 1. Cluster modules
# ---------------------------------------------------------------------------
log "Loading cluster modules"
if command -v module &>/dev/null; then
  module load libs/nvidia-cuda/11.8.0 2>/dev/null \
    || module load cuda/11.8 2>/dev/null \
    || warn "no CUDA 11.8 module; jax's pip wheels bundle their own runtime"
  module load apps/anaconda3/2024.06/bin 2>/dev/null \
    || module load apps/anaconda3 2>/dev/null \
    || module load anaconda3 2>/dev/null \
    || warn "could not load an anaconda3 module"
  module list 2>&1 | sed 's/^/  /' || true
else
  warn "no 'module' command; assuming conda is already on PATH"
fi

# ---------------------------------------------------------------------------
# 2. Storage: ~12 GB, which will exceed a Kelvin-2 home quota
# ---------------------------------------------------------------------------
log "Choosing install location"
if [[ -z "${AF_STORE:-}" ]]; then
  for cand in "/mnt/scratch2/users/$USER" "/mnt/scratch/users/$USER" \
              "/mnt/scratch2/$USER" "$HOME/sharedscratch" "$HOME/scratch" \
              "/scratch/$USER"; do
    if [[ -d "$cand" && -w "$cand" ]]; then AF_STORE="$cand/tfr1_af_store"; break; fi
  done
fi
: "${AF_STORE:=$ROOT/store}"
mkdir -p "$AF_STORE" || die "cannot create $AF_STORE"
echo "  AF_STORE = $AF_STORE"
if [[ "$AF_STORE" == "$ROOT/store" ]]; then
  warn "no scratch directory found, so 12 GB goes inside the project dir."
  warn "Re-run with: AF_STORE=/mnt/scratch2/users/\$USER/af bash install_af.sh --build"
fi

AVAIL_G=$(( $(df -Pk "$AF_STORE" | awk 'NR==2{print $4}') / 1024 / 1024 ))
echo "  free space here: ${AVAIL_G} GB"
[[ "$AVAIL_G" -lt 15 ]] && warn "under 15 GB free -- the install may fail partway"

ENVS="$AF_STORE/envs"
AF_ENV="$ENVS/af"
PARAMS="$AF_STORE/af_params"
PIPCACHE="$AF_STORE/.pip_cache"
export CONDA_PKGS_DIRS="$AF_STORE/.conda_pkgs"
mkdir -p "$ENVS" "$PARAMS" "$PIPCACHE" "$CONDA_PKGS_DIRS" \
         "$AF_STORE/xla_cache" "$AF_STORE/logs"

# Neutralise anything that could redirect imports or installs. On a shared
# cluster a loaded module or a ~/.config/pip/pip.conf can set a different
# install prefix, and pip then reports "Successfully installed" while the files
# land where the interpreter never looks. Every pip call also passes
# --isolated, which makes pip ignore config files AND PIP_* variables.
unset PYTHONPATH PYTHONHOME PYTHONSTARTUP PYTHONUSERBASE
unset PIP_TARGET PIP_PREFIX PIP_USER PIP_ROOT PIP_REQUIRE_VIRTUALENV
export PYTHONNOUSERSITE=1

# ---------------------------------------------------------------------------
# 3. THE CUDA 11 vs 12 DECISION
# ---------------------------------------------------------------------------
# jax's pip wheels bundle their own CUDA, so the cluster's CUDA module does not
# decide this -- the NVIDIA *driver* does. cuda12 wheels need driver >= 525.60;
# cuda11 wheels need >= 450. Kelvin-2's V100 nodes are old enough that this is
# a real question, and getting it wrong produces, at import time,
#   "The NVIDIA driver's CUDA version is insufficient for the CUDA runtime"
#
# A login node usually has no GPU, so nvidia-smi may not be here. Order of
# preference: an explicit override, nvidia-smi, the kernel module's own
# version file, then a documented default.
log "Deciding CUDA wheel flavour"
DRIVER=""
if [[ -n "${AF_CUDA:-}" ]]; then
  echo "  AF_CUDA=$AF_CUDA given; skipping the driver probe"
else
  if command -v nvidia-smi &>/dev/null; then
    DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null \
             | head -1 | tr -d ' ') || true
  fi
  if [[ -z "$DRIVER" && -r /proc/driver/nvidia/version ]]; then
    DRIVER=$(grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' /proc/driver/nvidia/version | head -1) || true
  fi
  if [[ -n "$DRIVER" ]]; then
    echo "  NVIDIA driver on this host: $DRIVER"
    if [[ "${DRIVER%%.*}" -ge 525 ]]; then AF_CUDA=12; else AF_CUDA=11; fi
  else
    # Login nodes commonly have no GPU and no nvidia kernel module. Default to
    # 11 rather than 12: cuda11 wheels run on any driver a cuda12 wheel would
    # accept, so this choice cannot be the thing that breaks, only the slower
    # of two working options. submit.sbatch re-checks on the node and says so.
    AF_CUDA=11
    warn "no GPU or driver info on this login node -- defaulting to cuda11,"
    warn "which works on every driver cuda12 would. 'bash run.sh --probe'"
    warn "prints the node's real driver; if it is >= 525 you may rebuild with"
    warn "    AF_CUDA=12 bash install_af.sh --build"
  fi
fi
case "$AF_CUDA" in
  11) JAX_EXTRA="jax[cuda11_pip]==0.4.20" ;;
  12) JAX_EXTRA="jax[cuda12_pip]==0.4.20" ;;
  *)  die "AF_CUDA must be 11 or 12, got '$AF_CUDA'" ;;
esac
ok "building against CUDA $AF_CUDA  ($JAX_EXTRA)"

# ---------------------------------------------------------------------------
# 4. Conda
# ---------------------------------------------------------------------------
log "Locating conda"
if ! command -v conda &>/dev/null; then
  for c in "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda; do
    [[ -x "$c/bin/conda" ]] && export PATH="$c/bin:$PATH" && break
  done
fi
command -v conda &>/dev/null \
  || die "conda not found. Try: module load apps/anaconda3/2024.06/bin"
echo "  conda      $(conda --version)"
echo "  conda base $(conda info --base)"

APY="$AF_ENV/bin/python"

# ---------------------------------------------------------------------------
# 5. The constraints file -- read the note at the top of this script
# ---------------------------------------------------------------------------
CONSTRAINTS="$AF_STORE/af_constraints.txt"
cat > "$CONSTRAINTS" <<'EOF'
# Applied to EVERY pip call in install_af.sh, transitive deps included.
# Written by install_af.sh; editing it by hand is the supported way to move a
# pin, because nothing here may drift as a side effect of installing something
# else.
jax==0.4.20
jaxlib==0.4.20
numpy<2.0.0
scipy<1.12
# The haiku/optax/chex/flax set is what an unpinned install silently upgrades
# jax through. These four are the last releases that target jax 0.4.20.
dm-haiku==0.0.11
optax==0.1.7
chex==0.1.85
flax==0.7.5
ml-collections==0.1.1
immutabledict==4.1.0
biopython==1.81
pandas<2.1
EOF
echo "  constraints -> $CONSTRAINTS"
sed 's/^/    /' "$CONSTRAINTS"

if [[ $BUILD -eq 1 ]]; then
  # -------------------------------------------------------------------------
  # 6. Create the env
  # -------------------------------------------------------------------------
  log "Creating env: $AF_ENV (python 3.10)"
  # pip/setuptools/wheel come from conda, NOT from a later pip --upgrade.
  # Upgrading setuptools with pip is what left distutils-precedence.pth
  # pointing at a missing _distutils_hack on challenge 1's first attempt.
  if [[ -x "$APY" ]]; then echo "  exists; reusing"
  else conda create -y --prefix "$AF_ENV" python=3.10 pip setuptools wheel; fi
  [[ -x "$APY" ]] || die "env creation failed: no interpreter at $APY
     Try: rm -rf '$AF_ENV' and re-run."
  echo "  interpreter: $APY"
  echo "  version    : $("$APY" --version 2>&1)"
  case "$("$APY" --version 2>&1)" in
    *" 3.10."*) ;;
    *) die "expected python 3.10 at $APY, got $("$APY" --version 2>&1)" ;;
  esac

  pip_af() {
    "$APY" -m pip install --isolated --no-input --progress-bar off \
           --cache-dir "$PIPCACHE" --constraint "$CONSTRAINTS" "$@"
  }

  # -------------------------------------------------------------------------
  # 7. jax
  # -------------------------------------------------------------------------
  log "jax 0.4.20 + CUDA $AF_CUDA wheels"
  pip_af "$JAX_EXTRA" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
    || die "jax install failed.
     The CUDA $AF_CUDA wheel for jax 0.4.20 may have been withdrawn. Open
       https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
     and check which jaxlib 0.4.20 cuda build is listed for cp310, then edit
     both this line and $CONSTRAINTS."

  log "AlphaFold / ColabDesign python deps"
  pip_af "numpy<2" "scipy<1.12" "pandas<2.1" "biopython==1.81" \
         "dm-haiku==0.0.11" "optax==0.1.7" "chex==0.1.85" "flax==0.7.5" \
         "ml-collections==0.1.1" "immutabledict==4.1.0" \
         dm-tree joblib matplotlib tqdm py3Dmol

  # --no-deps: ColabDesign's setup.py lists a bare `jax`, which would upgrade
  # straight past the pin. The constraints file would catch it, but refusing
  # to consider it at all is clearer. Every dependency it needs is installed
  # by hand above -- including matplotlib, which is NOT obvious: with --no-deps
  # it goes missing and only surfaces as an ImportError deep inside
  # colabdesign.shared.plot on the first design call.
  log "ColabDesign (--no-deps)"
  pip_af "git+https://github.com/sokrypton/ColabDesign.git" --no-deps

  # -------------------------------------------------------------------------
  # 8. AF2 parameters (~5.3 GB) -- 5 ptm + 5 multimer + 5 original
  # -------------------------------------------------------------------------
  # ColabDesign wants a data_dir that CONTAINS a params/ subdirectory, and
  # loads e.g. params/params_model_1_ptm.npz from it. Extracting the tarball
  # straight into $PARAMS/params gives exactly that.
  log "AlphaFold2 parameters"
  PDIR="$PARAMS/params"
  mkdir -p "$PDIR"
  TAR="$PARAMS/alphafold_params_2022-12-06.tar"
  URL="https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar"
  if [[ -s "$PDIR/params_model_1_ptm.npz" && -s "$PDIR/params_model_5_ptm.npz" ]]; then
    echo "  already extracted ($(du -sh "$PDIR" | cut -f1))"
  else
    # A leftover .part from a FAILED attempt may hold an HTML error page.
    # Resuming on top of that silently corrupts the archive, so discard
    # anything too small to be a real partial download.
    if [[ -f "$TAR.part" ]]; then
      psz=$(stat -c%s "$TAR.part" 2>/dev/null || stat -f%z "$TAR.part" 2>/dev/null || echo 0)
      if [[ "$psz" -lt 1048576 ]]; then
        echo "  discarding stale partial (${psz} bytes)"; rm -f "$TAR.part"
      else
        echo "  resuming from $((psz / 1048576)) MB"
      fi
    fi
    if [[ ! -s "$TAR" ]]; then
      echo "  downloading 5.3 GB -> $TAR"
      curl -fL -C - --retry 5 --retry-delay 5 --progress-bar \
           -o "$TAR.part" "$URL" || { rm -f "$TAR.part"; die "download failed: $URL"; }
      mv "$TAR.part" "$TAR"
    fi
    # Validate before extracting: curl will happily have saved a 404 page.
    tar tf "$TAR" >/dev/null 2>&1 || { rm -f "$TAR"; die "corrupt download; re-run"; }
    sz=$(du -m "$TAR" | cut -f1)
    [[ "$sz" -gt 4000 ]] || { rm -f "$TAR"; die "archive is only ${sz} MB, expected ~5300"; }
    echo "  extracting (${sz} MB)"
    tar -xf "$TAR" -C "$PDIR"
    rm -f "$TAR" || warn "could not remove $TAR"
  fi
  [[ -s "$PDIR/params_model_1_ptm.npz" ]] \
    || die "params_model_1_ptm.npz missing after extraction"
  ok "parameters in $PDIR ($(ls "$PDIR" | wc -l | tr -d ' ') files)"

  # -------------------------------------------------------------------------
  # 9. pip check -- the last line of defence against a silent upgrade
  # -------------------------------------------------------------------------
  log "pip check"
  "$APY" -m pip check 2>&1 | sed 's/^/  /' \
    || warn "pip check reported conflicts above; read them before trusting a run"
fi

# ---------------------------------------------------------------------------
# 10. Verify -- this runs with or without --build
# ---------------------------------------------------------------------------
[[ -x "$APY" ]] || die "no interpreter at $APY. Run with --build on a login node."

log "Verifying the env"
"$APY" - <<'PY' || die "verification failed; re-run with --build"
import sys
print(f"  python  {sys.version.split()[0]}")

import numpy, scipy
print(f"  numpy   {numpy.__version__}")
assert numpy.__version__.startswith("1."), \
    f"numpy {numpy.__version__} -- this stack predates the NumPy 2 ABI break"

import jax, jaxlib
print(f"  jax     {jax.__version__}")
print(f"  jaxlib  {jaxlib.__version__}")
assert jax.__version__ == "0.4.20", (
    f"jax is {jax.__version__}, not 0.4.20. Something upgraded it -- this is "
    "the exact failure the constraints file exists to prevent.")

import haiku, optax, chex
print(f"  haiku   {haiku.__version__}")
print(f"  optax   {optax.__version__}")

import colabdesign
from colabdesign import mk_afdesign_model          # noqa: F401
from colabdesign.af.alphafold.common import residue_constants
h = residue_constants.restypes.index("H")
print(f"  colabdesign OK (histidine is index {h} of its 20-letter alphabet)")

import matplotlib                                  # --no-deps casualty
print(f"  matplotlib {matplotlib.__version__}")

# jax.devices() on a login node with no GPU returns CPU and prints a warning
# rather than raising, so this proves nothing about the compute node. What it
# does prove is that the CUDA plugin loaded at all.
devs = jax.devices()
print(f"  devices {[d.device_kind for d in devs]}")
if all(d.platform == "cpu" for d in devs):
    print("  no GPU here (normal on a login node);")
    print("  'bash run.sh --probe' verifies sm_70 and real GPU timings on a node")
PY
ok "env verified"

# ---------------------------------------------------------------------------
# 11. The target construct
# ---------------------------------------------------------------------------
# AfDesign needs ONE sequence-contiguous chain. The main pipeline's cropped
# targets are not that, and worse, they LIE about it: target_cropped_k382.pdb
# is renumbered 1..N, so it reports a single contiguous segment while actually
# containing 21 chain breaks with a maximum CA-CA distance of 36.2 A. AF2 would
# model that as one polypeptide and hallucinate loops to close the gaps.
log "Building the contiguous target construct"
if [[ -s "$ROOT/work/target_af_k382.pdb" ]]; then
  echo "  already present: work/target_af_k382.pdb"
else
  "$APY" -u "$ROOT/build_af_target.py" --out "$ROOT/work/target_af_k382.pdb" \
    || die "build_af_target.py failed -- read its output above"
fi
ok "target: work/target_af_k382.pdb"

# ---------------------------------------------------------------------------
# 12. Record everything the job scripts need
# ---------------------------------------------------------------------------
cat > "$ROOT/env.sh" <<EOF
# Written by install_af.sh on $(date -u +%Y-%m-%dT%H:%M:%SZ) from $(hostname).
# Absolute paths on Kelvin-2 scratch; meaningless anywhere else, which is why
# push.sh refuses to upload this file.
export AF_ROOT="$ROOT"
export AF_STORE="$AF_STORE"
export AF_ENV="$AF_ENV"
# Call this interpreter directly. Do NOT rely on 'conda activate' in a script:
# it can return 0 without changing PATH, putting pip and python in different
# environments.
export AF_PY="$APY"
export AF_PARAMS="$PARAMS"
export AF_CUDA="$AF_CUDA"
export AF_CONSTRAINTS="$CONSTRAINTS"
export AF_TARGET="$ROOT/work/target_af_k382.pdb"

# Caches off the home filesystem
export PIP_CACHE_DIR="$PIPCACHE"
export CONDA_PKGS_DIRS="$AF_STORE/.conda_pkgs"
# XLA recompiles the whole AF2 graph on every process start otherwise, which is
# most of the ~250 s first step. Persisting it across jobs is the single
# cheapest win available here.
export JAX_COMPILATION_CACHE_DIR="$AF_STORE/xla_cache"

# Keep imports unambiguous at run time as well as at install time. A module
# that exports PYTHONPATH, or a stray ~/.local/lib/python3.10, can shadow the
# pinned packages with incompatible versions.
unset PYTHONPATH PYTHONHOME PYTHONSTARTUP PYTHONUSERBASE
export PYTHONNOUSERSITE=1

# V100 is sm_70: no bf16, no TF32. AF2 runs fp32 anyway; saying so explicitly
# stops jax from quietly choosing otherwise on a future version.
export NVIDIA_TF32_OVERRIDE=0
# jax grabs 75% of VRAM up front by default. One AF2 model on a 241+55 residue
# complex does not need 24 GB, and preallocating it makes the node unshareable.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.85

export AF_MODULES="libs/nvidia-cuda/11.8.0 apps/anaconda3/2024.06/bin"
EOF
ok "wrote $ROOT/env.sh"

log "Done"
cat <<EOF

  project : $ROOT
  store   : $AF_STORE  ($(du -sh "$AF_STORE" 2>/dev/null | cut -f1) used)
  env     : $AF_ENV
  params  : $PARAMS/params
  target  : $ROOT/work/target_af_k382.pdb

Next -- a 15 minute probe BEFORE spending a design allocation. It answers the
only question that matters after the MI300A: is AF2 running on the GPU, and
how fast is one optimisation step?

  bash run.sh --probe

Then the real run:

  bash run.sh                # 8 designs, ~2 h wall clock
  bash run.sh --full         # 24 designs, 24 h cap

EOF

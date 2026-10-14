#!/usr/bin/env bash
# =============================================================================
# FULL INSTALLER -- builds the shared conda envs and third-party tools.
#
#   bash install_env.sh
#
# You almost certainly do NOT need to run this directly. `setup_env.sh` checks
# whether the store already exists and only calls this if it does not. On
# Kelvin-2 challenge 1 already built it, so setup_env.sh takes the fast path.
#
# This is challenge 1's installer, vendored so that uploading chal_2/ on its own
# is sufficient. Only two things were changed: the structures it downloads
# (1TNF/3ALQ/7KP7 + P01375/P06804 instead of 6ARU + EGFR), and env.sh now also
# exports the TNF_* interpreter names that pipeline.py looks for first.
#
# Everything else -- python 3.9/3.10 env creation, torch 2.0.1+cu118, the
# dgl 1.1.2+cu118 pin, e3nn, the editable SE3Transformer install, and the
# RFdiffusion checkpoint download with resume and size validation -- is copied
# verbatim. Those pins are fragile and were already proven on this cluster by
# job 10073878's selftest; re-deriving them would be the main way to break this.
#
# Takes 40-70 minutes and about 20 GB of scratch.
# =============================================================================
# =============================================================================
# EGFR pH-selective binder design -- environment setup
# Configured for Kelvin-2 (QUB): partition k2-gpu-v100, CUDA 11.8.
#
# RUN THIS ON A LOGIN NODE (needs internet). Compute nodes stay offline.
#
# Creates two conda envs pinned for NVIDIA V100 / sm_70:
#   rfdiff  py3.9   torch 2.0.1+cu118  dgl  e3nn        -> stages 0,1,2,4,5
#   fold    py3.10  torch 2.0.1+cu118  transformers     -> stage 3
#
# WHY CONDA AND NOT A VENV:
#   `python -m venv` inherits the interpreter you run it with, and the
#   anaconda3/2024.06 module is Python 3.12. torch 2.0.1+cu118 publishes no
#   cp312 wheel, and RFdiffusion's dgl/e3nn stack wants 3.9. Only conda can
#   pin the interpreter version itself, so venvs cannot work here.
#
# WHY THESE VERSIONS:
#   V100 is sm_70: no bfloat16, no FlashAttention-2. torch 2.0.1+cu118 is the
#   last comfortable version shipping sm_70 kernels AND matching a dgl wheel,
#   and cu118 lines up with libs/nvidia-cuda/11.8.0 on this cluster.
#   numpy<2 everywhere: this stack predates the NumPy 2 ABI break.
#
# STORAGE:
#   Weights + envs + caches are ~20 GB, which will usually exceed a Kelvin-2
#   home quota. They go to $EGFR_STORE, auto-detected on scratch. Override:
#       EGFR_STORE=/mnt/scratch2/users/$USER/egfr bash setup_env.sh
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IN="$ROOT/inputs"
mkdir -p "$IN" "$ROOT/work" "$ROOT/results" "$ROOT/logs"

log()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[warn] %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31m[err]  %s\033[0m\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 0. Cluster modules (Kelvin-2 names; harmless elsewhere)
# ---------------------------------------------------------------------------
log "Loading cluster modules"
if command -v module &>/dev/null; then
  module load libs/nvidia-cuda/11.8.0 2>/dev/null \
    || module load cuda/11.8 2>/dev/null \
    || warn "could not load a CUDA 11.8 module; torch bundles its own runtime, continuing"
  module load apps/anaconda3/2024.06/bin 2>/dev/null \
    || module load apps/anaconda3 2>/dev/null \
    || module load anaconda3 2>/dev/null \
    || warn "could not load an anaconda3 module"
  module list 2>&1 | sed 's/^/  /' || true
else
  warn "no 'module' command; assuming conda and CUDA are already on PATH"
fi

# ---------------------------------------------------------------------------
# 1. Pick a storage location with room for ~20 GB
# ---------------------------------------------------------------------------
log "Choosing install location"
if [[ -z "${EGFR_STORE:-}" ]]; then
  for cand in "/mnt/scratch2/users/$USER" "/mnt/scratch/users/$USER" \
              "/mnt/scratch2/$USER" "$HOME/scratch" "/scratch/$USER"; do
    if [[ -d "$cand" && -w "$cand" ]]; then
      EGFR_STORE="$cand/egfr_binder_store"; break
    fi
  done
fi
: "${EGFR_STORE:=$ROOT/store}"
mkdir -p "$EGFR_STORE" || die "cannot create $EGFR_STORE"
echo "  EGFR_STORE = $EGFR_STORE"
if [[ "$EGFR_STORE" == "$ROOT/store" ]]; then
  warn "no scratch directory found, so bulk data goes inside the project dir."
  warn "If this is on your home filesystem you may hit a quota (~20 GB needed)."
  warn "Re-run with: EGFR_STORE=/mnt/scratch2/users/\$USER/egfr bash setup_env.sh"
fi

AVAIL_K=$(df -Pk "$EGFR_STORE" | awk 'NR==2{print $4}')
AVAIL_G=$(( AVAIL_K / 1024 / 1024 ))
echo "  free space here: ${AVAIL_G} GB"
[[ "$AVAIL_G" -lt 25 ]] && warn "under 25 GB free -- install may fail partway"
echo "  your quota, for reference:"
quota -s 2>/dev/null | sed 's/^/    /' || echo "    (quota command unavailable)"

TP="$EGFR_STORE/third_party"
HF="$EGFR_STORE/hf_cache"
ENVS="$EGFR_STORE/envs"
mkdir -p "$TP" "$HF" "$ENVS"

# Keep every cache off the home filesystem.
PIPCACHE="$EGFR_STORE/.pip_cache"
export CONDA_PKGS_DIRS="$EGFR_STORE/.conda_pkgs"
export HF_HOME="$HF"
export TORCH_HOME="$EGFR_STORE/.torch"
mkdir -p "$PIPCACHE" "$CONDA_PKGS_DIRS" "$TORCH_HOME"

# ---------------------------------------------------------------------------
# Neutralise anything that could redirect imports or installs.
#
# On a shared cluster, a loaded module or a ~/.config/pip/pip.conf can set a
# different install prefix (target/prefix/user). pip then reports "Successfully
# installed" while the files land somewhere the interpreter never looks -- the
# exact "success but ModuleNotFoundError" we hit. Every pip call below also
# passes --isolated, which makes pip ignore config files AND PIP_* variables.
# ---------------------------------------------------------------------------
unset PYTHONPATH PYTHONHOME PYTHONSTARTUP PYTHONUSERBASE
unset PIP_TARGET PIP_PREFIX PIP_USER PIP_ROOT PIP_REQUIRE_VIRTUALENV
export PYTHONNOUSERSITE=1   # ignore ~/.local/lib/pythonX.Y/site-packages

diagnose_env() {  # diagnose_env <interpreter>
  local py="$1" envroot
  envroot="$(dirname "$(dirname "$py")")"
  warn "---------------- diagnostics for $py ----------------"
  echo "  interpreter identity:"
  "$py" -c 'import sys
print("    executable :", sys.executable)
print("    version    :", sys.version.split()[0])
print("    prefix     :", sys.prefix)
print("    sys.path   :")
[print("       ", p or "(cwd)") for p in sys.path]' 2>&1 | sed 's/^/  /'
  echo "  PYTHON*/PIP* in the environment:"
  env | grep -iE '^(PYTHON|PIP|CONDA)' | sed 's/^/    /' || echo "    (none)"
  echo "  pip configuration files present:"
  local found=0
  for f in /etc/pip.conf /etc/xdg/pip/pip.conf \
           "$HOME/.config/pip/pip.conf" "$HOME/.pip/pip.conf" "$envroot/pip.conf"; do
    if [[ -f "$f" ]]; then found=1; echo "    --- $f"; sed 's/^/        /' "$f"; fi
  done
  [[ "$found" == 0 ]] && echo "    (none)"
  echo "  pip's own report:"
  "$py" -m pip --version 2>&1 | sed 's/^/    /'
  "$py" -m pip show torch 2>&1 | grep -iE '^(Name|Version|Location)' | sed 's/^/    /' \
    || echo "    pip does not think torch is installed"
  echo "  is torch physically inside the env?"
  ls -d "$envroot"/lib/python*/site-packages/torch 2>&1 | sed 's/^/    /'
  echo "  did it leak into user-site instead?"
  ls -d "$HOME"/.local/lib/python*/site-packages/torch 2>&1 | sed 's/^/    /'
  warn "-----------------------------------------------------"
}

# ---------------------------------------------------------------------------
# 2. Conda
# ---------------------------------------------------------------------------
log "Locating conda"
if ! command -v conda &>/dev/null; then
  for c in "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda; do
    [[ -x "$c/bin/conda" ]] && export PATH="$c/bin:$PATH" && break
  done
fi
command -v conda &>/dev/null \
  || die "conda not found. Try: module load apps/anaconda3/2024.06/bin"

CONDA_BASE="$(conda info --base)"
echo "  conda      $(conda --version)"
echo "  conda base $CONDA_BASE"
# NOTE: we deliberately do NOT use `conda activate` anywhere in this script.
# In a non-interactive shell its shell-function integration is unreliable, and
# when it fails it can return 0 WITHOUT changing PATH -- so `pip` installs into
# one interpreter while `python` is another, and the only symptom is a
# "No module named torch" much later. Every install below names the target
# interpreter by absolute path instead, which cannot be ambiguous.

RFDIFF_ENV="$ENVS/rfdiff"
FOLD_ENV="$ENVS/fold"
TORCH_IDX="https://download.pytorch.org/whl/cu118"

# ---------------------------------------------------------------------------
# 3. Source repos
# ---------------------------------------------------------------------------
log "Fetching source repositories"
clone() { if [[ -d "$2/.git" ]]; then echo "  already present: $2";
          else git clone --depth 1 "$1" "$2"; fi; }
clone https://github.com/RosettaCommons/RFdiffusion.git "$TP/RFdiffusion"
clone https://github.com/dauparas/ProteinMPNN.git       "$TP/ProteinMPNN"

# ---------------------------------------------------------------------------
# 4. RFdiffusion weights (~3 GB)
# ---------------------------------------------------------------------------
log "Downloading RFdiffusion weights"
WDIR="$TP/RFdiffusion/models"; mkdir -p "$WDIR"

# IPD serves each checkpoint from an MD5-hash subdirectory, NOT from a flat
# path -- http://files.ipd.uw.edu/pub/RFdiffusion/<hash>/<name>.pt
# Rather than trust these hashes forever, prefer the ones in the repo we just
# cloned (scripts/download_models.sh), and fall back to the known values.
DL_SCRIPT="$TP/RFdiffusion/scripts/download_models.sh"

url_for() {  # url_for <checkpoint name>
  local name="$1" url=""
  if [[ -f "$DL_SCRIPT" ]]; then
    url=$(grep -oE "https?://[^[:space:]\"']*/${name}" "$DL_SCRIPT" | head -1 || true)
  fi
  if [[ -z "$url" ]]; then
    case "$name" in
      Base_ckpt.pt)         url="http://files.ipd.uw.edu/pub/RFdiffusion/6f5902ac237024bdd0c176cb93063dc4/Base_ckpt.pt" ;;
      Complex_base_ckpt.pt) url="http://files.ipd.uw.edu/pub/RFdiffusion/e29311f6f1bf1af907f9ef9f44b8328b/Complex_base_ckpt.pt" ;;
    esac
  fi
  echo "$url"
}

get_ckpt() {  # get_ckpt <name> <required|optional>
  local name="$1" need="$2" url sz
  # A checkpoint is ~1.3-1.7 GB; anything under 100 MB is a truncated file or
  # an HTML error page that curl happily saved.
  if [[ -s "$WDIR/$name" ]]; then
    sz=$(du -m "$WDIR/$name" | cut -f1)
    if [[ "$sz" -gt 100 ]]; then echo "  have $name (${sz} MB)"; return 0; fi
    warn "  $name is only ${sz} MB -- truncated, re-downloading"
    rm -f "$WDIR/$name"
  fi
  url="$(url_for "$name")"
  [[ -n "$url" ]] || { [[ "$need" == required ]] && die "no URL known for $name"; return 0; }
  echo "  -> $name"
  echo "     $url"
  # A leftover .part from a previous FAILED attempt may hold an HTML error
  # page (an earlier version of this script used a wrong URL and 404'd).
  # Resuming on top of that would silently corrupt the checkpoint, so discard
  # anything too small to be a real partial download.
  if [[ -f "$WDIR/$name.part" ]]; then
    psz=$(stat -c%s "$WDIR/$name.part" 2>/dev/null || stat -f%z "$WDIR/$name.part" 2>/dev/null || echo 0)
    if [[ "$psz" -lt 1048576 ]]; then
      echo "     discarding stale partial (${psz} bytes -- too small to be real)"
      rm -f "$WDIR/$name.part"
    else
      echo "     resuming from $((psz / 1048576)) MB"
    fi
  fi
  # -C - resumes a genuine partial download; these are large files.
  if curl -fL -C - --retry 5 --retry-delay 5 --progress-bar \
          -o "$WDIR/$name.part" "$url"; then
    mv "$WDIR/$name.part" "$WDIR/$name"
    sz=$(du -m "$WDIR/$name" | cut -f1)
    if [[ "$sz" -lt 100 ]]; then
      rm -f "$WDIR/$name"
      [[ "$need" == required ]] && die "$name downloaded but is only ${sz} MB"
      warn "  $name too small, skipping (optional)"
      return 0
    fi
    echo "     ok (${sz} MB)"
  else
    rm -f "$WDIR/$name.part"
    if [[ "$need" == required ]]; then
      die "failed to download $name from
       $url
     If IPD has moved the file, find the current URL in:
       $DL_SCRIPT
     then drop the file into $WDIR/ by hand and re-run this script."
    fi
    warn "  could not fetch $name (optional) -- continuing"
  fi
}

# Complex_base is the PPI/binder model: the only one this pipeline requires.
get_ckpt Complex_base_ckpt.pt required
# Base is only for unconditional generation -- handy to experiment with, but
# the pipeline never loads it, so a failure here is not fatal.
get_ckpt Base_ckpt.pt optional

# ---------------------------------------------------------------------------
# 5. Inputs (small; these stay with the project)
# ---------------------------------------------------------------------------
log "Downloading target structure and sequences"
fetch() { [[ -s "$2" ]] && { echo "  have $(basename "$2")"; return; }
          echo "  -> $(basename "$2")"
          curl -fL --retry 3 --retry-delay 5 -o "$2" "$1" || die "failed: $1"; }
fetch "https://files.rcsb.org/download/1TNF.pdb"        "$IN/1TNF.pdb"
fetch "https://files.rcsb.org/download/3ALQ.pdb"        "$IN/3ALQ.pdb"
fetch "https://files.rcsb.org/download/7KP7.pdb"        "$IN/7KP7.pdb"
fetch "https://rest.uniprot.org/uniprotkb/P01375.fasta" "$IN/P01375.fasta"
fetch "https://rest.uniprot.org/uniprotkb/P06804.fasta" "$IN/P06804.fasta"
grep -q "^ATOM" "$IN/1TNF.pdb" || die "1TNF.pdb has no ATOM records -- download corrupt"

# ---------------------------------------------------------------------------
# 6. env: rfdiff  (python 3.9)
# ---------------------------------------------------------------------------
log "Creating env: $RFDIFF_ENV (python 3.9)"
# pip/setuptools/wheel come from conda, NOT from a later `pip install --upgrade`.
# Upgrading setuptools with pip is what left distutils-precedence.pth pointing
# at a missing _distutils_hack module on the first attempt.
if [[ -x "$RFDIFF_ENV/bin/python" ]]; then echo "  exists; reusing"
else conda create -y --prefix "$RFDIFF_ENV" python=3.9 pip setuptools wheel; fi

# Absolute interpreter path -- see the note in section 2.
RPY="$RFDIFF_ENV/bin/python"
[[ -x "$RPY" ]] || die "env creation failed: no interpreter at $RPY
     Try removing $RFDIFF_ENV and re-running."
echo "  interpreter: $RPY"
echo "  version    : $("$RPY" --version 2>&1)"
case "$("$RPY" --version 2>&1)" in
  *" 3.9."*) ;;
  *) die "expected python 3.9 at $RPY but got $("$RPY" --version 2>&1).
     torch 2.0.1+cu118 has no wheel for other versions." ;;
esac
# --isolated: ignore pip.conf and PIP_* env vars, which on shared clusters are
# the usual reason a package installs "successfully" into the wrong prefix.
pip_rf() {
  "$RPY" -m pip install --isolated --no-input --progress-bar off \
         --cache-dir "$PIPCACHE" "$@"
}

log "  torch 2.0.1+cu118 (ships sm_70 kernels for V100)"
pip_rf "torch==2.0.1+cu118" "torchvision==0.15.2+cu118" --index-url "$TORCH_IDX"
if ! "$RPY" -c "import torch, sys; print('    torch', torch.__version__, 'in', sys.executable)"; then
  diagnose_env "$RPY"
  die "torch installed but is not importable from $RPY.
     The diagnostics above show where it actually went. Most likely one of:
       * a pip.conf setting target/prefix/user  -> listed above if present
       * a PYTHON*/PIP* variable from a module  -> listed above if set
       * a broken setuptools (.pth) in the env  -> rm -rf '$RFDIFF_ENV' and re-run
     Paste that block and I can pinpoint it."
fi

log "  dgl (SE3Transformer dependency -- the most fragile pin here)"
if ! pip_rf "dgl==1.1.2+cu118" -f https://data.dgl.ai/wheels/cu118/repo.html; then
  warn "dgl 1.1.2+cu118 unavailable; trying 1.1.3+cu118"
  pip_rf "dgl==1.1.3+cu118" -f https://data.dgl.ai/wheels/cu118/repo.html \
    || die "dgl install failed. Open https://data.dgl.ai/wheels/cu118/repo.html,
       pick an available +cu118 wheel for cp39, and edit this line."
fi

log "  RFdiffusion python deps"
pip_rf \
  "numpy<2" "scipy<1.12" "pandas<2.1" \
  "e3nn==0.5.1" "hydra-core==1.3.2" "omegaconf==2.3.0" \
  "opt_einsum==3.3.0" "opt_einsum_fx==0.1.4" \
  icecream pyrsistent decorator psutil tqdm requests networkx \
  "biopython==1.81"

log "  SE3Transformer + RFdiffusion (editable)"
pip_rf -e "$TP/RFdiffusion/env/SE3Transformer"
# --no-deps: RFdiffusion's setup.py would otherwise upgrade torch and silently
# destroy sm_70 support.
pip_rf -e "$TP/RFdiffusion" --no-deps

log "  verifying rfdiff env"
"$RPY" - <<'PY'
import sys, torch, numpy, scipy
print(f"  python     {sys.version.split()[0]}")
print(f"  torch      {torch.__version__}")
print(f"  numpy      {numpy.__version__}")

# torch.cuda.get_arch_list() short-circuits on is_available() and returns []
# whenever no GPU is visible, so on a LOGIN NODE it proves nothing at all.
# What is verifiable here is that this is a CUDA 11.8 build rather than a
# CPU-only wheel; the sm_70 check runs on the compute node (pipeline.py).
cu = torch.version.cuda
print(f"  cuda build {cu}")
assert cu is not None, \
    "CPU-only torch build! Reinstall with --index-url https://download.pytorch.org/whl/cu118"
assert cu.startswith("11.8"), f"expected a CUDA 11.8 build, got {cu}"

# Best effort: the compiled arch flags are a baked-in constant, so this often
# works with no device attached. Informational -- never fail the install on it.
try:
    flags = torch._C._cuda_getArchFlags()
    if flags:
        print(f"  compiled for {flags}")
        if "70" not in flags:
            print("  WARNING: sm_70 absent from the compiled arch list!")
except Exception:
    pass

if torch.cuda.is_available():
    arch = torch.cuda.get_arch_list()
    print(f"  arch list  {arch}")
    assert any("70" in a for a in arch), "torch build lacks sm_70 kernels for V100!"
    print(f"  device     {torch.cuda.get_device_name(0)}  (sm_70 OK)")
else:
    print("  no GPU here (normal on a login node);")
    print("  sm_70 is verified on the compute node before any model runs")
import dgl, e3nn, se3_transformer
print(f"  dgl        {dgl.__version__}")
print(f"  e3nn       {e3nn.__version__}")
print("  se3_transformer OK")
PY

# ---------------------------------------------------------------------------
# 7. env: fold  (python 3.10)
# ---------------------------------------------------------------------------
log "Creating env: $FOLD_ENV (python 3.10)"
if [[ -x "$FOLD_ENV/bin/python" ]]; then echo "  exists; reusing"
else conda create -y --prefix "$FOLD_ENV" python=3.10 pip setuptools wheel; fi

FPY="$FOLD_ENV/bin/python"
[[ -x "$FPY" ]] || die "env creation failed: no interpreter at $FPY"
echo "  interpreter: $FPY"
echo "  version    : $("$FPY" --version 2>&1)"
pip_fold() {
  "$FPY" -m pip install --isolated --no-input --progress-bar off \
         --cache-dir "$PIPCACHE" "$@"
}

pip_fold "torch==2.0.1+cu118" --index-url "$TORCH_IDX"
if ! "$FPY" -c "import torch, sys; print('    torch', torch.__version__, 'in', sys.executable)"; then
  diagnose_env "$FPY"
  die "torch installed but is not importable from $FPY (see diagnostics above)"
fi
pip_fold "transformers==4.35.2" "accelerate==0.24.1" \
  "numpy<2" "scipy<1.12" "biopython==1.81" "pandas<2.1" tqdm

log "  caching ESMFold weights (~2.6 GB) so compute nodes need no network"
"$FPY" - <<'PY'
import os
from transformers import AutoTokenizer, EsmForProteinFolding
mid = "facebook/esmfold_v1"
AutoTokenizer.from_pretrained(mid)
EsmForProteinFolding.from_pretrained(mid, low_cpu_mem_usage=True)
print("  cached under", os.environ.get("HF_HOME"))
PY

log "  verifying fold env"
"$FPY" - <<'PY'
import torch, transformers
from transformers import EsmForProteinFolding
print(f"  torch        {torch.__version__}")
print(f"  cuda build   {torch.version.cuda}")
print(f"  transformers {transformers.__version__}")
# A CPU-only wheel here would not error, it would just make ESMFold crawl.
assert torch.version.cuda is not None, "CPU-only torch build in the fold env!"
print("  EsmForProteinFolding OK (pure PyTorch, no OpenFold kernels needed)")
PY

# ---------------------------------------------------------------------------
# 8. Record everything the job scripts need
# ---------------------------------------------------------------------------
cat > "$ROOT/env.sh" <<EOF
# Auto-generated by setup_env.sh on $(date -u +%Y-%m-%dT%H:%M:%SZ)
# Cluster: Kelvin-2 (QUB)
export EGFR_ROOT="$ROOT"
export EGFR_STORE="$EGFR_STORE"
export EGFR_TP="$TP"
export EGFR_INPUTS="$IN"
export EGFR_RFDIFF_ENV="$RFDIFF_ENV"
export EGFR_FOLD_ENV="$FOLD_ENV"
# Call these interpreters directly. Do NOT rely on 'conda activate' in a
# script -- it can return 0 without changing PATH, which puts pip and python
# in different environments.
export EGFR_RFDIFF_PY="$RFDIFF_ENV/bin/python"
export EGFR_FOLD_PY="$FOLD_ENV/bin/python"
export TNF_RFDIFF_PY="$RFDIFF_ENV/bin/python"
export TNF_FOLD_PY="$FOLD_ENV/bin/python"
export TNF_STORE="$EGFR_STORE"
export TNF_ROOT="$ROOT"
export CONDA_BASE="$CONDA_BASE"

# Caches off the home filesystem
export HF_HOME="$HF"
export TORCH_HOME="$EGFR_STORE/.torch"
export CONDA_PKGS_DIRS="$EGFR_STORE/.conda_pkgs"
export PIP_CACHE_DIR="$PIPCACHE"

# Keep imports unambiguous at run time as well as at install time. A module
# that exports PYTHONPATH/PYTHONHOME, or a stray ~/.local/lib/python3.9, can
# shadow the pinned packages in these envs with incompatible versions.
unset PYTHONPATH PYTHONHOME PYTHONSTARTUP PYTHONUSERBASE
export PYTHONNOUSERSITE=1

# Compute nodes have no internet; fail loudly instead of hanging on a download.
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# V100 has no bf16; keep numerics predictable.
export NVIDIA_TF32_OVERRIDE=0

# Kelvin-2 module names
export EGFR_MODULES="libs/nvidia-cuda/11.8.0 apps/anaconda3/2024.06/bin"
EOF

log "Setup complete"
cat <<EOF

  project   : $ROOT
  store     : $EGFR_STORE   ($(du -sh "$EGFR_STORE" 2>/dev/null | cut -f1) used)
  envs      : $RFDIFF_ENV
              $FOLD_ENV

Next -- validate on CPU (~20 s) before using GPU time:

  cd $ROOT
  source env.sh
  "\$EGFR_RFDIFF_PY" pipeline.py --selftest

Then run:

  bash run_interactive.sh        # interactive V100, ~30-60 min, live output
  sbatch submit.sbatch           # or batch

EOF

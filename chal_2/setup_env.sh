#!/usr/bin/env bash
# =============================================================================
# Challenge 02 (TNF-alpha) environment setup for Kelvin-2.
#
#   bash setup_env.sh
#
# This does NOT build anything by default. Challenge 1 already created two
# conda envs and cloned RFdiffusion and ProteinMPNN into a scratch store, and
# job 10073878's selftest confirmed all of it works:
#
#     rfdiff  py3.9   torch 2.0.1+cu118  dgl  e3nn     -> stages 0,1,2,4,5
#     fold    py3.10  torch 2.0.1+cu118  transformers  -> stage 3
#
# Challenge 2 needs exactly the same tools against a different target, so this
# script's job is to FIND that store, verify it, and write an env.sh pointing at
# it. Rebuilding would take an hour, consume another ~20 GB of scratch, and risk
# ending up with a subtly different torch.
#
# If the store is missing, this script says so and tells you to run challenge
# 1's setup_env.sh, which is the thing that knows how to build it. There is no
# second copy of that logic here to drift out of step.
#
# Override the location if it is somewhere else:
#     TNF_STORE=/mnt/scratch2/users/$USER/egfr_binder_store bash setup_env.sh
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

c_i=$'\033[1;36m'; c_o=$'\033[1;32m'; c_w=$'\033[1;33m'; c_e=$'\033[1;31m'; c_x=$'\033[0m'
log()  { echo "${c_i}[info]${c_x} $*"; }
ok()   { echo "${c_o}[  ok]${c_x} $*"; }
warn() { echo "${c_w}[warn]${c_x} $*"; }
die()  { echo "${c_e}[ err]${c_x} $*" >&2; exit 1; }

echo "=============================================================="
echo " Challenge 02 (TNF-alpha) -- environment setup"
echo " host: $(hostname)   user: ${USER:-unknown}"
echo "=============================================================="

# --- Modules (same as challenge 1) ------------------------------------------
if command -v module &>/dev/null; then
  module load libs/nvidia-cuda/11.8.0 2>/dev/null \
    || module load cuda/11.8 2>/dev/null \
    || warn "no CUDA 11.8 module; torch bundles its own runtime, continuing"
  module load apps/anaconda3/2024.06/bin 2>/dev/null \
    || module load apps/anaconda3 2>/dev/null \
    || module load anaconda3 2>/dev/null \
    || warn "no anaconda3 module loaded"
fi

# --- Locate the shared store -------------------------------------------------
if [[ -z "${TNF_STORE:-}" ]]; then
  # Prefer whatever challenge 1 recorded; it is authoritative.
  if [[ -f "$ROOT/../chal_1/env.sh" ]]; then
    # shellcheck disable=SC1091
    TNF_STORE="$(bash -c "source '$ROOT/../chal_1/env.sh' >/dev/null 2>&1; \
                          echo \"\${EGFR_STORE:-}\"")"
    [[ -n "$TNF_STORE" ]] && log "store taken from ../chal_1/env.sh"
  fi
fi
if [[ -z "${TNF_STORE:-}" ]]; then
  for cand in "/mnt/scratch2/users/${USER:-}" "/mnt/scratch/users/${USER:-}" \
              "/users/${USER:-}/sharedscratch"; do
    if [[ -d "$cand/egfr_binder_store" ]]; then
      TNF_STORE="$cand/egfr_binder_store"; break
    fi
  done
fi
: "${TNF_STORE:=$ROOT/store}"
log "TNF_STORE = $TNF_STORE"

TP="$TNF_STORE/third_party"
ENVS="$TNF_STORE/envs"
HF="$TNF_STORE/hf_cache"
RFDIFF_ENV="$ENVS/rfdiff"
FOLD_ENV="$ENVS/fold"

# --- Verify, do not rebuild --------------------------------------------------
missing=0
check () {  # check <path> <label>
  if [[ -e "$1" ]]; then ok "$2"; else warn "MISSING: $2 ($1)"; missing=$((missing+1)); fi
}

log "checking the shared conda environments"
check "$RFDIFF_ENV/bin/python" "rfdiff env interpreter"
check "$FOLD_ENV/bin/python"   "fold env interpreter"

log "checking the shared third-party tools"
check "$TP/RFdiffusion/scripts/run_inference.py"            "RFdiffusion run_inference.py"
check "$TP/RFdiffusion/models/Complex_base_ckpt.pt"         "RFdiffusion Complex_base_ckpt.pt"
check "$TP/ProteinMPNN/protein_mpnn_run.py"                 "ProteinMPNN protein_mpnn_run.py"
check "$TP/ProteinMPNN/helper_scripts/parse_multiple_chains.py" "ProteinMPNN parse_multiple_chains.py"
check "$TP/ProteinMPNN/helper_scripts/assign_fixed_chains.py"   "ProteinMPNN assign_fixed_chains.py"

if (( missing > 0 )); then
  echo
  warn "$missing component(s) missing from $TNF_STORE"
  if [[ "${NO_INSTALL:-0}" == "1" ]]; then
    die "NO_INSTALL=1 set, so not building. Point TNF_STORE at an existing
store, or re-run without NO_INSTALL to build one."
  fi
  # Guard against running the installer on a laptop by mistake. It would get
  # some way in before conda or the CUDA module failed, which is a confusing
  # several minutes rather than an immediate answer.
  if ! command -v sbatch &>/dev/null && [[ "${FORCE_INSTALL:-0}" != "1" ]]; then
    die "this does not look like an HPC login node (no sbatch on PATH), and
the shared store was not found at
    $TNF_STORE

If you are on your laptop: you do not want to build a 20 GB CUDA environment
here. Upload the directory first and run this on Kelvin-2:

    bash push.sh
    ssh -i ~/.ssh/my-kelvin-key 40423097@login.kelvin.alces.network
    cd /users/40423097/sharedscratch/Protein_create/chal_2 && bash run.sh

If the store really does live somewhere else, point at it:

    TNF_STORE=/path/to/egfr_binder_store bash setup_env.sh

To build anyway (you almost certainly do not want this): FORCE_INSTALL=1"
  fi

  echo
  log "falling back to the full installer (install_env.sh)"
  log "this builds two conda envs, clones RFdiffusion and ProteinMPNN, and"
  log "downloads the Complex_base checkpoint: 40-70 min, ~20 GB of scratch."
  echo
  if [[ ! -f "$ROOT/install_env.sh" ]]; then
    die "install_env.sh is missing from $ROOT -- re-upload the directory."
  fi
  # The installer writes its own env.sh with both TNF_* and EGFR_* names, so
  # after it finishes there is nothing left for this script to do.
  EGFR_STORE="$TNF_STORE" bash "$ROOT/install_env.sh"
  echo
  ok "installer finished; env.sh written by install_env.sh"
  echo
  echo "=============================================================="
  echo " Next:"
  echo "   source env.sh"
  echo "   \$TNF_RFDIFF_PY pipeline.py --selftest"
  echo "   sbatch --export=ALL,LEARN=1 submit.sbatch"
  echo "=============================================================="
  exit 0
fi

# --- Soluble ProteinMPNN weights --------------------------------------------
# Stage mpnn passes --use_soluble_model. The weights ship with the repo, but an
# old checkout may predate them, and the failure mode is an unhelpful KeyError
# several minutes in.
if [[ -f "$TP/ProteinMPNN/soluble_model_weights/v_48_020.pt" ]]; then
  ok "ProteinMPNN soluble weights present"
else
  warn "soluble_model_weights/v_48_020.pt not found in the ProteinMPNN checkout"
  warn "  stage mpnn will fail with --use_soluble_model; either update the"
  warn "  checkout or run: python pipeline.py --stage mpnn  with that flag removed"
fi

# --- Quick import check in each env -----------------------------------------
log "verifying imports inside each env"
"$RFDIFF_ENV/bin/python" - <<'PY' || die "rfdiff env cannot import its own stack"
import sys
import numpy, scipy, Bio, torch
print(f"  rfdiff: python {sys.version.split()[0]}, torch {torch.__version__}, "
      f"numpy {numpy.__version__}, biopython {Bio.__version__}")
try:
    import dgl
    print(f"  rfdiff: dgl {dgl.__version__} (RFdiffusion needs this)")
except ImportError:
    print("  rfdiff: WARNING dgl missing -- RFdiffusion will not run")
PY
fold_check () {
  "$FOLD_ENV/bin/python" - <<'PY'
import sys, torch, transformers, huggingface_hub
print(f"  fold:   python {sys.version.split()[0]}, torch {torch.__version__}, "
      f"transformers {transformers.__version__}, "
      f"huggingface_hub {huggingface_hub.__version__}")
PY
}

if ! fold_check; then
  # Dependency drift, and this one is predictable enough to repair rather than
  # just report. huggingface_hub released 1.x; the transformers pinned in this
  # env requires >=0.16.4,<1.0, so an unrelated `pip install -U` anywhere in
  # the env silently breaks every `import transformers`:
  #
  #   ImportError: huggingface-hub>=0.16.4,<1.0 is required ... found 1.14.0
  #
  # The upper bound below is 0.26, not 1.0: `cached_download` was removed in
  # huggingface_hub 0.26 and the transformers versions that pair with torch
  # 2.0.1 still import it, so "newest thing under 1.0" is not safe either.
  warn "fold env import failed -- attempting the known repair"
  warn "  pinning huggingface_hub to >=0.16.4,<0.26 inside $FOLD_ENV"
  if "$FOLD_ENV/bin/python" -m pip install --isolated --no-input \
       --progress-bar off "huggingface_hub>=0.16.4,<0.26"; then
    if fold_check; then
      ok "repaired: fold env imports again"
    else
      die "fold env still cannot import after pinning huggingface_hub.
Show the traceback above to diagnose, or run the stages without ESMFold:

    sbatch --export=ALL,SKIP_ESMFOLD=1 submit.sbatch

That skips stage 'fold', which costs you the pLDDT and RMSD filters -- the
most valuable ones in the pipeline -- so treat it as a way to keep moving,
not as a fix."
    fi
  else
    die "could not pin huggingface_hub in $FOLD_ENV (no network on this node?).
Run it yourself on a login node:

    source env.sh
    \"\$TNF_FOLD_PY\" -m pip install --isolated \"huggingface_hub>=0.16.4,<0.26\""
  fi
fi
ok "both envs import cleanly"

# --- Inputs ------------------------------------------------------------------
IN="$ROOT/inputs"
mkdir -p "$IN" "$ROOT/work" "$ROOT/results" "$ROOT/logs"
log "checking target inputs (downloaded on first use if absent)"
for f in 1TNF.pdb 3ALQ.pdb P01375.fasta P06804.fasta; do
  if [[ -f "$IN/$f" ]]; then ok "  $f"; else warn "  $f absent (will download)"; fi
done

# --- Write env.sh ------------------------------------------------------------
cat > "$ROOT/env.sh" <<EOF
# Auto-generated by setup_env.sh on $(date -u +%Y-%m-%dT%H:%M:%SZ)
# Cluster: Kelvin-2 (QUB). Challenge 02, TNF-alpha.
#
# The envs and third-party tools are SHARED with challenge 1 -- this file only
# points at them. Both the TNF_* and EGFR_* names are exported because
# src/tnf_hpc/tools.py falls back to the EGFR_* ones, so an env.sh written by
# either project works for either pipeline.
export TNF_ROOT="$ROOT"
export TNF_STORE="$TNF_STORE"
export TNF_TP="$TP"
export TNF_INPUTS="$IN"
export TNF_RFDIFF_ENV="$RFDIFF_ENV"
export TNF_FOLD_ENV="$FOLD_ENV"

# Call these interpreters directly. Do NOT rely on 'conda activate' in a
# script -- it can return 0 without changing PATH, which puts pip and python in
# different environments.
export TNF_RFDIFF_PY="$RFDIFF_ENV/bin/python"
export TNF_FOLD_PY="$FOLD_ENV/bin/python"

# Back-compat aliases
export EGFR_STORE="$TNF_STORE"
export EGFR_RFDIFF_PY="$RFDIFF_ENV/bin/python"
export EGFR_FOLD_PY="$FOLD_ENV/bin/python"

# Caches off the home filesystem (home quota on Kelvin-2 is small)
export HF_HOME="$HF"
export TORCH_HOME="$TNF_STORE/.torch"
export CONDA_PKGS_DIRS="$TNF_STORE/.conda_pkgs"
export PIP_CACHE_DIR="$TNF_STORE/.pip_cache"

# Keep imports unambiguous at run time. A module that exports PYTHONPATH, or a
# stray ~/.local/lib/python3.9, can shadow the pinned packages in these envs.
unset PYTHONPATH PYTHONHOME PYTHONSTARTUP PYTHONUSERBASE
export PYTHONNOUSERSITE=1
EOF
ok "wrote $ROOT/env.sh"

echo
echo "=============================================================="
echo " Next:"
echo "   source env.sh"
echo "   \$TNF_RFDIFF_PY pipeline.py --selftest       # ~60 s, CPU only"
echo "   sbatch submit.sbatch                        # the real run"
echo "   sbatch --export=ALL,LEARN=1 submit.sbatch    # ~1 h smoke test"
echo "=============================================================="

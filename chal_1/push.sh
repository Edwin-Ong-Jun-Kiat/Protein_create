#!/usr/bin/env bash
# =============================================================================
# Sync this project to Kelvin-2 (QUB), and pull results back.
# RUN THIS ON YOUR MAC, not on the cluster.
#
#   bash push.sh              # upload the scripts (fast, default)
#   bash push.sh --results    # download results/ back to your Mac
#   bash push.sh --shell      # just ssh in, landing in the project directory
#   bash push.sh --dry        # show what would transfer, change nothing
# =============================================================================
set -euo pipefail

# --- Connection ------------------------------------------------------------
KEY="${KELVIN_KEY:-$HOME/.ssh/my-kelvin-key}"
PORT="${KELVIN_PORT:-55890}"
HOST="${KELVIN_HOST:-40423097@login.kelvin.alces.network}"
# ~/sharedscratch is a symlink to /mnt/scratch2/users/<id> on Kelvin.
REMOTE="${KELVIN_REMOTE:-/users/40423097/sharedscratch/Protein_create/chal_1}"

LOCAL="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SSH_OPTS=(-p "$PORT" -i "$KEY")

[[ -f "$KEY" ]] || { echo "ERROR: ssh key not found at $KEY" >&2; exit 1; }

MODE="push"; DRY=""
for a in "$@"; do
  case "$a" in
    --results|--pull) MODE="pull" ;;
    --shell|--ssh)    MODE="shell" ;;
    --dry|--dry-run)  DRY="1" ;;
    *) echo "unknown option: $a" >&2; exit 1 ;;
  esac
done

# rsync is much better here (only sends changed bytes), but it must exist on
# BOTH ends. Fall back to scp if the cluster does not have it.
have_remote_rsync() {
  ssh "${SSH_OPTS[@]}" -o BatchMode=yes "$HOST" 'command -v rsync >/dev/null 2>&1'
}

case "$MODE" in
shell)
  echo "connecting; you will land in $REMOTE"
  exec ssh "${SSH_OPTS[@]}" -t "$HOST" "cd '$REMOTE' 2>/dev/null; exec \$SHELL -l"
  ;;

pull)
  mkdir -p "$LOCAL/results_from_kelvin"
  echo "==> downloading results/ -> $LOCAL/results_from_kelvin/"
  if have_remote_rsync; then
    rsync -avz ${DRY:+--dry-run} -e "ssh ${SSH_OPTS[*]}" \
      "$HOST:$REMOTE/results/" "$LOCAL/results_from_kelvin/"
  else
    [[ -n "$DRY" ]] && { echo "(scp fallback has no dry-run; skipping)"; exit 0; }
    scp -r "${SSH_OPTS[@]}" "$HOST:$REMOTE/results/*" "$LOCAL/results_from_kelvin/"
  fi
  echo
  echo "pulled into $LOCAL/results_from_kelvin/"
  ls -la "$LOCAL/results_from_kelvin/" 2>/dev/null | sed 's/^/  /'
  ;;

push)
  # Only the scripts live here; work/, results/ and store/ are generated on the
  # cluster and must never be overwritten from the Mac.
  FILES=(pipeline.py setup_env.sh submit.sbatch run_interactive.sh README.md push.sh)
  echo "==> uploading to $HOST:$REMOTE"
  for f in "${FILES[@]}"; do
    [[ -f "$LOCAL/$f" ]] || { echo "  missing locally: $f" >&2; exit 1; }
    printf '  %s\n' "$f"
  done

  # Named files only (never a directory), so cluster-generated work/, results/,
  # logs/, inputs/ and env.sh can never be clobbered from the Mac -- and there
  # is no risk of scp nesting a second copy at $REMOTE/chal_1 the way
  # `scp -r dir host:parent` does when the target already exists.
  SRC=(); for f in "${FILES[@]}"; do SRC+=("$LOCAL/$f"); done

  ssh "${SSH_OPTS[@]}" "$HOST" "mkdir -p '$REMOTE'"

  sent=""
  if have_remote_rsync && command -v rsync >/dev/null; then
    # macOS ships openrsync, not GNU rsync, and it rejects some options --
    # so treat any rsync failure as "fall back to scp" rather than fatal.
    if rsync -avz ${DRY:+--dry-run} -e "ssh ${SSH_OPTS[*]}" \
         "${SRC[@]}" "$HOST:$REMOTE/"; then
      sent="rsync"
    else
      echo "  (rsync failed -- falling back to scp)"
    fi
  fi
  if [[ -z "$sent" ]]; then
    [[ -n "$DRY" ]] && { echo "(scp has no dry-run; nothing sent)"; exit 0; }
    scp "${SSH_OPTS[@]}" "${SRC[@]}" "$HOST:$REMOTE/"
    sent="scp"
  fi
  echo "  transferred via $sent"

  echo
  echo "done. Next, on Kelvin-2:"
  echo "  cd $REMOTE"
  echo "  bash setup_env.sh                        # if setup is not finished"
  echo "  source env.sh && \"\$EGFR_RFDIFF_PY\" pipeline.py --selftest"
  echo "  bash run_interactive.sh"
  ;;
esac

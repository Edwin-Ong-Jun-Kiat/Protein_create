#!/usr/bin/env bash
# =============================================================================
# Upload this whole directory to Kelvin-2.
#
#   bash push.sh              upload everything (the normal case)
#   bash push.sh --dry-run    show what would be sent
#   bash push.sh --pull       fetch results/ back into results_from_kelvin/
#   bash push.sh --shell      just ssh in, landing in the project directory
#
# Unlike challenge 1's version this sends the DIRECTORY, not a list of named
# files -- challenge 2 is not a single script, it needs src/tnf_hpc/ as well,
# and a named-file list silently omits it.
#
# Cluster-generated state is excluded rather than protected by omission:
# work/, results/, logs/ and env.sh all live on the cluster and must never be
# overwritten from the Mac. env.sh in particular holds absolute scratch paths
# that mean nothing here.
# =============================================================================
set -euo pipefail

LOCAL="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KEY="${KELVIN_KEY:-$HOME/.ssh/my-kelvin-key}"
HOST="${KELVIN_HOST:-40423097@login.kelvin.alces.network}"
REMOTE="${KELVIN_REMOTE:-/users/40423097/sharedscratch/Protein_create/chal_2}"
# Kelvin-2 does not listen on 22. The first version of this script assumed it
# did and timed out with "Operation timed out", which looks like a network
# problem rather than a wrong port.
PORT="${KELVIN_PORT:-55890}"

# ssh takes -p, scp takes -P. Keeping two arrays avoids getting that backwards.
SSH_OPTS=(-p "$PORT" -i "$KEY" -o StrictHostKeyChecking=accept-new)
SCP_OPTS=(-P "$PORT" -i "$KEY" -o StrictHostKeyChecking=accept-new)
[[ -f "$KEY" ]] || { echo "ERROR: ssh key not found at $KEY" >&2; exit 1; }

MODE="push"
DRY=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run|-n)  DRY=1 ;;
    --pull)        MODE="pull" ;;
    --shell|--ssh) MODE="shell" ;;
    -h|--help)     sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
  shift
done

# Never send these. The first four are produced on the cluster; the rest is
# local noise that would only confuse a diff.
TAR_EXCL=(--exclude ./work --exclude ./results --exclude ./logs
          --exclude ./env.sh --exclude ./store --exclude __pycache__
          --exclude '*.pyc' --exclude .DS_Store --exclude '*.bak'
          --exclude ./results_from_kelvin)
RSYNC_EXCL=(--exclude 'work/' --exclude 'results/' --exclude 'logs/'
            --exclude 'env.sh' --exclude 'store/' --exclude '__pycache__/'
            --exclude '*.pyc' --exclude '.DS_Store' --exclude '*.bak'
            --exclude 'results_from_kelvin/')

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
  echo "==> pulling $HOST:$REMOTE/results/"
  if have_remote_rsync && command -v rsync >/dev/null; then
    rsync -avz ${DRY:+--dry-run} -e "ssh ${SSH_OPTS[*]}" \
      "$HOST:$REMOTE/results/" "$LOCAL/results_from_kelvin/"
  else
    [[ -n "$DRY" ]] && { echo "(scp has no dry-run)"; exit 0; }
    scp -r "${SCP_OPTS[@]}" "$HOST:$REMOTE/results/*" "$LOCAL/results_from_kelvin/"
  fi
  echo "done -> $LOCAL/results_from_kelvin/"
  ;;

push)
  echo "==> uploading $LOCAL"
  echo "    to  $HOST:$REMOTE"
  echo "    port $PORT, key $KEY"
  echo "    excluding work/ results/ logs/ env.sh store/ __pycache__"
  echo

  # A dry run must not touch the cluster at all -- not even to mkdir, and not
  # even to probe for rsync, since both need a working connection and the
  # point of --dry-run is often to check the file list while offline.
  if [[ -n "$DRY" ]]; then
    echo "(dry run) files that would be sent:"
    tar -C "$LOCAL" "${TAR_EXCL[@]}" -cf - . | tar -tf - \
      | grep -v '/$' | sort | sed 's/^/  /'
    echo
    echo "  $(tar -C "$LOCAL" "${TAR_EXCL[@]}" -cf - . | tar -tf - \
              | grep -cv '/$') files, $(du -sh "$LOCAL" | cut -f1) on disk"
    echo "  nothing was sent and the cluster was not contacted"
    exit 0
  fi

  ssh "${SSH_OPTS[@]}" "$HOST" "mkdir -p '$REMOTE'"

  sent=""
  if have_remote_rsync && command -v rsync >/dev/null; then
    # The trailing slash on the source is load-bearing: "$LOCAL/" copies the
    # CONTENTS into $REMOTE. Without it rsync creates $REMOTE/chal_2/.
    # macOS ships openrsync, which rejects some GNU options, so a failure here
    # falls back to tar-over-ssh rather than being fatal.
    if rsync -avz "${RSYNC_EXCL[@]}" \
         -e "ssh -p $PORT -i $KEY -o StrictHostKeyChecking=accept-new" \
         "$LOCAL/" "$HOST:$REMOTE/"; then
      sent="rsync"
    else
      echo "  (rsync failed -- falling back to tar over ssh)"
    fi
  fi

  if [[ -z "$sent" ]]; then
    # tar over ssh beats scp -r here: one stream, it honours the excludes, and
    # it will not nest a second copy at $REMOTE/chal_2 the way scp -r does when
    # the target directory already exists.
    tar -C "$LOCAL" "${TAR_EXCL[@]}" -czf - . \
      | ssh "${SSH_OPTS[@]}" "$HOST" "tar -C '$REMOTE' -xzf -"
    sent="tar+ssh"
  fi
  echo
  echo "  transferred via $sent"

  if [[ -z "$DRY" ]]; then
    ssh "${SSH_OPTS[@]}" "$HOST" \
      "chmod +x '$REMOTE'/*.sh 2>/dev/null; \
       find '$REMOTE/src' -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null; \
       mkdir -p '$REMOTE/work' '$REMOTE/results' '$REMOTE/logs'; true"
    echo
    echo "done. Now, on Kelvin-2:"
    echo
    echo "  ssh -p $PORT -i $KEY $HOST"
    echo "  cd $REMOTE"
    echo "  bash run.sh            # setup + selftest + submit a ~1 h test"
    echo "  bash run.sh --full     # setup + selftest + submit the full run"
  fi
  ;;
esac

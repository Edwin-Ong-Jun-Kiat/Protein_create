#!/usr/bin/env bash
# =============================================================================
# Upload this directory to Kelvin-2 (QUB).
#
#   bash push.sh              upload everything (the normal case)
#   bash push.sh --dry-run    show what would be sent, contacting nothing
#   bash push.sh --pull       fetch results/ back into results_from_kelvin/
#   bash push.sh --shell      just ssh in, landing in the project directory
#   bash push.sh --stop       close the shared connection now
#
# Connection details are challenge 1's and 2's, unchanged:
#
#   host  40423097@login.kelvin.alces.network
#   port  55890          -- Kelvin-2 does NOT listen on 22. The first version
#                           of chal_1/push.sh assumed it did and timed out with
#                           "Operation timed out", which reads like a network
#                           outage rather than a wrong port.
#   key   ~/.ssh/my-kelvin-key
#
# YOU SHOULD BE ASKED TO AUTHENTICATE AT MOST ONCE PER RUN
# -------------------------------------------------------
# A push needs four round trips to the cluster -- probe for rsync, mkdir the
# remote directory, transfer, then fix permissions. Each is a separate ssh, and
# a separate ssh means a separate passphrase prompt. The first version of this
# script hid connection reuse behind an opt-in --mux flag, so the default path
# prompted four times. That was the wrong default and this is the fix.
#
# Now every ssh, scp and rsync here shares ONE connection:
#
#   ControlMaster=auto   the first connection becomes a master and listens on
#                        a local socket; later ones reuse it instead of
#                        re-authenticating
#   ControlPath=~/.ssh/cm-%C   %C is a hash of user+host+port, so the socket
#                        name cannot exceed the ~104 char unix socket limit
#                        the way a literal %r@%h:%p path can
#   ControlPersist=10m   the socket outlives this script, so a push followed
#                        by a --pull a minute later does not prompt again.
#                        Override with KELVIN_PERSIST=1h, or close it early
#                        with `bash push.sh --stop`.
#
# The master is opened by an explicit warm-up call before any transfer, not
# lazily by whichever command happens to run first. That matters: the tar path
# pipes data into ssh's stdin, and a passphrase prompt appearing in the middle
# of a pipeline is at best confusing and at worst reads the wrong input.
#
# TO BE ASKED ZERO TIMES, load the key into your agent once per login:
#
#   ssh-add --apple-use-keychain ~/.ssh/my-kelvin-key    # macOS; stores it in
#                                                        # the login keychain
#   ssh-add -l                                           # check what is loaded
#
# This script reports which of the three cases you are in before it connects.
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
PORT="${KELVIN_PORT:-55890}"
REMOTE="${KELVIN_REMOTE:-/users/40423097/sharedscratch/Protein_create/chal_4_af}"
PERSIST="${KELVIN_PERSIST:-10m}"

MODE="push"; DRY=""; MUX=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run|-n)  DRY=1 ;;
    --pull)        MODE="pull" ;;
    --shell|--ssh) MODE="shell" ;;
    --stop)        MODE="stop" ;;
    --no-mux)      MUX=0 ;;   # escape hatch if multiplexing ever misbehaves
    -h|--help)     sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
  shift
done

# --- Connection options -----------------------------------------------------
# ssh takes -p, scp takes -P. Keeping two arrays avoids getting that backwards.
SSH_OPTS=(-p "$PORT" -o StrictHostKeyChecking=accept-new)
SCP_OPTS=(-P "$PORT" -o StrictHostKeyChecking=accept-new)

# The key is optional, unlike in chal_1/chal_2. Kelvin-2 may be back to
# password + verification for this account, and refusing to run without a key
# file would block the one workflow that still works.
if [[ -f "$KEY" ]]; then
  SSH_OPTS+=(-i "$KEY"); SCP_OPTS+=(-i "$KEY")
else
  echo "note: no ssh key at $KEY -- expect a password prompt." >&2
  KEY=""
fi

mkdir -p "$HOME/.ssh" 2>/dev/null || true
CM_PATH="${KELVIN_SOCK:-$HOME/.ssh/cm-%C}"
if [[ "$MUX" == "1" ]]; then
  CM_OPTS=(-o ControlMaster=auto -o "ControlPath=$CM_PATH"
           -o "ControlPersist=$PERSIST")
  SSH_OPTS+=("${CM_OPTS[@]}"); SCP_OPTS+=("${CM_OPTS[@]}")
fi

# Report which of the three auth cases we are in, before connecting, so a
# prompt is never a surprise.
auth_note() {
  [[ "$MODE" == "stop" || -n "$DRY" ]] && return 0
  if [[ -n "$KEY" ]] && ssh-add -l 2>/dev/null | grep -qF "$(
        ssh-keygen -lf "$KEY" 2>/dev/null | awk '{print $2}')"; then
    echo "auth: key is loaded in ssh-agent -- you should not be prompted at all"
  elif [[ "$MUX" == "1" ]]; then
    echo "auth: one shared connection (ControlPersist=$PERSIST)"
    echo "      -> at most ONE prompt for this run, and none for $PERSIST after"
    [[ -n "$KEY" ]] && echo "      to stop being asked entirely:" \
      && echo "        ssh-add --apple-use-keychain $KEY"
  else
    echo "auth: --no-mux given, so every round trip authenticates separately"
  fi
}

# Open the master connection in the foreground, where a prompt can be answered
# cleanly. `true` is the cheapest possible remote command.
warm_connection() {
  [[ "$MUX" == "1" ]] || return 0
  if ssh -O check "${SSH_OPTS[@]}" "$HOST" &>/dev/null; then
    echo "==> reusing the existing shared connection"
    return 0
  fi
  echo "==> opening one shared connection to $HOST"
  ssh "${SSH_OPTS[@]}" "$HOST" true \
    || { echo "ERROR: could not connect to $HOST on port $PORT" >&2; exit 1; }
}

# Never send these. The first four are produced on the cluster; .venv and
# store/ are machine-specific and would be actively harmful to copy.
TAR_EXCL=(--exclude ./work --exclude ./results --exclude ./logs
          --exclude ./env.sh --exclude ./store --exclude ./.venv
          --exclude __pycache__ --exclude '*.pyc' --exclude .DS_Store
          --exclude '*.bak' --exclude ./results_from_kelvin)
RSYNC_EXCL=(--exclude 'work/' --exclude 'results/' --exclude 'logs/'
            --exclude 'env.sh' --exclude 'store/' --exclude '.venv/'
            --exclude '__pycache__/' --exclude '*.pyc' --exclude '.DS_Store'
            --exclude '*.bak' --exclude 'results_from_kelvin/')

# No BatchMode here: it would make this probe fail outright rather than
# authenticate, on the very run where nothing is cached yet.
have_remote_rsync() {
  ssh "${SSH_OPTS[@]}" "$HOST" 'command -v rsync >/dev/null 2>&1'
}

case "$MODE" in
stop)
  if ssh -O exit "${SSH_OPTS[@]}" "$HOST" 2>/dev/null; then
    echo "closed the shared connection"
  else
    echo "no shared connection was open"
  fi
  ;;

shell)
  auth_note
  echo "connecting; you will land in $REMOTE"
  exec ssh "${SSH_OPTS[@]}" -t "$HOST" "cd '$REMOTE' 2>/dev/null; exec \$SHELL -l"
  ;;

pull)
  auth_note
  mkdir -p "$LOCAL/results_from_kelvin"
  warm_connection
  echo "==> pulling $HOST:$REMOTE/results/"
  if command -v rsync >/dev/null && have_remote_rsync; then
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
  echo "    to  $HOST:$REMOTE  (port $PORT${KEY:+, key $KEY})"
  echo "    excluding work/ results/ logs/ env.sh store/ .venv/ __pycache__"
  auth_note
  echo

  # A dry run must not touch the cluster at all -- not even to mkdir, and not
  # even to probe for rsync, since both need a working connection and the
  # point of --dry-run is often to check the file list while offline. It
  # therefore never prompts for anything.
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

  warm_connection

  sent=""
  if command -v rsync >/dev/null && have_remote_rsync; then
    ssh "${SSH_OPTS[@]}" "$HOST" "mkdir -p '$REMOTE'"
    # The trailing slash on the source is load-bearing: "$LOCAL/" copies the
    # CONTENTS into $REMOTE. Without it rsync creates $REMOTE/chal_4_af/.
    # macOS ships openrsync, which rejects some GNU options, so a failure here
    # falls back to tar-over-ssh rather than being fatal.
    if rsync -avz "${RSYNC_EXCL[@]}" -e "ssh ${SSH_OPTS[*]}" \
         "$LOCAL/" "$HOST:$REMOTE/"; then
      sent="rsync"
    else
      echo "  (rsync failed -- falling back to tar over ssh)"
    fi
  fi

  if [[ -z "$sent" ]]; then
    # tar over ssh beats scp -r here: one stream, it honours the excludes, and
    # it will not nest a second copy at $REMOTE/chal_4_af the way scp -r does
    # when the target directory already exists. mkdir is folded into the same
    # remote command so the no-rsync path is a single round trip.
    tar -C "$LOCAL" "${TAR_EXCL[@]}" -czf - . \
      | ssh "${SSH_OPTS[@]}" "$HOST" "mkdir -p '$REMOTE' && tar -C '$REMOTE' -xzf -"
    sent="tar+ssh"
  fi
  echo
  echo "  transferred via $sent"

  ssh "${SSH_OPTS[@]}" "$HOST" \
    "chmod +x '$REMOTE'/*.sh 2>/dev/null; \
     mkdir -p '$REMOTE/work' '$REMOTE/results' '$REMOTE/logs'; true"

  cat <<EOF

done. Now, on Kelvin-2:

  bash push.sh --shell         # reuses the connection just opened, no prompt
  cd $REMOTE

  bash install_af.sh --build   # ONCE, on the login node. ~40-70 min, ~12 GB.
  bash run.sh --probe          # 20 min V100 job: is AF2 actually on the GPU?
  bash run.sh                  # the real design run

The shared connection stays open for $PERSIST, so the commands above and a
later 'bash push.sh --pull' will not ask again. Close it early with:

  bash push.sh --stop

EOF
  ;;
esac

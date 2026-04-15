#!/usr/bin/env bash
# Kill a training run by PID on a remote host.
# Usage: bash scripts/kill.sh <host> <pid>
set -e
HOST=$1
PID=$2
if [[ -z "$HOST" || -z "$PID" ]]; then
    echo "Usage: $0 <host> <pid>" >&2
    exit 1
fi
ssh "$HOST" "kill $PID && echo 'Killed PID $PID on $HOST'"

#!/usr/bin/env bash
# Kill a training run by PID on a remote host, taking its worker processes with it.
#
# A plain `kill <pid>` only signals the parent. DataLoader workers are orphaned and
# keep their CUDA contexts alive, so VRAM stays allocated and the next launch fails
# the GPU-free check (or OOMs). Signalling the whole process GROUP avoids that.
#
# Usage: bash scripts/kill.sh <host> <pid> [--force]
#   --force : follow up with SIGKILL if the group is still alive after the grace period
set -e
HOST=$1
PID=$2
FORCE=${3:-}
GRACE=8   # seconds to allow for a clean shutdown before checking/escalating

if [[ -z "$HOST" || -z "$PID" ]]; then
    echo "Usage: $0 <host> <pid> [--force]" >&2
    exit 1
fi

ssh "$HOST" bash -s -- "$PID" "$GRACE" "$FORCE" << 'ENDSSH'
PID=$1; GRACE=$2; FORCE=$3

if ! kill -0 "$PID" 2>/dev/null; then
    echo "PID $PID is not running"
else
    PGID=$(ps -o pgid= -p "$PID" | tr -d ' ')
    if [[ -n "$PGID" ]]; then
        echo "Killing process group $PGID (leader $PID)..."
        kill -TERM -- "-$PGID" 2>/dev/null || kill -TERM "$PID"
    else
        kill -TERM "$PID"
    fi
    sleep "$GRACE"
    if kill -0 "$PID" 2>/dev/null; then
        if [[ "$FORCE" == "--force" ]]; then
            echo "Still alive after ${GRACE}s; sending SIGKILL to group $PGID"
            kill -KILL -- "-$PGID" 2>/dev/null || kill -KILL "$PID"
            sleep 3
        else
            echo "WARNING: PID $PID still alive after ${GRACE}s (re-run with --force)"
        fi
    fi
fi

# Report whether the GPU actually came back, so leaked VRAM is noticed now rather
# than at the next launch.
if command -v nvidia-smi > /dev/null 2>&1; then
    echo "--- GPU after kill ---"
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
    LEFT=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader)
    if [[ -n "$LEFT" ]]; then
        echo "Processes still holding VRAM:"
        echo "$LEFT"
    else
        echo "No processes holding VRAM."
    fi
fi
ENDSSH

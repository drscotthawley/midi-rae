#!/bin/bash
# Block until the most recent midi-rae run on a remote host finishes.
# Polls status every INTERVAL seconds, printing output each time.
#
# Usage:
#   ./scripts/wait.sh <host>
#   ./scripts/wait.sh <host> <run_dir>          # wait on a specific run
#   ./scripts/wait.sh <host> "" <interval>      # custom poll interval (seconds)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

HOST="${1:?Usage: $0 <host> [run_dir] [interval_seconds]}"
RUN_DIR="${2:-}"
INTERVAL="${3:-120}"  # default: 2 minutes

while true; do
    echo "=== $(date '+%Y-%m-%d %H:%M:%S') ==="
    output=$(bash "${SCRIPT_DIR}/status.sh" "${HOST}" "${RUN_DIR}")
    echo "$output"
    if ! echo "$output" | grep -q "Status: RUNNING"; then
        break
    fi
    echo "(next check in ${INTERVAL}s...)"
    sleep "${INTERVAL}"
done

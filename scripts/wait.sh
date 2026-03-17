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
MAX_FAILURES=5        # give up only after this many consecutive SSH failures

failures=0
while true; do
    echo "=== $(date '+%Y-%m-%d %H:%M:%S') ==="
    output=$(bash "${SCRIPT_DIR}/status.sh" "${HOST}" "${RUN_DIR}" 2>&1)
    status=$?
    echo "$output"
    if [[ $status -ne 0 ]] || ! echo "$output" | grep -q "Status:"; then
        failures=$((failures + 1))
        echo "[warn] SSH/status failed (attempt ${failures}/${MAX_FAILURES}); will retry..."
        if [[ $failures -ge $MAX_FAILURES ]]; then
            echo "[error] ${MAX_FAILURES} consecutive failures — giving up."
            exit 1
        fi
    else
        failures=0
        if ! echo "$output" | grep -q "Status: RUNNING"; then
            break
        fi
    fi
    echo "(next check in ${INTERVAL}s...)"
    sleep "${INTERVAL}"
done

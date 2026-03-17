#!/bin/bash
# Check the status of the most recent midi-rae training run on a remote host.
#
# Usage:
#   ./scripts/status.sh <host>
#   ./scripts/status.sh <host> <run_dir>   # check a specific run directory

HOST="${1:?Usage: $0 <host> [run_dir]}"
RUNS_DIR="~/runs/midi-rae"
RUN_DIR="${2:-}"

ssh -o ClearAllForwardings=yes "${HOST}" bash << ENDSSH
RUN_DIR="${RUN_DIR:-\$(ls -td ${RUNS_DIR}/*/ 2>/dev/null | head -1)}"
if [ -z "\$RUN_DIR" ]; then echo "No runs found under ${RUNS_DIR}"; exit 1; fi
echo "Run: \$RUN_DIR"
echo ""

# Main process only: lowest-PID python -m midi_rae.train process
MAIN_PID=\$(ps aux | grep '[p]ython -m midi_rae.train' | awk '{print \$2}' | sort -n | head -1)
if [ -n "\$MAIN_PID" ]; then
    echo "Status: RUNNING (PID \$MAIN_PID)"
    echo "  \$(ps -p \$MAIN_PID -o cmd=)"
else
    echo "Status: NOT RUNNING"
fi
echo ""

LOG="\$RUN_DIR/run.log"
if [ ! -f "\$LOG" ]; then
    echo "(no run.log found)"
elif grep -q "FINISHED\. Best metric" "\$LOG"; then
    RESULT=\$(grep "FINISHED\. Best metric" "\$LOG" | tail -1)
    echo "Result: \$RESULT"
    echo ""
    echo "--- Last 5 lines of run.log ---"
    tail -n 5 "\$LOG"
elif [ -z "\$MAIN_PID" ]; then  # not running and no FINISHED line = crash
    echo "Result: CRASHED"
    echo ""
    echo "--- Traceback (or last 50 lines) ---"
    grep -A 999 "Traceback" "\$LOG" 2>/dev/null || tail -n 50 "\$LOG"
else
    echo "--- Last 5 lines of run.log ---"
    tail -n 5 "\$LOG"
fi
ENDSSH
echo ""
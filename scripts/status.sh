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

# Check if log is growing (most reliable running indicator)
LOG_GROWING=0
if [ -f "\$RUN_DIR/run.log" ]; then
    SIZE1=\$(wc -c < "\$RUN_DIR/run.log")
    sleep 2
    SIZE2=\$(wc -c < "\$RUN_DIR/run.log")
    [ "\$SIZE2" -gt "\$SIZE1" ] && LOG_GROWING=1
fi

# Main process only: lowest-PID midi_rae training process (module or script form)
MAIN_PID=\$(ps aux | grep -E '[p]ython -m midi_rae|[p]ython train_cfm_midi|[p]ython train_flow' | awk '{print \$2}' | sort -n | head -1)
if [ "\$LOG_GROWING" -eq 1 ]; then
    echo "Status: RUNNING (log growing$([ -n "\$MAIN_PID" ] && echo ", PID \$MAIN_PID"))"
    [ -n "\$MAIN_PID" ] && echo "  \$(ps -p \$MAIN_PID -o cmd=)"
elif [ -n "\$MAIN_PID" ]; then
    echo "Status: RUNNING (PID \$MAIN_PID, log not growing)"
    echo "  \$(ps -p \$MAIN_PID -o cmd=)"
else
    echo "Status: NOT RUNNING"
fi
echo ""

LOG="\$RUN_DIR/run.log"

# Progress line for step-based trainers (train_cfm_midi). They print "step N loss X"
# appended to the tqdm bar, so once \\r becomes \\n the step line matches the
# it/s filter below and is thrown away with the progress-bar noise -- which left the
# tail showing nothing but stale header lines. Surface it explicitly instead.
if [ -f "\$LOG" ]; then
    STEP=\$(tr '\r' '\n' < "\$LOG" | grep -oE 'step [0-9]+ loss [0-9.]+' | tail -1)
    RATE=\$(tr '\r' '\n' < "\$LOG" | grep -oE '[0-9.]+it/s' | tail -1)
    [ -n "\$STEP" ] && echo "Progress: \$STEP\$([ -n "\$RATE" ] && echo "  (\$RATE)")" && echo ""
fi

if [ ! -f "\$LOG" ]; then
    echo "(no run.log found)"
elif grep -q "FINISHED" "\$LOG"; then
    RESULT=\$(grep "FINISHED" "\$LOG" | tail -1)
    echo "Result: \$RESULT"
    echo ""
    echo "--- Last 5 lines of run.log ---"
    tr '\r' '\n' < "\$LOG" | grep -vE '[0-9]+it/s|%\|' | grep -v '^\s*$' | tail -10
elif [ -z "\$MAIN_PID" ]; then  # not running and no FINISHED line = crash
    echo "Result: CRASHED"
    echo ""
    echo "--- Traceback (or last 50 lines) ---"
    grep -A 999 "Traceback" "\$LOG" 2>/dev/null || tr '\r' '\n' < "\$LOG" | grep -vE '[0-9]+it/s|%\|' | tail -50
else
    echo "--- Last 5 lines of run.log ---"
    tr '\r' '\n' < "\$LOG" | grep -vE '[0-9]+it/s|%\|' | grep -v '^\s*$' | tail -5
fi
ENDSSH
echo ""

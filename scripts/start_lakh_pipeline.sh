#!/usr/bin/env bash
# Remote-side launcher: backgrounds lakh_pipeline.sh with nohup and exits
# immediately, so the ssh session that invokes this doesn't hang waiting on
# open file descriptors. Run via: ssh <host> "bash ~/lakh_scripts/start_lakh_pipeline.sh"
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
nohup bash lakh_pipeline.sh > "$HOME/lakh_pipeline.log" 2>&1 < /dev/null &
disown
echo "Launched lakh_pipeline.sh in background, PID $!. Logging to $HOME/lakh_pipeline.log"

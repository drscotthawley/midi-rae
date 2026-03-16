#!/bin/bash
# Check if the GPU is free for compute on the local machine.
# Uses nvidia-smi --query-compute-apps which reports only CUDA compute
# contexts (not display/compositor processes), so no false positives.
#
# Exits 0 (free) or 1 (busy).
# If busy, prints the PID, memory used (MiB), and process name.
#
# Usage:
#   ./scripts/is_gpu_free.sh        # check GPU 0
#   ./scripts/is_gpu_free.sh 1      # check GPU 1

GPU_IDX="${1:-0}"

if ! command -v nvidia-smi &>/dev/null; then
    echo "GPU ${GPU_IDX}: NO_NVIDIA_SMI"
    exit 2
fi

apps=$(nvidia-smi -i ${GPU_IDX} --query-compute-apps=pid,used_gpu_memory,name --format=csv,noheader 2>/dev/null)

if [ -z "$apps" ]; then
    echo "GPU ${GPU_IDX}: FREE"
    exit 0
else
    echo "GPU ${GPU_IDX}: BUSY"
    while IFS=',' read -r pid mem name; do
        pid="${pid// /}"
        cmdline=$(ps -p "$pid" -o cmd= 2>/dev/null || echo "(process gone)")
        echo "  PID $pid |$mem | $cmdline"
    done <<< "$apps"
    exit 1
fi

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

BUSY_THRESHOLD_MIB="${2:-2000}"  # GPU is "busy" only if total usage exceeds this

if [ -z "$apps" ]; then
    echo "GPU ${GPU_IDX}: FREE"
    exit 0
fi

total_mib=0
while IFS=',' read -r pid mem name; do
    mib=$(echo "$mem" | tr -dc '0-9')
    total_mib=$((total_mib + mib))
done <<< "$apps"

if [ "$total_mib" -lt "$BUSY_THRESHOLD_MIB" ]; then
    echo "GPU ${GPU_IDX}: FREE (${total_mib} MiB in use, below ${BUSY_THRESHOLD_MIB} MiB threshold)"
    exit 0
else
    echo "GPU ${GPU_IDX}: BUSY (${total_mib} MiB in use)"
    while IFS=',' read -r pid mem name; do
        pid="${pid// /}"
        cmdline=$(ps -p "$pid" -o cmd= 2>/dev/null || echo "(process gone)")
        echo "  PID $pid |$mem | $cmdline"
    done <<< "$apps"
    exit 1
fi

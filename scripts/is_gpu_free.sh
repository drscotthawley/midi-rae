#!/bin/bash
# Check if the GPU is free for compute on the local machine.
# Tries nvidia-smi first; if unavailable (e.g. WSL), falls back to
# checking /dev/nvidia* via fuser/lsof; if neither works, assumes free.
#
# Exits 0 (free) or 1 (busy).
#
# Usage:
#   ./scripts/is_gpu_free.sh        # check GPU 0
#   ./scripts/is_gpu_free.sh 1      # check GPU 1

GPU_IDX="${1:-0}"
BUSY_THRESHOLD_MIB="${2:-2000}"

# --- Method 1: nvidia-smi ---
if command -v nvidia-smi &>/dev/null; then
    apps=$(nvidia-smi -i ${GPU_IDX} --query-compute-apps=pid,used_gpu_memory,name --format=csv,noheader 2>/dev/null)
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
fi

# --- Method 2: fuser/lsof on /dev/nvidia* (works on some WSL setups) ---
if ls /dev/nvidia* &>/dev/null 2>&1; then
    procs=$(fuser /dev/nvidia* 2>/dev/null | tr ' ' '\n' | grep -v '^$' || true)
    if [ -z "$procs" ]; then
        echo "GPU ${GPU_IDX}: FREE (nvidia-smi unavailable; no /dev/nvidia* users)"
        exit 0
    else
        echo "GPU ${GPU_IDX}: BUSY (nvidia-smi unavailable; /dev/nvidia* in use by PIDs: $procs)"
        exit 1
    fi
fi

# --- Fallback: assume free ---
echo "GPU ${GPU_IDX}: FREE (nvidia-smi unavailable; no GPU device files found; assuming free)"
exit 0

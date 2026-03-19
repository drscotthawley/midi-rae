#!/bin/bash
# Run eval_encoder.py on a remote host against one or more encoder checkpoints.
# Copies the eval script to the host, runs it CPU-only, prints JSON results.
#
# Usage:
#   ./scripts/eval_enc.sh <host> <run_tag> [run_tag2 ...]
#
# Examples:
#   ./scripts/eval_enc.sh lecun exp25_edckd7
#   ./scripts/eval_enc.sh lecun exp25_edckd7 exp24_Lcbvz8 exp23_deltamep_hHkNan

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST="${1:?Usage: $0 <host> <run_tag> [run_tag2 ...]}"
shift
TAGS=("$@")
if [ ${#TAGS[@]} -eq 0 ]; then
    echo "Error: at least one run_tag required"
    exit 1
fi

SSH="ssh -o ClearAllForwardings=yes"
RUNS_DIR="~/runs/midi-rae"

# Copy latest eval_encoder.py and inspect module to the host
echo "Copying eval_encoder.py and inspect.py to ${HOST}..."
scp "${SCRIPT_DIR}/eval_encoder.py" "${HOST}:~/eval_encoder.py"
scp "${SCRIPT_DIR}/../midi_rae/inspect.py" "${HOST}:~/inspect_module.py"

for TAG in "${TAGS[@]}"; do
    RUN_DIR="${RUNS_DIR}/${TAG}"
    echo ""
    echo "=== ${TAG} ==="

    # Find the best encoder checkpoint
    CKPT=$($SSH "${HOST}" "ls ${RUN_DIR}/checkpoints/SwinEncoder_*_best.pt 2>/dev/null | head -1")
    if [ -z "$CKPT" ]; then
        echo "  No best checkpoint found in ${RUN_DIR}/checkpoints/ — skipping"
        continue
    fi
    echo "  Checkpoint: ${CKPT}"

    # Find the config (prefer config_swin_razer, fallback to config_swin_lecun, then config_swin)
    CONFIG=$($SSH "${HOST}" "
        for c in config_swin_razer config_swin_lecun config_swin; do
            f=${RUN_DIR}/configs/\${c}.yaml
            if [ -f \"\$f\" ]; then echo \"\$f\"; break; fi
        done
    ")
    if [ -z "$CONFIG" ]; then
        echo "  No config found in ${RUN_DIR}/configs/ — skipping"
        continue
    fi
    echo "  Config: ${CONFIG}"

    # Inject latest inspect.py into the run's midi_rae/ (older runs may not have it)
    $SSH "${HOST}" "cp ~/inspect_module.py ${RUN_DIR}/midi_rae/inspect.py"

    # Run eval
    $SSH "${HOST}" "
        source ~/envs/midi-rae/bin/activate
        cd ${RUN_DIR}
        PYTHONPATH=. python ~/eval_encoder.py '${CONFIG}' '${CKPT}' --device cpu --data-path ~/datasets/POP909_images_basic
    "
done
echo ""
echo "Done."

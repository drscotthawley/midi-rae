#!/bin/bash
# Launch encoder or decoder training on a remote GPU machine via SSH.
# Creates a unique run directory under ~/runs/, copies source + config there,
# and launches with PYTHONPATH pointing at that snapshot.
#
# Usage:
#   ./scripts/launch.sh <host> <enc|dec> <config> <tag> [ckpt_host:ckpt_path]
#
#   host          — SSH host (as defined in ~/.ssh/config)
#   type          — "enc" or "dec"
#   config        — config name without .yaml (e.g. config_swin_razer)
#   tag           — short descriptive label (e.g. "dec1"); a 6-char random suffix is appended
#   ckpt_host:path — (optional) source of encoder checkpoint for decoder runs,
#                    e.g. lecun:~/runs/midi-rae/exp16_l9hXFF/checkpoints/SwinEncoder_exp16_l9hXFF_best.pt
#
# Example:
#   ./scripts/launch.sh lecun enc config_swin exp18
#   ./scripts/launch.sh razer dec config_swin_razer dec1 lecun:~/runs/midi-rae/exp16_l9hXFF/checkpoints/SwinEncoder_exp16_l9hXFF_best.pt

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

HOST="${1:?Usage: $0 <host> <enc|dec> <config> <tag> [ckpt_host:ckpt_path]}"
TYPE="${2:?Usage: $0 <host> <enc|dec> <config> <tag> [ckpt_host:ckpt_path]}"
CONFIG="${3:?Usage: $0 <host> <enc|dec> <config> <tag> [ckpt_host:ckpt_path]}"
TAG="${4:?Usage: $0 <host> <enc|dec> <config> <tag> [ckpt_host:ckpt_path]}"
CKPT_SRC="${5:-}"  # optional: host:remote_path

SSH="ssh -o ClearAllForwardings=yes"

if [[ "$TYPE" != "enc" && "$TYPE" != "dec" ]]; then
    echo "Error: type must be 'enc' or 'dec', got '${TYPE}'"
    exit 1
fi

# Generate unique run tag: user tag + 6-char random alphanumeric suffix
HASH=$(LC_ALL=C tr -dc 'A-Za-z0-9' < /dev/urandom | head -c 6)
RUN_TAG="${TAG}_${HASH}"
RUN_DIR="~/runs/midi-rae/${RUN_TAG}"

echo "Run tag: ${RUN_TAG}"

# Check GPU availability on the remote host
echo "Checking GPU on ${HOST}..."
if ! $SSH "${HOST}" 'bash -s' < "${SCRIPT_DIR}/is_gpu_free.sh"; then
    echo "Aborting: GPU is busy."
    exit 1
fi

# Export notebooks to .py files before copying
echo "Running nbdev-export..."
cd "${REPO_DIR}" && nbdev-export
cd - > /dev/null

# Create run directory structure on remote
echo "Creating run directory ${RUN_DIR} on ${HOST}..."
$SSH "${HOST}" "mkdir -p ${RUN_DIR}/midi_rae ${RUN_DIR}/configs ${RUN_DIR}/checkpoints"

# Copy source snapshot and all config files (parent configs needed for Hydra defaults inheritance)
echo "Copying midi_rae/*.py and configs/ to ${HOST}:${RUN_DIR}/ ..."
scp "${REPO_DIR}"/midi_rae/*.py "${HOST}:${RUN_DIR}/midi_rae/"
scp "${REPO_DIR}"/configs/*.yaml "${HOST}:${RUN_DIR}/configs/"

# Transfer encoder checkpoint if specified (for decoder runs)
CKPT_OVERRIDE=""
if [[ -n "$CKPT_SRC" ]]; then
    CKPT_HOST="${CKPT_SRC%%:*}"
    CKPT_PATH="${CKPT_SRC#*:}"
    CKPT_FILE="$(basename "${CKPT_PATH}")"
    echo "Transferring checkpoint ${CKPT_FILE} from ${CKPT_HOST} to ${HOST}..."
    scp "${CKPT_HOST}:${CKPT_PATH}" "/tmp/${CKPT_FILE}"
    scp "/tmp/${CKPT_FILE}" "${HOST}:${RUN_DIR}/checkpoints/${CKPT_FILE}"
    rm -f "/tmp/${CKPT_FILE}"
    CKPT_OVERRIDE="++encoder_ckpt=${RUN_DIR}/checkpoints/${CKPT_FILE}"
    echo "Checkpoint staged at ${RUN_DIR}/checkpoints/${CKPT_FILE}"
fi

# Write a self-contained run script to the run directory and execute it.
# Using a script file prevents mp.Manager() child processes from inheriting
# the SSH socket file descriptors, which would otherwise keep SSH open.
cat > /tmp/midi_rae_run.sh << EOF
#!/bin/bash
source ~/envs/midi-rae/bin/activate
cd ${RUN_DIR}
PYTHONPATH=${RUN_DIR} nohup python -m midi_rae.train_${TYPE} --config-name ${CONFIG} ++tag=${RUN_TAG} ${CKPT_OVERRIDE} > ${RUN_DIR}/run.log 2>&1 &
echo \$!
EOF

scp /tmp/midi_rae_run.sh "${HOST}:${RUN_DIR}/run.sh"

echo "Launching train_${TYPE} on ${HOST} (tag=${RUN_TAG}, config=${CONFIG})..."
PID=$($SSH "${HOST}" "bash ${RUN_DIR}/run.sh")
echo "Launched PID ${PID} → ${RUN_DIR}/run.log"
echo "Launch script finished."

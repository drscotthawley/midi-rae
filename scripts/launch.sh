#!/bin/bash
# Launch encoder or decoder training on a remote GPU machine via SSH.
# Creates a unique run directory under ~/runs/, copies source + config there,
# and launches with PYTHONPATH pointing at that snapshot.
#
# Usage:
#   ./scripts/launch.sh <host> <enc|dec|hmep> <config> <tag> [hydra_overrides...]
#
#   host              — SSH host (as defined in ~/.ssh/config)
#   type              — "enc", "dec", "hmep", "preencode", or "fitpca"
#   config            — config name without .yaml (e.g. config_swin_razer)
#   tag               — short descriptive label (e.g. "dec1"); a 6-char random suffix is appended
#   hydra_overrides   — (optional) any number of Hydra overrides, e.g. ++training.dec_epochs=200
#
# Example:
#   ./scripts/launch.sh lecun enc config_swin exp18
#   ./scripts/launch.sh razer dec config_swin_razer dec1 ++training.dec_epochs=200
#   ./scripts/launch.sh razer dec config_swin_razer dec1 ++encoder_ckpt=~/runs/midi-rae/exp22/.../best.pt

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

HOST="${1:?Usage: $0 <host> <enc|dec|hmep> <config> <tag> [hydra_overrides...]}"
TYPE="${2:?Usage: $0 <host> <enc|dec|hmep> <config> <tag> [hydra_overrides...]}"
CONFIG="${3:?Usage: $0 <host> <enc|dec|hmep> <config> <tag> [hydra_overrides...]}"
TAG="${4:?Usage: $0 <host> <enc|dec|hmep> <config> <tag> [hydra_overrides...]}"
shift 4
EXTRA_OVERRIDES="$*"  # all remaining args passed directly to Hydra

SSH="ssh -o ClearAllForwardings=yes"

# Hydra's grammar treats ~ as its delete operator, so expand ~ to the remote $HOME
REMOTE_HOME=$($SSH "${HOST}" "echo \$HOME" 2>/dev/null || true)
EXTRA_OVERRIDES="${EXTRA_OVERRIDES//\~/$REMOTE_HOME}"

if [[ "$TYPE" != "enc" && "$TYPE" != "dec" && "$TYPE" != "hmep" && "$TYPE" != "preencode" && "$TYPE" != "fitpca" ]]; then
    echo "Error: type must be 'enc', 'dec', 'hmep', 'preencode', or 'fitpca', got '${TYPE}'"
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

# Write a self-contained run script to the run directory and execute it.
# Using a script file prevents mp.Manager() child processes from inheriting
# the SSH socket file descriptors, which would otherwise keep SSH open.
if [[ "$TYPE" = "preencode" ]]; then
    MODULE="midi_rae.preencode"
elif [[ "$TYPE" = "fitpca" ]]; then
    MODULE="midi_rae.fit_pca"
else
    MODULE="midi_rae.train_${TYPE}"
fi

cat > /tmp/midi_rae_run.sh << EOF
#!/bin/bash
source ~/envs/midi-rae/bin/activate
cd ${RUN_DIR}
PYTHONPATH=${RUN_DIR} nohup python -m ${MODULE} --config-name ${CONFIG} ++tag=${RUN_TAG} ${EXTRA_OVERRIDES} > ${RUN_DIR}/run.log 2>&1 &
echo \$!
EOF

scp /tmp/midi_rae_run.sh "${HOST}:${RUN_DIR}/run.sh"

echo "Launching train_${TYPE} on ${HOST} (tag=${RUN_TAG}, config=${CONFIG})..."
PID=$($SSH "${HOST}" "bash ${RUN_DIR}/run.sh")
echo "Launched PID ${PID} → ${RUN_DIR}/run.log"
echo "Launch script finished."

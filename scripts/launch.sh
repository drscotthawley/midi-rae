#!/bin/bash
# Launch encoder or decoder training on a remote GPU machine via SSH.
# Creates a unique run directory under ~/runs/, copies source + config there,
# and launches with PYTHONPATH pointing at that snapshot.
#
# Usage:
#   ./scripts/launch.sh <host> <enc|dec|hmep> <config> <tag> [hydra_overrides...]
#   ./scripts/launch.sh <host> ssm - <tag> [argparse_args...]
#
#   host              — SSH host (as defined in ~/.ssh/config)
#   type              — "enc", "dec", "hmep", "preencode", "fitpca", "flow", or "ssm"
#   config            — config name without .yaml (e.g. config_swin_razer); use "-" for ssm
#   tag               — short descriptive label (e.g. "dec1"); a 6-char random suffix is appended
#   hydra_overrides   — (optional) Hydra overrides, or argparse args for ssm type
#
# Example:
#   ./scripts/launch.sh lecun enc config_swin exp18
#   ./scripts/launch.sh razer dec config_swin_razer dec1 ++training.dec_epochs=200
#   ./scripts/launch.sh razer ssm - ssm_survey --n_songs 20

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

FORCE=0
if [[ "${1}" == "--force" ]]; then FORCE=1; shift; fi

HOST="${1:?Usage: $0 [--force] <host> <enc|dec|hmep> <config> <tag> [hydra_overrides...]}"
TYPE="${2:?Usage: $0 [--force] <host> <enc|dec|hmep> <config> <tag> [hydra_overrides...]}"
CONFIG="${3:?Usage: $0 [--force] <host> <enc|dec|hmep> <config> <tag> [hydra_overrides...]}"
TAG="${4:?Usage: $0 [--force] <host> <enc|dec|hmep> <config> <tag> [hydra_overrides...]}"
shift 4
EXTRA_OVERRIDES="$*"  # all remaining args passed directly to Hydra

SSH="ssh -o ClearAllForwardings=yes"

# Hydra's grammar treats ~ as its delete operator, so expand ~ to the remote $HOME
REMOTE_HOME=$($SSH "${HOST}" "echo \$HOME" 2>/dev/null || true)
EXTRA_OVERRIDES="${EXTRA_OVERRIDES//\~/$REMOTE_HOME}"

if [[ "$TYPE" != "enc" && "$TYPE" != "dec" && "$TYPE" != "hmep" && "$TYPE" != "preencode" && "$TYPE" != "fitpca" && "$TYPE" != "flow" && "$TYPE" != "flow2" && "$TYPE" != "generate" && "$TYPE" != "ssm" && "$TYPE" != "cfm" ]]; then
    echo "Error: type must be 'enc', 'dec', 'hmep', 'preencode', 'fitpca', 'flow', 'flow2', 'generate', 'ssm', or 'cfm', got '${TYPE}'"
    exit 1
fi

# Generate unique run tag: user tag + 6-char random alphanumeric suffix
HASH=$(LC_ALL=C tr -dc 'A-Za-z0-9' < /dev/urandom | head -c 6)
RUN_TAG="${TAG}_${HASH}"
RUN_DIR="~/runs/midi-rae/${RUN_TAG}"

echo "Run tag: ${RUN_TAG}"

# Check GPU availability on the remote host (skip for CPU-only analysis types)
echo "Checking GPU on ${HOST}..."
if [[ "$TYPE" == "ssm" ]]; then
    echo "(ssm: CPU-only analysis, skipping GPU check)"
elif [[ $FORCE -eq 1 ]]; then
    echo "(--force: skipping GPU check)"
elif ! $SSH "${HOST}" 'bash -s' < "${SCRIPT_DIR}/is_gpu_free.sh"; then
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

# Copy source snapshot (and configs for Hydra-based types)
echo "Copying midi_rae/*.py to ${HOST}:${RUN_DIR}/ ..."
scp "${REPO_DIR}"/midi_rae/*.py "${HOST}:${RUN_DIR}/midi_rae/"
if [[ "$TYPE" == "cfm" ]]; then
    echo "Copying train_cfm_midi.py to ${HOST}:${RUN_DIR}/ ..."
    scp "${REPO_DIR}"/train_cfm_midi.py "${HOST}:${RUN_DIR}/"
    echo "Copying unet_mlc.py to ${HOST} site-packages ..."
    scp "${REPO_DIR}"/conditional-flow-matching/torchcfm/models/unet/unet_mlc.py \
        "${HOST}:~/envs/midi-rae/lib/python3.10/site-packages/torchcfm/models/unet/unet_mlc.py"
elif [[ "$TYPE" != "ssm" ]]; then
    echo "Copying configs/ to ${HOST}:${RUN_DIR}/ ..."
    scp "${REPO_DIR}"/configs/*.yaml "${HOST}:${RUN_DIR}/configs/"
fi

# Write a self-contained run script to the run directory and execute it.
# Using a script file prevents mp.Manager() child processes from inheriting
# the SSH socket file descriptors, which would otherwise keep SSH open.
if [[ "$TYPE" = "preencode" ]]; then
    MODULE="midi_rae.preencode"
elif [[ "$TYPE" = "fitpca" ]]; then
    MODULE="midi_rae.fit_pca"
elif [[ "$TYPE" = "generate" ]]; then
    MODULE="midi_rae.generate"
elif [[ "$TYPE" = "ssm" ]]; then
    MODULE="midi_rae.ssm_analysis"
elif [[ "$TYPE" = "flow2" ]]; then
    MODULE="midi_rae.train_flow"   # _run_flow2 dispatched via ++flow_stage=2 Hydra override
    EXTRA_OVERRIDES="++flow_stage=2 ${EXTRA_OVERRIDES}"
elif [[ "$TYPE" != "cfm" ]]; then
    MODULE="midi_rae.train_${TYPE}"
fi

if [[ "$TYPE" = "cfm" ]]; then
cat > /tmp/midi_rae_run.sh << EOF
#!/bin/bash
source ~/envs/midi-rae/bin/activate
cd ${RUN_DIR}
PYTHONPATH=${RUN_DIR} nohup python train_cfm_midi.py \
    --run_name ${RUN_TAG} \
    ${EXTRA_OVERRIDES} \
    > ${RUN_DIR}/run.log 2>&1 &
echo \$!
EOF
elif [[ "$TYPE" = "ssm" ]]; then
cat > /tmp/midi_rae_run.sh << EOF
#!/bin/bash
source ~/envs/midi-rae/bin/activate
cd ${RUN_DIR}
PYTHONPATH=${RUN_DIR} nohup python -m ${MODULE} --out_dir ${RUN_DIR}/results/ssm ${EXTRA_OVERRIDES} > ${RUN_DIR}/run.log 2>&1 &
echo \$!
EOF
else
cat > /tmp/midi_rae_run.sh << EOF
#!/bin/bash
source ~/envs/midi-rae/bin/activate
cd ${RUN_DIR}
PYTHONPATH=${RUN_DIR} nohup python -m ${MODULE} --config-name ${CONFIG} ++tag=${RUN_TAG} ${EXTRA_OVERRIDES} > ${RUN_DIR}/run.log 2>&1 &
echo \$!
EOF
fi

scp /tmp/midi_rae_run.sh "${HOST}:${RUN_DIR}/run.sh"

echo "Launching train_${TYPE} on ${HOST} (tag=${RUN_TAG}, config=${CONFIG})..."
PID=$($SSH "${HOST}" "bash ${RUN_DIR}/run.sh")
echo "Launched PID ${PID} → ${RUN_DIR}/run.log"
echo "Launch script finished."

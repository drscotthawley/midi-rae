#!/bin/bash
# Launch training or analysis jobs on a remote GPU machine via SSH.
# Creates a unique timestamped run directory under ~/runs/midi-rae/<RUN_TAG>/,
# copies the relevant source snapshot there, and launches in the background.
#
# ── Usage ────────────────────────────────────────────────────────────────────
#
#   ./scripts/launch.sh [--force] <host> <type> <config> <tag> [extra_args...]
#
#   --force   Skip GPU availability check (razer only — never use on lecun)
#   host      SSH alias from ~/.ssh/config  (lecun | razer-docker | oryxpro)
#   type      Job type — see table below
#   config    Meaning depends on type — see table below
#   tag       Short label; a 6-char random suffix is appended to form the run tag
#   extra_args  Passed verbatim to the remote process (Hydra overrides or argparse flags)
#
# ── Job types ────────────────────────────────────────────────────────────────
#
#   Type        Config arg                  What runs
#   ─────────── ─────────────────────────── ──────────────────────────────────
#   enc         config name (no .yaml)      train_enc.py  (Hydra)
#   dec         config name                 train_dec.py  (Hydra)
#   hmep        config name                 train_hmep.py (Hydra)
#   preencode   config name                 preencode.py  (Hydra)
#   fitpca      config name                 fit_pca.py    (Hydra)
#   flow        config name                 train_flow.py (Hydra, stage 1)
#   flow2       config name                 train_flow.py (Hydra, stage 2)
#   generate    config name                 generate.py   (Hydra)
#   cfm         config name                 train_cfm_midi.py (argparse)
#   ssm         - (unused, pass literal -)  ssm_analysis.py   (argparse, CPU)
#   probe       encoder run-dir name        probe_musicality.py (argparse, CPU)
#
# ── probe type details ───────────────────────────────────────────────────────
#
#   config = the encoder run directory name under ~/runs/midi-rae/ on the remote
#            host (e.g. "exp26_best").  The script auto-discovers the checkpoint
#            as the first SwinEncoder_*_best.pt in that directory's checkpoints/.
#            Data paths (POP909_images_basic, POP909_images, EMOPIA) are baked in.
#            GPU check is skipped — add "--device cuda" in extra_args for GPU.
#
#   Typical usage:
#     # Run only the fast sklearn probes (chord, density, chroma, key) — a few minutes:
#     ./scripts/launch.sh lecun probe exp26_best probe_exp26 \
#         --no_transposition --no_time --no_emopia --no_melody --no_wandb
#
#     # Full probe suite including equivariance curves and EMOPIA (~30 min on GPU):
#     ./scripts/launch.sh lecun probe exp26_best probe_exp26_full --device cuda
#
# ── Examples ─────────────────────────────────────────────────────────────────
#
#   ./scripts/launch.sh lecun enc config_swin exp18
#   ./scripts/launch.sh razer dec config_swin_razer dec1 ++training.dec_epochs=200
#   ./scripts/launch.sh razer ssm - ssm_survey --n_songs 20
#   ./scripts/launch.sh lecun probe exp26_best probe_exp26 --no_transposition --no_time --no_wandb
#   ./scripts/launch.sh --rerun u3s5_inv1_eWwtSU tsrazer-ts-docker enc config_swin_u3s5_inv1 u3s5_inv1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

FORCE=0
RERUN=""
while true; do
    if [[ "${1}" == "--force" ]]; then FORCE=1; shift;
    elif [[ "${1}" == "--rerun" ]]; then RERUN="${2}"; shift 2;
    else break; fi
done

HOST="${1:?Usage: $0 [--force] [--rerun <existing_tag>] <host> <enc|dec|hmep> <config> <tag> [hydra_overrides...]}"
TYPE="${2:?Usage: $0 [--force] [--rerun <existing_tag>] <host> <enc|dec|hmep> <config> <tag> [hydra_overrides...]}"
CONFIG="${3:?Usage: $0 [--force] [--rerun <existing_tag>] <host> <enc|dec|hmep> <config> <tag> [hydra_overrides...]}"
TAG="${4:?Usage: $0 [--force] [--rerun <existing_tag>] <host> <enc|dec|hmep> <config> <tag> [hydra_overrides...]}"
shift 4
EXTRA_OVERRIDES="$*"  # all remaining args passed directly to Hydra

SSH="ssh -o ClearAllForwardings=yes"

# Hydra's grammar treats ~ as its delete operator, so expand ~ to the remote $HOME
REMOTE_HOME=$($SSH "${HOST}" "echo \$HOME" 2>/dev/null || true)
EXTRA_OVERRIDES="${EXTRA_OVERRIDES//\~/$REMOTE_HOME}"

if [[ "$TYPE" != "enc" && "$TYPE" != "dec" && "$TYPE" != "hmep" && "$TYPE" != "preencode" && "$TYPE" != "fitpca" && "$TYPE" != "flow" && "$TYPE" != "flow2" && "$TYPE" != "generate" && "$TYPE" != "ssm" && "$TYPE" != "cfm" && "$TYPE" != "probe" ]]; then
    echo "Error: type must be 'enc', 'dec', 'hmep', 'preencode', 'fitpca', 'flow', 'flow2', 'generate', 'ssm', 'cfm', or 'probe', got '${TYPE}'"
    exit 1
fi

# Generate unique run tag, or reuse existing directory if --rerun was given
if [[ -n "${RERUN}" ]]; then
    RUN_TAG="${RERUN}"
    echo "Rerunning in existing directory: ${RUN_TAG}"
else
    HASH=$(LC_ALL=C tr -dc 'A-Za-z0-9' < /dev/urandom | head -c 6)
    RUN_TAG="${TAG}_${HASH}"
fi
RUN_DIR="~/runs/midi-rae/${RUN_TAG}"

echo "Run tag: ${RUN_TAG}"

# Check GPU availability on the remote host (skip for CPU-only analysis types)
echo "Checking GPU on ${HOST}..."
if [[ "$TYPE" == "ssm" || "$TYPE" == "probe" ]]; then
    echo "(${TYPE}: analysis run, skipping GPU check — add --device cuda to use GPU)"
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
    echo "Staging custom torchcfm unet files to ${HOST}:${RUN_DIR}/ (deployed into torchcfm at run time, version-agnostic) ..."
    scp "${REPO_DIR}"/conditional-flow-matching/torchcfm/models/unet/unet_mlc.py \
        "${REPO_DIR}"/conditional-flow-matching/torchcfm/models/unet/unet.py \
        "${REPO_DIR}"/conditional-flow-matching/torchcfm/models/unet/nn.py \
        "${REPO_DIR}"/conditional-flow-matching/torchcfm/models/unet/logger.py \
        "${HOST}:${RUN_DIR}/"
elif [[ "$TYPE" == "probe" ]]; then
    echo "Copying probe_musicality.py to ${HOST}:${RUN_DIR}/ ..."
    scp "${REPO_DIR}"/probe_musicality.py "${HOST}:${RUN_DIR}/"
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
# Deploy custom torchcfm UNet files into the installed torchcfm (version-agnostic path)
UDIR=\$(python -c 'import torchcfm, os; print(os.path.join(os.path.dirname(torchcfm.__file__), "models", "unet"))')
cp -f ${RUN_DIR}/unet_mlc.py ${RUN_DIR}/unet.py ${RUN_DIR}/nn.py ${RUN_DIR}/logger.py "\$UDIR/"
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
elif [[ "$TYPE" = "probe" ]]; then
# CONFIG = encoder run directory name (e.g. exp26_best); checkpoint auto-discovered via glob.
cat > /tmp/midi_rae_run.sh << EOF
#!/bin/bash
source ~/envs/midi-rae/bin/activate
cd ${RUN_DIR}
CKPT=\$(ls ${REMOTE_HOME}/runs/midi-rae/${CONFIG}/checkpoints/SwinEncoder_*_best.pt 2>/dev/null | head -1)
if [[ -z "\$CKPT" ]]; then echo "ERROR: no SwinEncoder checkpoint found in ${REMOTE_HOME}/runs/midi-rae/${CONFIG}/checkpoints/"; exit 1; fi
echo "Using checkpoint: \$CKPT"
PYTHONPATH=${RUN_DIR} nohup python probe_musicality.py \
    --ckpt \$CKPT \
    --config ${REMOTE_HOME}/runs/midi-rae/${CONFIG}/configs/config_swin.yaml \
    --data ${REMOTE_HOME}/datasets/POP909_images_basic \
    --pop909 ${REMOTE_HOME}/datasets/POP909_images \
    --emopia ${REMOTE_HOME}/datasets/EMOPIA \
    ${EXTRA_OVERRIDES} \
    > ${RUN_DIR}/run.log 2>&1 &
echo \$!
EOF
else
cat > /tmp/midi_rae_run.sh << EOF
#!/bin/bash
source ~/envs/midi-rae/bin/activate
cd ${RUN_DIR}
ulimit -n 65536
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=${RUN_DIR} nohup python -m ${MODULE} --config-name ${CONFIG} ++tag=${RUN_TAG} ${EXTRA_OVERRIDES} > ${RUN_DIR}/run.log 2>&1 &
echo \$!
EOF
fi

scp /tmp/midi_rae_run.sh "${HOST}:${RUN_DIR}/run.sh"

echo "Launching train_${TYPE} on ${HOST} (tag=${RUN_TAG}, config=${CONFIG})..."
PID=$($SSH "${HOST}" "bash ${RUN_DIR}/run.sh")
echo "Launched PID ${PID} → ${RUN_DIR}/run.log"
echo "Launch script finished."

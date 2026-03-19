#!/bin/bash
# Quick smoke-test for HMEP training on the local Mac (outside the devcontainer).
# Runs 2 epochs with batch_size=4 to verify the full forward/backward pass works.
# Usage: bash scripts/test_hmep_local.sh
cd "$(dirname "$0")/.."
CKPT_DIR=/workspaces/ClaudeCode-Mar12/runs/midi-rae/exp24_Lcbvz8/checkpoints

PYTHONPATH=$(pwd) python -m midi_rae.train_hmep \
    --config-name config_swin_hmep_local \
    ++tag=hmep_test \
    ++training.hmep_epochs=2 \
    ++training.batch_size=4 \
    ++training.num_workers=[0,0] \
    ++no_wandb=true \
    ++use_preencoded=false \
    ++data.path=/workspaces/ClaudeCode-Mar12/datasets/POP909_images_basic \
    ++encoder_ckpt=${CKPT_DIR}/SwinEncoder_exp24_Lcbvz8_best.pt \
    ++hmep_init_ckpt=${CKPT_DIR}/SwinMaskedEmbeddingPredictor_exp24_Lcbvz8_best.pt \
    2>&1 | tee /tmp/hmep_test.log
echo "Exit code: $?"

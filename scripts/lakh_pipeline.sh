#!/usr/bin/env bash
# End-to-end Lakh MIDI pipeline for expanding the encoder training set:
#   1. Download LMD-full (raw MIDI)
#   2. Clean it (quarantine corrupt/unparseable files)
#   3. Convert Lakh MIDI -> piano-roll images (Lakh only; POP909 images already exist)
#   4. Merge with existing POP909 images via symlinks (no copying/reconversion)
#   5. Mirror the cleaned raw Lakh MIDI dataset to a second host (e.g. tsrazer)
#
# Every stage already reports progress via wget's own progress bar or tqdm
# (with ETA, even under multiprocessing -- see clean_lakh.py / midi2img_lakh.py).
# Safe to re-run: each stage skips work that's already done.
#
# Usage: ./lakh_pipeline.sh [--mirror-host HOST]
set -euo pipefail

DATASETS_DIR="$HOME/datasets"
LAKH_DIR="$DATASETS_DIR/lmd_full"
POP909_IMAGES="$DATASETS_DIR/POP909_images_basic"
LAKH_IMAGES="$DATASETS_DIR/Lakh_images"
MERGED_DIR="$DATASETS_DIR/POP909_Lakh_images"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MIRROR_HOST="tsrazer-ts-docker"

source ~/envs/midi-rae/bin/activate

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mirror-host) MIRROR_HOST="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

echo "=========================================="
echo "STAGE 1/5: Download LMD-full"
echo "=========================================="
bash "$SCRIPT_DIR/download_lakh.sh" "$DATASETS_DIR"

echo
echo "=========================================="
echo "STAGE 2/5: Clean (quarantine corrupt files)"
echo "=========================================="
python "$SCRIPT_DIR/clean_lakh.py" "$LAKH_DIR"

echo
echo "=========================================="
echo "STAGE 3/5: Convert Lakh MIDI -> piano-roll images"
echo "=========================================="
python "$SCRIPT_DIR/midi2img_lakh.py" "$LAKH_DIR" "$LAKH_IMAGES"

echo
echo "=========================================="
echo "STAGE 4/5: Merge POP909 + Lakh images (via symlinks, no copying)"
echo "=========================================="
mkdir -p "$MERGED_DIR"
ln -sfn "$POP909_IMAGES" "$MERGED_DIR/pop909"
ln -sfn "$LAKH_IMAGES" "$MERGED_DIR/lakh"
echo "Merged dataset available at: $MERGED_DIR"
echo "  -> point config_swin.yaml's data.path at this directory to train on the union."

echo
echo "=========================================="
echo "STAGE 5/5: Mirror cleaned Lakh MIDI to $MIRROR_HOST"
echo "=========================================="
rsync -a --info=progress2 "$LAKH_DIR/" "$MIRROR_HOST:$DATASETS_DIR/lmd_full/"

echo
echo "All done."
echo "  Raw Lakh MIDI:       $LAKH_DIR  (mirrored to $MIRROR_HOST:$DATASETS_DIR/lmd_full/)"
echo "  Lakh images:         $LAKH_IMAGES"
echo "  Merged for training: $MERGED_DIR"

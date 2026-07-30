#!/usr/bin/env bash
# Downloads and extracts LMD-full (the official, deduped 176,581-file Lakh MIDI
# Dataset) into a destination directory. Idempotent: skips if already present.
#
# Usage: ./download_lakh.sh [dest_dir]   (default: $HOME/datasets)
set -euo pipefail

DEST_DIR="${1:-$HOME/datasets}"
URL="http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz"

mkdir -p "$DEST_DIR"
cd "$DEST_DIR"

if [ -d "lmd_full" ]; then
  echo "lmd_full already exists at $DEST_DIR/lmd_full -- skipping download."
  exit 0
fi

echo "Downloading LMD-full to $DEST_DIR ..."
wget -c "$URL" -O lmd_full.tar.gz

echo "Extracting..."
tar -xzf lmd_full.tar.gz

echo "Done. lmd_full/ now in $DEST_DIR"

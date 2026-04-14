#!/usr/bin/env bash
# Run nbdev-export from the repo root (where nbs/ lives).
# Usage: bash scripts/nbdev_export.sh
set -e
cd "$(dirname "$0")/.."
nbdev-export "$@"

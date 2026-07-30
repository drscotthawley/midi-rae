"""Materialise an explicit train/ and val/ split inside an image dataset directory.

Why: the previous split was computed at load time as a seeded shuffle of glob()
output. glob() returns filesystem order, so the same 909 files produced THREE
DIFFERENT partitions across three machines -- making val losses and 'best'
checkpoints incomparable between runs. An on-disk split is a property of the data
and travels with it.

Idempotent: if train/ and val/ already exist, it reports and exits.

Usage: python scripts/make_split_dirs.py <dataset_dir> [--val-fraction 0.1] [--seed 42]
"""
import argparse
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_dir")
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    root = Path(a.dataset_dir).expanduser()
    train_d, val_d = root / "train", root / "val"
    if train_d.is_dir() and val_d.is_dir():
        print(f"already split: {len(list(train_d.glob('*.png')))} train / "
              f"{len(list(val_d.glob('*.png')))} val")
        return

    files = sorted(p for p in root.glob("*.png"))     # sorted => machine-independent
    if not files:
        sys.exit(f"no .png files directly under {root}")
    rng = random.Random(a.seed)
    shuffled = files.copy()
    rng.shuffle(shuffled)
    cut = int(len(shuffled) * (1 - a.val_fraction))
    train, val = shuffled[:cut], shuffled[cut:]

    train_d.mkdir(); val_d.mkdir()
    for f in train: shutil.move(str(f), train_d / f.name)
    for f in val:   shutil.move(str(f), val_d / f.name)

    names = sorted(f.name for f in val)
    manifest = {"seed": a.seed, "val_fraction": a.val_fraction,
                "n_total": len(files), "n_train": len(train), "n_val": len(val),
                "val_md5": hashlib.md5("\n".join(names).encode()).hexdigest()[:12],
                "val": names}
    (root / "split_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"{len(train)} train / {len(val)} val  (val_md5={manifest['val_md5']})")
    print(f"wrote {root/'split_manifest.json'}")


if __name__ == "__main__":
    main()

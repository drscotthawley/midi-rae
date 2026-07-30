"""Replace demo/examples/*.png with songs from the HELD-OUT split.

The demo previously shipped six training songs, which invites the objection that
generations are recited from memory. The authoritative split lives in
POP909_split.json, dumped on the machine that trained the model -- glob() ordering
is filesystem-dependent, so recomputing it locally gives a DIFFERENT split and
must not be trusted.

Picks val songs with enough note density to be worth looking at. Loads no model.
Run: python demo/_pick_val_examples.py [source_dir] [--apply]
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else \
    Path("/workspaces/ClaudeCode-Mar12/pop909_tmp/POP909_images_basic")
N_WANT = 6

sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import flow_infer


def main():
    split = json.loads((HERE / "POP909_split.json").read_text())
    val = split["val"]
    print(f"split: {split['n_train']} train / {split['n_val']} val "
          f"(from {split['data_dir']})")

    scored = []
    for name in val:
        p = SRC / name
        if not p.exists():
            continue
        arr = flow_infer._load_binary_array(p)
        if arr.shape[1] < 512:                     # need room to scroll the strip
            continue
        # density of the best 128-wide window: a song that is mostly silence is a
        # poor demo even if it is held out
        cx = int(flow_infer.best_crop_x(p))
        win = arr[:, cx:cx + flow_infer.IMAGE_SIZE]
        scored.append((float(win.mean()), arr.shape[1], name))

    scored.sort(reverse=True)
    picks = [s for s in scored[:N_WANT]]
    print(f"\n{len(scored)} val songs usable; top {N_WANT} by windowed density:")
    for d, w, n in picks:
        print(f"  {n:20s} density={d:.4f}  width={w}")

    if "--apply" not in sys.argv:
        print("\n(dry run — pass --apply to replace demo/examples/)")
        return
    ex = HERE / "examples"
    for old in ex.glob("*.png"):
        old.unlink()
    for _, _, n in picks:
        shutil.copy(SRC / n, ex / n)
    print(f"\nreplaced demo/examples/ with {len(picks)} held-out songs")


if __name__ == "__main__":
    main()

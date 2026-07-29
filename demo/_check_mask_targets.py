"""Is 'span restores 101%' real, or just an easy denominator?

All the vs-target numbers are relative to the density the mask erased. A
full-height time span spans the whole pitch range, most of which is empty in a
piano roll, so its target density may simply be much lower than a blob sitting
on the dense mid-register. No model needed -- this is a property of the data.
Run: python demo/_check_mask_targets.py
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))

import flow_infer

EXAMPLES_DIR = HERE / "examples"


def masks():
    yy, xx = np.mgrid[0:128, 0:128]
    blob = (((xx - 78) / 26.0) ** 2 + ((yy - 64) / 30.0) ** 2) < 1.0
    span = np.zeros((128, 128), bool); span[:, 48:80] = True
    band = np.zeros((128, 128), bool); band[40:88, 32:96] = True
    return {"blob": blob, "span": span, "band": band}


def main():
    songs = sorted(p.name for p in EXAMPLES_DIR.glob("*.png"))
    shapes = masks()
    rows = {m: [] for m in shapes}
    whole = []
    for song in songs:
        cx = flow_infer.best_crop_x(EXAMPLES_DIR / song)
        img = flow_infer.image_to_binary_tensor(EXAMPLES_DIR / song, crop_x=int(cx))
        roll = img[0, 0].numpy() > 0.5
        whole.append(roll.mean())
        for m, h in shapes.items():
            rows[m].append(roll[h].mean())

    print(f"whole-window density: {np.mean(whole):.4f}\n")
    print(f"{'mask':6s} {'coverage':>9s} {'target density':>16s} {'vs whole-window':>16s}")
    for m, h in shapes.items():
        t = np.mean(rows[m])
        print(f"{m:6s} {h.mean():9.1%} {t:16.4f} {t / np.mean(whole):15.2f}x")
    print("\nA LOW target density makes 'percent of erased density restored' easy to hit:")
    print("the model need only produce its usual amount of content to score ~100%.")


if __name__ == "__main__":
    main()

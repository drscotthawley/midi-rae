"""Sanity-check the note-event matcher. Loads no model.

If note_f1(real, real) is not exactly 1.0, the matcher is broken and every number
derived from it is meaningless. Also checks a deliberately perturbed copy so we
know the metric responds sensibly rather than always returning 1.0.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))

import flow_infer
from _recon_note_f1 import notes, note_f1

EX = HERE / "examples"


def main():
    p = sorted(EX.glob("*.png"))[0]
    img = flow_infer.image_to_binary_tensor(p, crop_x=int(flow_infer.best_crop_x(p)))
    real = img[0, 0].numpy() > 0.5
    ref = notes(real)
    print(f"{p.name}: {len(ref)} note events\n")

    print(f"identical            F1 = {note_f1(ref, ref, 0)[2]:.4f}   (must be 1.0000)")

    shifted = np.roll(real, 1, axis=1)          # 1 step later in time
    print(f"shifted +1 step      F1 = {note_f1(notes(shifted), ref, 0)[2]:.4f} "
          f"(tol 0) / {note_f1(notes(shifted), ref, 1)[2]:.4f} (tol 1)")

    transposed = np.roll(real, 1, axis=0)       # 1 semitone up
    print(f"transposed +1 semi   F1 = {note_f1(notes(transposed), ref, 3)[2]:.4f} "
          f"(should be ~0: pitch must match)")

    rng = np.random.default_rng(0)
    speckled = real.copy()
    noise = rng.random(real.shape) < 0.02       # 2% spurious pixels, like the speckle
    speckled |= noise
    print(f"real + 2% speckle    F1 = {note_f1(notes(speckled), ref, 0)[2]:.4f} "
          f"({len(notes(speckled))} events vs {len(ref)})")


if __name__ == "__main__":
    main()

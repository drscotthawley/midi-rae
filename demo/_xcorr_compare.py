"""Normalised cross-correlation between input and generated roll, over 2-D shifts.

Reports the peak NCC, the shift that achieves it, and NCC at zero shift. A peak at
dy != 0 means the generated roll is transposed in pitch; dx != 0 means it is
displaced in time. Both would look "similar but wrong" and would tank any
alignment-sensitive metric while sounding close.

Also reports best-shift pixel F1 for interpretability, and the fraction of windows
whose peak sits at exactly (0,0).

Select model with MIDIRAE_MODELS / MIDIRAE_CFM_CKPT. Run on a GPU host.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))

import pcfm_infer
import flow_infer

DEV = "cuda"
STEPS, CFG, N = 20, 4.0, 30
RAD = 12                      # search +/- this many pixels in pitch and time
POP909 = Path("/home/shawley/datasets/POP909_images_basic")


def ncc_map(a, b, rad=RAD):
    """Normalised cross-correlation over circular shifts of b, within +/-rad."""
    a = a.astype(np.float32); b = b.astype(np.float32)
    a = a - a.mean(); b = b - b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return None
    out = np.zeros((2 * rad + 1, 2 * rad + 1), np.float32)
    for i, dy in enumerate(range(-rad, rad + 1)):
        rolled_y = np.roll(b, dy, axis=0)
        for j, dx in enumerate(range(-rad, rad + 1)):
            out[i, j] = float((a * np.roll(rolled_y, dx, axis=1)).sum() / (na * nb))
    return out


def f1(pred, real):
    tp = float((pred & real).sum()); fp = float((pred & ~real).sum()); fn = float((~pred & real).sum())
    p = tp / max(tp + fp, 1e-9); r = tp / max(tp + fn, 1e-9)
    return 2 * p * r / max(p + r, 1e-9)


def main():
    demo = pcfm_infer.get_demo(); demo.set_device(DEV)
    print(f"MODEL_SET={pcfm_infer.MODEL_SET}  flow={pcfm_infer.CFM_CKPT.name}\n")

    peaks, zeros, shifts, bf1, zf1 = [], [], [], [], []
    for p in sorted(POP909.glob("*.png")):
        img = flow_infer.image_to_binary_tensor(p, crop_x=int(flow_infer.best_crop_x(p)))
        if float((img > 0.5).float().mean()) < 0.01:
            continue
        real = img[0, 0].numpy() > 0.5
        gen = demo.generate(img, n_steps=STEPS, seed=0, cfg_strength=CFG, device=DEV) > 0.5
        m = ncc_map(real, gen)
        if m is None:
            continue
        k = np.unravel_index(np.argmax(m), m.shape)
        dy, dx = k[0] - RAD, k[1] - RAD
        peaks.append(float(m[k])); zeros.append(float(m[RAD, RAD])); shifts.append((dy, dx))
        bf1.append(f1(np.roll(np.roll(gen, dy, axis=0), dx, axis=1), real))
        zf1.append(f1(gen, real))
        if len(peaks) >= N:
            break

    sh = np.array(shifts)
    at_zero = float(np.mean((sh[:, 0] == 0) & (sh[:, 1] == 0)))
    print(f"{len(peaks)} windows, search radius +/-{RAD}\n")
    print(f"{'NCC at zero shift':26s} {np.mean(zeros):.3f}")
    print(f"{'NCC at peak':26s} {np.mean(peaks):.3f}")
    print(f"{'pixel F1 at zero shift':26s} {np.mean(zf1):.3f}")
    print(f"{'pixel F1 at peak shift':26s} {np.mean(bf1):.3f}")
    print(f"{'peak exactly at (0,0)':26s} {at_zero:.0%} of windows")
    print(f"\nmedian shift (dy pitch, dx time): ({np.median(sh[:,0]):.0f}, {np.median(sh[:,1]):.0f})")
    print(f"mean |dy| = {np.abs(sh[:,0]).mean():.1f} semitones, "
          f"mean |dx| = {np.abs(sh[:,1]).mean():.1f} steps")
    uy, cy = np.unique(sh[:, 0], return_counts=True)
    print("dy histogram:", {int(a): int(b) for a, b in zip(uy, cy)})


if __name__ == "__main__":
    main()

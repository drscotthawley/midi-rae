"""Why does pixel F1 say 0.27 when the outputs look similar?

Piano-roll notes are thin lines on a sparse canvas: a one-pixel shift in pitch or
time looks identical to the eye but scores near-zero overlap. Compare exact F1
against F1 with a tolerance (match if a note pixel exists within +/-k pixels), and
against the best F1 over small rigid shifts of the whole roll.

  exact       - strict pixel agreement
  tol +/-1,2  - dilate the target before matching (local slop allowed)
  best-shift  - max F1 over global shifts, catching systematic misalignment

Also dumps a real-vs-generated panel so we can look.
Run on a GPU host: ~/envs/midi-rae/bin/python _recon_tolerance.py
"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))

import pcfm_infer
import flow_infer

DEV = "cuda" if torch.cuda.is_available() else "cpu"
STEPS, CFG, N = 20, 8.0, 20
POP909 = Path("/home/shawley/datasets/POP909_images_basic")


def f1(pred, real):
    tp = float((pred & real).sum()); fp = float((pred & ~real).sum()); fn = float((~pred & real).sum())
    p = tp / max(tp + fp, 1e-9); r = tp / max(tp + fn, 1e-9)
    return 2 * p * r / max(p + r, 1e-9)


def dilate(m, k):
    """Binary dilation by k pixels in both axes (square structuring element)."""
    out = m.copy()
    for dy in range(-k, k + 1):
        for dx in range(-k, k + 1):
            out |= np.roll(np.roll(m, dy, axis=0), dx, axis=1)
    return out


def tol_f1(pred, real, k):
    """Symmetric tolerant F1: predictions count if near a real note and vice versa."""
    tp_p = float((pred & dilate(real, k)).sum())
    tp_r = float((real & dilate(pred, k)).sum())
    p = tp_p / max(float(pred.sum()), 1e-9)
    r = tp_r / max(float(real.sum()), 1e-9)
    return 2 * p * r / max(p + r, 1e-9)


def best_shift_f1(pred, real, rad=3):
    best = 0.0
    for dy in range(-rad, rad + 1):
        for dx in range(-rad, rad + 1):
            best = max(best, f1(np.roll(np.roll(pred, dy, axis=0), dx, axis=1), real))
    return best


def main():
    demo = pcfm_infer.get_demo(); demo.set_device(DEV)
    pngs = sorted(POP909.glob("*.png"))
    rows, panels = [], []
    for p in pngs:
        img = flow_infer.image_to_binary_tensor(p, crop_x=int(flow_infer.best_crop_x(p)))
        if float((img > 0.5).float().mean()) < 0.01:
            continue
        real = img[0, 0].numpy() > 0.5
        gen = demo.generate(img, n_steps=STEPS, seed=0, cfg_strength=CFG, device=DEV) > 0.5
        rows.append((f1(gen, real), tol_f1(gen, real, 1), tol_f1(gen, real, 2),
                     best_shift_f1(gen, real)))
        if len(panels) < 4:
            panels.append((real, gen))
        if len(rows) >= N:
            break

    a = np.array(rows)
    print(f"ckpt={pcfm_infer.CFM_CKPT.name}  {len(rows)} windows  steps={STEPS} cfg={CFG}\n")
    for name, col in zip(["exact F1", "tol +/-1", "tol +/-2", "best-shift F1"], a.T):
        print(f"{name:16s} {col.mean():.3f} +/- {col.std():.3f}")

    # panel: real (blue) above generated (green), 4 windows
    tiles = []
    for real, gen in panels:
        t = np.zeros((128 * 2 + 4, 128, 3), np.uint8)
        t[:128][real] = (120, 180, 255)
        t[132:][gen] = (80, 255, 120)
        tiles.append(t)
    strip = tiles[0]
    for t in tiles[1:]:
        strip = np.concatenate([strip, np.full((strip.shape[0], 4, 3), 60, np.uint8), t], axis=1)
    Image.fromarray(strip).resize((strip.shape[1] * 3, strip.shape[0] * 3),
                                  Image.NEAREST).save(HERE / "_recon_panel.png")
    print("\nsaved _recon_panel.png (top row real, bottom row generated)")


if __name__ == "__main__":
    main()

"""Sweep token-mask dilation x CFG strength for XMEP inpainting, on GPU.

Two metrics, both measured INSIDE the painted hole and compared to what the
brush erased:
  density  - how much content came back
  run-len  - mean horizontal run of note pixels. Real MIDI notes are horizontal
             bars, so long runs = notes and short runs = speckle/static. This is
             what separates 'filled with music' from 'filled with noise'.
Run on hsrazer: ~/envs/midi-rae/bin/python _sweep_inpaint.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))

import pcfm_infer
import guided_sample as gs
import flow_infer

EXAMPLES_DIR = HERE / "examples"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_STEPS, SEEDS = 20, (0, 1)
DILATES, CFGS = (0, 1, 2), (1.0, 2.0, 4.0, 8.0)


def blob_mask(h=128, w=128):
    yy, xx = np.mgrid[0:h, 0:w]
    return (((xx - 78) / 26.0) ** 2 + ((yy - 64) / 30.0) ** 2) < 1.0


def mean_run_len(binary2d):
    """Mean horizontal run length of True pixels (notes are horizontal bars)."""
    m = np.pad(binary2d.astype(np.int8), ((0, 0), (1, 1)))
    d = np.diff(m, axis=1)
    starts = np.argwhere(d == 1)
    ends = np.argwhere(d == -1)
    if len(starts) == 0:
        return 0.0
    return float((ends[:, 1] - starts[:, 1]).mean())


def main():
    ex = sorted(p.name for p in EXAMPLES_DIR.glob("*.png"))[0]
    cx = flow_infer.best_crop_x(EXAMPLES_DIR / ex)
    img = flow_infer.image_to_binary_tensor(EXAMPLES_DIR / ex, crop_x=int(cx))
    hole = blob_mask()
    orig = img[0, 0].numpy()
    demo = pcfm_infer.get_demo(); demo.set_device(DEV)

    img_holed = img.clone(); img_holed[..., torch.from_numpy(hole)] = 0.0
    x_known = img_holed * 2 - 1
    mask_t = torch.from_numpy((~hole).astype("float32")).view(1, 1, 128, 128)
    grad = gs.make_inpaint_grad(x_known, mask_t)

    t_dens = float((orig[hole] > 0.5).mean())
    t_run = mean_run_len((orig > 0.5) & hole)
    print(f"{ex} dev={DEV} ckpt={pcfm_infer.CFM_CKPT.name}")
    print(f"TARGET (erased content): density={t_dens:.3f}  run-len={t_run:.2f}\n")
    print(f"{'dilate':>6s} {'cfg':>5s} {'density':>9s} {'vs tgt':>7s} "
          f"{'run-len':>9s} {'vs tgt':>7s} {'masked L0..L5':>22s}")

    for dil in DILATES:
        mlc, st = demo.encode_to_mlcond_filled(img_holed, hole, dilate=dil, return_stats=True)
        frac = "/".join(f"{s[1]}" for s in st)
        for cfg in CFGS:
            ds, rs = [], []
            for s in SEEDS:
                g = gs.pnpflow_generate(demo, img_holed, grad, n_steps=N_STEPS, seed=s,
                                        cfg_strength=cfg, device=DEV, alpha=0.5,
                                        strength=1.0, num_avg=1, mlcond=mlc)
                ds.append(float((g[hole] > 0.5).mean()))
                rs.append(mean_run_len((g > 0.5) & hole))
            d, r = float(np.mean(ds)), float(np.mean(rs))
            print(f"{dil:6d} {cfg:5.1f} {d:9.3f} {d / t_dens:6.0%} "
                  f"{r:9.2f} {r / t_run:6.0%} {frac:>22s}")

    print("\nmasked L0..L5 = tokens repredicted per level (L0 has only 1 token total).")
    print("Want density and run-len both near 100%: enough content, and shaped like notes.")


if __name__ == "__main__":
    main()

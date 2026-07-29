"""Which mask shape x sampling config actually fills the hole?

My melody-band figure came out empty at CFG 0.8 / 10 steps while an earlier blob
filled fine at CFG 4.0 / 20 steps -- two variables changed at once. Separate them.

Mask shapes:
  blob         - compact, inside the occupied register (what my tests used)
  mel-full     - upper voice erased across the FULL width (no temporal context)
  mel-local    - upper voice erased over a time WINDOW (context to left and right,
                 i.e. what a brush stroke actually does)
Run on hsrazer: ~/envs/midi-rae/bin/python _grid_mask_config.py
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
SEEDS = (0, 1)
CONFIGS = [(10, 0.8), (20, 0.8), (10, 4.0), (20, 4.0), (10, 1.5), (30, 0.8)]


def shapes_for(real):
    yy, xx = np.mgrid[0:128, 0:128]
    blob = (((xx - 78) / 26.0) ** 2 + ((yy - 64) / 30.0) ** 2) < 1.0
    rows = np.where(real.any(axis=1))[0]
    top, bot = rows.min(), rows.max()
    cut = int(top + 0.35 * (bot - top))
    mel_full = np.zeros_like(real, bool); mel_full[top:cut + 1, :] = True
    mel_local = np.zeros_like(real, bool); mel_local[top:cut + 1, 40:90] = True
    return {"blob": blob, "mel-full": mel_full, "mel-local": mel_local}


def main():
    demo = pcfm_infer.get_demo(); demo.set_device(DEV)
    songs = sorted(p.name for p in EXAMPLES_DIR.glob("*.png"))[:3]
    print(f"dev={DEV} ckpt={pcfm_infer.CFM_CKPT.name}  pnpflow+XMEP, dilate=0")
    print("cell = generated hole density / erased hole density (100% = matched)\n")

    hdr = "  ".join(f"{s:>10s}" for s, _ in [("blob", 0), ("mel-full", 0), ("mel-local", 0)])
    print(f"{'steps':>5s} {'cfg':>4s}  {hdr}")

    for steps, cfg in CONFIGS:
        cells = {}
        for song in songs:
            cx = flow_infer.best_crop_x(EXAMPLES_DIR / song)
            img = flow_infer.image_to_binary_tensor(EXAMPLES_DIR / song, crop_x=int(cx))
            real = img[0, 0].numpy() > 0.5
            if not real.any():
                continue
            for name, hole in shapes_for(real).items():
                t = float(real[hole].mean())
                if t < 0.01:
                    continue
                ih = img.clone(); ih[..., torch.from_numpy(hole)] = 0.0
                xk = ih * 2 - 1
                mt = torch.from_numpy((~hole).astype("float32")).view(1, 1, 128, 128)
                mlc = demo.encode_to_mlcond_filled(ih, hole, dilate=0)
                grad = gs.make_inpaint_grad(xk, mt)
                for s in SEEDS:
                    g = gs.pnpflow_generate(demo, ih, grad, n_steps=steps, seed=s,
                                            cfg_strength=cfg, device=DEV, alpha=0.5,
                                            strength=1.0, num_avg=1, mlcond=mlc)
                    cells.setdefault(name, []).append(float((g[hole] > 0.5).mean()) / t)
        row = "  ".join(f"{np.mean(cells.get(n, [0])):9.0%}"
                        for n in ("blob", "mel-full", "mel-local"))
        print(f"{steps:5d} {cfg:4.1f}   {row}")

    print("\nIf mel-local fills but mel-full doesn't, temporal context is what matters,")
    print("and a brush stroke (local) is fundamentally easier than erasing a whole voice.")


if __name__ == "__main__":
    main()

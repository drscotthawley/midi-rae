"""Definitive inpaint comparison: method x conditioning, averaged over seeds.

No gradio import -- runs on the headless GPU box. Reports, inside the painted
hole, the generated note density (target = the real density it replaced) and,
outside it, how much of the real context survived.
Run on hsrazer: ~/envs/midi-rae/bin/python _test_methods_gpu.py
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
N_STEPS, CFG, SEEDS = 20, 4.0, (0, 1, 2)


def blob_mask(h=128, w=128):
    yy, xx = np.mgrid[0:h, 0:w]
    return (((xx - 78) / 26.0) ** 2 + ((yy - 64) / 30.0) ** 2) < 1.0


def run(demo, method, img_holed, x_known, mask_t, hole, mlcond, seed):
    if method == "hard":
        g = torch.Generator().manual_seed(seed)
        x0n = torch.randn(1, 1, 128, 128, generator=g)
        return gs.guided_generate(demo, img_holed, n_steps=N_STEPS, seed=seed, cfg_strength=CFG,
                                  device=DEV, mlcond=mlcond,
                                  project=gs.make_inpaint_project(x_known, mask_t, x0n))
    if method == "soft":
        return gs.guided_generate(demo, img_holed, n_steps=N_STEPS, seed=seed, cfg_strength=CFG,
                                  device=DEV, mlcond=mlcond, init_known=x_known, init_t0=0.2,
                                  guide_fn=gs.make_soft_inpaint_guidance(x_known, mask_t,
                                                                        eta=1.0, t_min=0.2))
    return gs.pnpflow_generate(demo, img_holed, gs.make_inpaint_grad(x_known, mask_t),
                               n_steps=N_STEPS, seed=seed, cfg_strength=CFG, device=DEV,
                               alpha=0.5, strength=1.0, num_avg=1, mlcond=mlcond)


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

    target = float((orig[hole] > 0.5).mean())
    print(f"{ex} crop_x={cx} dev={DEV} ckpt={pcfm_infer.CFM_CKPT.name}")
    print(f"TARGET hole density (what the brush erased) = {target:.3f}\n")
    print(f"{'method':10s} {'cond':9s} {'hole-density':>22s} {'vs target':>10s} {'known-kept':>11s}")

    for method in ("pnpflow", "hard", "soft"):
        for fill in (False, True):
            mlc = demo.encode_to_mlcond_filled(img_holed, hole, dilate=1) if fill else None
            ds, ks = [], []
            for s in SEEDS:
                g = run(demo, method, img_holed, x_known, mask_t, hole, mlc, s)
                ds.append(float((g[hole] > 0.5).mean()))
                ks.append(float(((g[~hole] > 0.5) == (orig[~hole] > 0.5)).mean()))
            m = float(np.mean(ds))
            print(f"{method:10s} {'XMEP' if fill else 'blanked':9s} "
                  f"{str([round(d, 3) for d in ds]):>22s} {m / target:9.0%} {np.mean(ks):11.3f}")

    print("\n'vs target' = fraction of the erased note density the model put back."
          "\nknown-kept ~1.0 means the surrounding music was preserved.")


if __name__ == "__main__":
    main()

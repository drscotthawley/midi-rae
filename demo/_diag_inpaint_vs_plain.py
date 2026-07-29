"""Is inpainting worse than plain generation, and is it the sampling config?

Fixes the earlier apples-to-oranges error: EVERY run-length here is measured on
the SAME hole region, so boundary truncation affects all rows equally. 'plain'
= ordinary generation with no mask and no constraint, then cropped to the hole;
it is the ceiling the inpainted hole should reach.

Also sweeps toward the config that produces the good-looking W&B samples:
train_cfm_midi.py's generate_samples() uses n_ode_steps=100 and cfg_strength=1.0
(it is called without a CFG argument), while the demo defaults to 20 steps and
CFG 4.0.

Run on hsrazer: ~/envs/midi-rae/bin/python _diag_inpaint_vs_plain.py
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
CONFIGS = [(20, 4.0), (20, 1.0), (100, 4.0), (100, 1.0)]   # last = training-eval config


def blob_mask(h=128, w=128):
    yy, xx = np.mgrid[0:h, 0:w]
    return (((xx - 78) / 26.0) ** 2 + ((yy - 64) / 30.0) ** 2) < 1.0


def run_len(b2d):
    m = np.pad(b2d.astype(np.int8), ((0, 0), (1, 1)))
    d = np.diff(m, axis=1)
    s, e = np.argwhere(d == 1), np.argwhere(d == -1)
    return float((e[:, 1] - s[:, 1]).mean()) if len(s) else 0.0


def main():
    demo = pcfm_infer.get_demo(); demo.set_device(DEV)
    songs = sorted(p.name for p in EXAMPLES_DIR.glob("*.png"))
    hole = blob_mask()
    print(f"dev={DEV} ckpt={pcfm_infer.CFM_CKPT.name}")
    print("All run-lengths measured INSIDE the hole (equal truncation).\n")
    print(f"{'steps':>5s} {'cfg':>4s} {'variant':18s} {'density':>9s} {'run-len':>9s} "
          f"{'vs real':>8s}")

    for steps, cfg in CONFIGS:
        agg = {k: [] for k in ("real", "plain", "hard-XMEP", "pnp-XMEP", "pnp-blank")}
        for song in songs:
            cx = flow_infer.best_crop_x(EXAMPLES_DIR / song)
            img = flow_infer.image_to_binary_tensor(EXAMPLES_DIR / song, crop_x=int(cx))
            orig = img[0, 0].numpy()
            if (orig[hole] > 0.5).mean() < 0.01:
                continue
            agg["real"].append((float((orig[hole] > 0.5).mean()), run_len((orig > 0.5) & hole)))

            img_holed = img.clone(); img_holed[..., torch.from_numpy(hole)] = 0.0
            x_known = img_holed * 2 - 1
            mask_t = torch.from_numpy((~hole).astype("float32")).view(1, 1, 128, 128)
            mlc = demo.encode_to_mlcond_filled(img_holed, hole, dilate=0)
            grad = gs.make_inpaint_grad(x_known, mask_t)

            for s in SEEDS:
                # plain generation, cropped to the hole = the ceiling
                g = demo.generate(img, n_steps=steps, seed=s, cfg_strength=cfg, device=DEV)
                agg["plain"].append((float((g[hole] > 0.5).mean()), run_len((g > 0.5) & hole)))

                gh = torch.Generator().manual_seed(s)
                x0n = torch.randn(1, 1, 128, 128, generator=gh)
                gg = gs.guided_generate(demo, img_holed, n_steps=steps, seed=s, cfg_strength=cfg,
                                        device=DEV, mlcond=mlc,
                                        project=gs.make_inpaint_project(x_known, mask_t, x0n))
                agg["hard-XMEP"].append((float((gg[hole] > 0.5).mean()),
                                         run_len((gg > 0.5) & hole)))

                for tag, m in (("pnp-XMEP", mlc), ("pnp-blank", None)):
                    gp = gs.pnpflow_generate(demo, img_holed, grad, n_steps=steps, seed=s,
                                             cfg_strength=cfg, device=DEV, alpha=0.5,
                                             strength=1.0, num_avg=1, mlcond=m)
                    agg[tag].append((float((gp[hole] > 0.5).mean()), run_len((gp > 0.5) & hole)))

        real_r = np.mean([r for _, r in agg["real"]])
        for k, v in agg.items():
            d, r = np.mean([x[0] for x in v]), np.mean([x[1] for x in v])
            print(f"{steps:5d} {cfg:4.1f} {k:18s} {d:9.3f} {r:9.2f} {r / real_r:7.0%}")
        print()

    print("If 'plain' is near 100% but the inpaint rows are far below, the inpainting")
    print("path degrades quality. If 'plain' is also low, it is the sampling config.")


if __name__ == "__main__":
    main()

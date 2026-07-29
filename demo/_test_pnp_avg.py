"""Why is the hole empty specifically for PnP-Flow (the GUI default)?

Hypothesis: num_avg averages endpoint estimates over independent noise draws.
Inside the hole nothing pins the content, so the draws disagree and average
toward grey -- which then thresholds to nothing. Outside the hole the constraint
keeps them agreeing, so only the hole washes out.
Run: python demo/_test_pnp_avg.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))

import pcfm_infer          # NOT app: the GPU test box has no gradio, and the UI
import guided_sample as gs  # layer is irrelevant to what this measures
import flow_infer

EXAMPLES_DIR = HERE / "examples"

N_STEPS, SEED, CFG = 20, 0, 4.0
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def blob_mask(h=128, w=128):
    yy, xx = np.mgrid[0:h, 0:w]
    return (((xx - 78) / 26.0) ** 2 + ((yy - 64) / 30.0) ** 2) < 1.0


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

    print(f"input hole density {(orig[hole] > 0.5).mean():.3f}\n")
    print(f"{'variant':28s} {'hole-density':>13s} {'hole-mean-val':>14s} {'known-kept':>11s}")

    for fill in (False, True):
        mlc = demo.encode_to_mlcond_filled(img_holed, hole, dilate=1) if fill else None
        tag = "XMEP" if fill else "blanked"
        for navg in (1, 4):
            g = gs.pnpflow_generate(demo, img_holed, grad, n_steps=N_STEPS, seed=SEED,
                                    cfg_strength=CFG, device=DEV, alpha=0.5,
                                    strength=1.0, num_avg=navg, mlcond=mlc)
            kept = float(((g[~hole] > 0.5) == (orig[~hole] > 0.5)).mean())
            print(f"pnpflow/{tag}/avg{navg:<14d} {(g[hole] > 0.5).mean():13.3f} "
                  f"{g[hole].mean():14.3f} {kept:11.3f}")

        # soft guidance for reference, same conditioning
        g = gs.guided_generate(demo, img_holed, n_steps=N_STEPS, seed=SEED, cfg_strength=CFG,
                               device=DEV, mlcond=mlc, init_known=x_known, init_t0=0.2,
                               guide_fn=gs.make_soft_inpaint_guidance(x_known, mask_t,
                                                                      eta=1.0, t_min=0.2))
        kept = float(((g[~hole] > 0.5) == (orig[~hole] > 0.5)).mean())
        print(f"soft/{tag:<23s} {(g[hole] > 0.5).mean():13.3f} "
              f"{g[hole].mean():14.3f} {kept:11.3f}")


if __name__ == "__main__":
    main()

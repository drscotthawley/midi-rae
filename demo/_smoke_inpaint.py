"""CPU smoke test: three inpainting methods on the pixel-CFM.
Run: PYTHONPATH=. python demo/_smoke_inpaint.py

Task: mask a central time-band (columns 48:80) and inpaint it; the rest is KNOWN.
Reports known-region fidelity (should be ~0 for hard-replace; small for soft) and
the density inside the filled hole (should be musical, not blank/saturated).
"""
import sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))

import pcfm_infer, flow_infer
import guided_sample as gs


def main():
    demo = pcfm_infer.PixelCFMDemo(device="cpu").load()
    ex = sorted((HERE / "examples").glob("*.png"))
    cx = flow_infer.best_crop_x(ex[0])
    img = flow_infer.image_to_binary_tensor(ex[0], crop_x=cx)     # (1,1,128,128) in [0,1]

    x_known = (img * 2 - 1)                                       # -> model space [-1,1]
    mask = torch.ones_like(x_known)
    mask[..., :, 48:80] = 0.0                                     # 0 = HOLE (columns 48:80)
    hole = (mask < 0.5)
    known = (mask > 0.5)

    def report(name, roll):
        r = torch.from_numpy(roll).view(1, 1, 128, 128) * 2 - 1   # back to [-1,1] for fidelity
        kmae = float((r - x_known).abs()[known].mean())
        hole_dens = float((roll > 0.5).reshape(1, 1, 128, 128)[hole.numpy()].mean())
        print(f"  {name:<12} known-MAE={kmae:.4f}  hole-density={hole_dens:.3f}")

    N, SEED, CFG = 20, 0, 4.0
    print("inpaint central band cols[48:80]:")

    # (a) hard replacement (RePaint-style)
    gen = torch.Generator().manual_seed(SEED)
    x0n = torch.randn(1, 1, 128, 128, generator=gen)
    proj = gs.make_inpaint_project(x_known, mask, x0n)
    report("hard", gs.guided_generate(demo, img, n_steps=N, seed=SEED, cfg_strength=CFG, project=proj))

    # (b) soft guidance (+ t0=0.2 warm start to dodge the (1-t)/t blow-up)
    guide = gs.make_soft_inpaint_guidance(x_known, mask, eta=1.0, t_min=0.2)
    report("soft", gs.guided_generate(demo, img, n_steps=N, seed=SEED, cfg_strength=CFG,
                                      guide_fn=guide, init_known=x_known, init_t0=0.2))

    # (c) PnP-Flow
    grad = gs.make_inpaint_grad(x_known, mask)
    report("pnpflow", gs.pnpflow_generate(demo, img, grad, n_steps=N, seed=SEED, cfg_strength=CFG,
                                          alpha=0.5, strength=1.0))
    report("pnpflow-avg4", gs.pnpflow_generate(demo, img, grad, n_steps=N, seed=SEED, cfg_strength=CFG,
                                               alpha=0.5, strength=1.0, num_avg=4))

    print("\nSanity: hard known-MAE ~0; soft/pnp small; all hole-density in a musical range.")


if __name__ == "__main__":
    main()

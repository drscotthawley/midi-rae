"""Visual seam comparison of the three inpaint methods. CPU.
Run: python demo/_inpaint_compare.py   ->  writes demo/_inpaint_compare.png

Panels: original | masked | hard-replace | soft-guidance | PnP-Flow(avg4).
The hole columns [48:80] are outlined so seams at the boundary are visible.
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))

import pcfm_infer, flow_infer
import guided_sample as gs

H0, H1 = 48, 80   # hole columns


def panel(roll01, title_band=False):
    """(128,128) [0,1] -> RGB uint8; notes white, hole boundary marked red."""
    a = (np.asarray(roll01) > 0.5).astype(np.uint8) * 255
    rgb = np.stack([a, a, a], axis=-1)
    rgb[:, H0, 0] = 200; rgb[:, H0, 1:] = 0     # left hole edge (red)
    rgb[:, H1 - 1, 0] = 200; rgb[:, H1 - 1, 1:] = 0
    return rgb


def main():
    demo = pcfm_infer.PixelCFMDemo(device="cpu").load()
    ex = sorted((HERE / "examples").glob("*.png"))
    cx = flow_infer.best_crop_x(ex[0])
    img = flow_infer.image_to_binary_tensor(ex[0], crop_x=cx)      # (1,1,128,128) [0,1]
    orig = img[0, 0].numpy()

    x_known = img * 2 - 1
    mask = torch.ones_like(x_known); mask[..., :, H0:H1] = 0.0
    masked_view = orig.copy(); masked_view[:, H0:H1] = 0.0

    N, SEED, CFG = 20, 0, 4.0
    gen = torch.Generator().manual_seed(SEED)
    x0n = torch.randn(1, 1, 128, 128, generator=gen)

    hard = gs.guided_generate(demo, img, n_steps=N, seed=SEED, cfg_strength=CFG,
                              project=gs.make_inpaint_project(x_known, mask, x0n))
    soft = gs.guided_generate(demo, img, n_steps=N, seed=SEED, cfg_strength=CFG,
                              guide_fn=gs.make_soft_inpaint_guidance(x_known, mask, eta=1.0, t_min=0.2),
                              init_known=x_known, init_t0=0.2)
    pnp = gs.pnpflow_generate(demo, img, gs.make_inpaint_grad(x_known, mask),
                              n_steps=N, seed=SEED, cfg_strength=CFG, alpha=0.5, strength=1.0, num_avg=4)

    panels = [panel(orig), panel(masked_view), panel(hard), panel(soft), panel(pnp)]
    gap = np.zeros((128, 6, 3), np.uint8); gap[:] = (40, 40, 40)
    strip = panels[0]
    for p in panels[1:]:
        strip = np.concatenate([strip, gap, p], axis=1)
    out = HERE / "_inpaint_compare.png"
    Image.fromarray(strip).resize((strip.shape[1] * 3, 128 * 3), Image.NEAREST).save(out)
    print("panels: original | masked | hard | soft-guidance | pnpflow-avg4")
    print("saved", out)


if __name__ == "__main__":
    main()

"""Does the XMEP latent fill actually put notes back in the hole?

A/B: identical mask, seed and method, conditioning either (a) read off the
blanked image, or (b) with the hole's embeddings repredicted by the XMEP.
The hole density is the number that matters -- input is ~0.10 in that region.
Run: python demo/_test_xmep_inpaint.py
"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))

import app
import flow_infer

N_STEPS, SEED, CFG = 20, 0, 4.0


def blob_mask(h=128, w=128):
    yy, xx = np.mgrid[0:h, 0:w]
    return (((xx - 78) / 26.0) ** 2 + ((yy - 64) / 30.0) ** 2) < 1.0


def panel(roll, color):
    m = (np.asarray(roll) > 0.5).astype(np.uint8)
    rgb = np.zeros((*m.shape, 3), np.uint8)
    for c in range(3): rgb[:, :, c] = m * color[c]
    return rgb


def main():
    ex = sorted(p.name for p in app.EXAMPLES_DIR.glob("*.png"))[0]
    cx = flow_infer.best_crop_x(app.EXAMPLES_DIR / ex)
    img = flow_infer.image_to_binary_tensor(app.EXAMPLES_DIR / ex, crop_x=int(cx))
    hole = blob_mask()
    orig = img[0, 0].numpy()
    demo = app.get_demo(); demo.set_device("cpu")

    print(f"{ex} crop_x={cx}  hole={hole.mean()*100:.1f}% of canvas")
    print(f"input density: inside hole {(orig[hole] > 0.5).mean():.3f}   "
          f"outside {(orig[~hole] > 0.5).mean():.3f}\n")

    # what the XMEP changes in latent space, per level (per-token norms, no pooling)
    img_holed = img.clone(); img_holed[..., torch.from_numpy(hole)] = 0.0
    _, stats = demo.encode_to_mlcond_filled(img_holed, hole, dilate=1, return_stats=True)
    print(f"{'level':6s} {'masked':>10s} {'of':>6s} {'||pred-enc|| hole':>18s} {'visible':>10s}")
    for i, n_hole, n_tot, d_hole, d_vis in stats:
        print(f"L{i:<5d} {n_hole:10d} {n_tot:6d} {d_hole:18.4f} {d_vis:10.4f}")

    outs = {}
    print(f"\n{'method':10s} {'cond':12s} {'hole-density':>13s} {'known-kept':>11s}")
    for method in ("pnpflow", "hard"):
        for fill in (False, True):
            g, _, _ = app.run_inpaint(
                {"layers": [np.dstack([np.zeros((128, 128, 3), np.uint8),
                                       (hole * 255).astype(np.uint8)])]},
                img, method, N_STEPS, CFG, SEED, "cpu", fill, 1)
            g = np.asarray(g.convert("L").resize((128, 128), Image.NEAREST)) / 255.0
            outs[(method, fill)] = g
            kept = float(((g[~hole] > 0.5) == (orig[~hole] > 0.5)).mean())
            print(f"{method:10s} {'XMEP' if fill else 'blanked':12s} "
                  f"{(g[hole] > 0.5).mean():13.3f} {kept:11.3f}")

    strip = [panel(orig, (120, 180, 255)), panel(np.where(hole, 0.0, orig), (120, 180, 255))]
    strip += [panel(outs[k], (80, 255, 120))
              for k in (("pnpflow", False), ("pnpflow", True), ("hard", False), ("hard", True))]
    gap = np.full((128, 4, 3), 60, np.uint8)
    out = strip[0]
    for p in strip[1:]: out = np.concatenate([out, gap, p], axis=1)
    dest = HERE / "_xmep_inpaint.png"
    Image.fromarray(out).resize((out.shape[1] * 2, 256), Image.NEAREST).save(dest)
    print("\npanels: input | holed | pnp/blanked | pnp/XMEP | hard/blanked | hard/XMEP")
    print("saved", dest)


if __name__ == "__main__":
    main()

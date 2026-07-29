"""Diagnose the GUI inpaint path with a BLOB mask (like a real brush stroke),
not the full-height band the earlier tests used.

Reports, per method: MAE in the KNOWN region (should be ~0 -- is context kept?)
and note density inside the HOLE (should be musical, ~0.03-0.08 -- did it fill?).
Saves a panel PNG for eyeballing.
Run: python demo/_diag_inpaint_blob.py
"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))

import app
import guided_sample as gs
import pcfm_infer


def blob_mask(h=128, w=128):
    """Elliptical blob in the centre-right, roughly like the screenshot stroke."""
    yy, xx = np.mgrid[0:h, 0:w]
    return (((xx - 78) / 26.0) ** 2 + ((yy - 64) / 30.0) ** 2) < 1.0


def panel(roll, color):
    m = (np.asarray(roll) > 0.5).astype(np.uint8)
    rgb = np.zeros((*m.shape, 3), np.uint8)
    for c in range(3): rgb[:, :, c] = m * color[c]
    return rgb


def main():
    ex = sorted(p.name for p in app.EXAMPLES_DIR.glob("*.png"))[0]
    demo = app.get_demo(); demo.set_device("cpu")
    cx = app.flow_infer.best_crop_x(app.EXAMPLES_DIR / ex)
    print(f"using best_crop_x={cx}")
    img = app.flow_infer.image_to_binary_tensor(app.EXAMPLES_DIR / ex, crop_x=int(cx))
    hole = blob_mask()
    print(f"example={ex}  hole covers {hole.mean()*100:.1f}% of canvas "
          f"({int(hole.sum())} cells)")

    orig = img[0, 0].numpy()
    n_in_hole = int((orig[hole] > 0.5).sum())
    print(f"input notes inside hole: {n_in_hole}  "
          f"(density {(orig[hole] > 0.5).mean():.3f})")
    print(f"input notes in known   : density {(orig[~hole] > 0.5).mean():.3f}\n")

    img_holed = img.clone()
    img_holed[..., torch.from_numpy(hole)] = 0.0
    x_known = img_holed * 2 - 1
    mask_t = torch.from_numpy((~hole).astype("float32")).view(1, 1, 128, 128)

    # A/B the CONDITIONING source: hole-zeroed (current GUI) vs full image.
    # Everything else identical. If 'full' fills the hole and 'holed' doesn't,
    # the blank region is telling the encoder "no notes here".
    outs = {}
    for tag, cond in (("holed", img_holed), ("full", img)):
        gen_t = torch.Generator().manual_seed(0)
        x0n = torch.randn(1, 1, 128, 128, generator=gen_t)
        outs[f"hard/{tag}"] = gs.guided_generate(
            demo, cond, n_steps=20, seed=0, cfg_strength=4.0, device="cpu",
            project=gs.make_inpaint_project(x_known, mask_t, x0n))
        outs[f"soft/{tag}"] = gs.guided_generate(
            demo, cond, n_steps=20, seed=0, cfg_strength=4.0, device="cpu",
            guide_fn=gs.make_soft_inpaint_guidance(x_known, mask_t, eta=1.0, t_min=0.2),
            init_known=x_known, init_t0=0.2)
        outs[f"pnpflow/{tag}"] = gs.pnpflow_generate(
            demo, cond, gs.make_inpaint_grad(x_known, mask_t), n_steps=20, seed=0,
            cfg_strength=4.0, device="cpu", alpha=0.5, strength=1.0, num_avg=4)

    print(f"{'method/cond':16s} {'known-MAE':>10s} {'known-kept':>11s} {'hole-density':>13s}"
          f"  (input hole density {(orig[hole] > 0.5).mean():.3f})")
    for k, g in outs.items():
        known_mae = float(np.abs(g[~hole] - orig[~hole]).mean())
        kept = float(((g[~hole] > 0.5) == (orig[~hole] > 0.5)).mean())
        print(f"{k:16s} {known_mae:10.4f} {kept:11.4f} {(g[hole] > 0.5).mean():13.3f}")

    strip = [panel(orig, (120, 180, 255)),
             panel(np.where(hole, 0.0, orig), (120, 180, 255))]
    strip += [panel(outs[k], (80, 255, 120))
              for k in ("hard/holed", "pnpflow/holed", "hard/full", "pnpflow/full")]
    gap = np.full((128, 4, 3), 60, np.uint8)
    out = strip[0]
    for p in strip[1:]: out = np.concatenate([out, gap, p], axis=1)
    dest = HERE / "_diag_inpaint_blob.png"
    Image.fromarray(out).resize((out.shape[1] * 2, 256), Image.NEAREST).save(dest)
    print("\npanels: input | holed-input | hard/holed | pnpflow/holed | hard/full | pnpflow/full")
    print("saved", dest)


if __name__ == "__main__":
    main()

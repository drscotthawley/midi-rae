"""Stop arguing from summary statistics -- render the images and look.

Saves a side-by-side panel (real / plain generation / inpainted) and reports
metrics that separate 'short notes' from 'speckle', which mean run-length
cannot: median run length, and the fraction of note pixels living in runs of
>= 4 px. A few long notes plus a lot of dots gives a healthy MEAN and a
terrible median/fraction.

Run on hsrazer: ~/envs/midi-rae/bin/python _panel_compare.py
"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))

import pcfm_infer
import guided_sample as gs
import flow_infer

EXAMPLES_DIR = HERE / "examples"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
STEPS, CFG, SEED = 20, 4.0, 0


def blob_mask(h=128, w=128):
    yy, xx = np.mgrid[0:h, 0:w]
    return (((xx - 78) / 26.0) ** 2 + ((yy - 64) / 30.0) ** 2) < 1.0


def run_stats(b2d):
    """(mean, median, frac of note pixels in runs >= 4px)."""
    m = np.pad(b2d.astype(np.int8), ((0, 0), (1, 1)))
    d = np.diff(m, axis=1)
    s, e = np.argwhere(d == 1), np.argwhere(d == -1)
    if len(s) == 0:
        return 0.0, 0.0, 0.0
    L = (e[:, 1] - s[:, 1]).astype(float)
    return float(L.mean()), float(np.median(L)), float(L[L >= 4].sum() / L.sum())


def colorize(roll, hole, base=(120, 180, 255), fill=(80, 255, 120)):
    m = roll > 0.5
    rgb = np.zeros((*m.shape, 3), np.uint8)
    for c in range(3):
        rgb[:, :, c] = np.where(hole, m * fill[c], m * base[c])
    return rgb


def main():
    demo = pcfm_infer.get_demo(); demo.set_device(DEV)
    song = sorted(p.name for p in EXAMPLES_DIR.glob("*.png"))[0]
    cx = flow_infer.best_crop_x(EXAMPLES_DIR / song)
    img = flow_infer.image_to_binary_tensor(EXAMPLES_DIR / song, crop_x=int(cx))
    orig = img[0, 0].numpy()
    hole = blob_mask()

    img_holed = img.clone(); img_holed[..., torch.from_numpy(hole)] = 0.0
    x_known = img_holed * 2 - 1
    mask_t = torch.from_numpy((~hole).astype("float32")).view(1, 1, 128, 128)
    mlc = demo.encode_to_mlcond_filled(img_holed, hole, dilate=0)

    outs = {"real": orig}
    outs["plain-gen"] = demo.generate(img, n_steps=STEPS, seed=SEED, cfg_strength=CFG, device=DEV)
    outs["pnp-XMEP"] = gs.pnpflow_generate(demo, img_holed, gs.make_inpaint_grad(x_known, mask_t),
                                           n_steps=STEPS, seed=SEED, cfg_strength=CFG, device=DEV,
                                           alpha=0.5, strength=1.0, num_avg=1, mlcond=mlc)
    g = torch.Generator().manual_seed(SEED)
    x0n = torch.randn(1, 1, 128, 128, generator=g)
    outs["hard-XMEP"] = gs.guided_generate(demo, img_holed, n_steps=STEPS, seed=SEED,
                                           cfg_strength=CFG, device=DEV, mlcond=mlc,
                                           project=gs.make_inpaint_project(x_known, mask_t, x0n))

    print(f"{song}  ckpt={pcfm_infer.CFM_CKPT.name}  steps={STEPS} cfg={CFG}")
    print("Metrics INSIDE the hole. speckle -> median 1, low frac>=4px\n")
    print(f"{'variant':12s} {'density':>8s} {'mean-run':>9s} {'median-run':>11s} {'frac>=4px':>10s}")
    for k, v in outs.items():
        mean, med, frac = run_stats((v > 0.5) & hole)
        print(f"{k:12s} {float((v[hole] > 0.5).mean()):8.3f} {mean:9.2f} {med:11.1f} {frac:10.0%}")

    panels = [colorize(outs["real"], np.zeros_like(hole)),
              colorize(np.where(hole, 0.0, orig), np.zeros_like(hole)),
              colorize(outs["plain-gen"], np.ones_like(hole)),
              colorize(outs["pnp-XMEP"], hole),
              colorize(outs["hard-XMEP"], hole)]
    gap = np.full((128, 4, 3), 70, np.uint8)
    strip = panels[0]
    for p in panels[1:]:
        strip = np.concatenate([strip, gap, p], axis=1)
    dest = HERE / "_panel_compare.png"
    Image.fromarray(strip).resize((strip.shape[1] * 4, 512), Image.NEAREST).save(dest)
    print("\npanels: real | holed | plain-gen | pnp-XMEP | hard-XMEP")
    print("(green = inside hole / model-generated, blue = real or pinned)")
    print("saved", dest)


if __name__ == "__main__":
    main()

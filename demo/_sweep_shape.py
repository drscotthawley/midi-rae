"""The fill has roughly the right amount of ink but the wrong shape (speckle,
not note bars). Three candidate levers, measured by run-length:

  binarize threshold - the model hedges (lots of mid-grey); 0.5 turns hedged
                       pixels into speckle. Swept post-hoc, no regeneration.
  cfg strength       - higher CFG sharpened run-length in the single-song sweep
  flow steps         - coarse integration may under-resolve note bars

Run on hsrazer: ~/envs/midi-rae/bin/python _sweep_shape.py
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
THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)
CONFIGS = [(20, 4.0), (20, 8.0), (50, 4.0), (50, 8.0)]


def blob_mask(h=128, w=128):
    yy, xx = np.mgrid[0:h, 0:w]
    return (((xx - 78) / 26.0) ** 2 + ((yy - 64) / 30.0) ** 2) < 1.0


def mean_run_len(b2d):
    m = np.pad(b2d.astype(np.int8), ((0, 0), (1, 1)))
    d = np.diff(m, axis=1)
    s, e = np.argwhere(d == 1), np.argwhere(d == -1)
    return float((e[:, 1] - s[:, 1]).mean()) if len(s) else 0.0


def main():
    demo = pcfm_infer.get_demo(); demo.set_device(DEV)
    songs = sorted(p.name for p in EXAMPLES_DIR.glob("*.png"))
    hole = blob_mask()
    print(f"dev={DEV} ckpt={pcfm_infer.CFM_CKPT.name}  pnpflow+XMEP dilate=0, blob mask")
    print(f"{len(songs)} songs x {len(SEEDS)} seeds\n")
    print(f"{'steps':>5s} {'cfg':>4s} {'thr':>4s} {'density vs tgt':>15s} {'run-len vs tgt':>15s}")

    for steps, cfg in CONFIGS:
        # generate once per (steps,cfg), then sweep the threshold on the SAME outputs
        gens, tgts = [], []
        for song in songs:
            cx = flow_infer.best_crop_x(EXAMPLES_DIR / song)
            img = flow_infer.image_to_binary_tensor(EXAMPLES_DIR / song, crop_x=int(cx))
            orig = img[0, 0].numpy()
            t_d = float((orig[hole] > 0.5).mean())
            t_r = mean_run_len((orig > 0.5) & hole)
            if t_d < 0.01 or t_r <= 0:
                continue
            img_holed = img.clone(); img_holed[..., torch.from_numpy(hole)] = 0.0
            x_known = img_holed * 2 - 1
            mask_t = torch.from_numpy((~hole).astype("float32")).view(1, 1, 128, 128)
            mlc = demo.encode_to_mlcond_filled(img_holed, hole, dilate=0)
            grad = gs.make_inpaint_grad(x_known, mask_t)
            for s in SEEDS:
                gens.append(gs.pnpflow_generate(demo, img_holed, grad, n_steps=steps, seed=s,
                                                cfg_strength=cfg, device=DEV, alpha=0.5,
                                                strength=1.0, num_avg=1, mlcond=mlc))
                tgts.append((t_d, t_r))
        for thr in THRESHOLDS:
            ds = [float((g[hole] > thr).mean()) / t[0] for g, t in zip(gens, tgts)]
            rs = [mean_run_len((g > thr) & hole) / t[1] for g, t in zip(gens, tgts)]
            print(f"{steps:5d} {cfg:4.1f} {thr:4.1f} {np.mean(ds):10.0%} ± {np.std(ds):3.0%} "
                  f"{np.mean(rs):10.0%} ± {np.std(rs):3.0%}")
        print()

    print("Target is 100%/100%: right amount of ink AND right note shape.")
    print("If a higher threshold lifts run-len without killing density, the model is")
    print("hedging and 0.5 is simply the wrong cut for rendering.")


if __name__ == "__main__":
    main()

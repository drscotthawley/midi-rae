"""How should a pixel mask project onto the hierarchical token grids?

`any pixel erased -> token is a hole` (thresh=0) is safe but throws away tokens
that are mostly known context, which hurts most when the mask isn't aligned to
the token grid (blobs, pitch bands). Sweep the fraction required to call a token
a hole, per mask shape, across all songs.

Run on hsrazer: ~/envs/midi-rae/bin/python _sweep_thresh.py
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
N_STEPS, CFG, SEEDS = 20, 4.0, (0, 1)
THRESHES = (0.0, 0.25, 0.5, 0.75)


def masks():
    yy, xx = np.mgrid[0:128, 0:128]
    blob = (((xx - 78) / 26.0) ** 2 + ((yy - 64) / 30.0) ** 2) < 1.0
    span = np.zeros((128, 128), bool); span[:, 48:80] = True
    band = np.zeros((128, 128), bool); band[40:88, 32:96] = True
    return {"blob": blob, "span": span, "band": band}


def mean_run_len(b2d):
    m = np.pad(b2d.astype(np.int8), ((0, 0), (1, 1)))
    d = np.diff(m, axis=1)
    s, e = np.argwhere(d == 1), np.argwhere(d == -1)
    return float((e[:, 1] - s[:, 1]).mean()) if len(s) else 0.0


def main():
    demo = pcfm_infer.get_demo(); demo.set_device(DEV)
    songs = sorted(p.name for p in EXAMPLES_DIR.glob("*.png"))
    print(f"dev={DEV} ckpt={pcfm_infer.CFM_CKPT.name} pnpflow+XMEP cfg={CFG} dilate=0")
    print(f"{len(songs)} songs x {len(SEEDS)} seeds per cell\n")
    print(f"{'mask':6s} {'thresh':>7s} {'density vs tgt':>16s} {'run-len vs tgt':>16s} "
          f"{'tokens masked':>14s}")

    for mname, hole in masks().items():
        for th in THRESHES:
            ds, rs, toks = [], [], []
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
                mlc, st = demo.encode_to_mlcond_filled(img_holed, hole, dilate=0, thresh=th,
                                                       return_stats=True)
                toks.append(sum(s[1] for s in st))
                grad = gs.make_inpaint_grad(x_known, mask_t)
                for s in SEEDS:
                    g = gs.pnpflow_generate(demo, img_holed, grad, n_steps=N_STEPS, seed=s,
                                            cfg_strength=CFG, device=DEV, alpha=0.5,
                                            strength=1.0, num_avg=1, mlcond=mlc)
                    ds.append(float((g[hole] > 0.5).mean()) / t_d)
                    rs.append(mean_run_len((g > 0.5) & hole) / t_r)
            print(f"{mname:6s} {th:7.2f} {np.mean(ds):11.0%} ± {np.std(ds):3.0%} "
                  f"{np.mean(rs):11.0%} ± {np.std(rs):3.0%} {np.mean(toks):14.0f}")
        print()

    print("thresh=0 -> any erased pixel masks the token (current default).")
    print("Higher thresh keeps partially-erased tokens as context for the XMEP.")


if __name__ == "__main__":
    main()

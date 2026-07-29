"""Paper-grade ablation: does XMEP latent inpainting beat blanked conditioning?

Varies song (6), mask shape (3), and seed (3) = 54 samples per cell, so the
comparison isn't an artefact of one hand-drawn blob on one song. The ONLY
difference between the two conditioning arms is whether the hole's embeddings
are repredicted by the masked-embedding predictor or read off a blanked image;
mask, seed, sampler and checkpoint are identical.

Metrics, measured inside the hole against the content the mask erased:
  density  - fraction of erased note density restored
  run-len  - mean horizontal run of note pixels (notes are bars; static is
             speckle), as a fraction of the erased content's run length
  kept     - agreement with the real music OUTSIDE the hole (should be ~1.0)

Run on hsrazer: ~/envs/midi-rae/bin/python _ablation_paper.py
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
N_STEPS, CFG, DILATE = 20, 4.0, 0
SEEDS = (0, 1, 2)
METHODS = ("pnpflow", "hard", "soft")


def masks():
    """Three mask shapes: a brush-like blob, a full-height time span, a pitch band."""
    yy, xx = np.mgrid[0:128, 0:128]
    blob = (((xx - 78) / 26.0) ** 2 + ((yy - 64) / 30.0) ** 2) < 1.0
    span = np.zeros((128, 128), bool); span[:, 48:80] = True
    band = np.zeros((128, 128), bool); band[40:88, 32:96] = True
    return {"blob": blob, "span": span, "band": band}


def mean_run_len(binary2d):
    m = np.pad(binary2d.astype(np.int8), ((0, 0), (1, 1)))
    d = np.diff(m, axis=1)
    starts, ends = np.argwhere(d == 1), np.argwhere(d == -1)
    return float((ends[:, 1] - starts[:, 1]).mean()) if len(starts) else 0.0


def run(demo, method, img_holed, x_known, mask_t, mlcond, seed):
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
    demo = pcfm_infer.get_demo(); demo.set_device(DEV)
    songs = sorted(p.name for p in EXAMPLES_DIR.glob("*.png"))
    shapes = masks()
    print(f"dev={DEV} ckpt={pcfm_infer.CFM_CKPT.name} cfg={CFG} dilate={DILATE} "
          f"steps={N_STEPS}")
    print(f"{len(songs)} songs x {len(shapes)} masks x {len(SEEDS)} seeds "
          f"= {len(songs)*len(shapes)*len(SEEDS)} samples per cell\n")

    acc = {(m, f): {"d": [], "r": [], "k": []} for m in METHODS for f in (False, True)}
    by_shape = {}     # (mask, fill) -> densities, for the variance breakdown
    by_song = {}      # (song, fill) -> densities
    skipped = 0

    for song in songs:
        cx = flow_infer.best_crop_x(EXAMPLES_DIR / song)
        img = flow_infer.image_to_binary_tensor(EXAMPLES_DIR / song, crop_x=int(cx))
        orig = img[0, 0].numpy()
        for mname, hole in shapes.items():
            t_dens = float((orig[hole] > 0.5).mean())
            t_run = mean_run_len((orig > 0.5) & hole)
            if t_dens < 0.01 or t_run <= 0:       # nothing erased -> ratio undefined
                skipped += 1
                continue
            img_holed = img.clone(); img_holed[..., torch.from_numpy(hole)] = 0.0
            x_known = img_holed * 2 - 1
            mask_t = torch.from_numpy((~hole).astype("float32")).view(1, 1, 128, 128)
            for fill in (False, True):
                mlc = (demo.encode_to_mlcond_filled(img_holed, hole, dilate=DILATE)
                       if fill else None)
                for method in METHODS:
                    for s in SEEDS:
                        g = run(demo, method, img_holed, x_known, mask_t, mlc, s)
                        a = acc[(method, fill)]
                        dr = float((g[hole] > 0.5).mean()) / t_dens
                        a["d"].append(dr)
                        a["r"].append(mean_run_len((g > 0.5) & hole) / t_run)
                        a["k"].append(float(((g[~hole] > 0.5) == (orig[~hole] > 0.5)).mean()))
                        if method == "pnpflow":     # breakdown on the headline arm only
                            by_shape.setdefault((mname, fill), []).append(dr)
                            by_song.setdefault((song, fill), []).append(dr)

    print(f"{'method':10s} {'cond':9s} {'density vs target':>20s} {'run-len vs target':>20s} "
          f"{'kept':>7s}")
    for method in METHODS:
        for fill in (False, True):
            a = acc[(method, fill)]
            if not a["d"]:
                continue
            print(f"{method:10s} {'XMEP' if fill else 'blanked':9s} "
                  f"{np.mean(a['d']):13.0%} ± {np.std(a['d']):3.0%} "
                  f"{np.mean(a['r']):13.0%} ± {np.std(a['r']):3.0%} "
                  f"{np.mean(a['k']):7.3f}")
    # Where does the spread come from? Break the pnpflow arm down by mask shape and song.
    print(f"\n--- pnpflow density vs target, by MASK SHAPE ---")
    print(f"{'mask':8s} {'blanked':>16s} {'XMEP':>16s} {'gain':>7s}")
    for mname in masks():
        b, x = by_shape.get((mname, False)), by_shape.get((mname, True))
        if not b or not x:
            continue
        print(f"{mname:8s} {np.mean(b):9.0%} ± {np.std(b):3.0%} "
              f"{np.mean(x):9.0%} ± {np.std(x):3.0%} {np.mean(x)/max(np.mean(b),1e-6):6.1f}x")

    print(f"\n--- pnpflow density vs target, by SONG ---")
    print(f"{'song':16s} {'blanked':>16s} {'XMEP':>16s} {'gain':>7s}")
    for song in sorted(p.name for p in EXAMPLES_DIR.glob("*.png")):
        b, x = by_song.get((song, False)), by_song.get((song, True))
        if not b or not x:
            continue
        print(f"{song:16s} {np.mean(b):9.0%} ± {np.std(b):3.0%} "
              f"{np.mean(x):9.0%} ± {np.std(x):3.0%} {np.mean(x)/max(np.mean(b),1e-6):6.1f}x")

    if skipped:
        print(f"\n({skipped} song/mask combos skipped: mask covered no notes)")


if __name__ == "__main__":
    main()

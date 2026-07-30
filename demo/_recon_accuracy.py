"""How accurately does the flow reproduce a roll from its own conditioning?

Backs the paper claim "the flow follows the conditioning signal closely". For each
window: encode the REAL roll (no mask, no XMEP), generate from noise conditioned on
those embeddings, and compare the result to the original.

Reported: F1 / precision / recall on binarised note pixels, plus the density ratio.
Swept over CFG strength and step count, because adherence is expected to rise with
both. The paper's decoder F1 (0.9953-0.9955 for MRJ-48^) is the "pure decoding"
ceiling to compare against -- this path has to reconstruct from a 25-channel PCA
summary instead, so it should be lower.

CAVEAT: samples windows from POP909_images without controlling train/val
membership. For a paper number, re-run against the actual held-out split.

Run on a GPU host: ~/envs/midi-rae/bin/python _recon_accuracy.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))

import pcfm_infer
import flow_infer

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CONFIGS = [(10, 1.0), (10, 2.0), (10, 4.0), (10, 8.0),
           (20, 1.0), (20, 2.0), (20, 4.0), (20, 8.0),
           (50, 4.0), (50, 8.0)]
N_WINDOWS = 40
# _basic holds flat NNN_TOTAL.png files; POP909_images is 909 per-song subdirs
POP909 = Path("/home/shawley/datasets/POP909_images_basic")


def f1(pred, real):
    tp = float((pred & real).sum())
    fp = float((pred & ~real).sum())
    fn = float((~pred & real).sum())
    prec = tp / max(tp + fp, 1e-9)
    rec = tp / max(tp + fn, 1e-9)
    return 2 * prec * rec / max(prec + rec, 1e-9), prec, rec


def windows(n):
    """Sample windows from as many distinct songs as possible."""
    src = POP909 if POP909.exists() else (HERE / "examples")
    pngs = sorted(src.glob("*.png"))
    if not pngs:
        raise SystemExit(f"no PNGs under {src}")
    out = []
    per_song = max(1, n // len(pngs)) if len(pngs) < n else 1
    for p in pngs:
        for k in range(per_song):
            try:
                cx = flow_infer.best_crop_x(p)
                # spread crops out so repeats of one song are not the same bars
                cx = int(cx) + k * 137
                img = flow_infer.image_to_binary_tensor(p, crop_x=cx)
            except Exception:
                continue
            if float((img > 0.5).float().mean()) < 0.01:
                continue
            out.append((p.name, cx, img))
            if len(out) >= n:
                return out
    return out


def main():
    demo = pcfm_infer.get_demo(); demo.set_device(DEV)
    wins = windows(N_WINDOWS)
    print(f"dev={DEV} ckpt={pcfm_infer.CFM_CKPT.name}")
    print(f"{len(wins)} windows from {len(set(w[0] for w in wins))} songs\n")
    print(f"{'steps':>5s} {'cfg':>5s} {'F1':>16s} {'precision':>10s} {'recall':>8s} "
          f"{'density ratio':>13s}")

    for steps, cfg in CONFIGS:
        fs, ps, rs, ds = [], [], [], []
        for name, cx, img in wins:
            real = img[0, 0].numpy() > 0.5
            gen = demo.generate(img, n_steps=steps, seed=0, cfg_strength=cfg, device=DEV)
            pred = gen > 0.5
            a, b, c = f1(pred, real)
            fs.append(a); ps.append(b); rs.append(c)
            ds.append(pred.mean() / max(real.mean(), 1e-9))
        print(f"{steps:5d} {cfg:5.1f} {np.mean(fs):11.4f} ± {np.std(fs):.3f} "
              f"{np.mean(ps):10.4f} {np.mean(rs):8.4f} {np.mean(ds):13.2f}")

    print("\nCompare against the paper's decoder F1 (~0.995) as the pure-decoding")
    print("ceiling; this path reconstructs from the 25-channel PCA summary instead.")


if __name__ == "__main__":
    main()

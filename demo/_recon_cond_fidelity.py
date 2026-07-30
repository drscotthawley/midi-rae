"""Does the flow reproduce the CONDITIONING SIGNAL faithfully?

Pixel F1 against the original roll (see _recon_accuracy.py) measures the wrong
thing for this claim: the conditioning is a 25-channel PCA summary -- one token at
L0 for the whole window -- so the flow is asked for *a* roll consistent with those
statistics, not *the* roll. A different but equally valid realisation scores ~0 on
pixel F1 while satisfying the conditioning perfectly.

Round trip instead:
    real roll -> encode -> mlcond  -> generate -> re-encode -> mlcond'
and compare mlcond' against mlcond, per level.

Controls:
  self     - re-encoding the REAL roll (upper bound; should be ~1.0)
  shuffle  - mlcond' compared against a DIFFERENT window's conditioning
             (chance level; shows the metric discriminates at all)

Run on a GPU host: ~/envs/midi-rae/bin/python _recon_cond_fidelity.py
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
# Steps made no measurable difference (0.710 -> 0.715 over 10..50), so hold steps
# fixed and sweep CFG. 1.0 = pure conditional velocity, no guidance amplification --
# what the model actually learned, and what the training-time W&B samples use.
# 0.8 is the inpainting sweet spot found empirically.
CONFIGS = [(20, 0.8), (20, 1.0), (20, 2.0), (20, 4.0), (20, 8.0)]
N_WINDOWS = 40
POP909 = Path("/home/shawley/datasets/POP909_images_basic")


def cond_vec(mlcond):
    """Flatten the 6 conditioning maps into one vector (drops the mean-pitch column,
    which is a global scalar and would inflate agreement)."""
    parts = []
    for c in mlcond:
        a = c[0, :-1].detach().float().cpu().numpy()   # (n_comp, sp, sp)
        parts.append(a.reshape(-1))
    return np.concatenate(parts)


def per_level(mlcond):
    return [c[0, :-1].detach().float().cpu().numpy().reshape(-1) for c in mlcond]


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / max(na * nb, 1e-9))


def r2(pred, true):
    ss_res = float(((true - pred) ** 2).sum())
    ss_tot = float(((true - true.mean()) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-9)


def windows(n):
    pngs = sorted(POP909.glob("*.png")) or sorted((HERE / "examples").glob("*.png"))
    out = []
    for p in pngs:
        try:
            cx = int(flow_infer.best_crop_x(p))
            img = flow_infer.image_to_binary_tensor(p, crop_x=cx)
        except Exception:
            continue
        if float((img > 0.5).float().mean()) < 0.01:
            continue
        out.append(img)
        if len(out) >= n:
            break
    return out


def main():
    demo = pcfm_infer.get_demo(); demo.set_device(DEV)
    wins = windows(N_WINDOWS)
    print(f"dev={DEV} ckpt={pcfm_infer.CFM_CKPT.name}  {len(wins)} windows\n")

    conds = [demo.encode_to_mlcond(w) for w in wins]
    vecs = [cond_vec(c) for c in conds]

    # upper bound: re-encoding the real roll is not exactly idempotent (float paths),
    # so measure it rather than assuming 1.0
    self_cos = [cos(cond_vec(demo.encode_to_mlcond(w)), v) for w, v in zip(wins, vecs)]
    shuf = [cos(vecs[i], vecs[(i + 1) % len(vecs)]) for i in range(len(vecs))]
    print(f"{'control':22s} {'cosine':>8s}")
    print(f"{'self (re-encode real)':22s} {np.mean(self_cos):8.4f}")
    print(f"{'shuffle (other window)':22s} {np.mean(shuf):8.4f}   <- chance level\n")

    print(f"{'steps':>5s} {'cfg':>5s} {'cosine':>16s} {'R^2':>8s}   per-level cosine L0..L5")
    for steps, cfg in CONFIGS:
        cs, rs, lev = [], [], []
        for w, c, v in zip(wins, conds, vecs):
            gen = demo.generate(w, n_steps=steps, seed=0, cfg_strength=cfg, device=DEV)
            # BINARISE before re-encoding: real inputs reach the encoder binary (via
            # image_to_binary_tensor), so feeding a continuous roll is off-distribution
            # and tanks the comparison for reasons that have nothing to do with the flow.
            gt = torch.from_numpy((gen > 0.5).astype("float32")).view(1, 1, 128, 128)
            c2 = demo.encode_to_mlcond(gt)
            v2 = cond_vec(c2)
            cs.append(cos(v2, v)); rs.append(r2(v2, v))
            lev.append([cos(a, b) for a, b in zip(per_level(c2), per_level(c))])
        lv = np.mean(np.array(lev), axis=0)
        print(f"{steps:5d} {cfg:5.1f} {np.mean(cs):11.4f} ± {np.std(cs):.3f} {np.mean(rs):8.4f}   "
              + " ".join(f"{x:.2f}" for x in lv))

    print("\nHigh cosine with low pixel-F1 = the flow honours the conditioning while")
    print("choosing a different realisation -- which is what a generative model should do.")


if __name__ == "__main__":
    main()

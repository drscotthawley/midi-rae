"""Level-averaged conditioning fidelity, with a matching chance floor.

The flattened-vector aggregate is dominated by L5 (3072 dims vs 3 at L0), so it
understates adherence. This reports the UNWEIGHTED mean of per-level cosines, and
computes the shuffle baseline the same way so the comparison is consistent.

Run on a GPU host: ~/envs/midi-rae/bin/python _cond_fidelity_agg.py
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
CFGS = [0.8, 1.0, 4.0, 8.0]
STEPS, N = 20, 40
POP909 = Path("/home/shawley/datasets/POP909_images_basic")


def per_level(mlcond):
    return [c[0, :-1].detach().float().cpu().numpy().reshape(-1) for c in mlcond]


def cos(a, b):
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-9))


def lvl_mean(c_a, c_b):
    """Unweighted mean of per-level cosines."""
    return float(np.mean([cos(a, b) for a, b in zip(per_level(c_a), per_level(c_b))]))


def main():
    demo = pcfm_infer.get_demo(); demo.set_device(DEV)
    pngs = sorted(POP909.glob("*.png"))
    wins = []
    for p in pngs:
        img = flow_infer.image_to_binary_tensor(p, crop_x=int(flow_infer.best_crop_x(p)))
        if float((img > 0.5).float().mean()) >= 0.01:
            wins.append(img)
        if len(wins) >= N:
            break
    conds = [demo.encode_to_mlcond(w) for w in wins]
    print(f"ckpt={pcfm_infer.CFM_CKPT.name}  {len(wins)} windows, steps={STEPS}")
    print("level-averaged cosine (unweighted mean over L0..L5)\n")

    shuf = [lvl_mean(conds[i], conds[(i + 1) % len(conds)]) for i in range(len(conds))]
    print(f"{'shuffle (chance)':24s} {np.mean(shuf):.3f} ± {np.std(shuf):.3f}")
    print(f"{'self (re-encode real)':24s} 1.000\n")

    print(f"{'cfg':>5s} {'level-avg cosine':>18s}")
    for cfg in CFGS:
        vals = []
        for w, c in zip(wins, conds):
            g = demo.generate(w, n_steps=STEPS, seed=0, cfg_strength=cfg, device=DEV)
            gt = torch.from_numpy((g > 0.5).astype("float32")).view(1, 1, 128, 128)
            vals.append(lvl_mean(demo.encode_to_mlcond(gt), c))
        print(f"{cfg:5.1f} {np.mean(vals):13.3f} ± {np.std(vals):.3f}")


if __name__ == "__main__":
    main()

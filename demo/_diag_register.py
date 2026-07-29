"""Does the model spray notes into pitch registers that are silent in real data?

The panel image suggests the speckle lives ABOVE and BELOW the occupied register,
not among the notes. Quantify it: split pitch rows into those that are occupied in
the real roll and those that are empty, and compare generated density in each.

A model that had learned the pitch marginal would put ~nothing in empty rows.
This also gives a metric that tracks the visual complaint, which mean run-length
did not.

Run on hsrazer: ~/envs/midi-rae/bin/python _diag_register.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))

import pcfm_infer
import flow_infer

EXAMPLES_DIR = HERE / "examples"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = (0, 1)
STEPS, CFG = 20, 4.0
CKPTS = ["otcfm_midi_weights_step_6000.pt", "otcfm_midi_weights_step_10000.pt",
         "otcfm_midi_weights_step_12000.pt", "otcfm_midi_weights_step_14000.pt"]


def main():
    import importlib
    import os

    print(f"dev={DEV}  plain generation, no mask, steps={STEPS} cfg={CFG}")
    print("'empty rows' = pitch rows with no notes in the REAL roll.\n")
    print(f"{'checkpoint':40s} {'occupied':>9s} {'empty':>8s} {'leak':>7s}")

    for name in CKPTS:
        if not (HERE / "checkpoints" / "c55" / name).exists():
            print(f"{name:40s}  (missing)")
            continue
        os.environ["MIDIRAE_CFM_CKPT"] = name
        for m in ("pcfm_infer",):
            if m in sys.modules:
                del sys.modules[m]
        import pcfm_infer as pi
        importlib.reload(pi)
        demo = pi.PixelCFMDemo(device=DEV).load()

        occ, emp = [], []
        for song in sorted(p.name for p in EXAMPLES_DIR.glob("*.png")):
            cx = flow_infer.best_crop_x(EXAMPLES_DIR / song)
            img = flow_infer.image_to_binary_tensor(EXAMPLES_DIR / song, crop_x=int(cx))
            real = img[0, 0].numpy() > 0.5
            rows_occupied = real.any(axis=1)          # (128,) True where real has notes
            if rows_occupied.sum() in (0, 128):
                continue
            for s in SEEDS:
                g = demo.generate(img, n_steps=STEPS, seed=s, cfg_strength=CFG, device=DEV) > 0.5
                occ.append(g[rows_occupied].mean())
                emp.append(g[~rows_occupied].mean())
        o, e = float(np.mean(occ)), float(np.mean(emp))
        print(f"{name:40s} {o:9.4f} {e:8.4f} {e / max(o, 1e-9):6.0%}")

    print("\n'leak' = density in silent rows as a fraction of density in occupied rows.")
    print("0% would mean the model respects the pitch range; high means it sprays.")


if __name__ == "__main__":
    main()

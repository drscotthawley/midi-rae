"""Parity + solver check for guided_sample.py. Run: PYTHONPATH=. python demo/_parity_guided.py

1. identity guidance (guide=None, project=None) reproduces demo.generate (euler, same seed/steps).
2. euler vs rk4 agree on the ~unguided field (cfg=1 => plain conditional OT field), and separate
   under strong CFG (cfg=8) -- the empirical OT-straightening check.
"""
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))

import pcfm_infer, flow_infer
import guided_sample as gs


def main():
    demo = pcfm_infer.PixelCFMDemo(device="cpu").load()
    ex = sorted((HERE / "examples").glob("*.png"))
    cx = flow_infer.best_crop_x(ex[0])
    img = flow_infer.image_to_binary_tensor(ex[0], crop_x=cx)

    N, SEED, CFG = 20, 0, 4.0

    # --- 1. parity: black-box NeuralODE vs explicit integrator, same solver ---
    ref = demo.generate(img, n_steps=N, seed=SEED, cfg_strength=CFG, solver="euler")
    mine = gs.guided_generate(demo, img, n_steps=N, seed=SEED, cfg_strength=CFG, solver="euler")
    d = float(np.abs(ref - mine).max())
    print(f"[parity] euler  max|torchdyn - integrate| = {d:.3e}   "
          f"({'PASS' if d < 1e-4 else 'FAIL'})")

    # --- 2. convergence vs a high-step reference (near-binary output => use
    #        mean|diff| and note-flip fraction, NOT max-abs which threshold-flips) ---
    def gen(cfg, solver, n):
        return gs.guided_generate(demo, img, n_steps=n, seed=SEED, cfg_strength=cfg, solver=solver)

    def cmp(a, b):
        mae = float(np.abs(a - b).mean())
        flip = float((((a > 0.5) != (b > 0.5)).mean()))   # fraction of pixels that cross threshold
        return mae, flip

    for cfg in (1.0, 8.0):
        ref = gen(cfg, "rk4", 200)                         # ~converged solution
        print(f"[converge] cfg={cfg}")
        for solver in ("euler", "rk4"):
            for n in (10, 20, 50):
                mae, flip = cmp(gen(cfg, solver, n), ref)
                print(f"   {solver:<5} n={n:<3}  mean|Δ|={mae:.4f}  note-flip={flip*100:.2f}%")

    print("\nExpect: parity PASS. Both solvers converge to the reference; rk4 needs fewer\n"
          "steps, and the gap between euler and rk4 widens at cfg=8 (guidance curves the field).")


if __name__ == "__main__":
    main()

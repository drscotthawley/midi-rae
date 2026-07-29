"""Is the hole filling with noise because the flow is undertrained?

Generates with NO inpainting at all, at two training checkpoints, and compares
note density against real windows. Real POP909 windows sit near 0.05; a much
denser output means the flow hasn't learned sparsity yet, which is what shows up
as 'noise' inside an inpainting hole (everywhere else is pinned to real notes).

Run: python demo/_test_ckpt_trend.py            (cheap: 20 steps, few seeds)
"""
import importlib
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))

STEPS, SEEDS = 20, (0, 1, 2)
CKPTS = ["otcfm_midi_weights_step_6000.pt", "otcfm_midi_weights_step_10000.pt",
         "otcfm_midi_weights_step_12000.pt"]


def run_for(ckpt_name):
    """Reload pcfm_infer pointed at a specific flow checkpoint."""
    os.environ["MIDIRAE_CFM_CKPT"] = ckpt_name
    for m in ("pcfm_infer", "flow_infer"):
        if m in sys.modules:
            del sys.modules[m]
    import pcfm_infer, flow_infer
    importlib.reload(pcfm_infer)

    demo = pcfm_infer.PixelCFMDemo(device="cpu").load()
    ex = sorted(p.name for p in (HERE / "examples").glob("*.png"))[0]
    cx = flow_infer.best_crop_x(HERE / "examples" / ex)
    img = flow_infer.image_to_binary_tensor(HERE / "examples" / ex, crop_x=int(cx))
    real = float((img.numpy() > 0.5).mean())

    dens = []
    for s in SEEDS:
        roll = demo.generate(img, n_steps=STEPS, seed=s, cfg_strength=4.0, device="cpu")
        dens.append(float((roll > 0.5).mean()))
    return real, dens


def main():
    print(f"{'checkpoint':40s} {'real':>7s} {'generated densities':>32s} {'ratio':>7s}")
    for name in CKPTS:
        if not (HERE / "checkpoints" / "c55" / name).exists():
            print(f"{name:40s}  (missing, skipped)")
            continue
        real, dens = run_for(name)
        mean = float(np.mean(dens))
        print(f"{name:40s} {real:7.4f} {str([round(d, 3) for d in dens]):>32s} "
              f"{mean / real:6.1f}x")
    print("\nRatio ~1x = learned real sparsity. Much >1x = still undertrained;"
          "\nthat surplus is what reads as 'noise' inside an inpainting hole.")


if __name__ == "__main__":
    main()

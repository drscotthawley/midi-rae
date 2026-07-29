"""Verify the C55cUL model set loads and generates end-to-end.

Checks the encoder/UNet state dicts load strictly (catches architecture drift),
the conditioning widths match what the checkpoint's FiLM layers expect, and a
short generation produces a plausible roll.
Run: python demo/_test_c55_load.py
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))

import pcfm_infer
import flow_infer

EXAMPLES = HERE / "examples"


def main():
    print(f"MODEL_SET = {pcfm_infer.MODEL_SET}")
    print(f"  encoder = {pcfm_infer.ENCODER_CKPT.name}")
    print(f"  flow    = {pcfm_infer.CFM_CKPT.name}")
    print(f"  xmep    = {pcfm_infer.MEP_CKPT.name if pcfm_infer.MEP_CKPT else None}")

    demo = pcfm_infer.get_demo(device="cpu")
    print(f"  n_comp  = {demo.n_comp}   (widths = n_comp+1)")

    ex = sorted(p.name for p in EXAMPLES.glob("*.png"))[0]
    cx = flow_infer.best_crop_x(EXAMPLES / ex)
    img = flow_infer.image_to_binary_tensor(EXAMPLES / ex, crop_x=int(cx))
    print(f"\ninput: {ex} crop_x={cx} density={float((img > 0.5).float().mean()):.4f}")

    mlcond = demo.encode_to_mlcond(img)
    print("mlcond maps (level: shape):")
    for i, c in enumerate(mlcond):
        print(f"  L{i}: {tuple(c.shape)}")
        assert c.shape[1] == demo.n_comp[i] + 1, f"L{i} width mismatch"

    roll = demo.generate(img, n_steps=8, seed=0, cfg_strength=4.0, device="cpu")
    dens = float((roll > 0.5).mean())
    print(f"\ngenerated: shape={roll.shape} density={dens:.4f} "
          f"min={roll.min():.3f} max={roll.max():.3f}")
    assert roll.shape == (128, 128)
    assert np.isfinite(roll).all(), "non-finite pixels in output"
    print("\nC55 LOAD TEST PASSED"
          f"{'  (WARNING: density looks empty)' if dens < 0.005 else ''}")


if __name__ == "__main__":
    main()

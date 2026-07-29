"""Regression test: inpaint hooks must tolerate constraint tensors that live on a
different device/dtype than the sampler state (the MPS crash of 2026-07-28).

app.py builds x_known/mask on CPU float32; the sampler runs on mps/cuda. The
hooks in guided_sample now re-home their captured tensors via _like().

This box is CPU-only, so we reproduce the mismatch with a 'meta'-device tensor
(same cross-device RuntimeError) and with a dtype mismatch.
Run: python demo/_test_hook_devices.py
"""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))

import guided_sample as gs


def cpu_constraints():
    """x_known / mask / noise exactly as app.py builds them: CPU, float32."""
    x_known = torch.zeros(1, 1, 8, 8)
    mask = torch.ones(1, 1, 8, 8); mask[..., 2:6] = 0.0   # centre band = hole
    return x_known, mask, torch.randn(1, 1, 8, 8)


def check(name, fn, ref):
    out = fn()
    assert out.device == ref.device, f"{name}: returned {out.device}, expected {ref.device}"
    assert out.dtype == ref.dtype, f"{name}: returned {out.dtype}, expected {ref.dtype}"
    print(f"[ok] {name:14s} -> {out.device}/{str(out.dtype).replace('torch.','')}")


def run_case(label, ref):
    print(f"\n--- sampler state on {ref.device}/{str(ref.dtype).replace('torch.','')} ---")
    x_known, mask, x0n = cpu_constraints()
    v = torch.zeros_like(ref)

    check("soft-guide", lambda: gs.make_soft_inpaint_guidance(x_known, mask, t_min=0.0)(0.5, ref, v), ref)
    check("hard-project", lambda: gs.make_inpaint_project(x_known, mask, x0n)(0.5, ref), ref)
    check("pnp-grad", lambda: gs.make_inpaint_grad(x_known, mask)(ref), ref)


def test_caching_rebinds():
    """_like should return the identical object once aligned (no per-step copy)."""
    x_known, mask, _ = cpu_constraints()
    ref = torch.zeros(1, 1, 8, 8, dtype=torch.float64)
    grad = gs.make_inpaint_grad(x_known, mask)
    grad(ref)                      # first call re-homes to float64
    before = torch.zeros(1, 1, 8, 8, dtype=torch.float64)
    assert gs._like(before, before) is before, "_like must no-op when aligned"
    print("\n[ok] _like no-ops when already aligned (cached rebind)")


def main():
    # dtype mismatch: constraints float32, sampler float64
    run_case("dtype", torch.zeros(1, 1, 8, 8, dtype=torch.float64))
    # device mismatch: 'meta' raises the same cross-device error as mps/cuda
    run_case("device", torch.zeros(1, 1, 8, 8, device="meta"))
    test_caching_rebinds()
    print("\nHOOK-DEVICE TEST PASSED")


if __name__ == "__main__":
    main()

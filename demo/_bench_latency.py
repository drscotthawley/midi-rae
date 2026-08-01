"""End-to-end inference latency for the pixel-CFM demo, on CPU, MPS or CUDA.

Backs the paper's claim of timely inference on CPU-only runtimes. What a user waits
for is the whole chain, not the flow's forward passes, so this times encode -> PCA ->
sample -> render as one path and also breaks it out.

Usage (one arm per invocation; the flag is the only thing that changes):

    python demo/_bench_latency.py --device cpu  --threads 2 4 0
    python demo/_bench_latency.py --device mps                  # host, not the devcontainer
    python demo/_bench_latency.py --device cuda
    python demo/_bench_latency.py --device cpu --solver rk4 --steps 3

MPS cannot be reached from inside the devcontainer (torch+cpu, no Metal passthrough),
so that arm has to be run on the Mac host.

Reports the median of --reps timed runs after --warmup discarded ones. The first call
on any accelerator pays lazy-init and kernel-compilation costs that a user only pays
once, so including it would overstate steady-state latency several-fold.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pcfm_infer            # noqa: E402
import flow_infer            # noqa: E402  image helpers
import img2midi              # noqa: E402


def _pick_example(path=None):
    """A held-out example window as (1,1,128,128) in [0,1]."""
    if path is None:
        cands = sorted((HERE / "examples").glob("*.png"))
        if not cands:
            sys.exit("no examples found in demo/examples/")
        path = cands[0]
    arr = flow_infer._load_binary_array(path)
    x0 = flow_infer.best_crop_x(path)
    crop = arr[:, x0:x0 + pcfm_infer.IMAGE_SIZE]
    return torch.from_numpy(crop).float().unsqueeze(0).unsqueeze(0), Path(path).name


def _sync(device):
    "Accelerator work is queued asynchronously; without this we would time the enqueue."
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def time_once(demo, img, device, steps, cfg, solver, seed):
    """Return per-stage seconds. Sampling is inferred by subtraction rather than by
    reaching into generate(), so this measures the same code path the demo runs."""
    t0 = time.perf_counter()
    _ = demo.encode_to_mlcond(img)
    _sync(device)
    t_encode = time.perf_counter() - t0

    t0 = time.perf_counter()
    gen = demo.generate(img, n_steps=steps, seed=seed, cfg_strength=cfg,
                        device=device, solver=solver)
    _sync(device)
    t_total_model = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = img2midi.roll_to_midi_file(gen)
    t_render = time.perf_counter() - t0

    return {"encode": t_encode,
            "sample": max(t_total_model - t_encode, 0.0),   # generate() re-encodes
            "render": t_render,
            "end_to_end": t_total_model + t_render}


def run_arm(demo, img, device, threads, args):
    if device == "cpu" and threads:
        torch.set_num_threads(threads)
    label = f"{device}" + (f"/{threads}thr" if device == "cpu" and threads else "")

    for _ in range(args.warmup):
        time_once(demo, img, device, args.steps, args.cfg, args.solver, seed=0)

    runs = [time_once(demo, img, device, args.steps, args.cfg, args.solver, seed=i)
            for i in range(args.reps)]
    med = {k: statistics.median(r[k] for r in runs) for k in runs[0]}

    # Function evaluations: euler is 1/step, rk4 is 4. Guidance != 1 runs a
    # conditional and an unconditional pass per evaluation, so it doubles the cost.
    nfe = args.steps * (4 if args.solver == "rk4" else 1)
    passes = nfe * (2 if abs(args.cfg - 1.0) > 1e-9 else 1)

    print(f"\n=== {label} | {args.solver} x{args.steps} steps | CFG {args.cfg} ===")
    print(f"  NFE {nfe}   forward passes {passes}"
          f"{'  (CFG doubles them)' if passes != nfe else ''}")
    for k in ("encode", "sample", "render", "end_to_end"):
        lo, hi = min(r[k] for r in runs), max(r[k] for r in runs)
        print(f"  {k:<11} {med[k]*1000:8.1f} ms   (min {lo*1000:.1f}, max {hi*1000:.1f})")
    return {"label": label, "device": device, "threads": threads,
            "solver": args.solver, "steps": args.steps, "cfg": args.cfg,
            "nfe": nfe, "forward_passes": passes, "reps": args.reps,
            "median_ms": {k: med[k] * 1000 for k in med}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default=None, help="cpu | mps | cuda (default: fastest available)")
    p.add_argument("--threads", type=int, nargs="*", default=[0],
                   help="CPU thread counts to sweep; 0 = torch default. Cheap cloud "
                        "instances are typically 2-4 vCPU, so measure there too.")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--solver", default="euler", choices=["euler", "rk4"])
    p.add_argument("--cfg", type=float, default=4.0)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--example", default=None)
    p.add_argument("--json", default=None, help="also append results to this JSON file")
    args = p.parse_args()

    device = args.device or pcfm_infer.default_device()
    if device not in pcfm_infer.available_devices():
        sys.exit(f"device '{device}' unavailable here; have {pcfm_infer.available_devices()}")

    img, name = _pick_example(args.example)
    demo = pcfm_infer.PixelCFMDemo(device=device)
    demo.load()

    print(f"example: {name}   torch {torch.__version__}   device {device}")
    results = []
    for thr in (args.threads if device == "cpu" else [0]):
        results.append(run_arm(demo, img, device, thr, args))

    if args.json:
        prev = json.loads(Path(args.json).read_text()) if Path(args.json).exists() else []
        Path(args.json).write_text(json.dumps(prev + results, indent=1))
        print(f"\nappended {len(results)} result(s) to {args.json}")


if __name__ == "__main__":
    main()

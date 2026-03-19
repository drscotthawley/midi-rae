#!/usr/bin/env python3
"""Fetch and display loss curves from W&B for midi-rae runs.

Usage:
    python scripts/check_wandb.py                    # show last N runs summary
    python scripts/check_wandb.py <run_id_or_tag>    # show curves for a specific run
    python scripts/check_wandb.py --last N           # show last N runs (default 5)

Prints per-epoch loss values for: val_loss, val_sim, val_sigreg, val_mep, val_fact,
and per-level MEP/sigreg if available.
"""
import sys, os
import wandb

ENTITY  = "drscotthawley"
PROJECT = "ar-mep-swin-midi-rae"

ENC_KEYS = ["val_loss", "val_sim", "val_sigreg", "val_mep", "val_fact"]
DEC_KEYS = ["val_dec", "val_bce", "val_mse", "F1_clean", "F1_masked"]
LEVEL_KEYS = [f"val_level{i}_{k}" for i in range(6) for k in ["mep", "sigreg", "sim", "fact"]]

def get_api():
    return wandb.Api(timeout=30)

def summarize_runs(n=5, project=PROJECT):
    api = get_api()
    runs = api.runs(f"{ENTITY}/{project}", order="-created_at")
    print(f"\n{'Run name':<35} {'Tag':<20} {'Epochs':>6} {'Best val_loss':>13}  State")
    print("-" * 85)
    count = 0
    for run in runs:
        if count >= n: break
        tag = run.config.get("tag", "")
        hist = run.scan_history(keys=["val_loss", "epoch"])
        rows = list(hist)
        if not rows:
            print(f"{run.name:<35} {tag:<20} {'?':>6} {'?':>13}  {run.state}")
            count += 1
            continue
        best = min(r["val_loss"] for r in rows if "val_loss" in r)
        epochs = max((r.get("epoch", 0) for r in rows), default=0)
        print(f"{run.name:<35} {tag:<20} {int(epochs):>6} {best:>13.6f}  {run.state}")
        count += 1

def show_run(identifier, project=PROJECT):
    api = get_api()
    runs = api.runs(f"{ENTITY}/{project}", order="-created_at")
    run = None
    for r in runs:
        tag = r.config.get("tag", "")
        if identifier in tag or identifier in r.id or identifier in r.name:
            run = r
            break
    if run is None:
        print(f"Run not found: {identifier}"); return

    # Auto-detect enc vs dec keys
    is_dec = project.startswith("dec-")
    KEYS = DEC_KEYS if is_dec else ENC_KEYS
    val_key = "val_dec" if is_dec else "val_loss"

    print(f"\nRun: {run.name}  tag={run.config.get('tag','')}  state={run.state}")
    print(f"{'Epoch':>6}  " + "  ".join(f"{k:>12}" for k in KEYS))
    print("-" * (8 + 14 * len(KEYS)))

    hist = run.scan_history(keys=["epoch"] + KEYS)
    rows = sorted((r for r in hist if val_key in r), key=lambda r: r.get("epoch", 0))
    for r in rows:
        ep = int(r.get("epoch", 0))
        vals = "  ".join(f"{r.get(k, float('nan')):>12.6f}" for k in KEYS)
        print(f"{ep:>6}  {vals}")

    if not is_dec:
        # Per-level summary at final epoch (enc only)
        level_keys_present = []
        for lk in LEVEL_KEYS:
            hist2 = list(run.scan_history(keys=["epoch", lk]))
            vals = [r[lk] for r in hist2 if lk in r]
            if vals:
                level_keys_present.append((lk, vals[-1]))
        if level_keys_present:
            print(f"\nPer-level values at final epoch:")
            for lk, v in level_keys_present:
                print(f"  {lk:<35} {v:.6f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("run", nargs="?", help="run id, tag, or name prefix")
    parser.add_argument("--last", type=int, default=5, help="number of recent runs to show")
    parser.add_argument("--project", default=PROJECT, help=f"W&B project name (default: {PROJECT})")
    a = parser.parse_args()
    if a.run:
        show_run(a.run, project=a.project)
    else:
        summarize_runs(a.last, project=a.project)

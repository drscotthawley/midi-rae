# midi-rae autoresearch

This is an autonomous experiment loop for the midi-rae project. The agent modifies training code and configs, launches runs on a remote GPU machine, waits for results, and iterates.

## Machines

| Host | VRAM | Availability | Default config |
|------|------|--------------|----------------|
| `lecun` | 24 GB | Always reachable (external server) | `config_swin` |
| `razer` | 16 GB | Home only — unreachable when user is away | `config_swin_razer` |
| `oryx`  | 8 GB  | Home only — unreachable when user is away | TBD |

**Default to `lecun` for all autonomous runs.** The home machines (`razer`, `oryx`) are only available when the user is at home. Do not attempt to launch on them unless the user confirms they are reachable.

Because different machines have different GPU architectures and CUDA versions, results are not directly comparable across machines. Each machine needs its own baseline. The home machines may also require smaller batch sizes to fit in VRAM — machine-specific configs handle this.

## Setup

To begin a new autoresearch session, work with the user to:

1. **Agree on a host**: default is `lecun`. All results within a session must be compared on the same machine — cross-machine quantitative comparisons are unreliable due to hardware FP differences.
2. **Agree on a run tag prefix**: something short and descriptive (e.g. `lr_sweep`, `swin_depth`). Each run gets a unique suffix appended automatically.
3. **Agree on scope**: encoder (`enc`) or decoder (`dec`) training, and which config to use (default: `config_swin`).
4. **Read the relevant source files**: Read the notebook(s) you'll be modifying — e.g. `nbs/06_train_enc.ipynb` for encoder, `nbs/09_train_dec.ipynb` for decoder. Also read the active config (`configs/config_swin.yaml`). Do not infer code structure — read the actual files.
5. **Establish a baseline**: The first run should be the unmodified code, to record the baseline metric on this machine.
6. **Initialize results.tsv**: Create `results.tsv` with just the header row (see format below).
7. **Confirm and go**.

## Scope and goals

**Focus: encoder training only** (`train_enc`, `nbs/06_train_enc.ipynb`).

**Primary goal**: minimize `best_metric` (validation loss) on the encoder. Downstream, the best encoder should also produce the best decoder scores — but encoder val loss is the working proxy metric.

**Active objectives** — all of these should remain active (non-zero lambda) in every run:
- **LeJEPA attraction loss** — core; do not disable
- **SIGReg / factorization loss** — keep on
- **MEP loss** — keep on; target EMA eta of ≥ 0.9 (currently may be lower — increasing it is a valid experiment)
- **EMA** — keep on

**Hard constraints**:
- `lambda_mae = 0` always — MAE is completely off for all runs, no exceptions
- No other lambda may be set to zero — use a small value instead if you want to de-emphasize a loss term
- Keep the Swin hierarchy (multi-scale structure)

**Fair game** (neither required nor forbidden):
- Curriculum learning on pitch/time deltas — currently off, can be re-enabled and tuned
- One-cycle LR schedule parameters (`pct_start`, `max_lr`, `final_lr`, etc.) — or replace with a different schedule entirely
- Number of training epochs — default 100 (~3.5 hrs on lecun); shorter runs are acceptable if they still discriminate between ideas
- Architecture changes within the Swin framework

**Promising architectural idea to try**: replace the finest (lowest) level of the Swin hierarchy with lightweight conv layers (e.g. `Conv2d → GELU → Conv2d`) and optionally disable the attraction loss at that level. This might allow adding an even finer level to the hierarchy.

## What you can modify

- **Notebook files** in `nbs/` — this is the primary target. Edit the relevant notebook cells, then `nbdev-export` is run automatically by `launch.sh` before copying files.
- **Config files** in `configs/` — hyperparameters, architecture settings, etc.

**Do NOT edit `.py` files in `midi_rae/` directly** — they are auto-generated from notebooks and will be overwritten.

## How to launch a run

```bash
./scripts/launch.sh <host> <enc|dec> <config> <tag>
```

- `host`: SSH host as defined in `~/.ssh/config` (e.g. `lecun`)
- `enc|dec`: encoder or decoder training
- `config`: config name without `.yaml` (e.g. `config_swin`)
- `tag`: short label; a 6-char random suffix is appended to form the unique run tag

The script will:
1. Check that the GPU is free — aborts if busy
2. Run `nbdev-export` locally to compile notebooks → `.py` files
3. Copy `midi_rae/*.py` and the config to a unique run directory on the remote host
4. Launch training in the background (non-blocking) and return immediately

## How to wait for a run to finish

```bash
./scripts/wait.sh <host>
```

This blocks, polling `status.sh` every 2 minutes, until the run is no longer RUNNING. It prints a timestamped status update each poll. Use this after every `launch.sh` call. Do not poll more frequently than every 2 minutes — a full 100-epoch run takes ~3.5 hours on lecun (~2 min/epoch).

## How to check status

```bash
./scripts/status.sh <host>
./scripts/status.sh <host> <run_dir>   # specific run
```

Output shows: run directory, RUNNING/FINISHED/KILLED/CRASHED status, and the last few lines of the log (tqdm lines filtered out).

## Output format

When training finishes successfully, the log ends with:

```
FINISHED. Best metric: 0.123456
```

Extract it with:

```bash
ssh <host> "grep 'FINISHED. Best metric' ~/runs/midi-rae/<run_tag>/run.log | tail -1"
```

Or just read the status output — the FINISHED case prints the result line directly.

## Git workflow

**Branch**: all autoresearch work lives on a dedicated branch, e.g. `autoresearch/mar16`. Never commit to `main`. Create the branch at session start:

```bash
git checkout -b autoresearch/<tag>   # e.g. autoresearch/mar16
```

**Do not push to GitHub** — these branches are local only. CI workflows are configured to ignore `autoresearch/**` branches anyway.

Before each experiment:
1. Make your changes to the notebook(s) and/or config
2. `git add` the changed notebook(s)/config(s)
3. `git commit -m "<short description of what this experiment tries>"`
4. Then launch

Commits track what changed in each experiment. The `.py` files in `midi_rae/` are generated — do not commit them; they are in `.gitignore`.

## Logging results

Record every run in `results.tsv` (tab-separated — commas break in descriptions):

```
commit	host	type	best_metric	status	description
```

1. `commit` — 7-char short git hash
2. `host` — machine the run was on (e.g. `lecun`)
3. `type` — `enc` or `dec`
4. `best_metric` — value from "FINISHED. Best metric:" line; use `0.000000` for crashes/kills
5. `status` — `keep`, `discard`, `crash`, or `killed`
6. `description` — short text description of what this experiment tried

Example:

```
commit	host	type	best_metric	status	description
a1b2c3d	lecun	enc	0.123456	keep	baseline
b2c3d4e	lecun	enc	0.119200	keep	increase LR to 3e-4
c3d4e5f	lecun	enc	0.131000	discard	add extra Swin stage
d4e5f6g	lecun	enc	0.000000	crash	reduce embed_dim to 32 (OOM)
```

## The experiment loop

LOOP FOREVER:

1. Look at git state: current branch and last commit
2. Choose an experiment — modify notebook(s) and/or config
3. `git commit` with a description of what you changed
4. `./scripts/launch.sh <host> <enc|dec> <config> <tag>`
5. `./scripts/wait.sh <host>` — blocks until done (hours)
6. Read the result from status output or grep the log
7. If the run crashed or was killed, inspect the log, decide whether to fix and retry or skip
8. Log the result in `results.tsv`
9. If `best_metric` improved (lower is better): keep the commit, continue from here
10. If `best_metric` is equal or worse: `git reset --hard HEAD~1` to revert, continue from the previous state

**NEVER STOP**: Once the loop has begun, do not pause to ask if you should continue. The user may be away for many hours. Run until manually interrupted.

**Crashes**: If a run crashes with a clear bug (typo, import error), fix and relaunch. If the idea is fundamentally broken, log it as `crash`, revert, and move on.

**Cross-machine comparisons**: Do not compare `best_metric` values across different machines. Always baseline on the same machine you're experimenting on.

**Simplicity criterion**: A small improvement that adds significant complexity may not be worth keeping. A simplification that achieves equal or better results is always worth keeping.

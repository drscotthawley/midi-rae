# midi-rae autoresearch

This is an autonomous experiment loop for the midi-rae project. The agent modifies training code and configs, launches runs on a remote GPU machine, waits for results, and iterates.

## Pending To-Dos (user action required)

- [x] **Finish setting up Tailscale on razer** — resolved 2026-03-19, razer-ts-docker accessible
- [ ] **Standardize CUDA versions across lecun and razer** — lecun upgraded to driver 595 (CUDA 13.2 max); razer believed to be on CUDA 13.1. Install CUDA 13.0 toolkit + PyTorch cu130 on lecun to match razer as closely as possible.
- [ ] **Add MIDI playback eval** — after fine flow training is validated, add eval code that converts generated piano rolls to MIDI and logs them to W&B using the [midi-player](https://github.com/drscotthawley/midi-player) wandb-compatible player. Allows listening to generated samples directly in the W&B dashboard instead of just looking at piano roll images.

## Standing Config Defaults

These apply to all encoder runs unless explicitly overridden:
- `lambda_fact=0` — factorization loss is tabled pending MEP investigation (see finding below)
- `n_skip_finest_levels=2` — skip LeJEPA at L4+L5
- `lambda_mep=0.1` — MEP loss weight

## Key Findings

### Coarse levels provide suppression context; fine levels drive recall — 2026-03-21

**Finding**: Decoder ablation (dec28, `mask_levels=[0,1,2,3]`, train + val masking applied) shows that training with only L4/L5 visible produces lower precision than recall (precision ~0.80, recall ~0.90 at epoch 20), compared to full-embedding runs where precision ≥ 0.91 by the same point.

**Interpretation**: The coarse levels (L0-L3) carry **global suppression context** — they tell the decoder "even though this local patch looks note-like, in this harmonic/temporal context there should be silence here." Without coarse context, the decoder defaults to a higher base rate of note prediction: it finds most real notes (high recall) but also hallucinates notes in empty regions (low precision). The fine levels (L4/L5) encode *what a note patch looks like locally*; the coarse levels encode *when not to place a note*.

**Implication**: Coarse levels are load-bearing for **precision** (knowing when not to place a note); fine levels handle **recall** (knowing where notes are). This functional decomposition suggests the coarse levels encode higher-level musical context (harmony, rhythm, phrase structure) while fine levels encode local note patterns. This also implies the flow model (which generates L0-L3) is contributing musical *coherence* to generated samples, not just fidelity — even if fine-level reconstruction quality is high with L4/L5 alone.

**Perceptual asymmetry for generation**: For the generative task, false positives (spurious notes) are more damaging than false negatives (missing notes). A spurious note clashes harmonically and rhythmically; a missing note just produces a sparser phrase. This argues for pos_weight < 1.0 for generative decoder training — biasing toward precision over recall — rather than the reconstruction-optimal value of ~2.5. The original pos_weight=2.5 was tuned to balance F1 for reconstruction; generation quality has a different objective.

**Caveat — pos_weight confound**: The same precision < recall pattern also appeared in dec27 (PCA-augmented fine-tune), where coarse levels *were* present but PCA-distorted. This undermines the "coarse suppression context" interpretation. A more parsimonious explanation: `pos_weight=2.5` inherently biases toward recall over precision; with clean embeddings the decoder is confident enough to overcome this bias, but any degradation (PCA distortion or zeroing) reduces decoder confidence and causes it to fall back on the pos_weight prior — predicting more notes to avoid false negatives. The precision deficit may be a pos_weight artifact rather than a direct indicator of missing suppression context. Future ablations should use `pos_weight=1.0` to decouple this effect before drawing conclusions about which levels encode suppression.

### Factorization loss hurts reconstruction — 2026-03-19

**Finding**: exp25 (lambda_fact=0.5) produced decoder F1=98.91% vs exp22 (lambda_fact=0.5 but trained on razer with slightly different dynamics) at 99.4%. Post-training eval shows exp25 has L0 cross_pt=0.007 (near-perfect pitch/time separation) while exp22 has L0 cross_pt=0.112 — yet exp22 produces a better decoder. The factorization loss over-constrains the embedding geometry in ways that hurt reconstruction.

**Decision**: Set `lambda_fact=0` for all subsequent runs. Factorization loss is scientifically interesting (produces cleaner equivariance metrics) but is tabled until MEP experiments are complete and reconstruction quality is fully understood.

**Next**: exp26 (lambda_fact=0, delta_mep=False) and exp27 (lambda_fact=0, delta_mep=True) will establish the clean baseline.

### Equivariance eval baseline — exp19 (2026-03-19)

First run of `scripts/eval_encoder.py` on exp19 checkpoint (n_skip_finest=2, 100 epochs, lecun).

**SVD (effective dims for 90% variance)**:
- L0 (coarsest): 13 dims, top-1 explains 16% — nicely compressed
- L1–L3: 18–22 dims, top-1 8–11%
- L4–L5 (LeJEPA skipped): only 6 and 3 dims, top-1 49% and 75% — barely using capacity, MEP-only supervision

**Factorization cosines** (cross ≈0 good, parallel ≈+1, anti ≈-1):
- Cross terms ≈0.000–0.007 at all levels — near-perfect pitch/time orthogonality
- Parallel/anti structure degrades at L4/L5, with L5 anti going positive (0.23) — expected, no LeJEPA there

**R² equivariance** (pitch/time shift magnitude linearly encoded):
- `r2_pitch`: L0=**0.986**, L1=**0.986**, L2=0.935, L3=0.761, L4=0.425, L5≈0 — pitch equivariance excellent at coarse levels
- `r2_time`: **≈0.0 at all levels** — time shift magnitude not linearly encoded anywhere

**Why r2_time≈0 is expected, not a bug**: Time shifts cause notes to appear/disappear at crop boundaries — the two crops contain partially *different content*, not just a translation of the same content. The model cannot form a consistent "time shift magnitude" vector because the information isn't preserved across the shift. Pitch shifts are clean (whole roll slides up/down, same notes visible). This asymmetry is fundamental to the data structure, not a training failure. The cross terms ≈0 still confirm the time direction is orthogonal to pitch.

### Skip LeJEPA at finest Swin level (L5) — 2026-03-17

**Finding**: Disabling the entire LeJEPA loss (attraction + SIGReg + factorization) at the finest Swin hierarchy level (L5) is a major win. Config: `skip_lejepa_levels: [5]`.

**Results** (25-epoch enc runs on lecun, lambda_fact=0.1):
- Without skip (exp4): enc val_loss = 0.787, dec F1 = ~99.2% (baseline)
- With skip (exp9): enc val_loss = 0.412, dec F1 = **99.66%**

**Why it works**: The finest-level patches are essentially a **discrete vocabulary of musical primitives** (individual notes, rests, partial note edges). SIGReg was forcing these into a Gaussian distribution — a fundamentally wrong prior for discrete/categorical data. Attraction loss compounds this by collapsing the vocabulary, pulling together patches with similar local pixel patterns regardless of musical context. Removing LeJEPA at L5 lets the discrete vocabulary organize itself naturally, shaped only by the Swin hierarchy connections and MEP. MEP is an appropriate objective at this level — it asks "predict the local patch content" without imposing a distributional prior. The hierarchy and output count are unchanged (still 6 levels); the finest level embeddings are still there, just no longer distorted by inappropriate supervision.

**Implication**: Do NOT apply LeJEPA loss at the finest Swin level. This is now hardcoded in `calc_enc_loss_multiscale` (`if lev < n_levels - 1`). There is no need to replace L5 with a CNN block or modify the patch embedding — the architecture is fine; the problem was imposing a Gaussian prior on what is naturally a discrete vocabulary.

**Also noted**: Skipping LeJEPA at L5 reduces training time by ~30% (SIGReg on 204,800 patches was the dominant cost).

**Decoder cross-validation (razer, 2026-03-17)**: An exp9-like encoder run on razer (RTX 2070, 16 GB, codebase not fully in sync with lecun) produced a decoder F1 of **99.78%** in 100 epochs / 73 minutes. Not directly comparable to the lecun dec1 result (99.66%) due to different GPU/CUDA, but corroborates that the skip-L5-LeJEPA architecture generalizes across hardware and produces strong decoders.

**MEP finding**: It is unnecessary to run MEP on the raw patch embedding outputs. Including only the 6 Swin stage outputs (L0–L5) in MEP supervision is sufficient — the finest MEP target is L5 (output of stage 0), which has already been through one round of Swin processing. When L5 has no LeJEPA losses, MEP alone provides sufficient supervision at that level. The raw patch embeddings remain completely unsupervised and the system works well — the Swin hierarchy provides all the structure needed above them.

**Config setting to always include**:
```yaml
skip_lejepa_levels: [5]   # disable entire LeJEPA loss at finest Swin level
```

### exp15 loss curve reinterpretation — 2026-03-18

**Observation**: val_sigreg in exp15 (blue line) drops sharply early then rises back in later epochs. Previously interpreted as SIGReg dominating — actually the opposite: the structural losses (sim, fact) were *pushing back against* SIGReg in later epochs, causing it to rise. val_mep kept descending steadily through all 100 epochs — MEP was never the problem. val_loss also kept improving through epoch 100.

**Implication**: exp15 may have been fine as-is, and lowering `sigreg_prefac` further (exp16) might not help — or could hurt if SIGReg is already being held in check by structural losses. Wait for exp16 result before concluding.

**Also**: MEP descending through epoch 100 suggests the prediction task may have been too easy (same-image, no EMA target challenge). Delta-conditioned MEP (predicting shifted image embeddings) is a harder, more meaningful task.

### SIGReg alone produces incoherent embeddings — exp11 evidence — 2026-03-17

**Observation**: exp11 (lambd curriculum: 0.99→0.3, meaning ~99% SIGReg in early epochs) showed very low SIGReg loss but significantly higher MEP loss compared to runs with attraction active.

**Interpretation**: SIGReg alone pushes each patch embedding independently toward a Gaussian — it produces the right *distribution* but no local *coherence*. Without attraction or factorization, there is no relationship between neighboring or shifted patches. MEP's job is to predict masked patch embeddings from unmasked ones; if embeddings are uncorrelated, MEP has nothing to latch onto and its loss rises.

**Implication**: The higher MEP loss under SIGReg-only is direct evidence that attraction/factorization are what create the predictable local structure that MEP exploits. This retroactively validates the exp12 design: keeping `lambd=0.3` fixed (so factorization is always active at 70%) and only ramping `lambda_sim` (attraction) from 0→1. Factorization builds pitch/time structure from epoch 1; MEP can predict into that structure; attraction is introduced once the geometry is established.

## Machines

| Host | GPU | VRAM | Speed | Availability | Default config |
|------|-----|------|-------|--------------|----------------|
| `lecun` | RTX 4090 | 24 GB | ~90s/epoch | Always reachable (external server) | `config_swin` |
| `razer-docker` | RTX 4090 Max-Q | 16 GB | ~90s/epoch | Home only | `config_swin_razer` |
| `oryxpro` | RTX 2070 | 8 GB | ~5min/epoch | Home only | `config_swin_oryxpro` |

**Default to `lecun` for all autonomous runs.** The home machines (`razer`, `oryxpro`) are only available when the user is at home. Do not attempt to launch on them unless the user confirms they are reachable.

**oryxpro is ~4x slower than lecun/razer** — use it for smoke tests, OOM checks, and decoder runs only. Not suitable for full screening runs.

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

## Project context

midi-rae is a **representation autoencoder for MIDI piano roll images**, designed to learn a structured latent space that captures musical semantics — particularly repeated motifs, and pitch/time shift structure. The architecture is a Swin Transformer-based encoder trained with LeJEPA (attraction + SIGReg factorization), MEP, and EMA.

The ultimate goal is a latent space good enough to support a **generative model** (flow matching or diffusion) trained directly in that space. A good latent space means:
- Repeated motifs cluster together
- Pitch shifts and time shifts correspond to structured, separable directions (factorization)
- The geometry is smooth enough for a generative model to learn

Val loss is a proxy for encoder quality, not a direct measure of latent structure richness. Keep this in mind when evaluating results — a small val_loss improvement that destroys latent structure geometry is not a win.

**The RAE philosophy**: The encoder is trained *without* a reconstruction objective, using only internal consistency requirements (attraction, factorization, MEP). The goal is representations rich enough that reconstruction works well *as a byproduct* — not because we optimized for it. Training end-to-end for reconstruction would maximize F1 but destroy the representation quality. The decoder F1 is a sanity check that the representations are usable, not the optimization target.

**Decoder F1 threshold**: 99.6% F1 is considered acceptable for downstream use — reconstructions at this level are visually error-free in practice. Current best: 99.78% (exp9-like encoder on razer, 100 epochs, 73 min). lecun best: 99.66% (exp9 encoder, dec1). Note: razer and lecun results are not directly comparable due to different GPU/CUDA versions.

**What we actually care about**: latent space quality — do repeated motifs cluster? Are pitch/time shifts encoded in separable, structured directions? Is the geometry smooth enough for a future generative model? Val_loss is a proxy for this; it is assumed (but not proven) to correlate with representation quality.

## Generative model plan

### Latent space structure (from exp19 eval)
PCA on each level's embeddings reveals effective dimensionality (90% variance):
- L0 (coarsest/CLS): **13 dims** — remarkably compressed; 128×128=16,384 pixels → 13 numbers
- L1–L3: 18–22 dims each
- L4–L5: 6 and 3 dims (LeJEPA-skipped, MEP-only)

### Generation strategies

**Option A — L0-only flat generation** (simplest starting point):
1. PCA(13) on training set L0 embeddings; save transform
2. Train small MLP flow model in 13-dim space
3. Generate → inverse PCA → 256-dim L0 → feed to decoder with L1-L5 zeroed/mean

Useful as a lower bound: if L0-only decoding is already decent, the encoder is doing heavy lifting. If it's terrible, robustness training is needed before cascaded generation.

**Option B — Joint multi-level generation** ← theoretically preferred:
Concatenate PCA projections of all levels into one vector (~13+20+20+18 ≈ 70 dims for L0-L3) and train a single flow model in the joint space. No error accumulation, still tractable for a small MLP.

**Theoretical justification (G2G framework, arXiv 2603.12288)**: "From Garbage to Gold" formally proves that for data generated by a hierarchical latent structure Y ← S¹ → S² → S'², a **Breadth strategy** (more independent proxies of S¹) asymptotically dominates a **Depth strategy** (improving a fixed set of proxies). In our generative context:
- S¹ = underlying musical content
- S² = coarse embeddings (L0–L2)
- S'² = fine embeddings (L3–L5)

Option C (cascaded) is a Depth strategy — it tries to perfect L0 then conditions each finer level on it. Each step accumulates **Structural Uncertainty**: the irreducible ambiguity from the probabilistic S¹→S² mapping that cannot be reduced by cleaning or improving any single level. Option B (joint) is a Breadth strategy — generating L0–L3 simultaneously provides more independent proxies of S¹, formally reducing total Structural Uncertainty. This is a stronger argument than just "avoiding error accumulation."

**Fine levels may not be needed for generation**: The low-rank structure and redundancy between levels (each level is an independent proxy of the same S¹) imply that L4–L5 are largely recoverable from L0–L3 via MEP. The G2G proof says each additional proxy reduces H(S¹|S'²); by L3 we may already have enough proxies that adding L4–L5 to the flow model adds minimal new S¹ information. Suggested pipeline: flow generates L0–L3 jointly (~70 dims), MEP predicts L4–L5 from those, decoder reconstructs from all 6 levels. This limits flow model complexity while leveraging the full hierarchy for decoding.

**Option C — Cascaded generation** (original plan):
Generate L0 first, condition L1 on L0, etc. Risk: error accumulation compounds across levels — formally: each conditioning step cannot reduce Structural Uncertainty, only propagate it. Defer unless A/B fail.

**Mitigations for any approach**:
- **Masked embedding decoder training** (implemented): per-patch masking at MEP-style ratios with learnable mask tokens — forces decoder robustness to imperfect generative model outputs
- Both the G2G redundancy argument and our dec_baseline (99.4% F1) vs dec_masked results confirm the decoder can handle missing fine-level information well

**Recommended order**: Try Option A first (fast lower bound), then Option B with L0–L3 joint flow + MEP for L4–L5.

**Mini-batch OT via `ann_repair` — theoretically and practically free**: Random source-target pairing produces a highly tangled velocity field (paths cross everywhere), creating high curvature that is hard to learn and unstable to integrate. `ann_repair` re-pairs each batch by sorting source and target along random projections (1-D OT), straightening the flow paths at O(B log B) cost. In practice this costs nothing: the embedding DataLoader is the bottleneck, and the GPU would otherwise sit idle waiting for the next batch. The sort ops run entirely on-device during that idle window. Observed effect: GPU utilization jumps from ~8% (random pairing) to >90% (with `ann_repair`) — the extra compute is genuinely free.

## Encoder probing / musical structure analysis (for paper — 2026-04-11)

**Goal**: demonstrate that the encoder captures musically meaningful structure, not just reconstruction-useful features. This is a key missing piece for the ISMIR submission.

### Self-similarity matrix (SSM) comparison — highest priority, ~1 day effort

Slide a 128×128 window across a full song at regular intervals (e.g. every 16 or 32 columns = every beat or 2 beats). Embed each crop with the encoder. Compute pairwise cosine similarity between all crop embeddings → embedding-space SSM. Compare to pixel-space SSM (raw binary roll similarity).

**What to look for**: chorus/verse repetition appears as off-diagonal bright blocks. If embedding SSM shows similar block structure to pixel SSM, the encoder captures phrase-level repetition without being trained to. If embedding SSM is *cleaner* (less noisy blocks), the encoder is doing something useful beyond pixel matching.

**Quantitative summary (no labels needed)**: Pearson correlation between flattened embedding SSM and pixel SSM across a set of songs. One number, interpretable, label-free.

**Key point**: no chord/key/verse labels needed — the SSM is fully unsupervised. The visual correspondence IS the result.

### Melody vs. accompaniment separation

POP909 provides labeled melody and accompaniment tracks. Natural experiment: does the embedding change more when the melody changes (accompaniment fixed) vs. when the accompaniment changes (melody fixed)? If L0 is sensitive to melody changes but robust to accompaniment changes (or vice versa), that suggests level-specific musical encoding.

### Motif / periodicity in embedding trajectory

For a single song, compute embeddings at regular time steps → sequence of vectors. Compute autocorrelation of the embedding trajectory. If the model encodes phrase structure, expect peaks at 4-bar (~32 columns at 16th-note resolution) and 8-bar intervals.

### Longer timescales (future architecture work)

Current model sees at most 2 measures (128 columns at 120bpm, 16th-note resolution). To capture song-level structure (verse/chorus/bridge), would need an additional hierarchy level below L0 treating each 128×128 crop as a token — essentially a "song-level transformer" on top of the current patch-level encoder. Noted as future work.

## Scope and goals

**Focus: encoder training only** (`train_enc`, `nbs/06_train_enc.ipynb`).

**Primary goal**: minimize `best_metric` (validation loss) on the encoder. Downstream, the best encoder should also produce the best decoder scores — but encoder val loss is the working proxy metric. Be aware that val_loss alone does not fully capture latent space quality.

**Role of each component** — clear separation of concerns:
- **SIGReg** — anti-collapse; prevents embedding space from degenerating. Keep on.
- **Attraction + factorization losses** — geometry; encode pitch/time shift structure. Keep on.
- **MEP** — prediction quality; drives representation richness. Keep on.
- **EMA** — generalization boost only; NOT needed for collapse prevention (SIGReg handles that). Defer EMA tuning until main dynamics are understood. Do not add EMA complexity until simpler levers are exhausted.

**Active objectives** — all of these should remain active (non-zero lambda) in every run:
- **LeJEPA attraction loss** — core; do not disable
- **SIGReg / factorization loss** — keep on
- **MEP loss** — keep on

**Hard constraints**:
- `lambda_mae = 0` always — MAE is completely off for all runs, no exceptions
- No other lambda may be set to zero — use a small value instead if you want to de-emphasize a loss term
- `lambda_fact >= 0.001` minimum, but **do not optimize by reducing lambda_fact** — lower values approach removing factorization entirely, which defeats the research objective. The goal is to keep factorization alive and find other ways to reduce val_loss. Use lambda_fact >= 0.1 for meaningful factorization signal.
- Keep the Swin hierarchy (multi-scale structure)

**Fair game** (neither required nor forbidden):
- Curriculum learning on pitch/time deltas — currently off, can be re-enabled and tuned
- One-cycle LR schedule parameters (`pct_start`, `max_lr`, `final_lr`, etc.) — or replace with a different schedule entirely
- Number of training epochs — default 100 (~3.5 hrs on lecun); shorter runs are acceptable if they still discriminate between ideas
- Architecture changes within the Swin framework

**Promising architectural idea to try**: replace the finest (lowest) level of the Swin hierarchy with lightweight conv layers (e.g. `Conv2d → GELU → Conv2d`) and optionally disable the attraction loss at that level. This might allow adding an even finer level to the hierarchy.

**Decoder robustness idea (for latent generative model prep)**: during decoder training, randomly mask some fraction of the encoder embeddings before passing them to the decoder, and train the decoder to reconstruct despite the missing embeddings. Motivation: a flow matching or diffusion generative model trained in the latent space will produce imperfect samples — the decoder needs to be robust to small inaccuracies in the embeddings it receives, not just trained on perfect encoder outputs. Masked decoder training is an explicit way to build in that robustness.

## What you can modify

- **Notebook files** in `nbs/` — this is the primary target. Edit the relevant notebook cells, then `nbdev-export` is run automatically by `launch.sh` before copying files.
- **Config files** in `configs/` — hyperparameters, architecture settings, etc. **Always edit the YAML directly. Never use Hydra CLI overrides (`++key=val`) when launching.** The user reads `config_swin.yaml` in the run directory to understand how a run was trained — CLI overrides make that file misleading. The YAML must be the authoritative record.

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

## Suggested experiment order

Start with low-risk, high-payoff changes before architectural surgery:

1. **Switch to flat + cosine-tail LR schedule, then re-baseline** — one-cycle normalizes its LR trajectory to total epochs, so early stopping is unreliable. A warmup → flat → short cosine tail (e.g. last 10 epochs) decouples LR from epoch count. `ReduceLROnPlateau` is another option. **Important**: after switching schedules, run a fresh 25-epoch baseline *before* making any other changes — results are not comparable across schedules. Only then use the tiered strategy: **25 epochs to screen** ideas (~50 min, 4x throughput), **50 to confirm**, **100 for final**. Do not compare 25-epoch flat-schedule results against the 100-epoch one-cycle baseline.
2. **Enable factorization loss** (`lambda_fact > 0`, e.g. 0.1–0.5) — currently off; enabling it switches to `ShiftedTripletDataset`. Low risk, likely impactful.
3. **Fix EMA eta schedule** — currently `ema_eta=1e-5` with a hardcoded jump to 0.96 at epoch 44. Replace with a cleaner schedule or fixed value ≥ 0.9. Note: lower eta = faster EMA update; 0.96+ means the EMA model changes slowly (more stable target).
3. **Tune lambda_mep, lambd** — the attraction/SIGReg balance and MEP weight are currently at defaults; sweep these.
4. **Disable attraction loss at finest Swin level** — the finest level captures local texture; applying attraction loss there may be counterproductive (pulls locally-similar but semantically-different patches together). Try `skip_attraction_levels=[0]` or equivalent. Can be done independently of the conv change.
5. **Conv layers at finest Swin level** — replace the finest (highest-resolution) Swin stage with `Conv2d → GELU → Conv2d`, combined with no attraction loss at that level. The design intent: let the finest level handle local reconstruction fidelity cheaply (convs are good at this), while reserving the coarser Swin levels purely for semantic representation (attraction + factorization). This separation of concerns may improve both reconstruction *and* representation quality simultaneously, and may allow adding an even finer level cheaply.
5. **Curriculum for factorization vs attraction** — train with only factorization loss for an initial warmup period (e.g. first 20–30 epochs), then phase in attraction loss. Motivation: factorization loss establishes the latent geometry (pitch/time orthogonal directions); if attraction loss is active too early it may distort that geometry before it forms. This is speculative but theoretically well-motivated.

## Future experiment ideas

### Representation-conditioned pixel-space flow (bypass decoder entirely)

**Motivation**: Pixel-space CFM works far better than representation-space flow. The decoder is a persistent source of brittleness (binarization artifacts, recall/precision imbalance, sensitivity to embedding quality). Rather than improving the decoder, eliminate it.

**Idea**: Flow directly in pixel space, using learned representations as a conditioning signal. Two variants:

1. **Flow decoder** — source = L5 (or coarse) embeddings, target = piano roll pixels. Representations define the starting point; flow finds the path to pixel space. Analogous to a learned stochastic decoder without the reconstruction bottleneck.

2. **Conditioned pixel flow** — source = Gaussian noise, target = pixels, conditioning = representations (L3/L4/L5 or full hierarchy). Representations guide generation but don't constrain it — flow has full geometric freedom. More powerful because it doesn't require representation space to be close to pixel space, only informationally sufficient.

**Key insight**: Pixel space can be viewed as the finest level of the existing hierarchy (L6, conceptually). The flow then completes the hierarchy from representation → pixel rather than representation → PCA code → decoder → pixel. This removes two lossy steps (PCA, decoder) at once.

**Open questions (TBD)**:
- Condition on L5 only, or the full coarse+fine hierarchy?
- Train with random hierarchy masking (drop random subsets of levels during training) so the model learns to generate from any sub-part of the hierarchy — enabling flexible conditioning at inference (full stack, coarse-only, single level, etc.). Analogous to classifier-free guidance dropout but over a structured hierarchy.
- The disjoint semantic clustering in coarse levels means conditioning on L0/L1/L2 alone may be sufficient for steerable generation (class-like behavior); fine levels add spatial detail.

**Status**: Deferred — validate representation-space flow first. High priority if e2e flow continues to underperform pixel CFM.

### Learned AE frontend replacing PCA (for decoder recall improvement)

**Motivation**: Decoder trained on PCA-roundtripped embeddings shows lower recall than precision, even at 95% variance. The 5% dropped variance likely contains the note/silence decision boundary (low-variance directions encode rare "note on" patches). PCA is blind to reconstruction loss — it cannot learn to preserve what matters for decoding.

**Idea**: Replace per-level PCA with a small nonlinear MLP autoencoder (encoder + decoder), same bottleneck dims as current PCA. Weights shared across patches within a level (same as encoder treats patches uniformly). Train AE jointly end-to-end with the piano roll decoder — reconstruction loss backpropagates through both.

**Pipeline (replaces fitpca + pca_aug)**:
1. Train AE + decoder jointly (AE replaces PCA roundtrip augmentation; no `pca_aug` needed)
2. Freeze AE; preencode dataset using frozen AE encoder → save latent codes (same format as current PCA chunks)
3. Train coarse flow in AE latent space (same as now)
4. Train fine flow with frozen coarse + frozen AE (for viz)

**Whitening**: AE latent space won't be whitened like PCA. Options: (a) add KL/whitening regularizer, (b) "residual PCA" variant — AE learns residual on top of PCA scaffold (`z = PCA(emb) + AE_enc(emb)`; `emb_recon = PCA_inv(z_pca) + AE_dec(z)`), which keeps flow-friendly whitening while capturing what PCA misses.

**Status**: Deferred — try 99% variance PCA first (cheaper). Revisit if recall gap persists.

### Revisit deeper finest-level encoder blocks (post-exp20)

Earlier experiments with `depths=[4,4,4,6,2,1]` (more transformer blocks at L4/L5) showed no benefit. However, those runs had LeJEPA losses active at the finest levels, meaning the extra capacity was being used to fight an inappropriate Gaussian prior. Now that `n_skip_finest_levels=2` skips LeJEPA at both L4 and L5, the finest-level blocks are supervised only by MEP — a much more appropriate objective. More capacity at these levels could now meaningfully improve MEP quality and the richness of the discrete vocabulary organized there.

**To try**: `depths=[4,4,4,6,2,1]` or `[2,2,4,6,2,1]` with `n_skip_finest_levels=2`. Use 50 epochs to screen, 100 to confirm — do not repeat the 200-epoch budget. Compare against exp19 (100 epochs, n_skip=2, val_loss=0.1418) as the baseline — not exp20, which ran 200 epochs and isn't directly comparable.

### Reduce lambda_mep (post-exp20)

`lambda_mep=1.0` has been fixed across all runs. In exp20, MEP loss is monotonically decreasing while other losses fluctuate — suggesting MEP may be dominating late-training optimization and crowding out structural losses.

**To try**: `lambda_mep=0.5` and/or `lambda_mep=0.1` with otherwise identical config to exp19. Use 100 epochs. Compare val_loss against exp19 (0.1418). If reducing MEP weight gives the structural losses more room, we may see improvement similar to the `sigreg_prefac` reduction in exp16.

### Delta-conditioned cross-view MEP
Instead of predicting unmasked img2 embeddings from masked img2 (same image), predict **img3 embeddings from masked img2**, conditioned on the shift delta between them. The MEP model signature becomes `mep_model(enc_out2, deltas)` → predicted embeddings of img3.

**Motivation**: Current MEP is a masked reconstruction task — useful but not directly tied to the pitch/time shift structure. Cross-view MEP with delta conditioning explicitly trains the model to answer "if I shift by delta, what does the representation look like?" This directly incentivizes factorized pitch/time directions in latent space, rather than relying on the factorization loss alone to produce them as an emergent property.

**Implementation notes**: Delta conditioning could be a learned linear projection of the delta vector added to the MEP query embeddings (similar to positional encodings). Requires the triplet dataset (img2, img3, deltas) to be available during MEP — which it already is when `lambda_fact > 0`.

## Embedding quality evaluation ideas

Current automated metrics (val_loss, decoder F1) are coarse. Below are more rigorous evaluation approaches to develop.

### PESTO-inspired transposition equivariance test

**Paper**: "PESTO: Pitch Estimation with Self-supervised Transposition-equivariant Objective" — Riou, Lattner, Hadjeres, Peeters. ISMIR 2023 Best Paper. ([arXiv](https://arxiv.org/abs/2309.02265), [GitHub — inference only](https://github.com/SonyCSLParis/pesto))

PESTO enforces that pitch-shifting the input by N semitones shifts the output by exactly N semitones — the same equivariance our factorization loss targets. Their datasets (MIR-1K, MDB-stem-synth) are audio-based and not directly usable for MIDI piano rolls.

**Borrowed evaluation idea**: Measure how *linearly predictable* the pitch shift amount is from the difference vector in latent space. Concretely, for pitch-shifted pairs (img1, img2) with known delta_pitch:
1. Compute `d = z2 - z1` (mean-pooled per level)
2. Fit a linear regression `delta_pitch → d · pitch_axis` where `pitch_axis` is the first PC of pitch difference vectors
3. Report R² per level — a good factorized encoder should have R² ≈ 1.0 on pitch axis, ≈ 0.0 on time axis (and vice versa)

This is a stricter version of what `factorization_metrics` already measures via cosine similarity. The cosine metrics tell us direction consistency; R² tells us whether the *magnitude* of the shift is also encoded linearly.

**Status**: Idea only — not yet implemented.

### Other evaluation directions to explore
- **Nearest-neighbor retrieval**: Given an anchor piano roll, does the top-K nearest neighbor in latent space contain the same melody/rhythm? Requires a labeled subset.
- **Linear probe on music attributes**: Train a linear classifier on frozen embeddings to predict key, tempo, or instrument — measures semantic content without fine-tuning.
- **Reconstruction quality vs. encoder quality correlation**: Track whether decoder F1 reliably tracks encoder val_loss across runs (confirmed for exp9→dec1, but more data points needed).

### Classifier-free guidance (CFG) for the conditional fine flow model

The `ConditionalFineFlowModel` conditions L4/L5 generation on the coarse L0-L3 flow output. CFG could sharpen the generated fine-level distribution by training with occasional conditioning dropout and amplifying the conditioning signal at inference.

**How it would work**: During training, randomly zero out (or replace with a learned null embedding) the coarse conditioning input ~10–20% of the time. The model learns both unconditional `p(x_fine)` and conditional `p(x_fine | x_coarse)`. At inference, run two forward passes per step and interpolate: `v_guided = v_uncond + w*(v_cond - v_uncond)`. Scale `w > 1` amplifies the conditioning, sharpening the distribution toward what the coarse context implies.

**Why it might help**: The current symptom — smooth generated marginals vs. spiked real data marginals — could reflect the model being insufficiently "committed" to what the coarse context implies. CFG w > 1 would push generated samples harder toward the conditional distribution, potentially producing tighter/spikier marginals that better match real data. Training cost is minimal (just a random dropout on the conditioning input, no architecture change).

**Why it may not help**: Our conditioning is "hard" — the coarse embeddings are concatenated directly as a dense input vector, giving the model a clear gradient path to use them. The model may already be exploiting the conditioning fully. The spiky-vs-smooth problem may be a capacity or training-time issue that more epochs will resolve on its own. CFG also doubles inference cost (two forward passes per ODE step).

**Verdict**: Low-cost experiment worth trying *after* seeing whether the `source_scales=[1.0,1.0]` and `ema_eta=0.9` fixes improve histogram matching. If marginals are still too smooth after adequate training, CFG w ∈ {1.5, 2.0} is a natural next lever. Implementation: add `cfg_dropout_prob=0.15` to `ConditionalFineFlowModel` training, pass `w` as an inference argument to `train_flow_conditional`/`generate`.

### PCA compression of fine-level embeddings ✓ IMPLEMENTED (n=3, fitpca running on lecun + razer)

**The core problem**: `flow2` currently trains in the raw fine embedding space: L4 = 256 patches × 16 dims = 4096 dims; L5 = 1024 patches × 8 dims = 8192 dims; total ~12,000 dims. But from PCA analysis, L4 has only ~6 effective dims and L5 only ~3. We are training a flow model in 12,000-dimensional space when the data lives on a ~9-dimensional manifold. The model wastes enormous capacity discovering that 11,991 directions have near-zero variance — likely explaining the "3 big blobs" result and slow convergence.

**The fix**: Fit PCA on the fine levels (L4, L5) exactly as we do for coarse (L0–L3), store the transforms in `POP909_pca_exp26/` as `pca_L4_n6.pkl` and `pca_L5_n3.pkl` (or use n=20 for consistency and let variance explain itself), then pass fine-level PCA projections to `ConditionalFlowDataset` the same way coarse ones are passed. `flow2` would then operate in ~9-dim (or ~40-dim with n=20) space instead of 12K-dim space.

**Why this is different from the coarse levels**: The coarse PCA is per-level (each level's patch embeddings concatenated across patches, then PCA'd). The same approach works for fine levels — treat each level's full embedding tensor as a single vector per sample, fit PCA, project.

**Expected impact**: Large. A 9-dim flow model trained on the same data as a 12K-dim one should converge orders of magnitude faster and produce much tighter distributions. This is the single highest-leverage change to try for the "3 blobs instead of 5 clusters" problem.

**Implementation**: Extend `fit_pca` (or the existing fitpca script) to also fit transforms for fine levels L4/L5. Then update `ConditionalFlowDataset` to optionally load and apply fine-level PCA transforms. The config would add `flow2.fine_pca_n_components: 20` (or similar).

### Local ancestry prediction (flow2 architecture replacement, high priority after PCA)

**The idea**: Instead of predicting ALL fine patches simultaneously conditioned on a flat global coarse vector, predict each fine patch (or small spatial block of fine patches) conditioned only on its direct ancestor chain in the hierarchy.

**VQ-VAE 2 analogy**: This is exactly the "bottom-level prior conditioned on top-level codes" from Razavi et al. 2019, but with continuous flow matching instead of discrete PixelCNN. The coarse flow sets global structure; the fine flow fills in local detail patch-by-patch.

**Ancestry chain**: For a fine patch at spatial position (i, j) in L4/L5, its ancestors are:
- L3 patch covering (i, j) — direct parent
- L2 patch covering (i, j) — grandparent
- L1 patch — great-grandparent
- L0 patch — great-great-grandparent

Each ancestor is a 20-dim PCA vector (already computed). So the conditioning input per fine patch is 4 levels × 20 dims = **80 dims** — vs. the current ~1700-dim global flat vector. The output per fine patch is **3 dims** (after fine PCA). The flow model solves 80-dim → 3-dim for each fine patch independently (or in small spatial blocks of 2×2 or 4×4 fine patches).

**Why the global vector is fine too**: You could still concatenate the full 1700-dim global coarse vector as extra conditioning (for long-range coherence) — the model just also always gets the local ancestry. In practice, starting with ancestry-only is cleaner: it explicitly encodes spatial position (each patch has a unique ancestor chain) without needing a separate positional embedding, and the model can't confuse "which part of the global blob is relevant to me."

**Expected impact**: The prediction problem shrinks from (1700-dim cond, ~200-dim output for all fine patches) to (80-dim cond, 3-dim output per patch), run in parallel across all patches. This is an astronomically more tractable problem that should converge rapidly.

**Data structure**: No change to files on disk. `ConditionalFlowDataset` needs a new mode that:
1. Loads coarse PCA chunk files (already done) — extract per-patch tensors per level, keeping spatial layout (N, H_l, W_l, 20) rather than flattening
2. Loads fine PCA chunk files (new) — per-patch tensors (N, H_l, W_l, 3)
3. For each sample, constructs all (ancestry_chain, fine_patch_code) pairs — or yields them grouped by spatial block
4. Batch contains many (80-dim, 3-dim) pairs from across all samples and spatial positions

**Model**: A small MLP or lightweight transformer taking 80-dim ancestry + time → 3-dim velocity. Much simpler than the current `ConditionalFineFlowModel`. Can optionally add a few neighboring ancestor patches (the "uncle" patches) for a little more spatial context without going fully global.

**Sequencing**: Implement after fine PCA runs are complete and results are evaluated.

### Ablation: is the conditioning signal actually being used?

Before reaching for advanced techniques, check whether `ConditionalFineFlowModel` is actually using the coarse conditioning input. **Test**: run inference with the coarse conditioning zeroed out vs. real coarse embeddings. If the histograms are identical, the model has learned to ignore the conditioning — the problem is architectural (conditioning not reaching the velocity prediction path) rather than optimization. If they differ noticeably, the conditioning is working and the problem is elsewhere.

### Advanced flow techniques (from Gemini suggestion, lower priority)

These were suggested in the context of disjoint target distributions. Note that our toy 1D Gaussian → 2 disjoint Gaussians worked fine, so none of these are likely the root cause.

- **Divergence matching**: Add `div(v)` term to the FM loss to enforce the continuity equation more strictly near bifurcation regions. Theoretically sound but expensive in high dimensions (needs Hutchinson trace estimator). Skip until the PCA compression approach is tried.

- **Soft-OT / Sinkhorn entropic transport**: Replace hard mini-batch OT (`ann_repair`) with entropy-regularized Sinkhorn assignments. Would smooth the velocity field near cluster boundaries. Harder to scale than `ann_repair` (O(B²) vs O(B log B)) and we already have approximate OT. Low priority.

- **Schrödinger Bridge / cluster-aware prior**: Instead of mapping a standard Gaussian source to the fine embeddings, initialize from a distribution that already knows about the cluster structure. In our case, the coarse embeddings already provide some of this signal via conditioning. A more explicit version: use a GMM fitted on the fine PCA projections as the source distribution instead of a standard Gaussian. This directly addresses the topological mismatch between a unimodal source and a multimodal target.

### Discrete flow matching on the piano roll lattice (speculative)

Piano roll pixels form a natural binary lattice (note on/off). Discrete flow matching (e.g. MDLM, SEDD, or discrete rectified flow) operates directly on categorical/binary spaces without needing a continuous embedding — the "flow" moves between discrete states rather than interpolating in ℝⁿ. Applied to piano rolls, this would mean learning a generative model directly over the pixel grid rather than in the encoder's continuous latent space.

**Why it might be interesting**: the finest Swin level already organizes a discrete vocabulary of musical primitives — this aligns conceptually with a discrete state space. A discrete flow model might better respect the on/off nature of notes and avoid the continuous-space mismatch where the generative model has to learn to "snap" back to plausible binary patterns.

**Why it's not obviously compatible**: our embeddings are continuous (the whole point of the encoder is to produce a smooth latent space for generation). A hybrid approach — discrete flow over raw pixels, continuous flow in the latent space — would require either abandoning the encoder pipeline or treating the two as separate generation strategies. The lattice structure also doesn't carry the hierarchical multi-scale geometry we've built into the encoder.

**Verdict**: speculative but scientifically interesting. Worth revisiting if the continuous flow model underperforms or if discrete generative models mature enough to handle long-range musical structure.

## Low priority future improvements

- **Optimize embedding dims per Swin level**: PCA analysis shows L0 (256-dim) needs only ~13 effective dims — a ~20x overparameterization. A tapered hierarchy (e.g. 16→32→64→64→32→16) could dramatically reduce parameter count. Requires customizing Swin's default doubling-per-stage behavior.

- **Weighted dataset sampling by song length**: Currently `file_idx` sampling gives shorter songs proportionally more coverage than longer ones. Fix with `torch.utils.data.WeightedRandomSampler`, weighting each file by its length in bars. One-liner change to `data.py` but not a correctness issue — current approach is fine for now.

- **Differentiable binarization / Schmitt trigger**: Current `binarize()` and `schmitt_binarize()` are non-differentiable (hard thresholds). Like a real diode vs an ideal one, a differentiable analogue with finite slope everywhere would allow learnable thresholds via backprop. E.g. replace hard threshold with a steep sigmoid (`σ(k*(x - θ))`, large k), and the Schmitt trigger's hysteresis with a recurrent sigmoid cell. Would enable end-to-end training of binarization parameters rather than hand-tuning low/high/init_thresh.

## Open questions

- **Does factorization loss actually help?** It's off in the baseline. Turning it on is the first test.
- **What is the right EMA eta?** The code jumps from 1e-5 → 0.96 at epoch 44. Is this optimal? Should it ramp gradually? Should it start high?
- **Attraction loss at finest level**: Should it be disabled? The finest level captures local texture — attraction loss there may conflate locally-similar but semantically-different patches.
- **Conv vs attention at finest level**: Does replacing the finest Swin stage with conv layers improve results? Does it allow adding a finer level?
- **Curriculum ordering of losses**: Does training factorization-first (then adding attraction) produce better latent geometry than training both simultaneously?
- **lambda_anchor**: Currently 0. Is there value in a small L2 penalty on empty patch embeddings?
- **LR schedule**: One-cycle normalizes to total epochs, making early stopping unreliable. Would a flat+cosine-tail or ReduceLROnPlateau schedule allow valid early stopping at 25 epochs, giving 4x throughput?
- **Epochs**: 100 epochs (~3.5 hrs) for final runs; with a flat LR schedule, 25 epochs (~50 min) may be sufficient to discriminate between ideas — 4x more experiments in the same wall time. Use a tiered strategy: 25 epochs to screen, 50 to confirm, 100 for final.
- **Val_loss vs decoder F1 correlation**: Is encoder val_loss actually predictive of decoder F1? Worth verifying by running the decoder on a few encoder checkpoints with different val_losses. If the correlation is weak, we need a better proxy metric.
- **Can decoder F1 exceed 99.2%?** This is the baseline with the current encoder. Does a better encoder push this higher, or is the decoder bottleneck elsewhere?

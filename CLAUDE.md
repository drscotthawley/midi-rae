# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Critical: nbdev Workflow

**All `.py` files in `midi_rae/` are auto-generated from notebooks in `nbs/`.** Never edit `.py` files directly — edits will be overwritten. Always edit the corresponding notebook (e.g., `nbs/06_train_enc.ipynb` → `midi_rae/train_enc.py`), then run:

```bash
nbdev_prepare   # compiles notebooks → modules, runs tests, cleans notebooks
```

**Import style in notebooks:** Always use absolute imports in notebook cells (e.g. `from midi_rae.utils import to_scalar`). nbdev automatically converts these to relative imports (e.g. `from .utils import to_scalar`) in the exported `.py` files. Writing relative imports directly in notebooks causes an `AssertionError` during export.

To compile without running tests:
```bash
nbdev_export    # notebooks → .py only
```

To run tests:
```bash
nbdev_test      # runs all notebook tests
nbdev_test --path nbs/03_losses.ipynb   # single notebook
```

## Training Commands

```bash
# Encoder training (Hydra config)
PYTHONPATH=$(pwd) python -m midi_rae.train_enc --config-name config_swin_razer.yaml

# Decoder training
PYTHONPATH=$(pwd) python -m midi_rae.train_dec

# Pre-encode images to embeddings (between encoder and decoder training)
PYTHONPATH=$(pwd) python -m midi_rae.preencode
```

Configs live in `configs/`. The active Swin configs are `config_swin*.yaml`. `config.yaml` is the older ViT config.

## Architecture

This project trains a **Masked Autoencoder for MIDI piano roll images** to learn compressed patch representations.

**Three-stage pipeline:**
1. **Encoder training** (`train_enc.py`) — learns patch embeddings via LeJEPA loss (attraction + SIGReg regularization), optional MAE/MEP objectives, curriculum learning on pitch/time shifts
2. **Pre-encoding** (`preencode.py`) — freezes encoder, encodes all training data to disk
3. **Decoder training** (`train_dec.py`) — reconstructs piano rolls from embeddings

**Key modules:**
- `core.py` — `PatchState`, `HierarchicalPatchState`, `EncoderOutput` data structures for multi-scale patch representations
- `swin.py` — Active encoder: Swin Transformer V2 (`SwinEncoder`, `SwinDecoder`, `SwinMaskedEmbeddingPredictor`), multi-stage hierarchical with FPN
- `vit.py` — Legacy ViT encoder (not currently used in active runs)
- `losses.py` — `calc_enc_loss` / `calc_enc_loss_multiscale`: LeJEPA attraction loss + SIGReg (complex number spectral regularization); `calc_mae_loss` for MAE objective
- `data.py` — `PRPairDataset`, `ShiftedTripletDataset`: paired/triplet piano roll sampling with pitch/time augmentation
- `utils.py` — `set_seed(seed=42, deterministic=False)`, EMA, checkpoint save/load
- `viz.py` — UMAP/PCA embedding visualization, MAE reconstruction viz (supports cuML GPU acceleration)

**Loss terminology:** "sim_loss" = "attraction_loss" in the code — these are used interchangeably.

## Reproducibility Notes

- `set_seed(deterministic=True)` enables `cudnn.deterministic=True` and `cudnn.benchmark=False`
- `train_enc.py` has a hardcoded `torch.backends.cudnn.benchmark = True` at module level that is overridden when `set_seed(deterministic=True)` is called in `train()`
- Cross-machine reproducibility is limited: SIGReg uses complex number numerical integration that is sensitive to FP precision differences across GPU architectures
- One-cycle LR schedule peaks at ~30% of total epochs (default `pct_start=0.3`), which is the point of maximum inter-machine divergence

## Code Navigation

Always read the relevant source file before suggesting changes to it. Do not infer code structure from patterns or generic examples — the actual code may differ significantly.

Preserve the structure of the original code. If replacing a one-liner, keep it a one-liner. Conserve vertical space for readability.

When outputting standalone Python code in chat, do not add leading indentation to top-level statements. Python indentation is significant and the user should not have to manually fix it.

## Experiment Tracking

Runs are logged to Weights & Biases. Three machines used for parallel runs: **oryx**, **lecun**, **razer**. Always compare variations against a baseline run on the *same machine* — cross-machine quantitative comparisons are unreliable due to hardware FP differences.

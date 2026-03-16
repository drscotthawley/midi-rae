# Research Log: midi-rae Encoder Training Diagnostics

## Project Goal
Build a representation autoencoder (RAE) in the style of DINOv2 — self-consistency losses only, no reconstruction loss required. The masked autoencoder (MAE) exists but is optional. LeJEPA handles collapse prevention; EMA provides the teacher-student dynamic for robust representations.

## Architecture Summary
- EMA encoder → z1 (no gradients, teacher)
- Online encoder → z2 (gets gradients, student)
- LeJEPA: collapse prevention via attraction loss (sim) + SIGReg
- SIGReg: spectral regularization via complex number numerical integration
- Swin Transformer V2 encoder (hierarchical, 6 levels L0=coarsest, L5=finest)
- One-cycle LR schedule, peaks at epoch 30 (pct_start=0.3) for 100-epoch runs

---

## Finding 1: Cross-Machine Reproducibility

Running the same code, same config, same seed on three machines (oryx, lecun, razer) produces diverging loss curves, particularly after epoch 30. The divergence is large (order of magnitude at epoch 35) and happens simultaneously across all levels.

**Cause:** One-cycle LR peaks at epoch 30, amplifying any accumulated floating-point differences across GPU architectures. SIGReg (complex number numerical integration) is particularly sensitive to FP precision differences.

**Conclusion:** Cross-machine quantitative comparisons are unreliable. Always compare a variation against a baseline run on the same machine.

---

## Finding 2: Non-Empty Patch Masking Imbalances Gradient Contributions Across Levels

Piano roll images are sparse. Non-empty patch fractions by level (approximate, batch_size=200):

| Level | Non-empty | Total | Fraction |
|-------|-----------|-------|----------|
| L0 (coarsest) | 200 | 200 | 100% |
| L1 | ~782 | 800 | ~98% |
| L2 | ~1563 | 3200 | ~49% |
| L3 | ~4600 | 12800 | ~36% |
| L4 | ~13400 | 51200 | ~26% |
| L5 (finest) | ~33000 | 204800 | ~16% |

The non-empty masking was inadvertently down-weighting fine levels (sparse → fewer valid patches → smaller gradient contribution) and up-weighting coarse levels. Removing the masking from the attraction loss balances gradient contributions across the hierarchy.

**Experiment:** Replaced non-empty masking with random sampling (`resample_nes`) using the same patch count as non-empty but random positions. Result: sim loss improved (less spatial inductive bias), sigreg worsened (empty patch embeddings corrupt spectral statistics).

**Resolution:** Apply different masking to each loss component — random/all-patches for attraction loss, non-empty only for SIGReg. Implemented via a separate `valid` argument to `LeJEPA`.

---

## Finding 3: EMA is the Primary Cause of sim_loss Degradation

The original diagnostic concern was that sim_loss had degraded between commits. The actual cause turned out to be the intentional addition of the EMA encoder, not a code bug.

### EMA effects tested:
- eta=0.96: severe sim_loss degradation, sigreg less affected
- eta=0.9: same
- eta=0.8: still too slow, similar behavior
- eta=0.001: still degraded — nearly identical curve to higher eta values

**Key insight:** The degradation is not primarily caused by the eta value. Even eta=0.001 (essentially a near-online encoder) produces similar degradation. The dominant effect is the **loss of gradients on z1** when EMA is active — the loss has only one gradient pathway (through z2) instead of two.

### SIGReg behavior with EMA:
SIGReg is less affected than sim loss because it only regularizes the embedding distribution and doesn't depend on the teacher-student relationship. When EMA is active, passing only z2 (not z1) to SIGReg gives a cleaner signal since z1 is a lagged duplicate anyway.

---

## Finding 4: One-Cycle Schedule May Be Incompatible with EMA

The majority of learning happens between epochs 25-35 (around the LR peak). With EMA active from epoch 0, the teacher is too slow-moving during the most critical learning phase. With EMA active from later epochs, the model has already converged and EMA has little effect.

**Proposed fix:** Staged training — no EMA for epochs 1-32 (full gradients on both z1 and z2), then engage EMA at epoch 33. At that point the EMA encoder initializes as an exact copy of the already-trained online encoder, avoiding a sudden shift in z1.

**Current experiment:** EMA turns on at epoch 33 with eta=0.001 to isolate the gradient loss effect from the eta timescale effect.

---

## Proposed Changes Not Yet Fully Evaluated

### EMA Cosine Schedule
Add cosine-scheduled eta to `EMAModel`: starts at `eta` (eta_start), increases to `eta_end` over `total_steps`. Config adds optional `ema_eta_end`; existing `ema_eta` becomes eta_start.

```python
def _scheduled_eta(self):
    if self.eta_end is None or self.total_steps is None: return self.eta
    k, K = self._steps.item(), self.total_steps
    return self.eta_end - (self.eta_end - self.eta) * (math.cos(math.pi * k / K) + 1) / 2
```

Pairs naturally with cosine decay LR schedule: high LR + low EMA eta early (both moving fast), decaying LR + high EMA eta late (both stabilizing).

### LR Schedule Change
100-epoch one-cycle runs are for rapid diagnostics only. For longer production runs, consider switching from one-cycle to cosine decay (no warmup — gradient clipping covers early instability, and low EMA eta early provides its own stabilization). ReduceLROnPlateau is another option that pairs well with EMA.

### Downstream Evaluation
Current downstream evaluation: decoder reconstruction quality. MAE was not a fair comparison since the encoder hadn't trained well with EMA on. A masked embedding predictor (MEP, à la I-JEPA) is implemented in the current HEAD but not in the diagnostic branch. MEP against EMA targets (rather than online encoder targets) avoids the "cheating" risk where MEP trivially exploits the attraction loss pulling representations together.

---

## Finding 5: Loss of Gradients on z1 is the Dominant Factor; LR Increase Compensates

**Definitive conclusion:** The primary cause of sim_loss degradation when using EMA is the loss of gradients on z1, not the eta value or the EMA timescale. This was confirmed by:
- Engaging EMA at epoch 44 with eta=0.001 (near-online teacher) still caused an immediate jump in sim_loss across all levels
- Using `torch.no_grad()` on z1 with the student encoder (no EMA at all) produced the same degradation pattern

**Compensation:** Increasing the learning rate from 2e-4 to 1e-3 was sufficient to recover desired sim_loss behavior with EMA active. SIGReg was less affected and flattened at a sufficiently low value (~0.001) indicating no collapse.

**Secondary findings ruled out as primary causes:**
- Loss function change (old weighted MSE → new hinge loss with delta-scaled margin): noticeable but qualitatively similar
- EMA eta value: secondary effect
- Non-empty masking changes: rebalances levels but doesn't break training
- Cross-machine FP differences: noise, not signal

---

## Finding 6: Decoder Improvements — Note Weights, MSE Loss, WSL OOM

### Note-length pixel weights
Added `note_length_weights(img)` to `AnchorDataset` — precomputed at init time (one-time cost), stored as float16 to halve RAM footprint. Weight per pixel is `(1/length)^power` normalized by median note length (robust to outliers). Passed to `F.binary_cross_entropy_with_logits` via the `weight=` argument, which is compatible with `pos_weight=` simultaneously.

Key design decisions:
- Median (not mean) used for normalization — mean is skewed by rare very long notes
- `power=0.5` for gentle initial weighting; can increase later
- Background pixels stay at weight 1.0 — foreground/background balance handled separately by `pos_weight`
- float16 storage sufficient; `torch.autocast` casts to bfloat16 at runtime

### MSE loss addition
Added small MSE term (`lambda_mse=0.1`) alongside BCE in `calc_dec_loss`. Motivation: MSE creates a smoother loss landscape at note boundaries, giving the optimizer partial credit for near-correct predictions. This helps resolve off-by-one boundary errors that BCE alone struggles with. The MSE "blur" is removed at evaluation time by binarization thresholding at 0.5.

### WSL2 OOM fix
DataLoader workers were killed by Linux OOM killer due to note_weights doubling dataset RAM footprint. Fix: store note_weights as float16 (`astype(np.float16)`) in the dataset. Halves memory cost, no precision issues since autocast handles it.

### nbdev import rule
In notebook cells, always use absolute imports (e.g. `from midi_rae.utils import to_scalar`). nbdev converts these to relative imports in the exported `.py` files. Writing relative imports directly in notebooks causes an `AssertionError` during `nbdev_export`.

---

## Open Questions
1. Does EMA actually improve downstream representation quality when given sufficient training time (longer runs, cosine schedule)?
2. What is the minimum training duration for EMA to be beneficial?
3. Does the higher LR (1e-3) remain stable for longer runs, or does it need to be scheduled down?

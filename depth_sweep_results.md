# Depth-sweep results

Baseline = MRJ-48∧ (exp26-wide3 on razer-docker), depths=[2,2,2,6,2,2], max_shift_x=48.
All depth-sweep variants (d_*) share lambd=0.15, lambda_fact=0.5, max_shift_x=48,
batch_size=300, 100 epochs, n_skip_finest_levels=2.

Variant 0 (lowsigreg2) uses max_shift_x=12 — included for reference only, not directly comparable.

## Probe 10 — Chroma R² (per-time-slice)

| Variant | Depths | Blocks | shift | L0 | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|---|---|---|---|
| baseline | [2,2,2,6,2,2] | 16 | 48 | -0.004 | 0.028 | 0.151 | 0.382 | 0.575 | 0.701 |
| lowsigreg2 ⚠️ | [2,2,2,6,2,2] | 16 | 12 | -0.016 | -0.094 | -0.013 | 0.290 | -0.210 | -50406389 |

## Probe 11 — Key Detection (24-class accuracy)

| Variant | Depths | Blocks | shift | L0 | L1 | L2 | L3 | L4 | L5 | chance |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | [2,2,2,6,2,2] | 16 | 48 | 0.047 | 0.059 | 0.061 | 0.093 | 0.139 | 0.178 | 0.042 |
| lowsigreg2 | [2,2,2,6,2,2] | 16 | 12 | 0.051 | 0.055 | 0.084 | 0.107 | 0.173 | 0.217 | 0.042 |

## Probe 5 — Density R²

| Variant | Depths | Blocks | shift | L0 | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|---|---|---|---|
| baseline | [2,2,2,6,2,2] | 16 | 48 | 0.357 | 0.490 | 0.682 | 0.779 | 0.818 | 0.893 |
| lowsigreg2 | [2,2,2,6,2,2] | 16 | 12 | 0.244 | 0.304 | 0.573 | 0.653 | 0.805 | 0.895 |

## Probe 2 — Root Note (12-class accuracy)

| Variant | Depths | Blocks | shift | L0 | L1 | L2 | L3 | L4 | L5 | chance |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | [2,2,2,6,2,2] | 16 | 48 | 0.086 | 0.104 | 0.138 | 0.196 | 0.249 | 0.264 | 0.083 |
| lowsigreg2 | [2,2,2,6,2,2] | 16 | 12 | 0.091 | 0.096 | 0.125 | 0.190 | 0.224 | 0.256 | 0.083 |

## Probe 4 — Cross-Song Ratio (lower = better)

| Variant | Depths | Blocks | shift | L0 | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|---|---|---|---|
| baseline | [2,2,2,6,2,2] | 16 | 48 | 0.631 | 0.642 | 0.680 | 0.633 | 0.699 | 0.830 |
| lowsigreg2 | [2,2,2,6,2,2] | 16 | 12 | 0.923 | 0.956 | 0.942 | 0.956 | 0.920 | 0.844 |

## Probe 1 — Chord Quality (4-class accuracy)

| Variant | Depths | Blocks | shift | L0 | L1 | L2 | L3 | L4 | L5 | chance |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | [2,2,2,6,2,2] | 16 | 48 | 0.532 | 0.544 | 0.553 | 0.551 | 0.541 | 0.529 | 0.500 |
| lowsigreg2 | [2,2,2,6,2,2] | 16 | 12 | 0.541 | 0.524 | 0.538 | 0.524 | 0.536 | 0.550 | 0.500 |

## Best metric (encoder val loss)

| Variant | Depths | Blocks | shift | Best metric | Run tag | Host |
|---|---|---|---|---|---|---|
| baseline | [2,2,2,6,2,2] | 16 | 48 | (not re-read) | exp26-wide3 | razer-docker |
| lowsigreg2 | [2,2,2,6,2,2] | 16 | 12 | 0.345709 | lowsigreg2_gh1jfm | tsrazer-ts-docker |

## Notes

**lowsigreg2 (variant 0):** L5 chroma R² = -50,406,389 — catastrophically pathological. L4 also negative (-0.210). The encoder trained normally (density R² = 0.895 at L5), so this is a chroma-specific collapse. This is worse than lowsigreg1 (-15.477), despite having the extra L5 depth block (depths[5]=2 vs 1). Conclusion: the extra L5 block did NOT fix the chroma collapse. **max_shift_x=48 (not depths) is the key differentiator between MRJ-48∧ (good chroma) and the lowsigreg runs (collapsed chroma).** All 8 depth-sweep variants use shift=48 and should not exhibit this failure.

Cross-song ratio for lowsigreg2 is also substantially worse than baseline across all levels (0.844–0.956 vs 0.631–0.830), suggesting shift=12 prevents the model from learning song-invariant structure at fine levels.

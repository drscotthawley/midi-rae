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
| d_uniform3 | [3,3,3,3,3,3] | 18 | 48 | -0.004 | 0.045 | 0.188 | 0.415 | 0.585 | **0.736** |
| d_plus1 | [3,3,3,6,3,3] | 22 | 48 | -0.004 | 0.034 | 0.161 | 0.355 | 0.494 | 0.583 |

## Probe 11 — Key Detection (24-class accuracy)

| Variant | Depths | Blocks | shift | L0 | L1 | L2 | L3 | L4 | L5 | chance |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | [2,2,2,6,2,2] | 16 | 48 | 0.047 | 0.059 | 0.061 | 0.093 | 0.139 | 0.178 | 0.042 |
| lowsigreg2 | [2,2,2,6,2,2] | 16 | 12 | 0.051 | 0.055 | 0.084 | 0.107 | 0.173 | 0.217 | 0.042 |
| d_uniform3 | [3,3,3,3,3,3] | 18 | 48 | 0.050 | 0.054 | 0.092 | 0.119 | 0.159 | **0.234** | 0.042 |
| d_plus1 | [3,3,3,6,3,3] | 22 | 48 | 0.053 | N/A | N/A | N/A | N/A | N/A | 0.058 |

## Probe 5 — Density R²

| Variant | Depths | Blocks | shift | L0 | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|---|---|---|---|
| baseline | [2,2,2,6,2,2] | 16 | 48 | 0.357 | 0.490 | 0.682 | 0.779 | 0.818 | 0.893 |
| lowsigreg2 | [2,2,2,6,2,2] | 16 | 12 | 0.244 | 0.304 | 0.573 | 0.653 | 0.805 | 0.895 |
| d_uniform3 | [3,3,3,3,3,3] | 18 | 48 | 0.267 | 0.347 | 0.558 | 0.624 | 0.735 | **0.907** |
| d_plus1 | [3,3,3,6,3,3] | 22 | 48 | 0.241 | 0.291 | 0.521 | 0.714 | 0.801 | 0.891 |

## Probe 2 — Root Note (12-class accuracy)

| Variant | Depths | Blocks | shift | L0 | L1 | L2 | L3 | L4 | L5 | chance |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | [2,2,2,6,2,2] | 16 | 48 | 0.086 | 0.104 | 0.138 | 0.196 | 0.249 | 0.264 | 0.083 |
| lowsigreg2 | [2,2,2,6,2,2] | 16 | 12 | 0.091 | 0.096 | 0.125 | 0.190 | 0.224 | 0.256 | 0.083 |
| d_uniform3 | [3,3,3,3,3,3] | 18 | 48 | 0.100 | 0.116 | 0.161 | 0.218 | 0.230 | **0.301** | 0.083 |
| d_plus1 | [3,3,3,6,3,3] | 22 | 48 | 0.070 | 0.086 | 0.145 | 0.191 | 0.233 | 0.263 | 0.083 |

## Probe 4 — Cross-Song Ratio (lower = better)

| Variant | Depths | Blocks | shift | L0 | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|---|---|---|---|
| baseline | [2,2,2,6,2,2] | 16 | 48 | 0.631 | 0.642 | 0.680 | 0.633 | 0.699 | 0.830 |
| lowsigreg2 | [2,2,2,6,2,2] | 16 | 12 | 0.923 | 0.956 | 0.942 | 0.956 | 0.920 | 0.844 |
| d_uniform3 | [3,3,3,3,3,3] | 18 | 48 | 0.715 | 0.781 | 0.736 | 0.745 | 0.748 | **0.722** |
| d_plus1 | [3,3,3,6,3,3] | 22 | 48 | 0.507 | 0.511 | 0.538 | 0.655 | 0.607 | **0.599** |

## Probe 1 — Chord Quality (4-class accuracy)

| Variant | Depths | Blocks | shift | L0 | L1 | L2 | L3 | L4 | L5 | chance |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | [2,2,2,6,2,2] | 16 | 48 | 0.532 | 0.544 | 0.553 | 0.551 | 0.541 | 0.529 | 0.500 |
| lowsigreg2 | [2,2,2,6,2,2] | 16 | 12 | 0.541 | 0.524 | 0.538 | 0.524 | 0.536 | 0.550 | 0.500 |
| d_uniform3 | [3,3,3,3,3,3] | 18 | 48 | 0.553 | 0.556 | 0.561 | 0.556 | 0.549 | 0.554 | 0.542 |
| d_plus1 | [3,3,3,6,3,3] | 22 | 48 | 0.544 | 0.550 | 0.537 | 0.545 | 0.534 | 0.540 | 0.545 |

## Best metric (encoder val loss)

| Variant | Depths | Blocks | shift | Best metric | Run tag | Host |
|---|---|---|---|---|---|---|
| baseline | [2,2,2,6,2,2] | 16 | 48 | (not re-read) | exp26-wide3 | razer-docker |
| lowsigreg2 | [2,2,2,6,2,2] | 16 | 12 | 0.345709 | lowsigreg2_gh1jfm | tsrazer-ts-docker |
| d_uniform3 | [3,3,3,3,3,3] | 18 | 48 | 0.357557 | d_uniform3_fJumfJ | razer-docker |
| d_plus1 | [3,3,3,6,3,3] | 22 | 48 | 0.339736 | d_plus1_uJjogu | lecun |
| d_deep8 | [2,2,2,8,2,2] | 18 | 48 | 0.359185 | d_deep8_ZCUKtb | razer-docker |
| d_plus2 | [4,4,4,6,4,4] | 26 | 48 | 0.334969 | d_plus2_rnhC3E | tsrazer-ts-docker |

## Notes

**lowsigreg2 (variant 0):** L5 chroma R² = -50,406,389 — catastrophically pathological. L4 also negative (-0.210). The encoder trained normally (density R² = 0.895 at L5), so this is a chroma-specific collapse. This is worse than lowsigreg1 (-15.477), despite having the extra L5 depth block (depths[5]=2 vs 1). Conclusion: the extra L5 block did NOT fix the chroma collapse. **max_shift_x=48 (not depths) is the key differentiator between MRJ-48∧ (good chroma) and the lowsigreg runs (collapsed chroma).** All 8 depth-sweep variants use shift=48 and should not exhibit this failure.

Cross-song ratio for lowsigreg2 is also substantially worse than baseline across all levels (0.844–0.956 vs 0.631–0.830), suggesting shift=12 prevents the model from learning song-invariant structure at fine levels.

**d_plus1 [3,3,3,6,3,3]:** Probe 11 key detection only completed L0=0.053 before the probe process exited (run.log ends there). Missing L1–L5. Chroma L5=0.583 is below baseline (0.701) and d_uniform3 (0.736). Cross-song L5=0.599 is the **best** of all variants so far (lower is better). The deep L3=6 bottleneck preserves cross-song separation but hurts chroma encoding vs. uniform depth=3.

**d_plus1 vs d_uniform3:** Keeping the L3 bottleneck (depth=6) improves cross-song separation (0.599 vs 0.722) but degrades chroma (0.583 vs 0.736). d_uniform3 wins on most musical content probes; d_plus1 wins on disentanglement.

**d_deep8 [2,2,2,8,2,2]:** Training done. Best metric 0.359185. Probe pending.

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
| d_uniform3 | [3,3,3,3,3,3] | 18 | 48 | -0.004 | 0.045 | 0.188 | 0.415 | 0.585 | 0.736 |
| d_plus1 | [3,3,3,6,3,3] | 22 | 48 | -0.004 | 0.034 | 0.161 | 0.355 | 0.494 | 0.583 |
| d_deep8 | [2,2,2,8,2,2] | 18 | 48 | -0.012 | 0.005 | 0.076 | 0.297 | 0.610 | **0.746** |
| d_plus2 ⚠️ | [4,4,4,6,4,4] | 26 | 48 | -0.003 | 0.043 | 0.138 | 0.351 | 0.617 | -125.515 |
| d_uniform4 | [4,4,4,4,4,4] | 24 | 48 | 0.007 | 0.050 | 0.172 | 0.389 | 0.529 | 0.537 |
| d_coarse ⚠️ | [4,4,4,6,2,2] | 22 | 48 | 0.002 | 0.047 | 0.160 | 0.326 | 0.339 | -712.675 |

## Probe 11 — Key Detection (24-class accuracy)

| Variant | Depths | Blocks | shift | L0 | L1 | L2 | L3 | L4 | L5 | chance |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | [2,2,2,6,2,2] | 16 | 48 | 0.047 | 0.059 | 0.061 | 0.093 | 0.139 | 0.178 | 0.042 |
| lowsigreg2 | [2,2,2,6,2,2] | 16 | 12 | 0.051 | 0.055 | 0.084 | 0.107 | 0.173 | 0.217 | 0.042 |
| d_uniform3 | [3,3,3,3,3,3] | 18 | 48 | 0.050 | 0.054 | 0.092 | 0.119 | 0.159 | **0.234** | 0.042 |
| d_plus1 | [3,3,3,6,3,3] | 22 | 48 | 0.053 | N/A | N/A | N/A | N/A | N/A | 0.058 |
| d_deep8 | [2,2,2,8,2,2] | 18 | 48 | 0.055 | 0.057 | 0.067 | 0.090 | 0.147 | 0.200 | 0.051 |
| d_plus2 | [4,4,4,6,4,4] | 26 | 48 | 0.059 | 0.057 | 0.088 | 0.110 | 0.136 | 0.192 | 0.057 |
| d_uniform4 | [4,4,4,4,4,4] | 24 | 48 | 0.065 | 0.067 | 0.099 | 0.127 | 0.141 | 0.210 | 0.052 |
| d_coarse | [4,4,4,6,2,2] | 22 | 48 | 0.059 | 0.080 | 0.116 | **0.146** | **0.171** | 0.222 | 0.055 |

## Probe 5 — Density R²

| Variant | Depths | Blocks | shift | L0 | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|---|---|---|---|
| baseline | [2,2,2,6,2,2] | 16 | 48 | 0.357 | 0.490 | 0.682 | 0.779 | 0.818 | 0.893 |
| lowsigreg2 | [2,2,2,6,2,2] | 16 | 12 | 0.244 | 0.304 | 0.573 | 0.653 | 0.805 | 0.895 |
| d_uniform3 | [3,3,3,3,3,3] | 18 | 48 | 0.267 | 0.347 | 0.558 | 0.624 | 0.735 | **0.907** |
| d_plus1 | [3,3,3,6,3,3] | 22 | 48 | 0.241 | 0.291 | 0.521 | 0.714 | 0.801 | 0.891 |
| d_deep8 | [2,2,2,8,2,2] | 18 | 48 | 0.338 | 0.355 | 0.644 | 0.740 | 0.840 | 0.892 |
| d_plus2 | [4,4,4,6,4,4] | 26 | 48 | 0.290 | **0.501** | 0.678 | 0.792 | 0.872 | 0.903 |
| d_uniform4 | [4,4,4,4,4,4] | 24 | 48 | 0.268 | 0.306 | 0.486 | 0.654 | 0.817 | 0.892 |
| d_coarse | [4,4,4,6,2,2] | 22 | 48 | 0.316 | 0.416 | 0.617 | 0.742 | 0.864 | **0.929** |

## Probe 2 — Root Note (12-class accuracy)

| Variant | Depths | Blocks | shift | L0 | L1 | L2 | L3 | L4 | L5 | chance |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | [2,2,2,6,2,2] | 16 | 48 | 0.086 | 0.104 | 0.138 | 0.196 | 0.249 | 0.264 | 0.083 |
| lowsigreg2 | [2,2,2,6,2,2] | 16 | 12 | 0.091 | 0.096 | 0.125 | 0.190 | 0.224 | 0.256 | 0.083 |
| d_uniform3 | [3,3,3,3,3,3] | 18 | 48 | 0.100 | 0.116 | 0.161 | 0.218 | 0.230 | **0.301** | 0.083 |
| d_plus1 | [3,3,3,6,3,3] | 22 | 48 | 0.070 | 0.086 | 0.145 | 0.191 | 0.233 | 0.263 | 0.083 |
| d_deep8 | [2,2,2,8,2,2] | 18 | 48 | 0.076 | 0.089 | 0.119 | 0.171 | 0.231 | 0.262 | 0.083 |
| d_plus2 | [4,4,4,6,4,4] | 26 | 48 | 0.104 | 0.106 | 0.131 | 0.172 | 0.203 | 0.265 | 0.083 |
| d_uniform4 | [4,4,4,4,4,4] | 24 | 48 | **0.108** | 0.131 | 0.148 | 0.199 | 0.241 | 0.264 | 0.083 |
| d_coarse | [4,4,4,6,2,2] | 22 | 48 | 0.101 | 0.124 | 0.164 | 0.217 | 0.233 | 0.272 | 0.083 |

## Probe 4 — Cross-Song Ratio (lower = better)

| Variant | Depths | Blocks | shift | L0 | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|---|---|---|---|
| baseline | [2,2,2,6,2,2] | 16 | 48 | 0.631 | 0.642 | 0.680 | **0.633** | 0.699 | 0.830 |
| lowsigreg2 | [2,2,2,6,2,2] | 16 | 12 | 0.923 | 0.956 | 0.942 | 0.956 | 0.920 | 0.844 |
| d_uniform3 | [3,3,3,3,3,3] | 18 | 48 | 0.715 | 0.781 | 0.736 | 0.745 | 0.748 | **0.722** |
| d_plus1 | [3,3,3,6,3,3] | 22 | 48 | **0.507** | **0.511** | **0.538** | 0.655 | **0.607** | **0.599** |
| d_deep8 | [2,2,2,8,2,2] | 18 | 48 | 0.815 | 0.835 | 0.854 | 0.903 | 0.822 | 0.727 |
| d_plus2 | [4,4,4,6,4,4] | 26 | 48 | 0.751 | 0.815 | 0.807 | 0.874 | 0.802 | 0.829 |
| d_uniform4 | [4,4,4,4,4,4] | 24 | 48 | 0.884 | 0.886 | 0.874 | 0.873 | 0.829 | 0.789 |
| d_coarse | [4,4,4,6,2,2] | 22 | 48 | 0.657 | 0.652 | 0.709 | 0.709 | 0.717 | 0.804 |

## Probe 1 — Chord Quality (4-class accuracy)

| Variant | Depths | Blocks | shift | L0 | L1 | L2 | L3 | L4 | L5 | chance |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | [2,2,2,6,2,2] | 16 | 48 | 0.532 | 0.544 | 0.553 | 0.551 | 0.541 | 0.529 | 0.500 |
| lowsigreg2 | [2,2,2,6,2,2] | 16 | 12 | 0.541 | 0.524 | 0.538 | 0.524 | 0.536 | 0.550 | 0.500 |
| d_uniform3 | [3,3,3,3,3,3] | 18 | 48 | 0.553 | 0.556 | 0.561 | 0.556 | 0.549 | 0.554 | 0.542 |
| d_plus1 | [3,3,3,6,3,3] | 22 | 48 | 0.544 | 0.550 | 0.537 | 0.545 | 0.534 | 0.540 | 0.545 |
| d_deep8 | [2,2,2,8,2,2] | 18 | 48 | 0.537 | 0.528 | 0.542 | 0.557 | 0.544 | 0.529 | 0.525 |
| d_plus2 ⚠️2cls | [4,4,4,6,4,4] | 26 | 48 | 0.533 | 0.526 | 0.526 | 0.529 | 0.529 | 0.526 | 0.506 |
| d_uniform4 | [4,4,4,4,4,4] | 24 | 48 | 0.524 | 0.527 | 0.534 | 0.520 | 0.532 | 0.523 | 0.498 |
| d_coarse | [4,4,4,6,2,2] | 22 | 48 | 0.514 | 0.516 | 0.541 | 0.530 | 0.538 | 0.545 | 0.522 |

## Best metric (encoder val loss)

| Variant | Depths | Blocks | shift | Best metric | Run tag | Host |
|---|---|---|---|---|---|---|
| baseline | [2,2,2,6,2,2] | 16 | 48 | (not re-read) | exp26-wide3 | razer-docker |
| lowsigreg2 | [2,2,2,6,2,2] | 16 | 12 | 0.345709 | lowsigreg2_gh1jfm | tsrazer-ts-docker |
| d_uniform3 | [3,3,3,3,3,3] | 18 | 48 | 0.357557 | d_uniform3_fJumfJ | razer-docker |
| d_plus1 | [3,3,3,6,3,3] | 22 | 48 | 0.339736 | d_plus1_uJjogu | lecun |
| d_deep8 | [2,2,2,8,2,2] | 18 | 48 | 0.359185 | d_deep8_ZCUKtb | razer-docker |
| d_plus2 ⚠️ | [4,4,4,6,4,4] | 26 | 48 | 0.334969 | d_plus2_rnhC3E | tsrazer-ts-docker |
| d_uniform4 | [4,4,4,4,4,4] | 24 | 48 | **0.321271** | d_uniform4_zPcusI | tsrazer-ts-docker |
| d_coarse | [4,4,4,6,2,2] | 22 | 48 | **0.321069** | d_coarse_BH3myr | lecun |

## Notes

**lowsigreg2 (variant 0):** L5 chroma R² = -50,406,389 — catastrophically pathological. L4 also negative (-0.210). The encoder trained normally (density R² = 0.895 at L5), so this is a chroma-specific collapse. This is worse than lowsigreg1 (-15.477), despite having the extra L5 depth block (depths[5]=2 vs 1). Conclusion: the extra L5 block did NOT fix the chroma collapse. **max_shift_x=48 (not depths) is the key differentiator between MRJ-48∧ (good chroma) and the lowsigreg runs (collapsed chroma).** All 8 depth-sweep variants use shift=48 and should not exhibit this failure.

Cross-song ratio for lowsigreg2 is also substantially worse than baseline across all levels (0.844–0.956 vs 0.631–0.830), suggesting shift=12 prevents the model from learning song-invariant structure at fine levels.

**d_plus1 [3,3,3,6,3,3]:** Probe 11 key detection only completed L0=0.053 before the probe process exited (run.log ends there). Missing L1–L5. Chroma L5=0.583 is below baseline (0.701) and d_uniform3 (0.736). Cross-song L5=0.599 is the **best** of all variants so far (lower is better). The deep L3=6 bottleneck preserves cross-song separation but hurts chroma encoding vs. uniform depth=3.

**d_plus1 vs d_uniform3:** Keeping the L3 bottleneck (depth=6) improves cross-song separation (0.599 vs 0.722) but degrades chroma (0.583 vs 0.736). d_uniform3 wins on most musical content probes; d_plus1 wins on disentanglement.

**d_plus2 [4,4,4,6,4,4] ⚠️:** Best encoder val loss 0.334969 (best of all variants). Despite the best training metric, chroma L5=-125.515 — catastrophic collapse similar to lowsigreg2, despite using max_shift_x=48. All other probes at L5 are fine (density=0.903, root=0.265, key=0.192, cross-song=0.829). The collapse is L5-specific and sudden (L4 chroma=0.617 is normal). With depths=[4,4,4,6,4,4], all fine levels have depth=4, suggesting that deep fine-level blocks with a strong L3 bottleneck (depth=6) may destabilize L5 chroma encoding despite normal val loss. Note: Probe 1 dropped to 2-class (maj/min only) for this run — values not comparable with other variants.

**d_deep8 [2,2,2,8,2,2]:** Best metric 0.359185. Chroma L5=**0.746** — new overall best, beating d_uniform3 (0.736) and baseline (0.701). Cross-song ratio is notably worse at intermediate levels (L0–L4: 0.815–0.903, vs baseline 0.631–0.699), recovering to 0.727 at L5 (comparable to d_uniform3 0.722). The deeper L3 bottleneck improves chroma at fine levels but degrades song disentanglement at coarse/mid levels. Density (0.892) and root note (0.262) at L5 are slightly below d_uniform3 (0.907, 0.301).

**d_uniform4 [4,4,4,4,4,4]:** Best metric 0.321271 (new best). No chroma collapse (L5=0.537). However, chroma L5=0.537 is **well below** baseline (0.701), d_uniform3 (0.736), and d_deep8 (0.746) — more uniform depth does not help chroma. Key detection L5=0.210 and root note L5=0.264 are also below d_uniform3. The best metric improvement over d_uniform3 (0.321 vs 0.358) did not translate to better representations. Cross-song L5=0.789 is slightly better than baseline (0.830) but worse than d_uniform3 (0.722).

**d_coarse [4,4,4,6,2,2] ⚠️:** Best metric 0.321069 (tied with d_uniform4). **Chroma collapse at L5=-712.675** despite shift=48 and shallow fine levels (L4=2, L5=2). L4 chroma=0.339 is already below baseline (0.575). This is now the 3rd chroma collapse pattern (with d_plus2 and lowsigreg2). **Emerging culprit: L0-L2=4 AND L3=6 together cause chroma collapse.** Baseline has L0-L2=2 (no collapse). d_uniform4 has L0-L2=4 but L3=4 (no collapse). d_plus2 and d_coarse both have L0-L2=4 AND L3=6 (collapse). d_deep8 has L0-L2=2, L3=8 (no collapse — deep L3 alone is safe). Density L5=**0.929** is the new best across all variants. Key detection wins at L3 (0.146) and L4 (0.171).

**Chroma collapse pattern summary:** Collapse occurs when coarse levels are deep (L0-L2≥4) AND the bottleneck is deep (L3≥6). Shallow coarse levels (L0-L2=2) appear to protect against collapse regardless of bottleneck depth. This suggests the coarse pathway dominates pitch encoding under the current SIGReg + shift=48 regime, and overparameterizing the coarse pathway destabilizes it.

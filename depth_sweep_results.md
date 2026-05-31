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
| d_pyramid ⚠️ | [2,3,4,6,4,3] | 22 | 48 | -0.020 | 0.030 | 0.125 | 0.335 | 0.425 | -76.685 |
| d_fine | [2,2,2,6,4,4] | 20 | 48 | -0.012 | 0.012 | 0.087 | 0.319 | 0.553 | -0.074 |

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
| d_pyramid | [2,3,4,6,4,3] | 22 | 48 | 0.056 | 0.064 | 0.092 | 0.112 | 0.162 | 0.218 | 0.059 |
| d_fine | [2,2,2,6,4,4] | 20 | 48 | 0.058 | 0.064 | 0.074 | 0.120 | 0.157 | 0.189 | 0.058 |

## Probe 5 — Density R²

| Variant | Depths | Blocks | shift | L0 | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|---|---|---|---|
| baseline | [2,2,2,6,2,2] | 16 | 48 | 0.357 | 0.490 | 0.682 | 0.779 | 0.818 | 0.893 |
| lowsigreg2 | [2,2,2,6,2,2] | 16 | 12 | 0.244 | 0.304 | 0.573 | 0.653 | 0.805 | 0.895 |
| d_uniform3 | [3,3,3,3,3,3] | 18 | 48 | 0.267 | 0.347 | 0.558 | 0.624 | 0.735 | 0.907 |
| d_plus1 | [3,3,3,6,3,3] | 22 | 48 | 0.241 | 0.291 | 0.521 | 0.714 | 0.801 | 0.891 |
| d_deep8 | [2,2,2,8,2,2] | 18 | 48 | 0.338 | 0.355 | 0.644 | 0.740 | 0.840 | 0.892 |
| d_plus2 | [4,4,4,6,4,4] | 26 | 48 | 0.290 | 0.501 | 0.678 | 0.792 | 0.872 | 0.903 |
| d_uniform4 | [4,4,4,4,4,4] | 24 | 48 | 0.268 | 0.306 | 0.486 | 0.654 | 0.817 | 0.892 |
| d_coarse | [4,4,4,6,2,2] | 22 | 48 | 0.316 | 0.416 | 0.617 | 0.742 | 0.864 | 0.929 |
| d_pyramid | [2,3,4,6,4,3] | 22 | 48 | 0.347 | **0.523** | 0.672 | 0.791 | 0.855 | **0.946** |
| d_fine | [2,2,2,6,4,4] | 20 | 48 | 0.210 | 0.292 | 0.477 | 0.573 | 0.773 | 0.920 |

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
| d_pyramid | [2,3,4,6,4,3] | 22 | 48 | 0.095 | 0.103 | 0.147 | 0.171 | 0.233 | 0.275 | 0.083 |
| d_fine | [2,2,2,6,4,4] | 20 | 48 | 0.091 | 0.085 | 0.112 | 0.170 | 0.219 | 0.248 | 0.083 |

## Probe 4 — Cross-Song Ratio (lower = better)

| Variant | Depths | Blocks | shift | L0 | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|---|---|---|---|
| baseline | [2,2,2,6,2,2] | 16 | 48 | 0.631 | 0.642 | 0.680 | **0.633** | 0.699 | 0.830 |
| lowsigreg2 | [2,2,2,6,2,2] | 16 | 12 | 0.923 | 0.956 | 0.942 | 0.956 | 0.920 | 0.844 |
| d_uniform3 | [3,3,3,3,3,3] | 18 | 48 | 0.715 | 0.781 | 0.736 | 0.745 | 0.748 | 0.722 |
| d_plus1 | [3,3,3,6,3,3] | 22 | 48 | **0.507** | **0.511** | **0.538** | 0.655 | **0.607** | **0.599** |
| d_deep8 | [2,2,2,8,2,2] | 18 | 48 | 0.815 | 0.835 | 0.854 | 0.903 | 0.822 | 0.727 |
| d_plus2 | [4,4,4,6,4,4] | 26 | 48 | 0.751 | 0.815 | 0.807 | 0.874 | 0.802 | 0.829 |
| d_uniform4 | [4,4,4,4,4,4] | 24 | 48 | 0.884 | 0.886 | 0.874 | 0.873 | 0.829 | 0.789 |
| d_coarse | [4,4,4,6,2,2] | 22 | 48 | 0.657 | 0.652 | 0.709 | 0.709 | 0.717 | 0.804 |
| d_pyramid | [2,3,4,6,4,3] | 22 | 48 | 0.662 | 0.687 | 0.661 | 0.802 | 0.682 | 0.720 |
| d_fine | [2,2,2,6,4,4] | 20 | 48 | 0.753 | 0.738 | 0.714 | 0.690 | 0.810 | 0.683 |

## Probe 1 — Chord Quality (accuracy)

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
| d_pyramid | [2,3,4,6,4,3] | 22 | 48 | 0.520 | 0.517 | 0.505 | 0.513 | 0.514 | 0.530 | 0.519 |
| d_fine | [2,2,2,6,4,4] | 20 | 48 | 0.522 | 0.533 | 0.534 | 0.547 | 0.541 | 0.560 | 0.503 |

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
| d_pyramid ⚠️ | [2,3,4,6,4,3] | 22 | 48 | 0.324533 | d_pyramid_6X6DEz | lecun |
| d_fine | [2,2,2,6,4,4] | 20 | 48 | 0.340836 | d_fine_6X1NIa | tsrazer-ts-docker |

## Notes

**lowsigreg2 (variant 0):** L5 chroma R² = -50,406,389 — catastrophically pathological. L4 also negative (-0.210). The encoder trained normally (density R² = 0.895 at L5), so this is a chroma-specific collapse. This is worse than lowsigreg1 (-15.477), despite having the extra L5 depth block (depths[5]=2 vs 1). Conclusion: the extra L5 block did NOT fix the chroma collapse. **max_shift_x=48 (not depths) is the key differentiator between MRJ-48∧ (good chroma) and the lowsigreg runs (collapsed chroma).** All 8 depth-sweep variants use shift=48 and should not exhibit this failure.

Cross-song ratio for lowsigreg2 is also substantially worse than baseline across all levels (0.844–0.956 vs 0.631–0.830), suggesting shift=12 prevents the model from learning song-invariant structure at fine levels.

**d_plus1 [3,3,3,6,3,3]:** Probe 11 key detection only completed L0=0.053 before the probe process exited (run.log ends there). Missing L1–L5. Chroma L5=0.583 is below baseline (0.701) and d_uniform3 (0.736). Cross-song L5=0.599 is the **best** of all variants so far (lower is better). The deep L3=6 bottleneck preserves cross-song separation but hurts chroma encoding vs. uniform depth=3.

**d_plus1 vs d_uniform3:** Keeping the L3 bottleneck (depth=6) improves cross-song separation (0.599 vs 0.722) but degrades chroma (0.583 vs 0.736). d_uniform3 wins on most musical content probes; d_plus1 wins on disentanglement.

**d_plus2 [4,4,4,6,4,4] ⚠️:** Best encoder val loss 0.334969 (best of all variants). Despite the best training metric, chroma L5=-125.515 — catastrophic collapse similar to lowsigreg2, despite using max_shift_x=48. All other probes at L5 are fine (density=0.903, root=0.265, key=0.192, cross-song=0.829). The collapse is L5-specific and sudden (L4 chroma=0.617 is normal). With depths=[4,4,4,6,4,4], all fine levels have depth=4, suggesting that deep fine-level blocks with a strong L3 bottleneck (depth=6) may destabilize L5 chroma encoding despite normal val loss. Note: Probe 1 dropped to 2-class (maj/min only) for this run — values not comparable with other variants.

**d_deep8 [2,2,2,8,2,2]:** Best metric 0.359185. Chroma L5=**0.746** — new overall best, beating d_uniform3 (0.736) and baseline (0.701). Cross-song ratio is notably worse at intermediate levels (L0–L4: 0.815–0.903, vs baseline 0.631–0.699), recovering to 0.727 at L5 (comparable to d_uniform3 0.722). The deeper L3 bottleneck improves chroma at fine levels but degrades song disentanglement at coarse/mid levels. Density (0.892) and root note (0.262) at L5 are slightly below d_uniform3 (0.907, 0.301).

**d_uniform4 [4,4,4,4,4,4]:** Best metric 0.321271 (new best). No chroma collapse (L5=0.537). However, chroma L5=0.537 is **well below** baseline (0.701), d_uniform3 (0.736), and d_deep8 (0.746) — more uniform depth does not help chroma. Key detection L5=0.210 and root note L5=0.264 are also below d_uniform3. The best metric improvement over d_uniform3 (0.321 vs 0.358) did not translate to better representations. Cross-song L5=0.789 is slightly better than baseline (0.830) but worse than d_uniform3 (0.722).

**d_coarse [4,4,4,6,2,2] ⚠️:** Best metric 0.321069 (tied with d_uniform4). **Chroma collapse at L5=-712.675** despite shift=48 and shallow fine levels (L4=2, L5=2). L4 chroma=0.339 is already below baseline (0.575). This is now the 3rd chroma collapse pattern (with d_plus2 and lowsigreg2). Density L5=0.929 is strong (second only to d_pyramid). Key detection wins at L3 (0.146) and L4 (0.171).

**d_pyramid [2,3,4,6,4,3] ⚠️:** Best metric 0.324533. **Chroma collapse at L5=-76.685** despite L0=2, L1=3. This refines the collapse pattern (see below). Density L5=**0.946** is the new best across all variants. Key detection and root note are mid-pack. Note that L5 density being high despite L5 chroma collapse confirms the collapse is chroma-specific.

**d_fine [2,2,2,6,4,4]:** Best metric 0.340836 (weakest of the depth-sweep variants). No catastrophic chroma collapse, though L5=-0.074 is mildly negative (essentially 0). L4 chroma=0.553 is below baseline L4=0.575, and L5 effectively zero — adding depth to fine levels (L4=4, L5=4) does not help chroma. Chord quality L5=0.560 vs chance=0.503 is the largest margin across all variants. Density and root note are consistently below baseline.

**Chroma collapse pattern — refined:** Collapse occurs when **L2≥4 AND L3=6**. L2≤3 is safe regardless of L3 depth or L0/L1 depth. Evidence: d_pyramid (L0=2, L1=3, L2=4, L3=6) → collapse; d_plus1 (L0=3, L1=3, L2=3, L3=6) → safe; baseline (L2=2, L3=6) → safe. The collapse is not about total coarse depth (d_plus1 has L0+L1+L2=9, same as d_pyramid, but is safe because L2=3). It is also not about deep fine levels — d_deep8 (L0-L2=2, L3=8) shows no collapse with a deeper bottleneck. This points to the L2→L3 transition specifically: when L2 is overparameterized and L3 is also deep, something in the gradient flow through that transition destabilizes the L5 chroma representation.

---

## Summary

**Winners by probe (finest usable level, excluding collapsed variants):**

| Probe | Winner | Value | Notes |
|---|---|---|---|
| Chroma R² (L5) | d_deep8 | **0.746** | Deepens only L3; L0-L2 stay at 2 |
| Chroma R² (L4) | d_deep8 | 0.610 | (d_plus2=0.617 but collapses at L5) |
| Key detection (L5) | d_uniform3 | **0.234** | Uniform depth=3 across all levels |
| Note density (L5) | d_fine | 0.920 | (d_pyramid=0.946 but collapses at L5) |
| Root note (L5) | d_uniform3 | **0.301** | |
| Cross-song ratio (L5) | d_plus1 | **0.599** | Lower is better |
| Best val loss | d_coarse | 0.321069 | ⚠️ collapses; low loss ≠ good probes |

**Patterns observed:**

1. **Lower val loss does not predict better representations.** d_uniform4 and d_coarse achieved the best val loss (~0.321) but rank near the bottom on chroma and tonal probes. Training loss is not a reliable selection criterion for this task.

2. **Chroma collapse: L2≥4 AND L3=6 → catastrophic L5 chroma failure.** Four variants collapsed (lowsigreg2 via shift=12; d_plus2, d_coarse, d_pyramid via the L2≥4+L3=6 combination). The only safe configurations with L3=6 have L2≤3 (baseline, d_plus1, d_fine).

3. **d_deep8 [2,2,2,8,2,2] is the overall winner for chroma/pitch content.** Concentrating depth in the bottleneck (L3) while keeping coarse levels shallow outperforms every other depth allocation for chroma. This is the strongest departure from baseline.

4. **d_uniform3 [3,3,3,3,3,3] is the runner-up and safest general choice.** It wins on root note and key detection, is second-best on chroma (0.736), and has no collapse risk. Only marginally more blocks than baseline (18 vs 16).

5. **Deeper fine levels (d_fine) do not help chroma.** L4=4, L5=4 in d_fine produces L4 chroma=0.553 and L5≈0 — worse than baseline (0.575, 0.701). The fine levels appear to be a representational "free zone" where more depth adds no pitch structure.

6. **Cross-song disentanglement and chroma trade off.** d_plus1 wins cross-song separation at the cost of chroma quality; d_deep8 wins chroma at the cost of cross-song separation at coarse/mid levels.

**Recommended next steps:**

- **Follow up on d_deep8:** It wins chroma decisively. Test whether d_deep8 + larger embed_dim (e.g., 16) or longer training further improves chroma. Alternatively, try d_deep10 [2,2,2,10,2,2] to probe the L3 depth ceiling.
- **Hybrid: d_deep8 + d_uniform3 blend:** Try [2,2,3,8,3,2] — keeps L3 deep for chroma, adds modest depth to L2/L4 for tonal probes, avoids the L2≥4+L3=6 trap.
- **Investigate the L2/L3 collapse mechanism:** Run ablations on L2=3 vs L2=4 with L3=6 fixed to confirm the threshold is L2=4 and not cumulative depth.
- **d_plus1 deserves a complete probe re-run.** Its key detection was truncated; it wins cross-song and may be the best choice if disentanglement is prioritized over chroma accuracy.

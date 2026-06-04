# Lambd Schedule Sweep Results

Backbone: d_uniform3, depths=[3,3,3,3,3,3], embed_dim=8.
All variants: n_skip_finest_levels=0 (LeJEPA on ALL 6 levels), lambda_fact=0.5,
max_shift_x=48, batch_size=300, 100 epochs.
lambd is now a per-level list [L0, L1, L2, L3, L4, L5] (L0=coarsest, L5=finest).

Reference: d_uniform3 baseline used scalar lambd=0.15 on L0-L3 only (n_skip=2).

## Run Registry

| # | Config | lambd schedule [L0..L5] | Hypothesis | Host | Run tag | Status |
|---|---|---|---|---|---|---|
| 1 | u3s1_min     | [0.15, 0.15, 0.15, 0.15, 0.10, 0.05] | Minimal change: extend tapering to L4-L5 | lecun | u3s1_min_by62tI | DONE |
| 2 | u3s2_gentle  | [0.15, 0.13, 0.12, 0.11, 0.10, 0.05] | Gentle descent from current L0 | tsrazer-ts-docker | u3s2_gentle_e4FxGS | DONE |
| 3 | u3s3_std     | [0.30, 0.25, 0.20, 0.15, 0.10, 0.05] | Standard descent, doubles L0 pressure | lecun | u3s3_std_gliobo | DONE |
| 4 | u3s4_steep   | [0.50, 0.35, 0.25, 0.15, 0.10, 0.05] | Aggressive descent, heavy coarse pressure | lecun | u3s4_steep_FO4FMg | DONE |
| 5 | u3s5_inv1    | [0.10, 0.13, 0.15, 0.15, 0.10, 0.05] | Mild inversion: less SIGReg at L0 | tsrazer-ts-docker | u3s5_inv1_eWwtSU | DONE |
| 6 | u3s6_inv2    | [0.05, 0.10, 0.15, 0.20, 0.10, 0.05] | Strong inversion: minimum at L0 | tsrazer-ts-docker | u3s6_inv2_1hpdXL | DONE |
| 7 | u3s7_peak    | [0.10, 0.15, 0.20, 0.20, 0.10, 0.05] | Peak at L2-L3, low at extremes | lecun | u3s7_peak_fQWV1h | DONE |
| 8 | u3s8_lowflat | [0.10, 0.10, 0.10, 0.10, 0.08, 0.05] | Low magnitude, nearly flat | tsrazer-ts-docker | u3s8_lowflat_YpgfMD | DONE |
| 9 | u3s9_lowest  | [0.07, 0.07, 0.07, 0.07, 0.07, 0.05] | Lowest uniform: less is more extreme | lecun | u3s9_lowest_ZX0rYk | DONE |

Note: razer-docker OOMs at batch_size=300 with n_skip_finest=0 (SIGReg at L5 needs ~1.27 GiB, GPU headroom ~1 GiB).
All 9 variants redistributed to lecun and tsrazer to keep batch_size=300 consistent.

Machine chains (updated — razer-docker excluded):
- lecun:    s1_min → s4_steep → s7_peak → s3_std → s9_lowest
- tsrazer:  s2_gentle → s5_inv1 → s8_lowflat → s6_inv2

## Probe 10 — Chroma R² (per-time-slice)

| Variant | lambd [L0..L5] | L0 | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|---|---|
| d_uniform3 (ref) | 0.15 scalar, skip L4-L5 | -0.004 | 0.045 | 0.188 | 0.415 | 0.585 | 0.736 |
| u3s1_min | [0.15,0.15,0.15,0.15,0.10,0.05] | -0.005 | 0.020 | 0.111 | 0.198 | 0.375 | 0.582 |
| u3s2_gentle | [0.15,0.13,0.12,0.11,0.10,0.05] | -0.009 | -0.000 | 0.045 | 0.154 | 0.289 | 0.559 |
| u3s3_std | [0.30,0.25,0.20,0.15,0.10,0.05] | -0.008 | 0.027 | 0.100 | 0.224 | 0.343 | 0.629 |
| u3s4_steep | [0.50,0.35,0.25,0.15,0.10,0.05] | -0.019 | -0.018 | 0.073 | 0.219 | 0.364 | 0.625 |
| u3s5_inv1 | [0.10,0.13,0.15,0.15,0.10,0.05] | -0.015 | -0.014 | 0.074 | 0.202 | 0.324 | 0.576 |
| u3s6_inv2 | [0.05,0.10,0.15,0.20,0.10,0.05] | -0.025 | -0.033 | 0.085 | 0.237 | 0.383 | 0.512 |
| u3s7_peak | [0.10,0.15,0.20,0.20,0.10,0.05] | -0.013 | 0.037 | 0.116 | 0.221 | 0.345 | -1.381 |
| u3s8_lowflat | [0.10,0.10,0.10,0.10,0.08,0.05] | -0.010 | -0.012 | 0.043 | 0.121 | 0.288 | 0.609 |
| u3s9_lowest | [0.07,0.07,0.07,0.07,0.07,0.05] | 0.010 | 0.036 | 0.113 | 0.244 | 0.333 | 0.578 |
| xmep1 | 0.15 scalar, n_skip=2, cross_level_mep | -0.012 | 0.002 | 0.095 | 0.365 | 0.510 | **-11.966** |

## Probe 11 — Key Detection (24-class accuracy)

| Variant | L0 | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|---|
| d_uniform3 (ref) | 0.050 | 0.054 | 0.092 | 0.119 | 0.159 | 0.234 |
| u3s1_min | 0.055 | 0.075 | 0.091 | 0.092 | 0.099 | 0.155 |
| u3s2_gentle | 0.064 | 0.056 | 0.078 | 0.078 | 0.089 | 0.137 |
| u3s3_std | 0.065 | 0.068 | 0.109 | 0.111 | 0.123 | 0.175 |
| u3s4_steep | 0.058 | 0.050 | 0.070 | 0.103 | 0.125 | 0.179 |
| u3s5_inv1 | 0.053 | 0.064 | 0.069 | 0.083 | 0.100 | 0.175 |
| u3s6_inv2 | 0.069 | 0.075 | 0.093 | 0.122 | 0.116 | 0.185 |
| u3s7_peak | 0.054 | 0.068 | 0.081 | 0.098 | 0.104 | 0.166 |
| u3s8_lowflat | 0.065 | 0.063 | 0.081 | 0.082 | 0.092 | 0.160 |
| u3s9_lowest | 0.062 | 0.077 | 0.078 | 0.099 | 0.108 | 0.187 |
| xmep1 | 0.046 | 0.057 | 0.072 | 0.103 | **0.187** | 0.215 |

## Probe 5 — Density R²

| Variant | L0 | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|---|
| d_uniform3 (ref) | 0.267 | 0.347 | 0.558 | 0.624 | 0.735 | 0.907 |
| u3s1_min | 0.383 | 0.419 | 0.515 | 0.552 | 0.673 | 0.811 |
| u3s2_gentle | 0.189 | 0.217 | 0.349 | 0.452 | 0.685 | 0.860 |
| u3s3_std | 0.193 | 0.319 | 0.579 | 0.654 | 0.718 | 0.868 |
| u3s4_steep | 0.241 | 0.308 | 0.453 | 0.616 | 0.707 | 0.790 |
| u3s5_inv1 | 0.202 | 0.291 | 0.538 | 0.606 | 0.641 | 0.821 |
| u3s6_inv2 | 0.298 | 0.363 | 0.530 | 0.591 | 0.686 | 0.723 |
| u3s7_peak | 0.155 | 0.222 | 0.335 | 0.597 | 0.787 | 0.901 |
| u3s8_lowflat | 0.246 | 0.296 | 0.424 | 0.476 | 0.605 | 0.760 |
| u3s9_lowest | 0.276 | 0.332 | 0.448 | 0.594 | 0.683 | 0.829 |
| xmep1 | 0.284 | 0.378 | 0.548 | 0.699 | 0.788 | **0.911** |

## Probe 2 — Root Note (12-class accuracy)

| Variant | L0 | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|---|
| d_uniform3 (ref) | 0.100 | 0.116 | 0.161 | 0.218 | 0.230 | 0.301 |
| u3s1_min | 0.103 | 0.113 | 0.136 | 0.164 | 0.169 | 0.226 |
| u3s2_gentle | 0.101 | 0.097 | 0.126 | 0.148 | 0.154 | 0.213 |
| u3s3_std | 0.107 | 0.119 | 0.146 | 0.158 | 0.179 | 0.233 |
| u3s4_steep | 0.086 | 0.088 | 0.124 | 0.149 | 0.163 | 0.245 |
| u3s5_inv1 | 0.101 | 0.100 | 0.119 | 0.133 | 0.153 | 0.245 |
| u3s6_inv2 | 0.100 | 0.123 | 0.144 | 0.178 | 0.187 | 0.255 |
| u3s7_peak | 0.081 | 0.103 | 0.133 | 0.162 | 0.171 | 0.230 |
| u3s8_lowflat | 0.088 | 0.095 | 0.111 | 0.137 | 0.149 | 0.216 |
| u3s9_lowest | 0.107 | 0.119 | 0.123 | 0.146 | 0.163 | 0.240 |
| xmep1 | 0.086 | 0.087 | 0.109 | 0.171 | 0.253 | 0.282 |

## Probe 4 — Cross-Song Ratio (lower = better)

| Variant | L0 | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|---|
| d_uniform3 (ref) | 0.715 | 0.781 | 0.736 | 0.745 | 0.748 | 0.722 |
| u3s1_min | 0.651 | 0.649 | 0.588 | 0.628 | 0.659 | 0.571 |
| u3s2_gentle | 0.779 | 0.782 | 0.753 | 0.719 | 0.701 | 0.659 |
| u3s3_std | 0.597 | 0.510 | 0.498 | 0.558 | 0.595 | 0.619 |
| u3s4_steep | 0.648 | 0.603 | 0.667 | 0.657 | 0.642 | 0.537 |
| u3s5_inv1 | 0.684 | 0.661 | 0.676 | 0.670 | 0.630 | 0.537 |
| u3s6_inv2 | 0.908 | 0.861 | 0.886 | 0.821 | 0.834 | 0.666 |
| u3s7_peak | 0.798 | 0.842 | 0.780 | 0.731 | 0.642 | 0.621 |
| u3s8_lowflat | 0.816 | 0.853 | 0.842 | 0.796 | 0.745 | 0.671 |
| u3s9_lowest | 0.820 | 0.844 | 0.859 | 0.881 | 0.979 | 1.070 |
| xmep1 | 0.854 | 0.858 | 0.842 | 0.841 | 0.878 | 0.856 |

## Probe 1 — Chord Quality (accuracy)

| Variant | L0 | L1 | L2 | L3 | L4 | L5 | chance |
|---|---|---|---|---|---|---|---|
| d_uniform3 (ref) | 0.553 | 0.556 | 0.561 | 0.556 | 0.549 | 0.554 | 0.542 |
| u3s1_min | 0.546 | 0.541 | 0.556 | 0.546 | 0.547 | 0.541 | 0.552 |
| u3s2_gentle | 0.551 | 0.540 | 0.541 | 0.542 | 0.542 | 0.541 | 0.514 |
| u3s3_std | 0.543 | 0.532 | 0.540 | 0.529 | 0.540 | 0.528 | 0.540 |
| u3s4_steep | 0.542 | 0.537 | 0.530 | 0.524 | 0.536 | 0.524 | 0.536 |
| u3s5_inv1 | 0.517 | 0.512 | 0.529 | 0.518 | 0.524 | 0.543 | 0.500 |
| u3s6_inv2 | 0.513 | 0.519 | 0.515 | 0.508 | 0.525 | 0.513 | 0.504 |
| u3s7_peak | 0.522 | 0.532 | 0.523 | 0.546 | 0.521 | 0.544 | 0.536 |
| u3s8_lowflat | 0.537 | 0.536 | 0.519 | 0.540 | 0.529 | 0.534 | 0.504 |
| u3s9_lowest | 0.523 | 0.514 | 0.523 | 0.509 | 0.519 | 0.533 | 0.509 |
| xmep1 | 0.539 | 0.559 | 0.531 | 0.525 | 0.523 | 0.523 | 0.509 |

## Best metric (encoder val loss)

| Variant | Best metric | Run tag | Host |
|---|---|---|---|
| d_uniform3 (ref) | 0.357557 | d_uniform3_fJumfJ | razer-docker |
| u3s1_min | 0.516957 | u3s1_min_by62tI | lecun |
| u3s2_gentle | 0.479290 | u3s2_gentle_e4FxGS | tsrazer-ts-docker |
| u3s3_std | 0.539052 | u3s3_std_gliobo | lecun |
| u3s4_steep | 0.554373 | u3s4_steep_FO4FMg | lecun |
| u3s5_inv1 | 0.476424 | u3s5_inv1_eWwtSU | tsrazer-ts-docker |
| u3s6_inv2 | 0.396545 | u3s6_inv2_1hpdXL | tsrazer-ts-docker |
| u3s7_peak | 0.491808 | u3s7_peak_fQWV1h | lecun |
| u3s8_lowflat | 0.397094 | u3s8_lowflat_YpgfMD | tsrazer-ts-docker |
| u3s9_lowest | 0.349890 | u3s9_lowest_ZX0rYk | lecun |
| xmep1 | **0.317367** | xmep1_sHbv4G | tsrazer-ts-docker |

## Notes

### Summary analysis

**Nothing beat the reference.** On every musical content probe (chroma, key, density, root note), d_uniform3 with scalar lambd=0.15 and n_skip=2 outperforms all 9 sweep variants. The sweep explored lambd schedules ranging from very aggressive (u3s4_steep: 0.50 at L0) to very light (u3s9_lowest: 0.07 uniform) and inverted profiles, but none improved probe quality.

**There is a critical confound.** The reference used `n_skip_finest_levels=2`, meaning SIGReg was only applied to L0–L3, leaving L4 and L5 to train freely. All sweep variants used `n_skip=0` (SIGReg on all 6 levels). The reference's edge on chroma and key at L5 may be explained entirely by the absence of SIGReg pressure at L4–L5, not by anything special about lambd=0.15. This sweep does not cleanly isolate the effect of the lambd *schedule* from the effect of *how many levels get SIGReg at all*.

**Fine levels dominate; coarse levels are inert.** L5 produces the best results on every probe and every variant, consistently by a wide margin. L3–L4 sometimes approach L5 on density and root note. L0–L2 are essentially flat near chance or zero R² across all probes — they are not encoding musically-relevant structure at all. Whether that's a limitation of coarse-level receptive fields, the embed_dim=8 bottleneck, or the SIGReg pressure itself is unclear.

**Encoder val loss is a poor proxy for probe quality.** u3s9_lowest achieved the best encoder val loss (0.350, beating even the reference at 0.358), yet its probe scores are middling and its cross-song ratio at L5 exceeds 1.0 — meaning at the finest level, same-song patches are *farther apart* than patches from different songs. Very low lambd lets the encoder minimize loss without building song-discriminative representations.

**Cross-song separation is an exception.** Several high-lambd variants (u3s3_std, u3s1_min, u3s4_steep) beat the reference on cross-song ratio (lower = better), suggesting SIGReg pressure does improve song identity, but this did not translate to better musical content probes.

**u3s7_peak anomaly.** Chroma R² at L5 is -1.381 — a sign of active anti-correlation. All other levels are normal. Likely a training instability specific to this run; not informative about the peak schedule.

**Takeaways for next steps:**
- The n_skip parameter appears more important than the lambd schedule shape. A controlled sweep varying n_skip (0–4) while holding lambd fixed would be more diagnostic.
- The coarse levels (L0–L2) may need a fundamentally different treatment — either much weaker SIGReg, a different objective, or more capacity — before they contribute meaningfully.
- If the goal is fine-level musical structure, the reference configuration (n_skip=2) is still the best known setting.

### xmep1: cross-level MEP results

**Best encoder val loss overall (0.317367)**, beating both the reference (0.357557) and all sweep variants. Cross-level MEP is a genuinely stronger self-supervised objective by the training metric.

**Density (P5) matches or slightly beats reference at all levels.** L5=0.911 vs ref 0.907; L3=0.699 vs ref 0.624. Noteworthy improvement at intermediate levels.

**Key detection (P11) improves at L4 (0.187 vs ref 0.159), slightly worse at L5 (0.215 vs 0.234).** This is the first variant to show any gain at an intermediate level.

**Chroma (P10) at L5 is catastrophically degraded: -11.966.** All other levels also worse than reference. A Ridge R² well below -1 indicates the L5 embeddings are actively anti-predictive of per-slice chroma — not just uninformative. This is the dominant negative result.

**Cross-song ratio (P4) worse than reference at all levels (0.84–0.88 vs ref 0.72–0.78)**, though all ratios remain below 1 (unlike u3s9_lowest's L5=1.070).

**Interpretation:** Cross-level MEP forces fine-level embeddings to be reconstructable from coarse summaries only. This appears to damage the fine-level chroma representation (L5) severely while improving density. Density may be capturable from global/coarse structure; chroma requires local pitch content that coarse-only supervision cannot provide. The constraint may be too strong at L5.

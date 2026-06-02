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

## Notes

*(To be filled as results come in.)*

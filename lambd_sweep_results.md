# Lambd Schedule Sweep Results

Backbone: d_uniform3, depths=[3,3,3,3,3,3], embed_dim=8.
All variants: n_skip_finest_levels=0 (LeJEPA on ALL 6 levels), lambda_fact=0.5,
max_shift_x=48, batch_size=300, 100 epochs.
lambd is now a per-level list [L0, L1, L2, L3, L4, L5] (L0=coarsest, L5=finest).

Reference: d_uniform3 baseline used scalar lambd=0.15 on L0-L3 only (n_skip=2).

## Run Registry

| # | Config | lambd schedule [L0..L5] | Hypothesis | Host | Run tag | Status |
|---|---|---|---|---|---|---|
| 1 | u3s1_min     | [0.15, 0.15, 0.15, 0.15, 0.10, 0.05] | Minimal change: extend tapering to L4-L5 | lecun | u3s1_min_XiKRDN | RUNNING |
| 2 | u3s2_gentle  | [0.15, 0.13, 0.12, 0.11, 0.10, 0.05] | Gentle descent from current L0 | tsrazer-ts-docker | u3s2_gentle_8ROxkz | RUNNING |
| 3 | u3s3_std     | [0.30, 0.25, 0.20, 0.15, 0.10, 0.05] | Standard descent, doubles L0 pressure | razer-docker | u3s3_std_i8Io5p | RUNNING |
| 4 | u3s4_steep   | [0.50, 0.35, 0.25, 0.15, 0.10, 0.05] | Aggressive descent, heavy coarse pressure | lecun | — | QUEUED |
| 5 | u3s5_inv1    | [0.10, 0.13, 0.15, 0.15, 0.10, 0.05] | Mild inversion: less SIGReg at L0 | tsrazer-ts-docker | — | QUEUED |
| 6 | u3s6_inv2    | [0.05, 0.10, 0.15, 0.20, 0.10, 0.05] | Strong inversion: minimum at L0 | razer-docker | — | QUEUED |
| 7 | u3s7_peak    | [0.10, 0.15, 0.20, 0.20, 0.10, 0.05] | Peak at L2-L3, low at extremes | lecun | — | QUEUED |
| 8 | u3s8_lowflat | [0.10, 0.10, 0.10, 0.10, 0.08, 0.05] | Low magnitude, nearly flat | tsrazer-ts-docker | — | QUEUED |
| 9 | u3s9_lowest  | [0.07, 0.07, 0.07, 0.07, 0.07, 0.05] | Lowest uniform: less is more extreme | razer-docker | — | QUEUED |

Machine chains (sequential within each):
- lecun:          s1_min → s4_steep → s7_peak
- tsrazer:        s2_gentle → s5_inv1 → s8_lowflat
- razer-docker:   s3_std → s6_inv2 → s9_lowest

## Probe 10 — Chroma R² (per-time-slice)

| Variant | lambd [L0..L5] | L0 | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|---|---|
| d_uniform3 (ref) | 0.15 scalar, skip L4-L5 | -0.004 | 0.045 | 0.188 | 0.415 | 0.585 | 0.736 |
| u3s1_min | [0.15,0.15,0.15,0.15,0.10,0.05] | | | | | | |
| u3s2_gentle | [0.15,0.13,0.12,0.11,0.10,0.05] | | | | | | |
| u3s3_std | [0.30,0.25,0.20,0.15,0.10,0.05] | | | | | | |
| u3s4_steep | [0.50,0.35,0.25,0.15,0.10,0.05] | | | | | | |
| u3s5_inv1 | [0.10,0.13,0.15,0.15,0.10,0.05] | | | | | | |
| u3s6_inv2 | [0.05,0.10,0.15,0.20,0.10,0.05] | | | | | | |
| u3s7_peak | [0.10,0.15,0.20,0.20,0.10,0.05] | | | | | | |
| u3s8_lowflat | [0.10,0.10,0.10,0.10,0.08,0.05] | | | | | | |
| u3s9_lowest | [0.07,0.07,0.07,0.07,0.07,0.05] | | | | | | |

## Probe 11 — Key Detection (24-class accuracy)

| Variant | L0 | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|---|
| d_uniform3 (ref) | 0.050 | 0.054 | 0.092 | 0.119 | 0.159 | 0.234 |
| u3s1_min | | | | | | |
| u3s2_gentle | | | | | | |
| u3s3_std | | | | | | |
| u3s4_steep | | | | | | |
| u3s5_inv1 | | | | | | |
| u3s6_inv2 | | | | | | |
| u3s7_peak | | | | | | |
| u3s8_lowflat | | | | | | |
| u3s9_lowest | | | | | | |

## Probe 5 — Density R²

| Variant | L0 | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|---|
| d_uniform3 (ref) | 0.267 | 0.347 | 0.558 | 0.624 | 0.735 | 0.907 |
| u3s1_min | | | | | | |
| u3s2_gentle | | | | | | |
| u3s3_std | | | | | | |
| u3s4_steep | | | | | | |
| u3s5_inv1 | | | | | | |
| u3s6_inv2 | | | | | | |
| u3s7_peak | | | | | | |
| u3s8_lowflat | | | | | | |
| u3s9_lowest | | | | | | |

## Probe 2 — Root Note (12-class accuracy)

| Variant | L0 | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|---|
| d_uniform3 (ref) | 0.100 | 0.116 | 0.161 | 0.218 | 0.230 | 0.301 |
| u3s1_min | | | | | | |
| u3s2_gentle | | | | | | |
| u3s3_std | | | | | | |
| u3s4_steep | | | | | | |
| u3s5_inv1 | | | | | | |
| u3s6_inv2 | | | | | | |
| u3s7_peak | | | | | | |
| u3s8_lowflat | | | | | | |
| u3s9_lowest | | | | | | |

## Probe 4 — Cross-Song Ratio (lower = better)

| Variant | L0 | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|---|
| d_uniform3 (ref) | 0.715 | 0.781 | 0.736 | 0.745 | 0.748 | 0.722 |
| u3s1_min | | | | | | |
| u3s2_gentle | | | | | | |
| u3s3_std | | | | | | |
| u3s4_steep | | | | | | |
| u3s5_inv1 | | | | | | |
| u3s6_inv2 | | | | | | |
| u3s7_peak | | | | | | |
| u3s8_lowflat | | | | | | |
| u3s9_lowest | | | | | | |

## Probe 1 — Chord Quality (accuracy)

| Variant | L0 | L1 | L2 | L3 | L4 | L5 | chance |
|---|---|---|---|---|---|---|---|
| d_uniform3 (ref) | 0.553 | 0.556 | 0.561 | 0.556 | 0.549 | 0.554 | 0.542 |
| u3s1_min | | | | | | | |
| u3s2_gentle | | | | | | | |
| u3s3_std | | | | | | | |
| u3s4_steep | | | | | | | |
| u3s5_inv1 | | | | | | | |
| u3s6_inv2 | | | | | | | |
| u3s7_peak | | | | | | | |
| u3s8_lowflat | | | | | | | |
| u3s9_lowest | | | | | | | |

## Best metric (encoder val loss)

| Variant | Best metric | Run tag | Host |
|---|---|---|---|
| d_uniform3 (ref) | 0.357557 | d_uniform3_fJumfJ | razer-docker |
| u3s1_min | | u3s1_min_XiKRDN | lecun |
| u3s2_gentle | | u3s2_gentle_8ROxkz | tsrazer-ts-docker |
| u3s3_std | | u3s3_std_i8Io5p | razer-docker |
| u3s4_steep | | | lecun |
| u3s5_inv1 | | | tsrazer-ts-docker |
| u3s6_inv2 | | | razer-docker |
| u3s7_peak | | | lecun |
| u3s8_lowflat | | | tsrazer-ts-docker |
| u3s9_lowest | | | razer-docker |

## Notes

*(To be filled as results come in.)*

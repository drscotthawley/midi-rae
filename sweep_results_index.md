# Sweep Results Index

## Active / Completed Sweeps

### 1. Depth Sweep
[[depth_sweep_results]]

Varied `depths` (transformer blocks per Swin level) across 8 configurations.
All variants: lambd=0.15 scalar, n_skip_finest=2, lambda_fact=0.5, max_shift_x=48, 100 epochs.
Backbone: embed_dim=8, depths vary, same base config otherwise.
**Status: COMPLETE.** All 8 variants trained and probed.

Key findings:
- **d_deep8 [2,2,2,8,2,2]** wins chroma (L5=0.746)
- **d_uniform3 [3,3,3,3,3,3]** wins key detection (L5=0.234) and root note (L5=0.301)
- Chroma collapse pattern: L2≥4 AND L3=6 → catastrophic L5 chroma failure
- Lower val loss ≠ better musical representations

---

### 2. Lambd Schedule Sweep
[[lambd_sweep_results]]

Varied per-level SIGReg strength (`lambd` list [L0..L5]) across 9 configurations.
All variants: d_uniform3 backbone (depths=[3,3,3,3,3,3]), n_skip_finest=0 (LeJEPA on all 6 levels),
lambda_fact=0.5, max_shift_x=48, 100 epochs.
**Status: IN PROGRESS.** Machines running round 1 of 3.

Machine chains:
- lecun: u3s1_min → u3s4_steep → u3s7_peak
- tsrazer-ts-docker: u3s2_gentle → u3s5_inv1 → u3s8_lowflat
- razer-docker: u3s3_std → u3s6_inv2 → u3s9_lowest

Key question: does a descending lambd schedule (high at L0, low at L5) or an inverted
schedule (low at L0, high at L3) produce better musical representations at coarse levels?

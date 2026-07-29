# midi-rae pixel-CFM demo / interactive test-bed

A small Gradio app that explores the **conditional generative capacity** of the
midi-rae pixel-space flow-matching model, and doubles as a research test-bed.
Runs locally on CPU or Apple-Silicon MPS.

## What it does

```
input piano-roll window (128×128 binary)
  → exp26 SwinEncoder            → per-level patch embeddings
  → per-level PCA + mean-pitch   → conditioning maps (mlcond), one per level L0..L5
  → [drop chosen levels]         ← the "level dropout" control (zero a level's map)
  → pixel-CFM UNet (otcfm)       → generates a NEW 128×128 piano roll from noise,
                                    conditioned on the maps (classifier-free guidance)
  → threshold → MIDI (played in-browser)
```

Unlike a decoder, this flows directly in **pixel space** — the UNet emits the piano-roll
image itself. Dropping conditioning levels shows what each scale contributes.

## Which model

The **pixel-CFM** stack (run `pixel_cfm_mlcdrop`, trained *with* per-level conditioning
dropout — so dropping conditioning levels is **in-distribution**):

| Piece | File in `demo/checkpoints/` |
|---|---|
| Pixel-CFM UNet (otcfm, step 84000, EMA) | `otcfm_mlcdrop_step84000.pt` |
| exp26 encoder (conditioning source) | `SwinEncoder_exp26_z1olvN_best.pt` |
| exp26 PCA (6 levels) | `pca_exp26/pca_L*_n*.pkl` |

Notes:
- The UNet is `UNetModelWrapperMLC`, **vendored** as `demo/unet_mlc_mlcdrop.py` (the
  Apr-21-2026 version that matches this checkpoint: dim-17 conditioning, no `middle_film`,
  input-block FiLM only). It's self-contained — it only imports the stable
  `torchcfm.models.unet.{fp16_util,nn}` helpers, so you do **not** need to manage which
  `unet_mlc.py` is in site-packages.
- Conditioning is PCA(exp26) per level — dims 17/31/37/23/7/3 for L0..L5, no mean-pitch
  column. `pcfm_infer.py` reproduces this.
- Note: for a 128px image with `channel_mult=[1,2,2,2,2]`, only L0 (global, via the time
  embedding) and L3/L4/L5 (spatial FiLM at feature-map resolutions 8/16/32) are actually
  consumed by the UNet; L1/L2 conditioning maps aren't referenced. So the L1/L2 drop
  checkboxes are effectively no-ops for this architecture (worth confirming empirically).

## Setup (Mac)

```bash
# from the repo root
pip install -e .                          # the midi_rae package
pip install -r demo/requirements.txt      # gradio, pretty_midi, midi-player, torchcfm, torchdyn
```

`pcfm_infer.py` imports `torchcfm.models.unet.unet_mlc.UNetModelWrapperMLC` — a **custom**
module (not in the pip `torchcfm`). Make sure the matching `unet_mlc.py` (the Apr-2026
version, with `middle_film`) is installed at
`<site-packages>/torchcfm/models/unet/unet_mlc.py`. On lecun it lives in the venv; copy
that file into your Mac's `torchcfm/models/unet/` if it isn't already there.

Checkpoints go in `demo/checkpoints/` (git-ignored). To fetch from lecun:

```bash
mkdir -p demo/checkpoints/pca_exp26
rsync -av lecun:/home/shawley/runs/midi-rae/pixel_cfm_Tkn0KT/checkpoints/otcfm_midi_weights_step_54000.pt demo/checkpoints/otcfm_Tkn0KT_step54000.pt
rsync -av lecun:/home/shawley/runs/midi-rae/exp26_z1olvN/checkpoints/SwinEncoder_exp26_z1olvN_best.pt demo/checkpoints/
rsync -av lecun:/home/shawley/datasets/POP909_pca_exp26/pca_L*_n*.pkl demo/checkpoints/pca_exp26/
```

## Run

```bash
python demo/app.py                    # http://localhost:7860
DEMO_PORT=8911 python demo/app.py     # custom port
DEMO_SHARE=1 python demo/app.py       # public *.gradio.live tunnel
```

Pick an example, choose a crop, pick a **device** (CPU/MPS radio — default is the fastest
available; switch to compare speeds), optionally drop levels, adjust **CFG strength**, and
hit **Generate**.

## Notes

- **Device radio**: CPU is always listed; CUDA/MPS appear when available; default is the
  fastest. Switching moves the models to the chosen device (a few seconds).
- `torch.compile` is not used (training-only; flaky on MPS). Integration is fixed-step
  Euler (`n_steps` slider) via a `torchdyn` NeuralODE, with classifier-free guidance.
- CPU generation is slow (~30s for 20 steps — the UNet is ~200M params); MPS is much faster.
- **Files**: `pcfm_infer.py` (encode → mlcond → CFG sample), `img2midi.py` (roll → MIDI,
  no control-toys dep), `app.py` (Gradio UI). `flow_infer.py` holds image helpers (and the
  older, non-converged e2e fine-flow backend, kept for reference). `_smoke_pcfm.py` /
  `_check_app.py` are dev sanity checks.
```

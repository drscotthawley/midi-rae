# Running the demo client-side (ONNX + WebGPU)

Researched 2026-07-29. **Shelved, not rejected** — the conclusion was "feasible and
well-suited, but not worth the time before the Aug 3 deadline". This area moves
fast, so re-check the specifics before acting on them.

## Why we were considering it

The actual goal was **an anonymised live demo for NeurIPS review** — an
alternative to posting a Hugging Face Space and hoping reviewers don't connect it
to the author. Client-side inference was a means to that end (a static page needs
no account and no server), not a goal in itself. Secondary benefits: no server GPU
cost, no cold starts, and "it runs on your own machine" is a nice story for a
creativity tool.

**Cheaper routes to the same goal** — consider these first:
- A static page of **pre-rendered examples** (original / masked / several
  regenerations, as audio + piano-roll images). Fully anonymous, no inference, and
  reviewers can listen immediately. We already have the pipeline to bulk-generate
  these.
- A **screen recording** of the paint-and-inpaint interaction, which covers the
  interactivity story without exposing a Space.

Either is an afternoon. The client-side port is a project.

### The pre-rendered route is mostly already built

- **Reuse the old Pictures-of-MIDI demo site as the template** — it already uses a
  standard layout with the html-midi-player JS, so it is largely a matter of
  swapping the text and the referenced MIDI files. Crucially it was **built to
  demo inpainting specifically**, so the before/after pairing and comparison
  layout already match this task; it is content substitution, not redesign. Using
  the same presentation as the earlier work also invites the direct comparison,
  which is where the argument should be made anyway.
- The generation side exists too: `img2midi.roll_to_midi_file()` produces the MIDI,
  and `app.py`'s `midi_player_html()` already wraps html-midi-player in an iframe
  `srcdoc`. A bulk script is mostly a loop over examples x mask geometries.
- Generating the examples is a GPU job — run it on hsrazer, not the Mac
  (see `feedback_gpu_workload_split` in memory; ask first).

**If the goal is anonymity, scrub the template properly.** An old site carries
identifying traces beyond the visible copy: author names in the footer, the domain
or repo it is served from, analytics IDs, favicon, ORCID or lab links in `<head>`,
and commit history if it is deployed from a personal repo. Replacing the body text
alone is not enough.

## What exists (as of mid-2026)

- **ONNX Runtime Web** has a WebGPU execution provider (since ORT 1.17, Feb 2024).
  Mature enough that people run Stable Diffusion in-browser; one demo fits SD +
  ControlNet in **under 250 MB**.
- **Transformers.js v3** wraps that same WebGPU EP — same ONNX graph targets
  WebGPU or WASM without reconversion.
- **WebGPU vs WASM is roughly 10-15x.** For a 43 M-parameter UNet, WebGPU is not
  optional; a WASM fallback would be painful.

### Gradio does NOT give this to us for free

Gradio-Lite runs Gradio in **Pyodide** — Python compiled to WASM. No PyTorch
there, and no WebGPU access from Python. The documented pattern is Gradio-Lite for
the UI plus **Transformers.js doing inference in JavaScript**. So the inference
path has to be written in JS regardless; Gradio only saves the widget layer.

## Why our model is a good fit

Smaller than what people already ship in browsers:

| piece | params / size |
|---|---|
| pixel-CFM UNet | 43.56 M -> ~174 MB fp32, **~87 MB fp16** |
| Swin encoder | 10 MB |
| XMEP | 24 MB |
| PCA transforms | kilobytes |

- Sampling is **10 Euler steps x 2 (CFG) = 20 UNet evaluations** on a 128x128x1
  image. Stable Diffusion does comparable step counts on a bigger network. Low
  seconds on WebGPU seems realistic.
- **Only the networks need ONNX.** The Euler loop, CFG blend, PCA projection (a
  25xD matmul), the mask, and the PnP-Flow / hard-replace constraints are all
  elementwise or trivial linear algebra — straightforward to write in JS.
- The brush is already client-side.

## The sharp edge: exporting the Swin encoder

Shifted-window attention uses `torch.roll`, which has a long history of ONNX
export trouble, and even at opset 17 there are reported Swin breakages. Workarounds
exist (replace `roll` with slice + concat). Budget a day, and verify numerical
parity against PyTorch afterwards — fp16 on WebGPU will not match fp32 exactly, so
check output quality, not just that it runs.

`SwinMaskedEmbeddingPredictor` uses `nn.MultiheadAttention`, which exports but can
need care.

### Staging that avoids the problem entirely

For a demo over a **fixed set of example songs**, the encoder isn't needed in the
browser at all:

1. Precompute each example's per-level embeddings server-side; ship them as a
   small binary alongside the models.
2. Client runs **XMEP fill -> PCA -> UNet sampling**.

That gives the full interactive inpainting loop client-side with no Swin export.
Arbitrary user-uploaded MIDI would need the encoder, and can come later.

## Caveats

- **Safari.** WebGPU is solid in Chrome/Edge, experimental in Safari, behind a flag
  in Firefox. Relevant if demoing from a Mac.
- **First-load download** of ~90-170 MB (cached afterwards).
- Numerical parity needs checking, not assuming.

## Sources

- [ONNX Runtime Web + WebGPU announcement](https://opensource.microsoft.com/blog/2024/02/29/onnx-runtime-web-unleashes-generative-ai-in-the-browser-using-webgpu/)
- [ORT WebGPU docs](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html)
- [Stable Diffusion in 250 MB in-browser](https://www.leebutterman.com/2025/03/01/running-stable-diffusion-in-250-megabytes-in-onnx-and-webgpu.html)
- [Gradio-Lite + Transformers.js](https://huggingface.co/blog/samihalawa/chrome-ai-agents-transformersjs-webgpu-gradiolite)
- [WebGPU vs WASM benchmarks](https://www.sitepoint.com/webgpu-vs-webasm-transformers-js/)
- [torch.roll ONNX export issue](https://github.com/pytorch/pytorch/issues/56355)
- [Swin opset 17 breakage](https://github.com/opencv/opencv/issues/28183)

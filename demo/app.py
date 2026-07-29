"""Gradio demo / interactive test-bed for the midi-rae pixel-CFM generative model.

Pick an example piano-roll window; the exp26 encoder turns it into per-level PCA
conditioning maps, and a pixel-space conditional flow-matching UNet (otcfm) generates
a *new* 128x128 piano roll from noise, conditioned on those maps. Toggle which
conditioning levels are active to hear/see what each scale contributes.

Run:  python demo/app.py [--port 9000] [--share] [--host 127.0.0.1]
      CLI flags win; otherwise DEMO_PORT / DEMO_SHARE / DEMO_HOST env vars apply.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import gradio as gr
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import pcfm_infer
import img2midi
import flow_infer  # image helpers
import guided_sample as gs

DISP = 512  # editor display resolution (4x the 128 roll) so the brush is usable

EXAMPLES_DIR = HERE / "examples"


def get_demo():
    return pcfm_infer.get_demo()


def midi_player_html(mid_path, title=""):
    """Return the html-midi-player, wrapped in an <iframe srcdoc> so its <script> runs.

    Gradio's gr.HTML injects markup via innerHTML, which does NOT execute <script> tags,
    so the raw MIDIPlayer.html (script + <midi-player> web component) renders blank. An
    iframe's srcdoc is parsed as its own document and DOES run scripts (same reason the
    player works in Jupyter/W&B, which also iframe-wrap it)."""
    import html as _html
    try:
        from midi_player import MIDIPlayer
        from midi_player.stylers import dark
        inner = MIDIPlayer(mid_path, 300, title=title, styler=dark).html
        srcdoc = _html.escape(inner, quote=True)
        return (f'<iframe srcdoc="{srcdoc}" width="100%" height="400" '
                f'style="border:none;overflow:hidden;" loading="lazy"></iframe>')
    except Exception as e:
        return (f"<div style='padding:8px;font-family:sans-serif'>MIDI ready: {Path(mid_path).name}"
                f"<br><small>player unavailable: {type(e).__name__}: {e}</small></div>")


def roll_to_display(roll, color=(80, 255, 120)):
    mask = (np.asarray(roll) > 0.5).astype(np.uint8)
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for c in range(3):
        rgb[:, :, c] = mask * color[c]
    return Image.fromarray(rgb, "RGB").resize((512, 512), Image.NEAREST)


# ---------------------------------------------------------------- inpaint mask
def _downsample_any(painted, out=128):
    """Downsample a boolean paint map to (out,out); ANY painted pixel in a block
    marks the block as hole (so thin strokes survive)."""
    H, W = painted.shape
    if H == out and W == out:
        return painted
    if H % out == 0 and W % out == 0:
        fh, fw = H // out, W // out
        return painted[:out * fh, :out * fw].reshape(out, fh, out, fw).any(axis=(1, 3))
    im = Image.fromarray((painted.astype(np.uint8) * 255)).resize((out, out), Image.BOX)
    return np.asarray(im) > 0


def extract_hole_mask(editor_value, out_hw=128):
    """gr.ImageEditor dict -> (out,out) bool, True where the user PAINTED (=hole).

    Uses the alpha of the drawn `layers` (the strokes on transparent RGBA), not
    the flattened `composite` -- that's exactly 'where the brush went'."""
    if not isinstance(editor_value, dict):
        return np.zeros((out_hw, out_hw), bool)
    painted = None
    for ly in (editor_value.get("layers") or []):
        ly = np.asarray(ly)
        if ly.ndim == 3 and ly.shape[-1] == 4:
            a = ly[..., 3] > 10                                   # alpha = painted
        elif ly.ndim == 3:
            a = ly.any(axis=-1)
        else:
            a = ly > 0
        painted = a if painted is None else (painted | a)
    if painted is None or not painted.any():
        return np.zeros((out_hw, out_hw), bool)
    return _downsample_any(painted, out_hw)


def load_window_for_paint(example_name, crop_x):
    """Render the chosen 128-window as a DISP-sized image for the editor, and
    stash the exact (1,1,128,128) input tensor in state for the inpaint run."""
    path = EXAMPLES_DIR / example_name
    img = flow_infer.image_to_binary_tensor(path, crop_x=int(crop_x))     # (1,1,128,128) [0,1]
    bg = roll_to_display(img[0, 0].numpy(), (120, 180, 255))              # PIL DISPxDISP
    return np.asarray(bg.convert("RGB")), img


def run_inpaint(editor_value, input_state, method, n_steps, cfg_strength, seed, device,
                latent_fill=True, dilate=0):
    if input_state is None:
        return None, "", "Load a window into the canvas first (▶ Load window)."
    hole = extract_hole_mask(editor_value, pcfm_infer.IMAGE_SIZE)          # (128,128) bool True=hole
    if not hole.any():
        return None, "", "Paint over the region you want the model to redraw, then Run."

    demo = get_demo(); demo.set_device(device)
    img = input_state                                                     # (1,1,128,128) [0,1]
    img_holed = img.clone()
    hole_t = torch.from_numpy(hole)
    img_holed[..., hole_t] = 0.0                                          # DELETE notes under the brush
    x_known = img_holed * 2 - 1                                           # model space [-1,1]
    mask_t = torch.from_numpy((~hole).astype("float32")).view(1, 1, *hole.shape)  # 1=known,0=hole

    # Conditioning. Blanked pixels read as "silence here" to the encoder, and the flow
    # renders that faithfully -- so the XMEP repredicts the hole's embeddings from
    # context instead. mlcond=None falls back to the plain (blanked) encoding.
    t0 = time.time()
    mlcond, note = None, "blanked-cond"
    if latent_fill:
        mlcond, st = demo.encode_to_mlcond_filled(img_holed, hole, dilate=int(dilate),
                                                  return_stats=True)
        n_tok = sum(s[1] for s in st)
        note = f"XMEP fill (dilate={int(dilate)}, {n_tok} tokens repredicted)"

    if method == "hard":
        gen_th = torch.Generator().manual_seed(int(seed))
        x0n = torch.randn(1, 1, pcfm_infer.IMAGE_SIZE, pcfm_infer.IMAGE_SIZE, generator=gen_th)
        gen = gs.guided_generate(demo, img_holed, n_steps=int(n_steps), seed=int(seed),
                                 cfg_strength=float(cfg_strength), device=device, mlcond=mlcond,
                                 project=gs.make_inpaint_project(x_known, mask_t, x0n))
    elif method == "soft":
        guide = gs.make_soft_inpaint_guidance(x_known, mask_t, eta=1.0, t_min=0.2)
        gen = gs.guided_generate(demo, img_holed, n_steps=int(n_steps), seed=int(seed),
                                 cfg_strength=float(cfg_strength), device=device, mlcond=mlcond,
                                 guide_fn=guide, init_known=x_known, init_t0=0.2)
    else:  # pnpflow
        grad = gs.make_inpaint_grad(x_known, mask_t)
        # num_avg=1: averaging over independent draws washes out the hole (nothing pins
        # it, so the draws disagree and average toward grey). Measured 2.5x less content
        # at avg4, which also hid the XMEP fill's benefit entirely.
        gen = gs.pnpflow_generate(demo, img_holed, grad, n_steps=int(n_steps), seed=int(seed),
                                  cfg_strength=float(cfg_strength), device=device, mlcond=mlcond,
                                  alpha=0.5, strength=1.0, num_avg=1)
    dt = time.time() - t0

    gen_mid = img2midi.roll_to_midi_file(gen)
    frac = float(hole.mean())
    in_hole = int((gen[hole] > 0.5).sum())
    status = (f"**{method}** · {note} · {device} · {dt:.1f}s ({n_steps} steps, CFG {cfg_strength}). "
              f"Repainted {frac*100:.1f}% of the window ({int(hole.sum())} of {hole.size} cells); "
              f"{in_hole} note-pixels generated INSIDE the hole "
              f"(density {(gen[hole] > 0.5).mean():.3f}).")
    return roll_to_display(gen, (80, 255, 120)), midi_player_html(gen_mid, "inpainted"), status


def on_example_change(example_name):
    path = EXAMPLES_DIR / example_name
    arr = flow_infer._load_binary_array(path)
    max_x = max(0, arr.shape[1] - flow_infer.IMAGE_SIZE)
    return gr.update(minimum=0, maximum=max_x, value=flow_infer.best_crop_x(path), step=1)


def run(example_name, crop_x, device, solver, n_steps, cfg_strength, seed,
        d0, d1, d2, d3, d4, d5):
    demo = get_demo()
    demo.set_device(device)
    path = EXAMPLES_DIR / example_name
    img = flow_infer.image_to_binary_tensor(path, crop_x=int(crop_x))
    drop = [i for i, d in enumerate((d0, d1, d2, d3, d4, d5)) if d]

    t0 = time.time()
    gen = demo.generate(img, drop_levels=drop, n_steps=int(n_steps), seed=int(seed),
                        cfg_strength=float(cfg_strength), device=device, solver=solver)
    dt = time.time() - t0

    input_roll = img[0, 0].numpy()
    gen_mid = img2midi.roll_to_midi_file(gen)
    input_mid = img2midi.roll_to_midi_file(input_roll)
    dropped = ", ".join(f"L{i}" for i in drop) if drop else "none (full conditioning)"
    status = (f"Device **{device}** · {solver} · {dt:.1f}s ({n_steps} steps, CFG {cfg_strength}). "
              f"Dropped: {dropped}. Generated {int((gen>0.5).sum())} note-pixels "
              f"(input {int(input_roll.sum())}).")
    return (roll_to_display(input_roll, (120, 180, 255)),
            roll_to_display(gen, (80, 255, 120)),
            midi_player_html(input_mid, "input"),
            midi_player_html(gen_mid, "generated"),
            status)


def build_ui():
    examples = sorted(p.name for p in EXAMPLES_DIR.glob("*.png"))
    default_example = examples[0] if examples else None
    if default_example:
        _arr = flow_infer._load_binary_array(EXAMPLES_DIR / default_example)
        _max_x = max(0, _arr.shape[1] - flow_infer.IMAGE_SIZE)
        _default_x = flow_infer.best_crop_x(EXAMPLES_DIR / default_example)
    else:
        _max_x, _default_x = 3000, 0

    devices = pcfm_infer.available_devices()

    with gr.Blocks(title="midi-rae pixel-CFM test-bed") as ui:
        gr.Markdown(
            "# midi-rae pixel-CFM — interactive generative test-bed\n"
            "The exp26 encoder turns an input piano-roll window into per-level PCA "
            "**conditioning maps**; a pixel-space flow-matching UNet then **generates a new "
            "piano roll** from noise, conditioned on those maps (classifier-free guidance). "
            "Drop conditioning levels to see what each scale contributes.\n\n"
            "*Model: `pixel_cfm_Tkn0KT` (full-conditioning run). Dropping levels is somewhat "
            "out-of-distribution for this checkpoint — a dropout-trained variant is a planned swap.*"
        )
        with gr.Tabs():
            # ---------------------------------------------------------- Generate
            with gr.Tab("Generate"):
                with gr.Row():
                    with gr.Column(scale=1):
                        example = gr.Dropdown(examples, value=default_example, label="Example song")
                        crop_x = gr.Slider(0, _max_x, value=_default_x, step=1, label="Crop position (time)")
                        device = gr.Radio(devices, value=devices[0], label="Device",
                                          info="fastest available is default; switch to compare speeds")
                        solver = gr.Dropdown([("Euler", "euler"), ("RK4", "rk4")], value="euler",
                                             label="Integration method",
                                             info="RK4 = 4 model evals/step (more accurate, slower)")
                        n_steps = gr.Slider(1, 60, value=10, step=1, label="Flow steps")
                        cfg = gr.Slider(0.0, 12.0, value=4.0, step=0.1, label="CFG strength",
                                        info="0 = unconditional · 1 = plain conditional · >1 = amplified guidance")
                        seed = gr.Number(value=0, precision=0, label="Seed")
                        gr.Markdown("**Drop conditioning levels** (zero = mean / no info):")
                        with gr.Row():
                            d0 = gr.Checkbox(False, label="L0")
                            d1 = gr.Checkbox(False, label="L1")
                            d2 = gr.Checkbox(False, label="L2")
                            d3 = gr.Checkbox(False, label="L3")
                            d4 = gr.Checkbox(False, label="L4")
                            d5 = gr.Checkbox(False, label="L5")
                        go = gr.Button("Generate", variant="primary")
                        status = gr.Markdown()
                    with gr.Column(scale=2):
                        with gr.Row():
                            in_img = gr.Image(label="Input window (conditioning source)", type="pil")
                            gen_img = gr.Image(label="Generated", type="pil")
                        gr.Markdown("### MIDI players")
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("**Input**")
                                in_player = gr.HTML()
                            with gr.Column(scale=1):
                                gr.Markdown("**Generated**")
                                gen_player = gr.HTML()

            # ----------------------------------------------------------- Inpaint
            with gr.Tab("Inpaint — paint the mask"):
                gr.Markdown(
                    "Load a window, then **paint over the region you want redrawn**. The brush "
                    "both *deletes* the notes under it (so the model can't just copy them) and "
                    "defines the inpainting mask. The flow runs in **pixel space**, so the mask "
                    "lives directly on the image — no spatially-aligned latent needed."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        ip_example = gr.Dropdown(examples, value=default_example, label="Example song")
                        ip_crop = gr.Slider(0, _max_x, value=_default_x, step=1, label="Crop position (time)")
                        ip_load = gr.Button("▶ Load window into canvas")
                        ip_method = gr.Dropdown(
                            [("PnP-Flow", "pnpflow"), ("Hard replace", "hard"),
                             ("Soft guidance ((1−t)/t)", "soft")], value="pnpflow",
                            label="Inpaint method",
                            info="PnP-Flow + XMEP fill generates the most content in the hole")
                        ip_fill = gr.Checkbox(value=True, label="XMEP latent fill",
                                              info="Repredict the hole's embeddings from context "
                                                   "instead of conditioning on a blanked region")
                        ip_dilate = gr.Slider(0, 3, value=0, step=1, label="Token-mask dilation",
                                              info="Also mask neighbouring tokens. Measured to "
                                                   "HURT (80%→57%→33% of erased density at 0/1/2) "
                                                   "— context matters more than contamination")
                        ip_device = gr.Radio(devices, value=devices[0], label="Device")
                        ip_steps = gr.Slider(1, 60, value=10, step=1, label="Flow steps",
                                             info="MORE steps hurts here — the hole empties out")
                        ip_cfg = gr.Slider(0.0, 12.0, value=0.8, step=0.1, label="CFG strength",
                                           info="~0.8 is the sweet spot: higher blanks the hole, "
                                                "near 0 gives noise")
                        ip_seed = gr.Number(value=0, precision=0, label="Seed")
                        ip_go = gr.Button("Inpaint", variant="primary")
                        ip_status = gr.Markdown()
                    with gr.Column(scale=2):
                        with gr.Row():
                            ip_canvas = gr.ImageEditor(
                                label="Paint the mask (brush = region to redraw)",
                                type="numpy", image_mode="RGB",
                                height=DISP, width=DISP,
                                brush=gr.Brush(colors=["#FF3B30"], default_size=16,
                                               color_mode="fixed"),
                                sources=(), interactive=True)
                            ip_out = gr.Image(label="Inpainted result", type="pil")
                        ip_player = gr.HTML()
                ip_state = gr.State(None)

        example.change(on_example_change, inputs=example, outputs=crop_x)
        go.click(run,
                 inputs=[example, crop_x, device, solver, n_steps, cfg, seed, d0, d1, d2, d3, d4, d5],
                 outputs=[in_img, gen_img, in_player, gen_player, status])

        ip_example.change(on_example_change, inputs=ip_example, outputs=ip_crop)
        ip_load.click(load_window_for_paint, inputs=[ip_example, ip_crop],
                      outputs=[ip_canvas, ip_state])
        ip_go.click(run_inpaint,
                    inputs=[ip_canvas, ip_state, ip_method, ip_steps, ip_cfg, ip_seed, ip_device,
                            ip_fill, ip_dilate],
                    outputs=[ip_out, ip_player, ip_status])
    return ui


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=int(os.environ.get("DEMO_PORT", "7860")),
                    help="port to serve on (overrides DEMO_PORT; default 7860)")
    ap.add_argument("--share", action="store_true", default=os.environ.get("DEMO_SHARE", "0") == "1",
                    help="create a public gradio link (overrides DEMO_SHARE)")
    ap.add_argument("--host", default=os.environ.get("DEMO_HOST", "0.0.0.0"),
                    help="interface to bind (overrides DEMO_HOST; default 0.0.0.0)")
    args = ap.parse_args()
    build_ui().launch(server_name=args.host, server_port=args.port, share=args.share)

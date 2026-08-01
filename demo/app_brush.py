"""Generation with spatially localised conditioning dropout.

This is the Generate panel from app.py with one change: the level dropout no longer
applies to the whole window, it applies only where you paint. Paint nothing and the
output is the input. Paint everything and it behaves exactly like the old Generate.

  levels   which scales the brush drops (L5 is finest, ~4 steps x 4 semitones per
           token, and carries note detail; coarse levels carry phrase shape)
  P        probability each brushed patch is dropped -- brush strength
  guidance how hard the flow follows the conditioning that survives

Outside the painted region the result is copied from the input, so untouched music is
preserved exactly rather than approximately.

Run:  python demo/app_brush.py [--port 9002]
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import gradio as gr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import pcfm_infer
import img2midi
import flow_infer
import guided_sample as gs
# Helpers only; app.py builds its UI inside build_ui(), so importing is side-effect free.
from app import (EXAMPLES_DIR, get_demo, midi_player_html, roll_to_display,
                 extract_hole_mask, load_song, click_strip)


def run_brush(editor_value, input_state, n_steps, cfg_log, seed, device, solver,
              drop_p, d0, d1, d2, d3, d4, d5):
    if input_state is None:
        gr.Warning("Pick a crop above first.")
        return None, "", "Pick a crop above first."
    hole = extract_hole_mask(editor_value, pcfm_infer.IMAGE_SIZE)
    if not hole.any():
        gr.Warning("Nothing painted, so nothing changes.")
        return None, "", "Paint where you want the model to rewrite."

    cfg_strength = 10.0 ** float(cfg_log)          # slider carries log10(guidance)
    levels = [i for i, on in enumerate((d0, d1, d2, d3, d4, d5)) if on]
    if not levels:
        gr.Warning("No levels ticked, so the brush drops nothing.")
        return None, "", "Tick at least one level for the brush to act on."

    demo = get_demo(); demo.set_device(device)
    img = input_state                                   # full input; nothing erased
    t0 = time.time()
    # Conditioning comes from the WHOLE input, then the brush zeroes patches inside the
    # painted region at the ticked levels. Pixels are never blanked: this is dropout,
    # not inpainting, so the encoder still sees the real music everywhere.
    mlcond, n_zeroed = demo.encode_to_mlcond_masked(
        img, hole, levels=levels, p=float(drop_p), seed=int(seed), skip_l0=False)

    gen = gs.guided_generate(demo, img, n_steps=int(n_steps), seed=int(seed),
                             cfg_strength=cfg_strength, device=device,
                             mlcond=mlcond, solver=solver)
    dt = time.time() - t0

    orig = img[0, 0].numpy()
    gen = np.where(hole, gen, orig)                     # untouched region is exact

    mid = img2midi.roll_to_midi_file(gen)
    frac = float(hole.mean())
    delta = int((gen[hole] > 0.5).sum()) - int((orig[hole] > 0.5).sum())
    status = (f"levels {levels} · P={float(drop_p):.2f} · guidance {cfg_strength:.2f} · "
              f"{solver} {n_steps} steps · {device} · {dt:.1f}s. "
              f"Brushed {frac*100:.1f}% of the window; patches dropped per level "
              f"{n_zeroed}; note-pixel change inside the brush {delta:+d}.")
    return roll_to_display(gen, (80, 255, 120)), midi_player_html(mid, "generated"), status


def build_ui():
    examples = sorted(p.name for p in EXAMPLES_DIR.glob("*.png"))
    default_example = examples[0] if examples else None
    if default_example:
        _arr = flow_infer._load_binary_array(EXAMPLES_DIR / default_example)
        _max_x = max(0, _arr.shape[1] - flow_infer.IMAGE_SIZE)
    else:
        _max_x = 3000
    devices = pcfm_infer.available_devices()

    with gr.Blocks(title="midi-rae brush") as ui:
        gr.Markdown("# Brush\nGeneration with the dropout confined to what you paint.")

        gr.Markdown("### Input selection")
        with gr.Group():
            example = gr.Dropdown(examples, value=default_example, label="Example song",
                                  scale=0, container=True)
            strip = gr.Image(label="Click to crop", type="numpy", interactive=False,
                             show_label=True, container=False)
            crop_note = gr.Markdown()
            crop_x = gr.Slider(0, _max_x, value=0, step=1, visible=False)

        with gr.Row():
            in_img = gr.Image(label="Input window", type="pil")
            in_player = gr.HTML()
        state = gr.State()

        gr.Markdown("## Brush")
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Row():
                    d0 = gr.Checkbox(False, label="L0")
                    d1 = gr.Checkbox(False, label="L1")
                    d2 = gr.Checkbox(False, label="L2")
                    d3 = gr.Checkbox(False, label="L3")
                    d4 = gr.Checkbox(False, label="L4")
                    d5 = gr.Checkbox(True, label="L5")
                    drop_p = gr.Slider(0.0, 1.0, value=0.75, step=0.05, label="P",
                                       container=False, scale=1, min_width=90)
                solver = gr.Dropdown([("Euler", "euler"), ("RK4", "rk4")], value="euler",
                                     label="Integration method")
                n_steps = gr.Slider(1, 120, value=10, step=1, label="Flow steps")
                cfg = gr.Slider(-2.0, 1.1, value=0.0, step=0.02,
                                label="Guidance strength (log₁₀)")
                cfg_readout = gr.Markdown("**guidance = 1.00**")
                cfg.change(lambda v: f"**guidance = {10.0 ** float(v):.2f}**",
                           inputs=cfg, outputs=cfg_readout)
                seed = gr.Number(value=0, precision=0, label="Seed")
                device = gr.Dropdown(devices, value=devices[0], label="Device")
                go = gr.Button("Generate", variant="primary")
            with gr.Column(scale=2):
                canvas = gr.ImageEditor(label="Paint where to rewrite", type="numpy",
                                        image_mode="RGB", height=520)
        with gr.Row():
            out_img = gr.Image(label="Result", type="pil")
            out_player = gr.HTML()
        status = gr.Markdown()

        # load_song / click_strip return, in order:
        #   strip rgb, crop_x, crop preview, note text, canvas image, input tensor, player
        outs = [strip, crop_x, in_img, crop_note, canvas, state, in_player]
        example.change(load_song, inputs=example, outputs=outs)
        strip.select(click_strip, inputs=example, outputs=outs)
        ui.load(load_song, inputs=example, outputs=outs)
        go.click(run_brush,
                 inputs=[canvas, state, n_steps, cfg, seed, device, solver, drop_p,
                         d0, d1, d2, d3, d4, d5],
                 outputs=[out_img, out_player, status])
    return ui


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=int(os.environ.get("DEMO_PORT", "9002")))
    ap.add_argument("--share", action="store_true",
                    default=os.environ.get("DEMO_SHARE", "0") == "1")
    ap.add_argument("--host", default=os.environ.get("DEMO_HOST", "0.0.0.0"))
    args = ap.parse_args()
    build_ui().launch(server_name=args.host, server_port=args.port, share=args.share)

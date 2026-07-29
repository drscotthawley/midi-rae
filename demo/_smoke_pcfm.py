"""CPU smoke test for the pixel-CFM backend. Run: PYTHONPATH=. python demo/_smoke_pcfm.py"""
import sys, time
from pathlib import Path
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))

import pcfm_infer, img2midi, flow_infer
import pretty_midi


def main():
    print("available_devices:", pcfm_infer.available_devices(), "default:", pcfm_infer.default_device())
    demo = pcfm_infer.PixelCFMDemo(device="cpu").load()

    ex = sorted((HERE / "examples").glob("*.png"))
    cx = flow_infer.best_crop_x(ex[0])
    img = flow_infer.image_to_binary_tensor(ex[0], crop_x=cx)
    print(f"input {ex[0].name} crop_x={cx} note_px_in={int(img.sum())}")

    t0 = time.time()
    roll = demo.generate(img, drop_levels=(), n_steps=20, seed=0, cfg_strength=4.0)
    dt = time.time() - t0
    dens = (roll > 0.5).mean()
    print(f"generated: shape={roll.shape} note_px={int((roll>0.5).sum())} density={dens:.3f} "
          f"min={roll.min():.2f} max={roll.max():.2f}  ({dt:.1f}s on cpu, 20 steps)")

    mid = img2midi.roll_to_midi_file(roll)
    pm = pretty_midi.PrettyMIDI(mid)
    nn = sum(len(i.notes) for i in pm.instruments)
    print(f"MIDI: n_notes={nn} end={pm.get_end_time():.2f}s -> {mid}")

    roll_dropL0 = demo.generate(img, drop_levels=(0,), n_steps=20, seed=0, cfg_strength=4.0)
    diff = float(np.abs(roll - roll_dropL0).sum())
    print(f"dropL0 density={(roll_dropL0>0.5).mean():.3f}  |full-dropL0| px_diff={diff}")

    # save generated PNG for visual inspection
    out = HERE / "_pcfm_gen_sample.png"
    Image.fromarray(((roll > 0.5) * 255).astype(np.uint8)).save(out)
    print("saved", out)
    assert nn > 0, "no notes generated"
    assert 0.01 < dens < 0.20, f"density {dens:.3f} out of musical range (expect sparse)"
    print("\nPCFM SMOKE TEST PASSED")


if __name__ == "__main__":
    main()

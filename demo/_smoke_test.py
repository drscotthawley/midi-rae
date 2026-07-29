"""CPU smoke test for the demo inference harness. Run:
    PYTHONPATH=. python demo/_smoke_test.py
"""
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))            # so `import flow_infer` works
sys.path.insert(0, str(HERE.parent))     # so `import midi_rae` works

import flow_infer
import img2midi
import pretty_midi


def main():
    dev = "cpu"
    demo = flow_infer.FlowDemo(device=dev).load()

    ex = sorted((HERE / "examples").glob("*.png"))
    print(f"examples found: {[p.name for p in ex]}")
    assert ex, "no example PNGs"
    from PIL import Image
    im = Image.open(ex[0])
    print(f"example[0] {ex[0].name} size(WxH)={im.size}")

    cx = flow_infer.best_crop_x(ex[0])
    img = flow_infer.image_to_binary_tensor(ex[0], crop_x=cx)
    print(f"best_crop_x={cx}  input tensor shape={tuple(img.shape)} note_px_in={int(img.sum())}")

    # 0) reconstruction sanity: encode -> real PCA -> decode (no fine flow)
    recon = demo.reconstruct(img)
    inp = img[0, 0].numpy()
    # input is (pitch-up) binary; recon is decoder/image orientation. Compare note counts + IoU-ish.
    inter = float(((recon > 0.5) & (inp > 0.5)).sum())
    union = float(((recon > 0.5) | (inp > 0.5)).sum())
    print(f"RECON note_px={int((recon>0.5).sum())} vs input note_px={int((inp>0.5).sum())} "
          f"overlap(inter/union)={inter/max(union,1):.3f}")

    # 1) generate with all levels kept
    roll_full = demo.generate(img, drop_levels=(), n_steps=20, seed=0)
    print(f"roll_full shape={roll_full.shape} note_px={int((roll_full>0.5).sum())} "
          f"min={roll_full.min():.2f} max={roll_full.max():.2f}")

    # 2) to MIDI
    mid_path = img2midi.roll_to_midi_file(roll_full)
    pm = pretty_midi.PrettyMIDI(mid_path)
    n_notes = sum(len(i.notes) for i in pm.instruments)
    print(f"MIDI written: {mid_path}  n_notes={n_notes}  end_time={pm.get_end_time():.2f}s")

    # 3) drop L0 -> should differ from full (same seed)
    roll_dropL0 = demo.generate(img, drop_levels=(0,), n_steps=20, seed=0)
    diff = float(np.abs(roll_full - roll_dropL0).sum())
    print(f"roll_dropL0 note_px={int((roll_dropL0>0.5).sum())}  |full-dropL0| px_diff={diff}")

    assert n_notes > 0, "generated MIDI has no notes"
    assert diff > 0, "dropping L0 produced an identical roll (conditioning not wired?)"
    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()

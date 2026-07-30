"""Which checkpoint went wrong, and how?

Sweeps c55 flow checkpoints on identical windows with identical settings.

  note count   - the speckle detector. Real windows have ~45 note events; speckle
                 shatters the roll into hundreds of 1-pixel "notes".
  density      - generated note density / real
  note F1      - pitch-exact, onset within 1 step
  pixel F1     - strict overlap

A sharp break at one checkpoint suggests a bad save; a monotone slide suggests the
training itself degraded.

Run on a GPU host: ~/envs/midi-rae/bin/python _ckpt_compare.py
"""
import importlib
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))

import flow_infer
from _recon_note_f1 import notes, note_f1

CKPTS = ["otcfm_midi_weights_step_12000.pt", "otcfm_midi_weights_step_14000.pt",
         "otcfm_midi_weights_step_16000.pt", "otcfm_midi_weights_step_18000.pt",
         "otcfm_midi_weights_step_20000.pt"]
STEPS, CFG, N = 20, 4.0, 20
POP909 = Path("/home/shawley/datasets/POP909_images_basic")


def pix_f1(pred, real):
    tp = float((pred & real).sum()); fp = float((pred & ~real).sum()); fn = float((~pred & real).sum())
    p = tp / max(tp + fp, 1e-9); r = tp / max(tp + fn, 1e-9)
    return 2 * p * r / max(p + r, 1e-9)


def main():
    pngs = sorted(POP909.glob("*.png"))
    wins = []
    for p in pngs:
        img = flow_infer.image_to_binary_tensor(p, crop_x=int(flow_infer.best_crop_x(p)))
        if float((img > 0.5).float().mean()) >= 0.01:
            wins.append(img)
        if len(wins) >= N:
            break
    real_n = np.mean([len(notes(w[0, 0].numpy() > 0.5)) for w in wins])
    print(f"{len(wins)} windows, steps={STEPS} cfg={CFG}")
    print(f"real: {real_n:.0f} note events per window\n")
    print(f"{'checkpoint':>10s} {'gen notes':>10s} {'density':>8s} {'note F1':>8s} {'pixel F1':>9s}")

    for name in CKPTS:
        if not (HERE / "checkpoints" / "c55" / name).exists():
            print(f"{name.split('_step_')[1][:-3]:>10s}  (missing)")
            continue
        os.environ["MIDIRAE_CFM_CKPT"] = name
        if "pcfm_infer" in sys.modules:
            del sys.modules["pcfm_infer"]
        import pcfm_infer as pi
        importlib.reload(pi)
        demo = pi.PixelCFMDemo(device="cuda").load()

        ns, ds, nf, pf = [], [], [], []
        for w in wins:
            real = w[0, 0].numpy() > 0.5
            gen = demo.generate(w, n_steps=STEPS, seed=0, cfg_strength=CFG,
                                device="cuda") > 0.5
            ns.append(len(notes(gen)))
            ds.append(gen.mean() / max(real.mean(), 1e-9))
            nf.append(note_f1(notes(gen), notes(real), 1)[2])
            pf.append(pix_f1(gen, real))
        step = name.split("_step_")[1][:-3]
        print(f"{step:>10s} {np.mean(ns):10.0f} {np.mean(ds):8.2f} {np.mean(nf):8.3f} "
              f"{np.mean(pf):9.3f}")


if __name__ == "__main__":
    main()

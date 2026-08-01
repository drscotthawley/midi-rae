"""One summary per model set, on identical windows.

Reports everything we have been arguing about in one place:

  gen notes     - note events per window; real is ~45. Speckle shatters the roll
                  into hundreds of 1-pixel events, so this is the speckle detector.
  density       - generated note density / real
  leak          - density in pitch rows that are SILENT in the real roll, as a
                  fraction of density in occupied rows. 0 = respects the register.
  note F1       - pitch-exact, onset within tol
  pixel F1      - strict pixel overlap
  pixel F1 (al) - pixel F1 after applying the shift that maximises normalised
                  cross-correlation. Sparse thin note-lines are punished twice by a
                  small global offset (miss + false alarm), so this separates "wrong
                  notes" from "right notes, displaced".
  shift         - mean |dy| (semitones) and |dx| (time steps) of that offset

Select the model set with MIDIRAE_MODELS=exp26|c55 and the flow step with
MIDIRAE_CFM_CKPT. Run on a GPU host.
"""
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))

import pcfm_infer
import flow_infer
from _recon_note_f1 import notes, note_f1
from _xcorr_compare import ncc_map, RAD

DEV = "cuda"
STEPS, CFG, N = 10, 4.0, 30   # 10 steps is what the demo ships
POP909 = Path("/home/shawley/datasets/POP909_images_basic/val")   # held-out split; the
                                                                  # files moved into train/ and
                                                                  # val/ when the split was made


def pix_f1(pred, real):
    tp = float((pred & real).sum()); fp = float((pred & ~real).sum()); fn = float((~pred & real).sum())
    p = tp / max(tp + fp, 1e-9); r = tp / max(tp + fn, 1e-9)
    return 2 * p * r / max(p + r, 1e-9)


def main():
    demo = pcfm_infer.get_demo(); demo.set_device(DEV)
    print(f"MODEL_SET={pcfm_infer.MODEL_SET}  flow={pcfm_infer.CFM_CKPT.name}")
    print(f"encoder={pcfm_infer.ENCODER_CKPT.name}  steps={STEPS} cfg={CFG}\n")

    rn, gn, ds, lk, nf0, nf1, nf2, pf = [], [], [], [], [], [], [], []
    paf, sdy, sdx = [], [], []          # aligned pixel F1 and the shift it needed
    for p in sorted(POP909.glob("*.png")):
        img = flow_infer.image_to_binary_tensor(p, crop_x=int(flow_infer.best_crop_x(p)))
        if float((img > 0.5).float().mean()) < 0.01:
            continue
        real = img[0, 0].numpy() > 0.5
        gen = demo.generate(img, n_steps=STEPS, seed=0, cfg_strength=CFG, device=DEV) > 0.5
        occ = real.any(axis=1)
        if occ.sum() in (0, 128):
            continue
        rn.append(len(notes(real))); gn.append(len(notes(gen)))
        ds.append(gen.mean() / max(real.mean(), 1e-9))
        # Skip windows that generated nothing in occupied rows: the ratio is undefined
        # there and a 1e-9 guard turns one such window into a millions-large outlier
        # that destroys the mean. Median over the rest for robustness.
        if gen[occ].mean() > 1e-6:
            lk.append(gen[~occ].mean() / gen[occ].mean())
        r_ev = notes(real); g_ev = notes(gen)
        nf0.append(note_f1(g_ev, r_ev, 0)[2])
        nf1.append(note_f1(g_ev, r_ev, 1)[2])
        nf2.append(note_f1(g_ev, r_ev, 2)[2])
        pf.append(pix_f1(gen, real))
        m = ncc_map(real, gen)                     # shift that best aligns the two
        if m is not None:
            k = np.unravel_index(np.argmax(m), m.shape)
            dy, dx = k[0] - RAD, k[1] - RAD
            paf.append(pix_f1(np.roll(np.roll(gen, dy, axis=0), dx, axis=1), real))
            sdy.append(abs(dy)); sdx.append(abs(dx))
        if len(rn) >= N:
            break

    print(f"{len(rn)} windows")
    print(f"{'real notes/window':22s} {np.mean(rn):8.0f}")
    print(f"{'gen notes/window':22s} {np.mean(gn):8.0f}   <- speckle detector")
    print(f"{'density ratio':22s} {np.mean(ds):8.2f}")
    print(f"{'register leak (median)':22s} {np.median(lk):8.3f}   <- 0 = respects pitch range")
    print(f"{'note F1 (tol 0)':22s} {np.mean(nf0):8.3f}")
    print(f"{'note F1 (tol 1)':22s} {np.mean(nf1):8.3f}")
    print(f"{'note F1 (tol 2)':22s} {np.mean(nf2):8.3f}")
    print(f"{'pixel F1':22s} {np.mean(pf):8.3f}")
    print(f"{'pixel F1 (aligned)':22s} {np.mean(paf):8.3f}   <- after best global shift")
    print(f"{'  shift |dy|,|dx|':22s} {np.mean(sdy):8.1f}, {np.mean(sdx):.1f}"
          f"   (semitones, time steps)")


if __name__ == "__main__":
    main()

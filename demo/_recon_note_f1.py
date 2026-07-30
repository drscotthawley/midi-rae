"""Note-event F1 for the pixel-CFM, the way transcription is normally scored.

Pixel overlap is the wrong unit for MIDI: a note starting one step late is the
same note, but pixel F1 charges it as both a miss and a false alarm. Following
mir_eval's transcription convention, extract note EVENTS (pitch, onset, offset)
from the binary roll and match on pitch + onset within a tolerance.

Pitch must match exactly (a semitone off is a different note). Onset tolerance is
swept. Matching is greedy nearest-onset, one reference note per prediction.

NOTE: not comparable to the paper's decoder F1 of 0.9955, which is pixel overlap.
Do not put these side by side without recomputing the decoder the same way.

Run on a GPU host: ~/envs/midi-rae/bin/python _recon_note_f1.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))

import pcfm_infer
import flow_infer

DEV = "cuda" if torch.cuda.is_available() else "cpu"
STEPS, CFG, N = 20, 8.0, 30
TOLS = [0, 1, 2, 3]
POP909 = Path("/home/shawley/datasets/POP909_images_basic")


def notes(roll):
    """binary (pitch, time) -> list of (pitch, onset, offset) runs."""
    out = []
    padded = np.pad(roll.astype(np.int8), ((0, 0), (1, 1)))
    d = np.diff(padded, axis=1)
    starts = np.argwhere(d == 1)
    ends = np.argwhere(d == -1)
    for (p, s), (p2, e) in zip(starts, ends):
        assert p == p2
        out.append((int(p), int(s), int(e)))
    return out


def note_f1(pred, ref, tol):
    """Greedy match on exact pitch + onset within tol. Returns (P, R, F1)."""
    by_pitch = {}
    for i, (p, s, e) in enumerate(ref):
        by_pitch.setdefault(p, []).append([s, i])
    used = set()
    tp = 0
    for p, s, e in sorted(pred, key=lambda n: n[1]):
        cands = by_pitch.get(p, [])
        best, bd = None, None
        for s_ref, idx in cands:
            if idx in used:
                continue
            dd = abs(s_ref - s)
            if dd <= tol and (bd is None or dd < bd):
                best, bd = idx, dd
        if best is not None:
            used.add(best); tp += 1
    prec = tp / max(len(pred), 1)
    rec = tp / max(len(ref), 1)
    return prec, rec, 2 * prec * rec / max(prec + rec, 1e-9)


def main():
    demo = pcfm_infer.get_demo(); demo.set_device(DEV)
    pngs = sorted(POP909.glob("*.png"))
    res = {t: [] for t in TOLS}
    counts = []
    for p in pngs:
        img = flow_infer.image_to_binary_tensor(p, crop_x=int(flow_infer.best_crop_x(p)))
        if float((img > 0.5).float().mean()) < 0.01:
            continue
        real = notes(img[0, 0].numpy() > 0.5)
        gen = notes(demo.generate(img, n_steps=STEPS, seed=0, cfg_strength=CFG,
                                  device=DEV) > 0.5)
        counts.append((len(real), len(gen)))
        for t in TOLS:
            res[t].append(note_f1(gen, real, t))
        if len(counts) >= N:
            break

    c = np.array(counts)
    print(f"ckpt={pcfm_infer.CFM_CKPT.name}  {len(counts)} windows  steps={STEPS} cfg={CFG}")
    print(f"notes per window: real {c[:,0].mean():.0f}, generated {c[:,1].mean():.0f}\n")
    print(f"{'onset tol':>10s} {'precision':>10s} {'recall':>8s} {'F1':>16s}")
    for t in TOLS:
        a = np.array(res[t])
        print(f"{t:10d} {a[:,0].mean():10.3f} {a[:,1].mean():8.3f} "
              f"{a[:,2].mean():11.3f} ± {a[:,2].std():.3f}")


if __name__ == "__main__":
    main()

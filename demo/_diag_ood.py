"""Is the speckle early-training sample quality, or out-of-distribution conditioning?

Speckle is OFF-manifold: the training data contains no speckle, so an undertrained
model should give bad-but-note-like output, not scattered dots. Two suspects:

  A. the flow is simply early in training  -> PLAIN generation (no mask, no XMEP,
     conditioned on a real unmodified image) should be speckly too.
  B. XMEP-filled conditioning is OOD       -> plain generation is clean, and only
     the inpainting path degrades. The XMEP is trained with an L2 objective, which
     is mean-seeking, so its predictions may be shrunk relative to real embeddings;
     the PCA and the UNet only ever saw real-embedding statistics.

Also dumps, per level, the norm of XMEP predictions vs real encoder embeddings and
the resulting PCA code ranges -- if those distributions don't line up, B is live.

Run on hsrazer: ~/envs/midi-rae/bin/python _diag_ood.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))

import pcfm_infer
import guided_sample as gs
import flow_infer

EXAMPLES_DIR = HERE / "examples"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = (0, 1)
STEPS, CFG = 20, 4.0


def blob_mask(h=128, w=128):
    yy, xx = np.mgrid[0:h, 0:w]
    return (((xx - 78) / 26.0) ** 2 + ((yy - 64) / 30.0) ** 2) < 1.0


def mean_run_len(b2d):
    m = np.pad(b2d.astype(np.int8), ((0, 0), (1, 1)))
    d = np.diff(m, axis=1)
    s, e = np.argwhere(d == 1), np.argwhere(d == -1)
    return float((e[:, 1] - s[:, 1]).mean()) if len(s) else 0.0


def main():
    demo = pcfm_infer.get_demo(); demo.set_device(DEV)
    songs = sorted(p.name for p in EXAMPLES_DIR.glob("*.png"))
    hole = blob_mask()
    print(f"dev={DEV} ckpt={pcfm_infer.CFM_CKPT.name}\n")

    # ---- A vs B: run-length of plain generation vs the two inpainting arms -----
    print("Run-length (bar-ness). REAL data is the reference; speckle -> ~1-2 px.")
    print(f"{'variant':22s} {'density':>9s} {'run-len':>9s} {'vs real run-len':>16s}")
    rows = {"real": [], "plain-gen": [], "inpaint-XMEP": [], "inpaint-blanked": []}
    for song in songs:
        cx = flow_infer.best_crop_x(EXAMPLES_DIR / song)
        img = flow_infer.image_to_binary_tensor(EXAMPLES_DIR / song, crop_x=int(cx))
        orig = img[0, 0].numpy()
        rows["real"].append((float((orig > 0.5).mean()), mean_run_len(orig > 0.5)))

        img_holed = img.clone(); img_holed[..., torch.from_numpy(hole)] = 0.0
        x_known = img_holed * 2 - 1
        mask_t = torch.from_numpy((~hole).astype("float32")).view(1, 1, 128, 128)
        grad = gs.make_inpaint_grad(x_known, mask_t)
        mlc = demo.encode_to_mlcond_filled(img_holed, hole, dilate=0)
        for s in SEEDS:
            # PLAIN: whole image generated, conditioned on the REAL unmodified roll
            g = demo.generate(img, n_steps=STEPS, seed=s, cfg_strength=CFG, device=DEV)
            rows["plain-gen"].append((float((g > 0.5).mean()), mean_run_len(g > 0.5)))
            for tag, m in (("inpaint-XMEP", mlc), ("inpaint-blanked", None)):
                gi = gs.pnpflow_generate(demo, img_holed, grad, n_steps=STEPS, seed=s,
                                         cfg_strength=CFG, device=DEV, alpha=0.5,
                                         strength=1.0, num_avg=1, mlcond=m)
                rows[tag].append((float((gi[hole] > 0.5).mean()),
                                  mean_run_len((gi > 0.5) & hole)))

    real_r = np.mean([r for _, r in rows["real"]])
    for k, v in rows.items():
        d, r = np.mean([x[0] for x in v]), np.mean([x[1] for x in v])
        print(f"{k:22s} {d:9.3f} {r:9.2f} {r / real_r:15.0%}")

    # ---- Is the XMEP's output distribution like the encoder's? -----------------
    print("\nXMEP predictions vs real encoder embeddings (per-token norms, no pooling):")
    print(f"{'level':6s} {'||enc||':>10s} {'||pred||':>10s} {'ratio':>7s} "
          f"{'PCA code std (enc)':>19s} {'(filled)':>10s}")
    song = songs[0]
    cx = flow_infer.best_crop_x(EXAMPLES_DIR / song)
    img = flow_infer.image_to_binary_tensor(EXAMPLES_DIR / song, crop_x=int(cx))
    img_holed = img.clone(); img_holed[..., torch.from_numpy(hole)] = 0.0

    mep = demo.load_mep()
    enc_full = demo.encoder(img.to(DEV))
    enc_hole = demo.encoder(img_holed.to(DEV))
    masks = [m.to(DEV) for m in demo.hole_to_token_masks(hole, dilate=0)]
    orig_fn = mep._make_level_masks
    mep._make_level_masks = lambda levels, device: masks
    try:
        preds, _ = mep(enc_hole)
    finally:
        mep._make_level_masks = orig_fn

    for i in range(pcfm_infer.N_LEVELS):
        e = enc_full.patches.levels[i].emb[0]
        p = preds[i][0]
        hole_tok = ~masks[i][0]
        if hole_tok.sum() == 0:
            continue
        ne = float(e[hole_tok].norm(dim=-1).mean())
        npd = float(p[hole_tok].norm(dim=-1).mean())
        ce = demo.pca[i].transform(e.float().cpu().numpy())
        filled = torch.where(masks[i][0].unsqueeze(-1), e, p.to(e.dtype))
        cf = demo.pca[i].transform(filled.float().cpu().numpy())
        print(f"L{i:<5d} {ne:10.3f} {npd:10.3f} {npd / max(ne, 1e-6):7.2f} "
              f"{ce.std():19.3f} {cf.std():10.3f}")

    print("\nIf ||pred|| << ||enc||, the XMEP is mean-seeking and its PCA codes are")
    print("shrunk -> conditioning the UNet never saw -> off-manifold output.")


if __name__ == "__main__":
    main()

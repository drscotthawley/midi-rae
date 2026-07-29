"""Paper figure: melody replacement by masked latent inpainting.

Two-row layout: the top row is the setup (original, and the same excerpt with the
upper voice erased); the bottom row is a fair sample of seeds, so the reader sees
the spread of proposals rather than one flattering pick.

The erased notes are removed from the ENCODER's input too, so the conditioning
cannot copy them back -- this is regeneration, not reconstruction.

Settings are the ones that work in practice: CFG 0.8, 10 Euler steps -- weak
guidance and coarse integration leave the erased region free to explore.

    python demo/_figure_inpaint.py            # restyle from cached rolls, no model
    python demo/_figure_inpaint.py --regen    # resample (needs the checkpoints)

The rolls are cached to _fig_inpaint_cache.npz, so colour/label/layout changes
cost nothing and need no GPU. Sampling is deterministic given the seeds -- a
local CPU run reproduces the GPU run's densities exactly.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))



import flow_infer

EXAMPLES_DIR = HERE / "examples"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
STEPS, CFG = 10, 0.8
SEEDS = (0, 1, 2, 3)   # four, not five: a 4-column grid gives noticeably bigger
                       # panels while keeping the setup row and the samples row
                       # separate, which is what makes the figure readable
SCALE = 6                      # 128px roll -> 768px panel, print resolution
BG = (255, 255, 255)
REAL = (18, 28, 54)            # kept music: very dark navy
BOX = (168, 168, 176)          # erased-region outline
ACCENTS = {"magenta": (194, 24, 91), "forest": (21, 115, 60)}


def melody_mask(real, frac=0.35, cols=(30, 100)):
    """Erase the upper voice over a TIME WINDOW -- what a brush stroke does.

    Erased across the full width instead, the hole comes back near-empty (13-17%
    of erased density vs 71-80% for a local window): with no surrounding melody
    in time, there is nothing to continue from."""
    rows = np.where(real.any(axis=1))[0]
    if len(rows) == 0:
        return np.zeros_like(real, bool)
    top, bot = rows.min(), rows.max()
    cut = int(top + frac * (bot - top))
    m = np.zeros_like(real, bool)
    m[top:cut + 1, cols[0]:cols[1]] = True
    return m


def colorize(roll, hole, accent, outline=None):
    """White page, dark notes for kept music, `accent` for generated notes."""
    m = np.asarray(roll) > 0.5
    rgb = np.full((*m.shape, 3), BG, np.uint8)
    for c in range(3):
        rgb[:, :, c] = np.where(m, np.where(hole, accent[c], REAL[c]), BG[c])
    if outline is not None and outline.any():
        r, c = np.where(outline)
        r0, r1, c0, c1 = r.min(), r.max(), c.min(), c.max()
        blank = ~m                       # draw the box only where there is no note
        for cc in range(3):
            for row in (r0, r1):
                seg = slice(c0, c1 + 1)
                rgb[row, seg, cc] = np.where(blank[row, seg], BOX[cc], rgb[row, seg, cc])
            for col in (c0, c1):
                seg = slice(r0, r1 + 1)
                rgb[seg, col, cc] = np.where(blank[seg, col], BOX[cc], rgb[seg, col, cc])
    return rgb


FONT_PX = 51           # baked-in labels for the portable PNG; kept in step with the
                       # LaTeX overlay's \footnotesize so the two versions match.
                       # Labels stay short to fit one panel -- "erased", not
                       # "upper voice erased".
CACHE = HERE / "_fig_inpaint_cache.npz"   # generated rolls, so restyling the figure
                                          # needs no model and no GPU


def grid_figure(rows, scale=SCALE, pad=110, gap=3, rowgap=16, draw_labels=True):
    """rows = list of (panels, titles). Shorter rows are centred.

    Returns (image, positions) where positions is [(text, xfrac, yfrac), ...] with
    fractions relative to the image, y measured from the BOTTOM to match TikZ's
    unit-square idiom. With draw_labels=False the label bands are left blank so
    LaTeX can typeset them in the document font instead."""
    h, w = rows[0][0][0].shape[:2]
    ncol = max(len(p) for p, _ in rows)
    W = ncol * w + (ncol - 1) * gap
    H = len(rows) * h + (len(rows) - 1) * rowgap
    canvas = np.full((H, W, 3), BG, np.uint8)
    offsets = []
    y = 0
    for panels, _ in rows:
        span = len(panels) * w + (len(panels) - 1) * gap
        x0 = (W - span) // 2
        offsets.append((x0, y))
        x = x0
        for p in panels:
            canvas[y:y + h, x:x + w] = p
            x += w + gap
        y += h + rowgap

    im = Image.fromarray(canvas).resize((W * scale, H * scale), Image.NEAREST)
    out = Image.new("RGB", (im.width, im.height + pad * len(rows)), BG)
    d = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", FONT_PX)
    except OSError:
        font = None

    # paste each row separately so a label band sits above it
    positions = []
    for i, ((panels, titles), (x0, y)) in enumerate(zip(rows, offsets)):
        band = im.crop((0, y * scale, im.width, (y + h) * scale))
        out.paste(band, (0, pad * (i + 1) + y * scale))
        for j, t in enumerate(titles):
            x = (x0 + j * (w + gap)) * scale + 3
            ytop = pad * (i + 1) + y * scale - pad + 6
            if draw_labels:
                d.text((x, ytop), t, fill=(20, 20, 20), font=font)
            positions.append((t, x / out.width, 1.0 - ytop / out.height))
    return out, positions


def write_tikz(positions, dest, image="inpaint_melody_replace_nolabel.png"):
    """Emit a TikZ overlay that puts the labels in the document's own font.

    Coordinates are fractions of the image node, so the labels track the figure
    through any \\includegraphics scaling. They do NOT track a re-crop: the panel
    crop is derived from the data, so regenerating from a checkpoint whose output
    occupies a different pitch range means regenerating this file too (the script
    does that automatically)."""
    lines = [
        "% Generated by demo/_figure_inpaint.py -- do not edit by hand.",
        "% Labels are typeset by LaTeX so they match the document font.",
        "\\begin{tikzpicture}",
        f"  \\node[anchor=south west,inner sep=0] (fig) at (0,0)"
        f" {{\\includegraphics[width=\\columnwidth]{{{image}}}}};",
        "  \\begin{scope}[x={(fig.south east)},y={(fig.north west)}]",
    ]
    for text, xf, yf in positions:
        lines.append(f"    \\node[anchor=north west,inner sep=0,font=\\footnotesize]"
                     f" at ({xf:.4f},{yf:.4f}) {{{text}}};")
    lines += ["  \\end{scope}", "\\end{tikzpicture}", ""]
    dest.write_text("\n".join(lines))


def generate(regen=False):
    """Return (orig, hole, gens), from cache unless regen is asked for.

    Only this needs a model; restyling the figure re-runs render() alone."""
    if CACHE.exists() and not regen:
        z = np.load(CACHE)
        print(f"loaded cached rolls from {CACHE.name} (pass --regen to resample)")
        return z["orig"], z["hole"].astype(bool), [z[f"g{i}"] for i in range(len(SEEDS))]

    import pcfm_infer
    import guided_sample as gs
    demo = pcfm_infer.get_demo(); demo.set_device(DEV)
    song = sorted(p.name for p in EXAMPLES_DIR.glob("*.png"))[0]
    cx = flow_infer.best_crop_x(EXAMPLES_DIR / song)
    img = flow_infer.image_to_binary_tensor(EXAMPLES_DIR / song, crop_x=int(cx))
    orig = img[0, 0].numpy()
    hole = melody_mask(orig > 0.5)

    img_holed = img.clone(); img_holed[..., torch.from_numpy(hole)] = 0.0
    x_known = img_holed * 2 - 1
    mask_t = torch.from_numpy((~hole).astype("float32")).view(1, 1, 128, 128)
    grad = gs.make_inpaint_grad(x_known, mask_t)
    mlc = demo.encode_to_mlcond_filled(img_holed, hole, dilate=0)

    erased = float((orig[hole] > 0.5).mean())
    print(f"{song} crop_x={cx} ckpt={pcfm_infer.CFM_CKPT.name} cfg={CFG} steps={STEPS}")
    print(f"mask covers {hole.mean():.1%} of the window, "
          f"{(orig[hole] > 0.5).sum()} note-pixels erased (density {erased:.3f})\n")

    gens = []
    for s in SEEDS:
        g = gs.pnpflow_generate(demo, img_holed, grad, n_steps=STEPS, seed=s, cfg_strength=CFG,
                                device=DEV, alpha=0.5, strength=1.0, num_avg=1, mlcond=mlc)
        gens.append(g)
        print(f"seed {s}: hole density {float((g[hole] > 0.5).mean()):.3f} "
              f"({float((g[hole] > 0.5).mean()) / erased:.0%} of erased)")
    np.savez_compressed(CACHE, orig=orig, hole=hole,
                        **{f"g{i}": g for i, g in enumerate(gens)})
    return orig, hole, gens


def main():
    regen = "--regen" in sys.argv
    orig, hole, gens = generate(regen)
    nothing = np.zeros_like(hole)

    # Crop vertically to the occupied pitch range: most of a piano roll's range is
    # empty, and the blank bands dominate the figure otherwise.
    occ = np.where((orig > 0.5).any(axis=1))[0]
    r0, r1 = max(0, occ.min() - 5), min(128, occ.max() + 6)

    def cr(a):                                   # crop to the occupied register
        return a[r0:r1]

    # Setup on top, samples below -- never mix a seed into the setup row, even to
    # fill space: the grouping is what tells the reader what they are looking at.
    for aname, accent in ACCENTS.items():
        pan = [colorize(cr(orig), cr(nothing), accent, cr(hole)),
               colorize(cr(np.where(hole, 0.0, orig)), cr(nothing), accent, cr(hole))]
        seed_pan = [colorize(cr(g), cr(hole), accent, cr(hole)) for g in gens]
        top = (pan, ["original", "erased"])
        bottom = (seed_pan, [f"seed {s}" for s in SEEDS])

        # (a) labels baked in -- portable: slides, README, anywhere outside the paper
        img_lab, _ = grid_figure([top, bottom], draw_labels=True)
        img_lab.save(HERE / f"_fig_inpaint_workflow_{aname}.png")

        # (b) blank label bands + a TikZ overlay -- typeset in the document font
        img_raw, pos = grid_figure([top, bottom], draw_labels=False)
        img_raw.save(HERE / f"_fig_inpaint_workflow_{aname}_nolabel.png")
        if aname == "magenta":
            write_tikz(pos, HERE / "_fig_inpaint_labels.tex")

    print("\nsaved _fig_inpaint_workflow_{magenta,forest}[_nolabel].png "
          "and _fig_inpaint_labels.tex")


if __name__ == "__main__":
    main()

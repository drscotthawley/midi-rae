"""Headless test of the inpaint UI logic (no Gradio server).
Run: python demo/_test_inpaint_ui.py
"""
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))

import app


def synthetic_editor(bg, cols):
    """A 512x512 canvas with one RGBA layer painted over `cols` (display space)."""
    H = W = app.DISP
    layer = np.zeros((H, W, 4), np.uint8)
    layer[:, cols[0]:cols[1], 0] = 255
    layer[:, cols[0]:cols[1], 3] = 255   # alpha = painted
    return {"background": bg, "layers": [layer], "composite": bg}


def main():
    # 1. build_ui constructs (validates every gr.* arg incl. Brush)
    ui = app.build_ui()
    print("[ok] build_ui() constructed:", type(ui).__name__)

    # 2. mask extraction: paint display cols [192:320] -> 128-space cols [48:80]
    ev = synthetic_editor(np.zeros((app.DISP, app.DISP, 3), np.uint8), (192, 320))
    hole = app.extract_hole_mask(ev, 128)
    cols_hole = np.where(hole.any(axis=0))[0]
    print(f"[mask] hole cols = [{cols_hole.min()}..{cols_hole.max()}] (expect ~48..79), "
          f"frac={hole.mean():.3f}")
    assert 46 <= cols_hole.min() <= 50 and 78 <= cols_hole.max() <= 81, "mask mapping wrong"

    # 3. empty paint -> no hole, graceful message
    empty = {"background": None, "layers": [], "composite": None}
    assert not app.extract_hole_mask(empty, 128).any()
    _, _, msg = app.run_inpaint(empty, None, "pnpflow", 6, 4.0, 0, "cpu")
    print("[guard] no-state message:", msg.split('(')[0].strip())

    # 4. full run_inpaint on a real window with a painted band (fast: 6 steps)
    ex = sorted(p.name for p in app.EXAMPLES_DIR.glob("*.png"))
    bg, state = app.load_window_for_paint(ex[0], 0)
    print(f"[load] canvas bg={bg.shape} input_state={tuple(state.shape)}")
    ev2 = synthetic_editor(bg, (192, 320))
    for method in ("pnpflow", "soft", "hard"):
        out, player, status = app.run_inpaint(ev2, state, method, 6, 4.0, 0, "cpu")
        assert out is not None, f"{method} produced no image"
        print(f"[run:{method}] out={out.size}  {status.split('.')[0]}")
    print("\nINPAINT-UI TEST PASSED")


if __name__ == "__main__":
    main()

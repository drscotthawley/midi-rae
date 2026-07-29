"""Build the pcfm app UI and exercise run() once (no server)."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))

import app, flow_infer, pcfm_infer

ui = app.build_ui()
print("build_ui OK; devices:", pcfm_infer.available_devices())

ex = sorted(p.name for p in (HERE / "examples").glob("*.png"))[0]
cx = flow_infer.best_crop_x(HERE / "examples" / ex)
# full conditioning, euler
out = app.run(ex, cx, "cpu", "euler", 20, 4.0, 0, False, False, False, False, False, False)
in_img, gen_img, in_html, gen_html, status = out
print("in:", in_img.size, "gen:", gen_img.size,
      "in_player midi:", "midi-player" in in_html, "gen_player midi:", "midi-player" in gen_html)
print("euler status:", status)
# RK4
out2 = app.run(ex, cx, "cpu", "rk4", 20, 4.0, 0, False, False, False, False, False, False)
print("rk4 status:", out2[4])
print("CHECK_APP OK")

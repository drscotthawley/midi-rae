"""Does conditional generation reproduce the input? (real mlcdrop, all conditioning on)"""
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import pcfm_infer, flow_infer

demo = pcfm_infer.PixelCFMDemo(device="cpu").load()
ex = sorted((HERE / "examples").glob("*.png"))[0]
cx = flow_infer.best_crop_x(ex)
img = flow_infer.image_to_binary_tensor(ex, crop_x=cx)
inp = (img[0, 0].numpy() > 0.5)
print(f"input note_px={inp.sum()}")

for cfg in [1.0, 3.0, 6.0, 10.0]:
    g = demo.generate(img, drop_levels=(), n_steps=40, seed=0, cfg_strength=cfg) > 0.5
    inter = float((g & inp).sum()); union = float((g | inp).sum())
    print(f"CFG={cfg:>4}: gen_px={int(g.sum())} IoU={inter/max(union,1):.3f} recall={inter/max(inp.sum(),1):.3f}")

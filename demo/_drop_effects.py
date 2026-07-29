"""Measure how much dropping each conditioning level changes the generation."""
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import pcfm_infer, flow_infer

demo = pcfm_infer.PixelCFMDemo(device="cpu").load()
ex = sorted((HERE / "examples").glob("*.png"))[0]
img = flow_infer.image_to_binary_tensor(ex, crop_x=flow_infer.best_crop_x(ex))

base = demo.generate(img, drop_levels=(), n_steps=20, seed=1, cfg_strength=4.0)
print(f"base note_px={int((base>0.5).sum())}")
for lv in range(6):
    g = demo.generate(img, drop_levels=(lv,), n_steps=20, seed=1, cfg_strength=4.0)
    print(f"drop L{lv}: px_diff={int(np.abs((g>0.5).astype(int)-(base>0.5).astype(int)).sum())}")

# also which levels are actually referenced by the UNet forward
m = demo.model
print("film_cond_map (input-block idx -> cond level):", m.film_cond_map)

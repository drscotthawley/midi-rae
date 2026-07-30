"""Write the resolved training config embedded in a checkpoint out as a YAML file.

Why: probes need to rebuild the encoder with the architecture it was TRAINED with.
Pointing them at configs/config_swin.yaml only works when the run used the base
config unmodified -- a run launched with, say, config_swin_exp26repro.yaml carries
`depths: [2,2,2,6,2,1]` while the base says [2,2,2,6,2,2], and OmegaConf.load does
not resolve Hydra `defaults:` inheritance, so loading the base silently yields the
wrong architecture and the state_dict fails to load.

save_checkpoint() stores `config: dict(cfg)` -- the fully-resolved config -- so the
checkpoint is a more reliable source than any file on disk.

Usage: python scripts/ckpt_config.py <checkpoint.pt> <out.yaml>
"""
import sys

import torch
from omegaconf import OmegaConf

def main():
    if len(sys.argv) != 3:
        sys.exit(f"usage: {sys.argv[0]} <checkpoint.pt> <out.yaml>")
    ckpt_path, out_path = sys.argv[1], sys.argv[2]
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "config" not in ckpt:
        sys.exit(f"{ckpt_path} has no embedded 'config' key")
    cfg = OmegaConf.create(ckpt["config"])
    OmegaConf.save(cfg, out_path)
    depths = cfg.get("model", {}).get("depths", "?")
    print(f"wrote {out_path} (epoch {ckpt.get('epoch', '?')}, depths {depths})")


if __name__ == "__main__":
    main()

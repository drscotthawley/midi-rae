#!/usr/bin/env python3
"""
Evaluate a trained encoder checkpoint and print scalar metrics as JSON.

Usage:
    python scripts/eval_encoder.py <config_name> <checkpoint_path> [--device cpu|cuda]

Example:
    python scripts/eval_encoder.py config_swin ~/runs/midi-rae/exp19_abc123/checkpoints/SwinEncoder_exp19_abc123_best.pt
"""
import argparse, json, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from omegaconf import OmegaConf
from midi_rae.swin import SwinEncoder
from midi_rae.utils import load_checkpoint
from midi_rae.inspect import eval_encoder

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Config name without .yaml, or full path to .yaml')
    parser.add_argument('checkpoint', help='Path to encoder checkpoint .pt file')
    parser.add_argument('--device', default='cpu', help='Device: cpu or cuda')
    parser.add_argument('--data-path', default=None, help='Override cfg.data.path')
    args = parser.parse_args()

    if os.path.isfile(args.config):
        cfg_path = args.config
    else:
        cfg_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'configs', args.config + '.yaml')
        if not os.path.isfile(cfg_path):  # fallback: look in cwd/configs/
            cfg_path = os.path.join('configs', args.config + '.yaml')
    # Resolve Hydra-style defaults manually: merge base configs before the target
    cfg_dir = os.path.dirname(os.path.abspath(cfg_path))
    raw = OmegaConf.load(cfg_path)
    defaults_node = raw.get('defaults', None)
    defaults = OmegaConf.to_container(defaults_node, resolve=False) if defaults_node is not None else []
    base_cfgs = []
    for d in defaults:
        if isinstance(d, str) and d != '_self_':
            base_path = os.path.join(cfg_dir, d + '.yaml')
            if os.path.isfile(base_path):
                base_cfgs.append(OmegaConf.load(base_path))
    cfg = OmegaConf.merge(*base_cfgs, raw) if base_cfgs else raw
    if args.data_path:
        OmegaConf.update(cfg, 'data.path', args.data_path)
    device = args.device

    encoder = SwinEncoder(
        img_height=cfg.data.image_size, img_width=cfg.data.image_size,
        patch_h=cfg.model.patch_h, patch_w=cfg.model.patch_w,
        embed_dim=cfg.model.embed_dim, depths=cfg.model.depths,
        num_heads=cfg.model.num_heads, window_size=cfg.model.window_size,
        mlp_ratio=cfg.model.mlp_ratio, drop_path_rate=cfg.model.drop_path_rate,
    ).to(device)
    encoder = load_checkpoint(encoder, args.checkpoint)
    encoder.eval()

    metrics = eval_encoder(cfg, encoder, device=device)
    print(json.dumps(metrics, indent=2))

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Evaluate a trained decoder checkpoint with a threshold sweep (ROC / PR curves).

Usage:
    python scripts/eval_decoder.py <config> <decoder_ckpt> [--device cpu|cuda]
                                   [--n-batches N] [--out roc.png]

Example:
    python scripts/eval_decoder.py config_swin \
        ~/runs/midi-rae/dec26_2JIbRt/checkpoints/SwinDecoder_dec26_2JIbRt_best.pt

Prints the best-F1 threshold, then saves ROC + PR curve PNGs.
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from midi_rae.swin import SwinDecoder, SwinEncoder
from midi_rae.utils import load_checkpoint
from midi_rae.train_dec import setup_dataloaders, setup_models, get_pos_cache, emb_levels_to_enc_out
from midi_rae.data import collate_preencode


def load_cfg(config_arg):
    if os.path.isfile(config_arg):
        cfg_path = config_arg
    else:
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cfg_path = os.path.join(repo, 'configs', config_arg + '.yaml')
    cfg_dir = os.path.dirname(os.path.abspath(cfg_path))
    raw = OmegaConf.load(cfg_path)
    defaults = OmegaConf.to_container(raw.get('defaults', []), resolve=False)
    base_cfgs = []
    for d in defaults:
        if isinstance(d, str) and d != '_self_':
            p = os.path.join(cfg_dir, d + '.yaml')
            if os.path.isfile(p):
                base_cfgs.append(OmegaConf.load(p))
    return OmegaConf.merge(*base_cfgs, raw) if base_cfgs else raw


@torch.no_grad()
def collect_logits(decoder, val_dl, pos_cache, device, n_batches):
    """Return flat numpy arrays: logits (sigmoid probabilities) and targets."""
    all_probs, all_targets = [], []
    for i, batch in enumerate(val_dl):
        if n_batches and i >= n_batches:
            break
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            emb_data, img_real = batch
            enc_out = emb_levels_to_enc_out(emb_data, pos_cache, device)
        else:
            img_real = batch['image'].to(device)
            enc_out = None  # shouldn't reach here for preencoded
        img_real = img_real.to(device)
        logits = decoder(enc_out)          # (B, N, patch_h*patch_w) or (B, C, H, W)
        probs = torch.sigmoid(logits).float().cpu().numpy().ravel()
        targets = img_real.float().cpu().numpy().ravel()
        all_probs.append(probs)
        all_targets.append(targets)
    return np.concatenate(all_probs), np.concatenate(all_targets)


def sweep_thresholds(probs, targets, thresholds):
    eps = 1e-8
    rows = []
    for t in thresholds:
        pred = (probs >= t).astype(np.float32)
        TP = (pred * targets).sum()
        FP = (pred * (1 - targets)).sum()
        FN = ((1 - pred) * targets).sum()
        TN = ((1 - pred) * (1 - targets)).sum()
        precision = TP / (TP + FP + eps)
        recall    = TP / (TP + FN + eps)
        f1        = 2 * precision * recall / (precision + recall + eps)
        fpr       = FP / (FP + TN + eps)
        rows.append(dict(t=t, precision=precision, recall=recall, f1=f1, fpr=fpr, tpr=recall))
    return rows


def plot_curves(rows, out_prefix):
    thresholds  = [r['t']         for r in rows]
    f1s         = [r['f1']        for r in rows]
    precisions  = [r['precision'] for r in rows]
    recalls     = [r['recall']    for r in rows]
    fprs        = [r['fpr']       for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # F1 vs threshold
    ax = axes[0]
    ax.plot(thresholds, f1s)
    best_i = int(np.argmax(f1s))
    ax.axvline(thresholds[best_i], color='r', linestyle='--',
               label=f'best t={thresholds[best_i]:.2f}  F1={f1s[best_i]:.4f}')
    ax.set_xlabel('threshold'); ax.set_ylabel('F1'); ax.set_title('F1 vs Threshold')
    ax.legend(); ax.grid(True)

    # Precision-Recall curve
    ax = axes[1]
    ax.plot(recalls, precisions)
    ax.scatter([recalls[best_i]], [precisions[best_i]], color='r', zorder=5,
               label=f't={thresholds[best_i]:.2f}')
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve'); ax.legend(); ax.grid(True)

    # ROC curve
    ax = axes[2]
    ax.plot(fprs, recalls, label='ROC')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8)
    ax.scatter([fprs[best_i]], [recalls[best_i]], color='r', zorder=5,
               label=f't={thresholds[best_i]:.2f}')
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
    ax.set_title('ROC Curve'); ax.legend(); ax.grid(True)

    plt.tight_layout()
    out_path = out_prefix + '_curves.png'
    plt.savefig(out_path, dpi=120)
    print(f'Saved: {out_path}')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config',   help='Config name (without .yaml) or full path')
    parser.add_argument('checkpoint', help='Path to decoder .pt checkpoint')
    parser.add_argument('--device',    default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--n-batches', type=int, default=0,
                        help='Number of val batches to evaluate (0 = full val set)')
    parser.add_argument('--out', default='eval_dec',
                        help='Output file prefix for saved plots')
    parser.add_argument('--thresholds', type=float, nargs='+',
                        default=list(np.arange(0.05, 1.0, 0.05).round(3)),
                        help='Threshold values to sweep')
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    # Patch in a dummy tag so Hydra required field doesn't block us
    OmegaConf.update(cfg, 'tag', 'eval', merge=True)

    device = args.device
    print(f'Device: {device}')

    preencoded = True   # always use preencoded for speed
    _, val_dl = setup_dataloaders(cfg, preencoded=preencoded)

    _, decoder = setup_models(cfg, device, preencoded=preencoded)
    decoder = load_checkpoint(decoder, os.path.expandvars(os.path.expanduser(args.checkpoint)))
    decoder.eval()

    # Build pos_cache from encoder
    from midi_rae.swin import SwinEncoder
    _enc = SwinEncoder(
        img_height=cfg.data.image_size, img_width=cfg.data.image_size,
        patch_h=cfg.model.patch_h, patch_w=cfg.model.patch_w,
        embed_dim=cfg.model.embed_dim, depths=cfg.model.depths,
        num_heads=cfg.model.num_heads, window_size=cfg.model.window_size,
        mlp_ratio=cfg.model.mlp_ratio, drop_path_rate=cfg.model.drop_path_rate,
    ).to(device)
    _enc = load_checkpoint(_enc, os.path.expandvars(os.path.expanduser(str(cfg.encoder_ckpt))))
    pos_cache = get_pos_cache(_enc, cfg.data.image_size, device)
    del _enc

    print(f'Collecting logits (n_batches={args.n_batches or "all"})...')
    probs, targets = collect_logits(decoder, val_dl, pos_cache, device, args.n_batches)
    print(f'Total pixels: {len(probs):,}  positive rate: {targets.mean():.4f}')

    rows = sweep_thresholds(probs, targets, args.thresholds)

    best = max(rows, key=lambda r: r['f1'])
    print(f'\nBest threshold: {best["t"]:.2f}  '
          f'F1={best["f1"]:.4f}  '
          f'precision={best["precision"]:.4f}  '
          f'recall={best["recall"]:.4f}')

    # Print full table
    print(f'\n{"thresh":>7}  {"F1":>7}  {"prec":>7}  {"recall":>7}  {"FPR":>7}')
    for r in rows:
        marker = ' <--' if r['t'] == best['t'] else ''
        print(f'{r["t"]:7.2f}  {r["f1"]:7.4f}  {r["precision"]:7.4f}  {r["recall"]:7.4f}  {r["fpr"]:7.4f}{marker}')

    plot_curves(rows, args.out)


if __name__ == '__main__':
    main()

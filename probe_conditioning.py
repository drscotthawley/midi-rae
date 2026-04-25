"""
probe_conditioning.py — Does the pixel_cfm conditioning signal actually do anything?

Core test: generate images from the SAME noise sample, with:
  - "matched"  conditioning: image i gets its own PCA embeddings
  - "shuffled" conditioning: image i gets image j's PCA embeddings (j != i)
  - "uncond"   no conditioning at all

Since x0 is fixed, any difference in the output is purely due to conditioning.
If conditioning works, matched generations should be closer to the real image.

Key metric: p(matched_MSE < shuffled_MSE), p(matched_MSE < uncond_MSE)

Usage (on lecun):
    python probe_conditioning.py \
        --ckpt          checkpoints/otcfm_midi_weights_step_32000.pt \
        --preencoded_dir ~/datasets/POP909_encoded_exp26 \
        --n_samples 100 --device cuda
"""
import copy
import os, sys
import numpy as np
import torch
import wandb
import matplotlib.pyplot as plt
from pathlib import Path
from absl import app, flags
from torchdyn.core import NeuralODE
from torchvision.utils import make_grid, save_image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from torchcfm.models.unet.unet_mlc import UNetModelWrapperMLC

FLAGS = flags.FLAGS
flags.DEFINE_string("ckpt",           None,         "pixel_cfm checkpoint (.pt)")
flags.DEFINE_string("preencoded_dir", None,         "dir with preencoded+pca chunks")
flags.DEFINE_integer("n_samples",     100,          "number of test pairs")
flags.DEFINE_integer("n_viz",         8,            "number of sample image grids to log")
flags.DEFINE_integer("n_ode_steps",   20,           "ODE integration steps")
flags.DEFINE_string("solver",         "euler",      "ODE solver (passed to torchdyn NeuralODE)")
flags.DEFINE_integer("crop_size",     128,          "spatial crop size")
flags.DEFINE_boolean("use_ema",       False,        "load ema_model weights from checkpoint")
flags.DEFINE_string("device",         "cpu",       "torch device")
flags.DEFINE_boolean("no_wandb",      False,        "disable wandb logging")
flags.DEFINE_string("wandb_project",  "cond-probe",  "wandb project name")
flags.DEFINE_string("run_name",       "",           "wandb run name (empty = auto)")
flags.DEFINE_float("cfg_strength",    10.0,         "CFG guidance strength (1.0 = no amplification)")


def load_val_samples(preencoded_dir, n_samples, split='val'):
    """Load images and PCA conditioning from preencoded + pca chunk files."""
    enc_dir = Path(os.path.expandvars(os.path.expanduser(preencoded_dir)))
    pca_dir = Path(str(enc_dir).replace('encoded', 'pca'))

    enc_files = sorted(enc_dir.glob(f'{split}_chunk*.pt'))
    pca_files = sorted(pca_dir.glob(f'{split}_chunk*_pca.pt'))
    assert enc_files, f"No {split}_chunk*.pt in {enc_dir}"
    assert pca_files, f"No {split}_chunk*_pca.pt in {pca_dir}"

    imgs, conds = [], []
    for enc_p, pca_p in zip(enc_files, pca_files):
        enc = torch.load(enc_p, map_location='cpu', weights_only=False)
        pca = torch.load(pca_p, map_location='cpu', weights_only=False)
        offset = 0
        for rec in enc:
            n = rec['img1'].shape[0]
            cond_levels = [pca[k][offset:offset+n] for k in sorted(pca.keys())]
            offset += n
            for i in range(n):
                imgs.append(rec['img1'][i].float())
                conds.append([c[i] for c in cond_levels])
                if len(imgs) >= n_samples * 2:
                    break
            if len(imgs) >= n_samples * 2:
                break
        if len(imgs) >= n_samples * 2:
            break

    return imgs, conds


def prep_cond(cond_list, device):
    """Reshape per-sample cond from list of (n_patches, n_comp) → list of (1, n_comp, sp, sp)."""
    result = []
    for c in cond_list:
        n_patches, n_comp = c.shape
        sp = int(round(n_patches ** 0.5))
        result.append(c.view(sp, sp, n_comp).permute(2, 0, 1).unsqueeze(0).to(device))
    return result


class CFGWrapper(torch.nn.Module):
    def __init__(self, m, c, strength):
        super().__init__(); 
        self.m = m; self.c = c; self.strength = strength
    def forward(self, t, x, strength=None, **kwargs):
        # Note: torchdyn will only call forward(t, x) so the strength= kwarg won't be reachable via the ODE solver
        strength = self.strength if strength is None else strength
        v_uncond = self.m(t, x, mlcond=None)
        v_cond   = self.m(t, x, mlcond=self.c)
        return v_uncond + strength * (v_cond - v_uncond)
    


def generate_samples(model, x0, cond, n_steps=20, solver='euler', cfg_strength=10.0):
    model_ = CFGWrapper(model, cond, cfg_strength) if cond is not None else model
    node = NeuralODE(model_, solver=solver, sensitivity="adjoint")
    with torch.no_grad():
        traj = node.trajectory(x0, t_span=torch.linspace(0, 1, n_steps, device=x0.device))
    return traj[-1].clip(-1, 1) / 2 + 0.5


def main(argv):
    assert FLAGS.ckpt, "must provide --ckpt"
    assert FLAGS.preencoded_dir, "must provide --preencoded_dir"
    device = FLAGS.device
    for k, v in sorted(FLAGS.flag_values_dict().items()):
        print(f"FLAG  {k}: {v}")


    print(f"Loading val samples from {FLAGS.preencoded_dir} ...")
    imgs, conds = load_val_samples(FLAGS.preencoded_dir, FLAGS.n_samples)
    N = min(FLAGS.n_samples, len(imgs) // 2)
    print(f"  {len(imgs)} samples available, probing {N}")

    ckpt = torch.load(os.path.expandvars(os.path.expanduser(FLAGS.ckpt)),
                      map_location=device, weights_only=False)
    key = 'ema_model' if (FLAGS.use_ema and 'ema_model' in ckpt) else 'net_model'
    state = ckpt[key]
    step  = ckpt.get('step', 0)
    num_channels = state['input_blocks.0.0.weight'].shape[0]
    print(f"  num_channels inferred from checkpoint: {num_channels}")

    sample_cond = prep_cond(conds[0], 'cpu')
    mlcond_shapes = {c.shape[2]: (i, c.shape[1]) for i, c in enumerate(sample_cond)}
    print(f"  mlcond_shapes: {mlcond_shapes}")

    model = UNetModelWrapperMLC(
        dim=(1, FLAGS.crop_size, FLAGS.crop_size),
        num_res_blocks=2,
        num_channels=num_channels,
        channel_mult=[1, 2, 2, 2, 2],
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="16",
        dropout=0.1,
        mlcond_shapes=mlcond_shapes,
    ).to(device).eval()
    model.load_state_dict(state)
    print(f"Loaded {key} from step {step}")

    if not FLAGS.no_wandb:
        wandb.init(project=FLAGS.wandb_project, name=FLAGS.run_name or None,
                   config=dict(ckpt=FLAGS.ckpt, n_samples=N, n_ode_steps=FLAGS.n_ode_steps,
                               solver=FLAGS.solver, step=step, use_ema=FLAGS.use_ema))

    matched_mses, shuffled_mses, uncond_mses = [], [], []
    viz_real, viz_matched, viz_shuffled, viz_uncond = [], [], [], []
    rng = torch.Generator(device=device)
    rng.manual_seed(42)

    for i in range(N):
        j = (i + N // 2) % len(imgs)
        real = imgs[i].unsqueeze(0).to(device)  # (1, 1, H, W)

        cond_m = prep_cond(conds[i], device)
        cond_s = prep_cond(conds[j], device)

        # Same noise for all three: isolates conditioning effect from stochasticity
        x0 = torch.randn(1, 1, FLAGS.crop_size, FLAGS.crop_size, device=device, generator=rng)

        gen_m = generate_samples(model, x0, cond_m, n_steps=FLAGS.n_ode_steps, solver=FLAGS.solver, cfg_strength=FLAGS.cfg_strength)
        gen_s = generate_samples(model, x0, cond_s, n_steps=FLAGS.n_ode_steps, solver=FLAGS.solver, cfg_strength=FLAGS.cfg_strength)
        gen_u = generate_samples(model, x0, None,   n_steps=FLAGS.n_ode_steps, solver=FLAGS.solver)

        mse_m = ((gen_m - real) ** 2).mean().item()
        mse_s = ((gen_s - real) ** 2).mean().item()
        mse_u = ((gen_u - real) ** 2).mean().item()
        matched_mses.append(mse_m)
        shuffled_mses.append(mse_s)
        uncond_mses.append(mse_u)

        if i < FLAGS.n_viz:
            viz_real.append(real.cpu());     viz_matched.append(gen_m.cpu())
            viz_shuffled.append(gen_s.cpu()); viz_uncond.append(gen_u.cpu())

        if i % 10 == 0:
            print(f"[{i:3d}/{N}]  matched={mse_m:.4f}  shuffled={mse_s:.4f}  uncond={mse_u:.4f}")

    matched_mses  = np.array(matched_mses)
    shuffled_mses = np.array(shuffled_mses)
    uncond_mses   = np.array(uncond_mses)
    p_vs_shuffled = (matched_mses < shuffled_mses).mean()
    p_vs_uncond   = (matched_mses < uncond_mses).mean()

    print(f"\n=== Results ===")
    print(f"  matched  MSE: {matched_mses.mean():.4f} ± {matched_mses.std():.4f}")
    print(f"  shuffled MSE: {shuffled_mses.mean():.4f} ± {shuffled_mses.std():.4f}")
    print(f"  uncond   MSE: {uncond_mses.mean():.4f} ± {uncond_mses.std():.4f}")
    print(f"  p(matched < shuffled): {p_vs_shuffled:.3f}  (0.5=chance, 1.0=perfect)")
    print(f"  p(matched < uncond):   {p_vs_uncond:.3f}  (0.5=chance, 1.0=perfect)")

    # Image grids: each row is one sample, columns are [real, matched, shuffled, uncond]
    nrow = len(viz_real)
    grid_all = torch.cat([torch.cat(viz_real), torch.cat(viz_matched),
                          torch.cat(viz_shuffled), torch.cat(viz_uncond)], dim=0)
    # interleave so each row is one sample's 4 versions
    idx = [i + offset * nrow for i in range(nrow) for offset in range(4)]
    grid_interleaved = grid_all[idx]
    grid_bin = (grid_interleaved > 0.5).float()

    save_image(grid_interleaved, 'probe_conditioning_grid.png', nrow=4)
    save_image(grid_bin,         'probe_conditioning_grid_bin.png', nrow=4)
    print("Saved → probe_conditioning_grid.png  (columns: real | matched | shuffled | uncond)")

    # Histogram
    fig, ax = plt.subplots(figsize=(8, 4))
    all_vals = np.concatenate([matched_mses, shuffled_mses, uncond_mses])
    bins = np.linspace(all_vals.min(), all_vals.max(), 40)
    ax.hist(matched_mses,  bins=bins, alpha=0.6, label='matched',  color='steelblue')
    ax.hist(shuffled_mses, bins=bins, alpha=0.6, label='shuffled', color='tomato')
    ax.hist(uncond_mses,   bins=bins, alpha=0.6, label='uncond',   color='gray')
    ax.axvline(matched_mses.mean(),  color='steelblue', ls='--', lw=2)
    ax.axvline(shuffled_mses.mean(), color='tomato',    ls='--', lw=2)
    ax.axvline(uncond_mses.mean(),   color='gray',      ls='--', lw=2)
    ax.set_xlabel('Pixel MSE vs real image')
    ax.set_title(f'pixel_cfm conditioning probe  |  p(matched<shuffled)={p_vs_shuffled:.3f}  p(matched<uncond)={p_vs_uncond:.3f}')
    ax.legend()
    plt.tight_layout()
    plt.savefig('probe_conditioning_dists.png', dpi=120)
    print("Saved → probe_conditioning_dists.png")

    if not FLAGS.no_wandb:
        wandb.log({
            "media/real_matched_shuffled_uncond":      wandb.Image(make_grid(grid_interleaved, nrow=4),
                                                        caption="real | matched | shuffled | uncond"),
            "media/real_matched_shuffled_uncond_bin":  wandb.Image(make_grid(grid_bin, nrow=4),
                                                        caption="binarized: real | matched | shuffled | uncond"),
            "eval/p_matched_lt_shuffled": p_vs_shuffled,
            "eval/p_matched_lt_uncond":   p_vs_uncond,
            "eval/matched_mse_mean":  matched_mses.mean(),
            "eval/shuffled_mse_mean": shuffled_mses.mean(),
            "eval/uncond_mse_mean":   uncond_mses.mean(),
            "media/hist": wandb.Image(fig),
        }, step=step)
        wandb.finish()


if __name__ == '__main__':
    app.run(main)

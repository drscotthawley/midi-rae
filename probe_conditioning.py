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
import os, sys
import numpy as np
import torch
import torch.nn.functional as F
import wandb
import matplotlib.pyplot as plt
from pathlib import Path
from absl import app, flags
from torchdyn.core import NeuralODE
from torchvision.utils import make_grid, save_image
from tqdm import tqdm


def xcorr_peak(a, b):
    """Shift-tolerant similarity: peak of normalized 2D cross-correlation via FFT."""
    a, b = a.squeeze().float(), b.squeeze().float()
    xcorr = torch.fft.irfft2(torch.fft.rfft2(a) * torch.fft.rfft2(b).conj())
    return (xcorr.max() / (a.norm() * b.norm()).clamp(min=1e-8)).item()


def pitch_profile_corr(a, b):
    """Pearson correlation of pitch-axis marginals (time-invariant)."""
    pa = a.squeeze().float().sum(dim=-1)   # (H,)
    pb = b.squeeze().float().sum(dim=-1)
    pa, pb = pa - pa.mean(), pb - pb.mean()
    denom = pa.norm() * pb.norm()
    return (pa @ pb / denom.clamp(min=1e-8)).item()


def rhythm_profile_corr(a, b):
    """Pearson correlation of time-axis marginals (pitch-invariant)."""
    ra = a.squeeze().float().sum(dim=-2)   # (W,)
    rb = b.squeeze().float().sum(dim=-2)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denom = ra.norm() * rb.norm()
    return (ra @ rb / denom.clamp(min=1e-8)).item()


def fft_mag_dist(a, b):
    """L2 distance between 2D FFT magnitude spectra (translation-invariant)."""
    ma = torch.fft.rfft2(a.squeeze().float()).abs()
    mb = torch.fft.rfft2(b.squeeze().float()).abs()
    return F.mse_loss(ma, mb).item()


def polyphony_corr(a, b, thresh=0.5):
    """Pearson correlation of notes-per-time-frame profiles."""
    pa = (a.squeeze() > thresh).float().sum(dim=-2)   # (W,)
    pb = (b.squeeze() > thresh).float().sum(dim=-2)
    pa, pb = pa - pa.mean(), pb - pb.mean()
    denom = pa.norm() * pb.norm()
    return (pa @ pb / denom.clamp(min=1e-8)).item()


def note_density_ratio(a, b, thresh=0.5):
    """Absolute log ratio of note counts (0=perfect match)."""
    na = (a.squeeze() > thresh).float().sum().clamp(min=1)
    nb = (b.squeeze() > thresh).float().sum().clamp(min=1)
    return abs(torch.log(na / nb)).item()


def mean_pitch_normalized(img):
    """Normalized mean pitch (0-1) for a (1, H, W) piano roll tensor."""
    H = img.shape[-2]
    density = img[0].float().sum(dim=-1)  # (H,)
    pitch_idx = torch.arange(H, dtype=torch.float32)
    mp = (pitch_idx * density).sum() / density.sum().clamp(min=1e-6)
    return mp / H


class TqdmWrapper(torch.nn.Module):
    def __init__(self, model, n_steps, solver='euler', desc='ODE'):
        super().__init__()
        self.model = model
        calls = {'euler': 1, 'rk4': 4, 'midpoint': 2}.get(solver, 1)
        self.pbar = tqdm(total=n_steps * calls, desc=desc)
    def forward(self, t, x, **kwargs):
        self.pbar.update(1)
        return self.model(t, x, **kwargs)
    def close(self):
        self.pbar.close()

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

    imgs, conds, embs = [], [], []
    for enc_p, pca_p in zip(enc_files, pca_files):
        enc = torch.load(enc_p, map_location='cpu', weights_only=False)
        pca = torch.load(pca_p, map_location='cpu', weights_only=False)
        offset = 0
        for rec in enc:
            n = rec['img1'].shape[0]
            cond_levels = [pca[k][offset:offset+n] for k in sorted(pca.keys())]
            offset += n
            for i in range(n):
                img_i = rec['img1'][i].float()
                mp = mean_pitch_normalized(img_i)
                imgs.append(img_i)
                conds.append([torch.cat([c[i], torch.full((c[i].shape[0], 1), mp.item())], dim=1) for c in cond_levels])
                embs.append([lvl[i].float() for lvl in rec['emb1']])
                if len(imgs) >= n_samples * 2:
                    break
            if len(imgs) >= n_samples * 2:
                break
        if len(imgs) >= n_samples * 2:
            break

    return imgs, conds, embs


def prep_cond(cond_list, device):
    """Reshape per-sample cond from list of (n_patches, n_comp) → list of (1, n_comp, sp, sp)."""
    result = []
    for c in cond_list:
        n_patches, n_comp = c.shape
        sp = int(round(n_patches ** 0.5))
        result.append(c.view(sp, sp, n_comp).permute(2, 0, 1).unsqueeze(0).to(device))
    return result


def prep_cond_batch(conds_list, device):
    """Batch N per-sample cond lists → list of (N, n_comp, sp, sp) tensors."""
    n_levels = len(conds_list[0])
    result = []
    for lev in range(n_levels):
        level_tensors = []
        for sample_cond in conds_list:
            c = sample_cond[lev]
            sp = int(round(c.shape[0] ** 0.5))
            level_tensors.append(c.view(sp, sp, c.shape[1]).permute(2, 0, 1))
        result.append(torch.stack(level_tensors).to(device))
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
    


def generate_samples(model, x0, cond, n_steps=20, solver='euler', cfg_strength=10.0, desc='ODE'):
    model_ = CFGWrapper(model, cond, cfg_strength) if cond is not None else model
    wrapped = TqdmWrapper(model_, n_steps, solver=solver, desc=desc)
    node = NeuralODE(wrapped, solver=solver, sensitivity="adjoint")
    with torch.no_grad():
        traj = node.trajectory(x0, t_span=torch.linspace(0, 1, n_steps, device=x0.device))
    wrapped.close()
    return traj[-1].clip(-1, 1) / 2 + 0.5


def main(argv):
    assert FLAGS.ckpt, "must provide --ckpt"
    assert FLAGS.preencoded_dir, "must provide --preencoded_dir"
    device = FLAGS.device
    for k, v in sorted(FLAGS.flag_values_dict().items()):
        print(f"FLAG  {k}: {v}")


    print(f"Loading val samples from {FLAGS.preencoded_dir} ...")
    imgs, conds, embs = load_val_samples(FLAGS.preencoded_dir, FLAGS.n_samples)
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

    rng = torch.Generator(device=device)
    rng.manual_seed(42)

    js = [(i + N // 2) % len(imgs) for i in range(N)]
    real_batch  = torch.stack([imgs[i] for i in range(N)]).to(device)   # (N,1,H,W)
    cond_m_batch = prep_cond_batch([conds[i] for i in range(N)], device)
    cond_s_batch = prep_cond_batch([conds[j] for j in js],       device)
    x0_batch     = torch.randn(N, 1, FLAGS.crop_size, FLAGS.crop_size, device=device, generator=rng)

    gen_m = generate_samples(model, x0_batch, cond_m_batch, n_steps=FLAGS.n_ode_steps, solver=FLAGS.solver, cfg_strength=FLAGS.cfg_strength, desc='matched')
    gen_s = generate_samples(model, x0_batch, cond_s_batch, n_steps=FLAGS.n_ode_steps, solver=FLAGS.solver, cfg_strength=FLAGS.cfg_strength, desc='shuffled')
    gen_u = generate_samples(model, x0_batch, None,         n_steps=FLAGS.n_ode_steps, solver=FLAGS.solver, desc='uncond')

    matched_mses  = ((gen_m - real_batch) ** 2).mean(dim=(1,2,3)).cpu().numpy()
    shuffled_mses = ((gen_s - real_batch) ** 2).mean(dim=(1,2,3)).cpu().numpy()
    uncond_mses   = ((gen_u - real_batch) ** 2).mean(dim=(1,2,3)).cpu().numpy()

    viz_real     = [real_batch[:FLAGS.n_viz].cpu()]
    viz_matched  = [gen_m[:FLAGS.n_viz].cpu()]
    viz_shuffled = [gen_s[:FLAGS.n_viz].cpu()]
    viz_uncond   = [gen_u[:FLAGS.n_viz].cpu()]

    # Per-sample similarity metrics (all higher = more similar to real, except fft_dist/density_ratio)
    def _per_sample(fn, gen):
        return np.array([fn(real_batch[i].cpu(), gen[i].cpu()) for i in range(N)])

    matched_xcorr   = _per_sample(xcorr_peak,          gen_m.cpu())
    shuffled_xcorr  = _per_sample(xcorr_peak,          gen_s.cpu())
    uncond_xcorr    = _per_sample(xcorr_peak,          gen_u.cpu())

    matched_pitch   = _per_sample(pitch_profile_corr,  gen_m.cpu())
    shuffled_pitch  = _per_sample(pitch_profile_corr,  gen_s.cpu())
    uncond_pitch    = _per_sample(pitch_profile_corr,  gen_u.cpu())

    matched_rhythm  = _per_sample(rhythm_profile_corr, gen_m.cpu())
    shuffled_rhythm = _per_sample(rhythm_profile_corr, gen_s.cpu())
    uncond_rhythm   = _per_sample(rhythm_profile_corr, gen_u.cpu())

    matched_fft     = _per_sample(fft_mag_dist,        gen_m.cpu())
    shuffled_fft    = _per_sample(fft_mag_dist,        gen_s.cpu())
    uncond_fft      = _per_sample(fft_mag_dist,        gen_u.cpu())

    matched_poly    = _per_sample(polyphony_corr,      gen_m.cpu())
    shuffled_poly   = _per_sample(polyphony_corr,      gen_s.cpu())
    uncond_poly     = _per_sample(polyphony_corr,      gen_u.cpu())

    matched_dens    = _per_sample(note_density_ratio,  gen_m.cpu())
    shuffled_dens   = _per_sample(note_density_ratio,  gen_s.cpu())
    uncond_dens     = _per_sample(note_density_ratio,  gen_u.cpu())

    def _report(name, m, s, u, higher_better=True):
        cmp = '<' if higher_better else '>'
        p_s = (m > s).mean() if higher_better else (m < s).mean()
        p_u = (m > u).mean() if higher_better else (m < u).mean()
        print(f"  {name:<22}  matched={m.mean():.4f}  shuffled={s.mean():.4f}  uncond={u.mean():.4f}"
              f"  p(m{cmp}s)={p_s:.3f}  p(m{cmp}u)={p_u:.3f}")

    print(f"\n=== Results ===")
    _report("MSE (lower=better)",      matched_mses,    shuffled_mses,    uncond_mses,    higher_better=False)
    _report("XCorr peak (↑)",          matched_xcorr,   shuffled_xcorr,   uncond_xcorr)
    _report("Pitch profile corr (↑)",  matched_pitch,   shuffled_pitch,   uncond_pitch)
    _report("Rhythm profile corr (↑)", matched_rhythm,  shuffled_rhythm,  uncond_rhythm)
    _report("FFT mag dist (↓)",        matched_fft,     shuffled_fft,     uncond_fft,     higher_better=False)
    _report("Polyphony corr (↑)",      matched_poly,    shuffled_poly,    uncond_poly)
    _report("Note density ratio (↓)",  matched_dens,    shuffled_dens,    uncond_dens,    higher_better=False)

    # Pitch regression: does PCA conditioning retain absolute pitch info?
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pitches = np.array([(mean_pitch_normalized(imgs[i]) * imgs[i].shape[-2]).item() for i in range(N)])
    n_levels = len(conds[0])
    last_col = np.array([conds[i][0].numpy().ravel()[-1] for i in range(N)])
    print(f"\n=== Mean Pitch Diagnostic ===")
    print(f"  conds[0][0].shape = {conds[0][0].shape}  (expect (n_patches, n_pca+1))")
    print(f"  Last col of L0: min={last_col.min():.4f} max={last_col.max():.4f} std={last_col.std():.4f}")
    print(f"  Pitches:        min={pitches.min():.2f}  max={pitches.max():.2f}  std={pitches.std():.2f}")
    print(f"  Corr(last_col, pitches/H): {np.corrcoef(last_col, pitches/imgs[0].shape[-2])[0,1]:.4f}")
    print(f"\n=== Pitch Regression from PCA Features (5-fold CV R²) ===")
    pitch_r2s = {}
    for lev in range(n_levels):
        X = np.array([conds[i][lev].numpy().ravel() for i in range(N)])
        r2 = cross_val_score(Pipeline([('scaler', StandardScaler()), ('ridge', Ridge())]), X, pitches, cv=5, scoring='r2').mean()
        pitch_r2s[f'L{lev}'] = r2
        print(f"  L{lev}  R²={r2:.4f}")

    print(f"\n=== Pitch Regression from Raw Encoder Embeddings (5-fold CV R²) ===")
    pitch_r2s_raw = {}
    for lev in range(len(embs[0])):
        X = np.array([embs[i][lev].numpy().ravel() for i in range(N)])
        r2 = cross_val_score(Pipeline([('scaler', StandardScaler()), ('ridge', Ridge())]), X, pitches, cv=5, scoring='r2').mean()
        pitch_r2s_raw[f'L{lev}'] = r2
        print(f"  L{lev}  R²={r2:.4f}")

    # Image grids: each row is one sample, columns are [real, matched, shuffled, uncond]
    nrow = FLAGS.n_viz
    grid_all = torch.cat([viz_real[0], viz_matched[0], viz_shuffled[0], viz_uncond[0]], dim=0)
    # interleave so each row is one sample's 4 versions
    idx = [i + offset * nrow for i in range(nrow) for offset in range(4)]
    grid_interleaved = grid_all[idx]
    grid_bin = (grid_interleaved > 0.5).float()

    save_image(grid_interleaved, 'probe_conditioning_grid.png', nrow=4)
    save_image(grid_bin,         'probe_conditioning_grid_bin.png', nrow=4)
    print("Saved → probe_conditioning_grid.png  (columns: real | matched | shuffled | uncond)")

    # Histograms: one panel per metric
    metrics = [
        ("MSE (↓)",              matched_mses,   shuffled_mses,   uncond_mses,   False),
        ("XCorr peak (↑)",       matched_xcorr,  shuffled_xcorr,  uncond_xcorr,  True),
        ("Pitch profile corr (↑)",matched_pitch,  shuffled_pitch,  uncond_pitch,  True),
        ("Rhythm profile corr (↑)",matched_rhythm,shuffled_rhythm, uncond_rhythm, True),
        ("FFT mag dist (↓)",     matched_fft,    shuffled_fft,    uncond_fft,    False),
        ("Polyphony corr (↑)",   matched_poly,   shuffled_poly,   uncond_poly,   True),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, (title, m, s, u, hi) in zip(axes.flat, metrics):
        all_v = np.concatenate([m, s, u])
        bins = np.linspace(all_v.min(), all_v.max(), 30)
        ax.hist(m, bins=bins, alpha=0.6, label='matched',  color='steelblue')
        ax.hist(s, bins=bins, alpha=0.6, label='shuffled', color='tomato')
        ax.hist(u, bins=bins, alpha=0.6, label='uncond',   color='gray')
        for val, col in zip([m.mean(), s.mean(), u.mean()], ['steelblue','tomato','gray']):
            ax.axvline(val, color=col, ls='--', lw=1.5)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7)
    plt.suptitle('pixel_cfm conditioning probe', fontsize=11)
    plt.tight_layout()
    plt.savefig('probe_conditioning_dists.png', dpi=120)
    print("Saved → probe_conditioning_dists.png")

    if not FLAGS.no_wandb:
        def _wandb_metric(name, m, s, u, hi=True):
            cmp = 'gt' if hi else 'lt'
            p_s = (m > s).mean() if hi else (m < s).mean()
            p_u = (m > u).mean() if hi else (m < u).mean()
            return {f"eval/{name}/matched": m.mean(), f"eval/{name}/shuffled": s.mean(),
                    f"eval/{name}/uncond": u.mean(),
                    f"eval/{name}/p_m_{cmp}_s": p_s, f"eval/{name}/p_m_{cmp}_u": p_u}
        wandb.log({
            "media/real_matched_shuffled_uncond":     wandb.Image(make_grid(grid_interleaved, nrow=4),
                                                       caption="real | matched | shuffled | uncond"),
            "media/real_matched_shuffled_uncond_bin": wandb.Image(make_grid(grid_bin, nrow=4),
                                                       caption="binarized: real | matched | shuffled | uncond"),
            "media/hist": wandb.Image(fig),
            **_wandb_metric("mse",           matched_mses,   shuffled_mses,   uncond_mses,   hi=False),
            **_wandb_metric("xcorr_peak",    matched_xcorr,  shuffled_xcorr,  uncond_xcorr),
            **_wandb_metric("pitch_corr",    matched_pitch,  shuffled_pitch,  uncond_pitch),
            **_wandb_metric("rhythm_corr",   matched_rhythm, shuffled_rhythm, uncond_rhythm),
            **_wandb_metric("fft_dist",      matched_fft,    shuffled_fft,    uncond_fft,    hi=False),
            **_wandb_metric("poly_corr",     matched_poly,   shuffled_poly,   uncond_poly),
            **_wandb_metric("density_ratio", matched_dens,   shuffled_dens,   uncond_dens,   hi=False),
            **{f"eval/pitch_r2_pca/{k}": v for k, v in pitch_r2s.items()},
            **{f"eval/pitch_r2_raw/{k}": v for k, v in pitch_r2s_raw.items()},
        }, step=step)
        wandb.finish()


if __name__ == '__main__':
    app.run(main)

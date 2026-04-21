# Flow matching on MIDI piano roll images, adapted from torchcfm CIFAR10 example.
# Uses OT-CFM (or other variants) with no conditioning.
#
# Usage:
#   python train_cfm_midi.py [--model otcfm] [--num_channel 128] [--total_steps 400001]

import copy
import os

import numpy as np
import torch
import wandb
from absl import app, flags
from scipy.stats import wasserstein_distance
from torchvision.utils import make_grid, save_image
from torchdyn.core import NeuralODE
from tqdm import trange
from reloading import reloading

from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
)
from torchcfm.models.unet.unet_mlc import UNetModelWrapperMLC

from pathlib import Path
from midi_rae.data import AnchorDataset


class PreencodedImageDataset(torch.utils.data.IterableDataset):
    """Stream img+PCA conditioning from pre-encoded chunk pairs."""
    def __init__(self, data_dir, crop_size=128, split='train'):
        self.enc_chunks = sorted(Path(data_dir).glob(f"{split}_chunk*.pt"))
        pca_dir = Path(str(data_dir).replace("encoded", "pca"))
        self.pca_chunks = sorted(pca_dir.glob(f"{split}_chunk*_pca.pt"))
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        pairs = list(zip(self.enc_chunks, self.pca_chunks))
        if worker_info is not None:
            pairs = pairs[worker_info.id::worker_info.num_workers]
        for enc_p, pca_p in pairs:
            enc = torch.load(enc_p, weights_only=False)
            pca = torch.load(pca_p, weights_only=False)
            offset = 0
            for rec in enc:
                n = rec['img1'].shape[0]
                cond = [pca[k][offset:offset+n] for k in sorted(pca.keys())]
                offset += n
                for i, img in enumerate(rec['img1'].unbind(0)):
                    yield {'img': img.float(), 'cond': [c[i] for c in cond]}

FLAGS = flags.FLAGS

flags.DEFINE_string("model", "otcfm", help="flow matching model type")
flags.DEFINE_string("output_dir", "./results/cfm_midi/", help="output directory")
flags.DEFINE_string("data_dir", "~/datasets/POP909_images_basic/", help="piano roll image dir")
flags.DEFINE_string("data_mode", "anchor", help="data loading: anchor or preencoded")
flags.DEFINE_string("preencoded_dir", "", help="dir of preencoded .pt chunks (data_mode=preencoded)")
# UNet
flags.DEFINE_integer("num_channel", 128, help="base channel of UNet")
# Training
flags.DEFINE_float("lr", 2e-4, help="target learning rate")
flags.DEFINE_float("grad_clip", 1.0, help="gradient norm clipping")
flags.DEFINE_integer("total_steps", 400001, help="total training steps")
flags.DEFINE_integer("warmup", 5000, help="learning rate warmup")
flags.DEFINE_integer("batch_size", 32, help="batch size")
flags.DEFINE_integer("num_workers", 4, help="workers of Dataloader")
flags.DEFINE_float("ema_decay", 0.9999, help="ema decay rate")
# Evaluation
flags.DEFINE_integer("eval_step", 1000, help="frequency of eval/wandb logging")
flags.DEFINE_integer("save_step", 10000, help="frequency of checkpoint saves, 0 to disable")
flags.DEFINE_boolean("use_checkpoint", False, help="gradient checkpointing to save VRAM (slower)")
flags.DEFINE_integer("crop_size", 128, help="spatial crop size (square)")
flags.DEFINE_integer("n_gen", 4*64, help="number of samples to generate for eval")
flags.DEFINE_integer("n_ode_steps", 100, help="ODE steps for sample generation")
flags.DEFINE_integer("ema_image_step", 20000, help="frequency of EMA image logging to wandb (0=same as eval_step)")
# Wandb
flags.DEFINE_string("wandb_project", "cfm-pixel", help="wandb project name")
flags.DEFINE_string("run_name", "", help="wandb run name (empty = auto)")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def warmup_lr(step):
    return min(step, FLAGS.warmup) / FLAGS.warmup


def ema(source, target, decay):
    src, tgt = source.state_dict(), target.state_dict()
    for key in src:
        tgt[key].data.copy_(tgt[key].data * decay + src[key].data * (1 - decay))


def infiniteloop(dataloader):
    while True:
        for batch in dataloader:
            #yield batch['img']   # (B, 1, H, W), values in [0, 1]
            cond = batch.get('cond', None)
            if cond is not None: # reshape to (B, n_comps, H, W)
                cond = [c.view(c.shape[0], int(c.shape[1]**0.5), int(c.shape[1]**0.5), c.shape[2]).permute(0,3,1,2) for c in cond]
            yield batch['img'], cond   # (B, 1, H, W), list of (B, cond_dim)


def mmd_rbf(x, y, n_sub=2000):
    """Unbiased MMD² with RBF kernel, median bandwidth heuristic. x, y: (N,D) tensors."""
    if x.size(0) > n_sub: x = x[torch.randperm(x.size(0))[:n_sub]]
    if y.size(0) > n_sub: y = y[torch.randperm(y.size(0))[:n_sub]]
    xy = torch.cat([x, y], dim=0)
    sigma2 = torch.cdist(xy, xy).median().pow(2).clamp(min=1e-6)
    def rbf(a, b): return torch.exp(-torch.cdist(a, b).pow(2) / (2 * sigma2))
    return (rbf(x, x).mean() + rbf(y, y).mean() - 2 * rbf(x, y).mean()).item()


def wasserstein_score(x, y, n_projections=200, n_sub=2000):
    """Sliced Wasserstein on flat images. x, y: (N,D) numpy arrays."""
    rng = np.random.default_rng(0)
    D = x.shape[1]
    projs = rng.standard_normal((D, n_projections))
    projs /= np.linalg.norm(projs, axis=0, keepdims=True)
    px, py = x[:n_sub] @ projs, y[:n_sub] @ projs
    return float(np.mean([wasserstein_distance(px[:, i], py[:, i]) for i in range(n_projections)]))


def compute_metrics(gen_gpu, real_cpu):
    """All metrics on CPU. gen_gpu: (N,1,H,W) GPU tensor in [0,1]."""
    from scipy.stats import skew, kurtosis
    gen  = (gen_gpu > 0.5).float().cpu()                        # binarize
    n_sub = min(2000, gen.shape[0], real_cpu.shape[0])
    g_flat_t = gen.view(gen.shape[0], -1)
    r_flat_t = real_cpu.float().view(real_cpu.shape[0], -1)
    g_flat = g_flat_t[:n_sub].numpy()
    r_flat = r_flat_t[:n_sub].numpy()
    # pitch marginals: mean over (batch, channel, time) → (H,)
    g_pitch = gen.mean(dim=(0, 1, 3)).numpy()
    r_pitch = real_cpu.mean(dim=(0, 1, 3)).numpy()
    metrics = {
        "note_density_gen":   gen.mean().item(),
        "note_density_real":  real_cpu.mean().item(),
        "gen_mean":           float(g_flat.mean()),
        "gen_std":            float(g_flat.std()),
        "gen_skew":           float(skew(g_flat.ravel())),
        "gen_kurt":           float(kurtosis(g_flat.ravel())),
        "real_mean":          float(r_flat.mean()),
        "real_std":           float(r_flat.std()),
        "real_skew":          float(skew(r_flat.ravel())),
        "real_kurt":          float(kurtosis(r_flat.ravel())),
        "mmd":                mmd_rbf(g_flat_t[:n_sub], r_flat_t[:n_sub]),
        "wasserstein":        wasserstein_score(g_flat, r_flat),
        "wasserstein_pitch":  float(wasserstein_distance(r_pitch, g_pitch)),
    }
    return metrics, gen

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

def generate_samples(model, savedir, step, net_="normal", n=64, crop_size=128, mlcond=None, cfg_strength=1.0):
    model.eval()
    model_ = copy.deepcopy(model)
    if mlcond is not None: model_ = CFGWrapper(model_, mlcond, cfg_strength)
    node_ = NeuralODE(model_, solver="euler", sensitivity="adjoint")
    with torch.no_grad():
        traj = node_.trajectory(
            torch.randn(n, 1, crop_size, crop_size, device=device),
            t_span=torch.linspace(0, 1, FLAGS.n_ode_steps, device=device),
        )
        traj = traj[-1, :].view([-1, 1, crop_size, crop_size]).clip(-1, 1)
        traj = traj / 2 + 0.5   # [0, 1]
    save_image(traj, os.path.join(savedir, f"{net_}_step_{step}.png"), nrow=8)
    model.train()
    return traj   # GPU tensor, [0,1]


def mlc_dropout(mlcond, p_uncond=0.1, p_keep_level=0.1, p_zero_level: list | None = None, p_patch: list | None = None):
    """ 'dropout' for multilevel conditioning"""
    if mlcond is None: return None 
    if p_zero_level is None:  p_zero_level = [0.2] * len(mlcond)
    if p_patch is None:  p_patch = [0.2] * len(mlcond)
 
    case = torch.rand(()).item()
    if case < p_uncond:  # case 1:
        return None   # classic CFG, 10% of time, drop all conditioning
    elif case < p_uncond + p_keep_level: # case 2: single level only
        keep = torch.randint(len(mlcond), ()).item()
        for i in range(len(mlcond)):
            if i != keep: mlcond[i] *= 0    # case 2: keep only single level, zero all others.
    else: # case 3: per-level/patch dropout
        B = mlcond[0].shape[0]               # batch size
        for i, cond in enumerate(mlcond):
            drop = torch.rand(B) < p_zero_level[i]
            if drop.any(): cond[drop] = 0   # drop entire level 
            elif (apply := torch.rand(B) < p_patch[i]).any(): # drop patches within levels
                H, W = cond.shape[2], cond.shape[3]
                mask = (torch.rand(B, 1, H, W, device=cond.device) > 0.5).float()
                mask[~apply] = 1.0
                mlcond[i] *= mask
    return mlcond



def train(argv):
    crop_size = FLAGS.crop_size
    print(f"lr={FLAGS.lr}, steps={FLAGS.total_steps}, ema={FLAGS.ema_decay}, model={FLAGS.model}")

    wandb.init(project=FLAGS.wandb_project, name=FLAGS.run_name or None,
               config=dict(model=FLAGS.model, lr=FLAGS.lr, batch_size=FLAGS.batch_size,
                           num_channel=FLAGS.num_channel, total_steps=FLAGS.total_steps,
                           ema_decay=FLAGS.ema_decay, crop_size=crop_size,
                           save_step=FLAGS.save_step))

    if FLAGS.data_mode == "preencoded":
        train_dataset = PreencodedImageDataset(FLAGS.preencoded_dir, crop_size=crop_size, split='train')
        val_dataset   = PreencodedImageDataset(FLAGS.preencoded_dir, crop_size=crop_size, split='val')
    else:
        train_dataset = AnchorDataset(
            image_dataset_dir=FLAGS.data_dir,
            crop_size=(crop_size, crop_size),
            split='train', aug_y_max=12, sigma=7,
        )
        val_dataset = AnchorDataset(
            image_dataset_dir=FLAGS.data_dir,
            crop_size=(crop_size, crop_size),
            split='val', verbose=False,
        )
    shuffle = not isinstance(train_dataset, torch.utils.data.IterableDataset)
    dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=FLAGS.batch_size,
        shuffle=shuffle, num_workers=FLAGS.num_workers, drop_last=True,
    )
    datalooper = infiniteloop(dataloader)

    # Fixed real reference batch for metrics (CPU, loaded once)
    val_shuffle = not isinstance(val_dataset, torch.utils.data.IterableDataset)
    val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=FLAGS.n_gen, shuffle=val_shuffle)
    real_batch = next(iter(val_dl))  # (n_gen, 1, H, W) CPU, [0,1]
    real_ref = real_batch['img']
    cond_ref = real_batch.get('cond', None)
    if cond_ref is not None:
        cond_ref = [c.view(c.shape[0], int(c.shape[1]**0.5), int(c.shape[1]**0.5), c.shape[2])
                    .permute(0,3,1,2) for c in cond_ref]

    mlcond_shapes = {c.shape[2]: (i, c.shape[1]) for i, c in enumerate(cond_ref)} if cond_ref is not None else None

    net_model = UNetModelWrapperMLC(
        dim=(1, crop_size, crop_size),
        num_res_blocks=2,
        num_channels=FLAGS.num_channel,
        channel_mult=[1, 2, 2, 2, 2],
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="16",
        dropout=0.1,
        use_checkpoint=FLAGS.use_checkpoint,
        mlcond_shapes=mlcond_shapes,
    ).to(device)

    ema_model = copy.deepcopy(net_model)
    optim = torch.optim.Adam(net_model.parameters(), lr=FLAGS.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)

    model_size = sum(p.data.nelement() for p in net_model.parameters())
    print(f"Model params: {model_size / 1024 / 1024:.2f} M")
    wandb.config.update({"n_params_M": model_size / 1024 / 1024})

    sigma = 0.0
    if FLAGS.model == "otcfm":
        FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
    elif FLAGS.model == "icfm":
        FM = ConditionalFlowMatcher(sigma=sigma)
    elif FLAGS.model == "fm":
        FM = TargetConditionalFlowMatcher(sigma=sigma)
    elif FLAGS.model == "si":
        FM = VariancePreservingConditionalFlowMatcher(sigma=sigma)
    else:
        raise NotImplementedError(f"Unknown model {FLAGS.model}")

    savedir = os.path.join(FLAGS.output_dir, FLAGS.model)
    os.makedirs(savedir, exist_ok=True)

    pbar = trange(FLAGS.total_steps, dynamic_ncols=True)
    for step in reloading(pbar):
            optim.zero_grad()
            x1, cond = next(datalooper)
            #cond = mlc_dropout(cond, p_uncond=0.1) 
            x1 = x1.to(device)
            if cond is not None:
                cond = [c.to(device) for c in cond]
            x1 = x1 * 2 - 1          # [0, 1] → [-1, 1]
            x0 = torch.randn_like(x1)
            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
            vt = net_model(t, xt, mlcond=cond)
            loss = torch.mean((vt - ut) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net_model.parameters(), FLAGS.grad_clip)
            optim.step()
            sched.step()
            ema(net_model, ema_model, FLAGS.ema_decay)
            loss_val = loss.item()
            pbar.set_postfix(loss=f"{loss_val:.4f}")
            if step % 100 == 0:
                print(f"step {step} loss {loss_val:.4f}", flush=True)
                wandb.log({"train/loss": loss_val, "train/lr": sched.get_last_lr()[0],
                           "train/cond_proj_norm": net_model.mlcond_global_proj.weight.norm().item()}, step=step)

            # SAMPLING & EVAL
            if FLAGS.eval_step > 0 and step > 0 and step % FLAGS.eval_step == 0:
                ema_img_step = FLAGS.ema_image_step if FLAGS.ema_image_step > 0 else FLAGS.eval_step
                log_ema_img  = (step % ema_img_step == 0)
                #gen_normal = generate_samples(net_model, savedir, step, "normal", crop_size=crop_size)
                cond_gpu = [c.to(device) for c in cond_ref] if cond_ref is not None else None
                log_dict = {}
                for gen_type, cond in zip(["uncond", "cond"], [None, cond_gpu]):
                    gen_normal = generate_samples(net_model, savedir, step, gen_type, crop_size=crop_size, mlcond=cond)
                    m_n, gen_bin_n = compute_metrics(gen_normal, real_ref)
                    for k, v in m_n.items(): log_dict[f"eval/{gen_type}/{k}"] = v
                    log_dict[f"media/normal_{gen_type}"] = wandb.Image(make_grid(gen_normal.cpu(), nrow=8),
                                                        caption=f"{gen_type} step {step}")
                    log_dict[f"media/normal_{gen_type}_binarized"] = wandb.Image(make_grid(gen_bin_n, nrow=8),
                                                                    caption=f"{gen_type} binarized step {step}")
                    del gen_normal, gen_bin_n
                if log_ema_img:
                    gen_ema = generate_samples(ema_model, savedir, step, "ema", crop_size=crop_size, mlcond=cond_gpu)
                    m_e, gen_bin_e = compute_metrics(gen_ema, real_ref)
                    for k, v in m_e.items(): log_dict[f"eval/ema/{k}"] = v
                    log_dict["media/ema"] = wandb.Image(make_grid(gen_ema.cpu(), nrow=8),
                                                        caption=f"ema step {step}")
                    log_dict["media/ema_binarized"] = wandb.Image(make_grid(gen_bin_e, nrow=8),
                                                                   caption=f"ema binarized step {step}")
                    del gen_ema, gen_bin_e
                wandb.log(log_dict, step=step)
            if FLAGS.save_step > 0 and step > 0 and step % FLAGS.save_step == 0:
                torch.save(
                    {"net_model": net_model.state_dict(), "ema_model": ema_model.state_dict(),
                     "sched": sched.state_dict(), "optim": optim.state_dict(), "step": step},
                    os.path.join(savedir, f"{FLAGS.model}_midi_weights_step_{step}.pt"),
                )

    wandb.finish()


if __name__ == "__main__":
    app.run(train)

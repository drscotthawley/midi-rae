# Flow matching on L5 PCA latent embeddings, reshaped as [3, 32, 32] "images".
# Isolates pixel-space vs. latent-space as the variable vs. train_cfm_midi_pixels.py.
# Same UNet, same torchcfm training loop — only the data changes.
#
# L5 PCA: n_comp=3, n_patches=1024 (32x32 grid) → reshape to [3, 32, 32]
# Uses unified pca chunk files: {split}_chunk*_pca.pt, key 'L5'
#
# Usage (from the directory containing this script, on lecun):
#   PYTHONPATH=$HOME/runs/midi-rae/<run> nohup python3 train_cfm_latent_l5.py \
#     --model otcfm --lr 2e-4 --ema_decay 0.9999 --batch_size 128 \
#     --total_steps 400001 --save_step 5000 &

import copy
import glob
import os

import torch
from absl import app, flags
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
from torchdyn.core import NeuralODE
from tqdm import trange

from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
)
from torchcfm.models.unet.unet import UNetModelWrapper

FLAGS = flags.FLAGS

flags.DEFINE_string("model", "otcfm", help="flow matching model type")
flags.DEFINE_string("output_dir", "./results/cfm_latent_l5/", help="output directory")
flags.DEFINE_string("pca_dir", "~/datasets/POP909_pca_exp26/", help="dir with *_pca.pt chunks")
flags.DEFINE_integer("num_channel", 128, help="base channel of UNet")
flags.DEFINE_float("lr", 2e-4, help="target learning rate")
flags.DEFINE_float("grad_clip", 1.0, help="gradient norm clipping")
flags.DEFINE_integer("total_steps", 400001, help="total training steps")
flags.DEFINE_integer("warmup", 5000, help="learning rate warmup")
flags.DEFINE_integer("batch_size", 128, help="batch size")
flags.DEFINE_integer("num_workers", 4, help="workers of Dataloader")
flags.DEFINE_float("ema_decay", 0.9999, help="ema decay rate")
flags.DEFINE_integer("save_step", 5000, help="frequency of saving checkpoints")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

L5_N_COMP    = 3    # PCA components at L5
L5_N_PATCHES = 1024 # 32x32 grid
L5_GRID      = 32


class L5PCADataset(Dataset):
    """Loads L5 PCA embeddings from unified chunk files, reshaped as [3, 32, 32]."""
    def __init__(self, pca_dir, split='train'):
        pca_dir = os.path.expandvars(os.path.expanduser(pca_dir))
        files = sorted(glob.glob(os.path.join(pca_dir, f'{split}_chunk*_pca.pt')))
        assert files, f"No {split}_chunk*_pca.pt found in {pca_dir}"
        print(f"Loading L5 from {len(files)} {split} chunks...", flush=True)
        chunks = []
        for f in files:
            data = torch.load(f, weights_only=False)
            l5 = data['L5'].float()   # [N, n_patches, n_comp] = [N, 1024, 3]
            chunks.append(l5)
        self.data = torch.cat(chunks, dim=0)  # [N_total, 1024, 3]
        print(f"Loaded {len(self.data)} samples, L5 shape per sample: {self.data.shape[1:]}", flush=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # [1024, 3] → [3, 32, 32]
        return self.data[idx].T.reshape(L5_N_COMP, L5_GRID, L5_GRID)


def warmup_lr(step):
    return min(step, FLAGS.warmup) / FLAGS.warmup


def ema(source, target, decay):
    src, tgt = source.state_dict(), target.state_dict()
    for key in src:
        tgt[key].data.copy_(tgt[key].data * decay + src[key].data * (1 - decay))


def infiniteloop(dataloader):
    while True:
        for x in dataloader:
            yield x


def generate_samples(model, savedir, step, net_="normal", n=64):
    model.eval()
    model_ = copy.deepcopy(model)
    node_ = NeuralODE(model_, solver="euler", sensitivity="adjoint")
    with torch.no_grad():
        traj = node_.trajectory(
            torch.randn(n, L5_N_COMP, L5_GRID, L5_GRID, device=device),
            t_span=torch.linspace(0, 1, 100, device=device),
        )
        traj = traj[-1, :].view([-1, L5_N_COMP, L5_GRID, L5_GRID])
        # Save each PCA component as a separate channel — normalize per-image for visibility
        traj_vis = traj - traj.flatten(1).min(1)[0].view(-1, 1, 1, 1)
        traj_vis = traj_vis / (traj_vis.flatten(1).max(1)[0].view(-1, 1, 1, 1) + 1e-8)
    save_image(traj_vis, os.path.join(savedir, f"{net_}_step_{step}.png"), nrow=8)
    model.train()


def train(argv):
    print(f"lr={FLAGS.lr}, steps={FLAGS.total_steps}, ema={FLAGS.ema_decay}, model={FLAGS.model}")

    dataset = L5PCADataset(FLAGS.pca_dir, split='train')
    dataloader = DataLoader(dataset, batch_size=FLAGS.batch_size, shuffle=True,
                            num_workers=FLAGS.num_workers, drop_last=True)
    datalooper = infiniteloop(dataloader)

    # 32px config — same as CIFAR10, just 3 channels of PCA instead of RGB
    net_model = UNetModelWrapper(
        dim=(L5_N_COMP, L5_GRID, L5_GRID),
        num_res_blocks=2,
        num_channels=FLAGS.num_channel,
        channel_mult=[1, 2, 2, 2],
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="16",
        dropout=0.1,
    ).to(device)

    ema_model = copy.deepcopy(net_model)
    optim = torch.optim.Adam(net_model.parameters(), lr=FLAGS.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)

    model_size = sum(p.data.nelement() for p in net_model.parameters())
    print(f"Model params: {model_size / 1024 / 1024:.2f} M")

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

    with trange(FLAGS.total_steps, dynamic_ncols=True) as pbar:
        for step in pbar:
            optim.zero_grad()
            x1 = next(datalooper).to(device)  # [B, 3, 32, 32] — already normalized by PCA whitening
            x0 = torch.randn_like(x1)
            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
            vt = net_model(t, xt)
            loss = torch.mean((vt - ut) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net_model.parameters(), FLAGS.grad_clip)
            optim.step()
            sched.step()
            ema(net_model, ema_model, FLAGS.ema_decay)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            if FLAGS.save_step > 0 and step % FLAGS.save_step == 0:
                generate_samples(net_model, savedir, step, net_="normal")
                generate_samples(ema_model, savedir, step, net_="ema")
                torch.save(
                    {
                        "net_model": net_model.state_dict(),
                        "ema_model": ema_model.state_dict(),
                        "sched": sched.state_dict(),
                        "optim": optim.state_dict(),
                        "step": step,
                    },
                    os.path.join(savedir, f"{FLAGS.model}_l5pca_weights_step_{step}.pt"),
                )


if __name__ == "__main__":
    app.run(train)

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
import pickle
import sys

import torch
import numpy as np
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
flags.DEFINE_string("dec_ckpt", "", help="path to SwinDecoder checkpoint for piano roll visualization")
flags.DEFINE_string("midi_rae_dir", "", help="path to midi_rae package dir (parent of midi_rae/); added to sys.path")
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


def load_decoder():
    """Load SwinDecoder + L5 PCA model for piano roll visualization. Returns (decoder, pca) or (None, None)."""
    if not FLAGS.dec_ckpt or not FLAGS.midi_rae_dir:
        return None, None
    if FLAGS.midi_rae_dir not in sys.path:
        sys.path.insert(0, os.path.expanduser(FLAGS.midi_rae_dir))
    from midi_rae.swin import SwinDecoder
    from midi_rae.utils import load_checkpoint
    ckpt_path = os.path.expanduser(FLAGS.dec_ckpt)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = ckpt['config']
    decoder = SwinDecoder(**{k: cfg['model'][k] for k in
                             ['patch_h','patch_w','embed_dim','depths','num_heads',
                              'window_size','mlp_ratio','drop_path_rate',
                              'dec_depths','dec_num_heads']},
                          in_channels=cfg['data']['in_channels'],
                          image_size=cfg['data']['image_size'])
    load_checkpoint(decoder, ckpt_path, weights_only=False)
    decoder = decoder.to(device).eval()
    # Load L5 PCA model
    pca_dir = os.path.expanduser(FLAGS.pca_dir)
    pca_matches = glob.glob(os.path.join(pca_dir, 'pca_L5_n*.pkl'))
    pca = None
    if pca_matches:
        path = max(pca_matches, key=os.path.getmtime)
        with open(path, 'rb') as f:
            pca = pickle.load(f)
        print(f"Loaded L5 PCA from {os.path.basename(path)}")
    return decoder, pca


def decode_l5_to_piano_rolls(latents, decoder, pca):
    """latents: [B, 3, 32, 32] PCA → inverse PCA → enc_out → decoder → [B, 1, 128, 128]

    Levels are coarsest-first (index 0 = L0 = 1×1×256, index 5 = L5 = 32×32×8).
    L0-L4 are filled with zeros; dec40 was trained with mask tokens for these levels.
    """
    from midi_rae.core import PatchState, HierarchicalPatchState, EncoderOutput
    B = latents.shape[0]
    # [B, 3, 32, 32] → [B*1024, 3]
    flat_pca = latents.permute(0, 2, 3, 1).reshape(B * L5_N_PATCHES, L5_N_COMP).cpu().numpy()
    flat_emb = pca.inverse_transform(flat_pca)  # [B*1024, 8]
    emb_l5 = torch.tensor(flat_emb, dtype=torch.float32, device=device).reshape(B, L5_N_PATCHES, -1)
    D_L5 = emb_l5.shape[-1]
    n_levels = 6
    # dims come from the decoder's lateral layer input sizes, not hardcoded
    dec_dims = [lp.weight.shape[1] for lp in decoder.laterals]  # coarsest-first
    levels = []
    for i in range(n_levels):
        g_i = 2 ** i
        n_i = g_i * g_i
        dim_i = dec_dims[i]
        rows, cols = torch.meshgrid(torch.arange(g_i), torch.arange(g_i), indexing='ij')
        pos_i = torch.stack([rows.flatten(), cols.flatten()], dim=1).float().to(device)
        if i == 5:
            levels.append(PatchState(emb=emb_l5, pos=pos_i, non_empty=None, mae_mask=None))
        else:
            dummy_emb = torch.zeros(B, n_i, dim_i, device=device)
            levels.append(PatchState(emb=dummy_emb, pos=pos_i, non_empty=None, mae_mask=None))
    enc_out = EncoderOutput(patches=HierarchicalPatchState(levels=levels),
                            full_pos=None, full_non_empty=None, mae_mask=None)
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        logits = decoder(enc_out)
    return torch.sigmoid(logits.float())


def generate_samples(model, savedir, step, net_="normal", n=64, decoder=None, pca=None):
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
    if decoder is not None and pca is not None:
        piano_rolls = decode_l5_to_piano_rolls(traj, decoder, pca)  # [n, 1, 128, 128]
        save_image(piano_rolls, os.path.join(savedir, f"{net_}_decoded_step_{step}.png"), nrow=8)
    model.train()


def train(argv):
    print(f"lr={FLAGS.lr}, steps={FLAGS.total_steps}, ema={FLAGS.ema_decay}, model={FLAGS.model}")
    decoder, pca = load_decoder()
    if decoder is not None:
        print("Decoder loaded — will generate piano roll visualizations.")
    else:
        print("No decoder — will save PCA latent visualizations only.")

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
                generate_samples(net_model, savedir, step, net_="normal", decoder=decoder, pca=pca)
                generate_samples(ema_model, savedir, step, net_="ema", decoder=decoder, pca=pca)
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

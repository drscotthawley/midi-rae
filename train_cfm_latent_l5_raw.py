# Flow matching on raw L5 encoder embeddings, reshaped as [8, 32, 32] "images".
# Companion to train_cfm_latent_l5.py (PCA version) — isolates PCA vs raw as the variable.
#
# L5 raw: n_dim=8, n_patches=1024 (32x32 grid) → reshape to [8, 32, 32]
# Uses preencoded chunk files: {split}_chunk*.pt, emb1[5] = L5 tensor [N, 1024, 8]
#
# Usage:
#   nohup ~/envs/torchcfm/bin/python3 train_cfm_latent_l5_raw.py \
#     --model otcfm --lr 2e-4 --ema_decay 0.9999 --batch_size 128 \
#     --total_steps 400001 --save_step 5000 \
#     --dec_ckpt $HOME/runs/midi-rae/dec40_l5only_NtjGJJ/checkpoints/SwinDecoder_dec40_l5only_NtjGJJ_best.pt \
#     --midi_rae_dir $HOME/runs/midi-rae/dec40_l5only_NtjGJJ &

import copy
import glob
import os
import sys

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
flags.DEFINE_string("output_dir", "./results/cfm_latent_l5_raw/", help="output directory")
flags.DEFINE_string("encoded_dir", "~/datasets/POP909_encoded_exp26/", help="dir with preencoded chunk files")
flags.DEFINE_string("dec_ckpt", "", help="path to SwinDecoder checkpoint for piano roll visualization")
flags.DEFINE_string("midi_rae_dir", "", help="parent dir of midi_rae/ package; added to sys.path")
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

L5_N_DIM    = 8     # raw embedding dim at L5
L5_N_PATCHES = 1024 # 32x32 grid
L5_GRID      = 32


class L5RawDataset(Dataset):
    """Lazily loads raw L5 encoder embeddings from preencoded chunk files, reshaped as [8, 32, 32].
    Chunks are loaded on demand to avoid exhausting system RAM."""
    def __init__(self, encoded_dir, split='train'):
        encoded_dir = os.path.expandvars(os.path.expanduser(encoded_dir))
        self.files = sorted(glob.glob(os.path.join(encoded_dir, f'{split}_chunk*.pt')))
        assert self.files, f"No {split}_chunk*.pt found in {encoded_dir}"
        print(f"Found {len(self.files)} {split} chunks (lazy loading).", flush=True)

        # Scan all chunks to build accurate index
        print(f"Scanning chunks...", flush=True)
        self.index = []
        for fi, f in enumerate(self.files):
            data = torch.load(f, weights_only=False)
            for item_i, item in enumerate(data):
                n = item['emb1'][5].shape[0]
                self.index.extend([(fi, item_i, s) for s in range(n)])
        print(f"Total samples: {len(self.index)}", flush=True)

        # Compute normalization stats from first chunk only (fast approximation)
        data0 = torch.load(self.files[0], weights_only=False)
        sample = torch.cat([item['emb1'][5].float() for item in data0], dim=0)
        flat = sample.reshape(-1, L5_N_DIM)
        self.mean = flat.mean(0)
        self.std  = flat.std(0).clamp(min=1e-6)
        print(f"Normalization stats from chunk 0 (approx).", flush=True)

        # Cache current loaded chunk
        self._cache_fi = None
        self._cache_data = None

    def _load_chunk(self, fi):
        if self._cache_fi != fi:
            self._cache_data = torch.load(self.files[fi], weights_only=False)
            self._cache_fi = fi

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        fi, item_i, s = self.index[idx]
        self._load_chunk(fi)
        emb = self._cache_data[item_i]['emb1'][5][s].float()  # [1024, 8]
        emb = (emb - self.mean) / self.std
        return emb.T.reshape(L5_N_DIM, L5_GRID, L5_GRID)


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
    """Load SwinDecoder for piano roll visualization. Returns decoder or None."""
    if not FLAGS.dec_ckpt or not FLAGS.midi_rae_dir:
        return None
    if FLAGS.midi_rae_dir not in sys.path:
        sys.path.insert(0, os.path.expanduser(FLAGS.midi_rae_dir))
    from midi_rae.swin import SwinDecoder
    from midi_rae.utils import load_checkpoint
    ckpt_path = os.path.expanduser(FLAGS.dec_ckpt)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = ckpt['config']
    m = cfg['model']
    decoder = SwinDecoder(
        patch_h=m['patch_h'], patch_w=m['patch_w'], embed_dim=m['embed_dim'],
        depths=m['depths'], num_heads=m['num_heads'],
        window_size=m['window_size'], mlp_ratio=m['mlp_ratio'],
        drop_path_rate=m['drop_path_rate'],
    )
    load_checkpoint(decoder, ckpt_path, weights_only=False)
    decoder = decoder.to(device).eval()
    print(f"Loaded decoder from {os.path.basename(ckpt_path)}")
    return decoder


def decode_l5_to_piano_rolls(latents, decoder, mean, std):
    """latents: [B, 8, 32, 32] normalized → unnormalize → enc_out → decoder → [B, 1, 128, 128]"""
    from midi_rae.core import PatchState, HierarchicalPatchState, EncoderOutput
    B = latents.shape[0]
    # Unnormalize: [B, 8, 32, 32] → [B, 1024, 8]
    emb_l5 = latents.permute(0, 2, 3, 1).reshape(B, L5_N_PATCHES, L5_N_DIM)
    emb_l5 = emb_l5 * std.to(device) + mean.to(device)
    # Build enc_out: L0-L4 zeros (dec40 mask tokens handle them), L5 = generated
    dec_dims = [lp.weight.shape[1] for lp in decoder.laterals]  # coarsest-first
    n_levels = 6
    levels = []
    for i in range(n_levels):
        g_i = 2 ** i
        n_i = g_i * g_i
        rows, cols = torch.meshgrid(torch.arange(g_i), torch.arange(g_i), indexing='ij')
        pos_i = torch.stack([rows.flatten(), cols.flatten()], dim=1).float().to(device)
        if i == 5:
            levels.append(PatchState(emb=emb_l5, pos=pos_i, non_empty=None, mae_mask=None))
        else:
            dummy_emb = torch.zeros(B, n_i, dec_dims[i], device=device)
            levels.append(PatchState(emb=dummy_emb, pos=pos_i, non_empty=None, mae_mask=None))
    enc_out = EncoderOutput(patches=HierarchicalPatchState(levels=levels),
                            full_pos=None, full_non_empty=None, mae_mask=None)
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        logits = decoder(enc_out)
    return torch.sigmoid(logits.float())


def generate_samples(model, savedir, step, net_="normal", n=64, decoder=None, mean=None, std=None):
    model.eval()
    model_ = copy.deepcopy(model)
    node_ = NeuralODE(model_, solver="euler", sensitivity="adjoint")
    with torch.no_grad():
        traj = node_.trajectory(
            torch.randn(n, L5_N_DIM, L5_GRID, L5_GRID, device=device),
            t_span=torch.linspace(0, 1, 100, device=device),
        )
        traj = traj[-1, :].view([-1, L5_N_DIM, L5_GRID, L5_GRID])
        # Save first 3 dims as RGB for visibility
        traj_vis = traj[:, :3]
        traj_vis = traj_vis - traj_vis.flatten(1).min(1)[0].view(-1, 1, 1, 1)
        traj_vis = traj_vis / (traj_vis.flatten(1).max(1)[0].view(-1, 1, 1, 1) + 1e-8)
    save_image(traj_vis, os.path.join(savedir, f"{net_}_step_{step}.png"), nrow=8)
    if decoder is not None:
        piano_rolls = decode_l5_to_piano_rolls(traj, decoder, mean, std)
        save_image(piano_rolls, os.path.join(savedir, f"{net_}_decoded_step_{step}.png"), nrow=8)
    model.train()


def train(argv):
    print(f"lr={FLAGS.lr}, steps={FLAGS.total_steps}, ema={FLAGS.ema_decay}, model={FLAGS.model}")
    decoder = load_decoder()
    if decoder is not None:
        print("Decoder loaded — will generate piano roll visualizations.")
    else:
        print("No decoder — will save latent visualizations only.")

    dataset = L5RawDataset(FLAGS.encoded_dir, split='train')
    mean, std = dataset.mean, dataset.std
    dataloader = DataLoader(dataset, batch_size=FLAGS.batch_size, shuffle=True,
                            num_workers=FLAGS.num_workers, drop_last=True)
    datalooper = infiniteloop(dataloader)

    # 32px config — same as PCA version, 8 channels instead of 3
    net_model = UNetModelWrapper(
        dim=(L5_N_DIM, L5_GRID, L5_GRID),
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
            x1 = next(datalooper).to(device)  # [B, 8, 32, 32]
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
                generate_samples(net_model, savedir, step, net_="normal", decoder=decoder, mean=mean, std=std)
                generate_samples(ema_model, savedir, step, net_="ema", decoder=decoder, mean=mean, std=std)
                torch.save(
                    {
                        "net_model": net_model.state_dict(),
                        "ema_model": ema_model.state_dict(),
                        "sched": sched.state_dict(),
                        "optim": optim.state_dict(),
                        "step": step,
                        "mean": mean,
                        "std": std,
                    },
                    os.path.join(savedir, f"{FLAGS.model}_l5raw_weights_step_{step}.pt"),
                )


if __name__ == "__main__":
    app.run(train)

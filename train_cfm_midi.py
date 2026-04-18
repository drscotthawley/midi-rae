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
from torchvision.utils import make_grid
from torchdyn.core import NeuralODE
from tqdm import trange

from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
)
from torchcfm.models.unet.unet import UNetModelWrapper

from midi_rae.data import AnchorDataset

FLAGS = flags.FLAGS

flags.DEFINE_string("model", "otcfm", help="flow matching model type")
flags.DEFINE_string("output_dir", "./results/cfm_midi/", help="output directory")
flags.DEFINE_string("data_dir", "~/datasets/POP909_images_basic/", help="piano roll image dir")
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
flags.DEFINE_integer("save_step", 20000, help="frequency of saving checkpoints, 0 to disable")
flags.DEFINE_integer("crop_size", 128, help="spatial crop size (square)")
# Logging
flags.DEFINE_string("wandb_project", "midi-rae-flow", help="wandb project name")
flags.DEFINE_string("run_name", "", help="wandb run name (empty = auto)")
flags.DEFINE_integer("log_every", 100, help="log loss to wandb every N steps")

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
            yield batch['img']   # (B, 1, H, W), values in [0, 1]


def generate_samples(model, step, net_="normal", n=64, crop_size=128):
    """Generate samples on GPU, move to CPU for viz — returns wandb.Image."""
    model.eval()
    model_ = copy.deepcopy(model).cpu()  # run ODE on CPU to avoid VRAM spike
    node_ = NeuralODE(model_, solver="euler", sensitivity="adjoint")
    with torch.no_grad():
        traj = node_.trajectory(
            torch.randn(n, 1, crop_size, crop_size),   # CPU
            t_span=torch.linspace(0, 1, 100),
        )
        traj = traj[-1].view(-1, 1, crop_size, crop_size).clip(-1, 1)
        traj = traj / 2 + 0.5  # [0,1]
    grid = make_grid(traj, nrow=8)           # (3, H, W) CPU tensor
    img_np = (grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    del model_, node_, traj, grid
    model.train()
    return wandb.Image(img_np, caption=f"{net_} step {step}")


def train(argv):
    crop_size = FLAGS.crop_size
    print(f"lr={FLAGS.lr}, steps={FLAGS.total_steps}, ema={FLAGS.ema_decay}, model={FLAGS.model}")

    wandb.init(
        project=FLAGS.wandb_project,
        name=FLAGS.run_name or None,
        config=dict(
            model=FLAGS.model, lr=FLAGS.lr, batch_size=FLAGS.batch_size,
            num_channel=FLAGS.num_channel, total_steps=FLAGS.total_steps,
            ema_decay=FLAGS.ema_decay, crop_size=crop_size,
        ),
    )

    # Dataset
    train_dataset = AnchorDataset(
        image_dataset_dir=FLAGS.data_dir,
        crop_size=(crop_size, crop_size),
        split='train',
        aug_y_max=12,
        sigma=7,
    )
    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        num_workers=FLAGS.num_workers,
        drop_last=True,
    )
    datalooper = infiniteloop(dataloader)

    # UNet
    net_model = UNetModelWrapper(
        dim=(1, crop_size, crop_size),
        num_res_blocks=2,
        num_channels=FLAGS.num_channel,
        channel_mult=[1, 2, 2, 2, 2],
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

    with trange(FLAGS.total_steps, dynamic_ncols=True) as pbar:
        for step in pbar:
            optim.zero_grad()
            x1 = next(datalooper).to(device)
            x1 = x1 * 2 - 1          # [0,1] → [-1,1]
            x0 = torch.randn_like(x1)
            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
            vt = net_model(t, xt)
            loss = torch.mean((vt - ut) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net_model.parameters(), FLAGS.grad_clip)
            optim.step()
            sched.step()
            ema(net_model, ema_model, FLAGS.ema_decay)
            loss_val = loss.item()
            pbar.set_postfix(loss=f"{loss_val:.4f}")

            if step % FLAGS.log_every == 0:
                wandb.log({"train/loss": loss_val,
                           "train/lr": sched.get_last_lr()[0]}, step=step)

            if FLAGS.save_step > 0 and step % FLAGS.save_step == 0:
                imgs = {
                    "samples/normal": generate_samples(net_model, step, "normal", crop_size=crop_size),
                    "samples/ema":    generate_samples(ema_model,  step, "ema",    crop_size=crop_size),
                }
                wandb.log(imgs, step=step)
                torch.save(
                    {
                        "net_model": net_model.state_dict(),
                        "ema_model": ema_model.state_dict(),
                        "sched": sched.state_dict(),
                        "optim": optim.state_dict(),
                        "step": step,
                    },
                    os.path.join(savedir, f"{FLAGS.model}_midi_weights_step_{step}.pt"),
                )

    wandb.finish()


if __name__ == "__main__":
    app.run(train)

"""Inference harness for the midi-rae conditional flow demo.

Pipeline (conditional generation, "e2e" latent space):

    input piano-roll image (128x128 binary)
        -> SwinEncoder (e2e_recon)         -> per-level patch embeddings
        -> per-level PCA.transform (L0/L1/L2) -> coarse conditioning vector  x_cond
        -> [optional] zero chosen coarse levels of x_cond  (level dropout)
        -> ConditionalFineFlowModel, Euler-integrated fine ODE conditioned on x_cond
                                           -> generated fine PCA embeddings (L3/L4/L5)
        -> inverse-PCA (coarse x_cond + generated fine) -> EncoderOutput
        -> SwinDecoder (e2e_recon) -> binarized 128x128 piano roll

This mirrors the teacher-forcing eval path (`_gen_chunked_tf` in train_flow.py):
real coarse embeddings as conditioning, plain-Gaussian fine source, 20-step Euler.
No coarse flow model is involved (the e2e run trained fine-only with teacher forcing).

Nothing here edits the midi_rae package; it is a pure consumer that imports it.
"""
import os
import sys
import glob
import pickle
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Make the midi_rae package importable when running from the demo dir.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from midi_rae.swin import SwinEncoder, SwinDecoder                       # noqa: E402
from midi_rae.train_flow import ConditionalFineFlowModel, sample_source, warp_time  # noqa: E402
from midi_rae.train_flow import decode_flow_to_piano_rolls              # noqa: E402
from midi_rae.utils import load_checkpoint                              # noqa: E402

HERE = Path(__file__).resolve().parent
CKPT_DIR = HERE / "checkpoints"
PCA_DIR = CKPT_DIR / "pca"

# --- e2e stack constants (from the discovered config_swin_e2e run config + run.log) ---
IMAGE_SIZE = 128
PATCH = 4
EMBED_DIM = 8
DEPTHS = [2, 2, 2, 6, 2, 1]
NUM_HEADS = [2, 2, 2, 4, 8, 16]
DEC_DEPTHS = [4, 4, 4, 6, 2, 1]
DEC_NUM_HEADS = [2, 2, 2, 4, 8, 16]
WINDOW_SIZE = 4
MLP_RATIO = 4.0

COARSE_LEVELS = [0, 1, 2]   # conditioning levels (droppable)
FINE_LEVELS = [3, 4, 5]     # generated levels

# ConditionalFineFlowModel hyperparameters (from flow: block of the e2e config)
FLOW_H_DIM = 128
FLOW_N_LAYERS = 6
FLOW_T_DIM = 64
FINE_ATTN_LAYERS = 2
FINE_LATERAL_ATTN_LAYERS = 1
WARP_S = 0.5

ENCODER_CKPT = CKPT_DIR / "SwinEncoder_e2e_recon_EVVFa3_best.pt"
DECODER_CKPT = CKPT_DIR / "SwinDecoder_e2e_recon_EVVFa3_best.pt"
FINE_CKPT = CKPT_DIR / "ConditionalFineFlowModel_e2e_flow_mlp_uQZ9lg_best.pt"
FINE_EMA_CKPT = CKPT_DIR / "EMAModel_e2e_flow_mlp_uQZ9lg_best.pt"

LEVEL_NAMES = ["L0 (global)", "L1", "L2"]


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_pca_dict(pca_dir=PCA_DIR, n_levels=6):
    """Load the 6 per-level sklearn PCA models as {level_idx: PCA}."""
    pca = {}
    for i in range(n_levels):
        matches = glob.glob(str(Path(pca_dir) / f"pca_L{i}_n*.pkl"))
        if not matches:
            raise FileNotFoundError(f"No pca_L{i}_n*.pkl in {pca_dir}")
        with open(max(matches, key=os.path.getmtime), "rb") as f:
            pca[i] = pickle.load(f)
    return pca


def _load_binary_array(img):
    """PIL image or path -> (128, W) binary float32 array (pitch axis padded to 128)."""
    if isinstance(img, (str, Path)):
        img = Image.open(img)
    arr = (np.array(img.convert("L"), dtype=np.uint8) > 0).astype(np.float32)  # (H, W)
    h, w = arr.shape
    if h != IMAGE_SIZE:
        out = np.zeros((IMAGE_SIZE, w), dtype=np.float32)
        out[: min(h, IMAGE_SIZE)] = arr[: min(h, IMAGE_SIZE)]
        arr = out
    if arr.shape[1] < IMAGE_SIZE:
        pad = np.zeros((IMAGE_SIZE, IMAGE_SIZE - arr.shape[1]), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=1)
    return arr


def best_crop_x(img):
    """Return the crop offset of the densest (most notes) 128-wide window.

    Full-song rolls often start with silence; this picks a musically populated
    window as a sensible default so the encoder sees actual content."""
    arr = _load_binary_array(img)
    w = arr.shape[1]
    if w <= IMAGE_SIZE:
        return 0
    col_density = arr.sum(axis=0)                      # notes per time column
    window = np.convolve(col_density, np.ones(IMAGE_SIZE), mode="valid")  # (w-127,)
    return int(np.argmax(window))


def image_to_binary_tensor(img, crop_x=0):
    """PIL image or path -> (1,1,128,128) binary float tensor.

    Matches the encoder's training preprocessing: grayscale -> (pixel>0) binary.
    A full-song roll (height 128, width W) is cropped to a 128-wide window
    starting at crop_x (clamped so the window stays in bounds)."""
    arr = _load_binary_array(img)
    w = arr.shape[1]
    crop_x = max(0, min(int(crop_x), w - IMAGE_SIZE))
    crop = arr[:, crop_x:crop_x + IMAGE_SIZE]
    return torch.from_numpy(crop).float().unsqueeze(0).unsqueeze(0)  # (1,1,128,128)


class FlowDemo:
    """Loads the e2e conditional-flow stack and generates piano rolls from an input crop."""

    def __init__(self, device=None):
        self.device = device or pick_device()
        self._loaded = False

    def load(self):
        dev = self.device
        print(f"[flow_infer] loading on device={dev}")

        self.pca = load_pca_dict()
        self.cond_n_comp = [self.pca[i].n_components_ for i in COARSE_LEVELS]   # e.g. [99,65,37]
        self.fine_n_comp = [self.pca[i].n_components_ for i in FINE_LEVELS]     # e.g. [21,8,4]
        self.cond_n_patches = [4 ** i for i in COARSE_LEVELS]                   # [1,4,16]
        self.fine_n_patches = [4 ** i for i in FINE_LEVELS]                     # [64,256,1024]
        self.cond_level_dims = [p * c for p, c in zip(self.cond_n_patches, self.cond_n_comp)]
        self.fine_level_dims = [p * c for p, c in zip(self.fine_n_patches, self.fine_n_comp)]
        print(f"[flow_infer] cond_level_dims={self.cond_level_dims} fine_level_dims={self.fine_level_dims}")

        self.encoder = SwinEncoder(
            img_height=IMAGE_SIZE, img_width=IMAGE_SIZE, patch_h=PATCH, patch_w=PATCH,
            embed_dim=EMBED_DIM, depths=DEPTHS, num_heads=NUM_HEADS, window_size=WINDOW_SIZE,
            mlp_ratio=MLP_RATIO, drop_path_rate=0.0,
        ).to(dev)
        load_checkpoint(self.encoder, str(ENCODER_CKPT))
        self.encoder.eval()

        self.decoder = SwinDecoder(
            img_height=IMAGE_SIZE, img_width=IMAGE_SIZE, patch_h=PATCH, patch_w=PATCH,
            out_channels=1, embed_dim=EMBED_DIM, depths=DEC_DEPTHS, num_heads=DEC_NUM_HEADS,
            window_size=WINDOW_SIZE, mlp_ratio=MLP_RATIO, drop_path_rate=0.0,
        ).to(dev)
        load_checkpoint(self.decoder, str(DECODER_CKPT))
        self.decoder.eval()

        self.fine = ConditionalFineFlowModel(
            cond_dims=self.cond_level_dims, target_dims=self.fine_level_dims,
            target_n_comp=self.fine_n_comp, cond_n_comp=self.cond_n_comp,
            h_dim=FLOW_H_DIM, n_layers=FLOW_N_LAYERS, t_dim=FLOW_T_DIM,
            n_fine_attn_layers=FINE_ATTN_LAYERS, n_lateral_attn_layers=FINE_LATERAL_ATTN_LAYERS,
            grad_checkpoint=False,
        ).to(dev)
        self._load_fine_weights(prefer_ema=True)
        self.fine.eval()

        for m in (self.encoder, self.decoder, self.fine):
            for p in m.parameters():
                p.requires_grad_(False)
        self._loaded = True
        return self

    def _load_fine_weights(self, prefer_ema=True):
        """Load the fine-flow weights. Prefer EMA (config eval'd on EMA); fall back to raw."""
        if prefer_ema and FINE_EMA_CKPT.exists():
            ckpt = torch.load(str(FINE_EMA_CKPT), map_location=self.device, weights_only=False)
            sd = ckpt["model_state_dict"]
            # EMAModel stores the averaged model under the `ema.` submodule (bf16); drop `_steps`.
            new = {}
            for k, v in sd.items():
                if k.startswith("ema."):
                    # fine model was trained with torch.compile -> keys are ema._orig_mod.*
                    nk = k[len("ema."):].replace("_orig_mod.", "")
                    new[nk] = v.float() if torch.is_floating_point(v) else v
            missing, unexpected = self.fine.load_state_dict(new, strict=False)
            print(f">>> Loaded EMA fine weights ({len(new)} tensors); "
                  f"missing={len(missing)} unexpected={len(unexpected)}")
            if len(new) == 0 or len(missing) > 0:
                print("    EMA load looked incomplete; falling back to raw fine checkpoint.")
                load_checkpoint(self.fine, str(FINE_CKPT))
        else:
            load_checkpoint(self.fine, str(FINE_CKPT))

    @torch.no_grad()
    def encode_to_coarse(self, img_tensor):
        """(1,1,128,128) binary image -> coarse conditioning vector (1, sum(cond_level_dims))."""
        enc = self.encoder(img_tensor.to(self.device))
        parts = []
        for li, i in enumerate(COARSE_LEVELS):
            emb = enc.patches.levels[i].emb           # (1, n_patches, D)
            B, npatch, D = emb.shape
            flat = emb.reshape(B * npatch, D).float().cpu().numpy()
            codes = self.pca[i].transform(flat)        # (B*n_patches, n_comp)
            parts.append(torch.from_numpy(codes.reshape(B, npatch * codes.shape[1])).float())
        return torch.cat(parts, dim=1).to(self.device)  # (1, 951)

    def _zero_dropped_levels(self, cond, drop_levels):
        """Zero (== PCA-whitened mean) the slices of chosen coarse levels in the conditioning vector."""
        if not drop_levels:
            return cond
        cond = cond.clone()
        offset = 0
        for i, d in enumerate(self.cond_level_dims):
            if i in drop_levels:
                cond[:, offset:offset + d] = 0.0
            offset += d
        return cond

    @torch.no_grad()
    def reconstruct(self, img_tensor):
        """Sanity check: encode -> REAL per-level PCA (all 6 levels) -> decode.

        No fine flow. If the e2e encoder/decoder + PCA wiring is correct this should
        closely reproduce the input crop, isolating decode correctness from the
        (separately trained) generative fine flow's quality."""
        assert self._loaded
        dev = self.device
        enc = self.encoder(img_tensor.to(dev))
        coarse_parts, fine_parts = [], []
        for i in range(6):
            emb = enc.patches.levels[i].emb
            B, npatch, D = emb.shape
            flat = emb.reshape(B * npatch, D).float().cpu().numpy()
            codes = self.pca[i].transform(flat)
            vec = torch.from_numpy(codes.reshape(B, npatch * codes.shape[1])).float().to(dev)
            (coarse_parts if i in COARSE_LEVELS else fine_parts).append(vec)
        coarse_emb = torch.cat(coarse_parts, dim=1)
        fine_emb = torch.cat(fine_parts, dim=1)
        rolls = decode_flow_to_piano_rolls(
            coarse_emb=coarse_emb, fine_emb=fine_emb, pca_models=self.pca,
            cond_level_dims=self.cond_level_dims, fine_level_dims=self.fine_level_dims,
            fine_levels_idx=FINE_LEVELS, cfg=None, decoder=self.decoder, device=dev, n_samples=1,
        )
        return rolls[0, 0].detach().cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def generate(self, img_tensor, drop_levels=(), n_steps=20, seed=0):
        """Conditional generation from one input crop.

        drop_levels: iterable of coarse level indices (0,1,2) to zero out of the conditioning.
        Returns a (128,128) binarized piano-roll numpy array (float {0,1}, image orientation).
        """
        assert self._loaded, "call .load() first"
        dev = self.device
        drop_levels = set(int(x) for x in drop_levels)

        cond = self.encode_to_coarse(img_tensor)           # (1, 951) real coarse
        cond = self._zero_dropped_levels(cond, drop_levels)

        # Plain-Gaussian fine source (matches the teacher-forcing eval path), reproducible.
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        fine_dim = sum(self.fine_level_dims)
        y_fine = torch.randn(1, fine_dim, generator=gen).to(dev)

        ts = warp_time(torch.linspace(0, 1, n_steps + 1), s=WARP_S).to(dev)
        for i in range(n_steps):
            dt = (ts[i + 1] - ts[i]).item()
            t = torch.full((1, 1), ts[i].item(), device=dev)
            y_fine = y_fine + self.fine(y_fine, t, cond) * dt

        rolls = decode_flow_to_piano_rolls(
            coarse_emb=cond, fine_emb=y_fine, pca_models=self.pca,
            cond_level_dims=self.cond_level_dims, fine_level_dims=self.fine_level_dims,
            fine_levels_idx=FINE_LEVELS, cfg=None, decoder=self.decoder, device=dev, n_samples=1,
        )
        return rolls[0, 0].detach().cpu().numpy().astype(np.float32)  # (128,128)


_DEMO = None


def get_demo(device=None):
    """Module-level singleton so the Gradio app loads models once."""
    global _DEMO
    if _DEMO is None:
        _DEMO = FlowDemo(device=device).load()
    return _DEMO

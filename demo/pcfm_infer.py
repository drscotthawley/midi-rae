"""Pixel-space conditional flow-matching (otcfm) inference backend for the demo.

This is the *working* generative model (run pixel_cfm_Tkn0KT). Unlike the e2e
fine-flow, it flows directly in pixel space: a UNet velocity field maps Gaussian
noise -> a 128x128 piano-roll image, conditioned on the exp26 encoder's per-level
PCA maps (`mlcond`), sampled with classifier-free guidance via a torchdyn NeuralODE.
No SwinDecoder, no inverse-PCA -- the UNet emits the image directly.

Conditioning pipeline (mirrors train_cfm_midi.py / probe_conditioning.py):
    input image (128x128 binary)
      -> exp26 SwinEncoder            -> 6 levels of patch embeddings
      -> per-level PCA.transform      -> (n_patches, n_comp) codes
      -> append normalized mean-pitch -> (n_patches, n_comp+1)
      -> reshape to (1, n_comp+1, sp, sp)  == mlcond[level]
    Level dropout = zero a chosen level's mlcond map.
"""
import os
import sys
import glob
import pickle
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from midi_rae.swin import SwinEncoder, SwinMaskedEmbeddingPredictor      # noqa: E402
from midi_rae.utils import load_checkpoint                              # noqa: E402
from torchdyn.core import NeuralODE                                     # noqa: E402
from torchcfm.models.unet.unet_mlc import UNetModelWrapperMLC           # noqa: E402

import flow_infer  # reuse image helpers (_load_binary_array, best_crop_x, image_to_binary_tensor)  # noqa: E402

CKPT_DIR = _HERE / "checkpoints"

# Which encoder+flow pair to serve. Override with MIDIRAE_MODELS=exp26 to get the old one.
#   c55   : wide3_xmep_C55cUL encoder (ep80, lambda_mep=1.0, cross_level_mep) + c55_cfm_MgB2XF
#           flow. Ships an XMEP checkpoint, so latent-space inpainting is available.
#   exp26 : the original pair. No XMEP on disk; predates unet_mlc's middle_film layers.
MODEL_SET = os.environ.get("MIDIRAE_MODELS", "c55").lower()

if MODEL_SET == "exp26":
    PCA_DIR = CKPT_DIR / "pca_exp26"
    CFM_CKPT = CKPT_DIR / "otcfm_mlcdrop_ckpt_step54000.pt"
    ENCODER_CKPT = CKPT_DIR / "SwinEncoder_exp26_z1olvN_best.pt"
    MEP_CKPT = None
    DEPTHS = [2, 2, 2, 6, 2, 1]
    HAS_MIDDLE_FILM = False
else:
    PCA_DIR = CKPT_DIR / "pca_c55"
    # c55_cfm_MgB2XF is still training; MIDIRAE_CFM_CKPT picks an earlier step.
    CFM_CKPT = CKPT_DIR / "c55" / os.environ.get("MIDIRAE_CFM_CKPT",
                                                 "otcfm_midi_weights_step_14000.pt")
    ENCODER_CKPT = CKPT_DIR / "c55" / "SwinEncoder_wide3_xmep_C55cUL_best.pt"
    MEP_CKPT = CKPT_DIR / "c55" / "SwinMaskedEmbeddingPredictor_wide3_xmep_C55cUL_best.pt"
    DEPTHS = [2, 2, 2, 6, 2, 2]
    HAS_MIDDLE_FILM = True


class _UNetNoMiddleFilm(UNetModelWrapperMLC):
    """exp26-era checkpoints predate the `middle_film` layers the current unet_mlc.py adds.
    Disable them so the architecture matches (forward already guards `is not None`)."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.middle_film_4 = None
        self.middle_film_2 = None

IMAGE_SIZE = 128
# Encoder architecture: shared across both sets except DEPTHS (set above).
EMBED_DIM = 8
NUM_HEADS = [2, 2, 2, 4, 8, 16]
WINDOW_SIZE = 4
MLP_RATIO = 4.0
PATCH = 4
# MEP geometry: n_summaries is hardcoded in train_enc.py, not stored in the config.
MEP_N_SUMMARIES = (1, 2, 4, 8, 32, 64)
MEP_CROSS_LEVEL = True

N_LEVELS = 6
LEVEL_NAMES = ["L0 (global)", "L1", "L2", "L3", "L4", "L5 (finest)"]


def available_devices():
    """CPU always; add CUDA / MPS when present. Ordered fastest-first."""
    devs = []
    if torch.cuda.is_available():
        devs.append("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        devs.append("mps")
    devs.append("cpu")
    return devs


def default_device():
    return available_devices()[0]


def mean_pitch_normalized(binary_2d):
    """Normalized mean pitch (0-1) from a (H, W) binary roll (matches train_cfm_midi)."""
    t = torch.as_tensor(binary_2d, dtype=torch.float32)
    H = t.shape[-2]
    density = t.sum(dim=-1)                       # (H,)
    idx = torch.arange(H, dtype=torch.float32)
    mp = (idx * density).sum() / density.sum().clamp(min=1e-6)
    return (mp / H).item()


class CFGWrapper(torch.nn.Module):
    """Classifier-free guidance: v = v_uncond + strength*(v_cond - v_uncond)."""
    def __init__(self, model, cond, strength):
        super().__init__()
        self.model, self.cond, self.strength = model, cond, strength

    def forward(self, t, x, **kwargs):
        v_uncond = self.model(t, x, mlcond=None)
        v_cond = self.model(t, x, mlcond=self.cond)
        return v_uncond + self.strength * (v_cond - v_uncond)


class PixelCFMDemo:
    def __init__(self, device=None):
        self.device = device or default_device()
        self._loaded = False

    def load(self):
        # PCA (6 levels), tiny
        self.pca = {}
        for i in range(N_LEVELS):
            m = glob.glob(str(PCA_DIR / f"pca_L{i}_n*.pkl"))
            if not m:
                raise FileNotFoundError(f"missing pca_L{i}_n*.pkl in {PCA_DIR}")
            with open(max(m, key=os.path.getmtime), "rb") as f:
                self.pca[i] = pickle.load(f)
        self.n_comp = [self.pca[i].n_components_ for i in range(N_LEVELS)]       # c55: [3,4,5,6,4,3]
        self.sp = [2 ** i for i in range(N_LEVELS)]                              # [1,2,4,8,16,32]
        # mlcond_shapes: {spatial_size: (level_idx, n_comp+1)}  (real mlcdrop = PCA + mean-pitch)
        self.mlcond_shapes = {self.sp[i]: (i, self.n_comp[i] + 1) for i in range(N_LEVELS)}

        # encoder (architecture per MODEL_SET)
        self.encoder = SwinEncoder(
            img_height=IMAGE_SIZE, img_width=IMAGE_SIZE, patch_h=PATCH, patch_w=PATCH,
            embed_dim=EMBED_DIM, depths=DEPTHS, num_heads=NUM_HEADS, window_size=WINDOW_SIZE,
            mlp_ratio=MLP_RATIO, drop_path_rate=0.0,
        )
        load_checkpoint(self.encoder, str(ENCODER_CKPT))
        self.encoder.eval()

        # pixel-CFM UNet (num_channels inferred from checkpoint)
        ckpt = torch.load(str(CFM_CKPT), map_location="cpu", weights_only=False)
        state = ckpt["ema_model"] if "ema_model" in ckpt else ckpt["net_model"]
        num_channels = state["input_blocks.0.0.weight"].shape[0]
        print(f"[pcfm] {MODEL_SET}: UNet num_channels={num_channels}  step={ckpt.get('step')}  "
              f"mlcond_shapes={self.mlcond_shapes}")
        unet_cls = UNetModelWrapperMLC if HAS_MIDDLE_FILM else _UNetNoMiddleFilm
        self.model = unet_cls(
            dim=(1, IMAGE_SIZE, IMAGE_SIZE), num_res_blocks=2, num_channels=num_channels,
            channel_mult=[1, 2, 2, 2, 2], num_heads=4, num_head_channels=64,
            attention_resolutions="16", dropout=0.1, mlcond_shapes=self.mlcond_shapes,
        )
        self.model.load_state_dict(state)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        for p in self.encoder.parameters():
            p.requires_grad_(False)

        self._loaded = True
        self._move_to(self.device)
        return self

    def _move_to(self, device):
        self.encoder.to(device)
        self.model.to(device)
        if getattr(self, "mep", None) is not None:
            self.mep.to(device)
        self.device = device

    def set_device(self, device):
        if self._loaded and device != self.device:
            print(f"[pcfm] moving models to {device}")
            self._move_to(device)

    @torch.no_grad()
    def encode_to_mlcond(self, img_tensor):
        """(1,1,128,128) binary -> list of 6 mlcond maps (1, n_comp+1, sp, sp) [PCA + mean-pitch]."""
        dev = self.device
        enc = self.encoder(img_tensor.to(dev))
        mp = mean_pitch_normalized(img_tensor[0, 0])
        mlcond = []
        for i in range(N_LEVELS):
            emb = enc.patches.levels[i].emb            # (1, n_patches, D)
            npatch, D = emb.shape[1], emb.shape[2]
            codes = self.pca[i].transform(emb.reshape(npatch, D).float().cpu().numpy())  # (npatch, n_comp)
            codes = torch.from_numpy(codes).float()
            mpcol = torch.full((npatch, 1), mp, dtype=torch.float32)
            cond_i = torch.cat([codes, mpcol], dim=1)  # (npatch, n_comp+1)
            sp = self.sp[i]
            cond_i = cond_i.view(sp, sp, -1).permute(2, 0, 1).unsqueeze(0)  # (1, n_comp+1, sp, sp)
            mlcond.append(cond_i.to(dev))
        return mlcond

    # ---------------------------------------------------------- XMEP latent inpainting
    def load_mep(self):
        """Lazily build+load the masked-embedding predictor (the latent inpainter).

        Trained jointly with this encoder to reconstruct embeddings at masked token
        positions from surrounding context -- exactly the operation we need for a
        painted hole. n_summaries is hardcoded in train_enc.py, not in the config."""
        if getattr(self, "mep", None) is not None:
            return self.mep
        if MEP_CKPT is None:
            raise RuntimeError(f"no XMEP checkpoint for MODEL_SET={MODEL_SET}")
        dims = tuple(lv for lv in (256, 128, 64, 32, 16, 8))
        self.mep = SwinMaskedEmbeddingPredictor(dims=dims, n_summaries=MEP_N_SUMMARIES,
                                                cross_level_mep=MEP_CROSS_LEVEL)
        load_checkpoint(self.mep, str(MEP_CKPT))
        self.mep.eval()
        for p in self.mep.parameters():
            p.requires_grad_(False)
        self.mep.to(self.device)
        return self.mep

    def hole_to_token_masks(self, hole, dilate=0, thresh=0.0):
        """(128,128) bool pixel hole -> per-level (1,N) bool masks, True=VISIBLE.

        Level i has sp=2**i tokens per side, and 128 % sp == 0 for every level.
        `thresh` is the fraction of a token's pixels that must be erased before the
        token counts as a hole: 0.0 = any pixel (maximally cautious, but for masks
        not aligned to the token grid it discards tokens that are mostly known
        context). `dilate` additionally grows the hole on the token grid -- measured
        to HURT (context matters more than the contamination it removes)."""
        masks = []
        for i in range(N_LEVELS):
            sp = self.sp[i]
            f = IMAGE_SIZE // sp
            frac = hole.reshape(sp, f, sp, f).mean(axis=(1, 3))        # (sp,sp) erased fraction
            blk = frac > thresh if thresh > 0 else frac > 0.0          # True=hole
            for _ in range(int(dilate)):
                g = blk.copy()
                g[1:, :] |= blk[:-1, :]; g[:-1, :] |= blk[1:, :]
                g[:, 1:] |= blk[:, :-1]; g[:, :-1] |= blk[:, 1:]
                blk = g
            vis = torch.from_numpy(~blk.reshape(-1)).unsqueeze(0)      # (1,N) True=visible
            masks.append(vis)
        return masks

    @torch.no_grad()
    def encode_to_mlcond_filled(self, img_tensor, hole, dilate=0, thresh=0.0,
                                return_stats=False):
        """Like encode_to_mlcond, but the hole's embeddings are PREDICTED by the XMEP.

        img_tensor must already be hole-zeroed, so the encoder never sees the erased
        notes. Zeroed pixels would otherwise make the conditioning say 'silence here',
        which the flow faithfully renders; the XMEP instead says 'plausible music here,
        given the surroundings'."""
        dev = self.device
        mep = self.load_mep()
        enc = self.encoder(img_tensor.to(dev))
        masks = [m.to(dev) for m in self.hole_to_token_masks(hole, dilate=dilate, thresh=thresh)]

        orig_fn = mep._make_level_masks                                 # inject our mask
        mep._make_level_masks = lambda levels, device: masks
        try:
            preds, _ = mep(enc)
        finally:
            mep._make_level_masks = orig_fn

        mp = mean_pitch_normalized(img_tensor[0, 0])
        mlcond, stats = [], []
        for i in range(N_LEVELS):
            emb = enc.patches.levels[i].emb                             # (1,N,D)
            vis = masks[i].unsqueeze(-1)                                # (1,N,1)
            filled = torch.where(vis, emb, preds[i].to(emb.dtype))
            if return_stats:
                n_hole = int((~masks[i]).sum())
                d = (preds[i] - emb).norm(dim=-1)[0]                    # per-token, no pooling
                stats.append((i, n_hole, int(masks[i].numel()),
                              float(d[~masks[i][0]].mean()) if n_hole else 0.0,
                              float(d[masks[i][0]].mean()) if n_hole < d.numel() else 0.0))
            npatch, D = filled.shape[1], filled.shape[2]
            codes = self.pca[i].transform(filled.reshape(npatch, D).float().cpu().numpy())
            codes = torch.from_numpy(codes).float()
            mpcol = torch.full((npatch, 1), mp, dtype=torch.float32)
            sp = self.sp[i]
            cond_i = torch.cat([codes, mpcol], dim=1).view(sp, sp, -1).permute(2, 0, 1).unsqueeze(0)
            mlcond.append(cond_i.to(dev))
        return (mlcond, stats) if return_stats else mlcond

    @torch.no_grad()
    def generate(self, img_tensor, drop_levels=(), n_steps=20, seed=0, cfg_strength=4.0,
                 device=None, solver="euler"):
        """Conditional pixel-CFM generation. Returns (128,128) float roll in [0,1].

        solver: torchdyn ODE solver — 'euler' (1 eval/step) or 'rk4' (4 evals/step)."""
        assert self._loaded, "call .load() first"
        if device is not None:
            self.set_device(device)
        dev = self.device
        drop = set(int(x) for x in drop_levels)

        mlcond = self.encode_to_mlcond(img_tensor)
        for i in drop:
            mlcond[i] = torch.zeros_like(mlcond[i])      # zero = drop this conditioning level

        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        x0 = torch.randn(1, 1, IMAGE_SIZE, IMAGE_SIZE, generator=gen).to(dev)

        wrapped = CFGWrapper(self.model, mlcond, float(cfg_strength))
        node = NeuralODE(wrapped, solver=solver, sensitivity="adjoint")
        # n_steps = number of integration *steps* -> n_steps+1 time points (>=2, so n_steps=1 works)
        t_span = torch.linspace(0, 1, int(n_steps) + 1, device=dev)
        traj = node.trajectory(x0, t_span=t_span)
        roll = (traj[-1].clip(-1, 1) / 2 + 0.5)[0, 0]     # (128,128) in [0,1]
        return roll.detach().cpu().numpy().astype(np.float32)


_DEMO = None


def get_demo(device=None):
    global _DEMO
    if _DEMO is None:
        _DEMO = PixelCFMDemo(device=device).load()
    return _DEMO

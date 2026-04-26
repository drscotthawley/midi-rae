"""
Musicality probes for midi-rae Swin encoder.

Probes:
  1. Chord quality linear probe  (maj / min / dom7 / dim / other)
  2. Root note linear probe      (C, C#, ..., B — 12 classes)
  3. Pitch transposition equivariance curve
  4. Cross-song similarity       (same-song crops closer than cross-song?)
  5. Note density regression     (R² per level)

All probes run at every encoder level (0=coarsest/256-dim … 5=finest/8-dim).
Coarsest level has 1 patch → its embedding IS the global representation.
Finer levels are mean-pooled across patches before probing.

Usage (from midi-rae repo root):
    python probe_musicality.py \
        --ckpt  ~/runs/midi-rae/exp26-wide1/checkpoints/SwinEncoder_exp26-wide1_best.pt \
        --config ~/runs/midi-rae/exp26-wide1/configs/config_swin.yaml \
        --data  ~/datasets/POP909_images_basic \
        --n_crops 2000 \
        --device cpu
"""
import argparse, sys, os, glob, urllib.request, zipfile, multiprocessing as mp, io
import wandb
from contextlib import redirect_stdout
import numpy as np
import torch
import pretty_midi
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')  # suppress sklearn CV/FutureWarnings
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from midi_rae.data import AnchorDataset, PRPairDataset
from midi_rae.swin import SwinEncoder
from midi_rae.utils import load_checkpoint

# ── Chord templates (from polyffusion chord_class.py) ───────────────────────

QUALITIES = {
    'maj':  [1,0,0,0,1,0,0,1,0,0,0,0],
    'min':  [1,0,0,1,0,0,0,1,0,0,0,0],
    '7':    [1,0,0,0,1,0,0,1,0,0,1,0],
    'maj7': [1,0,0,0,1,0,0,1,0,0,0,1],
    'min7': [1,0,0,1,0,0,0,1,0,0,1,0],
    'dim':  [1,0,0,1,0,0,1,0,0,0,0,0],
    'dim7': [1,0,0,1,0,0,1,0,0,1,0,0],
    'aug':  [1,0,0,0,1,0,0,0,1,0,0,0],
    'sus4': [1,0,0,0,0,1,0,1,0,0,0,0],
}
NOTE_NAMES = ['C','C#','D','Eb','E','F','F#','G','Ab','A','Bb','B']

# Build (n_templates, 12) matrix and corresponding label lists
_templates, _root_labels, _quality_labels, _chord_labels = [], [], [], []
for root in range(12):
    for q, tmpl in QUALITIES.items():
        _templates.append(np.roll(tmpl, root))
        _root_labels.append(root)
        _quality_labels.append(q)
        _chord_labels.append(f"{NOTE_NAMES[root]}:{q}")

TEMPLATES = np.array(_templates, dtype=np.float32)       # (108, 12)
ROOT_LABELS    = np.array(_root_labels)
QUALITY_LABELS = np.array(_quality_labels)

# Simplified quality groups for easier probing
QUALITY_GROUPS = {
    'maj': 'maj', 'maj7': 'maj', 'sus4': 'maj',
    'min': 'min', 'min7': 'min',
    '7':   'dom7',
    'dim': 'dim', 'dim7': 'dim',
    'aug': 'aug',
}


# ── Chroma extraction from piano roll image ──────────────────────────────────

def image_to_chroma(img: np.ndarray) -> np.ndarray:
    """
    img: (H, W) binary uint8, H=128 pitch rows (MIDI note 0..127), W=time steps.
    Returns (12,) chroma vector (summed across time and octaves).
    """
    chroma = np.zeros(12, dtype=np.float32)
    for row in range(img.shape[0]):
        chroma[row % 12] += img[row].sum()
    norm = chroma.sum()
    return chroma / norm if norm > 0 else chroma


def chroma_to_chord(chroma: np.ndarray):
    """
    Nearest-template chord recognition via cosine similarity.
    Returns (root_idx, quality_str, chord_str, quality_group_str).
    """
    if chroma.sum() == 0:
        return None, 'N', 'N', 'other'
    # cosine similarity against all templates
    scores = TEMPLATES @ chroma / (np.linalg.norm(TEMPLATES, axis=1) * np.linalg.norm(chroma) + 1e-8)
    best = int(scores.argmax())
    root = ROOT_LABELS[best]
    qual = QUALITY_LABELS[best]
    chord = _chord_labels[best]
    group = QUALITY_GROUPS.get(qual, 'other')
    return root, qual, chord, group


# ── Encoder helpers ──────────────────────────────────────────────────────────

def build_encoder(cfg, device):
    m = cfg.model
    enc = SwinEncoder(
        img_height=cfg.data.image_size, img_width=cfg.data.image_size,
        patch_h=m.patch_h, patch_w=m.patch_w,
        embed_dim=m.embed_dim, depths=tuple(m.depths),
        num_heads=tuple(m.num_heads), window_size=m.window_size,
        mlp_ratio=m.mlp_ratio, drop_path_rate=m.drop_path_rate,
    ).to(device)
    return enc


@torch.no_grad()
def get_level_embeddings(enc_out) -> list[np.ndarray]:
    """Return per-level mean-pooled embeddings. Level 0 = coarsest (1 patch)."""
    embs = []
    for lvl in enc_out.patches.levels:
        e = lvl.emb.cpu().float()           # (B, n_patches, dim)
        embs.append(e.mean(dim=1).numpy())  # (B, dim) — mean pool over patches
    return embs


# ── Probe 1 & 2: chord quality + root linear probes ─────────────────────────

def run_chord_probes(embeddings_per_level, root_labels, group_labels, n_levels):
    # Drop rare quality classes (need at least n_splits=5 samples per class)
    min_count = 5
    counts = {q: (group_labels == q).sum() for q in np.unique(group_labels)}
    keep_mask = np.array([counts[g] >= min_count for g in group_labels])
    kept_groups = group_labels[keep_mask]
    kept_roots  = root_labels[keep_mask]
    kept_embs   = [e[keep_mask] for e in embeddings_per_level]
    dropped = [q for q, c in counts.items() if c < min_count]
    if dropped:
        print(f"  (dropping rare quality classes with <{min_count} samples: {dropped})")

    quality_le = LabelEncoder().fit(kept_groups)
    y_qual = quality_le.transform(kept_groups)
    chance_qual = np.bincount(y_qual).max() / len(y_qual)
    n_classes = len(quality_le.classes_)

    print(f"\n=== Probe 1: Chord Quality ({n_classes}-class: {list(quality_le.classes_)}) ===")
    print(f"{'Level':>6}  {'Acc':>8}  {'Chance':>8}")
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))

    qual_accs, root_accs = [], []
    for lev in range(n_levels):
        acc = cross_val_score(clf,kept_embs[lev], y_qual, cv=5, scoring='accuracy', n_jobs=1).mean()
        qual_accs.append(acc)
        print(f"  L{lev}    {acc:.3f}    {chance_qual:.3f}")

    print(f"\n=== Probe 2: Root Note (12-class) ===")
    print(f"{'Level':>6}  {'Acc':>8}  {'Chance':>8}")
    root_le = LabelEncoder().fit(kept_roots)
    y_root = root_le.transform(kept_roots)
    chance_root = 1/12
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))

    for lev in range(n_levels):
        acc = cross_val_score(clf,kept_embs[lev], y_root, cv=5, scoring='accuracy', n_jobs=1).mean()
        root_accs.append(acc)
        print(f"  L{lev}    {acc:.3f}    {chance_root:.3f}")

    return qual_accs, root_accs, chance_qual, chance_root


# ── Probe 3: pitch transposition equivariance ────────────────────────────────

def run_transposition_probe(encoder, dataset, device, n_samples=200, max_shift=24, batch_size=32):
    print(f"\n=== Probe 3: Pitch Transposition Equivariance ===")
    shifts = list(range(0, max_shift + 1, 2))
    n_levels = None
    dist_by_shift = None

    for batch_start in tqdm(range(0, n_samples, batch_size), desc='Transposition', leave=False):
        B_actual = min(batch_size, n_samples - batch_start)
        for shift in shifts:
            items = [dataset.__getitem__(0, shift_x=0, shift_y=shift) for _ in range(B_actual)]
            originals = torch.stack([it['img1'] for it in items]).to(device)  # (B, 1, H, W)
            shifted   = torch.stack([it['img2'] for it in items]).to(device)
            combined  = torch.cat([originals, shifted], dim=0)  # (2B, 1, H, W)
            with torch.no_grad():
                enc = encoder(combined)
            if n_levels is None:
                n_levels = enc.patches.num_levels
                dist_by_shift = {lev: {s: [] for s in shifts} for lev in range(n_levels)}
            B = originals.shape[0]
            for lev in range(n_levels):
                emb = enc.patches.levels[lev].emb  # (2B, N, D)
                z1 = emb[:B].cpu().float()           # (B, N, D)
                z2 = emb[B:].cpu().float()
                patch_dists = (z1 - z2).norm(dim=-1).mean(dim=1)  # (B,)
                dist_by_shift[lev][shift].extend(patch_dists.tolist())

    colors = plt.cm.viridis(np.linspace(0, 1, n_levels))
    fig, ax = plt.subplots(figsize=(8, 5))
    all_means = {}
    for lev in range(n_levels):
        mean_dists = [np.mean(dist_by_shift[lev][s]) for s in shifts]
        std_dists  = [np.std(dist_by_shift[lev][s])  for s in shifts]
        all_means[lev] = mean_dists
        patch_px = 128 // (2 ** lev)
        ax.errorbar(shifts, mean_dists, yerr=std_dists, marker='o', capsize=3,
                    color=colors[lev], label=f'L{lev} ({patch_px}×{patch_px} patch)')
    ax.set_xlabel('Pitch shift (semitones)')
    ax.set_ylabel('Euclidean distance in embedding space (symlog)')
    ax.set_title('Pitch Transposition Equivariance — all levels')
    ax.set_yscale('symlog', linthresh=0.01)
    ax.axhline(0, color='gray', ls='--', alpha=0.4)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig('probe_transposition.png', dpi=120)
    mean_dists_l0 = all_means[0]
    print(f"  shift=0  dist(L0)={mean_dists_l0[0]:.3f}  (should be ~0)")
    print(f"  shift=12 dist(L0)={mean_dists_l0[shifts.index(12)]:.3f}  (octave)")
    print(f"  Saved → probe_transposition.png")
    return shifts, all_means


# ── Probe 6: time translation equivariance ───────────────────────────────────

def run_time_translation_probe(encoder, dataset, device, n_samples=200, max_shift=120, batch_size=32):
    """Probe time translation equivariance using PRPairDataset with explicit shift_x.

    Calls dataset.__getitem__(shift_x=shift, shift_y=0) so both crops come from
    the same real song with no zero-padding and no pitch offset between them.
    Short songs are handled by clamping in PRPairDataset.__getitem__.
    """
    print(f"\n=== Probe 6: Time Translation Equivariance ===")
    shifts = list(range(0, max_shift + 1, 4))
    n_levels = None
    dist_by_shift = None

    for batch_start in tqdm(range(0, n_samples, batch_size), desc='Time translation', leave=False):
        B_actual = min(batch_size, n_samples - batch_start)
        for shift in shifts:
            items = [dataset.__getitem__(0, shift_x=shift, shift_y=0) for _ in range(B_actual)]
            originals = torch.stack([it['img1'] for it in items]).to(device)  # (B, 1, H, W)
            shifted   = torch.stack([it['img2'] for it in items]).to(device)
            combined  = torch.cat([originals, shifted], dim=0)  # (2B, 1, H, W)
            with torch.no_grad():
                enc = encoder(combined)
            if n_levels is None:
                n_levels = enc.patches.num_levels
                dist_by_shift = {lev: {sh: [] for sh in shifts} for lev in range(n_levels)}
            for lev in range(n_levels):
                emb = enc.patches.levels[lev].emb  # (2B, N, D)
                z1 = emb[:B_actual].cpu().float()
                z2 = emb[B_actual:].cpu().float()
                patch_dists = (z1 - z2).norm(dim=-1).mean(dim=1)
                dist_by_shift[lev][shift].extend(patch_dists.tolist())

    colors = plt.cm.viridis(np.linspace(0, 1, n_levels))
    fig, ax = plt.subplots(figsize=(8, 5))
    all_means = {}
    for lev in range(n_levels):
        mean_dists = [np.mean(dist_by_shift[lev][s]) for s in shifts]
        std_dists  = [np.std(dist_by_shift[lev][s])  for s in shifts]
        all_means[lev] = mean_dists
        patch_px = 128 // (2 ** lev)
        ax.errorbar(shifts, mean_dists, yerr=std_dists, marker='o', capsize=3,
                    color=colors[lev], label=f'L{lev} ({patch_px}×{patch_px} patch)')
    ax.set_xlabel('Time shift (pixels)')
    ax.set_ylabel('Euclidean distance in embedding space (symlog)')
    ax.set_title('Time Translation Equivariance — all levels')
    ax.set_yscale('symlog', linthresh=0.01)
    ax.axhline(0, color='gray', ls='--', alpha=0.4)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig('probe_time_translation.png', dpi=120)
    mean_dists_l0 = all_means[0]
    print(f"  shift=0   dist(L0)={mean_dists_l0[0]:.3f}  (should be ~0)")
    print(f"  shift=64  dist(L0)={mean_dists_l0[-1]:.3f}")
    print(f"  Saved → probe_time_translation.png")
    return shifts, all_means


# ── Probe 7: temporal distance regression ─────────────────────────────────────

def run_temporal_distance_probe(encoder, dataset, device, n_pairs=1000):
    """
    Sample pairs of crops from the same file at known temporal offsets.
    Regress embedding distance → temporal distance. R² measures how well
    the encoder encodes temporal position (want high for equivariance).
    Uses PRPairDataset.__getitem__(shift_x=shift, shift_y=0) for real crops.
    """
    print(f"\n=== Probe 7: Temporal Distance Regression ===")
    max_shift = dataset.crop_size if isinstance(dataset.crop_size, int) else dataset.crop_size[1]
    n_levels = None
    temporal_dists, emb_dists_per_level = [], []

    for _ in tqdm(range(n_pairs), desc='Temporal distance pairs', leave=False):
        shift = int(np.random.randint(0, max_shift))
        item = dataset.__getitem__(0, shift_x=shift, shift_y=0)
        x1 = item['img1'].unsqueeze(0).to(device)
        x2 = item['img2'].unsqueeze(0).to(device)
        with torch.no_grad():
            e1 = encoder(x1)
            e2 = encoder(x2)
        if n_levels is None:
            n_levels = e1.patches.num_levels
            emb_dists_per_level = [[] for _ in range(n_levels)]
        for lev in range(n_levels):
            z1 = e1.patches.levels[lev].emb.cpu().float()  # (1, N, D)
            z2 = e2.patches.levels[lev].emb.cpu().float()
            emb_dists_per_level[lev].append((z1 - z2).norm(dim=-1).mean().item())
        temporal_dists.append(item['deltas'][0].item())

    temporal_dists = np.array(temporal_dists).reshape(-1, 1)
    print(f"{'Level':>6}  {'R²':>8}  {'corr':>8}")
    r2s = []
    for lev in range(n_levels):
        d = np.array(emb_dists_per_level[lev]).reshape(-1, 1)
        r2 = cross_val_score(Ridge(), d, temporal_dists.ravel(), cv=5, scoring='r2', n_jobs=1).mean()
        corr = float(np.corrcoef(d.ravel(), temporal_dists.ravel())[0, 1])
        r2s.append(r2)
        print(f"  L{lev}    {r2:.3f}    {corr:.3f}")
    return r2s


# ── Probe 4: cross-song similarity ───────────────────────────────────────────

def run_cross_song_probe(embeddings_per_level, file_idxs, n_levels):
    print(f"\n=== Probe 4: Cross-Song Similarity ===")
    print(f"{'Level':>6}  {'Same-song dist':>16}  {'Diff-song dist':>16}  {'Ratio':>8}")
    file_idxs = np.array(file_idxs)
    ratios = []
    for lev in range(n_levels):
        X = torch.tensor(embeddings_per_level[lev])
        X = F.normalize(X, dim=-1)
        n = len(X)
        rng = np.random.default_rng(42)
        idx_a = rng.integers(0, n, size=2000)
        idx_b = rng.integers(0, n, size=2000)
        same = file_idxs[idx_a] == file_idxs[idx_b]
        dists = (X[idx_a] - X[idx_b]).norm(dim=-1).numpy()
        same_dist = dists[same].mean()
        diff_dist = dists[~same].mean()
        ratio = same_dist / diff_dist if diff_dist > 0 else float('nan')
        print(f"  L{lev}    {same_dist:.4f}           {diff_dist:.4f}           {ratio:.3f}  (want < 1)")
        ratios.append(ratio)
    return ratios


# ── Probe 5: note density regression ─────────────────────────────────────────

def run_density_probe(embeddings_per_level, densities, n_levels):
    print(f"\n=== Probe 5: Note Density Regression (R²) ===")
    print(f"{'Level':>6}  {'R²':>8}")
    y = np.array(densities)
    reg = make_pipeline(StandardScaler(), Ridge())
    r2s = []
    for lev in range(n_levels):
        r2 = cross_val_score(reg, embeddings_per_level[lev], y, cv=5, scoring='r2', n_jobs=1).mean()
        print(f"  L{lev}    {r2:.3f}")
        r2s.append(r2)
    return r2s


# ── Probe 9: EMOPIA emotion ───────────────────────────────────────────────────

def _download_emopia(emopia_dir):
    url = 'https://zenodo.org/record/5257995/files/EMOPIA_2.2.zip'
    zip_path = os.path.join(emopia_dir, 'EMOPIA_2.2.zip')
    print(f"  Downloading EMOPIA 2.2 from Zenodo...")
    urllib.request.urlretrieve(url, zip_path)
    print(f"  Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(emopia_dir)
    os.remove(zip_path)


def run_emopia_emotion_probe(encoder, cfg, emopia_dir, device):
    """
    Render EMOPIA MIDI clips → piano roll → encode → 4-class emotion linear probe.
    Labels Q1-Q4 are parsed from the filename or parent directory name.
    Also reports binary arousal (Q1+Q2 vs Q3+Q4) and valence (Q1+Q4 vs Q2+Q3).
    """
    emopia_dir = os.path.expandvars(os.path.expanduser(emopia_dir))
    print(f"\n=== Probe 9: EMOPIA Emotion (4-class quadrant) ===")

    if not os.path.exists(emopia_dir):
        os.makedirs(emopia_dir, exist_ok=True)
        _download_emopia(emopia_dir)

    midi_files = sorted(glob.glob(os.path.join(emopia_dir, '**', '*.mid'), recursive=True))
    if not midi_files:
        print(f"  No .mid files found in {emopia_dir} — skipping.")
        return
    print(f"  Found {len(midi_files)} MIDI clips")

    crop_size = cfg.data.image_size
    rng = np.random.default_rng(42)
    n_levels = None
    embeddings_per_level = None
    emotion_labels = []
    skipped = 0

    for midi_path in tqdm(midi_files, desc='EMOPIA emotion', leave=False):
        # Parse Q label from filename (Q1_xxxx.mid) or parent dir (Q1/xxxx.mid)
        fname  = os.path.basename(midi_path)
        parent = os.path.basename(os.path.dirname(midi_path))
        if   fname[0] == 'Q' and fname[1].isdigit():   q = int(fname[1])
        elif parent[0] == 'Q' and parent[1].isdigit(): q = int(parent[1])
        else: skipped += 1; continue

        try:
            mid = pretty_midi.PrettyMIDI(midi_path)
            tc  = mid.get_tempo_changes()
            bps = (tc[1][0] if len(tc[1]) > 0 else 120.0) / 60.0
            fs  = bps * 4.0 * 2                  # same formula as midi_to_pr_img
            pr  = mid.get_piano_roll(fs=fs)       # (128, N)
        except Exception:
            skipped += 1; continue

        if pr.shape[1] < crop_size:
            skipped += 1; continue

        pr_bin = (pr > 0).astype(np.float32)
        x = int(rng.integers(0, pr_bin.shape[1] - crop_size))
        crop = pr_bin[:, x:x + crop_size]         # (128, crop_size)

        x_t = torch.from_numpy(crop).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            enc_out = encoder(x_t)

        level_embs = get_level_embeddings(enc_out)
        if n_levels is None:
            n_levels = len(level_embs)
            embeddings_per_level = [[] for _ in range(n_levels)]
        for lev, e in enumerate(level_embs):
            embeddings_per_level[lev].append(e[0])
        emotion_labels.append(q - 1)   # 0-indexed

    if n_levels is None or len(emotion_labels) < 20:
        print(f"  Too few valid clips ({len(emotion_labels)}) — skipping.")
        return
    if skipped:
        print(f"  (skipped {skipped} clips)")

    embeddings_per_level = [np.stack(embeddings_per_level[lev]) for lev in range(n_levels)]
    y = np.array(emotion_labels)
    counts = np.bincount(y, minlength=4)
    qnames = ['Q1 hi-A/pos-V', 'Q2 hi-A/neg-V', 'Q3 lo-A/neg-V', 'Q4 lo-A/pos-V']
    for i, name in enumerate(qnames): print(f"  {name}: {counts[i]}")
    chance4 = counts.max() / len(y)

    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))
    print(f"\n{'Level':>6}  {'4-class':>9}  {'Arousal':>9}  {'Valence':>9}  (chance: {chance4:.2f} / 0.50 / 0.50)")
    y_arousal = (y < 2).astype(int)                        # Q1+Q2=high, Q3+Q4=low
    y_valence = ((y == 0) | (y == 3)).astype(int)          # Q1+Q4=pos,  Q2+Q3=neg
    a4s, aas, avs = [], [], []
    for lev in range(n_levels):
        X = embeddings_per_level[lev]
        a4 = cross_val_score(clf,X, y,         cv=5, scoring='accuracy', n_jobs=1).mean()
        aa = cross_val_score(clf,X, y_arousal, cv=5, scoring='accuracy', n_jobs=1).mean()
        av = cross_val_score(clf,X, y_valence, cv=5, scoring='accuracy', n_jobs=1).mean()
        print(f"  L{lev}    {a4:.3f}      {aa:.3f}      {av:.3f}")
        a4s.append(float(a4)); aas.append(float(aa)); avs.append(float(av))
    return a4s, aas, avs


# ── Haar baseline for EMOPIA (compare to Probe 9) ────────────────────────────

def run_haar_emopia_baseline(emopia_dir, cfg, n_depths=5):
    """
    Haar 1D-time baseline probe on EMOPIA — same crops as run_emopia_emotion_probe
    (same RNG seed + sorted file list) so the comparison is fair.
    Feature at depth d: [mean, std] of LL and XOR subbands across time → (512,) max.
    """
    emopia_dir = os.path.expandvars(os.path.expanduser(emopia_dir))
    print(f"\n=== Haar Baseline: EMOPIA Emotion ===")
    midi_files = sorted(glob.glob(os.path.join(emopia_dir, '**', '*.mid'), recursive=True))
    if not midi_files:
        print(f"  No .mid files found — skipping.")
        return

    crop_size = cfg.data.image_size
    rng = np.random.default_rng(42)   # same seed as encoder probe
    feats_per_depth = [[] for _ in range(n_depths + 1)]
    labels = []
    skipped = 0

    for midi_path in tqdm(midi_files, desc='Haar baseline', leave=False):
        fname  = os.path.basename(midi_path)
        parent = os.path.basename(os.path.dirname(midi_path))
        if   fname[0]   == 'Q' and fname[1].isdigit():   q = int(fname[1])
        elif parent[0]  == 'Q' and parent[1].isdigit():  q = int(parent[1])
        else: skipped += 1; continue

        try:
            mid = pretty_midi.PrettyMIDI(midi_path)
            tc  = mid.get_tempo_changes()
            bps = (tc[1][0] if len(tc[1]) > 0 else 120.0) / 60.0
            pr  = mid.get_piano_roll(fs=bps * 4.0 * 2)
        except Exception:
            skipped += 1; continue

        if pr.shape[1] < crop_size:
            skipped += 1; continue

        pr_bin = (pr > 0).astype(np.uint8)
        x = int(rng.integers(0, pr_bin.shape[1] - crop_size))
        crop = pr_bin[:, x:x + crop_size].astype(np.float32)   # (128, T)

        cur = crop
        feats_per_depth[0].append(np.concatenate([cur.mean(axis=1), cur.std(axis=1)]))
        for d in range(1, n_depths + 1):
            n = (cur.shape[1] // 2) * 2
            if n < 2: break
            u = (cur > 0).astype(np.uint8)
            ll  = (u[:, 0:n:2] & u[:, 1:n:2]).astype(np.float32)
            xor = (u[:, 0:n:2] ^ u[:, 1:n:2]).astype(np.float32)
            feats_per_depth[d].append(np.concatenate([ll.mean(axis=1),  ll.std(axis=1),
                                                       xor.mean(axis=1), xor.std(axis=1)]))
            cur = ll
        labels.append(q - 1)

    if not labels:
        print("  No valid clips — skipping.")
        return
    if skipped:
        print(f"  (skipped {skipped} clips)")

    y         = np.array(labels)
    y_arousal = (y < 2).astype(int)
    y_valence = ((y == 0) | (y == 3)).astype(int)
    chance4   = np.bincount(y, minlength=4).max() / len(y)

    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))
    print(f"\n{'Depth':>6}  {'4-class':>9}  {'Arousal':>9}  {'Valence':>9}  (chance: {chance4:.2f} / 0.50 / 0.50)")
    for d in range(n_depths + 1):
        X_list = feats_per_depth[d]
        if len(X_list) < 20: continue
        X  = np.stack(X_list)
        yl = y[:len(X)]   # trim if some depths ran short
        a4 = cross_val_score(clf,X, yl,                cv=5, scoring='accuracy', n_jobs=1).mean()
        aa = cross_val_score(clf,X, y_arousal[:len(X)], cv=5, scoring='accuracy', n_jobs=1).mean()
        av = cross_val_score(clf,X, y_valence[:len(X)], cv=5, scoring='accuracy', n_jobs=1).mean()
        print(f"  D{d}    {a4:.3f}      {aa:.3f}      {av:.3f}")
    print("  (D0=raw, D1..5=progressively coarser Haar LL+XOR subbands)")


# ── Probe 8: melody / accompaniment separation (POP909) ──────────────────────

def run_melody_separation_probe(encoder, cfg, pop909_dir, device, n_songs=200):
    """
    Load MELODY/PIANO/TOTAL triplets from POP909_images.
    For each song take a random same-position crop of each, encode, and measure
    pairwise distances per level. Good content-sensitive encoders should show
    MELODY↔PIANO > MELODY↔TOTAL ≈ PIANO↔TOTAL.
    """
    pop909_dir = os.path.expandvars(os.path.expanduser(pop909_dir))
    totals = sorted(glob.glob(os.path.join(pop909_dir, '*', '*_TOTAL.png')))
    print(f"\n=== Probe 8: Melody/Accompaniment Separation (POP909) ===")
    if not totals:
        print(f"  No *_TOTAL.png found in {pop909_dir} — skipping.")
        return

    rng = np.random.default_rng(42)
    totals = list(rng.permutation(totals)[:n_songs])
    crop_size = cfg.data.image_size

    n_levels = None
    dists_per_level = {}
    skipped = 0

    for tot_path in tqdm(totals, desc='Melody probe', leave=False):
        song_dir = os.path.dirname(tot_path)
        base = os.path.basename(tot_path).replace('_TOTAL.png', '')
        mel_path = os.path.join(song_dir, f'{base}_MELODY.png')
        pia_path = os.path.join(song_dir, f'{base}_PIANO.png')
        if not os.path.exists(mel_path) or not os.path.exists(pia_path):
            skipped += 1
            continue

        try:
            def load_bin(path):
                arr = np.array(Image.open(path).convert('RGB'))
                return (arr > 0).any(axis=2).astype(np.float32)   # (H, W)
            bin_mel = load_bin(mel_path)
            bin_pia = load_bin(pia_path)
            bin_tot = load_bin(tot_path)
        except Exception:
            skipped += 1
            continue

        H, W = bin_mel.shape
        # detect long axis (time) and crop
        if H >= W:  # (N_time, 128)
            long_len = H
            make_crop = lambda img, x: img[x:x+crop_size, :]
        else:        # (128, N_time)
            long_len = W
            make_crop = lambda img, x: img[:, x:x+crop_size]

        if long_len < crop_size:
            skipped += 1
            continue

        x = int(rng.integers(0, long_len - crop_size))
        crop_mel = make_crop(bin_mel, x)
        crop_pia = make_crop(bin_pia, x)
        crop_tot = make_crop(bin_tot, x)

        def to_tensor(crop):
            return torch.from_numpy(crop).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            e_mel = encoder(to_tensor(crop_mel))
            e_pia = encoder(to_tensor(crop_pia))
            e_tot = encoder(to_tensor(crop_tot))

        if n_levels is None:
            n_levels = e_mel.patches.num_levels
            dists_per_level = {k: [[] for _ in range(n_levels)]
                               for k in ('mel_pia', 'mel_tot', 'pia_tot')}

        for lev in range(n_levels):
            z_mel = e_mel.patches.levels[lev].emb.cpu().float()  # (1, N, D)
            z_pia = e_pia.patches.levels[lev].emb.cpu().float()
            z_tot = e_tot.patches.levels[lev].emb.cpu().float()
            dists_per_level['mel_pia'][lev].append((z_mel - z_pia).norm(dim=-1).mean().item())
            dists_per_level['mel_tot'][lev].append((z_mel - z_tot).norm(dim=-1).mean().item())
            dists_per_level['pia_tot'][lev].append((z_pia - z_tot).norm(dim=-1).mean().item())

    if n_levels is None:
        print("  No valid triplets found — skipping.")
        return

    if skipped:
        print(f"  (skipped {skipped}/{len(totals)} songs)")
    print(f"{'Level':>6}  {'mel↔pia':>10}  {'mel↔tot':>10}  {'pia↔tot':>10}")
    for lev in range(n_levels):
        d_mp = np.mean(dists_per_level['mel_pia'][lev])
        d_mt = np.mean(dists_per_level['mel_tot'][lev])
        d_pt = np.mean(dists_per_level['pia_tot'][lev])
        print(f"  L{lev}    {d_mp:.4f}      {d_mt:.4f}      {d_pt:.4f}")
    print("  (mel↔pia > mel↔tot ≈ pia↔tot → content-sensitive embedding)")


# ── Summary plot ─────────────────────────────────────────────────────────────

def plot_summary(n_levels, qual_accs, root_accs, chance_qual, chance_root):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    levels = list(range(n_levels))
    ax1.plot(levels, qual_accs, marker='o', label='Chord quality')
    ax1.axhline(chance_qual, color='gray', ls='--', label=f'Chance ({chance_qual:.2f})')
    ax1.set_xlabel('Level (0=coarsest)'); ax1.set_ylabel('5-fold CV accuracy')
    ax1.set_title('Chord Quality Probe'); ax1.legend(); ax1.set_ylim(0, 1)

    ax2.plot(levels, root_accs, marker='o', color='orange', label='Root note')
    ax2.axhline(1/12, color='gray', ls='--', label=f'Chance ({1/12:.2f})')
    ax2.set_xlabel('Level (0=coarsest)'); ax2.set_ylabel('5-fold CV accuracy')
    ax2.set_title('Root Note Probe'); ax2.legend(); ax2.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig('probe_summary.png', dpi=120)
    print(f"\nSaved → probe_summary.png")


# ── Parallel worker functions ─────────────────────────────────────────────────

class _Tee(io.TextIOBase):
    """Write to both live stdout and a buffer simultaneously."""
    def __init__(self, live, buf):
        self._live, self._buf = live, buf
    def write(self, s):
        self._live.write(s); self._live.flush()
        self._buf.write(s)
        return len(s)
    def flush(self):
        self._live.flush()


def _worker_group_a(embeddings_per_level, root_labels, group_labels, file_idxs, densities, n_levels):
    """Probes 1, 2, 4, 5 — sklearn only, reads pre-computed embeddings."""
    import sys
    buf = io.StringIO()
    sys.stdout = _Tee(sys.__stdout__, buf)
    try:
        qual_accs, root_accs, chance_qual, chance_root = run_chord_probes(
            embeddings_per_level, root_labels, group_labels, n_levels)
        cross_song_ratios = run_cross_song_probe(embeddings_per_level, file_idxs, n_levels)
        density_r2s = run_density_probe(embeddings_per_level, densities, n_levels)
    finally:
        sys.stdout = sys.__stdout__
    return buf.getvalue(), qual_accs, root_accs, chance_qual, chance_root, cross_song_ratios, density_r2s


def _worker_group_b(ckpt, config_path, data_dir, device, n_crops, emopia_dir, pop909_dir, flags):
    """Probes 3, 6, 7, 8, 9 — loads its own encoder, does forward passes."""
    import sys
    cfg = OmegaConf.load(config_path)
    encoder = build_encoder(cfg, device)
    encoder = load_checkpoint(encoder, ckpt)
    encoder.eval()
    ds = PRPairDataset(image_dataset_dir=os.path.expandvars(os.path.expanduser(data_dir)),
                       split='val', verbose=False)
    buf = io.StringIO()
    sys.stdout = _Tee(sys.__stdout__, buf)
    transp_shifts, transp_means = None, None
    time_shifts, time_means = None, None
    temp_r2s = None
    emopia_data = None
    try:
        if not flags['no_transposition']:
            transp_shifts, transp_means = run_transposition_probe(encoder, ds, device, n_samples=n_crops)
        if not flags['no_time']:
            time_shifts, time_means = run_time_translation_probe(encoder, ds, device, n_samples=n_crops)
            temp_r2s = run_temporal_distance_probe(encoder, ds, device, n_pairs=n_crops)
        if not flags['no_emopia']:
            result = run_emopia_emotion_probe(encoder, cfg, emopia_dir, device)
            if result is not None:
                emopia_data = result
            if not flags['no_haar_baseline']:
                run_haar_emopia_baseline(emopia_dir, cfg)
        if not flags['no_melody']:
            run_melody_separation_probe(encoder, cfg, pop909_dir, device)
    finally:
        sys.stdout = sys.__stdout__
    return buf.getvalue(), transp_shifts, transp_means, time_shifts, time_means, temp_r2s, emopia_data


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',    required=True, help='Path to SwinEncoder checkpoint')
    p.add_argument('--config',  required=True, help='Path to config_swin.yaml used for that run')
    p.add_argument('--data',    default='$HOME/datasets/POP909_images_basic')
    p.add_argument('--n_crops', type=int, default=2000)
    p.add_argument('--device',  default='cpu')
    p.add_argument('--pop909',   default='$HOME/datasets/POP909_images', help='POP909_images dir for probe 8')
    p.add_argument('--emopia',   default='$HOME/datasets/EMOPIA', help='EMOPIA dir for probe 9 (auto-downloaded if absent)')
    p.add_argument('--no_transposition', action='store_true', help='Skip pitch transposition probe')
    p.add_argument('--no_time',   action='store_true', help='Skip time translation + temporal distance probes')
    p.add_argument('--no_melody', action='store_true', help='Skip melody/accompaniment separation probe')
    p.add_argument('--no_emopia',        action='store_true', help='Skip EMOPIA emotion probe')
    p.add_argument('--no_haar_baseline', action='store_true', help='Skip Haar baseline on EMOPIA')
    p.add_argument('--no_wandb',  action='store_true', help='Disable W&B logging')
    p.add_argument('--wandb_tag', default=None, help='W&B run name (defaults to checkpoint basename)')
    args = p.parse_args()

    device = args.device
    cfg = OmegaConf.load(args.config)

    print("Loading encoder...")
    encoder = build_encoder(cfg, device)
    encoder = load_checkpoint(encoder, args.ckpt)
    encoder.eval()

    print(f"Loading {args.n_crops} crops from AnchorDataset...")
    ds = AnchorDataset(image_dataset_dir=os.path.expandvars(os.path.expanduser(args.data)), split='val', verbose=True)

    embeddings_per_level = None
    root_labels, group_labels, file_idxs, densities = [], [], [], []
    n_levels = None

    for i in tqdm(range(args.n_crops), desc='Encoding crops'):
        sample = ds[i]
        img_t = sample['img'].unsqueeze(0).to(device)   # (1, 1, H, W)
        img_np = sample['img'].squeeze(0).numpy().astype(np.uint8)  # (H, W)

        # chord label from chroma
        chroma = image_to_chroma(img_np)
        root, qual, chord, group = chroma_to_chord(chroma)
        if root is None:
            continue  # skip empty crops
        root_labels.append(root)
        group_labels.append(group)
        file_idxs.append(int(sample['file_idx']))
        densities.append(float(img_np.mean()))

        # encoder forward
        with torch.no_grad():
            enc_out = encoder(img_t)

        level_embs = get_level_embeddings(enc_out)  # list of (1, dim) arrays
        if embeddings_per_level is None:
            n_levels = len(level_embs)
            embeddings_per_level = [[] for _ in range(n_levels)]
        for lev, e in enumerate(level_embs):
            embeddings_per_level[lev].append(e[0])   # (dim,)

    # stack
    embeddings_per_level = [np.stack(embeddings_per_level[lev]) for lev in range(n_levels)]
    root_labels  = np.array(root_labels)
    group_labels = np.array(group_labels)
    file_idxs    = np.array(file_idxs)
    densities    = np.array(densities)

    print(f"\nCollected {len(root_labels)} crops across {len(set(file_idxs))} files.")
    unique, counts = np.unique(group_labels, return_counts=True)
    print("Chord quality distribution:")
    for q, c in sorted(zip(unique, counts), key=lambda x: -x[1]):
        print(f"  {q:8s}: {c:5d}  ({100*c/len(group_labels):.1f}%)")

    # Run probes in parallel: group A (sklearn) and group B (encoder forward passes)
    flags = {k: getattr(args, k) for k in
             ('no_transposition', 'no_time', 'no_emopia', 'no_haar_baseline', 'no_melody')}
    print("\nLaunching probe groups in parallel...")
    ctx = mp.get_context('spawn')
    with ctx.Pool(2) as pool:
        r_a = pool.apply_async(_worker_group_a, (
            embeddings_per_level, root_labels, group_labels, file_idxs, densities, n_levels))
        r_b = pool.apply_async(_worker_group_b, (
            args.ckpt, args.config, args.data, args.device, args.n_crops,
            args.emopia, args.pop909, flags))
        text_a, qual_accs, root_accs, chance_qual, chance_root, cross_song_ratios, density_r2s = r_a.get()
        text_b, transp_shifts, transp_means, time_shifts, time_means, temp_r2s, emopia_data = r_b.get()

    log_path = 'stormbird_results.log'
    with open(log_path, 'w') as f:
        f.write(text_a)
        f.write(text_b)
    print(f"\n── Ordered results ──")
    print(text_a)
    print(text_b)
    print(f"Ordered log saved → {log_path}")
    plot_summary(n_levels, qual_accs, root_accs, chance_qual, chance_root)

    # ── W&B logging ──────────────────────────────────────────────────────────
    if not args.no_wandb:
        run_name = args.wandb_tag or os.path.splitext(os.path.basename(args.ckpt))[0]
        wandb.init(project='probe_musicality', name=run_name,
                   config={**OmegaConf.to_container(cfg, resolve=True),
                           'ckpt': args.ckpt, 'n_crops': args.n_crops, 'device': args.device})
        levels = list(range(n_levels))
        level_keys = [f'L{l}' for l in levels]

        # Scalar metrics per level
        log_dict = {}
        for l in levels:
            log_dict[f'chord_quality/L{l}']   = qual_accs[l]
            log_dict[f'root_note/L{l}']        = root_accs[l]
            if cross_song_ratios: log_dict[f'cross_song_ratio/L{l}'] = cross_song_ratios[l]
            if density_r2s:       log_dict[f'density_r2/L{l}']       = density_r2s[l]
            if temp_r2s:          log_dict[f'temporal_dist_r2/L{l}']  = temp_r2s[l]
            if emopia_data:
                log_dict[f'emopia_4class/L{l}'] = emopia_data[0][l]
                log_dict[f'emopia_arousal/L{l}'] = emopia_data[1][l]
                log_dict[f'emopia_valence/L{l}'] = emopia_data[2][l]
        wandb.log(log_dict)

        # Equivariance curves — one metric per level with custom x-axis step.
        # In the W&B UI: create one panel, filter on pitch_transposition/* or
        # time_translation/*, set x-axis to the step metric → multi-line panel
        # with log-scale toggle available via the pencil icon.
        if transp_shifts and transp_means:
            wandb.define_metric('pitch_shift_semitones')
            wandb.define_metric('pitch_transposition/*', step_metric='pitch_shift_semitones')
            for i, shift in enumerate(transp_shifts):
                wandb.log({'pitch_shift_semitones': shift,
                           **{f'pitch_transposition/L{l}_{128//2**l}px': transp_means[l][i]
                              for l in levels}})
        if time_shifts and time_means:
            wandb.define_metric('time_shift_pixels')
            wandb.define_metric('time_translation/*', step_metric='time_shift_pixels')
            for i, shift in enumerate(time_shifts):
                wandb.log({'time_shift_pixels': shift,
                           **{f'time_translation/L{l}_{128//2**l}px': time_means[l][i]
                              for l in levels}})

        wandb.finish()
        print("W&B run logged.")

    print("\nDone.")


if __name__ == '__main__':
    main()

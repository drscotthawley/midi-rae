"""Project freshly pre-encoded embeddings using the ORIGINAL PCA transforms.

fit_pca wipes and refits, which would give a transform the existing flow
checkpoints were never trained against -- PCA component signs are only stable for
identical input, so a refit can silently invalidate resuming. This reuses the
saved pickles and only re-projects.

Run on tsrazer: ~/envs/midi-rae/bin/python _project_saved_pca.py
"""
import pickle
import shutil
import sys
from pathlib import Path

from midi_rae.fit_pca import _project_chunks

ENC = Path("/home/shawley/datasets/POP909_encoded_C55cUL")
OUT = Path("/home/shawley/datasets/POP909_pca_C55cUL")
ORIG = Path("/home/shawley/datasets/PCA_ORIGINAL_C55cUL")
LEVELS = [0, 1, 2, 3, 4, 5]
KEY = "emb2"


def main():
    if not ORIG.exists():
        sys.exit(f"missing original PCA pickles at {ORIG}")
    OUT.mkdir(parents=True, exist_ok=True)

    pcas = []
    for i in LEVELS:
        hits = sorted(ORIG.glob(f"pca_L{i}_n*.pkl"))
        if not hits:
            sys.exit(f"no saved PCA for level {i} in {ORIG}")
        with open(hits[0], "rb") as f:
            pcas.append(pickle.load(f))
        shutil.copy(hits[0], OUT / hits[0].name)      # keep pickles beside the chunks
        print(f"L{i}: reusing {hits[0].name} ({pcas[-1].n_components_} components)")

    for split in ("train", "val"):
        n = len(list(ENC.glob(f"{split}_chunk*.pt")))
        print(f"\nprojecting {n} {split} chunks with the saved transforms...")
        _project_chunks(ENC, OUT, LEVELS, pcas, KEY, split)

    made = sorted(OUT.glob("*_pca.pt"))
    print(f"\nwrote {len(made)} projected chunks to {OUT}")


if __name__ == "__main__":
    main()

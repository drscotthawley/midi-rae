#!/usr/bin/env python3
"""
Lakh-specific data-cleaning pass: walk every MIDI file under a directory, try
to parse it with pretty_midi, and move any file that fails to parse into a
sibling `_corrupted` directory (preserving relative path), so corrupt files
are distinguishable from files our conversion pipeline itself skips/mishandles.

Embarrassingly parallel across CPU cores via multiprocessing.Pool.

Usage:
    python clean_lakh.py <midi_dir> [--corrupted-dir DIR] [--workers N]
"""
import os
import shutil
import argparse
from multiprocessing import Pool, cpu_count
from functools import partial
import pretty_midi
from tqdm import tqdm


def fast_scandir(root, exts):
    subdirs, files = [], []
    for entry in os.scandir(root):
        if entry.is_dir(follow_symlinks=False):
            subdirs.append(entry.path)
        elif entry.is_file() and entry.name.rsplit('.', 1)[-1].lower() in exts:
            files.append(entry.path)
    for d in list(subdirs):
        sd, sf = fast_scandir(d, exts)
        subdirs += sd
        files += sf
    return subdirs, files


def check_one(midi_file, root_dir, corrupted_dir):
    try:
        pretty_midi.PrettyMIDI(midi_file)
        return None  # valid, leave in place
    except Exception as e:
        rel = os.path.relpath(midi_file, root_dir)
        dest = os.path.join(corrupted_dir, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        try:
            shutil.move(midi_file, dest)
        except Exception:
            pass
        return (rel, f"{type(e).__name__}: {e}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('midi_dir', help='Directory containing (nested) MIDI files, e.g. lmd_full/')
    p.add_argument('--corrupted-dir', default=None,
                    help='Where to move corrupt files (default: <midi_dir>_corrupted, sibling directory)')
    p.add_argument('--workers', type=int, default=cpu_count(), help='Number of parallel workers')
    args = p.parse_args()

    midi_dir = os.path.abspath(args.midi_dir)
    corrupted_dir = args.corrupted_dir or (midi_dir.rstrip('/') + '_corrupted')
    os.makedirs(corrupted_dir, exist_ok=True)

    print(f"Scanning {midi_dir} for .mid/.midi files...")
    _, midi_files = fast_scandir(midi_dir, {'mid', 'midi'})
    print(f"Found {len(midi_files)} files. Checking with {args.workers} workers...")

    worker = partial(check_one, root_dir=midi_dir, corrupted_dir=corrupted_dir)
    corrupt_log = []
    with Pool(args.workers) as pool:
        for result in tqdm(pool.imap_unordered(worker, midi_files), total=len(midi_files), desc='Checking MIDI files'):
            if result is not None:
                corrupt_log.append(result)

    print(f"\nDone. {len(corrupt_log)} / {len(midi_files)} files were corrupt and moved to {corrupted_dir}")
    log_path = os.path.join(corrupted_dir, '_corrupt_log.txt')
    with open(log_path, 'w') as f:
        for rel, err in corrupt_log:
            f.write(f"{rel}\t{err}\n")
    print(f"Log written to {log_path}")

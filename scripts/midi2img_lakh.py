#!/usr/bin/env python3
"""
Standalone MIDI -> piano-roll image converter for the Lakh MIDI dataset.
No dependency on control-toys -- just pretty_midi, numpy, and PIL.

Unlike POP909 (which has real MELODY/PIANO-named tracks), arbitrary scraped
MIDI like Lakh has no consistent track-naming convention. Files with no
MELODY/PIANO tracks get all their instruments flattened together and saved
as a single "_TOTAL" image (matching POP909's convention where TOTAL means
"all instruments combined, no melody/accompaniment distinction"). Corrupt or
unparseable files are skipped (not crashed on).

Usage:
    python midi2img_lakh.py <midi_dir> <output_dir> [--workers N]
"""
import os
import argparse
from multiprocessing import Pool, cpu_count
from functools import partial

import numpy as np
import pretty_midi
from PIL import Image
from tqdm import tqdm

ONSET_STYLE = 'new'


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


def find_first_note_start(midi):
    first_start = 10000.0
    for instrument in midi.instruments:
        for note in instrument.notes:
            if note.start < first_start:
                first_start = note.start
    return first_start


def check_for_melody_piano(midi: pretty_midi.PrettyMIDI):
    has_melody, has_piano = False, False
    for instrument in midi.instruments:
        name = instrument.name.upper()
        if name == 'MELODY': has_melody = True
        if name == 'PIANO':  has_piano = True
    if len(midi.instruments) == 1 and midi.instruments[0].name == '':
        has_piano = True
        midi.instruments[0].name = 'PIANO'
    return has_melody, has_piano


def flatten_to_total(midi: pretty_midi.PrettyMIDI):
    """No real MELODY/PIANO separation exists -- rename every instrument to
    'PIANO' so all notes combine into one flattened track; the caller then
    saves only the resulting TOTAL image (PIANO/MELODY would be redundant/blank)."""
    for instrument in midi.instruments:
        instrument.name = 'PIANO'


def get_piano_rolls(midi, fs, remove_leading_silence=True):
    duration = midi.get_end_time()
    n_frames = int(np.ceil(duration * fs))
    piano_rolls = {'PIANO':  np.zeros((128, n_frames)),
                   'MELODY': np.zeros((128, n_frames)),
                   'TOTAL':  np.zeros((128, n_frames))}
    if remove_leading_silence:
        first_start = find_first_note_start(midi)

    for instrument in midi.instruments:
        name = instrument.name.upper()
        if name in ['MELODY', 'PIANO']:
            for note in instrument.notes:
                if remove_leading_silence:
                    note.start -= first_start
                    note.end -= first_start
                start = int(np.round(note.start * fs))
                dur = (note.end - note.start) * fs
                end = start + int(np.round(dur))
                if end == start: end = start + 1
                piano_rolls[name][note.pitch, start:end] = note.velocity
                piano_rolls['TOTAL'][note.pitch, start:end] = note.velocity
                piano_rolls[name][note.pitch, start - 1] = 0
                piano_rolls['TOTAL'][note.pitch, start - 1] = 0
    return piano_rolls


def is_green(r, g, b, thresh=20): return r < thresh and g > thresh and b < thresh
def is_black(r, g, b, thresh=20): return r < thresh and g < thresh and b < thresh


def piano_roll_to_img(pr_frame, output_dir, midi_name, instrument, add_onsets=True):
    # Skip empty/zero-duration rolls: a 0-frame roll makes a zero-width image, and
    # the onset loop's getpixel((0, y)) below would raise IndexError on it.
    if pr_frame.ndim != 2 or 0 in pr_frame.shape:
        return
    os.makedirs(f"{output_dir}/{midi_name}", exist_ok=True)
    filename = f"{output_dir}/{midi_name}/{midi_name}_{instrument}.png"

    scale_factor = 2
    green_channel = np.clip(np.round(pr_frame * scale_factor), 0, 255).astype(np.uint8)
    rgb_image = np.dstack((np.zeros_like(green_channel), green_channel, np.zeros_like(green_channel)))
    img = Image.fromarray(rgb_image, 'RGB')

    if add_onsets:
        for y in range(img.size[-1]):
            x = 0
            pxl = img.getpixel((x, y))
            if is_green(*pxl):
                img.putpixel((0, y), (pxl[1], 0, 0))
            for x in range(1, img.size[0]):
                pxl = img.getpixel((x, y))
                if is_green(*pxl) and is_black(*img.getpixel((x - 1, y))):
                    img.putpixel((x, y), (pxl[1], 0, 0))

    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    if 0 in img.size:
        print(f"Error: img.size = {img.size}. Skipping this file.")
        return
    img.save(filename)


def midi_to_pr_img(midi_file, output_dir, add_onsets=True, remove_leading_silence=True, filter_mp=True, debug=False):
    if debug: print(f"midi_to_pr_img: Processing {midi_file}")
    try:
        midi = pretty_midi.PrettyMIDI(midi_file)
    except Exception as e:
        print(f"Corrupt/unparseable: File {midi_file} raised {type(e).__name__}: {e}. Skipping")
        return

    has_melody, has_piano = check_for_melody_piano(midi)
    flattened = not (has_melody or has_piano)
    if flattened:
        flatten_to_total(midi)

    tempo_changes = midi.get_tempo_changes()
    if len(tempo_changes[1]) == 0:
        print(f"No tempo info: File {midi_file}. Skipping")
        return
    start_tempo = tempo_changes[1][0]
    if start_tempo <= 0:
        print(f"Invalid tempo ({start_tempo}): File {midi_file}. Skipping")
        return
    bps = start_tempo / 60.0
    fs = bps * 4.0 * 2

    if filter_mp:
        midi.instruments = [i for i in midi.instruments if i.name.upper() in ['MELODY', 'PIANO']]

    piano_rolls = get_piano_rolls(midi, fs, remove_leading_silence=remove_leading_silence)
    midi_name = os.path.basename(midi_file).split('.')[0]

    instruments_to_save = ['TOTAL'] if flattened else list(piano_rolls.keys())
    for instrument in instruments_to_save:
        piano_roll_to_img(piano_rolls[instrument], output_dir, midi_name, instrument, add_onsets=add_onsets)


def wrapper(args, midi_file):
    # Resilience: never let one unexpected per-file error kill the whole 174K-file pool.
    try:
        return midi_to_pr_img(midi_file, args.output_dir, add_onsets=args.onsets,
                              remove_leading_silence=(not args.silence), filter_mp=args.filter_mp)
    except Exception as e:
        print(f"Skipping {midi_file}: {type(e).__name__}: {e}")
        return None


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('midi_dir', help='Directory containing (nested) MIDI files')
    p.add_argument('output_dir', help='Output directory for piano-roll images')
    p.add_argument('--onsets', default=True, action=argparse.BooleanOptionalAction, help='Add onset markers')
    p.add_argument('--silence', default=False, action=argparse.BooleanOptionalAction,
                   help='Leave silence at start of song (True) or remove it (False, default)')
    p.add_argument('--filter-mp', default=True, action=argparse.BooleanOptionalAction,
                   help='Filter out non-piano, non-melody instruments before rendering')
    p.add_argument('--workers', type=int, default=cpu_count(), help='Number of parallel workers')
    args = p.parse_args()
    print("args =", args)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Scanning {args.midi_dir} for .mid/.midi files...")
    _, midi_files = fast_scandir(args.midi_dir, {'mid', 'midi'})
    print(f"Found {len(midi_files)} files.")

    process_one = partial(wrapper, args)
    with Pool(args.workers) as pool:
        list(tqdm(pool.imap(process_one, midi_files), total=len(midi_files), desc='Converting MIDI files'))

    print("Finished")

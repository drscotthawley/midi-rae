"""Lean piano-roll-image -> MIDI conversion for the demo.

The generated frames are single 128x128 rolls, so we don't need control-toys'
grid/panorama machinery (regroup_lines / filter_redgreen / CHORD_BORDER) which
targets multi-crop stitched images. We port only the ~30-line, self-contained
`piano_roll_to_pretty_midi` (originally from pretty_midi's reverse_pianoroll
example) and add a small front-end that un-flips the decoder's image-orientation
roll into (pitch, frames) and assigns a constant velocity.

fs=16 corresponds to the forward pipeline's `bps*4*2` at 120 BPM (see
scripts/midi2img_lakh.py), so playback timing matches the training convention.
"""
from pathlib import Path
import tempfile

import numpy as np
import pretty_midi

FS_DEFAULT = 16          # frames/sec == bps*4*2 at 120 BPM
DEFAULT_VELOCITY = 96


def piano_roll_to_pretty_midi(piano_roll, fs=8, program=0):
    """Convert a (128, frames) piano-roll array into a pretty_midi.PrettyMIDI.

    Ported verbatim (behaviour-preserving) from
    control-toys/control_toys/pianoroll.py, which copied it from
    github.com/jsleep/pretty-midi reverse_pianoroll.py example.
    """
    notes, frames = piano_roll.shape
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)

    # pad 1 column of zeros so we can acknowledge initial and ending events
    piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], "constant")

    # use changes in velocities to find note on / note off events
    velocity_changes = np.nonzero(np.diff(piano_roll).T)

    prev_velocities = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)

    for time, note in zip(*velocity_changes):
        velocity = np.clip(piano_roll[note, time + 1], 0, 127)
        time = time / fs
        if velocity > 0:
            if prev_velocities[note] == 0:
                note_on_time[note] = time
                prev_velocities[note] = velocity
        else:
            pm_note = pretty_midi.Note(
                velocity=int(prev_velocities[note]),
                pitch=int(note),
                start=note_on_time[note],
                end=time,
            )
            instrument.notes.append(pm_note)
            prev_velocities[note] = 0
    pm.instruments.append(instrument)
    return pm


def roll_to_pretty_midi(roll, fs=FS_DEFAULT, velocity=DEFAULT_VELOCITY, image_orientation=True):
    """(128,128) note-presence roll -> pretty_midi.PrettyMIDI.

    image_orientation=True: input row 0 is the TOP of the piano-roll image
    (highest pitch), as produced by the decoder / midi2img forward pass, so we
    flip vertically to get (pitch 0..127, frames) before decoding to notes.
    """
    roll = np.asarray(roll, dtype=np.float32)
    if image_orientation:
        roll = np.flipud(roll)
    binary = (roll > 0.5).astype(np.int32)
    pr = binary * int(velocity)          # constant-velocity notes
    return piano_roll_to_pretty_midi(pr, fs=fs)


def roll_to_midi_file(roll, out_path=None, fs=FS_DEFAULT, velocity=DEFAULT_VELOCITY,
                      image_orientation=True):
    """Write a roll to a .mid file and return its path (a temp file if out_path is None)."""
    pm = roll_to_pretty_midi(roll, fs=fs, velocity=velocity, image_orientation=image_orientation)
    if out_path is None:
        fd, out_path = tempfile.mkstemp(suffix=".mid", prefix="midirae_gen_")
        import os
        os.close(fd)
    out_path = str(out_path)
    pm.write(out_path)
    return out_path

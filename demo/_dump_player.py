"""Inspect what MIDIPlayer.html emits, to diagnose why it renders blank in gr.HTML."""
import sys, tempfile, os
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import numpy as np, img2midi

# make a tiny valid midi
roll = np.zeros((128, 128), np.float32); roll[60, 10:40] = 1; roll[64, 20:50] = 1
mid = img2midi.roll_to_midi_file(roll)

from midi_player import MIDIPlayer
from midi_player.stylers import dark
html = MIDIPlayer(mid, 300, title="t", styler=dark).html

print("=== length:", len(html))
print("=== contains <iframe:", "<iframe" in html, " srcdoc:", "srcdoc" in html)
print("=== contains <script:", "<script" in html, " <midi-player:", "<midi-player" in html)
print("=== data uri base64:", "data:audio/midi;base64" in html)
print("=== FIRST 1500 CHARS ===")
print(html[:1500])
print("=== LAST 600 CHARS ===")
print(html[-600:])

# playwright available?
try:
    import playwright  # noqa
    print("\n=== playwright: AVAILABLE", playwright.__version__ if hasattr(playwright,'__version__') else '')
except Exception as e:
    print("\n=== playwright: NOT available:", e)

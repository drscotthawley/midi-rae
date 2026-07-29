"""Headless-browser check that the iframe-wrapped MIDI player actually renders.

Loads the exact HTML app.midi_player_html() produces, waits for the CDN scripts,
and verifies the <midi-player> custom element registers and builds its shadow DOM.
"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))

import numpy as np
import img2midi
import app
from playwright.sync_api import sync_playwright

roll = np.zeros((128, 128), np.float32); roll[60, 10:40] = 1; roll[64, 20:55] = 1
mid = img2midi.roll_to_midi_file(roll)
player_html = app.midi_player_html(mid)
page = f"<!doctype html><html><head><meta charset='utf-8'></head><body>{player_html}</body></html>"

with sync_playwright() as p:
    browser = p.chromium.launch()
    pg = browser.new_page()
    errors = []
    pg.on("console", lambda m: errors.append(f"{m.type}: {m.text}") if m.type == "error" else None)
    pg.set_content(page, wait_until="load")
    # wait for the CDN <script> in the srcdoc iframe to define the element
    iframe_el = pg.wait_for_selector("iframe")
    frame = iframe_el.content_frame()
    ok = False
    for _ in range(20):
        pg.wait_for_timeout(500)
        defined = frame.evaluate("() => !!customElements.get('midi-player')")
        shadow = frame.evaluate("""() => {
            const mp = document.querySelector('midi-player');
            return !!(mp && mp.shadowRoot && mp.shadowRoot.querySelectorAll('*').length > 3);
        }""")
        if defined and shadow:
            ok = True
            break
    n_shadow = frame.evaluate("""() => {
        const mp = document.querySelector('midi-player');
        return (mp && mp.shadowRoot) ? mp.shadowRoot.querySelectorAll('*').length : -1;
    }""")
    is_defined = frame.evaluate("() => !!customElements.get('midi-player')")
    print(f"customElements 'midi-player' defined: {is_defined}")
    print(f"shadow DOM element count: {n_shadow}")
    print(f"console errors: {errors[:5]}")
    print("RENDER OK" if ok else "RENDER FAILED")
    browser.close()

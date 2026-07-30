"""Construct the Gradio UI and exit. Loads no model -- get_demo() is lazy, so this
only exercises component construction and event wiring (where version-specific
kwargs like show_download_button blow up)."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))

import app

ui = app.build_ui()
print(f"build_ui() ok: {type(ui).__name__}")

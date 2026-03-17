#!/usr/bin/env python3
"""Read cell source(s) from a Jupyter notebook.

Usage:
  python scripts/read_nb.py <notebook.ipynb> [cell_id ...]

With no cell IDs: prints all cell IDs and first 60 chars of source.
With cell IDs: prints full source of matching cells.
Cell IDs can be partial matches (substring).
"""
import json, sys

nb_path = sys.argv[1]
cell_ids = sys.argv[2:]

with open(nb_path) as f:
    nb = json.load(f)

for cell in nb['cells']:
    cid = cell.get('id', '')
    src = ''.join(cell['source'])
    if not cell_ids:
        print(f"{cid}  {src[:60].replace(chr(10), ' ')}")
    elif any(q in cid for q in cell_ids):
        print(f"=== {cid} ===")
        print(src)
        print()

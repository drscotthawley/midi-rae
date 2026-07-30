#!/usr/bin/env python3
"""Read cell source(s) from a Jupyter notebook.

Usage:
  python scripts/read_nb.py <notebook.ipynb> [cell_id ...]

With no cell IDs: prints all cell IDs and first 60 chars of source.
With --all: prints full source of all cells.
With cell IDs: prints full source of matching cells.
Cell IDs can be partial matches (substring).
"""
import json, sys

nb_path = sys.argv[1]
args = sys.argv[2:]
show_all = '--all' in args
cell_ids = [a for a in args if a != '--all']

with open(nb_path) as f:
    nb = json.load(f)

for cell in nb['cells']:
    cid = cell.get('id', '')
    src = ''.join(cell['source'])
    if show_all or (cell_ids and any(q in cid for q in cell_ids)):
        print(f"=== {cid} ===")
        print(src)
        print()
    elif not cell_ids:
        print(f"{cid}  {src[:60].replace(chr(10), ' ')}")

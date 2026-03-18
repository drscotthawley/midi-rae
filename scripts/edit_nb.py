#!/usr/bin/env python3
"""Edit a cell in a Jupyter notebook by replacing a string in its source.

Usage:
    python scripts/edit_nb.py <notebook> <cell_id> <old_string> <new_string>

The cell_id can be a prefix. old_string and new_string are read from files
if they start with '@' (e.g. @/tmp/old.txt), otherwise used as-is.
"""
import json, sys, pathlib

def load_str(s):
    if s.startswith('@'):
        return pathlib.Path(s[1:]).read_text()
    return s

nb_path, cell_id, old_str, new_str = sys.argv[1], sys.argv[2], load_str(sys.argv[3]), load_str(sys.argv[4])

nb = json.load(open(nb_path))
cell = next((c for c in nb['cells'] if c.get('id','').startswith(cell_id)), None)
if cell is None:
    print(f"ERROR: cell {cell_id!r} not found", file=sys.stderr); sys.exit(1)

src = ''.join(cell['source'])
if old_str not in src:
    # Try matching with trailing whitespace stripped per line
    def strip_trailing(s): return '\n'.join(l.rstrip() for l in s.split('\n'))
    src_stripped = strip_trailing(src)
    old_stripped = strip_trailing(old_str)
    if old_stripped not in src_stripped:
        print(f"ERROR: old_string not found in cell {cell_id}", file=sys.stderr); sys.exit(1)
    # Reconstruct: find the region in original src that matches stripped version
    src = src_stripped
    old_str = old_stripped

cell['source'] = src.replace(old_str, new_str, 1)
json.dump(nb, open(nb_path, 'w'), indent=1)
print(f"OK: updated cell {cell_id} in {nb_path}")

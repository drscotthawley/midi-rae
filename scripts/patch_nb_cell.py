"""Patch a specific cell in a Jupyter notebook by cell ID.
Usage: python scripts/patch_nb_cell.py <notebook_path> <cell_id> <new_source_file>
   or: python scripts/patch_nb_cell.py <notebook_path> <cell_id> --stdin  (reads from stdin)
"""
import json, sys

nb_path   = sys.argv[1]
cell_id   = sys.argv[2]
src_path  = sys.argv[3] if len(sys.argv) > 3 else None

if src_path == '--stdin':
    new_source = sys.stdin.read()
else:
    with open(src_path) as f:
        new_source = f.read()

with open(nb_path) as f:
    nb = json.load(f)

found = False
for cell in nb['cells']:
    if cell.get('id') == cell_id:
        cell['source'] = new_source
        found = True
        break

if not found:
    print(f"ERROR: cell {cell_id} not found in {nb_path}", file=sys.stderr)
    sys.exit(1)

with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=1)
print(f"Patched cell {cell_id} in {nb_path}")

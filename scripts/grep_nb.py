#!/usr/bin/env python3
"""Grep cell sources in a Jupyter notebook.

Usage:
  python scripts/grep_nb.py <notebook.ipynb> <pattern> [-C <context>] [-n]

Options:
  -C N   show N lines of context before and after each match (default 0)
  -n     show line numbers within the cell source

Prints cell ID, matching line numbers, and optionally surrounding context.
"""
import json, sys, re

def main():
    args = sys.argv[1:]
    context = 0
    show_line_numbers = False
    positional = []
    i = 0
    while i < len(args):
        if args[i] == '-C' and i + 1 < len(args):
            context = int(args[i+1]); i += 2
        elif args[i] == '-n':
            show_line_numbers = True; i += 1
        else:
            positional.append(args[i]); i += 1

    if len(positional) < 2:
        print(__doc__); sys.exit(1)

    nb_path, pattern = positional[0], positional[1]

    with open(nb_path) as f:
        nb = json.load(f)

    found_any = False
    for cell in nb['cells']:
        cid = cell.get('id', '')
        src = ''.join(cell.get('source', []))
        lines = src.split('\n')
        match_indices = [i for i, l in enumerate(lines) if re.search(pattern, l)]
        if not match_indices:
            continue
        found_any = True
        print(f"=== {cid} ===")
        shown = set()
        for mi in match_indices:
            lo = max(0, mi - context)
            hi = min(len(lines) - 1, mi + context)
            if lo > 0 and (shown and min(shown) > lo + 1):
                print('--')
            for li in range(lo, hi + 1):
                if li not in shown:
                    prefix = f"{li:4d}: " if show_line_numbers else ""
                    marker = ">" if li == mi else " "
                    print(f"{marker}{prefix}{lines[li]}")
                    shown.add(li)
        print()

    if not found_any:
        sys.exit(1)

if __name__ == '__main__':
    main()

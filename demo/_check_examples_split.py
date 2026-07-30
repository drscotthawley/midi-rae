"""Are the demo's example songs held out, or from training?

Reads POP909_split.json -- the split dumped ON THE MACHINE THAT TRAINED THE MODEL.
Do NOT recompute the split locally: it is a seeded shuffle of glob() output, and
glob() ordering is filesystem-dependent, so a local recomputation gives a different
(wrong) answer. That is not hypothetical -- it previously reported 4 of 6 examples
as training songs when the true figure was 6 of 6.

Regenerate the JSON with scratchpad/dump_split.py on the training host if the
dataset or split parameters ever change.

Run: python demo/_check_examples_split.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    sp = HERE / "POP909_split.json"
    if not sp.exists():
        sys.exit(f"missing {sp} — dump it on the training host first")
    d = json.loads(sp.read_text())
    train, val = set(d["train"]), set(d["val"])
    print(f"split from {d['data_dir']}: {d['n_train']} train / {d['n_val']} val "
          f"(seed={d['seed']}, val_fraction={d['val_fraction']})\n")

    examples = sorted(p.name for p in (HERE / "examples").glob("*.png"))
    bad = 0
    for name in examples:
        if name in val:
            where = "val (held out)"
        elif name in train:
            where = "TRAIN  <-- retrain-set leakage"
            bad += 1
        else:
            where = "NOT IN SPLIT (renamed? different dataset?)"
            bad += 1
        print(f"  {name:20s} {where}")

    print(f"\n{len(examples) - bad} of {len(examples)} demo examples are held out.")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

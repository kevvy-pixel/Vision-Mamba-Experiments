"""Create the single fixed patient-level split used by every reviewer experiment."""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

CLASSES = ("dvd_no", "eso_no", "eso_V", "exo_A", "exo_no", "exo_V")
SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    grouped = {}
    for name in CLASSES:
        paths = sorted(p.resolve() for p in (args.data_dir / name).iterdir()
                       if p.is_file() and p.suffix.lower() in SUFFIXES)
        grouped[name] = paths
    rng = random.Random(args.seed)
    rows = []
    for name in CLASSES:
        paths = list(grouped[name]); rng.shuffle(paths)
        n_val = max(1, round(len(paths) * .2)); n_test = max(1, round(len(paths) * .1))
        rows.extend(("val", name, p) for p in paths[:n_val])
        rows.extend(("test", name, p) for p in paths[n_val:n_val+n_test])
        rows.extend(("train", name, p) for p in paths[n_val+n_test:])
    rng.shuffle(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(("split", "class", "patient_id", "path"))
        for split, cls, path in rows:
            w.writerow((split, cls, path.stem, str(path)))
    counts = {s: dict(sorted(Counter(cls for ss, cls, _ in rows if ss == s).items()))
              for s in ("train", "val", "test")}
    summary = {"total": len(rows), "split_seed": args.seed, "patient_level": True,
               "one_image_per_patient": True, "counts": counts,
               "split_sizes": {k: sum(v.values()) for k, v in counts.items()}}
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

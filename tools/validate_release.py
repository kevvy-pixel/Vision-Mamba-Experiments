from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
EXPERIMENTS = REPO / "experiments"
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".ckpt", ".npy", ".pyc"}
SENSITIVE_PATTERNS = [
    re.compile(r"D:\\Source_9gaze_composed", re.IGNORECASE),
    re.compile(r"C:\\Users\\27356", re.IGNORECASE),
    re.compile(r"20\d{6}[a-z]+-[a-z]+", re.IGNORECASE),
]


def main() -> int:
    errors: list[str] = []
    experiment_dirs = sorted(path for path in EXPERIMENTS.iterdir() if path.is_dir())
    for experiment in experiment_dirs:
        for required in ("README.md", "code", "results"):
            if not (experiment / required).exists():
                errors.append(f"{experiment.name}: missing {required}")
        if not any((experiment / "code").rglob("*")):
            errors.append(f"{experiment.name}: code/ is empty")
        if not any((experiment / "results").rglob("*")):
            errors.append(f"{experiment.name}: results/ is empty")

    for path in REPO.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"forbidden binary/cache file: {path.relative_to(REPO)}")
        if path.stat().st_size >= 100 * 1024 * 1024:
            errors.append(f"GitHub 100 MiB limit exceeded: {path.relative_to(REPO)}")
        if path.suffix.lower() == ".json":
            try:
                json.loads(path.read_text(encoding="utf-8-sig"))
            except Exception as exc:
                errors.append(f"invalid JSON {path.relative_to(REPO)}: {exc}")
        if path.name == "split_manifest.csv":
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            if not rows or "path" not in rows[0]:
                errors.append(f"invalid split manifest: {path.relative_to(REPO)}")
            elif any(not row["path"].startswith("sample_") for row in rows):
                errors.append(f"non-deidentified split manifest: {path.relative_to(REPO)}")
        if path.suffix.lower() in {".md", ".txt", ".log", ".csv", ".json", ".py", ".ps1", ".sh"}:
            text = path.read_text(encoding="utf-8-sig", errors="replace")
            for pattern in SENSITIVE_PATTERNS:
                if pattern.search(text):
                    errors.append(f"sensitive path/name pattern in {path.relative_to(REPO)}")
                    break

    if errors:
        print("release validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"release validation passed: {len(experiment_dirs)} experiment groups")
    return 0


if __name__ == "__main__":
    sys.exit(main())


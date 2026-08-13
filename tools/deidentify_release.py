from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
PATH_RE = re.compile(r"(?:[A-Za-z]:\\[^\s,;\"']+|/[A-Za-z0-9_./-]+)")


def sample_id(raw_path: str) -> str:
    normalized = raw_path.replace("\\", "/").strip().lower()
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    suffix = Path(normalized).suffix or ".png"
    return f"sample_{digest}{suffix}"


def deidentify_manifest(path: Path) -> None:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    if "path" not in fieldnames:
        return
    for row in rows:
        row["path"] = sample_id(row["path"])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def deidentify_prediction_csv(path: Path) -> None:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    path_field = next((name for name in fieldnames if name.lower() in {"path", "image_path", "file", "filename"}), None)
    if path_field is None:
        return
    for row in rows:
        row[path_field] = sample_id(row[path_field])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def scrub_text(path: Path) -> None:
    text = path.read_text(encoding="utf-8", errors="replace")
    text = re.sub(r"D:\\Source_9gaze_composed", "<DATASET_ROOT>", text, flags=re.IGNORECASE)
    text = re.sub(r"C:\\Users\\27356", "<USER_HOME>", text, flags=re.IGNORECASE)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    manifests = list(REPO.rglob("split_manifest.csv"))
    for path in manifests:
        deidentify_manifest(path)
    prediction_files = list(REPO.rglob("predictions.csv"))
    for path in prediction_files:
        deidentify_prediction_csv(path)
    text_suffixes = {".log", ".txt", ".json", ".md", ".ps1", ".py", ".sh", ".yml", ".yaml", ".cfg"}
    text_files = [path for path in REPO.rglob("*") if path.is_file() and path.suffix.lower() in text_suffixes]
    for path in text_files:
        scrub_text(path)
    print(
        f"deidentified {len(manifests)} split manifests, "
        f"{len(prediction_files)} prediction tables, and scrubbed {len(text_files)} text files"
    )


if __name__ == "__main__":
    main()


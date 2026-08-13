from __future__ import annotations

import csv
import hashlib
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[2]
SOURCE = WORKSPACE / "outputs" / "vision_mamba_experiments_seed42_20260812"
OUTPUT = Path(__file__).resolve().parents[1] / "artifacts" / "EXCLUDED_BINARY_ARTIFACTS.csv"
EXCLUDED_SUFFIXES = {".pt", ".pth", ".npy", ".zip"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    files = sorted(
        (path for path in SOURCE.rglob("*") if path.is_file() and path.suffix.lower() in EXCLUDED_SUFFIXES),
        key=lambda path: path.as_posix(),
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["relative_path", "suffix", "bytes", "mib", "sha256"])
        for path in files:
            size = path.stat().st_size
            writer.writerow(
                [
                    path.relative_to(SOURCE).as_posix(),
                    path.suffix.lower(),
                    size,
                    f"{size / 1024 / 1024:.2f}",
                    sha256(path),
                ]
            )
    print(f"wrote {len(files)} entries to {OUTPUT}")


if __name__ == "__main__":
    main()


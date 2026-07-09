"""Download the MUCS 2021 Hindi-English code-switching corpus from OpenSLR 104.

Source: https://www.openslr.org/104/   (CC BY-SA 4.0)
The `openslr.trmal.net` mirror is password-free.

Downloads resume on interruption (HTTP Range), so a dropped 7.3 GB transfer
does not start over.
"""

from __future__ import annotations

import argparse
import sys
import tarfile
from pathlib import Path

import requests
from tqdm import tqdm

__all__ = ["RESOURCES", "download", "extract"]

BASE = "https://openslr.trmal.net/resources/104"

RESOURCES: dict[str, dict] = {
    "train": {"name": "Hindi-English_train.tar.gz", "approx_bytes": 7_300_000_000},
    "test": {"name": "Hindi-English_test.tar.gz", "approx_bytes": 443_000_000},
}

_CHUNK = 1 << 20  # 1 MiB


def download(split: str, dest_dir: Path, *, timeout: int = 60) -> Path:
    """Fetch one split's tarball with resume support. Returns the local path."""
    if split not in RESOURCES:
        raise ValueError(f"unknown split {split!r}; choose from {sorted(RESOURCES)}")

    name = RESOURCES[split]["name"]
    url = f"{BASE}/{name}"
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / name

    have = out.stat().st_size if out.exists() else 0
    headers = {"Range": f"bytes={have}-"} if have else {}

    with requests.get(url, stream=True, headers=headers, timeout=timeout) as r:
        if have and r.status_code == 416:
            print(f"[download] {name}: already complete ({have:,} bytes)")
            return out
        r.raise_for_status()

        remaining = int(r.headers.get("Content-Length", 0))
        total = have + remaining
        mode = "ab" if have and r.status_code == 206 else "wb"
        if mode == "wb":
            have = 0

        with out.open(mode) as fh, tqdm(
            total=total or None,
            initial=have,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=name,
        ) as bar:
            for chunk in r.iter_content(chunk_size=_CHUNK):
                fh.write(chunk)
                bar.update(len(chunk))

    print(f"[download] {name}: {out.stat().st_size:,} bytes -> {out}")
    return out


def extract(tarball: Path, dest_dir: Path) -> Path:
    """Extract a tarball, skipping if the destination already has content."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    marker = dest_dir / f".{tarball.name}.extracted"
    if marker.exists():
        print(f"[extract] {tarball.name}: already extracted")
        return dest_dir

    print(f"[extract] {tarball.name} -> {dest_dir}")
    with tarfile.open(tarball, "r:gz") as tf:
        members = tf.getmembers()
        for m in tqdm(members, desc=f"extract {tarball.name}", unit="file"):
            tf.extract(m, dest_dir)  # noqa: S202 - trusted OpenSLR archive

    marker.touch()
    return dest_dir


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", nargs="+", default=["train", "test"], choices=list(RESOURCES))
    ap.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--extract-dir", type=Path, default=Path("data/mucs"))
    ap.add_argument("--no-extract", action="store_true")
    args = ap.parse_args(argv)

    for split in args.splits:
        tarball = download(split, args.raw_dir)
        if not args.no_extract:
            extract(tarball, args.extract_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

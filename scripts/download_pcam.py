#!/usr/bin/env python3
"""
PCam downloader/organizer.

Primary source (default): torchvision.datasets.PCAM(download=True)
- Requires torchvision>=0.19 and 'gdown' during the first download.
- We only use TV to download/cache the official files, then we place/symlink them
  into your repo at:  src/data/pcam/

Fallback source: --source http --urls urls.json
- 'urls.json' should map canonical filenames to direct HTTP(S) URLs.
  Example 'urls.json':
  {
    "camelyonpatch_level_2_split_train_x.h5.gz": "https://.../camelyonpatch_level_2_split_train_x.h5.gz",
    "camelyonpatch_level_2_split_train_y.h5.gz": "https://.../camelyonpatch_level_2_split_train_y.h5.gz",
    "camelyonpatch_level_2_split_valid_x.h5.gz": "https://.../camelyonpatch_level_2_split_valid_x.h5.gz",
    "camelyonpatch_level_2_split_valid_y.h5.gz": "https://.../camelyonpatch_level_2_split_valid_y.h5.gz",
    "camelyonpatch_level_2_split_test_x.h5.gz":  "https://.../camelyonpatch_level_2_split_test_x.h5.gz",
    "camelyonpatch_level_2_split_test_y.h5.gz":  "https://.../camelyonpatch_level_2_split_test_y.h5.gz",
    "camelyonpatch_level_2_split_train_meta.csv":"https://.../camelyonpatch_level_2_split_train_meta.csv",
    "camelyonpatch_level_2_split_valid_meta.csv":"https://.../camelyonpatch_level_2_split_valid_meta.csv",
    "camelyonpatch_level_2_split_test_meta.csv": "https://.../camelyonpatch_level_2_split_test_meta.csv",
    "camelyonpatch_level_2_split_train_mask.h5": "https://.../camelyonpatch_level_2_split_train_mask.h5"
  }
- .gz files are auto-decompressed to .h5 in the output directory.

Usage:
  python -m scripts.download_pcam --out src/data/pcam
  python -m scripts.download_pcam --out src/data/pcam --copy
  python -m scripts.download_pcam --source http --urls urls.json --out src/data/pcam

"""

from __future__ import annotations
import argparse
import gzip
import io
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Iterable

REQUIRED_FILES: List[str] = [
    # HDF5
    "camelyonpatch_level_2_split_train_x.h5",
    "camelyonpatch_level_2_split_train_y.h5",
    "camelyonpatch_level_2_split_valid_x.h5",
    "camelyonpatch_level_2_split_valid_y.h5",
    "camelyonpatch_level_2_split_test_x.h5",
    "camelyonpatch_level_2_split_test_y.h5",
]

# Optional (not needed to train/eval with current code, but nice to have)
OPTIONAL_FILES: List[str] = [
    "camelyonpatch_level_2_split_train_mask.h5",
    # CSV metadata
    "camelyonpatch_level_2_split_train_meta.csv",
    "camelyonpatch_level_2_split_valid_meta.csv",
    "camelyonpatch_level_2_split_test_meta.csv",
]

CANONICAL_BASENAMES = set(REQUIRED_FILES + OPTIONAL_FILES)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def maybe_symlink_or_copy(src: Path, dst: Path, do_copy: bool = False) -> None:
    if dst.exists():
        return
    if do_copy:
        shutil.copy2(src, dst)
    else:
        try:
            dst.symlink_to(src.resolve())
        except OSError:
            # Filesystems w/o symlink support (e.g., some shared FS) → fallback to copy
            shutil.copy2(src, dst)


def find_downloaded_files(root: Path) -> Dict[str, Path]:
    """
    Search under 'root' for any of the canonical basenames, return {basename: fullpath}.
    """
    found: Dict[str, Path] = {}
    for p in root.rglob("*"):
        if p.is_file() and p.name in CANONICAL_BASENAMES:
            found[p.name] = p
    return found


def have_all_required(found: Dict[str, Path]) -> bool:
    return all(name in found for name in REQUIRED_FILES)


def decompress_gz_to(src_gz: Path, dst_h5: Path) -> None:
    with gzip.open(src_gz, "rb") as f_in, open(dst_h5, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)


def http_fetch(url: str, out: Path) -> None:
    import requests
    from tqdm import tqdm

    out.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        pbar = tqdm(total=total, unit="B", unit_scale=True, desc=out.name)
        with open(out, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):  # 1 MiB
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        pbar.close()


def from_torchvision(cache_root: Path) -> Dict[str, Path]:
    """
    Use torchvision.datasets.PCAM to download/cache files, then return the file map.
    """
    try:
        import torchvision  # noqa: F401
        from torchvision.datasets import PCAM
    except Exception as e:
        print(
            "torchvision import failed. Please `pip install torchvision>=0.19 gdown`.",
            file=sys.stderr,
        )
        raise

    # Some envs require gdown for the first fetch.
    try:
        import gdown  # noqa: F401
    except Exception:
        print(
            "Note: installing `gdown` may be required for the initial download.",
            file=sys.stderr,
        )

    # PCAM(download=True) typically pulls *all* files once.
    # Create datasets for the three splits to be safe (idempotent after first call).
    for split in ("train", "val", "test"):
        _ = PCAM(root=str(cache_root), split=split, download=True)

    # Gather files from whatever folder torchvision used (usually <root>/pcam/)
    return find_downloaded_files(cache_root)


def from_http(urls_json: Path, work_dir: Path) -> Dict[str, Path]:
    """
    Download from explicit URLs (you provide), decompress .gz if needed, return file map.
    """
    with open(urls_json, "r") as f:
        mapping: Dict[str, str] = json.load(f)

    out_map: Dict[str, Path] = {}
    tmp_dl = work_dir / "http_tmp"
    ensure_dir(tmp_dl)

    # First fetch everything we have URLs for
    for name, url in mapping.items():
        dst = tmp_dl / name
        if not dst.exists():
            print(f"Downloading {name} ...")
            http_fetch(url, dst)
        out_map[name] = dst

    # Decompress any .gz into .h5 alongside and replace the map entries
    decompressed: Dict[str, Path] = {}
    for name, path in list(out_map.items()):
        if name.endswith(".gz"):
            target_name = name[:-3]  # strip .gz
            target_path = tmp_dl / target_name
            if not target_path.exists():
                print(f"Decompressing {name} -> {target_name}")
                decompress_gz_to(path, target_path)
            decompressed[target_name] = target_path
    out_map.update(decompressed)

    # Keep only the canonical basenames we care about
    final_map: Dict[str, Path] = {}
    for name in CANONICAL_BASENAMES:
        # Prefer decompressed (.h5) over .gz
        if name in out_map:
            final_map[name] = out_map[name]
        elif (name + ".gz") in out_map:
            # shouldn't happen after decompression, but just in case
            final_map[name] = out_map[name + ".gz"]
    return final_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("src/data/pcam"),
        help="Destination directory (your code expects these filenames here).",
    )
    ap.add_argument(
        "--cache",
        type=Path,
        default=Path("~/.cache/pcam").expanduser(),
        help="Cache dir (used by torchvision/http downloads).",
    )
    ap.add_argument(
        "--copy", action="store_true", help="Copy files instead of creating symlinks."
    )
    ap.add_argument(
        "--source",
        choices=["torchvision", "http"],
        default="torchvision",
        help="Where to download PCam from.",
    )
    ap.add_argument(
        "--urls", type=Path, default=None, help="JSON mapping (only for --source http)."
    )
    args = ap.parse_args()

    ensure_dir(args.out)
    ensure_dir(args.cache)

    file_map: Dict[str, Path] = {}
    if args.source == "torchvision":
        print(
            "→ Downloading with torchvision.datasets.PCAM (this may pull all splits at once)."
        )
        file_map = from_torchvision(args.cache)
    else:
        if args.urls is None:
            print(
                "Please provide --urls <urls.json> for --source http", file=sys.stderr
            )
            sys.exit(2)
        print("→ Downloading with explicit HTTP URLs from:", args.urls)
        file_map = from_http(args.urls, args.cache)

    # Validate & place files
    if not file_map:
        print("No files found/downloaded. Abort.", file=sys.stderr)
        sys.exit(1)

    have_optional = []
    for req in REQUIRED_FILES:
        if req not in file_map:
            print(f"Missing required file after download: {req}", file=sys.stderr)
            # Keep going to list them all, then exit.
    missing = [req for req in REQUIRED_FILES if req not in file_map]
    if missing:
        print(
            "\nSome required files are missing:\n  - " + "\n  - ".join(missing),
            file=sys.stderr,
        )
        print(
            "If you used --source torchvision, please ensure torchvision>=0.19 and gdown are available.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Place required files
    for name in REQUIRED_FILES:
        src = file_map[name]
        dst = args.out / name
        maybe_symlink_or_copy(src, dst, do_copy=args.copy)

    # Place optional files when present
    for name in OPTIONAL_FILES:
        if name in file_map:
            src = file_map[name]
            dst = args.out / name
            maybe_symlink_or_copy(src, dst, do_copy=args.copy)
            have_optional.append(name)

    print(f"\nPCam is ready in: {args.out.resolve()}")
    print("Placed (or already present):")
    for name in REQUIRED_FILES:
        print("  -", name)
    for name in have_optional:
        print("  -", name, "(optional)")


if __name__ == "__main__":
    main()

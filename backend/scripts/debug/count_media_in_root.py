from __future__ import annotations

import argparse
import os
from pathlib import Path


VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".wmv", ".m4v", ".webm", ".3gp"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic"}

EXCLUDE_TOP_DEFAULT = {
    "_faces",
    "_no_faces",
    "_duplicates",
    "_animals",
    "_non_media",
    "_broken_media",
}


def main() -> int:
    ap = argparse.ArgumentParser(description="Count images/videos/other files under a root directory (excluding service folders).")
    ap.add_argument("--root", required=True, help="Root directory, e.g. C:\\tmp\\Photo")
    ap.add_argument("--exclude-top", action="append", default=[], help="Top-level folder name to exclude (repeatable)")
    args = ap.parse_args()

    root = os.path.abspath(str(args.root))
    exclude = set(EXCLUDE_TOP_DEFAULT)
    for x in args.exclude_top or []:
        s = str(x or "").strip()
        if s:
            exclude.add(s)

    total = images = videos = other = 0
    for dirpath, dirnames, filenames in os.walk(root):
        rel = os.path.relpath(dirpath, root)
        top = rel.split(os.sep, 1)[0] if rel not in (".", "") else ""
        if top and top in exclude:
            dirnames[:] = []
            continue
        for fn in filenames:
            total += 1
            ext = Path(fn).suffix.lower()
            if ext in IMAGE_EXT:
                images += 1
            elif ext in VIDEO_EXT:
                videos += 1
            else:
                other += 1

    print(f"root={root}")
    print(f"total={total}")
    print(f"images={images}")
    print(f"videos={videos}")
    print(f"other={other}")
    print(f"non_images(total - images)={total - images}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


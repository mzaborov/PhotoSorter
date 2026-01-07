from __future__ import annotations

import argparse
from typing import Any, Optional

from yadisk_client import get_disk


def _get(item: Any, key: str) -> Optional[Any]:
    if isinstance(item, dict):
        return item.get(key)
    return getattr(item, key, None)


def main() -> int:
    parser = argparse.ArgumentParser(description="List first-level folders on Yandex.Disk")
    parser.add_argument(
        "--path",
        default="/Фото",
        help="Путь на Яндекс.Диске, например /Фото (по умолчанию: /Фото)",
    )
    args = parser.parse_args()

    disk = get_disk()
    base = args.path

    for item in disk.listdir(base):
        if _get(item, "type") != "dir":
            continue
        name = _get(item, "name") or ""
        path = _get(item, "path") or ""
        print(f"{name}\t{path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())



















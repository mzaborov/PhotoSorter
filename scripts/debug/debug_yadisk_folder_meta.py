from __future__ import annotations

import argparse
from typing import Any, Optional

from yadisk_client import get_disk


def normalize(path: str) -> str:
    p = (path or "").strip()
    if p.startswith("disk:"):
        p = p[len("disk:") :]
    if not p.startswith("/"):
        p = "/" + p
    return p


def as_dict(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "to_json"):
        try:
            return obj.to_json()  # type: ignore[attr-defined]
        except Exception:
            pass
    # fallback: try public attrs
    out: dict[str, Any] = {}
    for k in dir(obj):
        if k.startswith("_"):
            continue
        try:
            v = getattr(obj, k)
        except Exception:
            continue
        if callable(v):
            continue
        out[k] = v
    return out


def pick(d: dict[str, Any], key: str) -> Optional[Any]:
    return d.get(key)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="disk:/Фото/Путешествия")
    args = parser.parse_args()

    disk = get_disk()
    p = normalize(args.path)

    # 1) Метаданные (как есть)
    meta1 = disk.get_meta(p)
    d1 = as_dict(meta1)

    # 2) Метаданные с embedded, но без items (если поддерживается)
    meta2 = disk.get_meta(p, limit=0)
    d2 = as_dict(meta2)

    def summarize(label: str, d: dict[str, Any]) -> None:
        embedded = d.get("embedded") if isinstance(d.get("embedded"), dict) else None
        print(f"\n== {label} ==")
        print("keys:", ", ".join(sorted(d.keys())[:60]), ("..." if len(d.keys()) > 60 else ""))
        print("type:", pick(d, "type"))
        print("name:", pick(d, "name"))
        print("path:", pick(d, "path"))
        print("size:", pick(d, "size"))
        if embedded is not None:
            print("embedded.keys:", ", ".join(sorted(embedded.keys())))
            print("embedded.total:", embedded.get("total"))
            items = embedded.get("items")
            if isinstance(items, list):
                print("embedded.items.len:", len(items))

        # Некоторые поля, которые иногда встречаются
        for k in ("file", "public_key", "md5", "sha256", "media_type", "mime_type", "created", "modified"):
            if k in d:
                print(f"{k}:", d.get(k))

    summarize("get_meta(default)", d1)
    summarize("get_meta(limit=0)", d2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())









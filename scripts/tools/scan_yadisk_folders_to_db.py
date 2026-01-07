from __future__ import annotations

import argparse
import re
from typing import Any, Optional

from DB.db import get_connection, init_db
from yadisk_client import get_disk


def _get(item: Any, key: str) -> Optional[Any]:
    if isinstance(item, dict):
        return item.get(key)
    return getattr(item, key, None)


_RU_TO_LAT = {
    "а": "a",
    "б": "b",
    "в": "v",
    "г": "g",
    "д": "d",
    "е": "e",
    "ё": "e",
    "ж": "zh",
    "з": "z",
    "и": "i",
    "й": "y",
    "к": "k",
    "л": "l",
    "м": "m",
    "н": "n",
    "о": "o",
    "п": "p",
    "р": "r",
    "с": "s",
    "т": "t",
    "у": "u",
    "ф": "f",
    "х": "h",
    "ц": "ts",
    "ч": "ch",
    "ш": "sh",
    "щ": "sch",
    "ъ": "",
    "ы": "y",
    "ь": "",
    "э": "e",
    "ю": "yu",
    "я": "ya",
}


def slugify_ru(text: str) -> str:
    s = text.strip().lower()
    out: list[str] = []
    for ch in s:
        if ch in _RU_TO_LAT:
            out.append(_RU_TO_LAT[ch])
        elif ch.isalnum():
            out.append(ch)
        else:
            out.append("_")
    slug = "".join(out)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "folder"


SORT_ORDER_BY_NAME: dict[str, int] = {
    "Агата": 2,
    "Санек": 3,
    "Нюся": 4,
    "Темка": 5,
    "Дети вместе": 6,
    "Миша и Аня": 7,
    "Бабушки и Дедушки": 8,
    "Семья": 9,
    "Котэ и сцобако": 10,
    "Другие люди": 11,
}


def upsert_folder(
    *,
    code_prefix: str,
    name: str,
    path: str,
    location: str = "yadisk",
    role: str = "target",
    sort_order: Optional[int],
) -> tuple[str, str]:
    """
    Возвращает (action, code), где action: inserted|updated|skipped.
    Upsert делаем по path (если запись уже есть).
    """
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT id, code FROM folders WHERE path = ? LIMIT 1", (path,))
        row = cur.fetchone()
        if row:
            folder_id = row["id"]
            code = row["code"]
            cur.execute(
                """
                UPDATE folders
                SET name = ?, location = ?, role = ?, sort_order = ?
                WHERE id = ?
                """,
                (name, location, role, sort_order, folder_id),
            )
            conn.commit()
            return ("updated", code)

        # вставка новой записи
        base_code = f"{code_prefix}{slugify_ru(name)}"
        code = base_code
        suffix = 2
        while True:
            cur.execute("SELECT 1 FROM folders WHERE code = ? LIMIT 1", (code,))
            if not cur.fetchone():
                break
            code = f"{base_code}_{suffix}"
            suffix += 1

        cur.execute(
            """
            INSERT INTO folders(code, path, name, location, role, sort_order, content_rule, priority_after_code)
            VALUES (?, ?, ?, ?, ?, ?, NULL, NULL)
            """,
            (code, path, name, location, role, sort_order),
        )
        conn.commit()
        return ("inserted", code)
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan first-level folders in /Фото and upsert into SQLite")
    parser.add_argument("--path", default="/Фото", help="Путь на Яндекс.Диске (по умолчанию: /Фото)")
    parser.add_argument(
        "--exclude-name",
        action="append",
        default=[],
        help="Исключить папку по имени (можно указывать несколько раз)",
    )
    parser.add_argument(
        "--code-prefix",
        default="yd_photo_",
        help="Префикс для кода папки при вставке (по умолчанию: yd_photo_)",
    )
    args = parser.parse_args()

    # гарантируем схему
    init_db()

    excluded = set(args.exclude_name)
    disk = get_disk()

    inserted = updated = skipped = 0

    for item in disk.listdir(args.path):
        if _get(item, "type") != "dir":
            continue
        name = str(_get(item, "name") or "")
        path = str(_get(item, "path") or "")

        if not name or not path:
            continue

        if name in excluded:
            skipped += 1
            continue

        sort_order = SORT_ORDER_BY_NAME.get(name)
        action, code = upsert_folder(
            code_prefix=args.code_prefix,
            name=name,
            path=path,
            sort_order=sort_order,
        )
        if action == "inserted":
            inserted += 1
        elif action == "updated":
            updated += 1
        else:
            skipped += 1

        print(f"{action}\t{code}\t{name}\t{path}\t{sort_order if sort_order is not None else ''}")

    print(f"\nRESULT: inserted={inserted}, updated={updated}, skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



















#!/usr/bin/env python3
"""
Удаляет дубликаты rect на кадрах видео (frame_idx 1..3) для указанного файла.

При перепривязке персоны могли сохраняться и старый rect (из кластера), и новый (ручной).
Скрипт оставляет по одному rect на каждое уникальное лицо (по bbox), приоритет — manual_person_id.

Использование:
  python backend/scripts/tools/cleanup_video_rect_duplicates.py "local:C:\tmp\Photo\_faces\VID-20231119-WA0006.mp4"
  python backend/scripts/tools/cleanup_video_rect_duplicates.py --dry-run "local:..."
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=_PROJECT_ROOT / "secrets.env", override=False)
    load_dotenv(dotenv_path=_PROJECT_ROOT / ".env", override=False)
except Exception:
    pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Очистка дубликатов rect на кадрах видео")
    parser.add_argument("path", help="Путь к файлу (local:... или disk:...)")
    parser.add_argument("--dry-run", action="store_true", help="Только показать, что будет удалено")
    args = parser.parse_args()

    from backend.common.db import get_connection

    conn = get_connection()
    conn.row_factory = lambda c, r: dict(zip([col[0] for col in c.description], r))
    cur = conn.cursor()

    # file_id по path
    path = (args.path or "").strip()
    cur.execute("SELECT id FROM files WHERE path = ? LIMIT 1", (path,))
    row = cur.fetchone()
    if not row:
        print(f"Файл не найден: {path}")
        return 1
    file_id = row["id"]

    # Все rect с frame_idx 1..3 для этого файла
    cur.execute(
        """
        SELECT pr.id, pr.frame_idx, pr.bbox_x, pr.bbox_y, pr.bbox_w, pr.bbox_h,
               pr.manual_person_id, pr.cluster_id, p.name AS person_name
        FROM photo_rectangles pr
        LEFT JOIN persons p ON p.id = pr.manual_person_id
        WHERE pr.file_id = ? AND pr.frame_idx IN (1, 2, 3) AND COALESCE(pr.ignore_flag, 0) = 0
        ORDER BY pr.frame_idx, pr.id
        """,
        (file_id,),
    )
    rows = cur.fetchall()

    if not rows:
        print("Нет rect для этого видео.")
        return 0

    # Группируем по (frame_idx, bbox) — дубликаты имеют одинаковые координаты
    def bbox_key(r):
        return (r["frame_idx"], r["bbox_x"], r["bbox_y"], r["bbox_w"], r["bbox_h"])

    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        k = bbox_key(r)
        if k not in groups:
            groups[k] = []
        groups[k].append(r)

    to_delete: list[int] = []
    for k, grp in groups.items():
        if len(grp) <= 1:
            continue
        # Сортируем: manual_person_id приоритетнее
        grp.sort(key=lambda x: (0 if x["manual_person_id"] else 1, x["id"]))
        keep = grp[0]
        for r in grp[1:]:
            to_delete.append(r["id"])
            print(f"  Удалить rect id={r['id']} frame={r['frame_idx']} "
                  f"(дубль, оставляем id={keep['id']})")

    if not to_delete:
        print("Дубликатов не найдено.")
        return 0

    print(f"\nБудет удалено rect: {len(to_delete)}")
    if args.dry_run:
        return 0

    for rid in to_delete:
        cur.execute("DELETE FROM photo_rectangles WHERE id = ?", (rid,))
    conn.commit()
    print(f"Удалено {len(to_delete)} rect.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Восстанавливает manual_person_id для лиц в ручных кластерах архива.

Берёт person_id из кластера и записывает в manual_person_id (где NULL).
Применимо после backfill, когда привязки были верифицированы.

Использование:
  python backend/scripts/tools/restore_manual_person_id_from_cluster.py
  python backend/scripts/tools/restore_manual_person_id_from_cluster.py --dry-run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from dotenv import load_dotenv  # type: ignore[import-not-found]
    load_dotenv(dotenv_path=str(_PROJECT_ROOT / "secrets.env"), override=False)
    load_dotenv(dotenv_path=str(_PROJECT_ROOT / ".env"), override=False)
except Exception:
    pass


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Восстановить manual_person_id из person_id кластера"
    )
    parser.add_argument("--dry-run", action="store_true", help="Только показать, что будет обновлено")
    args = parser.parse_args()

    from backend.common.db import get_connection

    conn = get_connection()
    cur = conn.cursor()

    if args.dry_run:
        cur.execute("""
            SELECT COUNT(*) AS cnt FROM photo_rectangles pr
            JOIN face_clusters fc ON fc.id = pr.cluster_id
            WHERE fc.archive_scope = 'archive' AND COALESCE(fc.method, '') = 'manual'
              AND pr.manual_person_id IS NULL AND fc.person_id IS NOT NULL
        """)
        cnt = cur.fetchone()[0]
        print(f"Будет обновлено записей: {cnt}")
        conn.close()
        return 0

    cur.execute("""
        UPDATE photo_rectangles
        SET manual_person_id = (
            SELECT fc.person_id FROM face_clusters fc
            WHERE fc.id = photo_rectangles.cluster_id
              AND fc.archive_scope = 'archive' AND COALESCE(fc.method, '') = 'manual'
        )
        WHERE cluster_id IN (
            SELECT id FROM face_clusters
            WHERE archive_scope = 'archive' AND COALESCE(method, '') = 'manual'
        )
        AND manual_person_id IS NULL
    """)
    updated = cur.rowcount
    conn.commit()
    conn.close()

    print(f"Обновлено записей: {updated}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Миграция: перенос данных из face_cluster_members в photo_rectangles.cluster_id, удаление таблицы face_cluster_members.

Перед запуском:
1. Бекап: python backend/scripts/tools/backup_database.py
2. Проверка: python backend/scripts/debug/verify_backup_integrity.py <путь_к_бекапу.db>
"""
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

from backend.common.db import get_connection, DB_PATH


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Перенести face_cluster_members -> photo_rectangles.cluster_id, DROP face_cluster_members")
    ap.add_argument("--dry-run", action="store_true", help="Только показать, что будет выполнено")
    args = ap.parse_args()

    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='face_cluster_members'")
        if not cur.fetchone():
            print("Таблица face_cluster_members уже отсутствует.")
            cur.execute("PRAGMA table_info(photo_rectangles)")
            cols = [r[1] for r in cur.fetchall()]
            if "cluster_id" not in cols:
                if not args.dry_run:
                    cur.execute("ALTER TABLE photo_rectangles ADD COLUMN cluster_id INTEGER REFERENCES face_clusters(id)")
                    conn.commit()
                    print("Добавлена колонка photo_rectangles.cluster_id.")
                else:
                    print("[DRY RUN] Будет добавлена колонка photo_rectangles.cluster_id.")
            return 0

        cur.execute("PRAGMA table_info(photo_rectangles)")
        cols = [r[1] for r in cur.fetchall()]
        if "cluster_id" not in cols:
            if args.dry_run:
                print("[DRY RUN] Будет добавлена колонка photo_rectangles.cluster_id.")
            else:
                cur.execute("ALTER TABLE photo_rectangles ADD COLUMN cluster_id INTEGER REFERENCES face_clusters(id)")
                conn.commit()
                print("Добавлена колонка photo_rectangles.cluster_id.")

        cur.execute("SELECT COUNT(*) FROM face_cluster_members")
        n = cur.fetchone()[0]
        if args.dry_run:
            print(f"[DRY RUN] Будет перенесено {n} записей из face_cluster_members в photo_rectangles.cluster_id.")
            print("[DRY RUN] Будет выполнено: DROP TABLE face_cluster_members;")
            return 0

        cur.execute("""
            UPDATE photo_rectangles
            SET cluster_id = (
                SELECT fcm.cluster_id FROM face_cluster_members fcm
                WHERE fcm.rectangle_id = photo_rectangles.id
                LIMIT 1
            )
            WHERE id IN (SELECT rectangle_id FROM face_cluster_members)
        """)
        updated = cur.rowcount
        cur.execute("DROP TABLE face_cluster_members")
        conn.commit()
        print(f"Перенесено записей: {updated}, таблица face_cluster_members удалена.")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

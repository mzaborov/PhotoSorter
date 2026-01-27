#!/usr/bin/env python3
"""
Миграция: перенос person_rectangle_manual_assignments → photo_rectangles.manual_person_id, удаление таблицы.

При конфликте (у прямоугольника уже есть cluster_id и есть ручная привязка): ручная перекрывает — SET manual_person_id = ?, cluster_id = NULL.
При нескольких записях на один rectangle_id берётся одна (MIN(id)).

Перед запуском: бекап и проверка целостности.
"""
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

from backend.common.db import get_connection


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Перенести person_rectangle_manual_assignments -> photo_rectangles.manual_person_id, DROP таблицы")
    ap.add_argument("--dry-run", action="store_true", help="Только показать, что будет выполнено")
    args = ap.parse_args()

    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='person_rectangle_manual_assignments'")
        if not cur.fetchone():
            print("Таблица person_rectangle_manual_assignments уже отсутствует.")
            return 0

        cur.execute("PRAGMA table_info(photo_rectangles)")
        cols = [r[1] for r in cur.fetchall()]
        if "manual_person_id" not in cols:
            if args.dry_run:
                print("[DRY RUN] Будет добавлена колонка photo_rectangles.manual_person_id")
            else:
                cur.execute("ALTER TABLE photo_rectangles ADD COLUMN manual_person_id INTEGER REFERENCES persons(id)")
                conn.commit()
                print("Добавлена колонка photo_rectangles.manual_person_id")

        cur.execute("SELECT COUNT(*) FROM person_rectangle_manual_assignments")
        n = cur.fetchone()[0]
        if args.dry_run:
            print(f"[DRY RUN] Будет перенесено до {n} записей в photo_rectangles.manual_person_id (ручная перекрывает cluster_id)")
            print("[DRY RUN] Будет выполнено: DROP TABLE person_rectangle_manual_assignments;")
            return 0

        # Одна запись на rectangle_id: берём по MIN(id)
        cur.execute("""
            UPDATE photo_rectangles
            SET manual_person_id = (
                SELECT prma.person_id
                FROM person_rectangle_manual_assignments prma
                WHERE prma.rectangle_id = photo_rectangles.id
                ORDER BY prma.id
                LIMIT 1
            ),
            cluster_id = NULL
            WHERE id IN (SELECT rectangle_id FROM person_rectangle_manual_assignments)
        """)
        updated = cur.rowcount
        cur.execute("DROP TABLE person_rectangle_manual_assignments")
        conn.commit()
        print(f"Перенесено записей: {updated}, таблица person_rectangle_manual_assignments удалена.")

        # CHECK: у прямоугольника только один способ привязки к персоне (SQLite 3.37+)
        try:
            cur.execute(
                "ALTER TABLE photo_rectangles ADD CONSTRAINT chk_rect_one_person_source "
                "CHECK (cluster_id IS NULL OR manual_person_id IS NULL)"
            )
            conn.commit()
            print("Добавлен CHECK (cluster_id IS NULL OR manual_person_id IS NULL).")
        except Exception as e:
            conn.rollback()
            err = str(e).lower()
            if "syntax error" in err or "constraint" in err or "near" in err or "operational" in err:
                print("CHECK не добавлен (требуется SQLite 3.37+ для ADD CONSTRAINT). Логика в коде при записи соблюдает правило.")
            else:
                raise
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

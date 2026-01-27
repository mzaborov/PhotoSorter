#!/usr/bin/env python3
"""
Добавляет CHECK (cluster_id IS NULL OR manual_person_id IS NULL) в photo_rectangles.

В SQLite нет ALTER TABLE ADD CONSTRAINT — ограничение добавляется только
пересозданием таблицы:
  1) новая таблица с CHECK в CREATE TABLE;
  2) копирование данных;
  3) сравнение числа строк (при несовпадении — откат);
  4) DROP старой таблицы;
  5) RENAME новой, пересоздание индексов.

Подключение — то же, что у migrate_person_rectangle_manual_to_photo_rectangles.py.

Запуск: python backend/scripts/migration/add_check_photo_rectangles_one_person_source.py
"""
import re
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

from backend.common.db import get_connection

CONSTRAINT_BODY = "CONSTRAINT chk_rect_one_person_source CHECK (cluster_id IS NULL OR manual_person_id IS NULL)"


def main() -> int:
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='photo_rectangles'"
        )
        row = cur.fetchone()
        if not row or not row[0]:
            print("Таблица photo_rectangles не найдена.")
            return 1
        create_sql = row[0]
        if "chk_rect_one_person_source" in create_sql:
            print("CHECK chk_rect_one_person_source уже присутствует в таблице.")
            return 0

        # Сохраняем определения индексов на photo_rectangles
        cur.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='index' AND tbl_name='photo_rectangles' AND sql IS NOT NULL"
        )
        indexes = [(r[0], r[1]) for r in cur.fetchall()]
        # Исключаем авто-индекс для PRIMARY KEY
        indexes = [(n, s) for n, s in indexes if not n.startswith("sqlite_")]

        # Вставляем CHECK перед закрывающей скобкой таблицы (sqlite_master может хранить ")" или ");")
        idx = create_sql.rfind(");")
        if idx == -1:
            idx = create_sql.rfind(")")
        if idx == -1:
            print("Не удалось найти закрывающую скобку в CREATE TABLE.")
            return 1
        tail = create_sql[idx:]
        head = create_sql[:idx].rstrip()
        # Не добавлять лишнюю запятую, если тело уже заканчивается на запятую
        need_comma = not head.endswith(",")
        check_clause = (", " if need_comma else " ") + CONSTRAINT_BODY
        new_create = head + check_clause + tail

        # Меняем имя таблицы на временное: CREATE TABLE ["]photo_rectangles["] (
        new_create = re.sub(
            r'(CREATE\s+TABLE\s+)(["]?)(photo_rectangles)\2(\s*\()',
            r'\1\2photo_rectangles_new\2\4',
            new_create,
            count=1,
            flags=re.IGNORECASE,
        )
        if "photo_rectangles_new" not in new_create:
            print("Не удалось подставить имя photo_rectangles_new в CREATE TABLE.")
            return 1

        conn.execute("PRAGMA foreign_keys=OFF")
        cur.execute("BEGIN")
        try:
            # 1) Новая таблица с CHECK
            cur.execute("DROP TABLE IF EXISTS photo_rectangles_new")
            cur.execute(new_create)
            # 2) Копирование данных
            cur.execute(
                "INSERT INTO photo_rectangles_new SELECT * FROM photo_rectangles"
            )
            # 3) Сравнение: число строк должно совпадать
            cur.execute("SELECT COUNT(*) FROM photo_rectangles")
            old_count = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM photo_rectangles_new")
            new_count = cur.fetchone()[0]
            if old_count != new_count:
                cur.execute("ROLLBACK")
                conn.execute("PRAGMA foreign_keys=ON")
                print(
                    f"Ошибка: после копирования число строк не совпадает — "
                    f"старая: {old_count}, новая: {new_count}. Откат."
                )
                return 1
            print(f"Сравнение: {old_count} строк в обеих таблицах, ок.")
            # 4) Удалить старую
            cur.execute("DROP TABLE photo_rectangles")
            # 5) Переименовать новую и восстановить индексы
            cur.execute("ALTER TABLE photo_rectangles_new RENAME TO photo_rectangles")
            for name, sql in indexes:
                if sql:
                    try:
                        cur.execute(sql)
                    except Exception as e:
                        print(f"Предупреждение: индекс {name} не пересоздан: {e}")
            cur.execute("COMMIT")
        except Exception as e:
            cur.execute("ROLLBACK")
            try:
                print("Фрагмент SQL, вызвавший ошибку (последние 800 символов):")
                print(new_create[-800:] if len(new_create) > 800 else new_create)
            except NameError:
                pass
            raise
        conn.execute("PRAGMA foreign_keys=ON")

        conn.commit()
        print("CHECK (cluster_id IS NULL OR manual_person_id IS NULL) добавлен (таблица пересоздана).")
    except Exception as e:
        conn.rollback()
        try:
            conn.execute("PRAGMA foreign_keys=ON")
        except Exception:
            pass
        print("Ошибка:", e)
        return 1
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

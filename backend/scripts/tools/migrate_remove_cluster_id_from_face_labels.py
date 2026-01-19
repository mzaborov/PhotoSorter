#!/usr/bin/env python3
"""
Миграция: удаляет колонку cluster_id из таблицы face_labels.

Это часть рефакторинга для устранения избыточности и рассинхронизации.
После миграции кластер определяется через JOIN с face_cluster_members.

ВАЖНО: Перед запуском миграции нужно обновить ВСЕ места в коде,
которые используют cluster_id из face_labels. Иначе после миграции
код перестанет работать.

Подробности: docs/REFACTORING_PLAN_remove_cluster_id_from_face_labels.md
"""

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=str(repo_root / "secrets.env"), override=False)
    load_dotenv(dotenv_path=str(repo_root / ".env"), override=False)
except Exception:
    pass

from backend.common.db import get_connection


def main() -> int:
    print("=" * 60)
    print("МИГРАЦИЯ: удаление cluster_id из face_labels")
    print("=" * 60)
    print()
    print("ВНИМАНИЕ: Перед запуском убедитесь, что ВСЕ места в коде")
    print("использующие cluster_id из face_labels уже обновлены!")
    print()
    
    response = input("Продолжить миграцию? (yes/no): ")
    if response.lower() != "yes":
        print("Миграция отменена.")
        return 1
    
    conn = get_connection()
    cur = conn.cursor()
    
    # Проверяем текущее состояние
    cur.execute("PRAGMA table_info(face_labels)")
    columns = cur.fetchall()
    cluster_id_col = next((c for c in columns if c["name"] == "cluster_id"), None)
    
    if not cluster_id_col:
        print("cluster_id уже удалён. Миграция не требуется.")
        return 0
    
    # Проверяем, есть ли данные
    cur.execute("SELECT COUNT(*) as cnt FROM face_labels")
    total_labels = cur.fetchone()["cnt"]
    print(f"Всего записей в face_labels: {total_labels}")
    
    if total_labels == 0:
        print("Таблица пуста, можно просто пересоздать.")
    else:
        print(f"Будет пересоздана таблица с {total_labels} записями.")
    
    print()
    print("Пересоздаём таблицу face_labels без cluster_id...")
    print()
    
    # 1. Создаём временную таблицу БЕЗ cluster_id
    cur.execute("""
        CREATE TABLE face_labels_new (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            face_rectangle_id   INTEGER NOT NULL,
            person_id           INTEGER NOT NULL,
            source              TEXT NOT NULL,
            confidence          REAL,
            created_at          TEXT NOT NULL
        )
    """)
    
    # 2. Копируем данные (БЕЗ cluster_id)
    print("Копируем данные (без cluster_id)...")
    cur.execute("""
        INSERT INTO face_labels_new (
            id, face_rectangle_id, person_id, source, confidence, created_at
        )
        SELECT 
            id, face_rectangle_id, person_id, source, confidence, created_at
        FROM face_labels
    """)
    
    copied = cur.rowcount
    print(f"Скопировано записей: {copied}")
    
    # 3. Удаляем старую таблицу и переименовываем новую
    print("Переименовываем таблицы...")
    cur.execute("DROP TABLE face_labels")
    cur.execute("ALTER TABLE face_labels_new RENAME TO face_labels")
    
    # 4. Восстанавливаем индексы (БЕЗ idx_face_labels_cluster)
    print("Восстанавливаем индексы...")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_face_labels_face ON face_labels(face_rectangle_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_face_labels_person ON face_labels(person_id)")
    # НЕ создаём idx_face_labels_cluster - колонки больше нет
    
    conn.commit()
    
    print()
    print("=" * 60)
    print("МИГРАЦИЯ ЗАВЕРШЕНА")
    print("=" * 60)
    print(f"Скопировано записей: {copied}")
    print()
    print("ВАЖНО: Убедитесь, что все места в коде обновлены!")
    print("Кластер теперь определяется через JOIN с face_cluster_members.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

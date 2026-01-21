#!/usr/bin/env python3
"""
Миграция данных: перенос person_id из face_labels в face_clusters.
Этап 3: Для каждого кластера находим персону через face_labels с source='cluster'.
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
    print("МИГРАЦИЯ ДАННЫХ: Перенос person_id из face_labels в face_clusters")
    print("=" * 60)
    print()
    
    conn = get_connection()
    cur = conn.cursor()
    
    # Проверяем кластеры с разными персонами
    print("Проверка кластеров с разными персонами...")
    cur.execute("""
        SELECT fc.id, COUNT(DISTINCT fl.person_id) as person_count, GROUP_CONCAT(DISTINCT fl.person_id) as person_ids
        FROM face_clusters fc
        JOIN face_cluster_members fcm ON fc.id = fcm.cluster_id
        JOIN face_labels fl ON fcm.face_rectangle_id = fl.face_rectangle_id
        WHERE fl.source = 'cluster'
        GROUP BY fc.id
        HAVING COUNT(DISTINCT fl.person_id) > 1
    """)
    problematic = cur.fetchall()
    
    if problematic:
        print(f"⚠️  Найдено {len(problematic)} кластеров с разными персонами:")
        for row in problematic:
            print(f"   Кластер {row['id']}: {row['person_count']} персон (IDs: {row['person_ids']})")
        print("   Будет использована первая попавшаяся персона.")
        print()
    
    # Миграция: для каждого кластера находим персону
    print("Миграция данных...")
    cur.execute("""
        UPDATE face_clusters
        SET person_id = (
            SELECT fl.person_id
            FROM face_cluster_members fcm
            JOIN face_labels fl ON fcm.face_rectangle_id = fl.face_rectangle_id
            WHERE fcm.cluster_id = face_clusters.id
            AND fl.source = 'cluster'
            LIMIT 1
        )
        WHERE EXISTS (
            SELECT 1
            FROM face_cluster_members fcm
            JOIN face_labels fl ON fcm.face_rectangle_id = fl.face_rectangle_id
            WHERE fcm.cluster_id = face_clusters.id
            AND fl.source = 'cluster'
        )
    """)
    
    updated = cur.rowcount
    print(f"Обновлено кластеров: {updated}")
    
    # Проверяем результат
    print()
    print("Проверка результата...")
    cur.execute("SELECT COUNT(*) as count FROM face_clusters WHERE person_id IS NOT NULL")
    with_person = cur.fetchone()['count']
    print(f"Кластеров с person_id: {with_person}")
    
    cur.execute("SELECT COUNT(*) as count FROM face_clusters WHERE person_id IS NULL")
    without_person = cur.fetchone()['count']
    print(f"Кластеров без person_id: {without_person}")
    
    # Проверяем архивные кластеры
    cur.execute("""
        SELECT COUNT(*) as count 
        FROM face_clusters 
        WHERE archive_scope = 'archive' AND person_id IS NULL
    """)
    archive_no_person = cur.fetchone()['count']
    if archive_no_person > 0:
        print(f"⚠️  Архивных кластеров без person_id: {archive_no_person}")
    else:
        print(f"✅ Все архивные кластеры имеют person_id")
    
    conn.commit()
    
    print()
    print("=" * 60)
    print("МИГРАЦИЯ ДАННЫХ ЗАВЕРШЕНА")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

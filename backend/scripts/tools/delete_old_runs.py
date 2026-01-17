"""Удаляет все прогоны кроме последнего."""

import sqlite3
import argparse
from pathlib import Path
from datetime import datetime


def main() -> None:
    parser = argparse.ArgumentParser(description="Удаляет все прогоны кроме последнего")
    parser.add_argument("--db", default="data/photosorter.db", help="Path to photosorter.db")
    parser.add_argument("--dry-run", action="store_true", help="Только показать, что будет удалено, без удаления")
    parser.add_argument("--confirm", action="store_true", help="Подтвердить удаление (без этого будет только dry-run)")
    args = parser.parse_args()
    
    db_path = Path(args.db)
    if not db_path.exists():
        print(f"БД не найдена: {db_path}")
        return
    
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    
    # Находим последний face_run (по started_at)
    cur.execute("""
        SELECT id, scope, root_path, status, started_at, finished_at, faces_found
        FROM face_runs
        ORDER BY started_at DESC
        LIMIT 1
    """)
    last_run = cur.fetchone()
    
    if not last_run:
        print("Нет прогонов для удаления")
        conn.close()
        return
    
    last_run_id = last_run["id"]
    print(f"Последний прогон (будет сохранён):")
    print(f"  ID: {last_run_id}")
    print(f"  Scope: {last_run['scope']}")
    print(f"  Root: {last_run['root_path']}")
    print(f"  Status: {last_run['status']}")
    print(f"  Started: {last_run['started_at']}")
    print(f"  Faces found: {last_run['faces_found']}")
    print()
    
    # Находим все остальные прогоны
    cur.execute("""
        SELECT id, scope, root_path, status, started_at, finished_at, faces_found
        FROM face_runs
        WHERE id != ?
        ORDER BY started_at DESC
    """, (last_run_id,))
    old_runs = cur.fetchall()
    
    if not old_runs:
        print("Нет старых прогонов для удаления")
        conn.close()
        return
    
    print(f"Будет удалено прогонов: {len(old_runs)}")
    print()
    print("Список прогонов для удаления:")
    for run in old_runs:
        print(f"  ID {run['id']}: {run['scope']} | {run['root_path']} | {run['status']} | {run['started_at']} | {run['faces_found']} лиц")
    print()
    
    # Подсчитываем связанные данные
    old_run_ids = [r["id"] for r in old_runs]
    placeholders = ",".join("?" * len(old_run_ids))
    
    cur.execute(f"""
        SELECT COUNT(*) as cnt FROM face_rectangles WHERE run_id IN ({placeholders})
    """, old_run_ids)
    faces_to_delete = cur.fetchone()["cnt"]
    
    cur.execute(f"""
        SELECT COUNT(*) as cnt FROM face_clusters WHERE run_id IN ({placeholders})
    """, old_run_ids)
    clusters_to_delete = cur.fetchone()["cnt"]
    
    cur.execute(f"""
        SELECT COUNT(*) as cnt 
        FROM face_cluster_members fcm
        JOIN face_clusters fc ON fcm.cluster_id = fc.id
        WHERE fc.run_id IN ({placeholders})
    """, old_run_ids)
    cluster_members_to_delete = cur.fetchone()["cnt"]
    
    cur.execute(f"""
        SELECT COUNT(*) as cnt 
        FROM face_labels fl
        JOIN face_clusters fc ON fl.cluster_id = fc.id
        WHERE fc.run_id IN ({placeholders})
    """, old_run_ids)
    labels_to_delete = cur.fetchone()["cnt"]
    
    print("Будет удалено:")
    print(f"  - {len(old_runs)} прогонов (face_runs)")
    print(f"  - {faces_to_delete:,} лиц (face_rectangles)")
    print(f"  - {clusters_to_delete:,} кластеров (face_clusters)")
    print(f"  - {cluster_members_to_delete:,} связей лиц с кластерами (face_cluster_members)")
    print(f"  - {labels_to_delete:,} меток персон (face_labels)")
    print()
    
    if args.dry_run or not args.confirm:
        print("DRY RUN - ничего не удалено")
        print("Для реального удаления запустите с --confirm")
        conn.close()
        return
    
    # Удаляем в правильном порядке (сначала зависимые данные)
    print("Удаление...")
    
    # 1. Удаляем метки персон для старых кластеров
    cur.execute(f"""
        DELETE FROM face_labels
        WHERE cluster_id IN (
            SELECT id FROM face_clusters WHERE run_id IN ({placeholders})
        )
    """, old_run_ids)
    labels_deleted = cur.rowcount
    print(f"  Удалено {labels_deleted:,} меток персон")
    
    # 2. Удаляем связи лиц с кластерами
    cur.execute(f"""
        DELETE FROM face_cluster_members
        WHERE cluster_id IN (
            SELECT id FROM face_clusters WHERE run_id IN ({placeholders})
        )
    """, old_run_ids)
    members_deleted = cur.rowcount
    print(f"  Удалено {members_deleted:,} связей лиц с кластерами")
    
    # 3. Удаляем кластеры
    cur.execute(f"""
        DELETE FROM face_clusters WHERE run_id IN ({placeholders})
    """, old_run_ids)
    clusters_deleted = cur.rowcount
    print(f"  Удалено {clusters_deleted:,} кластеров")
    
    # 4. Удаляем лица
    cur.execute(f"""
        DELETE FROM face_rectangles WHERE run_id IN ({placeholders})
    """, old_run_ids)
    faces_deleted = cur.rowcount
    print(f"  Удалено {faces_deleted:,} лиц")
    
    # 5. Удаляем прогоны
    cur.execute(f"""
        DELETE FROM face_runs WHERE id IN ({placeholders})
    """, old_run_ids)
    runs_deleted = cur.rowcount
    print(f"  Удалено {runs_deleted:,} прогонов")
    
    conn.commit()
    print()
    print("Удаление завершено!")
    
    # Показываем новый размер БД
    db_size = db_path.stat().st_size
    print(f"Новый размер БД: {db_size / (1024**3):.2f} GB ({db_size:,} байт)")
    
    conn.close()


if __name__ == "__main__":
    main()

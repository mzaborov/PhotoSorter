#!/usr/bin/env python3
"""Удаляет все пустые кластеры (кластеры без записей в face_cluster_members)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from backend.common.db import get_connection

def delete_empty_clusters(dry_run: bool = False) -> None:
    """
    Находит и удаляет все пустые кластеры.
    
    Args:
        dry_run: Если True, только показывает, что будет удалено, но не удаляет
    """
    conn = get_connection()
    cur = conn.cursor()
    
    # Находим все пустые кластеры (кластеры без записей в face_cluster_members)
    cur.execute("""
        SELECT fc.id
        FROM face_clusters fc
        LEFT JOIN face_cluster_members fcm ON fc.id = fcm.cluster_id
        WHERE fcm.cluster_id IS NULL
        ORDER BY fc.id
    """)
    
    empty_cluster_ids = [row["id"] for row in cur.fetchall()]
    
    if len(empty_cluster_ids) == 0:
        print("Пустых кластеров не найдено.")
        return
    
    print(f"Найдено пустых кластеров: {len(empty_cluster_ids)}")
    
    if dry_run:
        print("\n[DRY RUN] Были бы удалены следующие кластеры:")
        for cluster_id in empty_cluster_ids[:20]:  # Показываем первые 20
            print(f"  - Кластер #{cluster_id}")
        if len(empty_cluster_ids) > 20:
            print(f"  ... и еще {len(empty_cluster_ids) - 20} кластеров")
        print("\nДля реального удаления запустите скрипт без --dry-run")
        return
    
    # Удаляем метки для пустых кластеров
    placeholders = ",".join("?" * len(empty_cluster_ids))
    cur.execute(f"DELETE FROM face_labels WHERE cluster_id IN ({placeholders})", empty_cluster_ids)
    labels_deleted = cur.rowcount
    
    # Удаляем сами пустые кластеры
    cur.execute(f"DELETE FROM face_clusters WHERE id IN ({placeholders})", empty_cluster_ids)
    clusters_deleted = cur.rowcount
    
    conn.commit()
    
    print(f"\nУспешно удалено:")
    print(f"  Кластеров: {clusters_deleted}")
    print(f"  Меток: {labels_deleted}")

if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    
    if dry_run:
        print("[DRY RUN MODE]")
    
    delete_empty_clusters(dry_run=dry_run)

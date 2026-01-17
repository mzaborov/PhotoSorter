#!/usr/bin/env python3
"""Удаляет кластер и все связанные данные."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from backend.common.db import get_connection

def delete_cluster(cluster_id: int) -> None:
    conn = get_connection()
    cur = conn.cursor()
    
    # Проверяем существование кластера
    cur.execute("SELECT id FROM face_clusters WHERE id = ?", (cluster_id,))
    if not cur.fetchone():
        print(f"Кластер #{cluster_id} не найден")
        return
    
    # Удаляем связи лиц с кластером
    cur.execute("DELETE FROM face_cluster_members WHERE cluster_id = ?", (cluster_id,))
    members_deleted = cur.rowcount
    
    # Удаляем метки лиц, связанные с кластером
    cur.execute("DELETE FROM face_labels WHERE cluster_id = ?", (cluster_id,))
    labels_deleted = cur.rowcount
    
    # Удаляем сам кластер
    cur.execute("DELETE FROM face_clusters WHERE id = ?", (cluster_id,))
    
    conn.commit()
    
    print(f"Кластер #{cluster_id} удален:")
    print(f"  Удалено связей лиц: {members_deleted}")
    print(f"  Удалено меток: {labels_deleted}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: delete_cluster.py <cluster_id>")
        sys.exit(1)
    
    cluster_id = int(sys.argv[1])
    delete_cluster(cluster_id)

#!/usr/bin/env python3
"""
Проверка прогресса извлечения embeddings для прогона.
"""

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

from backend.common.db import get_connection

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=int, required=True)
    args = parser.parse_args()
    
    conn = get_connection()
    cur = conn.cursor()
    
    # Статистика по embeddings
    cur.execute("""
        SELECT 
            COUNT(*) AS total_faces,
            COUNT(embedding) AS faces_with_embedding,
            COUNT(*) - COUNT(embedding) AS faces_without_embedding
        FROM face_rectangles
        WHERE run_id = ? AND COALESCE(ignore_flag, 0) = 0
    """, (args.run_id,))
    
    stats = cur.fetchone()
    total = stats['total_faces'] or 0
    with_emb = stats['faces_with_embedding'] or 0
    without_emb = stats['faces_without_embedding'] or 0
    
    print("=" * 80)
    print(f"Прогресс извлечения embeddings для run_id={args.run_id}")
    print("=" * 80)
    print(f"Всего лиц: {total}")
    print(f"С embeddings: {with_emb}")
    print(f"Без embeddings: {without_emb}")
    if total > 0:
        percent = (with_emb / total) * 100
        print(f"Процент готовности: {percent:.1f}%")
    print("=" * 80)
    
    if without_emb > 0:
        print(f"\n⚠️  Осталось обработать: {without_emb} лиц")
        print("Можно перезапустить скрипт:")
        print(f"  python backend/scripts/tools/backfill_embeddings.py --run-id {args.run_id}")
    else:
        print("\n✓ Все embeddings извлечены!")

if __name__ == "__main__":
    main()

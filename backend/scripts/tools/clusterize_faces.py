#!/usr/bin/env python3
"""
Запускает кластеризацию лиц для указанного face_run_id.
"""

import argparse
import sys
from pathlib import Path

# Добавляем корень проекта в путь
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from backend.logic.face_recognition import cluster_face_embeddings
from backend.common.db import get_connection


def main() -> int:
    parser = argparse.ArgumentParser(description="Кластеризация лиц для face_run_id")
    parser.add_argument("--run-id", type=int, required=True, help="ID прогона детекции лиц (face_run_id)")
    parser.add_argument("--eps", type=float, default=0.4, help="Параметр eps для DBSCAN (по умолчанию 0.4)")
    parser.add_argument("--min-samples", type=int, default=2, help="Минимальное количество точек для кластера (по умолчанию 2)")
    args = parser.parse_args()

    print(f"Запуск кластеризации для face_run_id={args.run_id}")
    print(f"Параметры: eps={args.eps}, min_samples={args.min_samples}")
    print()

    try:
        result = cluster_face_embeddings(
            run_id=args.run_id,
            eps=args.eps,
            min_samples=args.min_samples,
            use_folder_context=True,
        )

        print("=" * 60)
        print("Результаты кластеризации:")
        print("=" * 60)
        print(f"Всего лиц: {result.get('total_faces', 0)}")
        print(f"Кластеров: {result.get('clusters_count', 0)}")
        print(f"Шум (не попали в кластеры): {result.get('noise_count', 0)}")
        if result.get('cluster_id'):
            print(f"ID кластера в БД: {result['cluster_id']}")
        print()

        clusters = result.get('clusters', {})
        if len(clusters) > 0:
            print("Кластеры:")
            for cluster_id, face_ids in clusters.items():
                print(f"  Кластер {cluster_id}: {len(face_ids)} лиц")
        
        return 0
    except Exception as e:
        print(f"Ошибка: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

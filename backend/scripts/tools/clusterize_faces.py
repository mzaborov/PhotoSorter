#!/usr/bin/env python3
"""
Запускает кластеризацию лиц для указанного face_run_id или для архива (--archive).
"""

import argparse
import sys
from pathlib import Path

# Добавляем корень проекта в путь
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from backend.logic.face_recognition import cluster_face_embeddings
from backend.common.db import get_connection


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Кластеризация лиц для face_run_id или для архива (--archive)."
    )
    parser.add_argument(
        "--run-id",
        type=int,
        default=None,
        help="ID прогона детекции лиц (face_run_id); обязателен, если не указан --archive",
    )
    parser.add_argument(
        "--archive",
        action="store_true",
        help="Кластеризовать лица архива (inventory_scope='archive')",
    )
    parser.add_argument("--eps", type=float, default=None, help="Параметр eps для DBSCAN (по умолчанию для архива: 0.44, иначе 0.4)")
    parser.add_argument("--min-samples", type=int, default=2, help="Минимальное количество точек для кластера (по умолчанию 2)")
    parser.add_argument(
        "--min-bbox-min",
        type=int,
        default=None,
        metavar="N",
        help="Для архива: учитывать только лица с min(bbox_w, bbox_h) >= N пикселей (по умолчанию при --archive: 70)",
    )
    args = parser.parse_args()

    if args.archive and args.run_id is not None:
        print("Ошибка: укажите либо --run-id, либо --archive, но не оба.", file=sys.stderr)
        return 1
    if not args.archive and args.run_id is None:
        print("Ошибка: укажите --run-id или --archive.", file=sys.stderr)
        return 1

    # Для архива — дефолты по результатам тюнинга (min_bbox_min=70, eps=0.44 → ARI_clustered ~0.36)
    if args.archive:
        if args.min_bbox_min is None:
            args.min_bbox_min = 70
        if args.eps is None:
            args.eps = 0.44
    elif args.eps is None:
        args.eps = 0.4
    if args.archive:
        print("Запуск кластеризации для архива (inventory_scope='archive')")
        if args.min_bbox_min is not None:
            print(f"Фильтр по качеству: min_bbox_min={args.min_bbox_min}")
    else:
        print(f"Запуск кластеризации для face_run_id={args.run_id}")
    print(f"Параметры: eps={args.eps}, min_samples={args.min_samples}")
    print()

    def progress(msg: str) -> None:
        print(f"  {msg}", flush=True)

    try:
        result = cluster_face_embeddings(
            run_id=args.run_id,
            archive_scope="archive" if args.archive else None,
            eps=args.eps,
            min_samples=args.min_samples,
            use_folder_context=True,
            progress_callback=progress,
            min_bbox_min=args.min_bbox_min if args.archive else None,
        )

        print("=" * 60)
        print("Результаты кластеризации:")
        print("=" * 60)
        if result.get('faces_added_to_existing', 0) > 0:
            print(f"Лиц добавлено в существующие кластеры: {result['faces_added_to_existing']}")
        print(f"Всего лиц обработано: {result.get('total_faces', 0)}")
        print(f"Новых кластеров создано: {result.get('clusters_count', 0)}")
        print(f"Шум (не попали в кластеры): {result.get('noise_count', 0)}")
        if result.get('cluster_id'):
            print(f"ID первого нового кластера в БД: {result['cluster_id']}")
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

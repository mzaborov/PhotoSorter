#!/usr/bin/env python3
"""
Копирует ground truth и эмбеддинги (по умолчанию) из основной БД в experiments.db
для тюнинга без обращения к основной БД. Запускать перед тюнингом и после обновления разметки.

Опционально — фильтр по качеству: только лица с достаточным confidence (детектор) и размером bbox.
Так можно оставить для кластеризации только «хорошие» кадры и сравнить ARI.

  python backend/scripts/tools/copy_archive_ground_truth_to_experiments.py
  python backend/scripts/tools/copy_archive_ground_truth_to_experiments.py --min-confidence 0.9 --min-bbox-min 50
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT))

from backend.common.db import get_connection
from backend.common.experiments_db import get_experiments_connection, ensure_experiments_tables


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Копирование ground truth архива в experiments.db для тюнинга (с опциональным фильтром по качеству)."
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=None,
        metavar="F",
        help="Минимальный confidence детектора (0..1). Не задано — без фильтра.",
    )
    parser.add_argument(
        "--min-bbox-min",
        type=int,
        default=None,
        metavar="N",
        help="Минимальная сторона bbox (min(bbox_w, bbox_h)) в пикселях. Отсекает мелкие/далёкие лица.",
    )
    args = parser.parse_args()

    main_conn = get_connection()
    main_cur = main_conn.cursor()
    main_cur.execute(
        """
        SELECT
            pr.id AS rectangle_id,
            COALESCE(pr.manual_person_id, fc.person_id) AS person_id,
            pr.embedding,
            pr.confidence,
            pr.bbox_w,
            pr.bbox_h
        FROM photo_rectangles pr
        JOIN files f ON f.id = pr.file_id
        LEFT JOIN face_clusters fc ON fc.id = pr.cluster_id
        WHERE f.inventory_scope = 'archive'
          AND pr.embedding IS NOT NULL
          AND pr.is_face = 1
          AND COALESCE(pr.ignore_flag, 0) = 0
          AND COALESCE(pr.manual_person_id, fc.person_id) IS NOT NULL
        ORDER BY pr.id
        """
    )
    rows = main_cur.fetchall()
    main_conn.close()

    if not rows:
        print("В основной БД нет размеченных лиц архива с эмбеддингами.")
        return 0

    # Фильтр по качеству
    if args.min_confidence is not None or args.min_bbox_min is not None:
        filtered = []
        for row in rows:
            d = dict(row)
            conf = d.get("confidence")
            if args.min_confidence is not None:
                if conf is None or float(conf) < args.min_confidence:
                    continue
            w = int(d.get("bbox_w") or 0)
            h = int(d.get("bbox_h") or 0)
            bbox_min = min(w, h)
            if args.min_bbox_min is not None and bbox_min < args.min_bbox_min:
                continue
            filtered.append(row)
        excluded = len(rows) - len(filtered)
        if excluded:
            print(f"По качеству исключено: {excluded} лиц (остаётся {len(filtered)})")
        rows = filtered

    if not rows:
        print("После фильтра по качеству не осталось лиц. Ослабьте --min-confidence / --min-bbox-min.")
        return 0

    exp_conn = get_experiments_connection()
    ensure_experiments_tables(exp_conn)
    exp_cur = exp_conn.cursor()
    exp_cur.execute("SELECT rectangle_id FROM experiment_outliers")
    exclude_ids = {row["rectangle_id"] for row in exp_cur.fetchall()}
    if exclude_ids:
        print(f"Исключаем из снапшота аутлайеров экспериментов: {len(exclude_ids)} лиц")
    exp_cur.execute("DELETE FROM tune_snapshot")
    copied = 0
    for row in rows:
        if row["rectangle_id"] in exclude_ids:
            continue
        exp_cur.execute(
            "INSERT INTO tune_snapshot (rectangle_id, person_id, embedding) VALUES (?, ?, ?)",
            (row["rectangle_id"], row["person_id"], row["embedding"]),
        )
        copied += 1
    exp_conn.commit()
    exp_conn.close()

    print(f"Скопировано в experiments.db: {copied} лиц (tune_snapshot).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Помечает аутлайеров в БД экспериментов: по tune_snapshot считает центроид каждой персоны (person_id),
расстояние каждого лица до центроида своей персоны, и записывает в experiment_outliers тех, кто
дальше всех (--top-pct) или с расстоянием выше порога (--threshold).
После этого copy_archive_ground_truth_to_experiments исключит их из снапшота.
Вся работа только в experiments.db, основная БД не трогается.

  python backend/scripts/tools/mark_outliers_in_experiments.py --top-pct 20
  python backend/scripts/tools/mark_outliers_in_experiments.py --threshold 0.35
  python backend/scripts/tools/mark_outliers_in_experiments.py --top-pct 15 --clear  # очистить аутлайеры и не помечать новых
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

try:
    import numpy as np
except ImportError:
    print("Ошибка: нужен numpy. Запустите из окружения с ML (например .venv-face).", file=sys.stderr)
    sys.exit(1)

from backend.common.experiments_db import get_experiments_connection, ensure_experiments_tables


def _cosine_distance(emb_norm: np.ndarray, centroid_norm: np.ndarray) -> float:
    return float(1.0 - np.dot(emb_norm, centroid_norm))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Пометить аутлайеров в experiments по расстоянию до центроида персоны (из tune_snapshot)."
    )
    parser.add_argument(
        "--top-pct",
        type=float,
        default=None,
        metavar="P",
        help="Исключить топ P%% самых далёких от центроида в каждой персоне (например 20)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        metavar="D",
        help="Исключить все лица с косинусным расстоянием до центроида > D (например 0.35)",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Только очистить experiment_outliers и выйти (сброс пометок)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Не писать в experiment_outliers")
    args = parser.parse_args()

    if not args.clear and args.top_pct is None and args.threshold is None:
        print("Укажите --top-pct и/или --threshold либо --clear", file=sys.stderr)
        return 1

    conn = get_experiments_connection()
    ensure_experiments_tables(conn)
    cur = conn.cursor()

    if args.clear:
        cur.execute("DELETE FROM experiment_outliers")
        conn.commit()
        print("experiment_outliers очищена.")
        return 0

    cur.execute("SELECT rectangle_id, person_id, embedding FROM tune_snapshot ORDER BY rectangle_id")
    rows = cur.fetchall()
    if not rows:
        print("tune_snapshot пуст. Сначала выполните copy_archive_ground_truth_to_experiments.py")
        return 0

    by_person: dict[int, list[tuple[int, np.ndarray]]] = {}
    for row in rows:
        d = dict(row)
        rid = d["rectangle_id"]
        pid = int(d["person_id"])
        emb = d["embedding"]
        if emb is None:
            continue
        try:
            raw = emb.decode("utf-8") if isinstance(emb, bytes) else emb
            arr = np.array(json.loads(raw), dtype=np.float32)
            if arr.size == 0 or np.isnan(arr).any():
                continue
            n = np.linalg.norm(arr)
            if n > 0:
                arr = arr / n
            if pid not in by_person:
                by_person[pid] = []
            by_person[pid].append((rid, arr))
        except Exception:
            continue

    outlier_rid_to_dist: dict[int, float] = {}
    for pid, rects in by_person.items():
        if len(rects) < 2:
            continue
        rect_ids = [r[0] for r in rects]
        embs = np.array([r[1] for r in rects])
        centroid = np.mean(embs, axis=0)
        c_norm = np.linalg.norm(centroid)
        if c_norm <= 0:
            continue
        centroid = centroid / c_norm
        dists = [(rect_ids[i], _cosine_distance(embs[i], centroid)) for i in range(len(rect_ids))]
        dists.sort(key=lambda x: x[1], reverse=True)
        if args.threshold is not None:
            for rid, d in dists:
                if d > args.threshold:
                    outlier_rid_to_dist[rid] = max(outlier_rid_to_dist.get(rid, 0), d)
        if args.top_pct is not None and args.top_pct > 0:
            k = max(1, int(round(len(dists) * args.top_pct / 100.0)))
            for rid, d in dists[:k]:
                outlier_rid_to_dist[rid] = max(outlier_rid_to_dist.get(rid, 0), d)
    outliers = list(outlier_rid_to_dist.items())

    run_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    n_total = sum(len(by_person[p]) for p in by_person)
    print(f"Аутлайеров по правилу: {len(outliers)} из {n_total} лиц")

    if not args.dry_run and outliers:
        cur.execute("DELETE FROM experiment_outliers")
        for rid, dist in outliers:
            cur.execute(
                "INSERT INTO experiment_outliers (rectangle_id, run_ts, cluster_id, distance) VALUES (?, ?, NULL, ?)",
                (rid, run_ts, dist),
            )
        conn.commit()
        print(f"Записано в experiment_outliers (run_ts={run_ts}). Дальше: copy_archive_ground_truth_to_experiments.py")
    elif args.dry_run:
        print("(dry-run: в БД не писали)")

    return 0


if __name__ == "__main__":
    sys.exit(main())

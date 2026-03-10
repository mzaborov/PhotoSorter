#!/usr/bin/env python3
"""
Поиск аутлайеров в кластерах персон архива. Центроид и расстояния — по данным из основной БД (только чтение).
Результат сохраняется только в БД экспериментов (experiments.db), основная БД не меняется.
При --save-to-experiments список аутлайеров пишется в experiment_outliers; при копировании снапшота они исключаются из тюнинга.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

try:
    import numpy as np
except ImportError:
    print("Ошибка: нужен numpy. Запустите из окружения с ML (например .venv-face).", file=sys.stderr)
    sys.exit(1)

from backend.common.db import get_connection
from backend.common.experiments_db import get_experiments_connection, ensure_experiments_tables


def _load_cluster_members(conn, person_id: int | None, progress: bool = True) -> list[tuple[int, int, bytes]]:
    """
    Загружает (rectangle_id, cluster_id, embedding) для кластеров архива.
    Если person_id задан — только кластеры этой персоны; иначе все кластеры архива с person_id.
    Только лица с ignore_flag=0.
    """
    if progress:
        print("  Загрузка членов кластеров из БД...", flush=True)
    cur = conn.cursor()
    if person_id is not None:
        cur.execute(
            """
            SELECT fr.id AS rectangle_id, fc.id AS cluster_id, fr.embedding
            FROM face_clusters fc
            JOIN photo_rectangles fr ON fr.cluster_id = fc.id
            WHERE fc.archive_scope = 'archive'
              AND fc.person_id = ?
              AND fr.embedding IS NOT NULL
              AND COALESCE(fr.ignore_flag, 0) = 0
            ORDER BY fc.id, fr.id
            """,
            (person_id,),
        )
    else:
        cur.execute(
            """
            SELECT fr.id AS rectangle_id, fc.id AS cluster_id, fr.embedding
            FROM face_clusters fc
            JOIN photo_rectangles fr ON fr.cluster_id = fc.id
            WHERE fc.archive_scope = 'archive'
              AND fc.person_id IS NOT NULL
              AND fr.embedding IS NOT NULL
              AND COALESCE(fr.ignore_flag, 0) = 0
            ORDER BY fc.id, fr.id
            """
        )
    rows = [(row["rectangle_id"], row["cluster_id"], row["embedding"]) for row in cur.fetchall()]
    if progress and rows:
        print(f"  Загружено записей: {len(rows)}", flush=True)
    return rows


def _cosine_distance(emb_norm: np.ndarray, centroid_norm: np.ndarray) -> float:
    """Косинусное расстояние = 1 - косинусное сходство (для нормализованных векторов)."""
    return float(1.0 - np.dot(emb_norm, centroid_norm))


def compute_outliers(
    members: list[tuple[int, int, bytes]],
    *,
    threshold: float | None = None,
    top_n: int | None = None,
    progress: bool = True,
) -> list[tuple[int, int, float]]:
    """
    По кластерам: центроид по ignore_flag=0 (уже отфильтровано в members),
    расстояния до центроида. Возвращает список (rectangle_id, cluster_id, distance)
    для кандидатов на исключение: либо distance > threshold, либо топ top_n по расстоянию в кластере.
    """
    if progress:
        print("  Группировка по кластерам...", flush=True)
    by_cluster: dict[int, list[tuple[int, bytes]]] = {}
    for rect_id, cluster_id, emb in members:
        if cluster_id not in by_cluster:
            by_cluster[cluster_id] = []
        by_cluster[cluster_id].append((rect_id, emb))
    n_clusters = len(by_cluster)
    if progress and n_clusters:
        print(f"  Кластеров: {n_clusters}. Вычисление центроидов и расстояний...", flush=True)

    outliers_set: set[tuple[int, int, float]] = set()
    done = 0
    for cluster_id, rects in by_cluster.items():
        if len(rects) < 2:
            continue
        embeddings_norm = []
        rect_ids = []
        for rect_id, emb_raw in rects:
            try:
                raw = emb_raw.decode("utf-8") if isinstance(emb_raw, bytes) else emb_raw
                arr = np.array(json.loads(raw), dtype=np.float32)
                if arr.size == 0 or np.isnan(arr).any():
                    continue
                n = np.linalg.norm(arr)
                if n > 0:
                    embeddings_norm.append(arr / n)
                    rect_ids.append(rect_id)
            except Exception:
                continue
        if len(embeddings_norm) < 2:
            continue
        centroid = np.mean(embeddings_norm, axis=0)
        c_norm = np.linalg.norm(centroid)
        if c_norm <= 0:
            continue
        centroid = centroid / c_norm
        dists = [(_rect_id, _cosine_distance(emb, centroid)) for _rect_id, emb in zip(rect_ids, embeddings_norm)]
        dists.sort(key=lambda x: x[1], reverse=True)  # худшие первые
        if threshold is not None:
            for rect_id, d in dists:
                if d > threshold:
                    outliers_set.add((rect_id, cluster_id, d))
        if top_n is not None and top_n > 0:
            for rect_id, d in dists[:top_n]:
                outliers_set.add((rect_id, cluster_id, d))
        done += 1
        if progress and n_clusters > 20 and done % 50 == 0:
            print(f"  Обработано кластеров: {done}/{n_clusters}", flush=True)
    return sorted(outliers_set, key=lambda x: x[2], reverse=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Поиск аутлайеров в кластерах персон архива по расстоянию до центроида."
    )
    parser.add_argument(
        "--person-id",
        type=int,
        default=None,
        help="ID персоны (только её кластеры); если не задан — все персоны архива",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Косинусное расстояние выше этого — кандидат на исключение (например 0.4)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="В каждом кластере помечать топ-N худших по расстоянию до центроида",
    )
    parser.add_argument(
        "--save-to-experiments",
        action="store_true",
        help="Сохранить список аутлайеров в experiments.db (experiment_outliers); основная БД не трогается",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Максимум строк в выводе (по умолчанию все)",
    )
    args = parser.parse_args()

    if args.threshold is None and args.top_n is None:
        print("Укажите --threshold и/или --top-n для выбора кандидатов на исключение.", file=sys.stderr)
        return 1
    if args.threshold is not None and args.threshold <= 0:
        print("--threshold должен быть > 0 (косинусное расстояние).", file=sys.stderr)
        return 1
    if args.top_n is not None and args.top_n < 1:
        print("--top-n должен быть >= 1.", file=sys.stderr)
        return 1

    conn = get_connection()
    try:
        members = _load_cluster_members(conn, args.person_id, progress=True)
    finally:
        conn.close()

    if not members:
        print("Нет данных (пустая выборка или неверный --person-id).")
        return 0

    outliers = compute_outliers(
        members,
        threshold=args.threshold,
        top_n=args.top_n,
        progress=True,
    )
    if not outliers:
        print("Кандидатов на исключение не найдено.")
        return 0

    # Сортируем по убыванию расстояния
    outliers.sort(key=lambda x: x[2], reverse=True)
    to_show = outliers
    if args.limit is not None and args.limit > 0:
        to_show = outliers[: args.limit]

    print(f"Кандидатов на исключение из тюнинга: {len(outliers)}")
    if args.limit:
        print(f"Показано первых {len(to_show)} (--limit {args.limit})")
    print()
    print("rectangle_id  cluster_id  distance")
    print("-" * 40)
    for rect_id, cluster_id, dist in to_show:
        print(f"{rect_id:<13} {cluster_id:<10} {dist:.4f}")

    if args.save_to_experiments and outliers:
        from datetime import datetime, timezone
        run_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        exp_conn = get_experiments_connection()
        ensure_experiments_tables(exp_conn)
        exp_cur = exp_conn.cursor()
        exp_cur.execute("DELETE FROM experiment_outliers")
        for rect_id, cluster_id, dist in outliers:
            exp_cur.execute(
                "INSERT INTO experiment_outliers (rectangle_id, run_ts, cluster_id, distance) VALUES (?, ?, ?, ?)",
                (rect_id, run_ts, cluster_id, dist),
            )
        exp_conn.commit()
        exp_conn.close()
        print()
        print(f"Сохранено в experiments.db (experiment_outliers): {len(outliers)} записей, run_ts={run_ts}")
    elif outliers and not args.save_to_experiments:
        print()
        print("Повторите с --save-to-experiments, чтобы сохранить список в experiments.db (основная БД не изменится).")

    return 0


if __name__ == "__main__":
    sys.exit(main())

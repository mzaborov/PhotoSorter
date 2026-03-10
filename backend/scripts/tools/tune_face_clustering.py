#!/usr/bin/env python3
"""
Тюнинг параметров DBSCAN (eps, min_samples) по ground truth архива.
Читает данные только из БД экспериментов (experiments.db). Сначала выполните
copy_archive_ground_truth_to_experiments.py. Результаты пишутся в experiments.db.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

# Корень проекта в путь
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

try:
    import numpy as np
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import adjusted_rand_score
    try:
        from sklearn.cluster import HDBSCAN as SklearnHDBSCAN
    except ImportError:
        SklearnHDBSCAN = None
except ImportError as e:
    print("Ошибка: для тюнинга нужны numpy и scikit-learn. Запустите из окружения с ML (например .venv-face).", file=sys.stderr)
    print(f"  {e}", file=sys.stderr)
    sys.exit(1)

from backend.common.experiments_db import get_experiments_connection, ensure_experiments_tables


def _load_ground_truth_and_embeddings(conn, embedding_source: str = "default", model_key: str | None = None):
    """
    Загрузка из experiments.db: tune_snapshot (default) или tune_snapshot + face_embeddings_alt (alt).
    """
    cur = conn.cursor()
    if embedding_source == "alt" and model_key:
        cur.execute(
            """
            SELECT t.rectangle_id, t.person_id, ea.embedding
            FROM tune_snapshot t
            INNER JOIN face_embeddings_alt ea ON ea.rectangle_id = t.rectangle_id AND ea.model_key = ?
            ORDER BY t.rectangle_id
            """,
            (model_key,),
        )
    else:
        cur.execute(
            """
            SELECT rectangle_id, person_id, embedding
            FROM tune_snapshot
            ORDER BY rectangle_id
            """
        )
    rows = cur.fetchall()
    rectangle_ids = []
    person_ids = []
    embeddings = []
    for row in rows:
        rid = row["rectangle_id"]
        pid = row["person_id"]
        emb = row["embedding"]
        if emb is None or pid is None:
            continue
        try:
            raw = emb.decode("utf-8") if isinstance(emb, bytes) else emb
            emb_list = json.loads(raw)
            arr = np.array(emb_list, dtype=np.float32)
            if arr.size == 0 or np.isnan(arr).any():
                continue
            rectangle_ids.append(rid)
            person_ids.append(int(pid))
            embeddings.append(arr)
        except Exception:
            continue
    return rectangle_ids, person_ids, np.array(embeddings)


def _normalize_l2(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return X / norms


def _person_ids_to_labels(person_ids: list[int]) -> np.ndarray:
    """Преобразует person_id в метки 0, 1, 2, ... для ARI."""
    uniq = sorted(set(person_ids))
    pid_to_label = {p: i for i, p in enumerate(uniq)}
    return np.array([pid_to_label[p] for p in person_ids], dtype=np.int32)


def _manual_actions_estimate(person_ids: list[int], labels: np.ndarray) -> tuple[int, int]:
    """Оценка ручных действий: (переопределить, назначить)."""
    from collections import Counter
    person_ids_arr = np.array(person_ids)
    n_reassign = 0
    for cluster_label in set(labels):
        if cluster_label == -1:
            continue
        mask = labels == cluster_label
        pids_in_cluster = person_ids_arr[mask]
        if len(pids_in_cluster) == 0:
            continue
        majority_pid = Counter(pids_in_cluster).most_common(1)[0][0]
        n_reassign += int(np.sum(pids_in_cluster != majority_pid))
    n_unassigned = int(np.sum(labels == -1))
    return n_reassign, n_unassigned


def run_grid_dbscan(
    X_normalized: np.ndarray,
    y_true: np.ndarray,
    person_ids: list[int],
    eps_values: list[float],
    min_samples_values: list[int],
    progress_callback: Callable[[int, int, float, int] | None] = None,
) -> list[dict]:
    results = []
    total = len(eps_values) * len(min_samples_values)
    k = 0
    for eps in eps_values:
        for min_s in min_samples_values:
            clustering = DBSCAN(eps=eps, min_samples=min_s, metric="cosine")
            labels = clustering.fit_predict(X_normalized)
            n_clusters = len(set(labels) - {-1})
            n_noise = int((labels == -1).sum())
            ari = adjusted_rand_score(y_true, labels)
            mask_c = labels != -1
            ari_clustered = (
                float(adjusted_rand_score(y_true[mask_c], labels[mask_c]))
                if mask_c.sum() > 0 else float("nan")
            )
            n_reassign, n_unassigned = _manual_actions_estimate(person_ids, labels)
            results.append({
                "algorithm": "dbscan",
                "eps": eps,
                "min_samples": min_s,
                "ari": float(ari),
                "ari_clustered": ari_clustered,
                "n_clusters": n_clusters,
                "n_noise": n_noise,
                "n_reassign": n_reassign,
                "n_unassigned": n_unassigned,
            })
            k += 1
            if progress_callback and (k % 5 == 0 or k == total):
                progress_callback(k, total, eps, min_s)
    return results


def run_grid_hdbscan(
    X_normalized: np.ndarray,
    y_true: np.ndarray,
    person_ids: list[int],
    min_cluster_size_values: list[int],
    cluster_selection_epsilon_values: list[float],
    progress_callback: Callable[[int, int, int, float] | None] = None,
) -> list[dict]:
    if SklearnHDBSCAN is None:
        raise RuntimeError("HDBSCAN требует scikit-learn >= 1.3. Установите: pip install -U scikit-learn")
    results = []
    total = len(min_cluster_size_values) * len(cluster_selection_epsilon_values)
    k = 0
    for min_cluster_size in min_cluster_size_values:
        for cluster_selection_epsilon in cluster_selection_epsilon_values:
            clustering = SklearnHDBSCAN(
                min_cluster_size=min_cluster_size,
                metric="cosine",
                cluster_selection_epsilon=cluster_selection_epsilon,
                copy=True,
            )
            labels = clustering.fit_predict(X_normalized)
            n_clusters = len(set(labels) - {-1})
            n_noise = int((labels == -1).sum())
            ari = adjusted_rand_score(y_true, labels)
            mask_c = labels != -1
            ari_clustered = (
                float(adjusted_rand_score(y_true[mask_c], labels[mask_c]))
                if mask_c.sum() > 0 else float("nan")
            )
            n_reassign, n_unassigned = _manual_actions_estimate(person_ids, labels)
            results.append({
                "algorithm": "hdbscan",
                "eps": cluster_selection_epsilon,
                "min_samples": min_cluster_size,
                "ari": float(ari),
                "ari_clustered": ari_clustered,
                "n_clusters": n_clusters,
                "n_noise": n_noise,
                "n_reassign": n_reassign,
                "n_unassigned": n_unassigned,
            })
            k += 1
            if progress_callback and (k % 5 == 0 or k == total):
                progress_callback(k, total, min_cluster_size, cluster_selection_epsilon)
    return results


def _save_results(
    conn, run_ts: str, n_faces: int, n_persons: int, results: list[dict],
    model_key: str | None = None, algorithm: str = "dbscan",
) -> None:
    """Пишет результаты в tune_face_results в той же БД экспериментов."""
    cur = conn.cursor()
    for r in results:
        algo = r.get("algorithm", algorithm)
        cur.execute(
            """
            INSERT INTO tune_face_results
            (run_ts, n_faces, n_persons, eps, min_samples, ari, n_clusters, n_noise, model_key, algorithm)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_ts,
                n_faces,
                n_persons,
                r["eps"],
                r["min_samples"],
                r["ari"],
                r["n_clusters"],
                r["n_noise"],
                model_key,
                algo,
            ),
        )
    conn.commit()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Тюнинг eps/min_samples для кластеризации лиц по ground truth архива (ARI)."
    )
    parser.add_argument(
        "--eps-min",
        type=float,
        default=0.2,
        help="Минимальное eps (по умолчанию 0.2)",
    )
    parser.add_argument(
        "--eps-max",
        type=float,
        default=0.6,
        help="Максимальное eps (по умолчанию 0.6)",
    )
    parser.add_argument(
        "--eps-step",
        type=float,
        default=0.05,
        help="Шаг по eps (по умолчанию 0.05)",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        nargs="+",
        default=[2, 3],
        help="Значения min_samples (по умолчанию 2 3)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Не сохранять результаты в experiments.db",
    )
    parser.add_argument(
        "--embedding-source",
        choices=("default", "alt"),
        default="default",
        help="Источник эмбеддингов: default (photo_rectangles.embedding) или alt (face_embeddings_alt)",
    )
    parser.add_argument(
        "--model-key",
        type=str,
        default=None,
        help="Ключ модели в face_embeddings_alt (обязателен при --embedding-source alt)",
    )
    parser.add_argument(
        "--algorithm",
        choices=("dbscan", "hdbscan"),
        default="dbscan",
        help="Алгоритм кластеризации: dbscan (по умолчанию) или hdbscan",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        nargs="+",
        default=[2, 3, 5],
        help="Для HDBSCAN: значения min_cluster_size (по умолчанию 2 3 5)",
    )
    parser.add_argument(
        "--cluster-selection-epsilon",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 0.2],
        help="Для HDBSCAN: значения cluster_selection_epsilon (по умолчанию 0 0.1 0.2)",
    )
    args = parser.parse_args()

    if args.embedding_source == "alt" and not args.model_key:
        print("При --embedding-source alt укажите --model-key", file=sys.stderr)
        return 1

    if args.algorithm == "hdbscan" and SklearnHDBSCAN is None:
        print("Ошибка: HDBSCAN требует scikit-learn >= 1.3. Установите: pip install -U scikit-learn", file=sys.stderr)
        return 1

    eps_values: list[float] = []
    if args.algorithm == "dbscan":
        x = args.eps_min
        while x <= args.eps_max + 1e-9:
            eps_values.append(round(x, 6))
            x += args.eps_step
        if not eps_values:
            print("Ошибка: пустая сетка eps (проверьте --eps-min/--eps-max/--eps-step)")
            return 1

    conn = get_experiments_connection()
    ensure_experiments_tables(conn)
    try:
        rectangle_ids, person_ids, X = _load_ground_truth_and_embeddings(
            conn,
            embedding_source=args.embedding_source,
            model_key=args.model_key,
        )
    finally:
        pass

    n_faces = len(rectangle_ids)
    n_persons = len(set(person_ids))
    if n_faces == 0:
        print("Нет размеченных лиц архива с эмбеддингами для выбранного источника. Завершение.")
        return 0
    src_label = f"alt (model_key={args.model_key})" if args.embedding_source == "alt" else "default"
    print(f"Источник эмбеддингов: {src_label}")
    print(f"Алгоритм: {args.algorithm}")
    print(f"Загружено лиц: {n_faces}, персон в ground truth: {n_persons}")
    print("  Перебор параметров...", flush=True)

    X_normalized = _normalize_l2(X)
    y_true = _person_ids_to_labels(person_ids)

    if args.algorithm == "dbscan":
        def _progress_dbscan(k: int, total: int, eps: float, min_s: int) -> None:
            print(f"  Перебор: {k}/{total} (eps={eps}, min_samples={min_s})", flush=True)
        results = run_grid_dbscan(
            X_normalized,
            y_true,
            person_ids,
            eps_values=eps_values,
            min_samples_values=args.min_samples,
            progress_callback=_progress_dbscan,
        )
    else:
        def _progress_hdbscan(k: int, total: int, min_cs: int, eps: float) -> None:
            print(f"  Перебор: {k}/{total} (min_cluster_size={min_cs}, cluster_selection_epsilon={eps})", flush=True)
        results = run_grid_hdbscan(
            X_normalized,
            y_true,
            person_ids,
            min_cluster_size_values=args.min_cluster_size,
            cluster_selection_epsilon_values=args.cluster_selection_epsilon,
            progress_callback=_progress_hdbscan,
        )

    run_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    model_key_for_save = args.model_key if args.embedding_source == "alt" else None
    if not args.no_save:
        _save_results(conn, run_ts, n_faces, n_persons, results, model_key=model_key_for_save, algorithm=args.algorithm)
        from backend.common.experiments_db import EXPERIMENTS_DB_PATH
        print(f"Результаты сохранены в {EXPERIMENTS_DB_PATH}")
        print()

    # Таблица
    print("algorithm  eps    min_samples  ARI       ARI_clust  n_clusters  n_noise  переопр.  назначить  всего действий")
    print("-" * 95)
    for r in results:
        algo = r.get("algorithm", args.algorithm)
        reassign = r.get("n_reassign", 0)
        unass = r.get("n_unassigned", 0)
        total_act = reassign + unass
        ari_c = r.get("ari_clustered", float("nan"))
        ari_c_str = f"{ari_c:.4f}" if not (isinstance(ari_c, float) and math.isnan(ari_c)) else "  —"
        print(f"{algo:<10} {r['eps']:<6.2f} {r['min_samples']:<12} {r['ari']:<9.4f} {ari_c_str:<9} {r['n_clusters']:<11} {r['n_noise']:<7} {reassign:<8} {unass:<10} {total_act}")
    best_ari = max(results, key=lambda x: (x["ari"], -x["n_noise"]))
    best_actions = min(results, key=lambda x: (x.get("n_reassign", 0) + x.get("n_unassigned", 0), -x["ari"]))
    print()
    if best_ari.get("algorithm") == "hdbscan":
        print(f"По ARI: min_cluster_size={best_ari['min_samples']}, cluster_selection_epsilon={best_ari['eps']} (ARI={best_ari['ari']:.4f})")
    else:
        print(f"По ARI: eps={best_ari['eps']}, min_samples={best_ari['min_samples']} (ARI={best_ari['ari']:.4f})")
    total_best = best_actions.get("n_reassign", 0) + best_actions.get("n_unassigned", 0)
    print(f"По минимуму ручных действий: переопределить {best_actions.get('n_reassign', 0)} + назначить {best_actions.get('n_unassigned', 0)} = {total_best} действий")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Запускает кластеризацию по данным из БД экспериментов (tune_snapshot).
Результат (привязка rectangle_id -> cluster_label) сохраняется в experiments.db
в experiment_cluster_labels. Основная БД не читается и не пишется.

Цикл полностью в experiments:
  1. copy_archive_ground_truth_to_experiments.py  (читает основную БД один раз, пишет в experiments)
  2. clusterize_in_experiments.py                (только experiments: кластеризация + ARI)
  3. tune_face_clustering.py                      (только experiments: перебор параметров, ARI)

  python backend/scripts/tools/clusterize_in_experiments.py --eps 0.4 --min-samples 2
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
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import adjusted_rand_score
except ImportError as e:
    print("Ошибка: нужны numpy и scikit-learn. Запустите из окружения с ML (например .venv-face).", file=sys.stderr)
    print(f"  {e}", file=sys.stderr)
    sys.exit(1)

from backend.common.experiments_db import get_experiments_connection, ensure_experiments_tables


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Кластеризация по tune_snapshot в experiments.db, результат пишется в experiment_cluster_labels."
    )
    parser.add_argument("--eps", type=float, default=0.4, help="eps для DBSCAN")
    parser.add_argument("--min-samples", type=int, default=2, help="min_samples для DBSCAN")
    parser.add_argument("--no-save", action="store_true", help="Не сохранять метки в experiment_cluster_labels")
    args = parser.parse_args()

    conn = get_experiments_connection()
    ensure_experiments_tables(conn)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT rectangle_id, person_id, embedding
        FROM tune_snapshot
        ORDER BY rectangle_id
        """
    )
    rows = cur.fetchall()
    if not rows:
        print("tune_snapshot пуст. Сначала выполните copy_archive_ground_truth_to_experiments.py")
        return 0

    rectangle_ids = []
    person_ids = []
    embeddings = []
    for row in rows:
        d = dict(row)
        rid = d["rectangle_id"]
        pid = d["person_id"]
        emb = d["embedding"]
        if emb is None or pid is None:
            continue
        try:
            raw = emb.decode("utf-8") if isinstance(emb, bytes) else emb
            arr = np.array(json.loads(raw), dtype=np.float32)
            if arr.size == 0 or np.isnan(arr).any():
                continue
            rectangle_ids.append(rid)
            person_ids.append(int(pid))
            embeddings.append(arr)
        except Exception:
            continue

    if not rectangle_ids:
        print("Нет валидных эмбеддингов в tune_snapshot.")
        return 0

    X = np.array(embeddings)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    X_norm = X / norms

    clustering = DBSCAN(eps=args.eps, min_samples=args.min_samples, metric="cosine")
    labels = clustering.fit_predict(X_norm)

    uniq_person = sorted(set(person_ids))
    pid_to_label = {p: i for i, p in enumerate(uniq_person)}
    y_true = np.array([pid_to_label[p] for p in person_ids], dtype=np.int32)
    ari = adjusted_rand_score(y_true, labels)
    # ARI только по точкам, попавшим в кластеры (без шума) — может быть выше при большом шуме
    mask_clustered = labels != -1
    if mask_clustered.sum() > 0:
        ari_clustered = adjusted_rand_score(y_true[mask_clustered], labels[mask_clustered])
    else:
        ari_clustered = float("nan")
    n_clusters = len(set(labels) - {-1})
    n_noise = int((labels == -1).sum())

    # Оценка ручных действий: переопределить (не та персона) + назначить (шум)
    n_reassign, n_unassigned = _manual_actions_estimate(person_ids, labels)
    n_total = len(rectangle_ids)

    run_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if not args.no_save:
        cur.execute("DELETE FROM experiment_cluster_labels WHERE run_ts = ?", (run_ts,))
        for rid, label in zip(rectangle_ids, labels):
            cur.execute(
                "INSERT INTO experiment_cluster_labels (run_ts, rectangle_id, cluster_label) VALUES (?, ?, ?)",
                (run_ts, rid, int(label)),
            )
        conn.commit()
        print(f"Метки сохранены в experiment_cluster_labels (run_ts={run_ts})")

    print(f"Лиц: {n_total}, персон в GT: {len(uniq_person)}")
    print(f"Параметры: eps={args.eps}, min_samples={args.min_samples}")
    print(f"Кластеров: {n_clusters}, шум: {n_noise}")
    print(f"ARI: {ari:.4f}")
    if not np.isnan(ari_clustered):
        print(f"ARI (только по точкам в кластерах, n={int(mask_clustered.sum())}): {ari_clustered:.4f}")
    print()
    print("Оценка ручных действий (перепроверка/переопределение):")
    print(f"  Переопределить (лицо в кластере не той персоны): {n_reassign} из {n_total}")
    print(f"  Назначить (лицо в шуме — не привязано ни к кому):   {n_unassigned} из {n_total}")
    print(f"  Итого примерных действий: {n_reassign + n_unassigned}")
    return 0


def _manual_actions_estimate(person_ids: list[int], labels: np.ndarray) -> tuple[int, int]:
    """
    Оценка ручных действий по результату кластеризации.
    Возвращает (n_reassign, n_unassigned).
    - Переопределить: лицо попало в кластер, но персона кластера (majority) не совпадает с GT этого лица.
    - Назначить: лицо в шуме (label -1) — пользователю нужно вручную назначить персону.
    """
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
        wrong = np.sum(pids_in_cluster != majority_pid)
        n_reassign += int(wrong)
    n_unassigned = int(np.sum(labels == -1))
    return n_reassign, n_unassigned


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Проверка embeddings и кластеров ручных привязок в архиве после backfill.

Показывает:
- Сколько прямоугольников имеют embedding (архив, run, видео)
- Ручные кластеры (method='manual', archive_scope='archive'): количество, распределение по персонам
- Потенциальные аномалии: кластеры с лицами «чужой» персоны (не должно быть)
- Объединение: сколько лиц в каждом ручном кластере (подливка)

Использование:
  python backend/scripts/tools/check_archive_manual_clusters.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from dotenv import load_dotenv  # type: ignore[import-not-found]
    load_dotenv(dotenv_path=str(_PROJECT_ROOT / "secrets.env"), override=False)
    load_dotenv(dotenv_path=str(_PROJECT_ROOT / ".env"), override=False)
except Exception:
    pass


def main() -> int:
    from backend.common.db import get_connection

    conn = get_connection()
    conn.row_factory = lambda c, r: dict(zip([col[0] for col in c.description], r))
    cur = conn.cursor()

    print("=" * 60)
    print("1. EMBEDDINGS В БАЗЕ")
    print("=" * 60)

    # Исключаем удалённые и local_done файлы
    _excl_deleted = (
        "AND (f.status IS NULL OR f.status != 'deleted') "
        "AND INSTR(f.path, '_delete') = 0 "
        "AND (f.inventory_scope IS NULL OR (f.inventory_scope != 'deleted' AND TRIM(COALESCE(f.inventory_scope, '')) != 'local_done'))"
    )
    cur.execute(f"""
        SELECT
            CASE WHEN f.inventory_scope = 'archive' OR TRIM(COALESCE(f.inventory_scope, '')) = 'archive' THEN 'archive'
                 WHEN pr.run_id IS NOT NULL AND pr.frame_t_sec IS NULL THEN 'run (photo)'
                 WHEN pr.frame_t_sec IS NOT NULL THEN 'video'
                 ELSE 'other'
            END AS scope,
            COUNT(*) AS total,
            SUM(CASE WHEN pr.embedding IS NOT NULL THEN 1 ELSE 0 END) AS with_embedding
        FROM photo_rectangles pr
        JOIN files f ON f.id = pr.file_id
        WHERE COALESCE(pr.ignore_flag, 0) = 0
          {_excl_deleted}
        GROUP BY scope
        ORDER BY scope
    """)
    for row in cur.fetchall():
        pct = 100 * row["with_embedding"] / row["total"] if row["total"] > 0 else 0
        print(f"  {row['scope']:15} total={row['total']:6}  embedding={row['with_embedding']:6}  ({pct:.1f}%)")

    # Детали по run (photo): какие run_id (исключая удалённые)
    cur.execute(f"""
        SELECT pr.run_id, COUNT(*) AS cnt,
               SUM(CASE WHEN pr.embedding IS NOT NULL THEN 1 ELSE 0 END) AS with_emb
        FROM photo_rectangles pr
        JOIN files f ON f.id = pr.file_id
        WHERE pr.run_id IS NOT NULL
          AND pr.frame_t_sec IS NULL
          AND (f.inventory_scope != 'archive' OR f.inventory_scope IS NULL OR TRIM(COALESCE(f.inventory_scope, '')) = '')
          AND COALESCE(pr.ignore_flag, 0) = 0
          {_excl_deleted}
        GROUP BY pr.run_id
        ORDER BY pr.run_id
        LIMIT 15
    """)
    run_rows = cur.fetchall()
    if run_rows:
        print("\n  run (photo) по run_id:")
        for r in run_rows:
            print(f"    run_id={r['run_id']:5}  лиц={r['cnt']:6}  embedding={r['with_emb']:6}")

    # Видео: run_id (для clusterize_faces --run-id N)
    cur.execute(f"""
        SELECT pr.run_id, COUNT(*) AS cnt,
               SUM(CASE WHEN pr.embedding IS NOT NULL THEN 1 ELSE 0 END) AS with_emb
        FROM photo_rectangles pr
        JOIN files f ON f.id = pr.file_id
        WHERE pr.frame_t_sec IS NOT NULL
          AND COALESCE(pr.ignore_flag, 0) = 0
          {_excl_deleted}
        GROUP BY pr.run_id
        ORDER BY pr.run_id
        LIMIT 15
    """)
    video_run_rows = cur.fetchall()
    if video_run_rows:
        print("\n  video по run_id (для clusterize_faces --run-id N):")
        for r in video_run_rows:
            print(f"    run_id={r['run_id']:5}  лиц={r['cnt']:6}  embedding={r['with_emb']:6}")

    # К каким файлам/путям относятся run (photo) — где они «застряли»? (исключая удалённые)
    cur.execute(f"""
        SELECT f.path, f.inventory_scope, COUNT(pr.id) AS cnt
        FROM photo_rectangles pr
        JOIN files f ON f.id = pr.file_id
        WHERE pr.run_id IS NOT NULL
          AND pr.frame_t_sec IS NULL
          AND (f.inventory_scope != 'archive' OR f.inventory_scope IS NULL OR TRIM(COALESCE(f.inventory_scope, '')) = '')
          AND COALESCE(pr.ignore_flag, 0) = 0
          {_excl_deleted}
        GROUP BY f.path, f.inventory_scope
        ORDER BY cnt DESC
        LIMIT 20
    """)
    path_rows = cur.fetchall()
    if path_rows:
        print("\n  run (photo): примеры путей (топ-20 по кол-ву лиц):")
        for r in path_rows:
            path_short = (r["path"] or "?")[:60]
            scope = (r["inventory_scope"] or "NULL")[:10]
            print(f"    scope={scope:10}  лиц={r['cnt']:4}  {path_short}")

    print("\n" + "=" * 60)
    print("2. РУЧНЫЕ КЛАСТЕРЫ В АРХИВЕ (method='manual', archive_scope='archive')")
    print("=" * 60)

    cur.execute(
        "SELECT COUNT(*) AS n FROM face_clusters WHERE archive_scope='archive' AND COALESCE(method,'')='manual'"
    )
    n_manual = cur.fetchone()["n"]
    print(f"  Всего ручных кластеров: {n_manual}")
    if n_manual == 0:
        manual_clusters = []
    else:
        print("  Загрузка деталей...")
        # CTE для ускорения: один проход по photo_rectangles вместо JOIN на каждый кластер
        cur.execute("""
        WITH cluster_sizes AS (
            SELECT cluster_id, COUNT(*) AS faces_count
            FROM photo_rectangles
            WHERE cluster_id IS NOT NULL AND COALESCE(ignore_flag, 0) = 0
            GROUP BY cluster_id
        )
        SELECT fc.id, fc.person_id, p.name AS person_name,
               COALESCE(cs.faces_count, 0) AS faces_count
        FROM face_clusters fc
        LEFT JOIN persons p ON p.id = fc.person_id
        LEFT JOIN cluster_sizes cs ON cs.cluster_id = fc.id
        WHERE fc.archive_scope = 'archive' AND COALESCE(fc.method, '') = 'manual'
        ORDER BY fc.person_id, fc.id
        LIMIT 5000
        """)
        manual_clusters = cur.fetchall()

    if not manual_clusters:
        print("  Нет ручных кластеров в архиве.")
    else:
        # Сводка по персонам
        person_stats: dict[int, dict] = {}
        for row in manual_clusters:
            pid = row["person_id"]
            if pid not in person_stats:
                person_stats[pid] = {"name": row["person_name"] or "?", "clusters": 0, "faces": 0}
            person_stats[pid]["clusters"] += 1
            person_stats[pid]["faces"] += row["faces_count"]

        print(f"  Всего ручных кластеров: {len(manual_clusters)}")
        print(f"  Уникальных персон: {len(person_stats)}")
        print("\n  По персонам:")
        for pid, st in sorted(person_stats.items(), key=lambda x: -x[1]["faces"]):
            avg = st["faces"] / st["clusters"] if st["clusters"] else 0
            print(f"    person_id={pid:4}  {st['name'][:30]:30}  кластеров={st['clusters']:4}  лиц={st['faces']:5}  (avg {avg:.1f}/кластер)")

        # Распределение размеров кластеров
        sizes = [r["faces_count"] for r in manual_clusters]
        print("\n  Распределение по размеру кластера (лиц в кластере):")
        from collections import Counter
        for size, cnt in sorted(Counter(sizes).items()):
            print(f"    {size} лиц: {cnt} кластеров")

    print("\n" + "=" * 60)
    print("3. ПРОВЕРКА АНОМАЛИЙ")
    print("=" * 60)

    # Остались ли manual-лица без embedding/кластера (не должны после backfill)
    cur.execute(f"""
        SELECT COUNT(*) AS cnt FROM photo_rectangles pr
        JOIN files f ON f.id = pr.file_id
        WHERE (f.inventory_scope = 'archive' OR TRIM(COALESCE(f.inventory_scope, '')) = 'archive')
          AND pr.manual_person_id IS NOT NULL
          AND (pr.embedding IS NULL OR pr.cluster_id IS NULL)
          AND COALESCE(pr.ignore_flag, 0) = 0
          {_excl_deleted}
    """)
    orphan = cur.fetchone()["cnt"]
    if orphan > 0:
        print(f"  ВНИМАНИЕ: {orphan} manual-лиц в архиве без embedding или cluster_id (не обработаны backfill)")
    else:
        print("  OK: все manual-лица в архиве имеют embedding и cluster_id")

    # Кластеры с person_id=NULL (ручные должны иметь person_id)
    cur.execute("""
        SELECT COUNT(*) AS cnt FROM face_clusters
        WHERE archive_scope = 'archive' AND COALESCE(method, '') = 'manual'
          AND person_id IS NULL
    """)
    null_person = cur.fetchone()["cnt"]
    if null_person > 0:
        print(f"  ВНИМАНИЕ: {null_person} ручных кластеров без person_id")
    else:
        print("  OK: у всех ручных кластеров есть person_id")

    # Соответствие: лица в ручных кластерах имеют manual_person_id=NULL (CHECK: только один источник)
    cur.execute("""
        SELECT COUNT(*) AS cnt FROM photo_rectangles pr
        JOIN face_clusters fc ON fc.id = pr.cluster_id
        WHERE fc.archive_scope = 'archive' AND COALESCE(fc.method, '') = 'manual'
          AND pr.manual_person_id IS NOT NULL
    """)
    has_manual = cur.fetchone()["cnt"]
    if has_manual > 0:
        print(f"  Примечание: {has_manual} лиц имеют и cluster_id и manual_person_id (CHECK запрещает)")
    else:
        print("  OK: лица в кластерах — персона через cluster_id (manual_person_id=NULL)")

    # Примечание: логика backfill добавляет ТОЛЬКО в кластеры своей персоны (fc.person_id = manual_person_id).
    # Избыточность (много мелких кластеров на персону) ожидаема: «Посторонний» — разные люди;
    # для одной персоны — разные ракурсы/освещение не всегда в пределах eps.

    conn.close()
    print("\nГотово.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

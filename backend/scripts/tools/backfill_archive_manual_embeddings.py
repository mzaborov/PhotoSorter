#!/usr/bin/env python3
"""
Backfill: вычисляет embeddings и создаёт/пополняет кластеры для ручных привязок
в архиве (inventory_scope='archive').

Находит файлы с photo_rectangles: manual_person_id IS NOT NULL и (embedding IS NULL
или cluster_id IS NULL). Извлекает embeddings, подливает в существующие кластеры
персоны (если расстояние ≤ eps) или создаёт новые.

Использование:
  python backend/scripts/tools/backfill_archive_manual_embeddings.py
  python backend/scripts/tools/backfill_archive_manual_embeddings.py --dry-run
  python backend/scripts/tools/backfill_archive_manual_embeddings.py --limit 100
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import traceback
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


def _path_to_local(path: str) -> str | None:
    """Преобразует path из БД в локальный путь для чтения файла."""
    if not path or not isinstance(path, str):
        return None
    path = path.strip()
    if path.startswith("local:"):
        p = path[6:].strip()
        return os.path.abspath(p) if p else None
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backfill embeddings и кластеров для manual-лиц в архиве"
    )
    parser.add_argument("--dry-run", action="store_true", help="Только показать, что будет обработано")
    parser.add_argument("--limit", type=int, default=None, help="Максимум файлов для обработки")
    parser.add_argument("-v", "--verbose", action="store_true", help="Печатать первые ошибки с traceback")
    args = parser.parse_args()

    from backend.common.db import get_connection, _get_file_id_from_path
    from backend.logic.face_recognition import process_manual_faces_for_archived_file

    conn = get_connection()
    conn.row_factory = lambda c, r: dict(zip([col[0] for col in c.description], r))
    cur = conn.cursor()

    cur.execute(
        """
        SELECT DISTINCT f.id AS file_id, f.path AS file_path
        FROM files f
        JOIN photo_rectangles pr ON pr.file_id = f.id
        WHERE (f.inventory_scope = 'archive' OR TRIM(COALESCE(f.inventory_scope, '')) = 'archive')
          AND pr.manual_person_id IS NOT NULL
          AND (pr.embedding IS NULL OR pr.cluster_id IS NULL)
          AND COALESCE(pr.ignore_flag, 0) = 0
        ORDER BY f.id
        """
    )
    rows = cur.fetchall()
    conn.close()

    if not rows:
        print("Нет файлов для обработки (все manual-лица уже с embedding и кластером).")
        return 0

    print(f"Найдено файлов для обработки: {len(rows)}")
    if args.limit:
        rows = rows[: args.limit]
        print(f"Ограничение: обрабатываем {len(rows)}")
    if args.dry_run:
        for r in rows[:20]:
            print(f"  file_id={r['file_id']} path={r['file_path']}")
        if len(rows) > 20:
            print(f"  ... и ещё {len(rows) - 20}")
        return 0

    print("Загрузка модели (при первом файле)...")
    total_processed = 0
    total_errors = 0
    files_ok = 0
    files_skip = 0

    disk = None
    try:
        from backend.common.yadisk_client import get_disk
        disk = get_disk()
    except Exception as e:
        print(f"YaDisk недоступен: {e}")
        if args.verbose:
            traceback.print_exc()

    verbose_errors_shown = 0
    for i, row in enumerate(rows):
        file_id = row["file_id"]
        db_path = row["file_path"] or ""

        local_path = _path_to_local(db_path)
        tmp_path = None

        if local_path and os.path.isfile(local_path):
            pass
        elif db_path.startswith("disk:") and disk:
            try:
                remote = db_path.replace("disk:/", "/")
                suf = Path(db_path).suffix or ".jpg"
                fd, tmp_path = tempfile.mkstemp(suffix=suf)
                os.close(fd)
                disk.download(remote, tmp_path)
                local_path = tmp_path
            except Exception as e:
                print(f"  file_id={file_id}: не удалось скачать {db_path}: {e}")
                files_skip += 1
                continue
        else:
            print(f"  file_id={file_id}: локальный путь недоступен ({db_path})")
            files_skip += 1
            continue

        try:
            conn = get_connection()
            conn.row_factory = lambda c, r: dict(zip([col[0] for col in c.description], r))
            error_samples: list[tuple[int, str, str]] = [] if args.verbose else []
            proc, errs = process_manual_faces_for_archived_file(
                conn=conn,
                file_id=file_id,
                local_file_path=local_path,
                error_samples=error_samples if args.verbose else None,
            )
            conn.close()

            total_processed += proc
            total_errors += errs
            if proc > 0:
                files_ok += 1
            if (i + 1) % 50 == 0 or proc > 0 or errs > 0:
                print(f"  [{i + 1}/{len(rows)}] file_id={file_id} processed={proc} errors={errs}")
            if errs > 0 and args.verbose and verbose_errors_shown < 2 and error_samples:
                verbose_errors_shown += 1
                for rect_id, ex_type, ex_msg in error_samples[:3]:
                    print(f"    rect_id={rect_id}: {ex_type}: {ex_msg}")
        except Exception as e:
            total_errors += 1
            print(f"  file_id={file_id}: исключение {type(e).__name__}: {e}")
            if args.verbose and verbose_errors_shown < 2:
                verbose_errors_shown += 1
                traceback.print_exc()
        finally:
            if tmp_path and os.path.isfile(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

    print(f"\nГотово. Файлов обработано: {files_ok}, пропущено: {files_skip}")
    print(f"Лиц: успешно={total_processed}, ошибок={total_errors}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Заполняет пустые taken_at для файлов (image/video) по приоритету:
EXIF → имя файла → системные атрибуты (mtime / modified).

Использование:
  python backend/scripts/tools/backfill_taken_at.py --dry-run --limit 10
  python backend/scripts/tools/backfill_taken_at.py --scope local
  python backend/scripts/tools/backfill_taken_at.py
"""

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
backend_dir = project_root / "backend"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(backend_dir))  # для local_sort: from common.db

from backend.common.db import DedupStore, get_connection
from backend.logic.pipeline.local_sort import _parse_date_from_filename, _try_exif_datetime_iso

# YaDisk — опционально, только при scope yadisk/all и наличии disk-путей
# Возвращает (disk или None, сообщение об ошибке или None)
def _get_disk_or_none():
    try:
        from backend.common.yadisk_client import get_disk
        return get_disk(), None
    except Exception as e:
        return None, str(e)


def _strip_local_prefix(p: str) -> str:
    return p[len("local:"):] if (p or "").startswith("local:") else p


def _get_date_from_system_attrs(
    path: str, is_yadisk: bool, row_modified: str | None, disk=None
) -> str | None:
    """Дата из системных атрибутов: для local — st_mtime, для YaDisk — modified из БД или disk.get_meta()."""
    if is_yadisk:
        if row_modified and len(str(row_modified).strip()) >= 10:
            s = str(row_modified).strip()
            if "T" in s or " " in s:
                return s[:19].replace(" ", "T") + "Z" if len(s) >= 19 else (s[:10] + "T00:00:00Z")
            return s[:10] + "T00:00:00Z"
        if disk:
            try:
                meta = disk.get_meta(path)
                if meta and getattr(meta, "modified", None):
                    mod = meta.modified
                    if hasattr(mod, "isoformat"):
                        return mod.isoformat().replace("+00:00", "Z")[:19] + "Z"
            except Exception:
                pass
        return None
    # local
    local_path = _strip_local_prefix(path)
    if not local_path or not os.path.isfile(local_path):
        return None
    try:
        st = os.stat(local_path)
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(st.st_mtime))
    except Exception:
        return None


def _get_taken_at_exif_local(abspath: str) -> str | None:
    """Читает EXIF DateTime для локального изображения."""
    try:
        from PIL import Image
        with Image.open(abspath) as img:
            return _try_exif_datetime_iso(img)
    except Exception:
        return None


def _get_taken_at_exif_yadisk(path: str, disk) -> str | None:
    """Скачивает файл во временный, читает EXIF, удаляет временный."""
    if not disk:
        return None
    try:
        from PIL import Image
        with tempfile.NamedTemporaryFile(suffix=os.path.splitext(path)[1] or ".jpg", delete=False) as f:
            tmp = f.name
        try:
            disk.download(path, tmp)
            with Image.open(tmp) as img:
                return _try_exif_datetime_iso(img)
        finally:
            try:
                os.unlink(tmp)
            except Exception:
                pass
    except Exception:
        return None


def process_file(
    row: dict,
    dedup: DedupStore,
    disk=None,
    dry_run: bool = False,
) -> tuple[bool, str | None]:
    """
    Определяет taken_at по приоритету EXIF → имя файла → системные атрибуты.
    Возвращает (updated, taken_at).
    """
    path = str(row.get("path") or "")
    name = str(row.get("name") or os.path.basename(path))
    media_type = (str(row.get("media_type") or "")).lower()
    is_image = media_type == "image"
    is_yadisk = path.startswith("disk:")

    taken_at = None
    # 1) EXIF — только для изображений
    if is_image:
        if is_yadisk and disk:
            taken_at = _get_taken_at_exif_yadisk(path, disk)
        else:
            local_path = _strip_local_prefix(path)
            if local_path and os.path.isfile(local_path):
                taken_at = _get_taken_at_exif_local(local_path)
    # 2) имя файла
    if taken_at is None:
        taken_at = _parse_date_from_filename(name)
    # 3) системные атрибуты
    if taken_at is None:
        taken_at = _get_date_from_system_attrs(
            path, is_yadisk, row.get("modified"), disk
        )

    if not taken_at:
        return False, None
    if not dry_run:
        dedup.set_taken_at_and_gps(path=path, taken_at=taken_at, gps_lat=None, gps_lon=None)
    return True, taken_at


def main() -> None:
    ap = argparse.ArgumentParser(description="Заполнение taken_at для файлов без даты съёмки")
    ap.add_argument("--dry-run", action="store_true", help="Не записывать в БД")
    ap.add_argument("--limit", type=int, default=0, help="Максимум файлов (0 = без ограничения)")
    ap.add_argument("--scope", choices=("local", "yadisk", "all"), default="all", help="Фильтр по источнику")
    args = ap.parse_args()

    # Загрузка .env из корня проекта и backend (как в migrate_archive_faces, main.py)
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=str(project_root / "secrets.env"), override=False)
        load_dotenv(dotenv_path=str(project_root / ".env"), override=False)
        load_dotenv(dotenv_path=str(backend_dir / "secrets.env"), override=False)
        load_dotenv(dotenv_path=str(backend_dir / ".env"), override=False)
    except Exception:
        pass

    conn = get_connection()
    cur = conn.cursor()
    scope = args.scope
    where = ["(taken_at IS NULL OR taken_at = '')", "media_type IN ('image', 'video')"]
    params = []
    if scope == "local":
        where.append("path LIKE 'local:%'")
    elif scope == "yadisk":
        where.append("path LIKE 'disk:%'")
    cur.execute(
        f"""
        SELECT id, path, name, media_type, modified
        FROM files
        WHERE {' AND '.join(where)}
        ORDER BY path
        """ + (" LIMIT ?" if args.limit else ""),
        params + ([args.limit] if args.limit else []),
    )
    rows = [dict(zip([c[0] for c in cur.description], r)) for r in cur.fetchall()]
    conn.close()

    total = len(rows)
    if total == 0:
        print("Нет файлов с пустым taken_at.")
        return
    print(f"Найдено файлов с пустым taken_at: {total}")

    disk = None
    if scope in ("yadisk", "all") and any(str(r.get("path") or "").startswith("disk:") for r in rows):
        disk, yadisk_err = _get_disk_or_none()
        if disk is None:
            reason = f": {yadisk_err}" if yadisk_err else ""
            print(f"Предупреждение: YaDisk недоступен{reason}. Для disk:-путей будет использоваться только имя файла и modified из БД.")

    updated = 0
    errors = 0
    step = 10  # прогресс каждые 10 файлов (и после первого)
    batch_size = 50  # между батчами закрываем соединение, чтобы не держать БД занятой
    done = 0

    def report():
        print(f"Обработано {done}/{total}…", flush=True)

    report()
    for start in range(0, total, batch_size):
        batch = rows[start : start + batch_size]
        dedup = DedupStore()
        try:
            for row in batch:
                try:
                    ok, taken = process_file(row, dedup, disk=disk, dry_run=args.dry_run)
                    if ok:
                        updated += 1
                except Exception as e:
                    errors += 1
                    print(f"Ошибка {row.get('path')}: {e}", flush=True)
                done += 1
                if done % step == 0 or done == total or done == 1:
                    report()
        finally:
            dedup.close()

    print(f"Обработано: {total}, обновлено: {updated}, ошибок: {errors}")
    if args.dry_run and updated:
        print("(режим --dry-run, в БД ничего не записано)")


if __name__ == "__main__":
    main()

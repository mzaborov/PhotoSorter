#!/usr/bin/env python3
"""
Заполняет пустые taken_at для файлов (image/video) по приоритету:
EXIF → имя файла → дата создания файла → дата изменения файла.

С флагом --fix-from-filename также исправляет уже записанные даты, если
по EXIF/имени файла/системе получается другая дата (например, в имени файла 20180819, а в БД 2025-12-27).

Использование:
  python backend/scripts/tools/backfill_taken_at.py --dry-run --limit 10
  python backend/scripts/tools/backfill_taken_at.py --scope local
  python backend/scripts/tools/backfill_taken_at.py --fix-from-filename --dry-run   # проверить исправления
  python backend/scripts/tools/backfill_taken_at.py --fix-from-filename              # исправить даты по имени/EXIF
  python backend/scripts/tools/backfill_taken_at.py --fix-from-filename --offset 5000 --limit 5000  # частями
  python backend/scripts/tools/backfill_taken_at.py --fix-from-filename --no-download-exif  # удалённые (YaDisk) без скачивания для EXIF — быстрее
"""

import argparse
import os
import re
import sys
import tempfile
import time
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
backend_dir = project_root / "backend"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(backend_dir))

from backend.common.db import DedupStore, get_connection

# Локальные копии парсера даты и EXIF, чтобы не тянуть local_sort (cv2)


def _parse_date_from_filename(filename: str) -> str | None:
    """Парсит дату/время из имени файла. Возвращает ISO 'YYYY-MM-DDTHH:MM:SSZ' или None."""
    if not filename or not isinstance(filename, str):
        return None
    name = os.path.splitext(filename)[0].strip()
    if not name:
        return None
    base = re.sub(r"^(IMG[-_]?|VID[-_]?|WA\d*[-_]?|PANO[-_]?)", "", name, flags=re.IGNORECASE).strip()
    if not base:
        return None

    def to_iso(y: str, m: str, d: str, h: str = "00", i: str = "00", s: str = "00") -> str | None:
        if len(y) == 4 and y.isdigit() and len(m) == 2 and m.isdigit() and len(d) == 2 and d.isdigit():
            if len(h) == 2 and h.isdigit() and len(i) == 2 and i.isdigit() and len(s) == 2 and s.isdigit():
                return f"{y}-{m}-{d}T{h}:{i}:{s}Z"
            return f"{y}-{m}-{d}T00:00:00Z"
        return None

    m = re.match(r"^(\d{4})(\d{2})(\d{2})[-_](\d{2})(\d{2})(\d{2})$", base)
    if m:
        return to_iso(m.group(1), m.group(2), m.group(3), m.group(4), m.group(5), m.group(6))
    m = re.match(r"^(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})$", base)
    if m:
        return to_iso(m.group(1), m.group(2), m.group(3), m.group(4), m.group(5), m.group(6))
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})[-_\s](\d{2})[-:](\d{2})[-:](\d{2})$", base)
    if m:
        return to_iso(m.group(1), m.group(2), m.group(3), m.group(4), m.group(5), m.group(6))
    m = re.match(r"^(\d{4})(\d{2})(\d{2})$", base)
    if m:
        return to_iso(m.group(1), m.group(2), m.group(3))
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", base)
    if m:
        return to_iso(m.group(1), m.group(2), m.group(3))
    # IMG-20180808-WA0009.jpg → после префикса base = "20180808-WA0009": 8 цифр в начале + суффикс
    m = re.match(r"^(\d{4})(\d{2})(\d{2})(?=[-_\s]|$)", base)
    if m:
        return to_iso(m.group(1), m.group(2), m.group(3))
    m = re.search(r"[-_](\d{4})(\d{2})(\d{2})(?:[-_](\d{2})(\d{2})(\d{2}))?$", base)
    if m:
        if m.group(4) is not None:
            return to_iso(m.group(1), m.group(2), m.group(3), m.group(4), m.group(5), m.group(6))
        return to_iso(m.group(1), m.group(2), m.group(3))
    return None


def _try_exif_datetime_iso(img) -> str | None:
    """EXIF DateTimeOriginal/DateTime → 'YYYY-MM-DDTHH:MM:SSZ'."""
    try:
        exif = img.getexif()
    except Exception:
        return None
    if not exif:
        return None
    dt = exif.get(36867) or exif.get(306)  # DateTimeOriginal or DateTime
    if not dt or not isinstance(dt, str):
        return None
    s = dt.strip()
    try:
        if len(s) >= 19 and s[4] == ":" and s[7] == ":" and s[10] in (" ", "T"):
            yyyy, mm, dd = s[0:4], s[5:7], s[8:10]
            hh, mi, ss = s[11:13], s[14:16], s[17:19]
            if yyyy.isdigit() and mm.isdigit() and dd.isdigit() and hh.isdigit() and mi.isdigit() and ss.isdigit():
                return f"{yyyy}-{mm}-{dd}T{hh}:{mi}:{ss}Z"
    except Exception:
        pass
    return None

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
    path: str, is_yadisk: bool, row_modified: str | None, row_created: str | None = None, disk=None
) -> str | None:
    """
    Дата из системных атрибутов.
    Приоритет: дата создания → дата изменения.
    Для local: st_birthtime (macOS/Linux) или st_ctime (Windows) как создание, затем st_mtime как изменение.
    Для YaDisk: created из БД/API, затем modified.
    """
    if is_yadisk:
        # Сначала пробуем API — там реальные даты создания/изменения файла на Диске.
        # В БД modified может быть датой последней синхронизации (2025–2026), а не съёмки.
        if disk:
            try:
                meta = disk.get_meta(path)
                if meta:
                    for attr in ("created", "modified"):
                        mod = getattr(meta, attr, None)
                        if mod is not None and hasattr(mod, "isoformat"):
                            return mod.isoformat().replace("+00:00", "Z")[:19] + "Z"
            except Exception:
                pass
        # Fallback: из БД (created, затем modified)
        for val in (row_created, row_modified):
            if val and len(str(val).strip()) >= 10:
                s = str(val).strip()
                if "T" in s or " " in s:
                    return s[:19].replace(" ", "T") + "Z" if len(s) >= 19 else (s[:10] + "T00:00:00Z")
                return s[:10] + "T00:00:00Z"
        return None
    # local: сначала дата создания, затем изменения
    local_path = _strip_local_prefix(path)
    if not local_path or not os.path.isfile(local_path):
        return None
    try:
        st = os.stat(local_path)
        # Дата создания: st_birthtime (macOS, Linux 4.11+), на Windows — st_ctime
        creation_ts = getattr(st, "st_birthtime", None) if hasattr(st, "st_birthtime") else None
        if creation_ts is None and os.name == "nt":
            creation_ts = st.st_ctime
        if creation_ts is not None:
            return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(creation_ts))
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
    fix_from_filename: bool = False,
    no_download_exif: bool = False,
) -> tuple[bool, str | None]:
    """
    Определяет taken_at по приоритету: EXIF → имя файла → дата создания файла → дата изменения файла.
    Если fix_from_filename=True и в БД уже есть дата — обновляет её, когда вычисленная дата отличается.
    Если no_download_exif=True — для файлов на YaDisk (disk:) не скачивать файл для EXIF, только имя файла и created/modified.
    Возвращает (updated, taken_at).
    """
    path = str(row.get("path") or "")
    name = str(row.get("name") or os.path.basename(path))
    media_type = (str(row.get("media_type") or "")).lower()
    is_image = media_type == "image"
    is_yadisk = path.startswith("disk:")
    current_taken_at = (row.get("taken_at") or "").strip() or None

    taken_at = None
    # 1) EXIF — только для изображений (для YaDisk пропускаем, если no_download_exif)
    if is_image:
        if is_yadisk and disk and not no_download_exif:
            taken_at = _get_taken_at_exif_yadisk(path, disk)
        elif not is_yadisk:
            local_path = _strip_local_prefix(path)
            if local_path and os.path.isfile(local_path):
                taken_at = _get_taken_at_exif_local(local_path)
    # 2) имя файла
    if taken_at is None:
        taken_at = _parse_date_from_filename(name)
    # 3) системные атрибуты: сначала дата создания, затем дата изменения
    if taken_at is None:
        taken_at = _get_date_from_system_attrs(
            path, is_yadisk, row.get("modified"), row.get("created"), disk
        )

    if not taken_at:
        return False, None
    # Не перезаписывать существующую дату, если не включено исправление и даты совпадают
    if current_taken_at and not fix_from_filename:
        return False, taken_at
    if current_taken_at and fix_from_filename:
        new_date = taken_at[:10] if len(taken_at) >= 10 else taken_at
        cur_date = current_taken_at[:10] if len(current_taken_at) >= 10 else current_taken_at
        if new_date == cur_date:
            return False, taken_at
    if not dry_run:
        dedup.set_taken_at_and_gps(path=path, taken_at=taken_at, gps_lat=None, gps_lon=None)
    return True, taken_at


def main() -> None:
    ap = argparse.ArgumentParser(description="Заполнение taken_at для файлов без даты съёмки")
    ap.add_argument("--dry-run", action="store_true", help="Не записывать в БД")
    ap.add_argument("--limit", type=int, default=0, help="Максимум файлов (0 = без ограничения)")
    ap.add_argument("--offset", type=int, default=0, help="Пропустить первые N файлов (для запуска частями)")
    ap.add_argument("--scope", choices=("local", "yadisk", "all"), default="all", help="Фильтр по источнику")
    ap.add_argument("--fix-from-filename", action="store_true", help="Также обрабатывать файлы с уже заполненной датой и исправлять, если по EXIF/имени файла дата другая")
    ap.add_argument("--no-download-exif", action="store_true", help="Для файлов на YaDisk (disk:) не скачивать файл для чтения EXIF — только имя файла и created/modified из БД (быстро для 27k удалённых)")
    ap.add_argument("--path", action="append", default=[], dest="paths", metavar="PATH", help="Обработать только указанный путь (можно повторять для нескольких файлов)")
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
    where = ["media_type IN ('image', 'video')", "(status IS NULL OR status != 'deleted')"]
    if not args.fix_from_filename:
        where.append("(taken_at IS NULL OR taken_at = '')")
    params = []
    paths_list = [p.strip() for p in (getattr(args, "paths", None) or []) if (p or "").strip()]
    use_path_filter = bool(paths_list)
    if use_path_filter:
        where.append("path IN (" + ",".join("?" * len(paths_list)) + ")")
        params.extend(paths_list)
    if not use_path_filter:
        if scope == "local":
            where.append("path LIKE 'local:%'")
        elif scope == "yadisk":
            where.append("path LIKE 'disk:%'")
    cur.execute(
        f"""
        SELECT id, path, name, media_type, modified, created, taken_at
        FROM files
        WHERE {' AND '.join(where)}
        ORDER BY path
        """
        + (" LIMIT ? OFFSET ?" if (args.limit or args.offset) and not use_path_filter else ""),
        params
        + (([args.limit or 999999999, args.offset] if (args.limit or args.offset) and not use_path_filter else [])),
    )
    rows = [dict(zip([c[0] for c in cur.description], r)) for r in cur.fetchall()]
    conn.close()

    total = len(rows)
    if total == 0:
        print("Нет файлов для обработки." if args.fix_from_filename else "Нет файлов с пустым taken_at.")
        return
    if args.fix_from_filename:
        print(f"Найдено файлов для проверки/исправления даты: {total}")
    else:
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
                    ok, taken = process_file(row, dedup, disk=disk, dry_run=args.dry_run, fix_from_filename=args.fix_from_filename, no_download_exif=args.no_download_exif)
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

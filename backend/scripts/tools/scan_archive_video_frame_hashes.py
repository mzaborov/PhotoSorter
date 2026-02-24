#!/usr/bin/env python3
"""
Досчитывает хеши кадров для архивных видео (только видео, фото не трогает).
Нужно для группировки «Визуально похожие видео» на странице /duplicates.

Запуск из корня репозитория (с UTF-8 для вывода кириллицы в путях):
  $env:PYTHONIOENCODING='utf-8'; python backend/scripts/tools/scan_archive_video_frame_hashes.py --limit 20
  $env:PYTHONIOENCODING='utf-8'; python backend/scripts/tools/scan_archive_video_frame_hashes.py --limit 50 --dry-run

Опции:
  --limit N    обработать не более N видео за запуск (по умолчанию 20)
  --dry-run    только вывести список путей, не качать и не писать в БД

БД: data/photosorter.db (от корня репозитория).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "backend"))

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=str(repo_root / "secrets.env"), override=False)
    load_dotenv(dotenv_path=str(repo_root / ".env"), override=False)
except Exception:
    pass

from common.db import DedupStore
from common.perceptual_hash import compute_phash_hex


def _normalize_yadisk_path(path: str) -> str:
    p = (path or "").strip()
    if p.startswith("disk:"):
        p = p[5:]
    if p and not p.startswith("/"):
        p = "/" + p
    return p


def _video_meta(local_path: str) -> dict | None:
    """Возвращает {duration_sec, times_sec} или None."""
    script = repo_root / "backend" / "scripts" / "tools" / "video_keyframes.py"
    if not script.exists():
        return None
    try:
        out = subprocess.run(
            [sys.executable, str(script), "--path", local_path, "--mode", "meta", "--samples", "3"],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(repo_root),
        )
        if out.returncode != 0:
            return None
        data = json.loads(out.stdout.strip() or "{}")
        if not data.get("ok"):
            return None
        return {
            "duration_sec": float(data.get("duration_sec") or 0),
            "times_sec": [float(x) for x in (data.get("times_sec") or [])[:3]],
        }
    except Exception:
        return None


def _extract_frame(local_path: str, t_sec: float, out_path: str, max_dim: int = 960) -> bool:
    script = repo_root / "backend" / "scripts" / "tools" / "video_keyframes.py"
    if not script.exists():
        return False
    try:
        out = subprocess.run(
            [
                sys.executable, str(script),
                "--path", local_path,
                "--mode", "extract",
                "--frame-idx", "1",
                "--t-sec", str(t_sec),
                "--out", out_path,
                "--max-dim", str(max_dim),
            ],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(repo_root),
        )
        return out.returncode == 0 and Path(out_path).is_file()
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Досчёт хешей кадров для архивных видео (похожие видео).")
    ap.add_argument("--limit", type=int, default=20, help="Макс. видео за запуск")
    ap.add_argument("--dry-run", action="store_true", help="Только вывести список, не обрабатывать")
    args = ap.parse_args()

    store = DedupStore()
    try:
        rows = store.list_archive_video_paths_without_frame_hashes(limit=args.limit)
    finally:
        store.close()

    if not rows:
        print("Нет архивных видео без хешей кадров.")
        return 0

    print(f"Найдено видео без хешей кадров: {len(rows)}")
    if args.dry_run:
        for path, file_id, dur in rows:
            print(f"  {file_id}: {path} (duration_sec={dur})")
        return 0

    disk = None
    if any((str(p or "").startswith("disk:") for p, _, _ in rows)):
        try:
            from common.yadisk_client import get_disk
            disk = get_disk()
        except Exception as e:
            print(f"Ошибка инициализации YaDisk: {e}", file=sys.stderr)
            return 1

    processed = 0
    errors = 0
    for path, file_id, duration_sec in rows:
        path = str(path or "")
        if not path:
            continue
        is_yadisk = path.startswith("disk:")
        local_path = path
        tmp_video = None
        try:
            if is_yadisk and disk:
                with tempfile.NamedTemporaryFile(prefix="ps_vf_", suffix=".mp4", delete=False) as f:
                    tmp_video = f.name
                try:
                    disk.download(_normalize_yadisk_path(path), tmp_video)
                except Exception as e:
                    print(f"  Ошибка загрузки {path}: {e}")
                    errors += 1
                    continue
                local_path = tmp_video
            elif not is_yadisk:
                # Локальный путь — проверяем существование
                if not Path(path.replace("local:", "") if path.startswith("local:") else path).is_file():
                    print(f"  Файл не найден: {path}")
                    errors += 1
                    continue
                local_path = path.replace("local:", "") if path.startswith("local:") else path

            meta = _video_meta(local_path)
            if not meta:
                print(f"  Не удалось получить длительность: {path}")
                errors += 1
                continue
            dur = meta["duration_sec"]
            times = meta.get("times_sec") or []
            if len(times) < 3:
                while len(times) < 3:
                    times.append(times[-1] if times else 0.0)
            times = times[:3]

            frames: list[tuple[float, str]] = []
            tmp_frames: list[str] = []
            try:
                for i, t in enumerate(times):
                    with tempfile.NamedTemporaryFile(prefix="ps_frame_", suffix=".jpg", delete=False) as f:
                        frame_path = f.name
                    tmp_frames.append(frame_path)
                    if not _extract_frame(local_path, t, frame_path):
                        break
                    phash_val = compute_phash_hex(frame_path)
                    if phash_val:
                        frames.append((t, phash_val))
                if len(frames) < 3:
                    print(f"  Меньше 3 кадров с pHash: {path}")
                    errors += 1
                    continue
                store2 = DedupStore()
                try:
                    store2.upsert_video_frame_hashes(file_id=file_id, duration_sec=dur, frames=frames[:3])
                    processed += 1
                    print(f"  OK file_id={file_id} {path[:60]}…")
                finally:
                    store2.close()
            finally:
                for fp in tmp_frames:
                    try:
                        os.unlink(fp)
                    except OSError:
                        pass
        except Exception as e:
            print(f"  Ошибка {path}: {e}")
            errors += 1
        finally:
            if tmp_video:
                try:
                    os.unlink(tmp_video)
                except OSError:
                    pass

    print(f"Готово. Обработано: {processed}, ошибок: {errors}")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Скачивает указанные фото из галереи fotografissimo (wfolio) и загружает на Яндекс.Диск.

Сценарий: часть файлов не долетела при «Сохранить в Яндекс.Диск» — добираем вручную скриптом.

Пример:
  python backend/scripts/tools/upload_fotografissimo_to_yadisk.py --dry-run
  python backend/scripts/tools/upload_fotografissimo_to_yadisk.py
  python backend/scripts/tools/upload_fotografissimo_to_yadisk.py --file IMG_3951.jpg --file IMG_3973.jpg

Нужен YADISK_ACCESS_TOKEN в secrets.env / .env (как у остальных скриптов PhotoSorter).
HTTP к fotografissimo идёт через curl (wfolio отклоняет urllib из Python).
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "backend"))

try:
    import yadisk.exceptions as yadisk_exceptions
except Exception:
    yadisk_exceptions = None

FOTOGRAFISSIMO_HOST = "https://fotografissimo.com"
DEFAULT_PROJECT_SLUG = "30-05-2026-artemiy-i-sabina-mdlm7l"
DEFAULT_FOLDER = "photos"
DEFAULT_TARGET_DIR = "disk:/Фото/Темка/Свадьба"

# 9 файлов, которые не загрузились через wfolio → ЯД (2026-07-02).
DEFAULT_FILES = [
    "IMG_3951.jpg",
    "IMG_3973.jpg",
    "IMG_4126.jpg",
    "IMG_4218.jpg",
    "IMG_4254.jpg",
    "IMG_4257.jpg",
    "IMG_4266.jpg",
    "IMG_4277.jpg",
    "IMG_4560.jpg",
]

_PIECE_RE = re.compile(
    r'data-gallery-title="(?P<name>[^"]+)"[^>]*data-gallery-piece-id="(?P<piece_id>\d+)"'
)
_CSRF_RE = re.compile(r'name="authenticity_token" value="([^"]+)"')
_REDIRECT_RE = re.compile(r'<turbo-stream action="redirect" target="([^"]+)"')


def _which_curl() -> str:
    for name in ("curl", "curl.exe"):
        path = shutil.which(name)
        if path:
            return path
    raise RuntimeError("curl не найден в PATH (нужен для скачивания с fotografissimo)")


class _CurlSession:
    """Минимальная HTTP-сессия через curl + cookie-jar."""

    def __init__(self, cookie_jar: Path) -> None:
        self._curl = _which_curl()
        self._cookie_jar = cookie_jar

    def _run(self, url: str, *, post_data: dict[str, str] | None = None) -> bytes:
        cmd = [
            self._curl,
            "-sL",
            "--fail",
            "-b",
            str(self._cookie_jar),
            "-c",
            str(self._cookie_jar),
        ]
        if post_data is not None:
            cmd += ["-X", "POST", "-d", urllib.parse.urlencode(post_data)]
        cmd.append(url)
        proc = subprocess.run(cmd, capture_output=True, check=False)
        if proc.returncode != 0:
            err = proc.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"curl exit {proc.returncode} for {url}: {err}")
        return proc.stdout

    def get(self, url: str) -> bytes:
        return self._run(url)

    def post(self, url: str, data: dict[str, str]) -> bytes:
        return self._run(url, post_data=data)


def _normalize_yadisk_path(path: str) -> str:
    p = (path or "").strip().replace("\\", "/")
    if p.lower().startswith("disk:"):
        p = p[5:].lstrip("/")
    if not p.startswith("/"):
        p = "/" + p
    return p


def _ensure_yadisk_folder(disk, folder_disk_path: str) -> None:
    norm = _normalize_yadisk_path(folder_disk_path)
    parts = [x for x in norm.split("/") if x]
    for i in range(1, len(parts) + 1):
        sub = "/" + "/".join(parts[:i])
        try:
            disk.mkdir(sub)
        except Exception as e:
            if yadisk_exceptions and isinstance(e, yadisk_exceptions.PathExistsError):
                pass
            elif "PathExistsError" in type(e).__name__ or "already exists" in str(e).lower():
                pass
            else:
                raise


def _fetch_piece_ids(
    session: _CurlSession,
    *,
    project_slug: str,
    folder_path: str,
) -> dict[str, str]:
    url = (
        f"{FOTOGRAFISSIMO_HOST}/disk/{project_slug}/pieces"
        f"?design_variant=storyboard&folder_path={urllib.parse.quote(folder_path)}"
    )
    html = session.get(url).decode("utf-8", errors="replace")
    return {m.group("name"): m.group("piece_id") for m in _PIECE_RE.finditer(html)}


def _download_original(
    session: _CurlSession,
    *,
    project_slug: str,
    piece_id: str,
    dest_path: Path,
) -> None:
    new_url = (
        f"{FOTOGRAFISSIMO_HOST}/disk/{project_slug}/pieces/downloads/new"
        f"?piece_id={urllib.parse.quote(piece_id)}"
    )
    modal_html = session.get(new_url).decode("utf-8", errors="replace")
    csrf_m = _CSRF_RE.search(modal_html)
    if not csrf_m:
        raise RuntimeError(f"CSRF token not found for piece_id={piece_id}")

    post_url = (
        f"{FOTOGRAFISSIMO_HOST}/disk/{project_slug}/pieces/downloads"
        f"?piece_id={urllib.parse.quote(piece_id)}"
    )
    turbo = session.post(
        post_url,
        {"authenticity_token": csrf_m.group(1), "size": "original"},
    ).decode("utf-8", errors="replace")
    redirect_m = _REDIRECT_RE.search(turbo)
    if not redirect_m:
        raise RuntimeError(f"Download redirect not found for piece_id={piece_id}")

    file_url = redirect_m.group(1)
    if file_url.startswith("//"):
        file_url = "https:" + file_url

    content = session.get(file_url)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_bytes(content)

    if content[:2] != b"\xff\xd8":
        if b"<html" in content[:200].lower() or b"turbo-stream" in content[:200].lower():
            raise RuntimeError(f"Expected image bytes, got HTML for piece_id={piece_id}")


def _yadisk_remote_path(target_dir: str, filename: str) -> str:
    base = target_dir.rstrip("/").replace("\\", "/")
    if base.lower().startswith("disk:"):
        base = base[5:].lstrip("/")
    return "/" + "/".join(x for x in (base, filename) if x)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Скачать фото из fotografissimo и загрузить на Яндекс.Диск"
    )
    parser.add_argument("--project-slug", default=DEFAULT_PROJECT_SLUG)
    parser.add_argument("--folder", default=DEFAULT_FOLDER, help="Папка в галерее (photos)")
    parser.add_argument("--target-dir", default=DEFAULT_TARGET_DIR, help="Папка на ЯД, disk:/...")
    parser.add_argument(
        "--file",
        dest="files",
        action="append",
        help="Имя файла (можно несколько раз). По умолчанию — 9 недостающих.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Скачать, но не загружать на ЯД")
    parser.add_argument("--skip-existing", action="store_true", help="Пропустить файлы, уже лежащие на ЯД")
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Сохранить копии в backend/scripts/debug/data/ (для отладки)",
    )
    args = parser.parse_args()

    filenames = args.files or list(DEFAULT_FILES)
    filenames = [f.strip() for f in filenames if f and f.strip()]
    if not filenames:
        print("Список файлов пуст.", file=sys.stderr)
        sys.exit(1)

    with tempfile.TemporaryDirectory(prefix="fotografissimo_cookies_") as cookie_dir:
        session = _CurlSession(Path(cookie_dir) / "cookies.txt")
        print(f"Загружаю список фото из галереи {args.project_slug}/{args.folder} ...")
        piece_ids = _fetch_piece_ids(
            session, project_slug=args.project_slug, folder_path=args.folder
        )
        print(f"В галерее найдено файлов: {len(piece_ids)}")

        missing_in_gallery = [f for f in filenames if f not in piece_ids]
        if missing_in_gallery:
            print("В галерее не найдены:", ", ".join(missing_in_gallery), file=sys.stderr)
            sys.exit(1)

        disk = None
        if not args.dry_run:
            from common.yadisk_client import get_disk

            disk = get_disk()
            _ensure_yadisk_folder(disk, args.target_dir)

        ok = 0
        skipped = 0
        failed: list[str] = []

        with tempfile.TemporaryDirectory(prefix="fotografissimo_yadisk_") as tmpdir:
            tmp = Path(tmpdir)
            for name in filenames:
                remote = _yadisk_remote_path(args.target_dir, name)
                piece_id = piece_ids[name]
                print(f"\n[{name}] piece_id={piece_id} → {remote}")

                if not args.dry_run and args.skip_existing and disk is not None:
                    try:
                        if disk.exists(remote):
                            print("  SKIP: уже есть на ЯД")
                            skipped += 1
                            continue
                    except Exception as e:
                        print(f"  WARN: не удалось проверить exists: {e}")

                local_path = tmp / name
                try:
                    print("  скачиваю оригинал с fotografissimo ...")
                    _download_original(
                        session,
                        project_slug=args.project_slug,
                        piece_id=piece_id,
                        dest_path=local_path,
                    )
                    size_mb = local_path.stat().st_size / (1024 * 1024)
                    print(f"  скачано: {size_mb:.1f} МБ")
                except Exception as e:
                    print(f"  ERROR download: {e}", file=sys.stderr)
                    failed.append(name)
                    continue

                if args.dry_run:
                    print("  DRY-RUN: на ЯД не загружаю")
                    ok += 1
                else:
                    try:
                        print("  загружаю на Яндекс.Диск ...")
                        disk.upload(str(local_path), remote, overwrite=False)
                        print("  OK")
                        ok += 1
                    except Exception as e:
                        if (
                            "PathExistsError" in type(e).__name__
                            or "already exists" in str(e).lower()
                            or "409" in str(e)
                        ):
                            print("  SKIP: файл уже существует на ЯД")
                            skipped += 1
                        else:
                            print(f"  ERROR upload: {e}", file=sys.stderr)
                            failed.append(name)

                if args.keep_temp and local_path.exists():
                    keep_dir = REPO_ROOT / "backend/scripts/debug/data"
                    keep_dir.mkdir(parents=True, exist_ok=True)
                    keep = keep_dir / name
                    keep.write_bytes(local_path.read_bytes())
                    print(f"  копия: {keep}")

    print("\n---")
    print(f"Готово: {ok}, пропущено: {skipped}, ошибок: {len(failed)}")
    if failed:
        print("Ошибки:", ", ".join(failed), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

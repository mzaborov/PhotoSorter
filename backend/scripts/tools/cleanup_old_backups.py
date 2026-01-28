#!/usr/bin/env python3
"""
Удаление старых бекапов из data/backups/, оставляя только указанный файл (или последний по времени).
Использование:
  python backend/scripts/tools/cleanup_old_backups.py --keep <путь_к_свежему_бекапу.db>
  python backend/scripts/tools/cleanup_old_backups.py --keep-recent 1   # оставить только 1 самый новый
"""
import sys
from pathlib import Path

# Корень проекта (backend/scripts/tools/ -> parents[3] = PhotoSorter)
project_root = Path(__file__).resolve().parents[3]
BACKUPS_DIR = project_root / "data" / "backups"


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Удалить старые бекапы, оставив только указанный или N последних.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--keep", type=Path, help="Путь к бекапу, который оставить (остальные удалить)")
    g.add_argument("--keep-recent", type=int, metavar="N", help="Оставить только N самых новых файлов")
    ap.add_argument("--dry-run", action="store_true", help="Не удалять, только показать, что будет удалено")
    args = ap.parse_args()

    if not BACKUPS_DIR.exists():
        print(f"Папка не найдена: {BACKUPS_DIR}")
        return 0

    files = sorted(BACKUPS_DIR.glob("*.db"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        print("Нет файлов .db в папке бекапов.")
        return 0

    if args.keep is not None:
        kept = Path(args.keep).resolve()
        if not kept.exists():
            print(f"Файл не найден: {kept}", file=sys.stderr)
            return 1
        if kept.parent != BACKUPS_DIR.resolve():
            print(f"Файл должен находиться в {BACKUPS_DIR}", file=sys.stderr)
            return 1
        to_keep = {kept}
    else:
        n = args.keep_recent
        to_keep = set(files[:n])

    to_delete = [p for p in files if p.resolve() not in to_keep]
    if not to_delete:
        print("Удалять нечего.")
        return 0

    if args.dry_run:
        print("Будет удалено:")
        for p in to_delete:
            print(f"  {p}")
        return 0

    for p in to_delete:
        p.unlink()
        print(f"Удалён: {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

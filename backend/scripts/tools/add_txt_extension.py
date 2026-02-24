#!/usr/bin/env python3
"""
Добавляет расширение .txt ко всем файлам в указанной папке.
Файлы, у которых расширение уже .txt (без учёта регистра), не изменяются.

Пример: History.log -> History.log.txt

Использование:
  python backend/scripts/tools/add_txt_extension.py [путь_к_папке]
  По умолчанию: C:\\tmp\\Sortet text
"""

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Добавить .txt ко всем файлам в папке")
    parser.add_argument(
        "directory",
        nargs="?",
        default=r"C:\tmp\Sortet text",
        help="Папка с файлами (по умолчанию: C:\\tmp\\Sortet text)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Только показать, что будет переименовано, не переименовывать",
    )
    args = parser.parse_args()

    root = Path(args.directory)
    if not root.is_dir():
        print(f"Ошибка: папка не найдена: {root}", file=sys.stderr)
        sys.exit(1)

    renamed = 0
    skipped_txt = 0
    errors = []

    for path in sorted(root.iterdir()):
        if not path.is_file():
            continue
        name = path.name
        if path.suffix.lower() == ".txt":
            skipped_txt += 1
            continue
        new_name = name + ".txt"
        new_path = path.parent / new_name
        if new_path.exists() and new_path != path:
            errors.append(f"Цель уже существует, пропуск: {name} -> {new_name}")
            continue
        try:
            if args.dry_run:
                print(f"[dry-run] {name} -> {new_name}")
            else:
                path.rename(new_path)
                print(f"{name} -> {new_name}")
            renamed += 1
        except OSError as e:
            errors.append(f"{name}: {e}")

    print()
    print(f"Готово. Переименовано: {renamed}, уже .txt (пропущено): {skipped_txt}.")
    if errors:
        for msg in errors:
            print(msg, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

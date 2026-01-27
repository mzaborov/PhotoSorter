#!/usr/bin/env python3
"""
Восстанавливает один файл из папки _delete обратно в корень прогона (или в указанную папку).

Использование:
  python backend/scripts/tools/restore_from_delete.py "C:\\tmp\\Photo\\_delete\\IMG-20241118-WA0021"
  python backend/scripts/tools/restore_from_delete.py "C:\\tmp\\Photo\\_delete\\IMG-20241118-WA0021.jpg" --original-dir "C:\\tmp\\Photo"
  python backend/scripts/tools/restore_from_delete.py "C:\\tmp\\Photo\\_delete\\IMG-20241118-WA0021" --pipeline-run-id 5

Путь к файлу в _delete можно указывать с расширением или без (скрипт попробует .jpg, .jpeg, .png).
По умолчанию original-dir = родитель папки _delete (т.е. корень прогона).
"""
import argparse
import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

# после insert project_root импорты backend.*
from backend.common.db import DedupStore, PipelineStore, get_connection


def _resolve_delete_file(p: str) -> str:
    """Возвращает полный путь к существующему файлу. Если передан путь без расширения — пробует .jpg, .jpeg, .png."""
    if os.path.isfile(p):
        return os.path.abspath(p)
    for ext in (".jpg", ".jpeg", ".png", ".webp", ".heic"):
        q = p + ext if not p.lower().endswith(ext) else p
        if os.path.isfile(q):
            return os.path.abspath(q)
    # как в WhatsApp — часто без расширения, но бывает .jpg
    base = os.path.dirname(p)
    name = os.path.basename(p)
    if os.path.isdir(base):
        for f in os.listdir(base):
            if f == name or f.startswith(name + ".") or (name and f.startswith(name)):
                return os.path.abspath(os.path.join(base, f))
    raise FileNotFoundError(f"Файл не найден: {p}")


def _find_pipeline_run_id_for_root(root_dir: str) -> int | None:
    """Ищет последний pipeline_run_id с root_path = root_dir или local:root_dir."""
    root_abs = os.path.abspath(root_dir)
    candidates = [root_abs, root_dir, "local:" + root_abs, "local:" + root_dir]
    conn = get_connection()
    try:
        cur = conn.cursor()
        placeholders = ",".join(["?"] * len(candidates))
        cur.execute(
            f"SELECT id FROM pipeline_runs WHERE root_path IN ({placeholders}) ORDER BY id DESC LIMIT 1",
            candidates,
        )
        row = cur.fetchone()
        return int(row["id"]) if row else None
    finally:
        conn.close()


def restore_from_delete(
    delete_path_local: str,
    original_dir: str | None = None,
    pipeline_run_id: int | None = None,
    dry_run: bool = False,
) -> None:
    delete_path_resolved = _resolve_delete_file(delete_path_local)
    delete_dir = os.path.dirname(delete_path_resolved)
    file_name = os.path.basename(delete_path_resolved)

    if original_dir is None:
        # родитель папки _delete = корень прогона
        original_dir = os.path.dirname(delete_dir)
    original_dir = os.path.abspath(original_dir)

    original_path_local = os.path.join(original_dir, file_name)
    delete_path = "local:" + delete_path_resolved
    original_path = "local:" + original_path_local
    original_parent_path = "local:" + original_dir

    if pipeline_run_id is None:
        pipeline_run_id = _find_pipeline_run_id_for_root(original_dir)
    if pipeline_run_id is None:
        raise RuntimeError(
            f"Не найден pipeline_run с root_path = {original_dir!r} или local:... . "
            "Задайте --pipeline-run-id вручную."
        )

    if dry_run:
        print(
            "[dry-run] Восстановление:",
            delete_path_resolved,
            "->",
            original_path_local,
            "pipeline_run_id=",
            pipeline_run_id,
        )
        return

    # 1) Переместить файл обратно
    if os.path.exists(original_path_local):
        raise FileExistsError(f"Цель уже существует: {original_path_local}")
    os.makedirs(original_dir, exist_ok=True)
    os.rename(delete_path_resolved, original_path_local)

    # 2) Обновить БД (path, manual labels, status)
    ds = DedupStore()
    try:
        ds.update_path(
            old_path=delete_path,
            new_path=original_path,
            new_name=file_name,
            new_parent_path=original_parent_path,
        )
        ds.update_run_manual_labels_path(
            pipeline_run_id=int(pipeline_run_id),
            old_path=delete_path,
            new_path=original_path,
        )
        cur = ds.conn.cursor()
        cur.execute("UPDATE files SET status = 'new', error = NULL WHERE path = ?", (original_path,))
        ds.conn.commit()
    finally:
        ds.close()

    print("Готово. Восстановлено:", original_path_local, "| pipeline_run_id =", pipeline_run_id)


def main() -> None:
    ap = argparse.ArgumentParser(description="Восстановить файл из _delete в корень прогона")
    ap.add_argument("delete_path", help="Путь к файлу в папке _delete (можно без расширения)")
    ap.add_argument(
        "--original-dir",
        default=None,
        help="Папка, куда вернуть файл (по умолчанию — родитель папки _delete)",
    )
    ap.add_argument("--pipeline-run-id", type=int, default=None, help="ID прогона (если не находится по root_path)")
    ap.add_argument("--dry-run", action="store_true", help="Только вывести, что будет сделано")
    args = ap.parse_args()

    restore_from_delete(
        delete_path_local=args.delete_path,
        original_dir=args.original_dir,
        pipeline_run_id=args.pipeline_run_id,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

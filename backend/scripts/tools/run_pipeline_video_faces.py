"""
Запуск прогона пайплайна с детекцией лиц по видео.

Создаёт новый run (или использует существующий), запускает local_sort с --video-samples 3.
Для несортированного видео: определяет лица на ключевых кадрах и записывает в photo_rectangles.

Примеры:
  python backend/scripts/tools/run_pipeline_video_faces.py --root C:\\tmp\\Photo
  python backend/scripts/tools/run_pipeline_video_faces.py --root C:\\tmp\\Photo --apply
  python backend/scripts/tools/run_pipeline_video_faces.py --root C:\\tmp\\Photo --pipeline-run-id 26 --video-samples 3
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main() -> int:
    p = argparse.ArgumentParser(
        description="Запуск прогона с детекцией лиц по видео (--video-samples 3)"
    )
    p.add_argument("--root", required=True, help="Корневая папка прогона (например C:\\tmp\\Photo)")
    p.add_argument("--apply", action="store_true", help="Применять перемещения файлов (по умолчанию dry-run)")
    p.add_argument("--skip-dedup", action="store_true", help="Пропустить шаг дедупликации")
    p.add_argument(
        "--pipeline-run-id",
        type=int,
        default=None,
        help="Использовать существующий run (если не указан — создаётся новый)",
    )
    p.add_argument(
        "--video-samples",
        type=int,
        default=3,
        choices=(1, 2, 3),
        help="Количество кадров на видео для детекции (по умолчанию 3)",
    )
    args = p.parse_args()

    root = str(args.root)
    if not os.path.isdir(root):
        print(f"Ошибка: папка не найдена: {root}", file=sys.stderr)
        return 1

    pipeline_run_id = args.pipeline_run_id
    if pipeline_run_id is None:
        from common.db import PipelineStore

        ps = PipelineStore()
        try:
            pipeline_run_id = ps.create_run(
                kind="local_sort",
                root_path=root,
                apply=bool(args.apply),
                skip_dedup=bool(args.skip_dedup),
            )
            print(f"Создан pipeline_run_id: {pipeline_run_id}")
        finally:
            ps.close()

    from logic.pipeline.local_sort import main as local_sort_main

    sys.argv = [
        "local_sort_by_faces",
        "--root",
        root,
        "--pipeline-run-id",
        str(pipeline_run_id),
        "--video-samples",
        str(args.video_samples),
    ]
    if args.apply:
        sys.argv.append("--apply")
    if args.skip_dedup:
        sys.argv.append("--skip-dedup")

    return local_sort_main()


if __name__ == "__main__":
    raise SystemExit(main())

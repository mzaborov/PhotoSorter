"""
ВРЕМЕННЫЙ скрипт: детекция лиц только для вкладки «Видео к разбору» (unsorted_videos).

Получает список файлов через API /api/faces/results (tab=no_faces, subtab=unsorted_videos),
запускает детекцию лиц по видео (3 кадра) и записывает в photo_rectangles.
Без перемещения файлов.

Требования: сервер должен быть запущен (uvicorn). Запускать из корня проекта.

Пример:
  python backend/scripts/tools/scan_unsorted_videos_faces.py --pipeline-run-id 26
  python backend/scripts/tools/scan_unsorted_videos_faces.py --pipeline-run-id 26 --api-base http://127.0.0.1:8000
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROJECT_ROOT = _REPO_ROOT.parent
for _p in (_PROJECT_ROOT, _REPO_ROOT):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)


def _fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return __import__("json").loads(resp.read().decode("utf-8"))


def _local_path_to_abs(path: str) -> str | None:
    r"""local:C:\... или local:/path -> абсолютный путь."""
    if not path or not path.startswith("local:"):
        return None
    p = path[6:].strip()
    if not p:
        return None
    return os.path.abspath(p)


def main() -> int:
    p = argparse.ArgumentParser(description="Детекция лиц только для «Видео к разбору»")
    p.add_argument("--pipeline-run-id", type=int, required=True, help="ID прогона")
    p.add_argument(
        "--api-base",
        default=os.getenv("API_BASE_URL", "http://127.0.0.1:8000"),
        help="Базовый URL API (по умолчанию http://127.0.0.1:8000)",
    )
    args = p.parse_args()

    run_id = int(args.pipeline_run_id)
    base = str(args.api_base).rstrip("/")

    # Собираем все пути из API (пагинация по 200)
    all_paths: list[str] = []
    total_api_items = 0
    page = 1
    while True:
        url = (
            f"{base}/api/faces/results?"
            + urllib.parse.urlencode({
                "pipeline_run_id": run_id,
                "tab": "no_faces",
                "subtab": "unsorted_videos",
                "page": page,
                "page_size": 200,
            })
        )
        try:
            data = _fetch_json(url)
        except Exception as e:
            print(f"Ошибка запроса к API: {e}", file=sys.stderr)
            return 1

        items = data.get("items") or []
        total_api_items += len(items)
        for it in items:
            fp = it.get("file_path") or it.get("path") or ""
            abs_p = _local_path_to_abs(fp)
            if abs_p and os.path.isfile(abs_p):
                all_paths.append(abs_p)

        total = int(data.get("total_count") or 0)
        print(f"Страница {page}: получено {len(items)} файлов, всего путей на диске: {len(all_paths)} (total_count={total})")
        if not items:
            break
        if len(items) < 200:
            break
        page += 1

    if not all_paths:
        print("Нет видео-файлов для обработки.")
        return 0

    # Убираем дубликаты, сохраняя порядок
    seen = set()
    paths = [p for p in all_paths if p not in seen and not seen.add(p)]
    skipped = total_api_items - len(paths)
    if skipped > 0:
        print(f"Обрабатываем {len(paths)} видео-файлов. Пропущено {skipped} (файл не найден на диске).")
    else:
        print(f"Обрабатываем {len(paths)} видео-файлов.")

    from common.db import PipelineStore
    from logic.pipeline.local_sort import scan_faces_local

    ps = PipelineStore()
    try:
        pr = ps.get_run_by_id(run_id=run_id)
        if not pr:
            print(f"Ошибка: pipeline_run_id {run_id} не найден.", file=sys.stderr)
            return 1
        face_run_id = pr.get("face_run_id")
        if not face_run_id:
            print("Ошибка: у прогона не задан face_run_id (шаг 3 не выполнялся).", file=sys.stderr)
            return 1
        root_path = str(pr.get("root_path") or "").replace("local:", "").strip()
        root_dir = os.path.abspath(root_path) if root_path and os.path.isdir(root_path) else (os.path.dirname(paths[0]) if paths else os.getcwd())
    finally:
        ps.close()

    pipeline = PipelineStore()
    try:
        run_id_face, stats = scan_faces_local(
            root_dir=root_dir,
            score_threshold=0.85,
            thumb_size=160,
            model_path=_REPO_ROOT.parent / "data" / "models" / "face_detection_yunet_2023mar.onnx",
            model_url="https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
            exclude_dir_names=(),
            exclude_paths=None,
            only_paths=paths,
            enable_qr=False,
            video_samples=3,
            video_max_dim=640,
            run_id=int(face_run_id),
            pipeline=pipeline,
            pipeline_run_id=run_id,
        )
    finally:
        pipeline.close()

    print(
        f"Готово. Обработано: {stats.images_scanned} видео, "
        f"лица: {stats.faces_found}, errors: {stats.errors}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

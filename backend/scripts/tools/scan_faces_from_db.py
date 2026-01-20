#!/usr/bin/env python3
"""
Скрипт для досканирования лиц из файлов, которые уже в БД, но не обработаны.
Обрабатывает файлы независимо от того, в каких папках они находятся (включая _faces, _no_faces и т.д.).
"""

import sys
import os
import argparse
from pathlib import Path

# Добавляем корень репозитория в путь
repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

from backend.common.db import get_connection, FaceStore
from backend.logic.pipeline.local_sort import scan_faces_local

def main():
    parser = argparse.ArgumentParser(
        description="Досканирование лиц для файлов из БД прогона"
    )
    parser.add_argument(
        "--pipeline-run-id",
        type=int,
        required=True,
        help="ID прогона pipeline"
    )
    parser.add_argument(
        "--model-path",
        default=str(Path("data") / "models" / "face_detection_yunet_2023mar.onnx"),
        help="Путь к модели детекции лиц"
    )
    parser.add_argument(
        "--model-url",
        default="https://github.com/opencv/opencv_zoo/raw/master/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        help="URL модели детекции лиц"
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.85,
        help="Порог уверенности для детекции лиц"
    )
    parser.add_argument(
        "--video-samples",
        type=int,
        default=0,
        help="Количество кадров для обработки видео (0-3)"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Применить изменения (без этого будет dry-run)"
    )
    args = parser.parse_args()
    
    conn = get_connection()
    cur = conn.cursor()
    
    # Получаем информацию о прогоне
    cur.execute("""
        SELECT id, face_run_id, root_path, status
        FROM pipeline_runs
        WHERE id = ?
    """, (args.pipeline_run_id,))
    
    run_info = cur.fetchone()
    if not run_info:
        print(f"❌ Прогон {args.pipeline_run_id} не найден!")
        return
    
    face_run_id = run_info['face_run_id']
    root_path = run_info['root_path']
    
    if not face_run_id:
        print(f"❌ У прогона {args.pipeline_run_id} нет face_run_id!")
        return
    
    # Убираем префикс "local:" если есть
    if root_path.startswith("local:"):
        root_path = root_path[6:]
    
    print("=" * 80)
    print(f"Досканирование лиц для прогона {args.pipeline_run_id}")
    print(f"  Face run ID: {face_run_id}")
    print(f"  Root path: {root_path}")
    print()
    
    # Находим файлы из БД, которые нужно обработать
    # Обрабатываем все файлы прогона, которые:
    # 1. Имеют faces_run_id = face_run_id
    # 2. ИЛИ не имеют faces_scanned_at
    # 3. ИЛИ имеют faces_scanned_at, но нет face_rectangles
    
    cur.execute("""
        SELECT DISTINCT path
        FROM files
        WHERE (
            faces_run_id = ? 
            OR (faces_run_id IS NULL AND path LIKE ?)
        )
        AND status != 'deleted'
        AND (
            faces_scanned_at IS NULL
            OR NOT EXISTS (
                SELECT 1 FROM face_rectangles 
                WHERE run_id = ? 
                  AND file_path = files.path 
                  AND COALESCE(ignore_flag, 0) = 0
            )
        )
    """, (face_run_id, f"local:{root_path}%", face_run_id))
    
    files_to_process = [row['path'] for row in cur.fetchall()]
    
    # Убираем префикс "local:" для работы с файловой системой
    files_to_process_abs = []
    for db_path in files_to_process:
        if db_path.startswith("local:"):
            abs_path = db_path[6:]
        else:
            abs_path = db_path
        
        if os.path.exists(abs_path):
            files_to_process_abs.append(abs_path)
        else:
            print(f"⚠️  Файл не найден: {abs_path}")
    
    print(f"Найдено файлов для обработки: {len(files_to_process_abs)}")
    print()
    
    if not files_to_process_abs:
        print("Нет файлов для обработки")
        return
    
    # Создаем временный список файлов и передаем его в scan_faces_local
    # Но scan_faces_local не принимает список файлов напрямую...
    # Нужно другой подход - модифицировать exclude_paths или создать обходной путь
    
    print("⚠️  ВНИМАНИЕ: Текущая версия scan_faces_local не поддерживает обработку")
    print("   конкретного списка файлов из БД. Нужна модификация кода.")
    print()
    print("Альтернативное решение:")
    print("1. Временно убрать исключаемые папки из списка исключений")
    print("2. Или обработать файлы по одному через отдельный механизм")
    print()
    print("Рекомендация: использовать --exclude-dirname для временного исключения")
    print("папок из списка исключений при запуске детекции.")

if __name__ == "__main__":
    main()

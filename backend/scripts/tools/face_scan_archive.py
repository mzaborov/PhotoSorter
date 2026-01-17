#!/usr/bin/env python3
"""
Сканирует все папки в фотоархиве (disk:/Фото) на наличие лиц.
Для каждой папки запускает face_scan.py, затем кластеризацию.
"""

import sys
import subprocess
from pathlib import Path

# Добавляем корень проекта в путь
repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

# Загружаем secrets.env/.env
try:
    from dotenv import load_dotenv  # type: ignore[import-not-found]
    load_dotenv(dotenv_path=str(repo_root / "secrets.env"), override=False)
    load_dotenv(dotenv_path=str(repo_root / ".env"), override=False)
except Exception:
    pass

from backend.common.yadisk_client import get_disk
from backend.common.db import get_connection
from backend.logic.face_recognition import cluster_face_embeddings


def list_folders_in_photo_archive(disk, root_path: str = "disk:/Фото") -> list[str]:
    """Получает список всех папок в фотоархиве."""
    folders = []
    
    # Нормализуем путь
    if root_path.startswith("disk:"):
        yadisk_path = root_path[5:]  # убираем "disk:"
    else:
        yadisk_path = root_path
    
    if not yadisk_path.startswith("/"):
        yadisk_path = "/" + yadisk_path
    
    try:
        # Получаем список элементов в корне фотоархива
        items = disk.listdir(yadisk_path)
        
        for item in items:
            if item.type == "dir":
                # Пропускаем служебные папки
                if item.name.startswith("_"):
                    continue
                
                folder_path = f"disk:{yadisk_path}/{item.name}"
                folders.append(folder_path)
        
        return sorted(folders)
    except Exception as e:
        print(f"Ошибка при получении списка папок: {e}")
        return []


def run_face_scan(folder_path: str) -> int | None:
    """Запускает face_scan.py для указанной папки."""
    script_path = Path(__file__).parent / "face_scan.py"
    venv_python = Path(__file__).resolve().parents[3] / ".venv-face" / "Scripts" / "python.exe"
    
    if not venv_python.exists():
        print(f"ERROR: .venv-face не найден: {venv_python}")
        return None
    
    cmd = [
        str(venv_python),
        str(script_path),
        "--path", folder_path,
    ]
    
    print(f"\n{'='*60}")
    print(f"Запуск сканирования для: {folder_path}")
    print(f"{'='*60}\n")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Ошибка при сканировании {folder_path}: {e}")
        return e.returncode
    except Exception as e:
        print(f"Ошибка: {e}")
        return None


def get_latest_run_id_for_path(root_path: str) -> int | None:
    """Получает ID последнего прогона для указанного пути."""
    conn = get_connection()
    cur = conn.cursor()
    
    cur.execute(
        """
        SELECT id FROM face_runs
        WHERE root_path = ? AND scope = 'yadisk'
        ORDER BY started_at DESC
        LIMIT 1
        """,
        (root_path,),
    )
    
    row = cur.fetchone()
    return row["id"] if row else None


def main() -> int:
    print("=" * 60)
    print("СКАНИРОВАНИЕ ФОТОАРХИВА НА ЛИЦА")
    print("=" * 60)
    
    # Получаем список папок
    disk = get_disk()
    folders = list_folders_in_photo_archive(disk)
    
    if not folders:
        print("Папки в фотоархиве не найдены")
        return 1
    
    print(f"\nНайдено папок: {len(folders)}")
    for folder in folders:
        print(f"  - {folder}")
    
    print(f"\nНачинаем сканирование...")
    
    # Сканируем каждую папку
    scanned_folders = []
    for i, folder in enumerate(folders, 1):
        print(f"\n[{i}/{len(folders)}] Обработка: {folder}")
        
        result = run_face_scan(folder)
        if result == 0:
            scanned_folders.append(folder)
            print(f"✓ Сканирование завершено для {folder}")
        else:
            print(f"✗ Ошибка при сканировании {folder}")
    
    # Запускаем кластеризацию для архива (используем archive_scope вместо run_id)
    print(f"\n{'='*60}")
    print("КЛАСТЕРИЗАЦИЯ АРХИВА")
    print(f"{'='*60}\n")
    
    # Для архива кластеризуем все лица вместе (archive_scope='archive')
    # не по отдельным прогонам, так как архив использует append без истории прогонов
    print("Кластеризация архива (archive_scope='archive')...")
    try:
        result = cluster_face_embeddings(
            run_id=None,
            archive_scope='archive',
            eps=0.2,
            min_samples=3,
            use_folder_context=True,
        )
        clusters_count = result.get("clusters_count", 0)
        noise_count = result.get("noise_count", 0)
        total_faces = result.get("total_faces", 0)
        print(f"✓ {clusters_count} кластеров, {noise_count} шум, всего лиц: {total_faces}")
    except Exception as e:
        print(f"✗ Ошибка кластеризации архива: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("ЗАВЕРШЕНО")
    print(f"{'='*60}")
    print(f"Обработано папок: {len(scanned_folders)}/{len(folders)}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Скачивает ONNX модель ArcFace для распознавания лиц.
Модель используется через onnxruntime (чистый Python, без компиляции C++).
"""

import os
import sys
import urllib.request
import shutil
from pathlib import Path

# Путь к директории с моделями
MODELS_DIR = Path(__file__).resolve().parents[3] / "models" / "face_recognition"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "w600k_r50.onnx"
MODEL_PATH = MODELS_DIR / MODEL_NAME


def download_via_insightface():
    """Пробует скачать модель через InsightFace (если установлен)."""
    try:
        import insightface  # type: ignore[import-untyped]
        from insightface.model_zoo import get_model
        
        print("Пробую скачать модель через InsightFace...")
        print("  (это может занять несколько минут, модель ~100MB)")
        
        # InsightFace автоматически скачает модель при первом использовании
        # Пробуем разные варианты названий моделей
        model_names = ["buffalo_l", "buffalo_s", "w600k_r50"]
        
        for model_name in model_names:
            try:
                print(f"  Пробую модель: {model_name}...")
                model = get_model(model_name)
                if model is not None:
                    print(f"  ✓ Модель {model_name} загружена")
                    break
            except Exception as e:
                print(f"  ✗ {model_name}: {e}")
                continue
        
        # Ищем скачанную модель в ~/.insightface/models/
        home = Path.home()
        possible_paths = [
            home / ".insightface" / "models" / "buffalo_l" / "w600k_r50.onnx",
            home / ".insightface" / "models" / "buffalo_s" / "w600k_r50.onnx",
            home / ".insightface" / "models" / "w600k_r50" / "w600k_r50.onnx",
        ]
        
        for insightface_model_path in possible_paths:
            if insightface_model_path.exists():
                # Копируем модель в нашу директорию
                print(f"  Найдена модель: {insightface_model_path}")
                shutil.copy2(insightface_model_path, MODEL_PATH)
                print(f"✓ Модель скопирована: {MODEL_PATH}")
                print(f"  Размер: {MODEL_PATH.stat().st_size / 1024 / 1024:.1f} MB")
                return True
        
        print("  Модель не найдена в стандартных путях InsightFace")
        return False
        
    except ImportError:
        print("  InsightFace не установлен")
        return False
    except Exception as e:
        print(f"  Ошибка: {e}")
        return False


def download_direct():
    """Пробует скачать модель напрямую с различных источников."""
    # Пробуем разные URL для скачивания
    urls = [
        # Google Drive (через gdown, если установлен)
        "https://drive.google.com/uc?id=1H37LK9FfzNz4qX9y7jS6_P9V4v8vJ8xQ",  # Пример, нужно найти реальный ID
        # Прямые ссылки на модели
        "https://github.com/deepinsight/insightface/releases/download/v0.7.3/buffalo_l.zip",  # Может быть архив
        "https://github.com/deepinsight/insightface/raw/master/python-package/insightface/model_zoo/models/w600k_r50.onnx",
    ]
    
    for url in urls:
        try:
            print(f"Пробую скачать с: {url}")
            print("  (это может занять несколько минут, модель ~100MB)")
            
            def progress_hook(count, block_size, total_size):
                if total_size > 0:
                    percent = int(count * block_size * 100 / total_size)
                    bar_length = 40
                    filled = int(bar_length * percent / 100)
                    bar = "=" * filled + "-" * (bar_length - filled)
                    print(f"\r  [{bar}] {percent}%", end="", flush=True)
            
            urllib.request.urlretrieve(url, MODEL_PATH, reporthook=progress_hook)
            print(f"\n✓ Модель успешно скачана: {MODEL_PATH}")
            print(f"  Размер: {MODEL_PATH.stat().st_size / 1024 / 1024:.1f} MB")
            return True
        except Exception as e:
            print(f"\n  ✗ Ошибка: {e}")
            continue
    
    return False


def main():
    """Основная функция."""
    print("=" * 60)
    print("Скачивание ONNX модели ArcFace для распознавания лиц")
    print("=" * 60)
    print(f"Целевой путь: {MODEL_PATH}")
    print()
    
    # Проверяем, не существует ли уже модель
    if MODEL_PATH.exists():
        size_mb = MODEL_PATH.stat().st_size / 1024 / 1024
        print(f"✓ Модель уже существует: {MODEL_PATH}")
        print(f"  Размер: {size_mb:.1f} MB")
        if size_mb < 10:
            print("  ⚠ Внимание: размер модели подозрительно мал, возможно файл повреждён")
            response = input("  Перезаписать? (y/n): ")
            if response.lower() != 'y':
                return 0
        else:
            return 0
    
    # Пробуем разные способы скачивания
    print("\nСпособ 1: Через InsightFace (рекомендуется)")
    if download_via_insightface():
        return 0
    
    print("\nСпособ 2: Прямое скачивание")
    if download_direct():
        return 0
    
    # Если ничего не помогло
    print("\n" + "=" * 60)
    print("Не удалось скачать модель автоматически.")
    print("=" * 60)
    print("\nРучная установка (выберите один из способов):")
    print()
    print("Способ А: Через InsightFace (требует установки):")
    print("   1. pip install insightface")
    print("   2. python -c \"from insightface.model_zoo import get_model; get_model('buffalo_l')\"")
    print("   3. Скопируйте модель:")
    print(f"      Из: %USERPROFILE%\\.insightface\\models\\buffalo_l\\w600k_r50.onnx")
    print(f"      В:  {MODEL_PATH}")
    print()
    print("Способ Б: Прямое скачивание:")
    print("   1. Откройте: https://github.com/deepinsight/insightface")
    print("   2. Найдите модель w600k_r50.onnx в репозитории")
    print("   3. Скачайте и сохраните в:")
    print(f"      {MODEL_PATH}")
    print()
    print("Примечание: Модель опциональна - pipeline работает и без неё.")
    print("            Embeddings будут извлекаться только если модель найдена.")
    
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nПрервано пользователем")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

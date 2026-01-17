#!/usr/bin/env python3
"""Скачивает модель через InsightFace API (правильный способ)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

target_dir = Path(__file__).resolve().parents[3] / "models" / "face_recognition"
target_dir.mkdir(parents=True, exist_ok=True)
target_path = target_dir / "w600k_r50.onnx"

print("Инициализация InsightFace для автоматической загрузки модели...")
print("Это может занять несколько минут (модель ~100MB)")

try:
    import insightface
    
    # Пробуем использовать правильный API для версии 0.2.1
    print("Пробую загрузить модель через insightface.model_zoo...")
    
    # Прямой импорт и использование
    from insightface import model_zoo
    
    # Пробуем разные варианты названий моделей
    model_names = ['buffalo_l', 'buffalo_s', 'w600k_r50', 'arcface_r50_v1']
    
    for model_name in model_names:
        try:
            print(f"Пробую модель: {model_name}...")
            # Прямой вызов через model_zoo
            model = model_zoo.get_model(model_name)
            if model is not None:
                print(f"✓ Модель {model_name} загружена")
                break
        except Exception as e:
            print(f"  {model_name}: {type(e).__name__}: {e}")
            continue
    
    # Проверяем, скачалась ли модель
    home = Path.home()
    insightface_dir = home / ".insightface"
    
    if insightface_dir.exists():
        print(f"\nПроверяю директорию: {insightface_dir}")
        onnx_files = list(insightface_dir.rglob("*.onnx"))
        if onnx_files:
            print(f"Найдено {len(onnx_files)} ONNX файлов:")
            for f in onnx_files:
                print(f"  {f}")
                if 'w600k_r50' in f.name or 'r50' in f.name.lower():
                    # Копируем найденную модель
                    import shutil
                    shutil.copy2(f, target_path)
                    print(f"\n✓ Модель скопирована: {target_path}")
                    print(f"  Размер: {target_path.stat().st_size / 1024 / 1024:.1f} MB")
                    sys.exit(0)
        else:
            print("ONNX файлы не найдены")
    else:
        print(f"Директория {insightface_dir} не существует")
    
    # Если не нашли, пробуем создать тестовое изображение и запустить распознавание
    # чтобы InsightFace скачал модель автоматически
    print("\nПробую запустить распознавание для автоматической загрузки модели...")
    try:
        import numpy as np
        from PIL import Image
        
        # Создаём тестовое изображение
        test_img = np.zeros((112, 112, 3), dtype=np.uint8)
        
        # Пробуем использовать FaceAnalysis
        try:
            from insightface.app import FaceAnalysis
            app = FaceAnalysis(name='buffalo_l')
            app.prepare(ctx_id=-1, det_size=(640, 640))
            # Запускаем на тестовом изображении
            faces = app.get(test_img)
            print("✓ Модель загружена через FaceAnalysis")
        except Exception as e:
            print(f"FaceAnalysis: {type(e).__name__}: {e}")
        
        # Проверяем снова
        if insightface_dir.exists():
            onnx_files = list(insightface_dir.rglob("w600k_r50.onnx"))
            if onnx_files:
                import shutil
                shutil.copy2(onnx_files[0], target_path)
                print(f"✓ Модель скопирована: {target_path}")
                sys.exit(0)
    except Exception as e:
        print(f"Ошибка при тестовом запуске: {type(e).__name__}: {e}")
    
    print("\n⚠ Модель не найдена автоматически")
    print("Попробуйте установить модель вручную:")
    print(f"  {target_path}")
    
except ImportError as e:
    print(f"Ошибка импорта: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Ошибка: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

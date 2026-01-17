#!/usr/bin/env python3
"""Скачивает модель ArcFace через InsightFace."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

print("Скачивание модели через InsightFace...")
print("Это может занять несколько минут (модель ~100MB)")

try:
    import insightface
    from insightface.app import FaceAnalysis
    
    # Пробуем загрузить модель через FaceAnalysis
    print("Инициализация FaceAnalysis с моделью 'buffalo_l'...")
    app = FaceAnalysis(providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    # Пробуем загрузить модель напрямую
    print("Загрузка модели через model_zoo...")
    from insightface.model_zoo import get_model
    model = get_model('buffalo_l')
    
    if model is None:
        # Пробуем другой способ
        print("Пробую альтернативный способ...")
        try:
            import insightface.model_zoo.model_zoo as mz
            # Прямой вызов загрузки
            print("Модель загружается автоматически при первом использовании")
        except Exception as e:
            print(f"Ошибка альтернативного способа: {e}")
    
    print("✓ Модель инициализирована")
    
    # Ищем скачанную модель
    home = Path.home()
    model_paths = [
        home / ".insightface" / "models" / "buffalo_l" / "w600k_r50.onnx",
        home / ".insightface" / "models" / "buffalo_s" / "w600k_r50.onnx",
    ]
    
    found_path = None
    for path in model_paths:
        if path.exists():
            found_path = path
            break
    
    if found_path:
        print(f"✓ Модель найдена: {found_path}")
        
        # Копируем в нашу директорию
        target_dir = Path(__file__).resolve().parents[3] / "models" / "face_recognition"
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / "w600k_r50.onnx"
        
        import shutil
        shutil.copy2(found_path, target_path)
        print(f"✓ Модель скопирована: {target_path}")
        print(f"  Размер: {target_path.stat().st_size / 1024 / 1024:.1f} MB")
    else:
        print("⚠ Модель загружена, но файл не найден в стандартных путях")
        print("Проверьте: ~/.insightface/models/buffalo_l/w600k_r50.onnx")
        
except ImportError as e:
    print(f"Ошибка импорта: {e}")
    print("Убедитесь, что InsightFace установлен: pip install insightface")
    sys.exit(1)
except Exception as e:
    print(f"Ошибка: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

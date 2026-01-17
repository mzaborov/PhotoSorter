#!/usr/bin/env python3
"""Скачивает модель через новую версию InsightFace."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

target_dir = Path(__file__).resolve().parents[3] / "models" / "face_recognition"
target_dir.mkdir(parents=True, exist_ok=True)
target_path = target_dir / "w600k_r50.onnx"

print("Использование новой версии InsightFace для загрузки модели...")
print("Это может занять несколько минут (модель ~100MB)")

try:
    import insightface
    print(f"Версия InsightFace: {insightface.__version__}")
    
    # Пробуем использовать API для версии 0.2.1
    print("\nПробую загрузить модель 'buffalo_l'...")
    
    # Способ через app.FaceAnalysis (версия 0.2.1)
    try:
        from insightface.app import FaceAnalysis
        
        # Инициализируем FaceAnalysis - это должно автоматически скачать модель
        print("Инициализация FaceAnalysis...")
        app = FaceAnalysis(name='buffalo_l')
        app.prepare(ctx_id=0, det_size=(640, 640))
        
        print("✓ FaceAnalysis инициализирован")
        
        # Проверяем, где сохранилась модель
        home = Path.home()
        insightface_dir = home / ".insightface"
        
        if insightface_dir.exists():
            print(f"\nПроверяю директорию: {insightface_dir}")
            # Ищем модель w600k_r50.onnx
            onnx_files = list(insightface_dir.rglob("w600k_r50.onnx"))
            if not onnx_files:
                # Ищем любые onnx файлы в buffalo_l
                onnx_files = list((insightface_dir / "models" / "buffalo_l").rglob("*.onnx"))
            
            if onnx_files:
                print(f"Найдено {len(onnx_files)} ONNX файлов:")
                for f in onnx_files:
                    print(f"  {f}")
                    size_mb = f.stat().st_size / 1024 / 1024
                    print(f"    Размер: {size_mb:.1f} MB")
                    
                    # Копируем модель если это w600k_r50 или если это единственный файл
                    if 'w600k_r50' in f.name or 'r50' in f.name.lower() or len(onnx_files) == 1:
                        import shutil
                        shutil.copy2(f, target_path)
                        print(f"\n✓ Модель скопирована: {target_path}")
                        print(f"  Размер: {target_path.stat().st_size / 1024 / 1024:.1f} MB")
                        sys.exit(0)
            else:
                print("ONNX файлы не найдены в стандартных путях")
        else:
            print(f"Директория {insightface_dir} не существует")
            
    except Exception as e:
        print(f"Ошибка FaceAnalysis: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    
    # Альтернативный способ - через model_zoo
    print("\nПробую альтернативный способ через model_zoo...")
    try:
        from insightface.model_zoo import get_model
        
        model = get_model('buffalo_l')
        if model:
            print("✓ Модель загружена через model_zoo")
            
            # Проверяем снова директорию
            home = Path.home()
            insightface_dir = home / ".insightface"
            if insightface_dir.exists():
                onnx_files = list(insightface_dir.rglob("w600k_r50.onnx"))
                if onnx_files:
                    import shutil
                    shutil.copy2(onnx_files[0], target_path)
                    print(f"✓ Модель скопирована: {target_path}")
                    sys.exit(0)
    except Exception as e:
        print(f"Ошибка model_zoo: {type(e).__name__}: {e}")
    
    print("\n⚠ Модель не найдена автоматически")
    print("Проверьте директорию вручную:")
    print(f"  {Path.home() / '.insightface'}")
    
except ImportError as e:
    print(f"Ошибка импорта: {e}")
    print("Убедитесь, что InsightFace установлен: pip install --upgrade insightface")
    sys.exit(1)
except Exception as e:
    print(f"Ошибка: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

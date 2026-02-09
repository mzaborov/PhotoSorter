#!/usr/bin/env python3
"""
Поиск модели распознавания лиц (ArcFace ONNX) в различных местах.
"""

import sys
from pathlib import Path

# Добавляем корень репозитория в путь
repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

def main():
    print("=" * 80)
    print("Поиск модели распознавания лиц (ArcFace ONNX)")
    print("=" * 80)
    print()
    
    found_models = []
    
    # 1. Проверяем локальные пути в проекте
    print("1. Проверка локальных путей в проекте:")
    model_dir = repo_root / "models" / "face_recognition"
    local_models = [
        model_dir / "w600k_r50.onnx",
        model_dir / "arcface_r50_v1.onnx",
        model_dir / "antelopev2" / "w600k_r50.onnx",
    ]
    
    for path in local_models:
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"   ✓ Найдено: {path} ({size_mb:.1f} MB)")
            found_models.append(path)
        else:
            print(f"   ✗ Не найдено: {path}")
    print()
    
    # 2. Проверяем домашнюю директорию InsightFace
    print("2. Проверка домашней директории InsightFace:")
    try:
        home = Path.home()
        insightface_dir = home / ".insightface"
        
        if insightface_dir.exists():
            print(f"   Директория существует: {insightface_dir}")
            
            # Ищем все ONNX файлы
            onnx_files = list(insightface_dir.rglob("*.onnx"))
            if onnx_files:
                for onnx_file in onnx_files[:10]:  # Показываем первые 10
                    size_mb = onnx_file.stat().st_size / (1024 * 1024)
                    print(f"   ✓ Найдено: {onnx_file} ({size_mb:.1f} MB)")
                    found_models.append(onnx_file)
            else:
                print("   ✗ ONNX файлы не найдены")
        else:
            print(f"   ✗ Директория не существует: {insightface_dir}")
    except Exception as e:
        print(f"   ✗ Ошибка: {e}")
    print()
    
    # 3. Проверяем, установлен ли InsightFace и может ли он загрузить модель
    print("3. Проверка InsightFace Python API:")
    try:
        import insightface  # type: ignore[import-untyped]
        from insightface.model_zoo import get_model
        
        print("   ✓ InsightFace установлен")
        
        # Пробуем загрузить модель
        try:
            model = get_model('buffalo_l')
            if model:
                print("   ✓ Модель 'buffalo_l' загружена через InsightFace API")
                # Проверяем, где она находится
                home = Path.home()
                insightface_dir = home / ".insightface"
                if insightface_dir.exists():
                    onnx_files = list(insightface_dir.rglob("w600k_r50.onnx"))
                    if onnx_files:
                        for onnx_file in onnx_files:
                            if onnx_file not in found_models:
                                size_mb = onnx_file.stat().st_size / (1024 * 1024)
                                print(f"   ✓ Модель находится: {onnx_file} ({size_mb:.1f} MB)")
                                found_models.append(onnx_file)
        except Exception as e:
            print(f"   ✗ Не удалось загрузить модель: {e}")
    except ImportError:
        print("   ✗ InsightFace не установлен")
    except Exception as e:
        print(f"   ✗ Ошибка: {e}")
    print()
    
    # 4. Проверяем embeddings в БД для архивных прогонов
    print("4. Проверка embeddings в БД (архивные прогоны):")
    try:
        from backend.common.db import get_connection
        
        conn = get_connection()
        cur = conn.cursor()
        
        # Проверяем архивные прогоны (лица в файлах с inventory_scope='archive')
        cur.execute("""
            SELECT 
                COUNT(*) AS total_faces,
                COUNT(pr.embedding) AS faces_with_embedding
            FROM photo_rectangles pr
            JOIN files f ON f.id = pr.file_id
            WHERE f.inventory_scope = 'archive'
              AND COALESCE(pr.ignore_flag, 0) = 0
        """)
        
        archive_stats = cur.fetchone()
        total = archive_stats['total_faces'] or 0
        with_emb = archive_stats['faces_with_embedding'] or 0
        
        print(f"   Всего лиц в архиве: {total}")
        print(f"   Лиц с embeddings: {with_emb}")
        if total > 0:
            percent = (with_emb / total) * 100
            print(f"   Процент с embeddings: {percent:.1f}%")
            
            if with_emb > 0:
                print("   ✓ Embeddings извлекались для архивных прогонов!")
                print("   → Значит модель где-то есть или использовалась")
    except Exception as e:
        print(f"   ✗ Ошибка: {e}")
    print()
    
    # Итоги
    print("=" * 80)
    print("Итоги:")
    if found_models:
        print(f"   ✓ Найдено {len(found_models)} модель(ей):")
        for model in found_models:
            print(f"      - {model}")
        print()
        print("   Рекомендация: Использовать найденную модель для извлечения embeddings")
    else:
        print("   ✗ Модели не найдены")
        print()
        print("   Рекомендация: Скачать модель через:")
        print("      python backend/scripts/setup/download_arcface_onnx.py")
    print("=" * 80)

if __name__ == "__main__":
    main()

# Face Recognition (Распознавание лиц)

## Обзор

В PhotoSorter реализовано извлечение face embeddings (векторных представлений лиц) для последующего распознавания и группировки похожих лиц.

## Архитектура

- **Face Detection**: YuNet (OpenCV) - находит лица на изображениях
- **Face Recognition**: ArcFace ONNX модель через onnxruntime - извлекает embeddings
- **Хранение**: Embeddings сохраняются в БД (таблица `face_rectangles`, колонка `embedding`)
- **Поиск похожих**: Функция `find_similar_faces()` в `FaceStore` (косинусное расстояние)

## Установка модели

### Вариант 1: Через InsightFace (рекомендуется)

1. Установите InsightFace (только для скачивания модели):
   ```bash
   pip install insightface
   ```

2. Запустите Python и скачайте модель:
   ```python
   from insightface.model_zoo import get_model
   model = get_model('buffalo_l')  # Модель автоматически скачается
   ```

3. Скопируйте ONNX файл:
   - Модель будет в `~/.insightface/models/buffalo_l/w600k_r50.onnx`
   - Скопируйте её в `models/face_recognition/w600k_r50.onnx`

### Вариант 2: Прямое скачивание

1. Перейдите на https://github.com/deepinsight/insightface
2. Найдите модель `w600k_r50.onnx` (ArcFace R50)
3. Сохраните в `models/face_recognition/w600k_r50.onnx`

### Вариант 3: Через скрипт

```bash
python backend/scripts/setup/download_arcface_onnx.py
```

## Использование

Модель загружается автоматически при запуске pipeline. Если модель не найдена, pipeline продолжит работу без извлечения embeddings (это нормально).

## API

### Извлечение embeddings

Embeddings извлекаются автоматически при детекции лиц в `scan_faces_local()`.

### Поиск похожих лиц

```python
from common.db import FaceStore

store = FaceStore()
similar = store.find_similar_faces(
    embedding_json=embedding_bytes,
    run_id=123,
    similarity_threshold=0.6,  # косинусное расстояние
    limit=10
)
```

## Формат embeddings

- **Тип**: JSON массив float32
- **Размер**: обычно 512 элементов (зависит от модели)
- **Нормализация**: L2 нормализованные векторы
- **Сравнение**: косинусное расстояние (dot product для нормализованных векторов)

## Производительность

- **Скорость**: ~50-100ms на лицо (CPU)
- **Память**: ~100MB для модели
- **Точность**: высокая (ArcFace - state-of-the-art модель)

## Примечания

- Модель работает только на CPU (через onnxruntime)
- Embeddings опциональны - pipeline работает и без них
- Для кластеризации похожих лиц используйте `find_similar_faces()` или внешние библиотеки (DBSCAN, HDBSCAN)

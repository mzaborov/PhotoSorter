#!/usr/bin/env python3
"""
Извлекает embeddings для существующих face_rectangles, у которых их ещё нет.
Перезапускает детекцию для указанных run_id или всех прогонов без embeddings.
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import cv2
import numpy as np
import json
import onnxruntime as ort
from backend.common.db import get_connection, FaceStore
from backend.common.yadisk_client import get_disk
from PIL import Image
import tempfile
import os


def extract_embedding_from_face(face_img_bgr: np.ndarray, recognition_model: dict) -> bytes | None:
    """Извлекает embedding из изображения лица."""
    try:
        sess = recognition_model["session"]
        input_name = recognition_model["input_name"]
        output_name = recognition_model["output_name"]
        input_shape = recognition_model["input_shape"]
        
        # ArcFace ожидает RGB и определённый размер (обычно 112x112)
        face_img_rgb = cv2.cvtColor(face_img_bgr, cv2.COLOR_BGR2RGB)
        
        # Ресайзим до нужного размера
        target_size = (input_shape[3], input_shape[2]) if len(input_shape) == 4 else (112, 112)
        face_resized = cv2.resize(face_img_rgb, target_size)
        
        # Нормализуем: (pixel - 127.5) / 128.0
        face_normalized = (face_resized.astype(np.float32) - 127.5) / 128.0
        
        # Транспонируем в формат [1, 3, H, W]
        face_transposed = np.transpose(face_normalized, (2, 0, 1))
        face_batch = np.expand_dims(face_transposed, axis=0)
        
        # Запускаем инференс
        outputs = sess.run([output_name], {input_name: face_batch})
        emb = outputs[0][0]
        
        # Нормализуем embedding (L2 нормализация)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        
        # Сериализуем в JSON
        return json.dumps(emb.tolist()).encode("utf-8")
    except Exception:
        return None


def load_recognition_model() -> dict | None:
    """Загружает модель распознавания."""
    try:
        repo_root = Path(__file__).resolve().parents[3]
        model_path = repo_root / "models" / "face_recognition" / "w600k_r50.onnx"
        
        if not model_path.exists():
            print(f"Модель не найдена: {model_path}")
            return None
        
        sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name
        input_shape = sess.get_inputs()[0].shape
        
        return {
            "session": sess,
            "input_name": input_name,
            "output_name": output_name,
            "input_shape": input_shape,
        }
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        return None


def process_run(run_id: int, recognition_model: dict) -> tuple[int, int]:
    """Обрабатывает один run_id: извлекает embeddings для лиц без них."""
    conn = get_connection()
    cur = conn.cursor()
    
    # Получаем все лица без embeddings для этого run_id
    cur.execute(
        """
        SELECT id, file_path, bbox_x, bbox_y, bbox_w, bbox_h, thumb_jpeg
        FROM face_rectangles
        WHERE run_id = ? 
          AND embedding IS NULL
          AND COALESCE(ignore_flag, 0) = 0
        ORDER BY id
        """,
        (run_id,),
    )
    
    faces = cur.fetchall()
    total = len(faces)
    
    if total == 0:
        return 0, 0
    
    print(f"\nRun {run_id}: найдено {total} лиц без embeddings")
    
    processed = 0
    errors = 0
    
    # Получаем информацию о прогоне
    cur.execute("SELECT scope, root_path FROM face_runs WHERE id = ?", (run_id,))
    run_info = cur.fetchone()
    if not run_info:
        print(f"  Run {run_id} не найден в БД")
        return 0, 0
    
    scope = run_info["scope"]
    root_path = run_info["root_path"]
    
    disk = None
    if scope == "yadisk":
        try:
            disk = get_disk()
        except Exception as e:
            print(f"  Ошибка подключения к YaDisk: {e}")
            return 0, 0
    
    for idx, face in enumerate(faces):
        face_id = face["id"]
        file_path = face["file_path"]
        bbox = (face["bbox_x"], face["bbox_y"], face["bbox_w"], face["bbox_h"])
        thumb_jpeg = face["thumb_jpeg"]
        
        if (idx + 1) % 10 == 0:
            print(f"  Обработано: {idx + 1}/{total}")
        
        try:
            # Пробуем извлечь embedding из thumb_jpeg (если есть)
            if thumb_jpeg:
                try:
                    # Декодируем JPEG
                    nparr = np.frombuffer(thumb_jpeg, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if img is not None:
                        # Ресайзим до нужного размера для распознавания
                        embedding = extract_embedding_from_face(img, recognition_model)
                        
                        if embedding:
                            # Сохраняем embedding в БД
                            cur.execute(
                                "UPDATE face_rectangles SET embedding = ? WHERE id = ?",
                                (embedding, face_id),
                            )
                            processed += 1
                            continue
                except Exception:
                    pass
            
            # Если не получилось из thumb, пробуем загрузить оригинальный файл
            if scope == "yadisk":
                # Загружаем файл с YaDisk
                try:
                    remote_path = file_path.replace("disk:/", "/")
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                        tmp_path = tmp.name
                    disk.download(remote_path, tmp_path)
                    
                    img_bgr = cv2.imread(tmp_path)
                    if img_bgr is not None:
                        # Кроп лица
                        x, y, w, h = bbox
                        pad = int(round(max(w, h) * 0.18))
                        x0 = max(0, x - pad)
                        y0 = max(0, y - pad)
                        x1 = min(img_bgr.shape[1], x + w + pad)
                        y1 = min(img_bgr.shape[0], y + h + pad)
                        face_crop = img_bgr[y0:y1, x0:x1]
                        
                        if face_crop.size > 0:
                            embedding = extract_embedding_from_face(face_crop, recognition_model)
                            if embedding:
                                cur.execute(
                                    "UPDATE face_rectangles SET embedding = ? WHERE id = ?",
                                    (embedding, face_id),
                                )
                                processed += 1
                                os.unlink(tmp_path)
                                continue
                    
                    os.unlink(tmp_path)
                except Exception as e:
                    errors += 1
                    if errors <= 5:
                        print(f"  Ошибка для {file_path}: {type(e).__name__}")
            elif scope == "local":
                # Локальный файл
                try:
                    local_path = file_path.replace("local:", "")
                    if os.path.exists(local_path):
                        img_bgr = cv2.imread(local_path)
                        if img_bgr is not None:
                            # Кроп лица
                            x, y, w, h = bbox
                            pad = int(round(max(w, h) * 0.18))
                            x0 = max(0, x - pad)
                            y0 = max(0, y - pad)
                            x1 = min(img_bgr.shape[1], x + w + pad)
                            y1 = min(img_bgr.shape[0], y + h + pad)
                            face_crop = img_bgr[y0:y1, x0:x1]
                            
                            if face_crop.size > 0:
                                embedding = extract_embedding_from_face(face_crop, recognition_model)
                                if embedding:
                                    cur.execute(
                                        "UPDATE face_rectangles SET embedding = ? WHERE id = ?",
                                        (embedding, face_id),
                                    )
                                    processed += 1
                                    continue
                except Exception as e:
                    errors += 1
                    if errors <= 5:
                        print(f"  Ошибка для {file_path}: {type(e).__name__}")
            
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  Ошибка для face_id={face_id}: {type(e).__name__}: {e}")
    
    conn.commit()
    print(f"  Run {run_id}: обработано {processed}/{total}, ошибок: {errors}")
    
    return processed, errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Извлекает embeddings для существующих лиц")
    parser.add_argument("--run-id", type=int, help="Обработать только указанный run_id")
    parser.add_argument("--all", action="store_true", help="Обработать все прогоны без embeddings")
    args = parser.parse_args()
    
    # Загружаем модель
    print("Загрузка модели распознавания...")
    recognition_model = load_recognition_model()
    if not recognition_model:
        print("✗ Модель не найдена. Убедитесь, что модель установлена.")
        return 1
    
    print("✓ Модель загружена")
    
    conn = get_connection()
    cur = conn.cursor()
    
    if args.run_id:
        # Обрабатываем один прогон
        run_ids = [args.run_id]
    elif args.all:
        # Находим все прогоны с лицами без embeddings
        cur.execute(
            """
            SELECT DISTINCT fr.id, fr.scope, fr.root_path, 
                   COUNT(fr2.id) as faces_without_emb
            FROM face_runs fr
            JOIN face_rectangles fr2 ON fr.id = fr2.run_id
            WHERE fr2.embedding IS NULL 
              AND COALESCE(fr2.ignore_flag, 0) = 0
            GROUP BY fr.id
            HAVING COUNT(fr2.id) > 0
            ORDER BY fr.id DESC
            """
        )
        run_ids = [row["id"] for row in cur.fetchall()]
        print(f"\nНайдено {len(run_ids)} прогонов без embeddings:")
        for row in cur.fetchall():
            print(f"  Run {row['id']}: {row['scope']} {row['root_path']} ({row['faces_without_emb']} лиц)")
    else:
        print("Укажите --run-id <id> или --all")
        return 1
    
    total_processed = 0
    total_errors = 0
    
    for run_id in run_ids:
        processed, errors = process_run(run_id, recognition_model)
        total_processed += processed
        total_errors += errors
    
    print(f"\n{'='*60}")
    print(f"Итого обработано: {total_processed} лиц")
    print(f"Ошибок: {total_errors}")
    print(f"{'='*60}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

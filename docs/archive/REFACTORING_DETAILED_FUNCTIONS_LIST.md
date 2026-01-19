# Детальный список функций и мест для изменения

## Файл: `backend/logic/face_recognition.py`

### 1. `assign_cluster_to_person()` (строки 360-418)
**Текущее использование `cluster_id`:**
- Строка 379: `DELETE FROM face_labels WHERE cluster_id = ?`
- Строка 412: `INSERT INTO face_labels (..., cluster_id, ...) VALUES (..., ?, ...)`

**Изменения:**
- Удаление: изменить на удаление по `face_rectangle_id` через JOIN с `face_cluster_members`
- Вставка: убрать `cluster_id` из INSERT

### 2. `get_cluster_info()` (строки 421-530)
**Текущее использование `cluster_id`:**
- Строка 504: `WHERE fl.cluster_id = ?`

**Изменения:**
- Переписать запрос через JOIN: `face_labels` → `face_cluster_members` → `face_clusters`

### 3. `find_closest_cluster_with_person_for_face()` (строки 647-787)
**Текущее использование `cluster_id`:**
- Строка 719: `JOIN face_labels fl ON fc.id = fl.cluster_id`

**Изменения:**
- Переписать JOIN через `face_cluster_members`: `face_labels` → `face_cluster_members` → `face_clusters`

### 4. `find_closest_cluster_with_person_for_face_by_min_distance()` (строки 787-930)
**Текущее использование `cluster_id`:**
- Строка 863: `JOIN face_labels fl ON fc.id = fl.cluster_id`

**Изменения:**
- Переписать JOIN через `face_cluster_members`

### 5. `find_small_clusters_to_merge_in_person()` (строки 1105-1340)
**Текущее использование `cluster_id`:**
- Строка 1158: `SELECT fl2.person_id FROM face_labels fl2 WHERE fl2.cluster_id = fc.id LIMIT 1`
- Строка 1172-1179: подзапросы с `fl2.cluster_id = fc.id`
- Строка 1189: подзапрос с `fl2.cluster_id = fc.id`

**Изменения:**
- Все подзапросы переписать через JOIN с `face_cluster_members`

### 6. `find_optimal_clusters_to_merge_in_person()` (строки 1342-1620)
**Текущее использование `cluster_id`:**
- Строка 1409: `SELECT fl2.person_id FROM face_labels fl2 WHERE fl2.cluster_id = fc.id LIMIT 1`
- Строка 1416-1420: подзапросы с `fl2.cluster_id = fc.id`
- Строка 1433: подзапрос с `fl2.cluster_id = fc.id`

**Изменения:**
- Все подзапросы переписать через JOIN с `face_cluster_members`

### 7. `merge_clusters()` (строки 1622-1802)
**Текущее использование `cluster_id`:**
- Строка 1682: `WHERE cluster_id = ?` (получение face_labels для source_cluster)
- Строка 1732: `WHERE face_rectangle_id = ? AND cluster_id = ? AND person_id = ?` (удаление)
- Строка 1741: `WHERE face_rectangle_id = ? AND cluster_id = ? AND person_id = ?` (проверка существования)
- Строка 1754: `WHERE face_rectangle_id = ? AND cluster_id = ? AND person_id = ?` (получение старой записи)
- Строка 1767: `INSERT INTO face_labels (..., cluster_id, ...)` (создание новой записи)
- Строка 1776: `INSERT INTO face_labels (..., cluster_id, ...)` (создание новой записи)
- Строка 1796: `DELETE FROM face_labels WHERE cluster_id = ?` (очистка)

**Изменения:**
- Получение face_labels: через JOIN с `face_cluster_members` по `face_rectangle_id`
- Удаление/проверка: убрать условие `cluster_id`, работать только с `face_rectangle_id` + `person_id`
- Вставка: убрать `cluster_id` из INSERT
- Очистка: удалять по `face_rectangle_id` через JOIN с `face_cluster_members`

## Файл: `backend/web_api/routers/face_clusters.py`

### 8. `api_face_clusters_list()` (строки ~100-200)
**Текущее использование `cluster_id`:**
- Строка 114: `SELECT p2.id FROM face_labels fl2 ... WHERE fl2.cluster_id = fc.id LIMIT 1`
- Строка 118: `SELECT p2.id FROM face_labels fl2 ... WHERE fl2.cluster_id = fc.id LIMIT 1 IS NULL`
- Строка 188: `SELECT p2.name FROM face_labels fl2 ... WHERE fl2.cluster_id = fc.id LIMIT 1`

**Изменения:**
- Все подзапросы переписать через JOIN с `face_cluster_members`

### 9. `api_face_clusters_suggest_optimal_merge()` (строки ~450-550)
**Текущее использование `cluster_id`:**
- Строка 834: `COUNT(DISTINCT fl.cluster_id) as clusters_count`

**Изменения:**
- Переписать через JOIN с `face_cluster_members`

### 10. `api_face_cluster_assign_person()` (строки ~1180-1230)
**Текущее использование `cluster_id`:**
- Строка 1193: `SELECT fl.id, fl.cluster_id`
- Строка 1221: `INSERT INTO face_labels (..., cluster_id, ...)`

**Изменения:**
- Убрать `cluster_id` из SELECT (если используется)
- Убрать `cluster_id` из INSERT

### 11. `api_person_face_assign()` (строки ~2070-2100)
**Текущее использование `cluster_id`:**
- Строка 2088: `INSERT INTO face_labels (..., cluster_id, ...)`

**Изменения:**
- Убрать `cluster_id` из INSERT

## Файл: `backend/common/db.py`

### 12. Схема БД (строки ~557-567)
**Текущее использование `cluster_id`:**
- Строка 562: `cluster_id INTEGER,` в CREATE TABLE
- Строка 586: `CREATE INDEX IF NOT EXISTS idx_face_labels_cluster ON face_labels(cluster_id);`

**Изменения:**
- Убрать колонку `cluster_id` из CREATE TABLE
- Убрать создание индекса

## Файл: `backend/scripts/tools/` (диагностические скрипты)

### 13. `count_person_faces_detailed.py`
**Текущее использование `cluster_id`:**
- Строки 104-105: `COUNT(DISTINCT fl.cluster_id)`, `GROUP_CONCAT(DISTINCT fl.cluster_id)`
- Строка 111: `HAVING cluster_count > 1`

**Изменения:**
- Переписать через JOIN с `face_cluster_members`
- Проверка дубликатов должна быть через `face_cluster_members` (должно быть 0)

### 14. `check_person_faces.py`
**Текущее использование `cluster_id`:**
- Строка 47: `COUNT(DISTINCT fl.cluster_id)`
- Строка 49: `JOIN face_cluster_members fcm ON fl.cluster_id = fcm.cluster_id` (неправильный JOIN!)
- Строки 83, 86, 90-91, 134-137: множественные использования

**Изменения:**
- Все переписать через правильный JOIN: `face_labels` → `face_cluster_members`

### 15. `check_person_stats.py`
**Текущее использование `cluster_id`:**
- Строка 61: `COUNT(DISTINCT fl.cluster_id)`
- Строка 63: `WHERE fl.person_id = ? AND fl.cluster_id IS NOT NULL`
- Строки 76, 79, 94, 97, 110, 113, 116, 118-119, 155: множественные использования

**Изменения:**
- Все переписать через JOIN с `face_cluster_members`
- Убрать фильтры `fl.cluster_id IS NOT NULL` (вместо этого LEFT JOIN)

### 16. `check_duplicate_face_labels.py`
**Текущее использование `cluster_id`:**
- Строка 47: `fl.cluster_id,`
- Строка 57: `GROUP BY fl.face_rectangle_id, fl.person_id, fl.cluster_id`

**Изменения:**
- Убрать `cluster_id` из SELECT и GROUP BY
- Если нужно проверить дубликаты по кластерам — через JOIN с `face_cluster_members`

### 17. `compare_person_snapshot.py`
**Текущее использование `cluster_id`:**
- Строка 54: `fl.cluster_id`
- Строка 100: `SELECT DISTINCT fl.face_rectangle_id as face_id, fl.cluster_id`
- Строка 229: `fl.cluster_id,`

**Изменения:**
- Добавить JOIN с `face_cluster_members` для получения `cluster_id`

### 18. `save_person_faces_snapshot.py`
**Текущее использование `cluster_id`:**
- Строка 59: `fl.cluster_id,`
- Строка 70: `ORDER BY fl.cluster_id, fl.face_rectangle_id`

**Изменения:**
- Добавить JOIN с `face_cluster_members` для получения `cluster_id`
- Изменить ORDER BY на `fcm.cluster_id, fl.face_rectangle_id`

### 19. `restore_faces_from_snapshot.py`
**Текущее использование `cluster_id`:**
- Строка 81: `INSERT INTO face_labels (..., cluster_id, ...)`

**Изменения:**
- Убрать `cluster_id` из INSERT
- Если нужно восстановить кластер — использовать отдельную логику через `face_cluster_members`

### 20. `restore_lost_faces_from_gold.py`
**Текущее использование `cluster_id`:**
- Строка 333: `SELECT fl.id, fl.cluster_id`
- Строка 393: `INSERT INTO face_labels (..., cluster_id, ...)`

**Изменения:**
- Убрать `cluster_id` из SELECT и INSERT
- Если нужно восстановить кластер — использовать отдельную логику

### 21. `check_duplicate_faces_in_clusters.py`
**Текущее использование `cluster_id`:**
- Строка 110: `WHERE fl.cluster_id = ?` (получение персоны для кластера)

**Изменения:**
- Переписать через JOIN с `face_cluster_members`

### 22. `delete_empty_clusters.py`
**Текущее использование `cluster_id`:**
- Строка 49: `DELETE FROM face_labels WHERE cluster_id IN (...)`

**Изменения:**
- Удалять по `face_rectangle_id` через JOIN с `face_cluster_members`

### 23. `delete_cluster.py`
**Текущее использование `cluster_id`:**
- Строка 26: `DELETE FROM face_labels WHERE cluster_id = ?`

**Изменения:**
- Удалять по `face_rectangle_id` через JOIN с `face_cluster_members`

### 24. `migrate_archive_faces.py`
**Текущее использование `cluster_id`:**
- Строка 109: `JOIN face_clusters fc ON fl.cluster_id = fc.id`

**Изменения:**
- Переписать через JOIN с `face_cluster_members`

### 25. `delete_old_runs.py`
**Текущее использование `cluster_id`:**
- Строка 95: `JOIN face_clusters fc ON fl.cluster_id = fc.id`

**Изменения:**
- Переписать через JOIN с `face_cluster_members`

### 26. `check_face_cluster_members_vs_labels.py`
**Текущее использование `cluster_id`:**
- Строка 55: `fl.cluster_id as label_cluster_id`
- Строка 60: `ORDER BY fl.face_rectangle_id, fl.cluster_id`

**Изменения:**
- Этот скрипт проверяет рассинхронизацию — после рефакторинга он должен показывать 0 проблем
- Можно оставить как есть для проверки или удалить после рефакторинга

## Итого

**Основные файлы:**
- `backend/logic/face_recognition.py` — 7 функций
- `backend/web_api/routers/face_clusters.py` — 4 функции/эндпойнта
- `backend/common/db.py` — схема БД
- `backend/scripts/tools/` — 13 скриптов

**Всего мест для изменения: ~50+**

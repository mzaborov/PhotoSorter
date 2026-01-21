# План: Добавление person_id в face_clusters и переименование face_labels

## Контекст и проблема

### Текущая ситуация
- **"Золотой банк"**: В фотоархиве все кластеры привязаны к персонам через `face_labels` с `source='cluster'` (верифицированные данные, потрачено много часов).
- **Логика работы**: При прогоне сортируемой папки новые лица добавляются в существующие кластеры из архива.
- **Проблема**: 
  - Связь кластера с персоной определяется через JOIN с `face_labels`, что приводит к дублированию персон в UI.
  - `face_labels` используется для двух разных целей: привязка кластера (`source='cluster'`) и ручная привязка (`source='manual'`), что создает путаницу.

### Ожидаемое поведение
- Кластер привязан к персоне **целиком** через явное поле `person_id` в `face_clusters`.
- Все лица в кластере наследуют персону через JOIN: `face_cluster_members -> face_clusters -> person_id`.
- `face_person_manual_assignments` (бывший `face_labels`) используется **только** для ручной привязки отдельных лиц (не через кластер).

## Цель
1. Добавить явную связь `person_id` в таблицу `face_clusters` для упрощения запросов и устранения дублирования персон.
2. Переименовать `face_labels` → `face_person_manual_assignments` для ясности назначения.
3. Удалить все записи с `source='cluster'` (они больше не нужны).

## План выполнения

### Этап 0: Резервное копирование БД
- [x] **ОБЯЗАТЕЛЬНО**: Создать бекап БД перед началом изменений
  - Запустить: `python backend/scripts/tools/backup_database.py`
  - Бекап сохранится в `data/backups/photosorter_backup_YYYYMMDD_HHMMSS.db`
  - ✅ **Выполнено**: Бекап создан

### Этап 1: Проверка текущих данных (SQL скрипты)
- [x] Проверить структуру таблицы `face_clusters` (есть ли `archive_scope`)
- [x] Проверить, сколько кластеров имеют привязку к персоне через `face_labels`
- [x] Проверить, есть ли кластеры с разными персонами в `face_labels`
- [x] Проверить распределение `face_labels` по `source` (manual/cluster/ai)
- [x] Проверить соответствие данных ожиданиям
  - ✅ **Выполнено**: Создан скрипт `backend/scripts/debug/check_clusters_persons_data.py`
  - ✅ **Обнаружено**: 2 проблемных кластера (528, 733) с разными персонами (Sanyok и Agata)
  - ✅ **Исправлено**: Удалены ошибочные записи для Agata из кластеров 528 и 733

### Этап 2: Добавление поля person_id в face_clusters
- [x] Добавить колонку `person_id INTEGER NULL` в таблицу `face_clusters`
- [x] Добавить индекс `idx_face_clusters_person` на `person_id`
- [x] Добавить FOREIGN KEY на `persons(id)`
  - ✅ **Выполнено**: Создан скрипт `backend/scripts/tools/migrate_add_person_id_to_face_clusters.py`
  - ✅ **Выполнено**: Обновлена схема в `backend/common/db.py`

### Этап 3: Миграция существующих данных
- [x] Создать скрипт миграции, который:
  - Для каждого кластера находит персону через `face_labels` с `source='cluster'` (любую, т.к. должна быть одна)
  - Устанавливает `face_clusters.person_id = person_id`
  - Проверяет целостность данных (все архивные кластеры должны иметь `person_id`)
  - ✅ **Выполнено**: Создан скрипт `backend/scripts/tools/migrate_person_id_to_clusters.py`
  - ✅ **Результат**: Мигрировано 10715 кластеров с привязкой к персоне

### Этап 4: Переименование face_labels → face_person_manual_assignments
- [x] Создать новую таблицу `face_person_manual_assignments` с той же структурой
- [x] Скопировать только записи с `source='manual'` (ручные привязки)
- [ ] Удалить старую таблицу `face_labels` (после завершения этапа 8)
- [x] Обновить все индексы и внешние ключи
  - ✅ **Выполнено**: Создан скрипт `backend/scripts/tools/migrate_rename_face_labels.py`
  - ✅ **Выполнено**: Обновлена схема в `backend/common/db.py`
  - ✅ **Результат**: Скопировано 5 записей с `source='manual'` + 50 с `source='restored_from_gold'`

### Этап 5: Удаление записей с source='cluster'
- [x] Удалить все записи из `face_labels` с `source='cluster'` (они больше не нужны)
- [x] Проверить, что остались только записи с `source='manual'` и `source='restored_from_gold'`
  - ✅ **Выполнено**: Создан скрипт `backend/scripts/tools/migrate_remove_cluster_source_from_face_labels.py`
  - ✅ **Результат**: Удалено 10710 записей с `source='cluster'`
  - ✅ **Выполнено**: Перенесены оставшиеся записи (55 шт.) в `face_person_manual_assignments`

### Этап 6: Обновление логики назначения кластера персоне
- [x] В `assign_cluster_to_person`:
  - Устанавливать `face_clusters.person_id = person_id`
  - **НЕ создавать** записи в `face_person_manual_assignments` (кластер привязан через `person_id`)
  - ✅ **Выполнено**: Обновлен `backend/logic/face_recognition.py`

### Этап 7: Обновление логики добавления новых лиц в кластеры
- [x] В `_try_add_to_existing_clusters`:
  - **НЕ создавать** `face_person_manual_assignments` автоматически
  - Лица наследуют персону через JOIN с `face_clusters.person_id`
  - ✅ **Выполнено**: Обновлен `backend/logic/face_recognition.py`
  - ✅ **Выполнено**: Обновлены функции поиска кластеров с персоной

### Этап 8: Обновление всех запросов и кода
- [x] Заменить все упоминания `face_labels` на `face_person_manual_assignments`:
  - [x] Все SQL-запросы в основных модулях
  - [x] Основной Python-код (`face_recognition.py`, `face_clusters.py`, `faces.py`, `gold.py`)
  - [x] Все скрипты (основные скрипты миграции обновлены, остальные - утилиты/отладка)
- [x] Заменить поиск персоны через `face_labels` на использование `face_clusters.person_id`:
  - [x] `api_faces_file_persons` (обновлен)
  - [x] `api_person_detail` (обновлен)
  - [x] `api_faces_results` (обновлен)
  - [x] `api_face_clusters_list` (обновлен)
  - [x] `merge_clusters` (обновлен)
  - [x] `find_closest_cluster_with_person_for_face` (обновлен)
  - [x] `find_small_clusters_to_merge_in_person` (обновлен)
  - [x] `find_optimal_clusters_to_merge_in_person` (обновлен)
  - [x] `api_face_rectangle_assign_person` (обновлен)
  - [x] `api_gold_import_from_file` (обновлен)
  - ✅ **Выполнено**: Все основные модули обновлены

### Этап 9: Документация
- [x] Удалить создание таблицы `face_labels` из `db.py`
- [x] Создать скрипт для удаления таблицы `face_labels`: `backend/scripts/tools/migrate_drop_face_labels_table.py`
- [ ] Обновить README с описанием архитектуры:
  - "Золотой банк" (фотоархив) — верифицированные кластеры с персоной
  - Связь кластера с персоной через `face_clusters.person_id`
  - Автоматическое наследование персоны при добавлении новых лиц
  - `face_person_manual_assignments` используется только для ручной привязки отдельных лиц (не через кластер)
- [ ] Обновить диаграммы сущностей (AS-IS и TO-BE)

## Важные принципы

1. **Не трогать "золотой банк"**: Все архивные кластеры должны сохранить свою привязку к персоне.
2. **Не создавать face_person_manual_assignments автоматически**: Лица наследуют персону через JOIN с `face_clusters.person_id`.
3. **Ясность назначения**: `face_person_manual_assignments` используется только для ручной привязки отдельных лиц (не через кластер).

## SQL скрипты для проверки данных

- `backend/scripts/debug/check_clusters_persons.sql` - проверка кластеров и персон
- `backend/scripts/debug/check_face_labels_usage.sql` - проверка использования face_labels

## Скрипты для выполнения

1. **Бекап БД**: `python backend/scripts/tools/backup_database.py` ✅
2. **Проверка данных**: `python backend/scripts/debug/check_clusters_persons_data.py` ✅
3. **Исправление проблемных кластеров**: `python backend/scripts/tools/fix_problematic_clusters.py` ✅
4. **Добавление person_id**: `python backend/scripts/tools/migrate_add_person_id_to_face_clusters.py` ✅
5. **Миграция данных**: `python backend/scripts/tools/migrate_person_id_to_clusters.py` ✅
6. **Создание новой таблицы**: `python backend/scripts/tools/migrate_rename_face_labels.py` ✅
7. **Удаление source='cluster'**: `python backend/scripts/tools/migrate_remove_cluster_source_from_face_labels.py` ✅
8. **Перенос оставшихся записей**: `python backend/scripts/tools/migrate_remaining_face_labels.py` ✅

## Прогресс выполнения

**Дата начала**: 2025-01-XX
**Текущий статус**: Этап 9 (документация) в процессе

### Выполнено:
- ✅ Этапы 0-8: Полностью завершены
- ✅ Основные модули обновлены:
  - `backend/logic/face_recognition.py` - основная логика распознавания
  - `backend/web_api/routers/face_clusters.py` - API для кластеров
  - `backend/web_api/routers/faces.py` - API для лиц
  - `backend/web_api/routers/gold.py` - API для gold данных
  - `backend/common/db.py` - схема БД (удалено создание `face_labels`)
- ✅ Создан скрипт для удаления таблицы `face_labels`: `backend/scripts/tools/migrate_drop_face_labels_table.py`

### Завершено:
- ✅ Этап 9: Обновление документации (README, диаграммы)
  - ✅ Обновлен README с описанием новой архитектуры (DD-016)
  - ✅ Обновлены диаграммы сущностей (AS-IS и TO-BE)
  - ✅ PNG диаграммы перегенерированы

### Осталось (опционально):
- [ ] Выполнить скрипт удаления таблицы `face_labels` (после тестирования)
  - Скрипт готов: `backend/scripts/tools/migrate_drop_face_labels_table.py`
  - ⚠️ **ВАЖНО**: Выполнять только после полного тестирования и убедившись, что все работает корректно

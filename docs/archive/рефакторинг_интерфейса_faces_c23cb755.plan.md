---
name: Рефакторинг интерфейса faces
overview: "Рефакторинг интерфейса faces: добавление закладки \"Лица\" с таблицами по персонам, изменение логики \"Много лиц\" для фильтрации неназначенных прямоугольников, реорганизация \"Посторонние лица\" с фильтром ручных привязок."
todos:
  - id: "1"
    content: Изменить фильтрацию 'Много лиц' - показывать только фото с неназначенными прямоугольниками
    status: completed
  - id: "2"
    content: Реорганизовать 'Посторонние лица' - переместить в конец, добавить фильтр 'только ручные/кроме ручных/все'
    status: completed
  - id: "3"
    content: Создать API endpoint /api/faces/all-persons-faces для получения данных по всем персонам
    status: completed
  - id: "4"
    content: Добавить закладку 'Лица' в UI с отображением таблиц по персонам
    status: completed
  - id: "5"
    content: Проверить и при необходимости исправить назначение группы из закладки "нет людей"
    status: completed
isProject: false
---

# Рефакторинг интерфейса faces

## Цель

Реорганизовать интерфейс faces согласно требованиям:

1. Добавить закладку "Лица" с таблицами по всем персонам (кроме Посторонних)
2. Изменить логику "Много лиц" - показывать только фото с неназначенными прямоугольниками
3. Реорганизовать "Посторонние лица" - сделать последней, добавить фильтр "только ручные/кроме ручных/все"
4. Проверить возможность назначения группы из "нет людей" (уже реализовано)

## Структура изменений

### 1. Добавление закладки "Лица"

**Файлы:**

- `backend/web_api/templates/faces.html` - добавление новой закладки и логики отображения
- `backend/web_api/routers/faces.py` - новый API endpoint для получения данных по всем персонам

**Изменения:**

- Добавить кнопку закладки "Лица" между "Животные" и "Нет людей"
- Создать API endpoint `/api/faces/all-persons-faces` который возвращает данные для всех персон (кроме Посторонних) из прогона
- Endpoint должен возвращать структуру: `{persons: [{person_id, person_name, faces: [...]}]}`
- Каждая запись в `faces` должна соответствовать данным из "Сортируется->Лица" в `person_detail.html`
- В `faces.html` добавить логику отображения: для каждой персоны показывать таблицу с файлами (аналогично person_detail, но упрощенную)
- Таблицы должны отображаться друг под другом без пагинации (все сразу)

**API endpoint структура:**

```python
@router.get("/api/faces/all-persons-faces")
def api_faces_all_persons_faces(pipeline_run_id: int) -> dict[str, Any]:
    # Возвращает данные для всех персон (кроме Посторонних) из прогона
    # Структура: {persons: [{person_id, person_name, faces: [...]}]}
    # faces - это файлы из "Сортируется->Лица" (через лица)
```

### 2. Изменение логики "Много лиц"

**Файлы:**

- `backend/web_api/routers/faces.py` - изменение фильтрации в `api_faces_results`

**Изменения:**

- В `api_faces_results` для `subtab_n == "many_faces"` добавить фильтр: показывать только файлы, где есть хотя бы один неназначенный прямоугольник
- Неназначенный прямоугольник = прямоугольник без привязки к персоне через:
  - `person_rectangle_manual_assignments` (ручные привязки)
  - `face_cluster_members` + `face_clusters` (через кластеры)
- SQL фильтр должен проверять: `EXISTS (SELECT 1 FROM photo_rectangles WHERE file_id = f.id AND run_id = ? AND NOT EXISTS (привязки к персоне))`

**Текущая логика (строка 840-841):**

```python
if tab_n == "faces" and subtab_n == "many_faces":
    sub_where = "COALESCE(faces_count, 0) >= 8"
```

**Новая логика:**

```python
if tab_n == "faces" and subtab_n == "many_faces":
    # Файлы с >= 8 лицами И с хотя бы одним неназначенным прямоугольником
    sub_where = "COALESCE(faces_count, 0) >= 8"
    # Добавить person_filter_sql для проверки неназначенных прямоугольников
```

### 3. Реорганизация "Посторонние лица"

**Файлы:**

- `backend/web_api/templates/faces.html` - перемещение закладки, добавление фильтра
- `backend/web_api/routers/faces.py` - добавление фильтрации по ручным привязкам

**Изменения:**

- Переместить закладку "Посторонние лица" в конец (после "Нет людей")
- Добавить фильтр "только ручные/кроме ручных/все" (аналогично `person_detail.html`)
- Фильтр должен работать на уровне API: параметр `manual_filter` со значениями `all|manual_only|no_manual`
- Логика фильтрации:
  - `manual_only`: только файлы, где все прямоугольники привязаны к Посторонним через `person_rectangle_manual_assignments`
  - `no_manual`: только файлы, где все прямоугольники привязаны к Посторонним через кластеры (не через ручные привязки)
  - `all`: все файлы с Посторонними (любой способ привязки)
- Условие: фото где ВСЕ прямоугольники либо посторонние, либо неназначенные (нет других персон)

**API изменения:**

- В `api_faces_results` добавить параметр `manual_filter: str | None = None`
- Для закладки "Посторонние лица" использовать `subtab_n = "outsider"` или `person_id_filter = 6` (ID персоны "Посторонний")
- Добавить SQL фильтр для проверки, что ВСЕ прямоугольники либо посторонние, либо неназначенные

### 4. Проверка назначения группы из "нет людей"

**Статус:** Уже реализовано через функцию `assignGroupToFile` в `faces.html` (строка 1095)

- Функция вызывает `/api/faces/assign-group`
- Модальное окно для выбора группы уже есть (`modalGroup`)
- Проверить, что функциональность работает корректно

## Порядок выполнения

1. **Этап 1: Изменение "Много лиц"** (самое простое)

   - Изменить фильтрацию в `api_faces_results`
   - Протестировать на реальных данных

2. **Этап 2: Реорганизация "Посторонние лица"**

   - Переместить закладку в конец
   - Добавить фильтр в UI и API
   - Протестировать фильтрацию

3. **Этап 3: Добавление закладки "Лица"**

   - Создать API endpoint
   - Добавить UI логику отображения
   - Протестировать отображение таблиц

4. **Этап 4: Проверка назначения группы**

   - Убедиться, что функциональность работает
   - При необходимости исправить баги

## Технические детали

### API endpoint для "Лица"

```python
@router.get("/api/faces/all-persons-faces")
def api_faces_all_persons_faces(pipeline_run_id: int) -> dict[str, Any]:
    """
    Возвращает данные для всех персон (кроме Посторонних) из прогона.
    Каждая персона содержит список файлов из "Сортируется->Лица" (через лица).
    """
    # 1. Получить список персон из прогона (через api_faces_persons_with_files)
    # 2. Для каждой персоны получить файлы из "Сортируется->Лица" (через person_id_filter)
    # 3. Исключить персону "Посторонний" (ID = 6)
    # 4. Вернуть структуру: {persons: [{person_id, person_name, files: [...]}]}
```

### Фильтр неназначенных прямоугольников для "Много лиц"

```sql
-- Файлы с >= 8 лицами И с хотя бы одним неназначенным прямоугольником
EXISTS (
    SELECT 1 FROM photo_rectangles fr
    WHERE fr.file_id = f.id 
      AND fr.run_id = ?
      AND COALESCE(fr.ignore_flag, 0) = 0
      AND NOT EXISTS (
          -- Нет ручной привязки
          SELECT 1 FROM person_rectangle_manual_assignments fpma
          WHERE fpma.rectangle_id = fr.id AND fpma.person_id IS NOT NULL
      )
      AND NOT EXISTS (
          -- Нет привязки через кластеры
          SELECT 1 FROM face_cluster_members fcm
          JOIN face_clusters fc ON fc.id = fcm.cluster_id
          WHERE fcm.rectangle_id = fr.id AND fc.person_id IS NOT NULL
      )
)
```

### Фильтр для "Посторонние лица"

```sql
-- Файлы где ВСЕ прямоугольники либо посторонние, либо неназначенные
-- (нет прямоугольников с другими персонами)
NOT EXISTS (
    SELECT 1 FROM photo_rectangles fr
    WHERE fr.file_id = f.id 
      AND fr.run_id = ?
      AND COALESCE(fr.ignore_flag, 0) = 0
      AND (
          -- Есть привязка к другой персоне (не Посторонний)
          EXISTS (
              SELECT 1 FROM person_rectangle_manual_assignments fpma
              WHERE fpma.rectangle_id = fr.id AND fpma.person_id != 6
          )
          OR EXISTS (
              SELECT 1 FROM face_cluster_members fcm
              JOIN face_clusters fc ON fc.id = fcm.cluster_id
              WHERE fcm.rectangle_id = fr.id AND fc.person_id != 6
          )
      )
)
```

## Файлы для изменения

1. `backend/web_api/templates/faces.html` - UI изменения
2. `backend/web_api/routers/faces.py` - API изменения
3. Возможно `backend/web_api/routers/face_clusters.py` - для получения ID персоны "Посторонний"
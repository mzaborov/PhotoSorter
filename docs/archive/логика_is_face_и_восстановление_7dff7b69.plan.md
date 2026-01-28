---
name: Логика is_face и восстановление
overview: Исправить отображение и сохранение галочки «Лицо» (is_face) в карточке фото и восстановить признак is_face=1 для прямоугольников с ручной привязкой к персоне, которые были сохранены с is_face=0.
todos:
  - id: investigate
    content: Выполнить исследование данных (скрипт ниже), зафиксировать счётчики и решить по восстановлению
    status: completed
  - id: debug-ui
    content: "Отладочный UI: список подозрительных прямоугольников, YuNet-предложения, правка is_face"
    status: pending
isProject: false
---

# План: логика галочки «Лицо» и восстановление is_face

## Исследование данных (перед действиями)

Перед правкой логики и массовым UPDATE нужно понять: сколько записей «испорчено», есть ли в БД другие следы «было лицо», и насколько старый бекап у нас есть.

### Бекапы

По результатам проверки репозитория:

- **Папка бекапов:** `data/backups/` (создаётся [backend/scripts/tools/backup_database.py](backend/scripts/tools/backup_database.py)).
- **Сейчас есть:** `photosorter_backup_20260127_112932.db` (27 января), `photosorter_backup_after_stage_1_complete_20260122_005316.db` (22 января).
- **Бекап от 22.01:** структура таблиц старая (нет `photo_rectangles`). Восстановление is_face **всё равно возможно** — по таблице **`face_cluster_members`**: `rectangle_id` из неё = прямоугольник был в кластере = был лицом (см. п. 2.2).

### Где в БД есть следы «лицо / не лицо»

Единственное явное поле — **`photo_rectangles.is_face`**. Других таблиц с историей «было ли это лицом» нет:

- **`face_cluster_members`** — удалена миграцией [migrate_face_cluster_members_to_photo_rectangles.py](backend/scripts/migration/migrate_face_cluster_members_to_photo_rectangles.py), данные перенесены в **`photo_rectangles.cluster_id`**.
- **`person_rectangle_manual_assignments`** — удалена миграцией [migrate_person_rectangle_manual_to_photo_rectangles.py](backend/scripts/migration/migrate_person_rectangle_manual_to_photo_rectangles.py), данные в **`photo_rectangles.manual_person_id`**.
- В **`face_clusters`** хранятся только кластеры и `person_id`; связи «прямоугольник → кластер» — только через **`photo_rectangles.cluster_id`**.

Косвенные признаки «это было лицо» в `photo_rectangles`:

| Признак | Смысл |
|---------|--------|
| `cluster_id IS NOT NULL` | Прямоугольник был в кластере; кластеры строятся по эмбеддингам лиц → с высокой вероятностью лицо. |
| `embedding IS NOT NULL` | Есть эмбеддинг → использовался в распознавании → лицо. |
| `is_manual = 0` и `confidence IS NOT NULL` | Детектор выдал область → по смыслу лицо. |
| `manual_person_id IS NOT NULL` | Назначена персона; по сценарию чаще всего это «лицо», а не «область без лица». |

Отдельной таблицы/поля «история is_face по времени» нет — только текущее значение в `photo_rectangles.is_face`.

### Скрипт исследования (read-only)

Нужно один раз прогнать скрипт, который **только читает** текущую БД и при необходимости бекап от 22 января, и выводит сводку. Разместить логично в [backend/scripts/debug/](backend/scripts/debug/), например `investigate_is_face_data.py`. Пример намерений запросов (без изменения БД):

1. **Текущая БД**

- Сколько записей: `manual_person_id IS NOT NULL AND (is_face IS NULL OR is_face = 0)` — считаем «испорченные».
- Из них: сколько с `cluster_id IS NOT NULL`, сколько с `embedding IS NOT NULL`, сколько с `is_manual = 0`.
- Общее число записей с `manual_person_id IS NOT NULL` и отдельно с `is_face = 1` / `is_face = 0` / `is_face IS NULL`.

2. **Бекап от 22 января** (если путь передан аргументом, например `--backup data/backups/photosorter_backup_after_stage_1_complete_20260122_005316.db`)

- Та же статистика по `photo_rectangles` в бекапе.
- Сравнение: стало ли «испорченных» больше в текущей БД, чем в бекапе, или примерно столько же — чтобы понять, была ли та же логика уже 22 января.

3. **Опционально:** для id из текущей БД с `manual_person_id IS NOT NULL AND (is_face = 0 OR is_face IS NULL)` — сколько из этих id есть в бекапе и с каким там `is_face`. Так проверяется, даёт ли бекап «лучшее» значение is_face для этих записей.

Результаты вывести в консоль и при желании — в файл в `backend/scripts/debug/data/` (например `investigate_is_face_YYYYMMDD.txt`).

После выполнения исследования — зафиксировать в плане или в отдельной заметке: объём «испорченных» записей, есть ли выигрыш от бекапа, и принимать решение по восстановлению (только правило «manual_person_id ⇒ is_face=1», или точечно из бекапа, или комбинированно с учётом cluster_id/embedding).

### Результаты исследования (2026-01-27)

Скрипт выполнен: [backend/scripts/debug/investigate_is_face_data.py](backend/scripts/debug/investigate_is_face_data.py).

**Текущая БД:**

- **Подозрительных прямоугольников (manual_person_id и is_face=0/NULL): 190** — это итого записей для разбора.
- Из них: с `cluster_id` — 0, с `embedding` — 0, с `is_manual=0` — **157**.
- Всего с manual_person_id: 963 (is_face=1: 773, is_face=0: 190, is_face NULL: 0).

**Бекап от 22.01:** структура старая (нет `photo_rectangles`). Восстановление **возможно** — по таблице **`face_cluster_members`** в бекапе: `rectangle_id` там = прямоугольник был в кластере = был лицом. Нужно доработать скрипт исследования/восстановления: при наличии в бекапе `face_cluster_members` считать пересечение подозрительных id с этими `rectangle_id` и восстанавливать is_face=1 для попавших в пересечение.

**Сколько из 190 можно обосновать по текущей БД:**

- **157** — с `is_manual=0` (созданы детектором). Изначально это были детектированные «лица» → можно уверенно выставить `is_face=1` по признаку «детектор создал область».
- **33** — с `is_manual=1` (ручные области). По полям в БД «лицо / не лицо» не различаем; варианты: YuNet по кропам или правило «manual_person_id ⇒ is_face=1».

**Вывод:** итого **190** прямоугольников. По текущей БД уверенно поправим **157** (по `is_manual=0`). Остальные **33** — по YuNet, по правилу «manual ⇒ is_face=1» или по бекапу: для id из пересечения с `face_cluster_members.rectangle_id` в бекапе выставить is_face=1.

**Рекомендуемая стратегия восстановления:** (1) из бекапа: взять `rectangle_id` из `face_cluster_members` → для подозрительных id из этого множества выставить is_face=1; (2) по текущей БД: для подозрительных с `is_manual=0` (157 записей), ещё не попавших в п.1, выставить is_face=1; (3) оставшиеся — отладочный UI + YuNet или правило «manual ⇒ is_face=1».

---

## Проблема

При назначении персоны прямоугольнику пользователь часто не обращал внимание на галочку «Лицо»; по умолчанию она оказывалась снятой (из‑за `rect.is_face === 1` при отсутствующем/нулевом значении). В результате в БД сохранялось `is_face=0` для областей, которые по смыслу были лицами. Прямоугольники с `manual_person_id` и `is_face=0` не учитываются в части логики (подсчёт лиц, кластеры, выборка «лиц на фото»), поэтому нужно: (1) скорректировать логику инициализации галочки и гарантировать наличие `is_face` в API; (2) массово вернуть `is_face=1` для таких записей (и при необходимости — восстановление из бекапа).

## 1. Исправить логику галочки «Лицо»

Правило: **если прямоугольник был с лицом — галочка включена, если без лица — снята.** Источник правды — поле `is_face`: только явное `0` значит «без лица», всё остальное (1, null, undefined) — «лицо».

### 1.1. Backend: вернуть is_face в ответ rectangles

Сейчас в ответ rectangles не попадает `is_face` в двух местах:

- **[backend/common/db.py](backend/common/db.py)** — `FaceStore.list_rectangles()` (около строк 840–849): в `SELECT` нет `fr.is_face`. Добавить в список полей `fr.is_face` (или `COALESCE(fr.is_face, 1) AS is_face` для обратной совместимости).
- **[backend/web_api/routers/faces.py](backend/web_api/routers/faces.py)** — ветка «архив» в `api_faces_rectangles` (около строк 1839–1853): в запросе к `photo_rectangles` нет `fr.is_face`. Добавить в SELECT, например `fr.is_face` или `COALESCE(fr.is_face, 1) AS is_face`.

Без этого фронт не может корректно инициализировать галочку по данным прямоугольника.

### 1.2. Frontend: инициализация галочки

В **[backend/web_api/static/photo_card.js](backend/web_api/static/photo_card.js)** в двух местах заменить правило «галочка = лицо только при is_face === 1» на «галочка = лицо, если не явное “без лица”»:

- **Подменю на плашке прямоугольника** (при выборе персоны из контекстного меню, ~строка 1539):
- Было: `const currentIsFace = rect ? (rect.is_face === 1 || rect.is_face === true) : true;`
- Стало: `const currentIsFace = rect ? (rect.is_face !== 0) : true;`
- **Модальное окно «Привязать персону»** (при открытии по выбранному прямоугольнику, ~строки 2877–2881):
- Было: `isFaceCheckbox.checked = (rect.is_face === 1 || rect.is_face === true);`
- Стало: `isFaceCheckbox.checked = (rect.is_face !== 0);`

В результате: при отсутствующем или не заданном `is_face` галочка будет включена; снятой она будет только при явном `is_face === 0`.

## 2. Восстановление is_face в БД

### 2.1. Скрипт массового исправления

Новый скрипт **[backend/scripts/tools/fix_is_face_manual_rectangles.py](backend/scripts/tools/fix_is_face_manual_rectangles.py)**:

- **Что делает:** выставляет `is_face = 1` для записей в `photo_rectangles`, у которых есть ручная привязка к персоне и при этом признак «лицо» не установлен:
- `manual_person_id IS NOT NULL AND (is_face IS NULL OR is_face = 0)`
- **Режимы:** по умолчанию `--dry-run` (только вывод, сколько строк затронуто и по каким id); с флагом `--apply` — выполнение `UPDATE`.
- **Перед --apply:** в описании скрипта и в выводе при запуске явно рекомендовать сделать бекап:  
`python backend/scripts/tools/backup_database.py`
- **Вывод в dry-run:** число затронутых строк, при необходимости — первые N примеров (id, file_id, manual_person_id) и предупреждение, что при наличии бекапов можно рассмотреть точечное восстановление из них (см. ниже).

Альтернатива имени: `restore_is_face_for_manual_rectangles.py` — на усмотрение, суть та же.

### 2.2. Восстановление из бекапа со старой схемой (через face_cluster_members)

Восстановление из бекапа **возможно**: в бекапе со старой схемой (без `photo_rectangles`) нужно смотреть таблицу **`face_cluster_members`**. В ней хранится связь прямоугольник → кластер: `rectangle_id` = id прямоугольника (в старой схеме — `face_rectangles.id`), после миграции это те же id, что в `photo_rectangles.id`. Прямоугольник, попавший в кластер, по смыслу был лицом.

**Логика восстановления:**

1. Подключиться к бекапу.
2. Если есть таблица `face_cluster_members`: выполнить `SELECT DISTINCT rectangle_id FROM face_cluster_members` — это множество id, которые в момент бекапа были в кластере (лица).
3. В текущей БД для подозрительных записей (`manual_person_id IS NOT NULL AND (is_face IS NULL OR is_face = 0)`): выставить `is_face = 1` там, где `id IN (множество rectangle_id из бекапа)`.

Так можно восстановить is_face для тех из 190 записей, чей id есть в `face_cluster_members` бекапа. Остальные — по правилу is_manual=0, YuNet или «manual ⇒ is_face=1».

**Реализация:** доработать скрипт исследования или сделать отдельный скрипт восстановления с флагом `--backup <path>`: при наличии в бекапе `face_cluster_members` — считать пересечение подозрительных id с `rectangle_id` из бекапа, в dry-run выводить объём, с `--apply` — выполнять `UPDATE photo_rectangles SET is_face = 1 WHERE id IN (...) AND manual_person_id IS NOT NULL AND (is_face IS NULL OR is_face = 0)`.

### 2.3. Вариант: прогнать YuNet (ONNX) по кропам

Идея: не считать все `manual_person_id` автоматически «лицом», а **по содержимому кропа** решить: есть ли в нём лицо. Тогда области «персона, но не лицо» (рисунок, силуэт и т.п.) останутся с `is_face=0`, а реальные лица получат `is_face=1`.

- **Где уже есть ONNX:** в проекте для детекции лиц используется **YuNet** (ONNX) в [backend/logic/pipeline/local_sort.py](backend/logic/pipeline/local_sort.py): `_create_face_detector()`, `_detect_faces(detector, img_bgr)`. Модель: `face_detection_yunet_2023mar.onnx` (OpenCV Zoo).
- **Кропы:** у прямоугольников в БД есть `photo_rectangles.thumb_jpeg` (JPEG-кроп области). Если его нет — кроп можно сгенерировать по `file_id`/path + `bbox_*`, как в [backend/web_api/routers/face_clusters.py](backend/web_api/routers/face_clusters.py) в `_generate_face_thumbnail_on_fly`.
- **Алгоритм (read-only тест или скрипт с `--apply`):**

1. Выбрать записи `photo_rectangles` с `manual_person_id IS NOT NULL AND (is_face IS NULL OR is_face = 0)` (или по решению — все с manual_person_id).
2. Для каждой записи: получить изображение кропа (из `thumb_jpeg` или собрать из файла по bbox + file_path).
3. Преобразовать в BGR (numpy/OpenCV), вызвать `_detect_faces(detector, crop_bgr)` (детектор создаётся один раз на скрипт).
4. Если найдено хотя бы одно лицо с `score >= порог` (например 0.5) → считать «лицо» → `is_face=1`, иначе → `is_face=0`.
5. При `--apply` — писать `UPDATE photo_rectangles SET is_face = ? WHERE id = ?`; при dry-run — только сводка (сколько бы поставили 1, сколько 0).

Размещение: отдельный скрипт в [backend/scripts/tools/](backend/scripts/tools/), например `fix_is_face_by_yunet_crops.py`, с зависимостью от пайплайна (import из `backend.logic.pipeline.local_sort` или вынос общей функции детекции в общий модуль). Нужен доступ к модели YuNet и к файлам изображений (для записей без thumb_jpeg); для `local:` и т.п. — учесть префиксы путей.

После исследования (п. 0) имеет смысл решить: только правило «manual ⇒ is_face=1», или дополнительно/вместо этого — прогон YuNet по кропам для объективного разбиения на лицо/не лицо.

## 5. Отладочный UI для разбора is_face

Цель: страница, где виден **список прямоугольников с возможной ошибкой** (ручная привязка к персоне, но `is_face=0` или NULL), для каждого — **предложение YuNet (лицо/не лицо)**, которое можно поменять и сохранить.

### 5.1. Сколько таких прямоугольников

Один запрос к БД:
`SELECT COUNT(*) FROM photo_rectangles WHERE manual_person_id IS NOT NULL AND (COALESCE(is_face, 0) = 0)`
Число возвращается в ответе списка и выводится на странице («Всего таких прямоугольников: N»).

### 5.2. Backend API

**1. Список подозрительных прямоугольников**

- **GET** `/api/debug/is-face-review/list`  
- Ответ: `{ "total": N, "items": [ ... ] }`.  
- Выборка: `photo_rectangles` с `manual_person_id IS NOT NULL AND (is_face IS NULL OR is_face = 0)`, JOIN `files` (path), JOIN `persons` (name).  
- Для каждого элемента: `id`, `file_id`, `path`, `bbox_x/y/w/h`, `person_name`, `is_face`, `run_id`, `pipeline_run_id` (если можно вывести по run_id), `thumb_base64`.  
- `thumb_base64`: из `thumb_jpeg`, если есть; иначе — генерация по path + bbox (повтор использования логики из [face_clusters._generate_face_thumbnail_on_fly](backend/web_api/routers/face_clusters.py) или общий хелпер).  
- Опционально: `?limit=` и `?offset=` для постраничной загрузки.

**2. Предложение YuNet по кропам**

- **POST** `/api/debug/is-face-review/compute-yunet`  
- Тело: `{ "rectangle_ids": [1, 2, 3] } `или `{}` — тогда по всем из текущего списка подозрительных.  
- Для каждого id: загрузить кроп (из `thumb_jpeg` или path+bbox), преобразовать в BGR, вызвать YuNet (`_create_face_detector` + `_detect_faces` из [local_sort](backend/logic/pipeline/local_sort.py)). Если есть хотя бы одно лицо с score ≥ порог → предложение 1, иначе 0.  
- Ответ: `{ "suggestions": { "1": 1, "2": 0, "3": 1 } }` (id → 1=лицо, 0=не лицо).  
- Реализация: ленивый import из `backend.logic.pipeline.local_sort` только внутри этого обработчика, чтобы не тянуть пайплайн в каждый запрос.

**3. Сохранение is_face**

- Используется существующий **POST** `/api/faces/rectangle/update` ([faces.py](backend/web_api/routers/faces.py) ~2331): в теле передаются `rectangle_id`, `is_face`, и при необходимости `file_id` или `path`, а для сортируемых — `pipeline_run_id`. Список из п.1 уже содержит `file_id`, `path`, `pipeline_run_id`, поэтому UI просто собирает payload и вызывает этот эндпоинт.

### 5.3. Страница отладки

- **GET** `/debug/is-face-review` — HTML-страница (шаблон в духе [debug_photo_card.html](backend/web_api/templates/debug_photo_card.html)).  
- Размещение роута: [backend/web_api/routers/faces.py](backend/web_api/routers/faces.py) (рядом с `/debug/photo-card`).  
- Содержимое страницы:

1. Заголовок: «Прямоугольники с ручной привязкой без лица (is_face=0/NULL)».
2. Блок «Всего таких прямоугольников: **N**» (N из `list.total`).
3. Кнопка «Прогнать YuNet по всем» → POST `compute-yunet` с id из текущего списка → в ответе подставляем предложения в таблицу.
4. Таблица (или карточки) по одному ряду на прямоугольник:

- превью (thumb_base64);
- путь к файлу (path);
- персона (person_name);
- текущий is_face (Лицо / Не лицо);
- столбец «YuNet»: после прогона — «Лицо»/«Не лицо» (из `suggestions[id]`);
- «Решение»: выпадающий список или переключатель «Лицо» / «Не лицо» (по умолчанию — предложение YuNet, если есть, иначе текущий is_face);
- кнопка «Сохранить» по строке → POST `/api/faces/rectangle/update` с `rectangle_id`, `is_face` из «Решение», `file_id`/`path`/`pipeline_run_id` из строки.

5. Опционально: кнопка «Применить YuNet ко всем» — для каждой строки вызвать update с `is_face = suggestions[id]` без явного ручного выбора.

### 5.4. Файлы

- Роут и оба API: [backend/web_api/routers/faces.py](backend/web_api/routers/faces.py).
- Шаблон: новый [backend/web_api/templates/debug_is_face_review.html](backend/web_api/templates/debug_is_face_review.html) (разметка, подключение статики, вызовы `/api/debug/is-face-review/list`, `/api/debug/is-face-review/compute-yunet`, `/api/faces/rectangle/update`).
- Логика YuNet: вызов существующих `_create_face_detector` и `_detect_faces` из [backend/logic/pipeline/local_sort.py](backend/logic/pipeline/local_sort.py); при отсутствии модели — в ответе compute-yunet возвращать ошибку или «не вычислено» по тем id, где не удалось прогнать.

### 5.5. Порядок по отношению к остальному плану

Отладочный UI можно делать до или после шага 0 (исследование): он даёт и счётчик («сколько таких прямоугольников»), и поштучный разбор с YuNet. Имеет смысл ввести его после исследования (чтобы уже понимать порядок величины N) или параллельно с ним — как удобнее для вашего потока.

## 3. Порядок внедрения

0. **Исследование данных:** выполнено ([backend/scripts/debug/investigate_is_face_data.py](backend/scripts/debug/investigate_is_face_data.py)); 190 подозрительных, 157 с is_manual=0, 33 с is_manual=1. При необходимости — доработать скрипт: при бекапе со старой схемой запрашивать `face_cluster_members.rectangle_id` и считать пересечение с подозрительными id.
1. Backend: добавить `is_face` в [backend/common/db.py](backend/common/db.py) (`list_rectangles`) и в [backend/web_api/routers/faces.py](backend/web_api/routers/faces.py) (архивная ветка `api_faces_rectangles`).
2. Frontend: поправить инициализацию галочки в [backend/web_api/static/photo_card.js](backend/web_api/static/photo_card.js) (подменю и модалка).
3. Скрипт восстановления: по стратегии из «Результаты исследования» — (1) из бекапа по `face_cluster_members.rectangle_id` (п. 2.2); (2) по текущей БД для подозрительных с `is_manual=0`; (3) оставшиеся — YuNet или правило «manual ⇒ is_face=1». Dry-run, затем при необходимости бекап и `--apply`.
4. Реализовать восстановление из бекапа (п. 2.2): скрипт с `--backup <path>` запрашивает в бекапе `SELECT DISTINCT rectangle_id FROM face_cluster_members`, для подозрительных id из этого множества выполняет `UPDATE photo_rectangles SET is_face = 1 WHERE id IN (...)`.
5. **Отладочный UI** (п. 5): страница `/debug/is-face-review`, API списка и compute-yunet, шаблон с таблицей, счётчиком, YuNet-предложениями и сохранением is_face через существующий `rectangle/update`. Можно делать после исследования или параллельно — UI сразу покажет «сколько таких прямоугольников» и даст поштучный разбор.

## 4. Замечания

- Существующий [backend/scripts/tools/fix_is_face_for_persons.py](backend/scripts/tools/fix_is_face_for_persons.py) делает обратное (ставит is_face=0 для выбранных файлов/персон) и опирается на старую таблицу `person_rectangle_manual_assignments`. Его не трогаем; новый скрипт работает только с `photo_rectangles.manual_person_id`.
- В `get_file_person_bindings` в db.py ветка по `manual_person_id` не фильтрует по `is_face`, поэтому привязки уже учитываются; но в других местах (например, face_clusters, подсчёт лиц) фильтр `is_face = 1` есть — после восстановления is_face все эти учёты начнут учитывать исправленные прямоугольники.
- После правок логики и скрипта восстановления имеет смысл добавить в History.log запись о внесённых изменениях и о запуске восстановления (если оно выполнялось).
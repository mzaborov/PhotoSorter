# Описание изменений: поворот фото в архиве, EXIF-даты, карточка поездки

## Обзор

Добавлена поддержка поворота фото для файлов на Яндекс.Диске (архив), запись даты в EXIF при повороте (в т.ч. из БД), восстановление даты модификации для локальных файлов, исправление отображения повёрнутых фото в плитке/карточке (обход кэша превью Яндекса), оптимизация перезагрузки страницы поездки при закрытии карточки.

---

## 1. Поворот фото для файлов из архива (Я.Диск)

**Файлы:** `backend/web_api/routers/faces.py`, `backend/web_api/static/photo_card.js`

- **Бэкенд:** API `/api/faces/rotate-photo` принимает пути как `local:`, так и `disk:`.
  - Для `disk:`: скачивание файла с Я.Диска → поворот пикселей (PIL) → загрузка обратно с `overwrite=True`.
  - Вспомогательная функция `_yadisk_api_path(path)` приводит путь к формату API Я.Диска (без префикса `disk:`, с ведущим `/`).
  - Явная обработка ошибок: при сбое `download`/`upload` возвращается HTTP 502 с текстом ошибки.

- **Фронт:** Кнопки поворота отображаются для изображений с путём `disk:` и `local:` (переменная `isRotatableImage`). В `handleRotatePhoto` разрешён путь `disk:` и отправка `path` в API.

---

## 2. EXIF при повороте: дата и ориентация

**Файлы:** `backend/web_api/routers/faces.py`, `requirements.txt`

- В **requirements.txt** добавлена зависимость **piexif==1.1.3**.

- При сохранении после поворота (и для локальных, и для Я.Диска) в EXIF записываются:
  - **Orientation = 1** (нормальная ориентация после поворота пикселей).
  - **DateTime / DateTimeOriginal / DateTimeDigitized** — по приоритету:
    1. Исходный EXIF файла;
    2. Дата модификации файла (для `local:`);
    3. **Дата из БД** (`files.taken_at`) — приоритетнее даты с Я.Диска после повторного поворота;
    4. Дата из метаданных файла на Я.Диске (`get_meta` → modified/created).

- Функции: `_exif_date_from_mtime`, `_exif_date_from_iso`, `_build_exif_after_rotate`. Для Я.Диска перед поворотом вызывается `get_meta` для получения `modified`/`created` как fallback для даты в EXIF.

---

## 3. Восстановление даты модификации (локальные файлы)

**Файл:** `backend/web_api/routers/faces.py`

- Для путей `local:` перед записью повёрнутого файла сохраняются `st_atime` и `st_mtime`.
- После `os.replace(temp_path, abs_path)` вызывается `os.utime(abs_path, (saved_atime, saved_mtime))`, чтобы не менять «Изменён» в свойствах файла.

---

## 4. Инвалидация кэша превью и отображение повёрнутых фото

**Файл:** `backend/web_api/main.py`

- **`_preview_cache_invalidate(path)`:** удаляет из кэша превью все записи по данному `path` (все размеры). Вызывается из обработчика поворота после успешной загрузки на Я.Диск. Одновременно путь помечается как «недавно повёрнутый».

- **«Недавно повёрнутые» пути:** словарь `_PREVIEW_RECENTLY_ROTATED` (path → timestamp), TTL 10 минут. Для таких путей при запросе превью отдаётся **ссылка на скачивание файла** (`get_download_link`), а не превью-URL Яндекса, чтобы в плитке и карточке отображалось актуальное (повёрнутое) изображение, а не закэшированное превью «на боку».

- **Параметр `_bust`** у `/api/yadisk/preview-image`: при наличии не используется кэш превью; для запроса с `_bust` (или для пути из «недавно повёрнутых») отдаётся redirect на download-ссылку с заголовками `Cache-Control: no-store`, `Pragma: no-cache`.

- **Фронт** (`photo_card.js`): после успешного поворота для `disk:` к URL изображения добавляется `_bust=Date.now()` при обновлении картинки в карточке.

---

## 5. Карточка поездки: перезагрузка только при изменениях

**Файлы:** `backend/web_api/static/photo_card.js`, `backend/web_api/static/video_card.js`, `backend/web_api/templates/trip_detail.html`

- **Флаг `cardHadChanges`** в состоянии карточки (photo и video). Устанавливается в `true` при:
  - повороте фото;
  - удалении файла;
  - назначении группы;
  - снятии/смене прямой привязки к персоне;
  - обновлении прямоугольника (персона, bbox, тип, «Посторонний», «Кот», удаление прямоугольника);
  - в video_card: сохранение кадров, назначение персоны/группы, «Кот», «Посторонний» и т.п.

- **Вызов `on_close`** при закрытии карточки передаёт аргумент: `on_close({ reload: true })` или `on_close({ reload: currentState.cardHadChanges })`. После удаления файла всегда `on_close({ reload: true })`.

- **trip_detail.html:** обработчик изменён с `on_close: function() { location.reload(); }` на `on_close: function(opts) { if (opts && opts.reload) location.reload(); }`. Полная перезагрузка страницы поездки выполняется только если в карточке были изменения; при простом закрытии без действий перезагрузки нет.

---

## Затронутые файлы (только эти изменения)

| Файл | Изменения |
|------|-----------|
| `backend/web_api/main.py` | Кэш превью: инвалидация, «недавно повёрнутые», _bust, отдача download-ссылки |
| `backend/web_api/routers/faces.py` | Поворот для disk:, EXIF (дата из БД/ЯД/mtime), восстановление mtime, вызов инвалидации кэша |
| `backend/web_api/static/photo_card.js` | Кнопки поворота для disk:, cardHadChanges, on_close({ reload }) |
| `backend/web_api/static/video_card.js` | cardHadChanges, on_close({ reload }) |
| `backend/web_api/templates/trip_detail.html` | on_close только с opts.reload → location.reload() |
| `requirements.txt` | piexif==1.1.3 |

---

## Полный diff

```bash
git diff backend/web_api/main.py backend/web_api/routers/faces.py \
  backend/web_api/static/photo_card.js backend/web_api/static/video_card.js \
  backend/web_api/templates/trip_detail.html requirements.txt
```

---
name: Отладочная страница is-face-review для подозрительных кропов
overview: Минимальный вариант — только показать кропы по селекту «manual_person_id и is_face=0/NULL» (сейчас 33). Один API списка id, одна страница с сеткой превью через существующий /api/face-rectangles/{id}/thumbnail.
todos:
  - id: api-list
    content: "API GET /api/debug/is-face-review/list в faces.py"
    status: completed
  - id: route-page
    content: "Роут GET /debug/is-face-review в faces.py"
    status: completed
  - id: template
    content: "Шаблон debug_is_face_review.html — сетка кропов по id из списка"
    status: completed
isProject: false
archived: "2026-01-27 — интерфейс реализован, проверка показала что все 33 кропа — не лица; интерфейс удалён по решению пользователя"
---

# Отладочная страница is-face-review (только показать 33 кропа)

**Архив:** интерфейс был сделан, по просмотру 33 кропов подтверждено, что это не лица; страница и API удалены, план перенесён в архив.

---

## Цель (на момент плана)

**Малой кровью:** одна страница, где видна сетка кропов по выборке «manual_person_id IS NOT NULL AND (is_face IS NULL OR is_face = 0)» (сейчас 33 записи). Без кнопок «Сохранить», без YuNet.

## 1. API списка id

**Роут:** `GET /api/debug/is-face-review/list`  
**Файл:** [backend/web_api/routers/faces.py](backend/web_api/routers/faces.py).

**Логика:** один запрос к БД:

```sql
SELECT pr.id, f.path, p.name AS person_name
FROM photo_rectangles pr
JOIN files f ON f.id = pr.file_id
LEFT JOIN persons p ON p.id = pr.manual_person_id
WHERE pr.manual_person_id IS NOT NULL AND (pr.is_face IS NULL OR pr.is_face = 0)
ORDER BY pr.id
```

**Ответ:** `{ "total": N, "items": [ { "id", "path", "person_name" }, ... ] }`

Превью **не** отдаём — страница берёт их по уже существующему `/api/face-rectangles/{rectangle_id}/thumbnail` из [face_clusters.py](backend/web_api/routers/face_clusters.py) (стр. 918). Никакой новой логики кропов, никаких импортов между роутерами.

## 2. Страница отладки

**Роут:** `GET /debug/is-face-review`  
**Файл:** [backend/web_api/routers/faces.py](backend/web_api/routers/faces.py) (по аналогии с `/debug/photo-card`).

**Шаблон:** [backend/web_api/templates/debug_is_face_review.html](backend/web_api/templates/debug_is_face_review.html).

**Содержимое (минимум):**

1. Заголовок: «Подозрительные кропы (manual_person_id и is_face=0/NULL)».
2. Счётчик: «Всего: **N**».
3. Сетка карточек: для каждого `item` из списка — блок с превью и подписью (path или person_name). Превью: JS вызывает `fetch("/api/face-rectangles/" + item.id + "/thumbnail")`, из ответа берёт `thumb_jpeg_base64`, вешает на `<img src="data:image/jpeg;base64,...">`. Как уже сделано в [persons_list.html](backend/web_api/templates/persons_list.html) (стр. 110–123): `getFaceThumbnail(faceId)` → `data:image/jpeg;base64,${data.thumb_jpeg_base64}`.

Без кнопки «Сохранить», без YuNet, без пагинации (N обычно 33).

## 3. Объём работ

- **faces.py:** два роута — `GET /api/debug/is-face-review/list` (один SELECT + формирование JSON) и `GET /debug/is-face-review` (TemplateResponse).
- **debug_is_face_review.html:** один HTML-файл с небольшим inline-скриптом: fetch list → для каждого item создать div с img (загрузка через `/api/face-rectangles/{id}/thumbnail`) и подпись path/person_name. Стили — в духе [debug_photo_card.html](backend/web_api/templates/debug_photo_card.html).

## 4. Файлы

| Действие | Файл |
|----------|------|
| Добавить 2 роута | [backend/web_api/routers/faces.py](backend/web_api/routers/faces.py) |
| Создать шаблон | [backend/web_api/templates/debug_is_face_review.html](backend/web_api/templates/debug_is_face_review.html) |

Кнопку «Сохранить», YuNet и пагинацию в этот план не включаем — при необходимости добавить отдельно.

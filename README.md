# PhotoSorter
Сортировщик фотографий на Яндекс Диске

## Web интерфейс (просмотр папок)

Установка зависимостей:

```bash
pip install -r requirements.txt
```

Запуск сервера (локально, рекомендуемый — без `--reload`):

```bash
uvicorn --app-dir . app.main:app --port 8000
```

Открыть в браузере:
- `http://127.0.0.1:8000/` — главная
- `http://127.0.0.1:8000/folders` — список папок (из SQLite)
- `http://127.0.0.1:8000/docs` — список API

## Быстрый старт после перезагрузки

1) Перейти в папку проекта.

2) Убедиться, что порт свободен:

```bash
netstat -ano | findstr :8000
```

3) Запустить сервер:

```bash
cd "C:\\Users\\mzaborov\\YandexDisk\\Работы, тексты, презентации\\PhotoSorter"
C:\\Users\\mzaborov\\AppData\\Local\\Python\\pythoncore-3.14-64\\python.exe -m uvicorn --app-dir . app.main:app --port 8000
```

4) Открыть:
- `/folders` — таблица папок
- В колонке "Путь" есть ссылка **↗** на просмотр вложенных папок (`/browse?path=...`)
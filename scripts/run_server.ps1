Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Запуск Web UI (FastAPI) в режиме разработки.
# Важно: этот скрипт можно запускать в отдельном "окне сервера" и держать запущенным.

$python = "C:\Users\mzaborov\AppData\Local\Python\pythoncore-3.14-64\python.exe"
if (-not (Test-Path $python)) {
  Write-Error "Python не найден по пути: $python"
}

Write-Host "Starting uvicorn with --reload..."
& $python -m uvicorn --reload --app-dir . app.main:app --host 127.0.0.1 --port 8000








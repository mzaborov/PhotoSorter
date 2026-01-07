Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Запуск Web UI (FastAPI) в режиме разработки.
# Важно: этот скрипт можно запускать в отдельном "окне сервера" и держать запущенным.

$python = "C:\Users\mzaborov\AppData\Local\Python\pythoncore-3.14-64\python.exe"
if (-not (Test-Path $python)) {
  Write-Error "Python не найден по пути: $python"
}

# repo root = .../backend/scripts -> ../..
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path

# ВАЖНО: web_api/common/logic лежат внутри backend/, поэтому добавляем backend/ в sys.path.
$backendDir = (Resolve-Path (Join-Path $repo "backend")).Path
$env:PYTHONPATH = $backendDir

Write-Host "Starting uvicorn with --reload..."
Push-Location $repo
try {
  & $python -m uvicorn --reload --app-dir $backendDir web_api.main:app --host 127.0.0.1 --port 8000
} finally {
  Pop-Location
}













Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Запуск Web UI (FastAPI) в режиме разработки.
# Важно: этот скрипт можно запускать в отдельном "окне сервера" и держать запущенным.

# repo root = .../backend/scripts -> ../..
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path

# Пробуем использовать Python из .venv-face (где установлены numpy/sklearn)
$venvPython = Join-Path $repo ".venv-face\Scripts\python.exe"
if (Test-Path $venvPython) {
  $python = $venvPython
  Write-Host "Using Python from .venv-face: $python"
} else {
  # Fallback: системный Python (может не иметь numpy/sklearn)
  $python = "C:\Users\mzaborov\AppData\Local\Python\pythoncore-3.14-64\python.exe"
  if (-not (Test-Path $python)) {
    Write-Error "Python не найден по пути: $python"
  } else {
    Write-Warning "Using system Python, numpy/sklearn may not be available!"
  }
}

# repo root = .../backend/scripts -> ../..
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path

# ВАЖНО: web_api/common/logic лежат внутри backend/, поэтому добавляем backend/ в sys.path.
# repo root уже вычислен выше
$backendDir = (Resolve-Path (Join-Path $repo "backend")).Path
$env:PYTHONPATH = $backendDir

Write-Host "Starting uvicorn with --reload..."
Write-Host "  Note: uvicorn watches for changes in Python files. Scripts in backend/scripts/ may trigger reloads."
Push-Location $repo
try {
  & $python -m uvicorn --reload --app-dir $backendDir web_api.main:app --host 127.0.0.1 --port 8000
} finally {
  Pop-Location
}













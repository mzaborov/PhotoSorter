# Безопасная обёртка для запуска Python команд в Cursor
# Предотвращает проблемы с терминалом при выполнении Python команд
#
# Использование:
#   .\backend\scripts\tools\run_python_safe.ps1 backend/scripts/debug/list_pipeline_runs.py --limit 15
#
# Или с полным путём:
#   .\backend\scripts\tools\run_python_safe.ps1 C:\Projects\PhotoSorter\backend\scripts\debug\list_pipeline_runs.py --limit 15

param(
  [Parameter(ValueFromRemainingArguments = $true, Mandatory = $true)]
  [string[]]$Args
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Определяем путь к Python
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path

# Пробуем использовать Python из .venv-face
$venvPython = Join-Path $repo ".venv-face\Scripts\python.exe"
if (Test-Path $venvPython) {
  $python = $venvPython
} else {
  # Fallback: системный Python
  $python = "C:\Users\mzaborov\AppData\Local\Python\pythoncore-3.14-64\python.exe"
  if (-not (Test-Path $python)) {
    Write-Error "Python не найден. Проверьте установку Python."
    exit 1
  }
}

# Устанавливаем PYTHONPATH
$backendPath = (Resolve-Path (Join-Path $repo "backend")).Path
$env:PYTHONPATH = $backendPath

# Важно: отключаем буферизацию для немедленного вывода
$env:PYTHONUNBUFFERED = "1"

# Первый аргумент - это скрипт Python
$scriptPath = $Args[0]
if (-not (Test-Path $scriptPath)) {
  # Пробуем относительно repo root
  $scriptPath = Join-Path $repo $Args[0]
  if (-not (Test-Path $scriptPath)) {
    Write-Error "Скрипт не найден: $($Args[0])"
    exit 1
  }
}

# Остальные аргументы передаём скрипту
$scriptArgs = $Args[1..($Args.Length - 1)]

Write-Host "Running Python script safely..." -ForegroundColor Cyan
Write-Host "  Python: $python" -ForegroundColor Gray
Write-Host "  Script: $scriptPath" -ForegroundColor Gray
if ($scriptArgs.Count -gt 0) {
  Write-Host "  Args: $($scriptArgs -join ' ')" -ForegroundColor Gray
}
Write-Host ""

# Запускаем Python скрипт
# Используем Start-Process для изоляции вывода и предотвращения проблем с терминалом
try {
  # Простой запуск с перенаправлением вывода
  & $python $scriptPath @scriptArgs
  
  # Пробрасываем код возврата
  exit $LASTEXITCODE
} catch {
  Write-Error "Ошибка выполнения Python скрипта: $_"
  exit 1
}

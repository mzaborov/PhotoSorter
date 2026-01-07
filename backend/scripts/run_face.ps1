# Запуск face-задач через отдельное окружение .venv-face (Python 3.12),
# без необходимости "активировать" venv руками.
#
# Пример:
#   .\scripts\run_face.ps1 scripts/tools/face_scan.py --help
#
# Пока это только обёртка под будущие команды (скан лиц/кластеризация).

param(
  [Parameter(ValueFromRemainingArguments = $true, Mandatory = $true)]
  [string[]]$Args
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$py = Join-Path $repo ".venv-face\Scripts\python.exe"
$py = (Resolve-Path $py).Path

if (-not (Test-Path $py)) {
  Write-Error "Не найден python в .venv-face: $py. Сначала создай .venv-face (Python 3.12)."
}

# Гарантируем импорт модулей из backend/ (там лежат logic/web_api/common).
$env:PYTHONPATH = (Resolve-Path (Join-Path $repo "backend")).Path

# Важно для UI: печатать прогресс сразу, без буферизации stdout (иначе прогресс-бары "висят" на заглушке).
$env:PYTHONUNBUFFERED = "1"

Write-Host "Running: $py $Args"
Write-Host ("Running: " + $py + " " + ($Args -join " "))
& $py @Args

# Важно: пробрасываем код возврата python-процесса наружу,
# чтобы Web API мог корректно выставить статус completed/failed.
exit $LASTEXITCODE



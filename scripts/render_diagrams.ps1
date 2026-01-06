Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Генерация PNG-диаграмм из PlantUML (*.puml) в docs/diagrams/.
# Не требует Java: используется публичный PlantUML server (URL-encoding внутри scripts/render_diagrams.py).

$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$py = Join-Path $root "python.exe"

if (Test-Path $py) {
  # если вдруг в корне лежит python.exe (обычно нет) — используем его
  $python = $py
} else {
  $python = "python"
}

Push-Location $root
try {
  & $python scripts/render_diagrams.py --in-dir "docs/diagrams"
} finally {
  Pop-Location
}








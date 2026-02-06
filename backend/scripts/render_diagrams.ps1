Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Генерация PNG-диаграмм из PlantUML (*.puml) в docs/diagrams/.
# Используется локальный plantuml.jar (путь: PLANTUML_JAR в secrets.env или backend/scripts/plantuml.jar).

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












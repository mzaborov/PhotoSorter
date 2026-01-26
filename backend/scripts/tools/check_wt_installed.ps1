# Проверка наличия Windows Terminal (wt.exe)
# Запуск: powershell.exe -ExecutionPolicy Bypass -File backend/scripts/tools/check_wt_installed.ps1

Write-Host "Проверка наличия Windows Terminal (wt.exe)..." -ForegroundColor Cyan

# Проверка через Get-Command
$wtCommand = Get-Command wt.exe -ErrorAction SilentlyContinue

if ($wtCommand) {
    Write-Host "✓ Windows Terminal найден!" -ForegroundColor Green
    Write-Host "  Путь: $($wtCommand.Source)" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Можно использовать 'wt.exe' в настройках Cursor" -ForegroundColor Green
    exit 0
} else {
    Write-Host "✗ Windows Terminal не найден" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Альтернативы:" -ForegroundColor Cyan
    Write-Host "  1. Установить Windows Terminal из Microsoft Store" -ForegroundColor Gray
    Write-Host "  2. Использовать 'powershell.exe' в настройках Cursor" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Для использования PowerShell измените в .vscode/settings.json:" -ForegroundColor Yellow
    Write-Host '  "terminal.external.windowsExec": "powershell.exe"' -ForegroundColor Gray
    exit 1
}

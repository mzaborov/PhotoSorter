# Скрипт для коммита изменений сессии.
# Сообщение коммита передаётся параметром -Message или первым позиционным аргументом.
# Пример: .\commit_session.ps1 -Message "Описание изменений"
#         .\commit_session.ps1 "Краткое описание"

param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Message
)

Write-Host "=== Проверка изменений ===" -ForegroundColor Cyan
git status --short

Write-Host "`n=== Добавление всех изменений ===" -ForegroundColor Cyan
git add -A

Write-Host "`n=== Коммит ===" -ForegroundColor Cyan
git commit -m $Message

Write-Host "`n=== Push в GitHub ===" -ForegroundColor Cyan
git push

Write-Host "`n=== Готово ===" -ForegroundColor Green

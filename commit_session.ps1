# Скрипт для коммита изменений сессии
# Убраны все автоматические вызовы DDL из runtime-кода

Write-Host "=== Проверка изменений ===" -ForegroundColor Cyan
git status --short

Write-Host "`n=== Добавление всех изменений ===" -ForegroundColor Cyan
git add -A

Write-Host "`n=== Коммит ===" -ForegroundColor Cyan
$commitMessage = @"
Убраны все автоматические вызовы DDL из runtime-кода

- Удалены вызовы init_db() из конструкторов FaceStore, DedupStore, PipelineStore
- Удален вызов init_db() из list_folders()
- Удален вызов _ensure_face_schema() из FaceStore.__init__()
- Исправлены синтаксические ошибки (отступы в try-except блоках)
- Обновлена документация (History.log, docs/README.md)

DDL операции теперь выполняются только явно через скрипты, не в runtime, что исключает риск порчи данных при работе приложения.
"@

git commit -m $commitMessage

Write-Host "`n=== Push в GitHub ===" -ForegroundColor Cyan
git push

Write-Host "`n=== Готово ===" -ForegroundColor Green

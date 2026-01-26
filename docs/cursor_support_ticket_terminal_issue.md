# Обращение в техподдержку Cursor

**Тема:** Cursor: после обновления не работают интегрированные терминалы, Internal Error в чате, ошибка сериализации

**Request ID:** `af8369f5-0104-4a3a-aa97-e456ced56ac5`

---

## Описание проблемы

После последнего обновления Cursor интегрированный терминал в проекте перестал выполнять **любые** команды (PowerShell). Ранее помогала очистка `%APPDATA%\Cursor\User\workspaceStorage\`, но теперь **не помогает** — терминалы "ломаются" сразу после обновления/запуска.

Параллельно в UI чата всплывает окно **Internal Error** и в логах/стеке вижу ошибку:
`[internal] serialize binary: invalid int 32: 3221226505`

## Окружение

- **OS**: Windows 10 (10.0.22631)
- **Shell**: PowerShell
- **Workspace**: `C:\Projects\PhotoSorter` (git repo)
- **Терминал**: integrated terminal в Cursor
- **Request ID**: `af8369f5-0104-4a3a-aa97-e456ced56ac5`

## Шаги воспроизведения

1. Запустить Cursor
2. Открыть workspace `C:\Projects\PhotoSorter`
3. Открыть integrated terminal (PowerShell)
4. Выполнить любую простую команду (например, `Get-Date` или любую другую)

## Фактический результат

- Команды в терминале не выполняются / терминал становится нерабочим
- Появляется всплывающее сообщение **Internal Error**
- Ошибка в стеке (см. ниже)
- Очистка `%APPDATA%\Cursor\User\workspaceStorage\` больше не восстанавливает работу

## Ожидаемый результат

- Терминал стабильно выполняет команды
- В чате Cursor не возникает Internal Error при работе с терминалом/агентом

## Логи/стек ошибки

**Request ID:** `af8369f5-0104-4a3a-aa97-e456ced56ac5`

```
[internal] serialize binary: invalid int 32: 3221226505
LTe: [internal] serialize binary: invalid int 32: 3221226505
    at kmf (vscode-file://vscode-app/c:/Program%20Files/cursor/resources/app/out/vs/workbench/workbench.desktop.main.js:9095:38337)
    at Cmf (vscode-file://vscode-app/c:/Program%20Files/cursor/resources/app/out/vs/workbench/workbench.desktop.main.js:9095:37240)
    at $mf (vscode-file://vscode-app/c:/Program%20Files/cursor/resources/app/out/vs/workbench/workbench.desktop.main.js:9096:4395)
    at ova.run (vscode-file://vscode-app/c:/Program%20Files/cursor/resources/app/out/vs/workbench/workbench.desktop.main.js:9096:8170)
    at async qyt.runAgentLoop (vscode-file://vscode-app/c:/Program%20Files/cursor/resources/app/out/vs/workbench/workbench.desktop.main.js:34190:57047)
    at async Wpc.streamFromAgentBackend (vscode-file://vscode-app/c:/Program%20Files/cursor/resources/app/out/vs/workbench/workbench.desktop.main.js:34239:7695)
    at async Wpc.getAgentStreamResponse (vscode-file://vscode-app/c:/Program%20Files/cursor/resources/app/out/vs/workbench/workbench.desktop.main.js:34239:8436)
    at async FTe.submitChatMaybeAbortCurrent (vscode-file://vscode-app/c:/Program%20Files/cursor/resources/app/out/vs/workbench/workbench.desktop.main.js:9170:14575)
    at async Oi (vscode-file://vscode-app/c:/Program%20Files/cursor/resources/app/out/vs/workbench/workbench.desktop.main.js:32991:3808)
```

## Анализ стека ошибки

Стек ошибки указывает на **проблему в JavaScript коде Cursor** (не в окружении пользователя):

1. **Все вызовы идут из скомпилированного JS кода Cursor:**
   - `vscode-file://vscode-app/c:/Program%20Files/cursor/resources/app/out/vs/workbench/workbench.desktop.main.js`
   - Это бандл фронтенда Cursor (Electron/VSCode-based приложение)

2. **Ошибка в сериализации бинарных данных:**
   - `serialize binary: invalid int 32: 3221226505`
   - Значение `3221226505` превышает максимальное значение для int32 (`2,147,483,647`)
   - Это указывает на **переполнение** или некорректное значение при сериализации данных

3. **Контекст вызова:**
   - `qyt.runAgentLoop` → `Wpc.streamFromAgentBackend` → `FTe.submitChatMaybeAbortCurrent`
   - Ошибка возникает при работе агента/чата, вероятно при передаче данных между фронтендом и бэкендом

**Вывод:** Это баг в коде Cursor — при сериализации бинарных данных (вероятно, при обмене с агентом/терминалом) передаётся значение, которое не укладывается в int32. Это может быть:
- Переполнение при вычислениях
- Некорректная обработка данных от терминала
- Баг в сериализации после обновления

## Что уже пробовал(а)

- Полная очистка `%APPDATA%\Cursor\User\workspaceStorage\` и перезапуск Cursor — **не помогло** (раньше временно помогало)

## Просьба к поддержке

1. Подтвердить, известна ли проблема с `serialize binary: invalid int 32: 3221226505` и поломкой integrated terminal после обновления
2. Подсказать, какие **дополнительные логи** собрать и куда их приложить (например, из `%APPDATA%\Cursor\logs\`), чтобы вы могли быстро найти первопричину
3. Если есть временный workaround (настройка терминала/PTY/shell integration) — дайте точные шаги

Если нужно — могу приложить скриншот всплывающего **Internal Error** и любые логи/дампы, которые вы укажете.

---

**Примечание:** Проблема появилась после последнего обновления Cursor. Ранее очистка workspaceStorage временно решала проблему, но теперь не помогает.

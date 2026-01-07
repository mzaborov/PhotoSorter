@echo off
setlocal enabledelayedexpansion

rem Run FastAPI server without Cursor (Windows cmd).
rem - Uses repo-root as working dir
rem - Adds backend/ to PYTHONPATH
rem - Starts uvicorn with --reload on 127.0.0.1:8000

set "REPO=%~dp0"
rem Remove trailing backslash
if "%REPO:~-1%"=="\" set "REPO=%REPO:~0,-1%"

set "BACKEND=%REPO%\backend"
set "PYTHONPATH=%BACKEND%"

set "PYEXE=C:\Users\mzaborov\AppData\Local\Python\pythoncore-3.14-64\python.exe"
if exist "%PYEXE%" (
  set "PY=%PYEXE%"
) else (
  rem Fallback: rely on python in PATH
  set "PY=python"
)

cd /d "%REPO%" || exit /b 1

echo Starting uvicorn...
echo   repo: %REPO%
echo   app-dir: %BACKEND%
echo   app: web_api.main:app
echo   host: 127.0.0.1
echo   port: 8000
echo.

"%PY%" -m uvicorn --reload --app-dir "%BACKEND%" web_api.main:app --host 127.0.0.1 --port 8000

endlocal


@echo off
chcp 65001 >nul
setlocal EnableExtensions
cd /d "%~dp0"

rem H2-3: один .bat для первого запуска и старта веб-UI.
rem CLI по-прежнему доступен: python main.py

set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHON="

rem Предпочитаем поддерживаемые версии 3.10-3.12, даже если python в PATH — 3.14.
py -3.12 -V >nul 2>nul
if not errorlevel 1 set "PYTHON=py -3.12"
if not defined PYTHON (
    py -3.11 -V >nul 2>nul
    if not errorlevel 1 set "PYTHON=py -3.11"
)
if not defined PYTHON (
    py -3.10 -V >nul 2>nul
    if not errorlevel 1 set "PYTHON=py -3.10"
)
if not defined PYTHON (
    where python >nul 2>nul
    if not errorlevel 1 set "PYTHON=python"
)

if not defined PYTHON (
    echo [H2-3] Python не найден. Установите Python 3.10-3.12 и добавьте его в PATH.
    goto :error
)

%PYTHON% -c "import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] <= (3, 12) else 1)"
if errorlevel 1 (
    echo [H2-3] Нужен Python 3.10-3.12. Найден:
    %PYTHON% -V
    goto :error
)

if not exist ".venv\Scripts\python.exe" (
    echo [H2-3] Первый запуск: создаю .venv и устанавливаю зависимости...
    %PYTHON% -m venv .venv
    if errorlevel 1 goto :error
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    if errorlevel 1 goto :error
    ".venv\Scripts\python.exe" -m pip install -e .
    if errorlevel 1 goto :error
)

if not exist ".env" (
    if exist ".env.example" (
        copy /Y ".env.example" ".env" >nul
        echo [H2-3] Создан .env из .env.example. Проверьте LLM_NAME, LLM_API_NAME, SAVE_DIR.
    )
)

".venv\Scripts\python.exe" scripts\start.py %*
if errorlevel 1 goto :error
exit /b 0

:error
echo.
echo [H2-3] Запуск не удался. Смотрите сообщение выше и docs/TROUBLESHOOTING.md.
pause
exit /b 1

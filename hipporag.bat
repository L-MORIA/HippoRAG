@echo off
chcp 65001 >nul
cd /d "%~dp0"
rem HippoRAG: запуск веб-сервера + автооткрытие веб-UI.
rem Если сервер уже работает - просто откроется браузер.
set "PY=venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
"%PY%" scripts\start.py --host 127.0.0.1 --port 8000 --open-browser
pause

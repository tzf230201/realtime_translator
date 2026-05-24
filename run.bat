@echo off
title Realtime Translator
cd /d "%~dp0backend"

if not exist ".venv\Scripts\python.exe" (
    echo [error] venv not found. Run backend\install.bat first.
    pause
    exit /b 1
)

REM Open the browser in the background after 25s so the model has time to load
start /b "" cmd /c "timeout /t 25 /nobreak >nul 2>&1 && start """" http://localhost:8000"

call .venv\Scripts\activate.bat
echo === Starting Realtime Translator at http://localhost:8000 ===
echo (Browser will open automatically once the models are ready)
echo.
python server.py
pause

@echo off
title Realtime Translator
cd /d "%~dp0backend"

if not exist ".venv\Scripts\python.exe" (
    echo [error] venv tidak ditemukan. Jalankan dulu: backend\install.bat
    pause
    exit /b 1
)

REM Buka browser di background setelah 25 detik (memberi waktu model load)
start /b "" cmd /c "timeout /t 25 /nobreak >nul 2>&1 && start """" http://localhost:8000"

call .venv\Scripts\activate.bat
echo === Starting Realtime Translator at http://localhost:8000 ===
echo (Browser akan terbuka otomatis setelah model siap)
echo.
python server.py
pause

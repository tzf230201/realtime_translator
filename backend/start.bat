@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
echo.
echo === Starting server at http://localhost:8000 ===
echo First run downloads the Whisper model (~1.5 GB). Subsequent runs load from cache.
echo.
python server.py
pause

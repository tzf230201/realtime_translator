@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
echo.
echo === Starting server at http://localhost:8000 ===
echo First run will download models (~5 GB). Subsequent runs are instant.
echo.
python server.py
pause

@echo off
REM One-time install: creates venv + installs packages + installs CUDA-enabled torch.
cd /d "%~dp0"

echo === Creating venv ===
py -3.12 -m venv .venv
if errorlevel 1 (
    echo Failed to create venv. Make sure Python 3.12 is installed.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat

echo === Upgrading pip ===
python -m pip install --upgrade pip

echo === Installing PyTorch with CUDA 12.8 (Blackwell support) ===
pip install torch --index-url https://download.pytorch.org/whl/cu128

echo === Installing other packages ===
pip install fastapi==0.115.6 "uvicorn[standard]==0.34.0" faster-whisper==1.1.1 transformers==4.47.1 sentencepiece==0.2.0 pykakasi==2.3.0 "numpy<2"

echo.
echo === Done. Run start.bat to launch the server. ===
pause

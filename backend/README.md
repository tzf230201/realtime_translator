# Offline Backend (CUDA)

Stack: faster-whisper (large-v3-turbo) + NLLB-200 distilled 1.3B + pykakasi,
served via FastAPI + WebSocket.

## Requirements

- Windows 10/11
- Python 3.12 (`py -3.12 --version`)
- NVIDIA GPU dengan CUDA driver (sudah terinstall — kamu pakai RTX 5070)
- ~10 GB free disk (models + packages + venv)

## Install (one time)

Double-click `install.bat`, tunggu selesai (~5-10 menit).

Ini akan:
1. Buat virtual environment di `backend/.venv/`
2. Install PyTorch dengan CUDA 12.4
3. Install FastAPI, faster-whisper, transformers, dll.

## Start the server

Double-click `start.bat`.

**Run pertama kali:** akan download models otomatis dari Hugging Face:
- `large-v3-turbo` (~1.5 GB)
- `nllb-200-distilled-1.3B` (~2.6 GB)

Disimpan di `C:\Users\teuku\.cache\huggingface\`. Run berikutnya langsung load dari cache (~30 detik).

Saat server siap, buka browser: **http://localhost:8000**

Di UI, pilih **Mode: Local (Offline, CUDA)**, lalu klik Mulai Rekam.

## Tuning

Edit `server.py`:

- **Model lebih kecil** (lebih cepat, sedikit kurang akurat):
  ```python
  WHISPER_MODEL = "medium"  # atau "small"
  ```
- **Translate lebih cepat** (600M vs 1.3B):
  ```python
  NLLB_MODEL = "facebook/nllb-200-distilled-600M"
  ```
- **Threshold silence** (kalau terlalu sering/jarang kepotong):
  ```python
  SILENCE_RMS_THRESHOLD = 0.008  # naikkan kalau mic noisy
  SILENCE_WINDOW = int(SAMPLE_RATE * 0.7)  # 0.7 detik
  ```

## Troubleshooting

- **"CUDA out of memory"** → ganti ke Whisper `medium` atau `small`.
- **"cudnn not found"** → install [cuDNN 9.x](https://developer.nvidia.com/cudnn) atau pakai PyTorch CPU build.
- **Download model lambat** → set proxy atau pakai `HF_HUB_ENABLE_HF_TRANSFER=1`.
- **Permission mikrofon ditolak** → buka via `http://localhost:8000` (BUKAN file://).

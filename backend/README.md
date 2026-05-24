# Backend — Realtime Translator

Stack:
- **Whisper** large-v3-turbo (faster-whisper, CUDA fp16) untuk speech-to-text
- **Qwen2.5-7B-Instruct** Q4_K_M via **Ollama** untuk terjemahan kontekstual
- **pykakasi** untuk romaji (Jepang)
- **FastAPI** + WebSocket

Per-WebSocket rolling context: 3 utterance terakhir di-feed ke LLM sebagai multi-turn message, sehingga kata ganti (それ, あれ, dia, itu) bisa di-resolve sesuai konteks pembicaraan.

## Requirements

- Windows 10/11
- Python 3.12 (`py -3.12 --version`)
- NVIDIA GPU + CUDA driver (project di-test pada RTX 5070, ~7GB VRAM total)
- [Ollama](https://ollama.com) (atau via `winget install Ollama.Ollama`)
- ~10 GB free disk (venv + Whisper model + Qwen model)

## Install (sekali saja)

1. **Install Ollama**

   ```powershell
   winget install --id Ollama.Ollama -e
   ```

   Setelah selesai, Ollama service auto-start setiap boot.

2. **Pull Qwen2.5-7B**

   ```powershell
   ollama pull qwen2.5:7b-instruct-q4_K_M
   ```

   Sekitar 4.7 GB.

3. **Setup Python venv**

   Double-click `install.bat`. Akan buat `.venv/` dan install PyTorch CUDA, FastAPI, faster-whisper, httpx, dll.

## Menjalankan

Double-click `start.bat` (atau `run.bat` di root project untuk start + buka browser otomatis).

Run pertama: download Whisper turbo (~1.5 GB) ke `C:\Users\<you>\.cache\huggingface\`. Run berikutnya load dari cache (~15 detik).

Saat siap, buka http://localhost:8000.

## Tuning

Edit `server.py`:

- **Model LLM lebih kecil/cepat** (jika VRAM mepet atau ingin latensi lebih rendah):
  ```python
  OLLAMA_MODEL = "qwen2.5:3b-instruct-q4_K_M"  # ~2 GB VRAM, lebih cepat, sedikit kurang akurat
  # atau "gemma2:9b-instruct-q4_K_M" (~5.5 GB, alternatif setara)
  ```
  Setelah ganti, pull dulu: `ollama pull <nama-model>`.

- **Lebih banyak konteks** untuk pembicaraan panjang:
  ```python
  TRANSLATION_HISTORY_TURNS = 5  # default 3
  ```

- **Whisper lebih cepat / kurang akurat**:
  ```python
  WHISPER_MODEL = "medium"  # atau "small"
  ```

- **Sensitivitas VAD**:
  ```python
  START_SPEECH_RMS_THRESHOLD = 0.006   # naikkan kalau mic noisy
  END_SILENCE_SAMPLES = int(SAMPLE_RATE * 0.45)
  ```

- **Ollama endpoint custom** (mis. di mesin lain):
  ```powershell
  $env:OLLAMA_URL = "http://192.168.1.10:11434"; python server.py
  ```

## Troubleshooting

- **`Ollama model 'qwen2.5:...' not found`** di log → jalankan `ollama pull qwen2.5:7b-instruct-q4_K_M`.
- **Translate hang / timeout** → cek `ollama list` (apakah model ter-load) dan `ollama ps` (apakah ada process aktif).
- **"CUDA out of memory"** → kombinasi Whisper + Qwen 7B Q4 butuh ~7 GB VRAM. Turunkan Whisper (`medium`) atau ganti LLM ke 3B.
- **Permission mikrofon ditolak** → buka via `http://localhost:8000` (BUKAN `file://`).
- **Latency tinggi (>2s per kalimat)** → cek `ollama ps` — kalau "Until" pendek (model dibuang dari VRAM), naikkan `OLLAMA_KEEP_ALIVE` di server.py.

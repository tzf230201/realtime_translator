# Backend — Realtime Translator

Stack:
- **Whisper** large-v3-turbo (faster-whisper, CUDA fp16) for speech-to-text
- **Qwen2.5-7B-Instruct** Q4_K_M served by **Ollama** for context-aware translation
- **Resemblyzer** for per-utterance speaker diarization (CPU)
- **FastAPI** + WebSocket

Each WebSocket keeps a rolling context of the last 3 utterances and feeds them
to the LLM as multi-turn messages, so pronouns and references (それ, あれ, "it",
"that one") are resolved against the ongoing conversation instead of being
translated in isolation.

## Requirements

- Windows 10/11
- Python 3.12 (`py -3.12 --version`)
- NVIDIA GPU + CUDA driver (tested on RTX 5070, ~7 GB VRAM in total)
- [Ollama](https://ollama.com) (or install via `winget install Ollama.Ollama`)
- ~10 GB free disk (venv + Whisper model + Qwen model)

## One-time install

1. **Install Ollama**

   ```powershell
   winget install --id Ollama.Ollama -e
   ```

   The Ollama service auto-starts on boot after installation.

2. **Pull Qwen2.5-7B**

   ```powershell
   ollama pull qwen2.5:7b-instruct-q4_K_M
   ```

   About 4.7 GB.

3. **Set up the Python venv**

   Double-click `install.bat`. It creates `.venv/` and installs FastAPI,
   faster-whisper, httpx, etc.

## Running

Double-click `start.bat`, or `run.bat` at the project root for a launch
that also opens the browser automatically.

First run downloads the Whisper turbo model (~1.5 GB) into
`C:\Users\<you>\.cache\huggingface\`. Subsequent runs load from cache
(~15 s).

Once the log shows `Uvicorn running`, open http://localhost:8000.

## Tuning

Edit `server.py`:

- **Smaller / faster LLM** (when VRAM is tight or you want lower latency):
  ```python
  OLLAMA_MODEL = "qwen2.5:3b-instruct-q4_K_M"  # ~2 GB VRAM, faster, slightly less accurate
  # or "gemma2:9b-instruct-q4_K_M" (~5.5 GB, comparable alternative)
  ```
  Remember to `ollama pull <name>` first.

- **Longer conversational memory**:
  ```python
  TRANSLATION_HISTORY_TURNS = 5  # default 3
  ```

- **Faster / less accurate Whisper**:
  ```python
  WHISPER_MODEL = "medium"  # or "small"
  ```

- **VAD sensitivity**:
  ```python
  START_SPEECH_RMS_THRESHOLD = 0.006   # raise if mic picks up noise
  END_SILENCE_SAMPLES = int(SAMPLE_RATE * 0.45)
  ```

- **Point Ollama at a different host** (e.g. another machine on the LAN):
  ```powershell
  $env:OLLAMA_URL = "http://192.168.1.10:11434"; python server.py
  ```

## Troubleshooting

- **`Ollama model 'qwen2.5:...' not found`** in the log → run
  `ollama pull qwen2.5:7b-instruct-q4_K_M`.
- **Translate hangs / times out** → check `ollama list` (is the model
  there?) and `ollama ps` (is a process loaded?).
- **"CUDA out of memory"** → Whisper + Qwen 7B Q4 together use ~7 GB
  VRAM. Drop Whisper to `medium` or switch the LLM to a 3B variant.
- **Microphone permission denied** → open via `http://localhost:8000`
  (not `file://`).
- **High latency (>2 s per sentence)** → check `ollama ps`. If the
  "Until" column shows a short keep-alive, raise `OLLAMA_KEEP_ALIVE`
  in `server.py`.

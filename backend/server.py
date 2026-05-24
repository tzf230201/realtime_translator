"""
Offline STT + Translate + Romaji server.
Serves the frontend at http://localhost:8000 and a WebSocket at /ws.
"""
import os

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import asyncio
import json
import re
from collections import deque
from pathlib import Path

import httpx
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketState

# ---------- Config ----------
WHISPER_MODEL = "large-v3-turbo"
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE_TYPE = "float16"

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b-instruct-q4_K_M")
OLLAMA_KEEP_ALIVE = "30m"
TRANSLATION_HISTORY_TURNS = 3  # rolling context per WebSocket connection

# Speaker diarization
SPEAKER_MATCH_THRESHOLD = 0.75   # cosine similarity to assign to existing speaker
SPEAKER_MIN_AUDIO_SECONDS = 1.2  # below this, reuse the previous label (avoid spurious new speakers)
SPEAKER_MAX_COUNT = 12

LANG_NAMES = {
    "ja": "Japanese",
    "en": "English",
    "id": "Indonesian",
}

SAMPLE_RATE = 16000
MIN_TRANSCRIBE_SAMPLES = int(SAMPLE_RATE * 0.35)
MIN_VOICED_SAMPLES = int(SAMPLE_RATE * 0.20)
MAX_UTTERANCE_SAMPLES = int(SAMPLE_RATE * 8.0)
PRE_ROLL_SAMPLES = int(SAMPLE_RATE * 0.25)
END_SILENCE_SAMPLES = int(SAMPLE_RATE * 0.45)
START_SPEECH_RMS_THRESHOLD = 0.006
CONTINUE_SPEECH_RMS_THRESHOLD = 0.003

PROMPT_LEAK_GUARDS = {
    "ja": (
        "句読点を正しく使ってください",
        "自然な日本語の文章",
    ),
    "en": (
        "clear english transcription",
        "proper punctuation",
    ),
    "id": (
        "transkrip bahasa indonesia",
        "tanda baca yang benar",
    ),
}

STATIC_DIR = Path(__file__).resolve().parent.parent

# ---------- Globals (populated in load_models) ----------
whisper = None
speaker_encoder = None


WHISPER_REPO_MAP = {
    "large-v3": "Systran/faster-whisper-large-v3",
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    "large-v2": "Systran/faster-whisper-large-v2",
    "medium": "Systran/faster-whisper-medium",
    "small": "Systran/faster-whisper-small",
}


def ensure_whisper_model(model_name: str) -> str:
    """Download Whisper model to a local dir (avoids Windows symlink permission issues)."""
    from huggingface_hub import snapshot_download

    models_dir = Path(__file__).resolve().parent / "models"
    models_dir.mkdir(exist_ok=True)
    repo_id = WHISPER_REPO_MAP.get(model_name, model_name)
    local_path = models_dir / repo_id.replace("/", "__")
    if not (local_path / "model.bin").exists():
        print(f"[download] {repo_id} -> {local_path}", flush=True)
        snapshot_download(repo_id=repo_id, local_dir=str(local_path))
    return str(local_path)


def warmup_ollama():
    """Ping Ollama and pre-load the model so the first WS translation is fast."""
    try:
        with httpx.Client(timeout=10.0) as client:
            tags = client.get(f"{OLLAMA_URL}/api/tags")
            tags.raise_for_status()
            names = [m.get("name", "") for m in tags.json().get("models", [])]
            if OLLAMA_MODEL not in names:
                print(
                    f"[warn] Ollama model '{OLLAMA_MODEL}' not found locally. "
                    f"Run: ollama pull {OLLAMA_MODEL}",
                    flush=True,
                )
                return
        print(f"[load] Warming up Ollama model '{OLLAMA_MODEL}'...", flush=True)
        with httpx.Client(timeout=120.0) as client:
            client.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": OLLAMA_MODEL,
                    "messages": [{"role": "user", "content": "ping"}],
                    "stream": False,
                    "keep_alive": OLLAMA_KEEP_ALIVE,
                    "options": {"num_predict": 1},
                },
            ).raise_for_status()
    except Exception as e:
        print(f"[warn] Ollama warmup failed: {e}", flush=True)


def load_models():
    """Load heavy models. Called only from __main__ to avoid Windows multiprocessing re-entry."""
    from faster_whisper import WhisperModel
    from resemblyzer import VoiceEncoder

    global whisper, speaker_encoder

    whisper_path = ensure_whisper_model(WHISPER_MODEL)
    print(f"[load] Whisper {WHISPER_MODEL} on {WHISPER_DEVICE} ({WHISPER_COMPUTE_TYPE})...", flush=True)
    whisper = WhisperModel(whisper_path, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE)

    print("[load] speaker encoder (Resemblyzer, CPU)...", flush=True)
    speaker_encoder = VoiceEncoder("cpu", verbose=False)

    warmup_ollama()
    print("[ready] All models loaded.", flush=True)


def _clean_for_translation(text: str) -> str:
    """Normalize whitespace and trim filler from Whisper output."""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" ,;:、")
    return text


_TRANSLATION_PREFIXES = (
    "translation:", "translated:", "indonesian:", "english:", "japanese:",
    "terjemahan:", "id:", "en:", "ja:",
)
_QUOTE_PAIRS = (('"', '"'), ("'", "'"), ("「", "」"), ("『", "』"), ("“", "”"), ("‘", "’"))


def _post_clean_llm(text: str) -> str:
    text = text.strip()
    for op, cl in _QUOTE_PAIRS:
        if text.startswith(op) and text.endswith(cl) and len(text) > len(op) + len(cl):
            text = text[len(op):-len(cl)].strip()
    lowered = text.lower()
    for prefix in _TRANSLATION_PREFIXES:
        if lowered.startswith(prefix):
            text = text[len(prefix):].strip()
            break
    return text


class TranslationContext:
    """Holds the last few translated turns per WebSocket for conversational context."""

    def __init__(self, max_turns: int = TRANSLATION_HISTORY_TURNS):
        self.turns: deque[tuple[str, str]] = deque(maxlen=max_turns)
        self.src_lang: str | None = None
        self.tgt_lang: str | None = None

    def reset_for(self, src: str, tgt: str):
        if (src, tgt) != (self.src_lang, self.tgt_lang):
            self.turns.clear()
            self.src_lang, self.tgt_lang = src, tgt

    def add(self, src_text: str, tgt_text: str):
        if src_text and tgt_text:
            self.turns.append((src_text, tgt_text))


def _build_translate_messages(text: str, src: str, tgt: str, ctx: TranslationContext) -> list[dict]:
    src_name = LANG_NAMES[src]
    tgt_name = LANG_NAMES[tgt]
    system = (
        f"You translate live speech from {src_name} to {tgt_name}. "
        f"Output ONLY the {tgt_name} translation — no quotes, no prefix, no notes, no explanation. "
        f"Use natural, conversational {tgt_name} that matches the speaker's register (casual stays casual, formal stays formal). "
        f"Refer to the prior turns for context (pronouns, references, ongoing topic) but never re-translate them. "
        f"If the input is empty or untranslatable, reply with an empty message."
    )
    messages: list[dict] = [{"role": "system", "content": system}]
    for src_h, tgt_h in ctx.turns:
        messages.append({"role": "user", "content": src_h})
        messages.append({"role": "assistant", "content": tgt_h})
    messages.append({"role": "user", "content": text})
    return messages


def translate(text: str, src: str, tgt: str, ctx: TranslationContext) -> str:
    text = _clean_for_translation(text)
    if not text or src == tgt:
        return text
    if src not in LANG_NAMES or tgt not in LANG_NAMES:
        return ""
    ctx.reset_for(src, tgt)

    payload = {
        "model": OLLAMA_MODEL,
        "messages": _build_translate_messages(text, src, tgt, ctx),
        "stream": False,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": {
            "temperature": 0.2,
            "top_p": 0.9,
            "num_predict": 512,
            "repeat_penalty": 1.05,
        },
    }
    try:
        with httpx.Client(timeout=60.0) as client:
            r = client.post(f"{OLLAMA_URL}/api/chat", json=payload)
            r.raise_for_status()
            data = r.json()
    except Exception as e:
        print(f"[translate-error] {e}", flush=True)
        return ""

    translation = _post_clean_llm(data.get("message", {}).get("content", ""))
    if translation:
        ctx.add(text, translation)
    return translation


class SpeakerRegistry:
    """Online clustering of speaker embeddings, one instance per WebSocket connection."""

    def __init__(self, threshold: float = SPEAKER_MATCH_THRESHOLD, max_speakers: int = SPEAKER_MAX_COUNT):
        self.centroids: list[np.ndarray] = []
        self.labels: list[str] = []
        self.counts: list[int] = []
        self.threshold = threshold
        self.max_speakers = max_speakers
        self.last_label: str | None = None

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
        return float(np.dot(a, b) / denom)

    def _new_label(self) -> str:
        i = len(self.centroids)
        return chr(ord("A") + i) if i < 26 else f"S{i + 1}"

    def assign(self, embedding: np.ndarray | None) -> str | None:
        if embedding is None:
            return self.last_label  # too short to embed — reuse previous label
        if not self.centroids:
            label = self._new_label()
            self.centroids.append(embedding.copy())
            self.labels.append(label)
            self.counts.append(1)
            self.last_label = label
            return label
        sims = [self._cosine(embedding, c) for c in self.centroids]
        best = int(np.argmax(sims))
        if sims[best] >= self.threshold:
            n = self.counts[best]
            self.centroids[best] = (self.centroids[best] * n + embedding) / (n + 1)
            self.counts[best] = n + 1
            self.last_label = self.labels[best]
            return self.last_label
        if len(self.centroids) >= self.max_speakers:
            # Out of slots — bucket into the closest existing speaker.
            self.last_label = self.labels[best]
            return self.last_label
        label = self._new_label()
        self.centroids.append(embedding.copy())
        self.labels.append(label)
        self.counts.append(1)
        self.last_label = label
        return label


def compute_speaker_embedding(audio: np.ndarray) -> np.ndarray | None:
    """Return a 256-d speaker embedding, or None if the clip is too short / fails."""
    if speaker_encoder is None:
        return None
    if len(audio) < int(SAMPLE_RATE * SPEAKER_MIN_AUDIO_SECONDS):
        return None
    try:
        from resemblyzer import preprocess_wav
        wav = preprocess_wav(audio.astype(np.float32), source_sr=SAMPLE_RATE)
        if len(wav) < int(SAMPLE_RATE * 1.0):
            return None
        return speaker_encoder.embed_utterance(wav)
    except Exception as e:
        print(f"[diarize-error] {e}", flush=True)
        return None


def audio_rms(audio: np.ndarray) -> float:
    if len(audio) == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio * audio)))


def trim_audio(audio: np.ndarray, limit: int) -> np.ndarray:
    if len(audio) <= limit:
        return audio
    return audio[-limit:]


def normalize_guard_text(text: str) -> str:
    return re.sub(r"[\W_]+", "", text.lower())


def looks_like_prompt_leak(text: str, lang: str | None) -> bool:
    normalized = normalize_guard_text(text)
    if not normalized:
        return False
    for phrase in PROMPT_LEAK_GUARDS.get(lang, ()):
        if normalize_guard_text(phrase) in normalized:
            return True
    return False


def is_suspicious_transcript(text: str, lang: str | None, audio_samples: int) -> bool:
    cleaned = text.strip()
    if not cleaned:
        return True
    if looks_like_prompt_leak(cleaned, lang):
        return True
    duration = audio_samples / SAMPLE_RATE
    compact_len = len(normalize_guard_text(cleaned))
    if duration < 1.0 and compact_len > 24:
        return True
    return False


def should_transcribe_chunk(chunk: np.ndarray, voiced_samples: int) -> bool:
    if len(chunk) < MIN_TRANSCRIBE_SAMPLES:
        return False
    if voiced_samples < MIN_VOICED_SAMPLES:
        return False
    return audio_rms(chunk) >= CONTINUE_SPEECH_RMS_THRESHOLD * 0.8


def transcribe(audio: np.ndarray, lang: str | None) -> str:
    if len(audio) < MIN_TRANSCRIBE_SAMPLES:
        return ""
    normalized_lang = lang if lang in ("ja", "en", "id") else None
    segments, _ = whisper.transcribe(
        audio,
        language=normalized_lang,
        beam_size=3,
        best_of=1,
        patience=1.0,
        temperature=0.0,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 250, "speech_pad_ms": 200},
        condition_on_previous_text=False,
        compression_ratio_threshold=2.0,
        log_prob_threshold=-0.7,
        no_speech_threshold=0.65,
        word_timestamps=False,
    )
    text = " ".join(s.text.strip() for s in segments).strip()
    if is_suspicious_transcript(text, normalized_lang, len(audio)):
        return ""
    return text


# ---------- App ----------
app = FastAPI()


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    src_lang = "ja"
    tgt_lang = "id"
    translation_ctx = TranslationContext()
    speaker_registry = SpeakerRegistry()
    lead_in = np.zeros(0, dtype=np.float32)
    utterance = np.zeros(0, dtype=np.float32)
    in_speech = False
    voiced_samples = 0
    trailing_silence_samples = 0
    processing = False
    pending_chunks: list[np.ndarray] = []

    async def safe_send(payload):
        if ws.client_state != WebSocketState.CONNECTED:
            return False
        try:
            await ws.send_json(payload)
            return True
        except Exception:
            return False

    await safe_send({"type": "status", "message": "ready"})

    def reset_utterance():
        nonlocal utterance, in_speech, voiced_samples, trailing_silence_samples
        utterance = np.zeros(0, dtype=np.float32)
        in_speech = False
        voiced_samples = 0
        trailing_silence_samples = 0

    def start_next_chunk():
        nonlocal processing
        if processing or not pending_chunks:
            return
        processing = True
        asyncio.create_task(process_chunk(pending_chunks.pop(0)))

    def enqueue_chunk(chunk: np.ndarray):
        pending_chunks.append(chunk)
        start_next_chunk()

    def finalize_utterance(force: bool = False):
        nonlocal lead_in
        if not in_speech:
            return
        chunk = utterance.copy()
        voiced = voiced_samples
        reset_utterance()
        lead_in = np.zeros(0, dtype=np.float32)
        if force:
            voiced = max(voiced, MIN_VOICED_SAMPLES)
        if should_transcribe_chunk(chunk, voiced):
            enqueue_chunk(chunk)

    async def process_chunk(chunk: np.ndarray):
        nonlocal processing
        try:
            # Whisper (GPU) and speaker embedding (CPU) can run in parallel.
            text, embedding = await asyncio.gather(
                asyncio.to_thread(transcribe, chunk, src_lang),
                asyncio.to_thread(compute_speaker_embedding, chunk),
            )
            if not text or ws.client_state != WebSocketState.CONNECTED:
                return
            speaker = speaker_registry.assign(embedding)
            await safe_send({"type": "partial", "text": text, "speaker": speaker})
            translation = await asyncio.to_thread(translate, text, src_lang, tgt_lang, translation_ctx)
            await safe_send({
                "type": "final",
                "text": text,
                "translation": translation,
                "speaker": speaker,
            })
        except Exception as e:
            print(f"[error] {e}", flush=True)
            await safe_send({"type": "error", "message": str(e)})
        finally:
            processing = False
            start_next_chunk()

    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break

            if msg.get("bytes") is not None:
                pcm = np.frombuffer(msg["bytes"], dtype=np.float32)
                frame_rms = audio_rms(pcm)

                if in_speech:
                    utterance = np.concatenate([utterance, pcm])
                    if frame_rms >= CONTINUE_SPEECH_RMS_THRESHOLD:
                        voiced_samples += len(pcm)
                        trailing_silence_samples = 0
                    else:
                        trailing_silence_samples += len(pcm)
                else:
                    lead_in = trim_audio(np.concatenate([lead_in, pcm]), PRE_ROLL_SAMPLES)
                    if frame_rms >= START_SPEECH_RMS_THRESHOLD:
                        in_speech = True
                        utterance = lead_in.copy()
                        voiced_samples = len(pcm)
                        trailing_silence_samples = 0
                        lead_in = np.zeros(0, dtype=np.float32)

                if not in_speech:
                    continue

                if len(utterance) >= MAX_UTTERANCE_SAMPLES:
                    finalize_utterance(force=True)
                elif voiced_samples >= MIN_VOICED_SAMPLES and trailing_silence_samples >= END_SILENCE_SAMPLES:
                    finalize_utterance()

            elif msg.get("text") is not None:
                try:
                    data = json.loads(msg["text"])
                except Exception:
                    continue
                t = data.get("type")
                if t == "config":
                    src_lang = data.get("srcLang", src_lang)
                    tgt_lang = data.get("tgtLang", tgt_lang)
                    lead_in = np.zeros(0, dtype=np.float32)
                    reset_utterance()
                    await ws.send_json({
                        "type": "status",
                        "message": f"lang: {src_lang} -> {tgt_lang}",
                    })
                elif t == "flush":
                    finalize_utterance(force=True)
                    lead_in = np.zeros(0, dtype=np.float32)

    except WebSocketDisconnect:
        pass


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()  # early-return for spawned child processes on Windows
    load_models()
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")

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
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketState

# ---------- Config ----------
WHISPER_MODEL = "large-v3-turbo"
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE_TYPE = "float16"

NLLB_MODEL = "facebook/nllb-200-3.3B"
NLLB_QUANTIZATION = "int8"  # int8 ~3.3GB VRAM, fp16 ~6.6GB

LANG_CODES = {
    "ja": "jpn_Jpan",
    "en": "eng_Latn",
    "id": "ind_Latn",
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
nllb_tokenizer = None
nllb_model = None
kks = None


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


def ensure_nllb_ct2(repo_id: str, quantization: str) -> str:
    """Download NLLB and convert to CTranslate2 format with int8 quantization."""
    from huggingface_hub import snapshot_download
    from ctranslate2.converters import TransformersConverter

    models_dir = Path(__file__).resolve().parent / "models"
    models_dir.mkdir(exist_ok=True)

    safe_id = repo_id.replace("/", "__")
    hf_path = models_dir / safe_id
    ct2_path = models_dir / f"{safe_id}_ct2_{quantization}"

    if (ct2_path / "model.bin").exists():
        return str(ct2_path)

    if not (hf_path / "config.json").exists():
        print(f"[download] {repo_id} -> {hf_path} (~6.6GB)", flush=True)
        snapshot_download(repo_id=repo_id, local_dir=str(hf_path))

    print(f"[convert] {hf_path} -> {ct2_path} ({quantization})", flush=True)
    converter = TransformersConverter(str(hf_path))
    converter.convert(str(ct2_path), quantization=quantization, force=False)
    return str(ct2_path)


def load_models():
    """Load heavy models. Called only from __main__ to avoid Windows multiprocessing re-entry."""
    import pykakasi
    import ctranslate2
    from faster_whisper import WhisperModel
    from transformers import AutoTokenizer

    global whisper, nllb_tokenizer, nllb_model, kks

    whisper_path = ensure_whisper_model(WHISPER_MODEL)
    print(f"[load] Whisper {WHISPER_MODEL} on {WHISPER_DEVICE} ({WHISPER_COMPUTE_TYPE})...", flush=True)
    whisper = WhisperModel(whisper_path, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE)

    nllb_path = ensure_nllb_ct2(NLLB_MODEL, NLLB_QUANTIZATION)
    print(f"[load] NLLB {NLLB_MODEL} (CT2 {NLLB_QUANTIZATION}) on cuda...", flush=True)
    nllb_tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL)
    nllb_model = ctranslate2.Translator(nllb_path, device="cuda", compute_type=NLLB_QUANTIZATION)

    print("[load] pykakasi...", flush=True)
    kks = pykakasi.kakasi()
    print("[ready] All models loaded.", flush=True)


def _clean_for_translation(text: str) -> str:
    """Normalize whitespace and trim filler from Whisper output."""
    import re
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" ,;:、")
    return text


def _split_sentences(text: str, src: str) -> list[str]:
    """Split into sentences for per-sentence translation (better NLLB quality)."""
    import re
    if src == "ja":
        parts = re.split(r"(?<=[。！？])", text)
    else:
        parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _translate_sentence(text: str, src_code: str, tgt_code: str) -> str:
    nllb_tokenizer.src_lang = src_code
    src_ids = nllb_tokenizer.encode(text, truncation=True, max_length=512)
    src_tokens = nllb_tokenizer.convert_ids_to_tokens(src_ids)
    results = nllb_model.translate_batch(
        [src_tokens],
        target_prefix=[[tgt_code]],
        beam_size=5,
        max_decoding_length=512,
        length_penalty=1.0,
        repetition_penalty=1.05,
        no_repeat_ngram_size=3,
    )
    target_tokens = results[0].hypotheses[0][1:]  # skip target language prefix
    target_ids = nllb_tokenizer.convert_tokens_to_ids(target_tokens)
    return nllb_tokenizer.decode(target_ids, skip_special_tokens=True).strip()


def translate(text: str, src: str, tgt: str) -> str:
    text = _clean_for_translation(text)
    if not text or src == tgt:
        return text
    src_code = LANG_CODES.get(src)
    tgt_code = LANG_CODES.get(tgt)
    en_code = LANG_CODES["en"]
    if not src_code or not tgt_code:
        return ""
    sentences = _split_sentences(text, src)
    if not sentences:
        return ""

    # Pivot through English when neither side is English.
    # NLLB has the most training data on EN pairs, so JA→EN→ID > JA→ID directly.
    use_pivot = src != "en" and tgt != "en"

    out = []
    for s in sentences:
        if use_pivot:
            mid = _translate_sentence(s, src_code, en_code)
            if mid:
                out.append(_translate_sentence(mid, en_code, tgt_code))
        else:
            out.append(_translate_sentence(s, src_code, tgt_code))
    return " ".join(t for t in out if t)


def to_romaji(text: str) -> str:
    parts = kks.convert(text)
    return " ".join(p["hepburn"] for p in parts if p.get("hepburn")).strip()


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
            text = await asyncio.to_thread(transcribe, chunk, src_lang)
            if not text or ws.client_state != WebSocketState.CONNECTED:
                return
            await safe_send({"type": "partial", "text": text})
            translation = await asyncio.to_thread(translate, text, src_lang, tgt_lang)
            romaji = ""
            if src_lang == "ja":
                romaji = await asyncio.to_thread(to_romaji, text)
            elif tgt_lang == "ja":
                romaji = await asyncio.to_thread(to_romaji, translation)
            await safe_send({
                "type": "final",
                "text": text,
                "translation": translation,
                "romaji": romaji,
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

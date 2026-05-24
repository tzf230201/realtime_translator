const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const downloadBtn = document.getElementById('downloadBtn');
const clearBtn = document.getElementById('clearBtn');
const sourceSel = document.getElementById('source');
const srcLangSel = document.getElementById('srcLang');
const tgtLangSel = document.getElementById('tgtLang');
const transcriptEl = document.getElementById('transcript');
const translationEl = document.getElementById('translation');
const statusEl = document.getElementById('status');

// ---------- State ----------
let mediaStream = null;
let mediaRecorder = null;
let recordedChunks = [];
let recordedBlobUrl = null;
let isRecording = false;

let transcriptSegs = [];
let translationSegs = [];
const FRESH_DURATION_MS = 5000;

// Local mode state
let ws = null;
let audioCtx = null;
let workletNode = null;
let sourceNode = null;

// Visualizer state
let vizCtx = null;
let vizAnalyser = null;
let vizAnimId = null;
const SPEAK_THRESHOLD = 0.018;

// ---------- UI helpers ----------
function setStatus(text, kind = 'idle') {
  statusEl.textContent = text;
  statusEl.className = `status ${kind}`;
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, c => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;',
  }[c]));
}

const stickyMap = new WeakMap();
const SCROLL_BOTTOM_TOLERANCE = 40;

function isAtBottom(el) {
  return el.scrollHeight - el.scrollTop - el.clientHeight <= SCROLL_BOTTOM_TOLERANCE;
}

function setupStickyScroll(el) {
  if (!el || stickyMap.has(el)) return;
  stickyMap.set(el, true);
  el.addEventListener('scroll', () => {
    stickyMap.set(el, isAtBottom(el));
  });
}

function speakerChipHtml(speaker) {
  if (!speaker) return '';
  const cls = /^[A-H]$/.test(speaker) ? `speaker-${speaker}` : 'speaker-default';
  return `<span class="speaker-chip ${cls}">${escapeHtml(speaker)}</span>`;
}

function renderSegs(el, segs, interim = '') {
  const sticky = stickyMap.get(el) ?? true;
  const prevScroll = el.scrollTop;
  const last = segs.length - 1;
  const html = segs.map((s, i) => {
    let cls = 'seg old';
    if (i === last) cls = 'seg latest';
    else if (i === last - 1) cls = 'seg recent';
    const sameSpeaker = i > 0 && segs[i - 1].speaker === s.speaker;
    const chip = sameSpeaker ? '' : speakerChipHtml(s.speaker);
    const br = i > 0 && !sameSpeaker ? '<br>' : '';
    return `${br}${chip}<span class="${cls}">${escapeHtml(s.text)}</span>`;
  }).join(' ');
  el.innerHTML = html + (interim ? ` <span class="interim">${escapeHtml(interim)}</span>` : '');
  if (sticky) {
    el.scrollTop = el.scrollHeight;
  } else {
    el.scrollTop = prevScroll;
  }
}

[transcriptEl, translationEl].forEach(setupStickyScroll);

function renderTranscript(interim = '') {
  renderSegs(transcriptEl, transcriptSegs, interim);
}

function renderTranslation() {
  renderSegs(translationEl, translationSegs);
}

function appendSeg(segs, text, renderFn, speaker = null) {
  if (!text) return;
  segs.push({ text, time: Date.now(), speaker });
  renderFn();
}

function langBase(bcp47) {
  return bcp47.split('-')[0];
}

// ---------- Visualizer ----------
function startVisualizer(stream) {
  const canvas = document.getElementById('visualizer');
  const indicator = document.getElementById('speakIndicator');
  const levelBar = document.getElementById('levelBar');
  if (!canvas) return;

  vizCtx = new (window.AudioContext || window.webkitAudioContext)();
  const source = vizCtx.createMediaStreamSource(stream);
  vizAnalyser = vizCtx.createAnalyser();
  vizAnalyser.fftSize = 2048;
  vizAnalyser.smoothingTimeConstant = 0.75;
  source.connect(vizAnalyser);

  const ctx2d = canvas.getContext('2d');
  const bufLen = vizAnalyser.fftSize;
  const timeArr = new Uint8Array(bufLen);

  function resize() {
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx2d.setTransform(1, 0, 0, 1, 0, 0);
    ctx2d.scale(dpr, dpr);
  }

  resize();
  window.addEventListener('resize', resize);

  let smoothedRms = 0;

  function draw() {
    vizAnimId = requestAnimationFrame(draw);
    vizAnalyser.getByteTimeDomainData(timeArr);

    let sum = 0;
    for (let i = 0; i < bufLen; i++) {
      const v = (timeArr[i] - 128) / 128;
      sum += v * v;
    }
    const rms = Math.sqrt(sum / bufLen);
    smoothedRms = smoothedRms * 0.7 + rms * 0.3;
    const speaking = smoothedRms > SPEAK_THRESHOLD;

    if (indicator) {
      indicator.textContent = speaking ? 'Speaking' : 'Silence';
      indicator.className = 'speak-indicator ' + (speaking ? 'active' : 'silent');
    }
    if (levelBar) {
      const pct = Math.min(100, smoothedRms * 500);
      levelBar.style.width = pct + '%';
    }

    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    ctx2d.fillStyle = '#0f1221';
    ctx2d.fillRect(0, 0, w, h);

    ctx2d.strokeStyle = 'rgba(154, 163, 199, 0.15)';
    ctx2d.lineWidth = 1;
    ctx2d.beginPath();
    ctx2d.moveTo(0, h / 2);
    ctx2d.lineTo(w, h / 2);
    ctx2d.stroke();

    ctx2d.lineWidth = 2;
    ctx2d.strokeStyle = speaking ? '#4ec9b0' : '#6c8cff';
    ctx2d.beginPath();
    const step = w / bufLen;
    for (let i = 0; i < bufLen; i++) {
      const v = timeArr[i] / 128 - 1;
      const x = i * step;
      const y = h / 2 + v * (h / 2 - 4);
      if (i === 0) ctx2d.moveTo(x, y);
      else ctx2d.lineTo(x, y);
    }
    ctx2d.stroke();
  }

  draw();
}

function stopVisualizer() {
  if (vizAnimId) cancelAnimationFrame(vizAnimId);
  vizAnimId = null;
  if (vizCtx) {
    try { vizCtx.close(); } catch (_) {}
    vizCtx = null;
  }
  vizAnalyser = null;

  const indicator = document.getElementById('speakIndicator');
  const levelBar = document.getElementById('levelBar');
  if (indicator) {
    indicator.textContent = 'Idle';
    indicator.className = 'speak-indicator';
  }
  if (levelBar) levelBar.style.width = '0%';

  const canvas = document.getElementById('visualizer');
  if (canvas) {
    const c = canvas.getContext('2d');
    c.fillStyle = '#0f1221';
    c.fillRect(0, 0, canvas.clientWidth, canvas.clientHeight);
  }
}

// ---------- Local mode: WebSocket + AudioWorklet ----------
const WORKLET_CODE = `
class PCMProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buffer = [];
    this.targetSize = 1600; // 100ms at 16kHz
  }
  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;
    const ch = input[0];
    for (let i = 0; i < ch.length; i++) this.buffer.push(ch[i]);
    if (this.buffer.length >= this.targetSize) {
      this.port.postMessage(new Float32Array(this.buffer));
      this.buffer = [];
    }
    return true;
  }
}
registerProcessor('pcm-processor', PCMProcessor);
`;

async function startLocalMode() {
  const wsProtocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${wsProtocol}//${location.host}/ws`);
  ws.binaryType = 'arraybuffer';
  const wsReady = new Promise((resolve, reject) => {
    const timeoutId = window.setTimeout(() => {
      reject(new Error('Connection to the local server timed out'));
    }, 5000);
    ws.addEventListener('open', () => {
      window.clearTimeout(timeoutId);
      resolve();
    }, { once: true });
    ws.addEventListener('error', () => {
      window.clearTimeout(timeoutId);
      reject(new Error('Failed to connect to the local server'));
    }, { once: true });
  });

  ws.onmessage = (ev) => {
    try {
      const data = JSON.parse(ev.data);
      if (data.type === 'status') {
        setStatus('Server: ' + data.message, 'ok');
      } else if (data.type === 'partial') {
        renderTranscript(data.text);
      } else if (data.type === 'final') {
        const sp = data.speaker || null;
        appendSeg(transcriptSegs, data.text, renderTranscript, sp);
        appendSeg(translationSegs, data.translation, renderTranslation, sp);
      } else if (data.type === 'error') {
        setStatus('Server error: ' + data.message, 'error');
      }
    } catch (_) {}
  };

  ws.onerror = () => setStatus('WebSocket error — is the server running?', 'error');
  ws.onclose = () => {
    if (isRecording) setStatus('WebSocket disconnected', 'error');
  };

  await wsReady;
  ws.send(JSON.stringify({
    type: 'config',
    srcLang: langBase(srcLangSel.value),
    tgtLang: tgtLangSel.value,
  }));

  audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
  const blob = new Blob([WORKLET_CODE], { type: 'application/javascript' });
  const url = URL.createObjectURL(blob);
  await audioCtx.audioWorklet.addModule(url);
  URL.revokeObjectURL(url);

  sourceNode = audioCtx.createMediaStreamSource(mediaStream);
  // Boost quiet mics. 3x ≈ +9.5 dB.
  const gainNode = audioCtx.createGain();
  gainNode.gain.value = 3.0;
  workletNode = new AudioWorkletNode(audioCtx, 'pcm-processor');
  workletNode.port.onmessage = (e) => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(e.data.buffer);
    }
  };
  sourceNode.connect(gainNode);
  gainNode.connect(workletNode);
}

function stopLocalMode() {
  if (ws && ws.readyState === WebSocket.OPEN) {
    try { ws.send(JSON.stringify({ type: 'flush' })); } catch (_) {}
    try { ws.close(); } catch (_) {}
  }
  ws = null;
  if (workletNode) {
    try { workletNode.disconnect(); } catch (_) {}
    workletNode = null;
  }
  if (sourceNode) {
    try { sourceNode.disconnect(); } catch (_) {}
    sourceNode = null;
  }
  if (audioCtx) {
    try { audioCtx.close(); } catch (_) {}
    audioCtx = null;
  }
}

// ---------- Audio source ----------
async function captureMic() {
  return navigator.mediaDevices.getUserMedia({
    audio: {
      echoCancellation: false,
      noiseSuppression: false,
      autoGainControl: false,
      channelCount: 1,
      sampleSize: 16,
    },
  });
}

async function captureSystemAudio() {
  // getDisplayMedia is the only way to capture tab/window/screen audio in browsers.
  // Chrome requires video:true in the request, but we discard the video tracks after.
  const display = await navigator.mediaDevices.getDisplayMedia({
    video: true,
    audio: {
      echoCancellation: false,
      noiseSuppression: false,
      autoGainControl: false,
    },
  });
  const audioTracks = display.getAudioTracks();
  if (audioTracks.length === 0) {
    display.getTracks().forEach(t => t.stop());
    throw new Error('No audio track. Make sure to tick "Share tab audio" / "Share system audio" in the picker.');
  }
  display.getVideoTracks().forEach(t => t.stop());
  return new MediaStream(audioTracks);
}

// ---------- Main record control ----------
async function start() {
  try {
    const source = sourceSel ? sourceSel.value : 'mic';
    if (source === 'system') {
      setStatus('Pick a tab/window and tick "Share audio"...', 'idle');
      mediaStream = await captureSystemAudio();
    } else {
      setStatus('Requesting microphone access...', 'idle');
      mediaStream = await captureMic();
    }

    recordedChunks = [];
    const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
      ? 'audio/webm;codecs=opus'
      : 'audio/webm';
    mediaRecorder = new MediaRecorder(mediaStream, { mimeType });
    mediaRecorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) recordedChunks.push(e.data);
    };
    mediaRecorder.onstop = () => {
      const blob = new Blob(recordedChunks, { type: mimeType });
      if (recordedBlobUrl) URL.revokeObjectURL(recordedBlobUrl);
      recordedBlobUrl = URL.createObjectURL(blob);
      downloadBtn.disabled = false;
    };
    mediaRecorder.start(1000);

    await startLocalMode();
    startVisualizer(mediaStream);

    isRecording = true;
    startBtn.disabled = true;
    startBtn.classList.add('recording');
    stopBtn.disabled = false;
    if (sourceSel) sourceSel.disabled = true;
    setStatus(`Recording [${source === 'system' ? 'system audio' : 'mic'}]...`, 'recording');

    // If user stops sharing system audio from browser UI, treat as stop
    if (mediaStream) {
      mediaStream.getAudioTracks().forEach(track => {
        track.addEventListener('ended', () => {
          if (isRecording) stop();
        });
      });
    }
  } catch (err) {
    console.error(err);
    setStatus('Failed: ' + err.message, 'error');
    cleanup();
  }
}

function stop() {
  isRecording = false;
  stopLocalMode();
  stopVisualizer();
  if (mediaRecorder && mediaRecorder.state !== 'inactive') mediaRecorder.stop();
  if (mediaStream) {
    mediaStream.getTracks().forEach(t => t.stop());
    mediaStream = null;
  }
  startBtn.disabled = false;
  startBtn.classList.remove('recording');
  stopBtn.disabled = true;
  if (sourceSel) sourceSel.disabled = false;
  setStatus('Stopped — audio ready to download', 'ok');
}

function cleanup() {
  isRecording = false;
  stopLocalMode();
  stopVisualizer();
  if (mediaStream) {
    mediaStream.getTracks().forEach(t => t.stop());
    mediaStream = null;
  }
  startBtn.disabled = false;
  startBtn.classList.remove('recording');
  stopBtn.disabled = true;
  if (sourceSel) sourceSel.disabled = false;
}

function downloadAudio() {
  if (!recordedBlobUrl) return;
  const a = document.createElement('a');
  a.href = recordedBlobUrl;
  a.download = `recording-${new Date().toISOString().replace(/[:.]/g, '-')}.webm`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}

function clearAll() {
  transcriptSegs = [];
  translationSegs = [];
  renderTranscript();
  renderTranslation();
}

// ---------- Events ----------
startBtn.addEventListener('click', start);
stopBtn.addEventListener('click', stop);
downloadBtn.addEventListener('click', downloadAudio);
clearBtn.addEventListener('click', clearAll);

srcLangSel.addEventListener('change', () => {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({
      type: 'config',
      srcLang: langBase(srcLangSel.value),
      tgtLang: tgtLangSel.value,
    }));
  }
});

tgtLangSel.addEventListener('change', () => {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({
      type: 'config',
      srcLang: langBase(srcLangSel.value),
      tgtLang: tgtLangSel.value,
    }));
  }
});

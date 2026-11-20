const API = "http://127.0.0.1:5000";

const messagesEl = document.getElementById("messages");
const inputEl    = document.getElementById("msgInput");
const sendBtn    = document.getElementById("sendBtn");
const micBtn     = document.getElementById("micBtn");
const clearBtn   = document.getElementById("clearBtn");
const docList    = document.getElementById("docList");
const fileInput  = document.getElementById("fileInput");
const uploadArea = document.getElementById("uploadArea");
const uploadStatus = document.getElementById("uploadStatus");

const autoSpeakBtn = document.getElementById("autoSpeakBtn");
let autoSpeak = true; // on by default
let isRecording = false;
let recognition = null;

autoSpeakBtn.addEventListener("click", () => {
  autoSpeak = !autoSpeak;
  autoSpeakBtn.style.opacity = autoSpeak ? "1" : "0.35";
  autoSpeakBtn.title = autoSpeak ? "Auto-speak ON (click to turn off)" : "Auto-speak OFF (click to turn on)";
  if (!autoSpeak) window.speechSynthesis && window.speechSynthesis.cancel();
});

// ── Auto-resize textarea ─────────────────────────────────────────
inputEl.addEventListener("input", () => {
  inputEl.style.height = "auto";
  inputEl.style.height = Math.min(inputEl.scrollHeight, 120) + "px";
});

// Enter to send (Shift+Enter = newline)
inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

sendBtn.addEventListener("click", sendMessage);
clearBtn.addEventListener("click", clearChat);

// ── Send message ─────────────────────────────────────────────────
async function sendMessage() {
  const text = inputEl.value.trim();
  if (!text) return;

  appendUserMsg(text);
  inputEl.value = "";
  inputEl.style.height = "auto";
  sendBtn.disabled = true;

  const typingId = appendTyping();

  try {
    const res = await fetch(`${API}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text }),
    });
    const data = await res.json();
    removeTyping(typingId);

    if (data.error) {
      appendBotMsg(`❌ Error: ${data.error}`, [], true, 0);
    } else {
      appendBotMsg(data.answer, data.sources, data.used_fallback, data.latency_ms);
      if (autoSpeak) speakText(data.answer);
    }
  } catch (err) {
    removeTyping(typingId);
    appendBotMsg("❌ Could not reach the server. Make sure the Python backend is running.", [], true, 0);
  }

  sendBtn.disabled = false;
}

function sendSuggestion(q) {
  inputEl.value = q;
  sendMessage();
}

// ── Append messages ──────────────────────────────────────────────
function appendUserMsg(text) {
  const div = document.createElement("div");
  div.className = "msg user-msg";
  div.innerHTML = `
    <div class="msg-avatar">👤</div>
    <div class="msg-body">
      <div class="msg-bubble">${escHtml(text)}</div>
      <div class="msg-meta">${timeStr()}</div>
    </div>`;
  messagesEl.appendChild(div);
  scrollBottom();
}

function appendBotMsg(text, sources, usedFallback, latencyMs) {
  const div = document.createElement("div");
  div.className = "msg bot-msg";

  const tag = usedFallback
    ? `<span class="tag tag-web">🌐 General Knowledge</span>`
    : `<span class="tag tag-doc">📄 From Documents</span>`;

  const srcHtml = sources && sources.length
    ? `<span>📎 ${sources.join(", ")}</span>`
    : "";

  const latHtml = latencyMs
    ? `<span>⏱ ${latencyMs}ms</span>`
    : "";

  const msgId = "msg_" + Date.now();

  div.innerHTML = `
    <div class="msg-avatar">🤖</div>
    <div class="msg-body">
      <div class="msg-bubble" id="${msgId}">${formatText(text)}</div>
      <div class="msg-meta">
        ${tag}
        ${srcHtml}
        ${latHtml}
        <span>${timeStr()}</span>
        <button class="speak-btn" onclick="speakText(${JSON.stringify(text)})" title="Speak answer">🔊</button>
      </div>
    </div>`;
  messagesEl.appendChild(div);
  scrollBottom();
}

function appendTyping() {
  const id = "typing_" + Date.now();
  const div = document.createElement("div");
  div.className = "msg bot-msg typing";
  div.id = id;
  div.innerHTML = `
    <div class="msg-avatar">🤖</div>
    <div class="msg-body">
      <div class="msg-bubble">
        <span class="dot"></span><span class="dot"></span><span class="dot"></span>
      </div>
    </div>`;
  messagesEl.appendChild(div);
  scrollBottom();
  return id;
}

function removeTyping(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

// ── Speak (browser-side TTS via Web Speech API) ──────────────────
function speakText(text) {
  if (!("speechSynthesis" in window)) {
    alert("Your browser doesn't support voice output. Try Chrome or Edge.");
    return;
  }

  // Cancel any ongoing speech first
  window.speechSynthesis.cancel();

  const utter = new SpeechSynthesisUtterance(text);
  utter.lang  = "en-US";
  utter.rate  = 1.0;
  utter.pitch = 1.0;

  // Pick a natural-sounding voice if available
  const voices = window.speechSynthesis.getVoices();
  const preferred = voices.find(v =>
    v.lang.startsWith("en") && (v.name.includes("Google") || v.name.includes("Natural") || v.name.includes("Female"))
  ) || voices.find(v => v.lang.startsWith("en"));
  if (preferred) utter.voice = preferred;

  window.speechSynthesis.speak(utter);
}

// ── Voice Input (Web Speech API) ─────────────────────────────────
micBtn.addEventListener("click", toggleRecording);

function toggleRecording() {
  if (!("webkitSpeechRecognition" in window) && !("SpeechRecognition" in window)) {
    alert("Your browser doesn't support voice input. Try Chrome.");
    return;
  }

  if (isRecording) {
    recognition && recognition.stop();
    return;
  }

  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  recognition = new SR();
  recognition.lang = "en-US";
  recognition.interimResults = false;

  recognition.onstart = () => {
    isRecording = true;
    micBtn.classList.add("recording");
    micBtn.title = "Recording... click to stop";
  };

  recognition.onresult = (e) => {
    const transcript = e.results[0][0].transcript;
    inputEl.value = transcript;
    inputEl.dispatchEvent(new Event("input"));
    sendMessage();
  };

  recognition.onerror = () => {
    isRecording = false;
    micBtn.classList.remove("recording");
  };

  recognition.onend = () => {
    isRecording = false;
    micBtn.classList.remove("recording");
    micBtn.title = "Voice input";
  };

  recognition.start();
}

// ── File Upload ──────────────────────────────────────────────────
uploadArea.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => handleFile(fileInput.files[0]));

uploadArea.addEventListener("dragover", (e) => { e.preventDefault(); uploadArea.classList.add("drag-over"); });
uploadArea.addEventListener("dragleave", () => uploadArea.classList.remove("drag-over"));
uploadArea.addEventListener("drop", (e) => {
  e.preventDefault();
  uploadArea.classList.remove("drag-over");
  handleFile(e.dataTransfer.files[0]);
});

async function handleFile(file) {
  if (!file) return;
  setUploadStatus("loading", `⏳ Uploading ${file.name}...`);

  const form = new FormData();
  form.append("file", file);

  try {
    const res  = await fetch(`${API}/api/upload`, { method: "POST", body: form });
    const data = await res.json();
    if (data.error) {
      setUploadStatus("error", `❌ ${data.error}`);
    } else {
      setUploadStatus("success", `✅ Indexed ${data.chunks} chunks from ${data.filename}`);
      loadDocs();
    }
  } catch (e) {
    setUploadStatus("error", "❌ Upload failed. Is the server running?");
  }

  fileInput.value = "";
}

function setUploadStatus(type, msg) {
  uploadStatus.textContent = msg;
  uploadStatus.className = "upload-status " + type;
  if (type === "success") setTimeout(() => { uploadStatus.textContent = ""; uploadStatus.className = "upload-status"; }, 4000);
}

// ── Load docs list ───────────────────────────────────────────────
async function loadDocs() {
  try {
    const res  = await fetch(`${API}/api/docs`);
    const data = await res.json();
    if (!data.docs || data.docs.length === 0) {
      docList.innerHTML = `<div class="doc-loading">No documents yet</div>`;
      return;
    }
    docList.innerHTML = data.docs.map(d => `
      <div class="doc-item">
        <div class="doc-icon">${docIcon(d.name)}</div>
        <div class="doc-info">
          <div class="doc-name">${escHtml(d.name)}</div>
          <div class="doc-size">${formatSize(d.size)}</div>
        </div>
      </div>`).join("");
  } catch {
    docList.innerHTML = `<div class="doc-loading">Server offline</div>`;
  }
}

// ── Clear chat ───────────────────────────────────────────────────
function clearChat() {
  messagesEl.innerHTML = `
    <div class="msg bot-msg welcome-msg">
      <div class="msg-avatar">🤖</div>
      <div class="msg-body">
        <div class="msg-bubble">
          <p>Chat cleared. Ask me anything!</p>
          <div class="suggestions">
            <div class="sug-label">Try asking:</div>
            <button class="sug-btn" onclick="sendSuggestion('What is CRR?')">What is CRR?</button>
            <button class="sug-btn" onclick="sendSuggestion('Explain Repo Rate')">Explain Repo Rate</button>
            <button class="sug-btn" onclick="sendSuggestion('What is NPA?')">What is NPA?</button>
            <button class="sug-btn" onclick="sendSuggestion('What is machine learning?')">What is machine learning?</button>
          </div>
        </div>
      </div>
    </div>`;
}

// ── Helpers ──────────────────────────────────────────────────────
function escHtml(s) {
  return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}

function formatText(text) {
  return escHtml(text)
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.*?)\*/g, "<em>$1</em>")
    .replace(/\n/g, "<br>");
}

function timeStr() {
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function formatSize(bytes) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024*1024) return (bytes/1024).toFixed(1) + " KB";
  return (bytes/(1024*1024)).toFixed(1) + " MB";
}

function docIcon(name) {
  const ext = name.split(".").pop().toLowerCase();
  return ext === "pdf" ? "📕" : ext === "docx" ? "📘" : "📄";
}

function scrollBottom() {
  setTimeout(() => messagesEl.scrollTop = messagesEl.scrollHeight, 50);
}

// ── Preload voices (Chrome loads them async) ─────────────────────
if ("speechSynthesis" in window) {
  window.speechSynthesis.getVoices();
  window.speechSynthesis.addEventListener("voiceschanged", () => {
    window.speechSynthesis.getVoices(); // cache them
  });
}

// ── Init ─────────────────────────────────────────────────────────
loadDocs();

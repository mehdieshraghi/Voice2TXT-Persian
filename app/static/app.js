/**
 * Browser microphone → WAV (16 kHz mono) for Vosk.
 * No server-side ffmpeg required.
 */

let mediaStream = null;
let audioContext = null;
let processor = null;
let recordedChunks = [];
let isRecording = false;

function encodeWAV(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  const writeString = (offset, str) => {
    for (let i = 0; i < str.length; i++) {
      view.setUint8(offset + i, str.charCodeAt(i));
    }
  };

  writeString(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(36, "data");
  view.setUint32(40, samples.length * 2, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }

  return new Blob([buffer], { type: "audio/wav" });
}

function downsampleBuffer(buffer, inputRate, outputRate) {
  if (outputRate === inputRate) return buffer;
  const ratio = inputRate / outputRate;
  const newLength = Math.round(buffer.length / ratio);
  const result = new Float32Array(newLength);
  for (let i = 0; i < newLength; i++) {
    result[i] = buffer[Math.round(i * ratio)];
  }
  return result;
}

async function startBrowserRecording() {
  if (isRecording) return;

  mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  audioContext = new AudioContext();
  const source = audioContext.createMediaStreamSource(mediaStream);
  processor = audioContext.createScriptProcessor(4096, 1, 1);
  recordedChunks = [];

  processor.onaudioprocess = (event) => {
    if (!isRecording) return;
    recordedChunks.push(new Float32Array(event.inputBuffer.getChannelData(0)));
  };

  source.connect(processor);
  processor.connect(audioContext.destination);
  isRecording = true;
}

async function stopBrowserRecording(targetSampleRate) {
  if (!isRecording) return null;

  isRecording = false;
  processor?.disconnect();
  mediaStream?.getTracks().forEach((t) => t.stop());

  const inputRate = audioContext?.sampleRate || 48000;
  await audioContext?.close();

  const totalLength = recordedChunks.reduce((sum, c) => sum + c.length, 0);
  const merged = new Float32Array(totalLength);
  let offset = 0;
  for (const chunk of recordedChunks) {
    merged.set(chunk, offset);
    offset += chunk.length;
  }

  const downsampled = downsampleBuffer(merged, inputRate, targetSampleRate);
  return encodeWAV(downsampled, targetSampleRate);
}

function isBrowserRecording() {
  return isRecording;
}

async function uploadWavBlob(blob) {
  const form = new FormData();
  form.append("audio", blob, "recording.wav");
  const response = await fetch("/api/transcribe/file", { method: "POST", body: form });
  return response.json();
}

function showStatus(el, message, type = "info") {
  el.textContent = message;
  el.className = `status show ${type}`;
}

function hideStatus(el) {
  el.className = "status";
  el.textContent = "";
}

function setLoading(btn, loading, label) {
  if (loading) {
    btn.disabled = true;
    btn.dataset.originalText = btn.textContent;
    btn.innerHTML = `<span class="spinner"></span> ${label || "در حال پردازش..."}`;
  } else {
    btn.disabled = false;
    btn.textContent = btn.dataset.originalText || btn.textContent;
  }
}

let installPollTimer = null;

const PHASE_LABELS = {
  preparing: "آماده‌سازی",
  downloading: "در حال دانلود",
  extracting: "استخراج فایل",
  configuring: "ذخیره تنظیمات",
  done: "تکمیل شد",
};

function formatBytes(bytes) {
  if (!bytes || bytes <= 0) return "—";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

function formatSpeed(bps) {
  if (!bps || bps <= 0) return "—";
  if (bps < 1024) return `${Math.round(bps)} B/s`;
  if (bps < 1024 * 1024) return `${(bps / 1024).toFixed(1)} KB/s`;
  return `${(bps / (1024 * 1024)).toFixed(2)} MB/s`;
}

function openInstallModal() {
  const modal = document.getElementById("install-modal");
  if (modal) modal.classList.remove("hidden");
}

function closeInstallModal() {
  const modal = document.getElementById("install-modal");
  if (modal) modal.classList.add("hidden");
}

async function loadModelCatalog() {
  const res = await fetch("/api/models/catalog?lang=fa");
  const data = await res.json();
  if (!data.ok) throw new Error(data.error || "Failed to load catalog");

  const select = document.getElementById("install-model-select");
  if (!select) return data;

  select.innerHTML = "";
  for (const m of data.models) {
    const opt = document.createElement("option");
    opt.value = m.id;
    opt.textContent = `${m.id} (${m.size})${m.recommended ? " — پیشنهادی" : ""} — ${m.description}`;
    if (m.id === data.default_model || m.recommended) opt.selected = true;
    select.appendChild(opt);
  }
  return data;
}

function updateInstallProgress(install) {
  const wrap = document.getElementById("install-progress");
  const fill = document.getElementById("install-progress-fill");
  const text = document.getElementById("install-progress-text");
  const stats = document.getElementById("install-progress-stats");
  const percentEl = document.getElementById("install-progress-percent");
  const phaseEl = document.getElementById("install-progress-phase");
  const status = document.getElementById("install-status");

  if (!wrap || !install) return;

  const progress = install.progress || 0;
  const phase = install.phase || "";

  if (install.status === "running") {
    wrap.classList.remove("hidden");
    if (fill) fill.style.width = `${progress}%`;
    if (percentEl) percentEl.textContent = `${progress}%`;
    if (phaseEl) phaseEl.textContent = PHASE_LABELS[phase] || install.message || "در حال پردازش...";
    if (text) text.textContent = install.message || "";

    if (stats) {
      if (phase === "downloading") {
        const downloaded = install.downloaded_bytes || 0;
        const total = install.total_bytes || 0;
        const speed = install.speed_bps || 0;
        const sizePart = total > 0
          ? `${formatBytes(downloaded)} / ${formatBytes(total)}`
          : `${formatBytes(downloaded)} دانلود شده`;
        stats.textContent = `${sizePart}  ·  ${formatSpeed(speed)}`;
      } else if (phase === "extracting") {
        stats.textContent = "در حال استخراج فایل فشرده...";
      } else if (phase === "configuring") {
        stats.textContent = "به‌روزرسانی settings.json...";
      } else {
        stats.textContent = "";
      }
    }
  } else if (install.status === "done") {
    if (fill) fill.style.width = "100%";
    if (percentEl) percentEl.textContent = "100%";
    if (phaseEl) phaseEl.textContent = PHASE_LABELS.done;
    if (text) text.textContent = "نصب کامل شد. در حال بارگذاری مجدد...";
    if (stats) stats.textContent = "";
    showStatus(status, "مدل با موفقیت نصب شد.", "success");
    setTimeout(() => location.reload(), 800);
  } else if (install.status === "error") {
    wrap.classList.remove("hidden");
    if (stats) stats.textContent = "";
    showStatus(status, install.error || install.message || "خطا در نصب", "error");
    const confirmBtn = document.getElementById("install-confirm-btn");
    if (confirmBtn) setLoading(confirmBtn, false);
  }
}

function pollInstallStatus() {
  if (installPollTimer) clearInterval(installPollTimer);
  installPollTimer = setInterval(async () => {
    try {
      const res = await fetch("/api/models/status");
      const data = await res.json();
      if (!data.ok) return;
      updateInstallProgress(data.install);
      if (data.install?.status === "done" || data.install?.status === "error") {
        clearInterval(installPollTimer);
        installPollTimer = null;
      }
    } catch (_) {
      /* ignore poll errors */
    }
  }, 400);
}

async function startModelInstall() {
  const select = document.getElementById("install-model-select");
  const confirmBtn = document.getElementById("install-confirm-btn");
  const status = document.getElementById("install-status");
  const modelId = select?.value;

  if (!modelId) {
    showStatus(status, "مدلی انتخاب نشده.", "error");
    return;
  }

  hideStatus(status);
  setLoading(confirmBtn, true, "در حال شروع...");
  document.getElementById("install-progress")?.classList.remove("hidden");

  try {
    const res = await fetch("/api/models/install", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_id: modelId }),
    });
    const data = await res.json();
    if (data.ok) {
      pollInstallStatus();
    } else {
      showStatus(status, data.error || "خطا", "error");
      setLoading(confirmBtn, false);
    }
  } catch (err) {
    showStatus(status, err.message, "error");
    setLoading(confirmBtn, false);
  }
}

async function initModelInstall() {
  const cfg = window.VOICE2TXT || {};
  const modal = document.getElementById("install-modal");
  if (!modal) return;

  document.getElementById("open-install-modal")?.addEventListener("click", openInstallModal);
  document.getElementById("install-modal-backdrop")?.addEventListener("click", closeInstallModal);
  document.getElementById("install-dismiss-btn")?.addEventListener("click", () => {
    sessionStorage.setItem("voice2txt_install_dismissed", "1");
    closeInstallModal();
  });
  document.getElementById("install-confirm-btn")?.addEventListener("click", startModelInstall);

  try {
    await loadModelCatalog();
  } catch (err) {
    showStatus(document.getElementById("install-status"), err.message, "error");
  }

  if (!cfg.modelReady) {
    const dismissed = sessionStorage.getItem("voice2txt_install_dismissed");
    const installRes = await fetch("/api/models/status");
    const installData = await installRes.json();
    if (installData.install?.status === "running") {
      openInstallModal();
      pollInstallStatus();
    } else if (!dismissed) {
      openInstallModal();
    }
  }
}

function ensureModelReady(showOn) {
  if (window.VOICE2TXT?.modelReady) return true;
  showStatus(showOn, "ابتدا مدل Vosk را نصب کنید.", "error");
  openInstallModal();
  return false;
}

document.addEventListener("DOMContentLoaded", () => {
  const settingsForm = document.getElementById("settings-form");
  const settingsStatus = document.getElementById("settings-status");
  const outputText = document.getElementById("output-text");
  const transcribeStatus = document.getElementById("transcribe-status");
  const targetRate = parseInt(document.body.dataset.sampleRate || "16000", 10);

  document.querySelectorAll(".main-tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".main-tab").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".main-tab-panel").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById(btn.dataset.mainTab)?.classList.add("active");
    });
  });

  const transcribePage = document.getElementById("page-transcribe");
  transcribePage?.querySelectorAll(".tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      transcribePage.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
      transcribePage.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById(btn.dataset.tab)?.classList.add("active");
    });
  });

  settingsForm?.addEventListener("submit", async (e) => {
    e.preventDefault();
    const formData = new FormData(settingsForm);
    const payload = Object.fromEntries(formData.entries());

    try {
      const res = await fetch("/api/settings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (data.ok) {
        showStatus(settingsStatus, "تنظیمات ذخیره شد.", "success");
        setTimeout(() => location.reload(), 600);
      } else {
        showStatus(settingsStatus, (data.errors || [data.error]).join(" — "), "error");
      }
    } catch (err) {
      showStatus(settingsStatus, err.message, "error");
    }
  });

  const dropZone = document.getElementById("drop-zone");
  const fileInput = document.getElementById("file-input");
  const uploadBtn = document.getElementById("upload-btn");

  dropZone?.addEventListener("click", () => fileInput.click());
  dropZone?.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  });
  dropZone?.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
  dropZone?.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    if (e.dataTransfer.files.length) {
      fileInput.files = e.dataTransfer.files;
    }
  });

  uploadBtn?.addEventListener("click", async () => {
    if (!ensureModelReady(transcribeStatus)) return;
    if (!fileInput.files.length) {
      showStatus(transcribeStatus, "لطفاً یک فایل WAV انتخاب کنید.", "error");
      return;
    }
    hideStatus(transcribeStatus);
    setLoading(uploadBtn, true);

    const form = new FormData();
    form.append("audio", fileInput.files[0]);

    try {
      const res = await fetch("/api/transcribe/file", { method: "POST", body: form });
      const data = await res.json();
      if (data.ok) {
        outputText.value = data.text || "";
        showStatus(
          transcribeStatus,
          data.text ? "تبدیل با موفقیت انجام شد." : "صدایی تشخیص داده نشد.",
          data.text ? "success" : "info"
        );
      } else {
        showStatus(transcribeStatus, data.error || "خطا در تبدیل", "error");
      }
    } catch (err) {
      showStatus(transcribeStatus, err.message, "error");
    } finally {
      setLoading(uploadBtn, false);
    }
  });

  const browserRecordBtn = document.getElementById("browser-record-btn");

  browserRecordBtn?.addEventListener("click", async () => {
    hideStatus(transcribeStatus);

    if (!isBrowserRecording()) {
      if (!ensureModelReady(transcribeStatus)) return;
      try {
        await startBrowserRecording();
        browserRecordBtn.textContent = "⏹ توقف ضبط";
        browserRecordBtn.classList.add("recording");
        showStatus(transcribeStatus, "در حال ضبط از میکروفون مرورگر...", "info");
      } catch (err) {
        showStatus(transcribeStatus, "دسترسی به میکروفون رد شد: " + err.message, "error");
      }
      return;
    }

    setLoading(browserRecordBtn, true, "در حال تبدیل...");
    browserRecordBtn.classList.remove("recording");

    try {
      const blob = await stopBrowserRecording(targetRate);
      browserRecordBtn.textContent = "🎤 ضبط با مرورگر";
      if (!blob) return;

      const data = await uploadWavBlob(blob);
      if (data.ok) {
        outputText.value = data.text || "";
        showStatus(
          transcribeStatus,
          data.text ? "تبدیل با موفقیت انجام شد." : "صدایی تشخیص داده نشد.",
          data.text ? "success" : "info"
        );
      } else {
        showStatus(transcribeStatus, data.error || "خطا", "error");
      }
    } catch (err) {
      showStatus(transcribeStatus, err.message, "error");
    } finally {
      setLoading(browserRecordBtn, false);
      browserRecordBtn.textContent = "🎤 ضبط با مرورگر";
    }
  });

  const serverRecordBtn = document.getElementById("server-record-btn");
  serverRecordBtn?.addEventListener("click", async () => {
    if (!ensureModelReady(transcribeStatus)) return;
    hideStatus(transcribeStatus);
    setLoading(serverRecordBtn, true);

    const duration = parseInt(document.getElementById("record-duration-live")?.value || "5", 10);

    try {
      const res = await fetch("/api/transcribe/record", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ duration }),
      });
      const data = await res.json();
      if (data.ok) {
        outputText.value = data.text || "";
        showStatus(
          transcribeStatus,
          data.text ? `تبدیل ${data.duration}s انجام شد.` : "صدایی تشخیص داده نشد.",
          data.text ? "success" : "info"
        );
      } else {
        showStatus(transcribeStatus, data.error || "خطا", "error");
      }
    } catch (err) {
      showStatus(transcribeStatus, err.message, "error");
    } finally {
      setLoading(serverRecordBtn, false);
    }
  });

  document.getElementById("copy-btn")?.addEventListener("click", () => {
    navigator.clipboard.writeText(outputText.value);
    showStatus(transcribeStatus, "متن کپی شد.", "success");
  });

  document.getElementById("save-btn")?.addEventListener("click", async () => {
    const text = outputText.value.trim();
    if (!text) {
      showStatus(transcribeStatus, "متنی برای ذخیره وجود ندارد.", "error");
      return;
    }
    try {
      const res = await fetch("/api/save", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      const data = await res.json();
      if (data.ok) {
        showStatus(transcribeStatus, `ذخیره شد: ${data.path}`, "success");
      } else {
        showStatus(transcribeStatus, data.error, "error");
      }
    } catch (err) {
      showStatus(transcribeStatus, err.message, "error");
    }
  });

  document.getElementById("clear-btn")?.addEventListener("click", () => {
    outputText.value = "";
    hideStatus(transcribeStatus);
  });

  initModelInstall();
});

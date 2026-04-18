/* app.js – Invoice AI frontend logic */

const dropZone     = document.getElementById("dropZone");
const fileInput    = document.getElementById("fileInput");
const processBtn   = document.getElementById("processBtn");
const clearBtn     = document.getElementById("clearBtn");
const previewStrip = document.getElementById("previewStrip");
const previewImg   = document.getElementById("previewImg");
const spinner      = document.getElementById("spinner");
const spinnerMsg   = document.getElementById("spinnerMsg");
const results      = document.getElementById("results");
const rejectedBanner = document.getElementById("rejectedBanner");
const rejectedReason = document.getElementById("rejectedReason");

let currentFile = null;
let currentResult = null;

const SPINNER_MSGS = [
  "Preprocessing image…",
  "Running OCR engine…",
  "Translating content…",
  "Summarising invoice…",
];
let spinnerInterval = null;

// ── Drag & Drop ───────────────────────────────────────────────────────────
dropZone.addEventListener("click", () => fileInput.click());

dropZone.addEventListener("dragover", e => {
  e.preventDefault();
  dropZone.classList.add("drag-over");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
dropZone.addEventListener("drop", e => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  const f = e.dataTransfer.files[0];
  if (f) loadFile(f);
});

fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) loadFile(fileInput.files[0]);
});

function loadFile(file) {
  if (!file.type.startsWith("image/")) {
    alert("Please upload an image file (JPG, PNG, etc.)");
    return;
  }
  currentFile = file;
  const reader = new FileReader();
  reader.onload = e => {
    previewImg.src  = e.target.result;
    previewStrip.style.display = "flex";
    processBtn.disabled = false;
    hideResults();
  };
  reader.readAsDataURL(file);
}

clearBtn.addEventListener("click", () => {
  currentFile = null;
  fileInput.value = "";
  previewStrip.style.display = "none";
  processBtn.disabled = true;
  hideResults();
});

// ── Process ───────────────────────────────────────────────────────────────
processBtn.addEventListener("click", runPipeline);

async function runPipeline() {
  if (!currentFile) return;

  hideResults();
  showSpinner();
  processBtn.disabled = true;

  const srcLang = document.getElementById("srcLang").value;
  const tgtLang = document.getElementById("tgtLang").value;

  // Convert to base64
  const b64 = await fileToBase64(currentFile);

  try {
    const res = await fetch("/process", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_b64: b64, src_lang: srcLang, tgt_lang: tgtLang }),
    });

    const data = await res.json();
    hideSpinner();
    processBtn.disabled = false;

    if (data.error) {
      alert("Error: " + data.error);
      return;
    }

    currentResult = data;

    if (data.status === "rejected") {
      rejectedReason.textContent = data.reject_reason || "Document does not appear to be a bill or invoice.";
      rejectedBanner.style.display = "flex";
    } else {
      showResults(data);
    }

  } catch (err) {
    hideSpinner();
    processBtn.disabled = false;
    alert("Connection error: " + err.message);
  }
}

function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload  = () => resolve(r.result);
    r.onerror = reject;
    r.readAsDataURL(file);
  });
}

// ── Results ───────────────────────────────────────────────────────────────
function showResults(data) {
  // Meta chips
  document.getElementById("metaLines").textContent = `${data.lines} lines`;
  document.getElementById("metaWords").textContent = `${data.words} words`;
  document.getElementById("metaTime").textContent  = `${data.processing_time_s}s`;
  document.getElementById("metaConf").textContent  = `Invoice confidence: ${(data.confidence * 100).toFixed(0)}%`;

  document.getElementById("extractedText").textContent  = data.extracted_text  || "(nothing extracted)";
  document.getElementById("translatedText").textContent = data.translated_text || "(no translation)";
  document.getElementById("summaryText").textContent    = data.summary         || "(no summary)";

  results.style.display = "block";

  // Activate first tab
  switchTab("extracted");
}

function hideResults() {
  results.style.display        = "none";
  rejectedBanner.style.display = "none";
}

// ── Tabs ──────────────────────────────────────────────────────────────────
document.querySelectorAll(".tab").forEach(btn => {
  btn.addEventListener("click", () => switchTab(btn.dataset.tab));
});

function switchTab(name) {
  document.querySelectorAll(".tab").forEach(b => b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll(".tab-content").forEach(c => {
    c.classList.toggle("active", c.id === `tab-${name}`);
  });
}

// ── Export ────────────────────────────────────────────────────────────────
document.getElementById("copyBtn").addEventListener("click", () => {
  if (!currentResult) return;
  const text = [
    "=== EXTRACTED ===\n" + currentResult.extracted_text,
    "=== TRANSLATED ===\n" + currentResult.translated_text,
    "=== SUMMARY ===\n" + currentResult.summary,
  ].join("\n\n");
  navigator.clipboard.writeText(text).then(() => flash("copyBtn", "✓ Copied!"));
});

document.getElementById("downloadBtn").addEventListener("click", () => {
  if (!currentResult) return;
  const text = [
    "INVOICE AI OUTPUT",
    "=================",
    `Source language : ${currentResult.src_lang}`,
    `Target language : ${currentResult.tgt_lang}`,
    `Processing time : ${currentResult.processing_time_s}s`,
    "",
    "EXTRACTED TEXT",
    "--------------",
    currentResult.extracted_text,
    "",
    "TRANSLATED TEXT",
    "---------------",
    currentResult.translated_text,
    "",
    "SUMMARY",
    "-------",
    currentResult.summary,
  ].join("\n");

  const blob = new Blob([text], { type: "text/plain" });
  const a    = document.createElement("a");
  a.href     = URL.createObjectURL(blob);
  a.download = "invoice_result.txt";
  a.click();
});

function flash(id, msg) {
  const btn = document.getElementById(id);
  const orig = btn.textContent;
  btn.textContent = msg;
  setTimeout(() => btn.textContent = orig, 2000);
}

// ── Spinner ───────────────────────────────────────────────────────────────
function showSpinner() {
  spinner.style.display = "flex";
  let i = 0;
  spinnerMsg.textContent = SPINNER_MSGS[0];
  spinnerInterval = setInterval(() => {
    i = (i + 1) % SPINNER_MSGS.length;
    spinnerMsg.textContent = SPINNER_MSGS[i];
  }, 3000);
}

function hideSpinner() {
  spinner.style.display = "none";
  clearInterval(spinnerInterval);
}

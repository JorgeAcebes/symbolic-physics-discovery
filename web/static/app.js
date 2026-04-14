/* ── Model catalogue ─────────────────────────────────────────────────────────── */
const MODELS = [
  {
    id: "MLP_Standard",
    name: "MLP Standard",
    desc: "Red neuronal densa con activación SiLU. Input→64→64→1.",
    tag: "nn",
    tagLabel: "Neural Net",
    params: [
      { key: "epochs", label: "Épocas",        type: "number", default: 100,   min: 1,       max: 5000,  step: 1     },
      { key: "lr",     label: "Learning Rate",  type: "number", default: 0.001, min: 0.00001, max: 1,     step: 0.0001 },
    ],
  },
  {
    id: "MLP_Sparse",
    name: "MLP Sparse (L1)",
    desc: "Arquitectura estrecha con regularización L1. Fuerza interpretabilidad.",
    tag: "nn",
    tagLabel: "Neural Net",
    params: [
      { key: "epochs",   label: "Épocas",     type: "number", default: 1000,  min: 1,       max: 10000, step: 1     },
      { key: "lr",       label: "Learning Rate", type: "number", default: 0.001, min: 0.00001, max: 1, step: 0.0001 },
      { key: "l1_alpha", label: "L1 Alpha",    type: "number", default: 0.001, min: 0.00001, max: 1,     step: 0.0001 },
    ],
  },
  {
    id: "MLP_Dropout",
    name: "MLP MC-Dropout",
    desc: "Dropout Monte Carlo al 20%. Cuantifica incertidumbre bayesiana.",
    tag: "nn",
    tagLabel: "Neural Net",
    params: [
      { key: "epochs",    label: "Épocas",      type: "number", default: 500, min: 1,    max: 5000, step: 1  },
      { key: "lr",        label: "Learning Rate", type: "number", default: 0.001, min: 0.00001, max: 1, step: 0.0001 },
      { key: "mc_samples", label: "Muestras MC", type: "number", default: 100, min: 10, max: 1000, step: 1 },
    ],
  },
  {
    id: "Polynomial",
    name: "Polynomial Regression",
    desc: "Regresión polinomial multivariable con recuperación analítica exacta.",
    tag: "math",
    tagLabel: "Analítico",
    params: [
      { key: "degree", label: "Grado", type: "number", default: 3, min: 1, max: 10, step: 1 },
    ],
  },
  {
    id: "PySR",
    name: "PySR",
    desc: "Regresión simbólica con algoritmos genéticos (Julia engine). Descubre la ecuación exacta.",
    tag: "sr",
    tagLabel: "Simbólico",
    params: [
      { key: "niterations", label: "Iteraciones",    type: "number", default: 50,  min: 1, max: 500,  step: 1 },
      { key: "populations", label: "Poblaciones",    type: "number", default: 10,  min: 1, max: 100,  step: 1 },
      { key: "maxsize",     label: "Max Complejidad", type: "number", default: 15,  min: 5, max: 50,   step: 1 },
    ],
  },
  {
    id: "GPLearn",
    name: "GPLearn",
    desc: "Programación genética en Python. Evoluciona expresiones simbólicas.",
    tag: "sr",
    tagLabel: "Simbólico",
    params: [
      { key: "generations",    label: "Generaciones", type: "number", default: 30,   min: 1,   max: 300,   step: 1 },
      { key: "population_size", label: "Población",    type: "number", default: 2000, min: 100, max: 10000, step: 100 },
    ],
  },
  {
    id: "PySINDy",
    name: "PySINDy",
    desc: "Identificación dispersa de dinámicas no lineales. Optimizador STLSQ.",
    tag: "sr",
    tagLabel: "Simbólico",
    params: [
      { key: "degree",    label: "Grado base",    type: "number", default: 3,   min: 1, max: 6,  step: 1    },
      { key: "threshold", label: "Umbral STLSQ",  type: "number", default: 0.1, min: 0.001, max: 1, step: 0.01 },
    ],
  },
  {
    id: "QLattice",
    name: "QLattice",
    desc: "Retículo probabilístico bayesiano. Requiere feyn instalado.",
    tag: "sr",
    tagLabel: "Simbólico",
    params: [
      { key: "epochs",         label: "Épocas",     type: "number", default: 15, min: 1,  max: 100, step: 1 },
      { key: "max_complexity", label: "Complejidad", type: "number", default: 7,  min: 2,  max: 20,  step: 1 },
    ],
  },
];

const TAG_CLASS = { nn: "tag-nn", sr: "tag-sr", math: "tag-math" };

/* ── State ───────────────────────────────────────────────────────────────────── */
const S = {
  activeTab:       "default",   // "default" | "upload"
  selectedDataset: null,        // { type, name?, file_id?, target_col, column_map }
  uploadedMeta:    null,        // { file_id, columns, preview, n_rows }
  colMap:          {},          // { original_name: new_name }
  targetCol:       null,
  selectedModels:  new Set(),
  hyperparams:     {},          // { model_id: { param: value } }
  jobId:           null,
  pollTimer:       null,
  logCount:        0,
};

/* ── DOM helpers ─────────────────────────────────────────────────────────────── */
const $  = id => document.getElementById(id);
const el = (tag, cls, html) => {
  const e = document.createElement(tag);
  if (cls)  e.className = cls;
  if (html) e.innerHTML = html;
  return e;
};

/* ── Init ────────────────────────────────────────────────────────────────────── */
document.addEventListener("DOMContentLoaded", () => {
  buildModelGrid();
  loadDefaultDatasets();
  setupDropZone();
  setupEventListeners();
});

function setupEventListeners() {
  // Tab switching
  document.querySelectorAll(".tab-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const tab = btn.dataset.tab;
      document.querySelectorAll(".tab-btn").forEach(b => b.classList.toggle("active", b === btn));
      document.querySelectorAll(".tab-pane").forEach(p => p.classList.toggle("active", p.id === `tab-${tab}`));
      S.activeTab = tab;
      if (tab === "default") { S.selectedDataset = null; S.uploadedMeta = null; }
      updateRunBtn();
    });
  });

  // Advanced toggle
  $("adv-toggle").addEventListener("click", () => {
    const body = $("adv-body");
    const tog  = $("adv-toggle");
    const open = body.classList.toggle("open");
    tog.classList.toggle("open", open);
    renderAdvancedOptions();
  });

  // Run
  $("run-btn").addEventListener("click", runExperiment);

  // Lightbox close
  $("lightbox").addEventListener("click", e => {
    if (e.target === $("lightbox") || e.target.closest(".lightbox-close")) {
      $("lightbox").classList.remove("open");
    }
  });
  document.addEventListener("keydown", e => {
    if (e.key === "Escape") $("lightbox").classList.remove("open");
  });
}

/* ── Default datasets ────────────────────────────────────────────────────────── */
async function loadDefaultDatasets() {
  try {
    const res  = await fetch("/api/datasets");
    const data = await res.json();
    renderDatasetCards(data);
  } catch {
    $("dataset-grid").innerHTML =
      `<p style="color:var(--text-muted);font-size:13px">⚠ No se pudieron cargar datasets. ¿Está en la carpeta data/?</p>`;
  }
}

function renderDatasetCards(datasets) {
  const grid = $("dataset-grid");
  grid.innerHTML = "";
  if (!datasets.length) {
    grid.innerHTML = `<p style="color:var(--text-muted);font-size:13px">No hay CSVs en data/</p>`;
    return;
  }
  datasets.forEach(ds => {
    const card = el("div", "dataset-card");
    card.innerHTML = `
      <div class="ds-name">${ds.name}</div>
      <div class="ds-cols">${ds.columns.join(" · ")}</div>`;
    card.addEventListener("click", () => selectDefaultDataset(ds, card));
    grid.appendChild(card);
  });
}

function selectDefaultDataset(ds, card) {
  document.querySelectorAll(".dataset-card").forEach(c => c.classList.remove("selected"));
  card.classList.add("selected");

  // Show inline target selector
  const targetCol = ds.columns[ds.columns.length - 1]; // default: last column
  S.selectedDataset = {
    type:       "default",
    name:       ds.name,
    target_col: targetCol,
    column_map: {},
  };
  showInlineTargetSelector(ds.columns, "default-target-row");
  updateRunBtn();
}

function showInlineTargetSelector(columns, rowId) {
  let row = $(rowId);
  if (!row) {
    row = el("div", "", "");
    row.id = rowId;
    row.style.marginTop = "16px";
    $("tab-default").appendChild(row);
  }
  row.innerHTML = `
    <div style="font-size:12px;color:var(--text-muted);margin-bottom:8px;font-family:var(--font-mono)">
      Columna objetivo (target):
    </div>
    <div style="display:flex;flex-wrap:wrap;gap:8px;">
      ${columns.map(c => `
        <label style="display:flex;align-items:center;gap:6px;cursor:pointer;font-family:var(--font-mono);font-size:12px;color:var(--text)">
          <input type="radio" name="default-target" value="${c}"
            ${c === columns[columns.length-1] ? "checked" : ""}
            style="accent-color:var(--accent)">
          ${c}
        </label>`).join("")}
    </div>`;
  row.querySelectorAll("input[name=default-target]").forEach(inp => {
    inp.addEventListener("change", () => {
      if (S.selectedDataset) S.selectedDataset.target_col = inp.value;
    });
  });
}

/* ── Upload ──────────────────────────────────────────────────────────────────── */
function setupDropZone() {
  const zone  = $("drop-zone");
  const input = $("file-input");

  zone.addEventListener("dragover",  e => { e.preventDefault(); zone.classList.add("drag-over"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("drag-over"));
  zone.addEventListener("drop",      e => {
    e.preventDefault();
    zone.classList.remove("drag-over");
    const file = e.dataTransfer.files[0];
    if (file) handleFileUpload(file);
  });
  input.addEventListener("change", () => {
    if (input.files[0]) handleFileUpload(input.files[0]);
  });
}

async function handleFileUpload(file) {
  $("upload-status").textContent = `Analizando ${file.name}…`;
  const formData = new FormData();
  formData.append("file", file);

  try {
    const res  = await fetch("/api/parse", { method: "POST", body: formData });
    const data = await res.json();
    if (data.error) { $("upload-status").textContent = `${data.error}`; return; }

    S.uploadedMeta = data;
    $("upload-status").textContent =
      `✓ ${file.name} — ${data.n_rows} filas, ${data.columns.length} columnas`;
    renderColumnConfigurator(data);
  } catch (e) {
    $("upload-status").textContent = `Error: ${e.message}`;
  }
}

function renderColumnConfigurator(meta) {
  const container = $("col-config");
  container.classList.remove("hidden");

  const rowsDiv = container.querySelector(".col-rows");
  rowsDiv.innerHTML = "";

  // Header row
  const header = el("div", "col-row", "");
  header.style.background = "none";
  header.style.border = "none";
  header.style.fontFamily = "var(--font-mono)";
  header.style.fontSize = "10px";
  header.style.color = "var(--text-muted)";
  header.innerHTML = `<span>Columna detectada / muestra</span><span>Nombre a usar</span><span title="Target">▶</span>`;
  rowsDiv.appendChild(header);

  meta.columns.forEach((col, idx) => {
    const preview = (meta.preview || []).slice(0, 3).map(r => r[idx]).join(", ");
    const row = el("div", "col-row", "");
    row.innerHTML = `
      <div class="col-preview">
        <span class="col-detected">${col}</span>
        <span>${preview}</span>
      </div>
      <input class="col-name-input" type="text" value="${col}" data-original="${col}" placeholder="${col}">
      <input class="col-target-radio" type="radio" name="upload-target" value="${col}"
        ${idx === meta.columns.length - 1 ? "checked" : ""}>`;
    rowsDiv.appendChild(row);
  });

  // Wire up events
  container.querySelectorAll(".col-name-input").forEach(inp => {
    inp.addEventListener("input", updateUploadDataset);
  });
  container.querySelectorAll(".col-target-radio").forEach(inp => {
    inp.addEventListener("change", updateUploadDataset);
  });

  updateUploadDataset();
}

function updateUploadDataset() {
  if (!S.uploadedMeta) return;
  const container = $("col-config");
  const colMap = {};
  container.querySelectorAll(".col-name-input").forEach(inp => {
    const orig = inp.dataset.original;
    const newName = inp.value.trim() || orig;
    if (newName !== orig) colMap[orig] = newName;
  });

  const targetRadio = container.querySelector(".col-target-radio:checked");
  const origTarget = targetRadio ? targetRadio.value : S.uploadedMeta.columns.slice(-1)[0];
  const finalTarget = colMap[origTarget] || origTarget;

  S.selectedDataset = {
    type:       "upload",
    file_id:    S.uploadedMeta.file_id,
    target_col: finalTarget,
    column_map: colMap,
  };
  updateRunBtn();
}

/* ── Model selection ─────────────────────────────────────────────────────────── */
function buildModelGrid() {
  const grid = $("model-grid");
  MODELS.forEach(m => {
    const card = el("div", "model-card");
    card.dataset.id = m.id;
    card.innerHTML = `
      <div class="model-name">${m.name}</div>
      <div class="model-desc">${m.desc}</div>
      <span class="model-tag ${TAG_CLASS[m.tag]}">${m.tagLabel}</span>`;
    card.addEventListener("click", () => toggleModel(m.id, card));
    grid.appendChild(card);
  });
}

function toggleModel(id, card) {
  if (S.selectedModels.has(id)) {
    S.selectedModels.delete(id);
    card.classList.remove("selected");
  } else {
    S.selectedModels.add(id);
    card.classList.add("selected");
  }
  // Re-render advanced options if open
  if ($("adv-body").classList.contains("open")) {
    renderAdvancedOptions();
  }
  updateRunBtn();
}

/* ── Advanced options ────────────────────────────────────────────────────────── */
function renderAdvancedOptions() {
  const body = $("adv-body");
  body.innerHTML = "";

  const selected = MODELS.filter(m => S.selectedModels.has(m.id));
  if (!selected.length) {
    body.innerHTML = `<p style="color:var(--text-muted);font-size:13px">Selecciona al menos un modelo primero.</p>`;
    return;
  }

  selected.forEach(m => {
    const section = el("div", "adv-model-section");
    section.innerHTML = `<div class="adv-model-title">${m.name}</div>`;
    const grid = el("div", "param-grid");

    m.params.forEach(p => {
      const current = S.hyperparams[m.id]?.[p.key] ?? p.default;
      const field = el("div", "param-field");
      field.innerHTML = `
        <label>${p.label} <span style="color:var(--text-dim)">(default: ${p.default})</span></label>
        <input type="${p.type}" value="${current}"
          min="${p.min ?? ""}" max="${p.max ?? ""}" step="${p.step ?? 1}"
          data-model="${m.id}" data-param="${p.key}">`;
      field.querySelector("input").addEventListener("input", e => {
        if (!S.hyperparams[m.id]) S.hyperparams[m.id] = {};
        S.hyperparams[m.id][e.target.dataset.param] =
          p.type === "number" ? parseFloat(e.target.value) : e.target.value;
      });
      grid.appendChild(field);
    });

    section.appendChild(grid);
    body.appendChild(section);
  });
}

/* ── Run button state ────────────────────────────────────────────────────────── */
function updateRunBtn() {
  const btn   = $("run-btn");
  const ready = S.selectedDataset && S.selectedModels.size > 0;
  btn.disabled = !ready;
  $("run-status").textContent = !S.selectedDataset
    ? "Elige un dataset primero"
    : !S.selectedModels.size
    ? "Selecciona al menos un modelo"
    : "";
}

/* ── Run experiment ──────────────────────────────────────────────────────────── */
async function runExperiment() {
  if (!S.selectedDataset || !S.selectedModels.size) return;

  // Reset
  clearLog();
  $("results-section").style.display = "none";
  $("log-section").style.display = "block";
  $("run-btn").disabled = true;
  $("run-status").innerHTML = `<span class="spinner"></span>`;

  const payload = {
    dataset:    S.selectedDataset,
    models:     Array.from(S.selectedModels),
    hyperparams: S.hyperparams,
  };

  try {
    const res  = await fetch("/api/run", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(payload),
    });
    const { job_id } = await res.json();
    S.jobId    = job_id;
    S.logCount = 0;
    startPolling(job_id);
  } catch (e) {
    logLine(`Error lanzando experimento: ${e.message}`, "error");
    $("run-btn").disabled = false;
    $("run-status").textContent = "";
  }
}

/* ── Polling ─────────────────────────────────────────────────────────────────── */
function startPolling(jobId) {
  stopPolling();
  S.pollTimer = setInterval(() => pollJob(jobId), 1800);
  pollJob(jobId);
}

function stopPolling() {
  if (S.pollTimer) { clearInterval(S.pollTimer); S.pollTimer = null; }
}

async function pollJob(jobId) {
  try {
    const res = await fetch(`/api/job/${jobId}`);
    const job = await res.json();

    // Append new log lines
    const newLines = (job.log || []).slice(S.logCount);
    newLines.forEach(line => logLine(line));
    S.logCount = (job.log || []).length;

    if (job.status === "done") {
      stopPolling();
      $("run-btn").disabled = false;
      $("run-status").innerHTML = `<span class="badge badge-done">✓ completado</span>`;
      displayResults(job);
    } else if (job.status === "error") {
      stopPolling();
      $("run-btn").disabled = false;
      $("run-status").innerHTML = `<span class="badge badge-error">✗ error</span>`;
    }
  } catch {/* network hiccup, keep polling */}
}

/* ── Terminal log ────────────────────────────────────────────────────────────── */
function clearLog() {
  $("terminal").innerHTML = `<span class="cursor"></span>`;
}

function logLine(msg, cls = "") {
  const term  = $("terminal");
  const cursor = term.querySelector(".cursor");

  const line = el("div", `log-line${cls ? " " + cls : ""}`);
  if (msg.toLowerCase().includes("error") || msg.includes("traceback")) {
    line.className = "log-line error";
  } else if (msg.startsWith("✓")) {
    line.className = "log-line success";
  } else if (msg.startsWith("⚠")) {
    line.className = "log-line warn";
  }
  line.textContent = msg;

  if (cursor) term.insertBefore(line, cursor);
  else term.appendChild(line);

  term.scrollTop = term.scrollHeight;
}

/* ── Display results ─────────────────────────────────────────────────────────── */
function displayResults(job) {
  const section = $("results-section");
  section.style.display = "block";
  section.scrollIntoView({ behavior: "smooth", block: "start" });

  // Metric cards
  const cardsDiv = $("metric-cards");
  cardsDiv.innerHTML = "";
  (job.summary || []).forEach(r => {
    const card = el("div", r.error ? "metric-card has-error" : "metric-card");
    card.innerHTML = r.error
      ? `<div class="mc-model">${r.model}</div>
         <div class="mc-error">${r.error}</div>`
      : `<div class="mc-model">${r.model}</div>
         <div class="mc-values">
           <div class="mc-val"><span class="mc-label">MSE</span><span class="mc-number">${fmtSci(r.mse)}</span></div>
           <div class="mc-val"><span class="mc-label">MAE</span><span class="mc-number">${fmtSci(r.mae)}</span></div>
         </div>
         <div class="mc-eq">${escHtml(String(r.equation))}</div>`;
    cardsDiv.appendChild(card);
  });

  // Gallery
  const images = (job.files || []).filter(f => f.type === "image");
  const galleryDiv = $("gallery-grid");
  galleryDiv.innerHTML = "";

  if (images.length) {
    $("gallery-section").style.display = "block";
    images.forEach(f => {
      const item = el("div", "gallery-item");
      const imgSrc = `/api/job/${job_id_from_url(job)}/image/${f.path}`;
      item.innerHTML = `
        <img src="${imgSrc}" alt="${f.name}" loading="lazy">
        <div class="gallery-item-footer">
          <span class="gallery-item-name">${f.name}</span>
          <a href="/api/job/${job_id_from_url(job)}/download/${f.path}"
             class="btn btn-ghost btn-sm" download>↓</a>
        </div>`;
      item.querySelector("img").addEventListener("click", () => openLightbox(imgSrc));
      galleryDiv.appendChild(item);
    });
  } else {
    $("gallery-section").style.display = "none";
  }

  // Downloads (text files)
  const texts = (job.files || []).filter(f => f.type === "text" && f.name !== "summary.json");
  const dlDiv = $("download-list");
  dlDiv.innerHTML = "";
  texts.forEach(f => {
    const a = el("a", "btn btn-ghost btn-sm");
    a.href     = `/api/job/${job_id_from_url(job)}/download/${f.path}`;
    a.download = f.name;
    a.textContent = `↓ ${f.name}`;
    dlDiv.appendChild(a);
  });
  // Also offer summary.json
  const summaryLink = el("a", "btn btn-ghost btn-sm");
  summaryLink.href     = `/api/job/${job_id_from_url(job)}/download/summary.json`;
  summaryLink.download = "summary.json";
  summaryLink.textContent = "↓ summary.json";
  dlDiv.appendChild(summaryLink);
}

/* ── Lightbox ────────────────────────────────────────────────────────────────── */
function openLightbox(src) {
  $("lightbox-img").src = src;
  $("lightbox").classList.add("open");
}

/* ── Helpers ─────────────────────────────────────────────────────────────────── */
function job_id_from_url(job) {
  // The job object doesn't carry its ID, but we stored it in S.jobId
  return S.jobId;
}

function fmtSci(n) {
  if (n === undefined || n === null) return "—";
  return n.toExponential(3);
}

function escHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
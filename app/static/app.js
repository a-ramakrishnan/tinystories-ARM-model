// TinyStories LM Studio: front end for app/server.py
// Polls /api/status once a second and renders each step from it.

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (n) => Number(n).toLocaleString();
const fmtLoss = (x) => (x == null ? "–" : Number(x).toFixed(3));
const cap = (x) => (x ? x[0].toUpperCase() + x.slice(1) : "");
const short = (n) => (n >= 1e12 ? `${(n / 1e12).toFixed(1)}T` : n >= 1e9 ? `${(n / 1e9).toFixed(1)}B` : n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : n >= 1e3 ? `${(n / 1e3).toFixed(1)}K` : `${n}`);
const bytes = (n) => (n >= 1e9 ? `${(n / 1e9).toFixed(1)} GB` : n >= 1e6 ? `${(n / 1e6).toFixed(1)} MB` : `${Math.max(1, Math.round(n / 1e3))} KB`);
const pctText = (p) => `${p < 1 ? p.toFixed(2) : p < 10 ? p.toFixed(1) : p.toFixed(0)}%`;
const shortUrl = (u) => { try { const x = new URL(u); return `${x.hostname} / ${x.pathname.split("/").pop()}`; } catch (e) { return u; } };
const dsLabel = (d) => (d.name.startsWith("http") ? shortUrl(d.name) : d.name) + (d.config_name ? ` · ${d.config_name}` : "");

const STEPS = [
  { id: "dataset", title: "Dataset", sub: "Choose training data" },
  { id: "tokenize", title: "Tokenize", sub: "Text → numbers" },
  { id: "batches", title: "Batches", sub: "Training windows" },
  { id: "model", title: "Model", sub: "Build the transformer" },
  { id: "attention", title: "Inside the transformer", sub: "Attention, step by step" },
  { id: "train", title: "Train", sub: "Watch it learn" },
  { id: "generate", title: "Generate", sub: "Write text" },
  { id: "models", title: "Models", sub: "Save, switch, compare" },
  { id: "lab", title: "Lab", sub: "Design, train, compare", explore: true, icon: "⚗" },
  { id: "journey", title: "Journey", sub: "How we got here", explore: true, icon: "⏱" },
];

const TRAIN_PRESETS = {
  quick: { max_iters: 2000, learning_rate: 6e-4, min_lr: 6e-5, warmup_steps: 200, eval_interval: 250, eval_iters: 50, gradient_accumulation_steps: 1 },
  full: { max_iters: 20000, learning_rate: 6e-4, min_lr: 6e-5, warmup_steps: 1000, eval_interval: 1000, eval_iters: 100, gradient_accumulation_steps: 1 },
};
const TRAIN_FIELDS = Object.keys(TRAIN_PRESETS.quick);

const MODEL_PRESETS = {
  tiny: { label: "Tiny", n_layer: 4, n_head: 4, n_embd: 256 },
  small: { label: "Small", n_layer: 6, n_head: 6, n_embd: 384 },
  medium: { label: "Medium", n_layer: 8, n_head: 8, n_embd: 512 },
  large: { label: "Large", n_layer: 12, n_head: 12, n_embd: 768 },
};
const VOCAB = 50257;
const BYTES_PER_PARAM = 16;

const ui = {
  step: "dataset", status: null, prevJob: null, filled: false,
  batchSampled: false, generated: false, inspected: false,
  sample: { key: null, split: "train", index: 0, loadedFor: null },
  walk: null, real: null, realHead: "avg",
};
try { ui.step = localStorage.getItem("lmstudio.step") || "dataset"; } catch (e) { /* storage unavailable */ }

// ---------- API ----------
async function api(path, body) {
  const opts = body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  const res = await fetch(path, opts);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const d = data.detail;
    throw new Error(Array.isArray(d) ? d.map((x) => `${x.loc?.slice(-1)[0]}: ${x.msg}`).join("; ") : d || res.statusText);
  }
  return data;
}

let toastTimer;
function toast(msg, isError = false) {
  const t = $("#toast");
  t.textContent = msg;
  t.className = "toast" + (isError ? " error" : "");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.add("hidden"), isError ? 6000 : 3000);
}

async function action(btn, fn) {
  if (btn) btn.disabled = true;
  try { return await fn(); }
  catch (e) { toast(e.message, true); }
  finally { if (btn) btn.disabled = false; }
}

// ---------- step logic ----------
function stepDone(id) {
  const s = ui.status, sel = s?.selection;
  switch (id) {
    case "dataset": return !!sel;
    case "tokenize": return !!sel?.tokenized;
    case "batches": return ui.batchSampled || !!s?.model;
    case "model": return !!s?.model;
    case "attention": return ui.inspected;
    case "train": return !!sel?.checkpoint;
    case "generate": return ui.generated;
    case "models": return (s?.library_count || 0) > 0;
    default: return false;
  }
}

function stepBlocker(id) {
  const s = ui.status, sel = s?.selection;
  if (id === "dataset" || id === "models" || id === "lab" || id === "journey") return null;
  if (id === "generate" && s.active) return null;
  if (!sel) return ["Add a dataset first, or pick a saved model in the Models step.", "dataset"];
  if (id === "tokenize") return null;
  if (!sel.tokenized) return ["Tokenize your training data first.", "tokenize"];
  if (id === "batches" || id === "model") return null;
  if (id === "attention") return s.model || sel.checkpoint ? null : ["Build a model first.", "model"];
  if (id === "train") return s.model || sel.checkpoint ? null : ["Build a model first.", "model"];
  if (id === "generate") return s.active ? null : ["Train a model first, or pick a saved model in the Models step.", "train"];
  return null;
}

function go(id) {
  ui.step = id;
  try { localStorage.setItem("lmstudio.step", id); } catch (e) { /* ignore */ }
  history.replaceState(null, "", `#${id}`);
  $$(".panel").forEach((p) => p.classList.toggle("active", p.dataset.step === id));
  window.scrollTo({ top: 0, behavior: "smooth" });
  if (id === "models") loadModels().catch((e) => toast(e.message, true));
  if (id === "lab" && typeof labOpen === "function") labOpen();
  if (id === "journey" && typeof journeyOpen === "function") journeyOpen();
  render();
  if (id === "batches") renderBatchExample();
  if (id === "tokenize") renderTokenExamples();
}

// ---------- rendering ----------
function render() {
  const s = ui.status;
  if (!s) return;
  if (ui.evalRunning && !(s.job.kind === "evaluate" && s.job.status === "running")) { ui.evalRunning = false; loadModels(); }
  if (s.job.kind === "evaluate" && s.job.status === "running") ui.evalRunning = true;
  if (!ui.filled) fillInputs(s);
  renderTopbar(s);
  renderSidebar(s);
  renderBlocker();
  renderJobs(s.job);
  renderDatasets(s);
  renderTokenize(s);
  renderEstimate();
  renderTrainLearn();
  renderTrain(s);
  renderGenerateCard(s);
  renderGallery();
  if (typeof labRender === "function") labRender(s);
}

function renderTopbar(s) {
  const names = { mps: "Apple GPU · Metal", cuda: "NVIDIA GPU · CUDA", cpu: "CPU only" };
  $("#device-pill").innerHTML = `<span class="dot" style="background:${s.device === "cpu" ? "var(--warn)" : "var(--good)"}"></span>${esc(s.chip)} · ${names[s.device] || s.device}`;
  $("#device-pill").title = `PyTorch ${s.torch} · ${s.dtype}${s.memory ? ` · ${bytes(s.memory)} memory` : ""}`;
  const j = s.job, pill = $("#job-pill");
  if (j.status === "running") {
    const pct = j.progress != null ? ` ${Math.round(j.progress * 100)}%` : "";
    pill.className = "pill running";
    pill.innerHTML = `<span class="dot"></span>${esc(cap(j.kind))}${pct}`;
  } else pill.className = "pill hidden";
}

function renderSidebar(s) {
  const busyStep = s.job.status === "running" ? s.job.kind : null;
  $("#steps").innerHTML = STEPS.map((st, i) => {
    const busy = busyStep === st.id || (st.id === "lab" && busyStep === "lab");
    const cls = ["step", st.explore && "explore", st.id === ui.step && "active", stepDone(st.id) && "done", busy && "busy"].filter(Boolean).join(" ");
    const head = st.explore && !STEPS[i - 1]?.explore ? '<li class="steps-group">Explore</li>' : "";
    return `${head}<li><button class="${cls}" data-go="${st.id}"><span class="num">${st.icon || (stepDone(st.id) && !busy ? "✓" : i + 1)}</span><span><div class="t">${st.title}</div><div class="s">${st.sub}</div></span></button></li>`;
  }).join("");
}

function renderBlocker() {
  const b = stepBlocker(ui.step), el = $("#blocked");
  if (!b) { el.classList.add("hidden"); return; }
  el.innerHTML = `<span>${esc(b[0])}</span><button class="btn small" data-go="${b[1]}">Go to ${STEPS.find((x) => x.id === b[1]).title}</button>`;
  el.classList.remove("hidden");
}

function renderJobs(j) {
  $$("[data-job]").forEach((el) => {
    if (j.kind !== el.dataset.job || j.status === "idle") { el.innerHTML = ""; return; }
    const running = j.status === "running";
    const pct = j.progress == null ? null : Math.round(j.progress * 100);
    el.innerHTML = `<div class="jobline ${j.status}">${running ? '<span class="spinner"></span>' : ""}<span>${esc(j.message || j.error || "")}</span></div>` +
      (running ? `<div class="bar ${pct == null ? "indeterminate" : ""}"><div style="width:${pct ?? 0}%"></div></div>` : "");
  });
  const running = j.status === "running";
  ["#ds-load", "#tok-run", "#b-run", "#m-run", "#t-start"].forEach((sel) => { $(sel).disabled = running; });
  $("#g-run").disabled = running && j.kind === "train";
  $("#t-stop").classList.toggle("hidden", !(running && j.kind === "train"));
}

function fillInputs(s) {
  $("#ds-name").value = s.default_dataset;
  $("#ds-config").value = s.default_config || "";
  $("#b-block").value = s.batch.block_size;
  $("#b-batch").value = s.batch.batch_size;
  $("#b-seed").value = s.batch.seed;
  setModelFields(s.model?.config || MODEL_PRESETS.small);
  setTrainPreset(s.train?.config ? matchPreset(s.train.config) : "quick", s.train?.config);
  ui.filled = true;
}

function stat(k, v, d = "", small = false) {
  return `<div class="stat"><div class="k">${k}</div><div class="v ${small ? "sm" : ""}">${v}</div>${d ? `<div class="d">${d}</div>` : ""}</div>`;
}

// ---------- 1. datasets ----------
const dsBody = () => ({ name: $("#ds-name").value.trim(), config_name: $("#ds-config").value.trim() || null, text_column: $("#ds-column").value.trim() || "text" });

function renderDatasets(s) {
  const list = s.datasets, sel = s.selection;
  $("#ds-selection").classList.toggle("hidden", !list.length);
  $("#ds-browse").classList.toggle("hidden", !list.length);
  if (!list.length) return;
  $("#ds-count").textContent = list.length > 1 ? `· ${list.length} datasets mixed` : "";
  $("#ds-mix-wrap").classList.toggle("hidden", list.length < 2);
  $$("#ds-mix button").forEach((b) => b.classList.toggle("on", b.dataset.mix === sel.mix_mode));
  const key = JSON.stringify([list.map((d) => [d.key, d.tokenized]), sel.shares, s.job.status]);
  if (key !== ui.dsKey) {
    ui.dsKey = key;
    $("#ds-list").innerHTML = list.map((d, i) => {
      const share = sel.shares ? sel.shares[i] : null;
      return `<div class="ds-row">
        <div class="ds-main"><div class="ds-name">${esc(dsLabel(d))}</div>
          <div class="ds-meta">${fmt(d.splits.train)} train / ${fmt(d.splits.validation)} validation documents · ${d.tokenized ? `${short(d.tokens.train)} tokens` : "not tokenized yet"} · <span class="mono">${esc(d.path)}</span></div></div>
        ${list.length > 1 ? `<div class="share">${share != null ? `<div class="share-bar"><div style="width:${Math.max(1, share * 100)}%"></div></div><span>${(share * 100).toFixed(share < 0.01 ? 2 : 0)}% of batches</span>` : '<span class="muted small">tokenize to see share</span>'}</div>` : ""}
        <button class="icon-btn" data-remove="${esc(d.key)}" title="Remove from training data" aria-label="Remove">×</button>
      </div>`;
    }).join("");
    $("#ds-mix-hint").innerHTML = list.length > 1
      ? `Each window in a batch comes from one dataset. <b>By size</b> draws in proportion to each dataset's tokens; <b>Equally</b> gives each the same share, so small datasets aren't drowned out. The trained model is saved in <code>${esc(sel.path)}</code>.`
      : `Add another dataset to train on a mix. The trained model is saved in <code>${esc(sel.path)}</code>.`;
    const opts = list.map((d) => `<option value="${esc(d.key)}">${esc(dsLabel(d))}</option>`).join("");
    const which = $("#ds-browse-which");
    if (which.innerHTML !== opts) which.innerHTML = opts;
    which.classList.toggle("hidden", list.length < 2);
  }
  const browseKey = list.find((d) => d.key === ui.sample.key) ? ui.sample.key : list[list.length - 1].key;
  if (ui.sample.loadedFor !== browseKey) {
    ui.sample = { key: browseKey, split: "train", index: 0, loadedFor: browseKey };
    $("#ds-browse-which").value = browseKey;
    loadSample(0);
  }
}

async function loadSample(index = ui.sample.index) {
  try {
    const r = await api(`/api/dataset/sample?key=${encodeURIComponent(ui.sample.key || "")}&split=${ui.sample.split}&index=${index}`);
    ui.sample.index = r.index;
    $("#ds-text").textContent = r.text;
    $("#ds-pos").textContent = `${fmt(r.index + 1)} / ${fmt(r.total)}`;
    $$("#ds-split button").forEach((b) => b.classList.toggle("on", b.dataset.split === r.split));
  } catch (e) { toast(e.message, true); }
}

async function checkDataset() {
  const el = $("#ds-check-result");
  el.className = "check";
  el.innerHTML = '<span class="spinner"></span> Checking the dataset…';
  const r = await api("/api/dataset/check", dsBody());
  if (!r.ok) {
    el.className = "check bad";
    el.innerHTML = `<div class="title">✕ Can't use this dataset</div><div class="row">${esc(r.error)}</div>` +
      (r.configs ? `<div class="configs">${r.configs.map((c) => `<button class="chip-btn" data-config="${esc(c)}">${esc(c)}</button>`).join("")}</div>` : "");
    return r;
  }
  const cols = r.columns
    ? Object.entries(r.columns).map(([k, v]) => `<span class="col ${k === r.text_column ? "chosen" : ""}" title="${esc(v)}">${esc(k)}</span>`).join("")
    : "<span>read after download</span>";
  const rows = r.rows ? Object.entries(r.rows).map(([k, v]) => `${esc(k)} ${fmt(v)}`).join(" · ") : "";
  el.className = "check ok";
  el.innerHTML = `<div class="title">✓ Text dataset${r.kind === "url" ? " (file link)" : ""}: ${r.selected ? "already in your training data" : "ready to add"}</div>` +
    `<div class="row"><span>Columns:</span>${cols}</div>` +
    `<div class="row">${r.download_size ? `<span>Download <b>${bytes(r.download_size)}</b></span>` : ""}${rows ? `<span>${rows}</span>` : ""}` +
    `<span>${r.downloaded ? "Already downloaded" : `Will be saved to <code>${esc(r.path)}</code>`}</span></div>`;
  return r;
}

let catalog = [], galleryTag = "All", galleryKey = null;
async function loadCatalog() {
  try { catalog = await api("/api/catalog"); galleryKey = null; renderGallery(); } catch (e) { /* server not ready */ }
}

function renderGallery() {
  const selectedKeys = new Set((ui.status?.datasets || []).map((d) => d.key));
  const key = JSON.stringify([galleryTag, [...selectedKeys], catalog.length]);
  if (key === galleryKey) return;
  galleryKey = key;
  const tags = ["All", ...new Set(catalog.flatMap((c) => c.tags))];
  $("#gallery-tags").innerHTML = tags.map((t) => `<button class="chip-btn ${t === galleryTag ? "on" : ""}" data-tag="${esc(t)}">${esc(t)}</button>`).join("");
  $("#gallery-grid").innerHTML = catalog.map((c, i) => {
    if (galleryTag !== "All" && !c.tags.includes(galleryTag)) return "";
    const id = c.dataset.startsWith("http") ? shortUrl(c.dataset) : c.dataset + (c.config ? ` · ${c.config}` : "");
    const isSel = selectedKeys.has(c.key);
    const status = isSel ? "In training data" : c.trained ? "Trained" : c.tokenized ? "Tokenized" : c.downloaded ? "Downloaded" : "";
    return `<div class="ds-card ${isSel ? "current" : ""}">
      <div class="top"><div><h4>${esc(c.title)}</h4><div class="id">${esc(id)}</div></div>${status ? `<span class="tag status">${status}</span>` : ""}</div>
      <div class="tags">${c.tags.map((t) => `<span class="tag">${esc(t)}</span>`).join("")}</div>
      <p>${esc(c.description)}</p>
      <div class="meta"><span><b>${esc(c.rows)}</b></span><span><b>${esc(c.size)}</b></span>${c.text_column !== "text" ? `<span>column <b>${esc(c.text_column)}</b></span>` : ""}</div>
      <div class="btns">${isSel ? `<button class="btn" data-remove="${esc(c.key)}">Remove</button>` : `<button class="btn primary" data-add="${i}">Add</button>`}<a class="btn" href="${esc(c.url)}" target="_blank" rel="noopener">View ↗</a></div>
    </div>`;
  }).join("");
}

async function addDataset(body) {
  await api("/api/dataset", body);
  await poll();
}

// ---------- 2. tokenize ----------
function tokenChip(t, colorKey, extra = "") {
  const shown = esc(t.text).replace(/ /g, '<span class="sp">·</span>').replace(/\n/g, '<span class="nl">↵</span>');
  return `<span class="tok c${colorKey % 6}" title="token id ${t.id}" ${extra}>${shown || "&nbsp;"}</span>`;
}

const TOKEN_EXAMPLES = [
  ["Once upon a time", "Common words are one token each. The space belongs to the word after it."],
  ["unbelievably", "A rarer word is split into familiar pieces."],
  ["Tom's dog didn't bark.", "Apostrophes and punctuation get tokens of their own."],
  ["In 2026 she was 7.", "Numbers are split into chunks, which is one reason small models are bad at arithmetic."],
  ["dog Dog DOG", "Capitalization changes the tokens completely: rarer spellings like “DOG” even split in two."],
  ["café 😀", "Accents and emoji are stored as bytes and can take several tokens. A � chip holds part of one character."],
];

async function renderTokenExamples() {
  if (ui.tokExamplesDone) return;
  try {
    const res = await api("/api/tokenize/many", { texts: TOKEN_EXAMPLES.map((e) => e[0]) });
    $("#tok-examples").innerHTML = res.map((r, i) => `<button class="example" data-example="${esc(r.text)}">
        <div class="ex-head"><code>${esc(r.text)}</code><span class="muted small">${r.text.length} chars → ${r.tokens.length} token${r.tokens.length > 1 ? "s" : ""}</span></div>
        <div class="chips tight">${r.tokens.map((t, j) => tokenChip(t, j)).join("")}</div>
        <div class="muted small">${esc(TOKEN_EXAMPLES[i][1])}</div></button>`).join("");
    ui.tokExamplesDone = true;
  } catch (e) { /* server not ready */ }
}

let previewTimer;
async function previewTokens() {
  const text = $("#tok-input").value;
  try {
    const r = await api("/api/tokenize/preview", { text });
    $("#tok-chips").innerHTML = r.tokens.map((t, i) => tokenChip(t, i)).join("");
    $("#tok-meta").textContent = `${fmt(r.chars)} characters → ${fmt(r.tokens.length)} tokens`;
  } catch (e) { /* server not ready yet */ }
}

function renderTokenize(s) {
  const list = s.datasets, st = $("#tok-status"), btn = $("#tok-run");
  if (!list.length) { st.innerHTML = '<p class="muted">Add a dataset in step 1 first.</p>'; btn.classList.add("hidden"); return; }
  st.innerHTML = `<div class="ds-list">${list.map((d) => `<div class="ds-row"><div class="ds-main"><div class="ds-name">${esc(dsLabel(d))}</div>
    <div class="ds-meta">${d.tokenized ? `<b>${fmt(d.tokens.train)}</b> training tokens and <b>${fmt(d.tokens.validation)}</b> validation tokens in <code>${esc(d.path)}/train.bin</code> and <code>validation.bin</code>` : "Not tokenized yet"}</div></div>
    <span class="tag ${d.tokenized ? "status" : ""}">${d.tokenized ? "Done" : "To do"}</span></div>`).join("")}</div>` +
    (s.selection.tokenized ? `<p class="hint">All done: ${fmt(s.selection.tokens.train)} training tokens in total. Tokenizing is skipped from now on.</p>`
      : '<p class="hint">Every document is converted to token IDs once and saved. Large datasets take a while.</p>');
  btn.classList.toggle("hidden", s.selection.tokenized);
}

// ---------- 3. batches ----------
async function renderBatchExample() {
  if (ui.batchExampleDone) return;
  try {
    const r = await api("/api/tokenize/preview", { text: "Once upon a time there was a little cat named Tom." });
    const t = r.tokens, n = 6;
    const x = t.slice(0, n), y = t.slice(1, n + 1);
    const row = (label, toks) => `<tr><th>${label}</th>${toks.map((tk) => `<td>${tokenChip(tk, tk.id)}</td>`).join("")}</tr>`;
    const pairs = x.map((_, i) => `<div class="pair"><span class="ctx">${x.slice(0, i + 1).map((tk) => tokenChip(tk, tk.id)).join("")}</span><span class="arrow">→</span>${tokenChip(y[i], y[i].id)}</div>`).join("");
    $("#b-example").innerHTML = `<div class="sentence">“Once upon a time there was a little cat named Tom.” → ${t.length} tokens</div>
      <div class="table-scroll"><table class="shift"><tbody>${row("Input x", x)}${row("Target y", y)}</tbody></table></div>
      <div class="pairs">${pairs}</div>`;
    ui.batchExampleDone = true;
  } catch (e) { /* server not ready */ }
}

function renderBatch(r) {
  const pos = r.input.map((_, i) => `<td>${i}</td>`).join("");
  const row = (label, toks) => `<tr><th>${label}</th>${toks.map((t) => `<td>${tokenChip(t, t.id)}</td>`).join("")}</tr>`;
  $("#b-table").innerHTML = `<tr class="pos"><th></th>${pos}</tr>` + row("Input", r.input) + row("Target", r.target);
  $("#b-shape").textContent = `${r.shape[0]} × ${r.shape[1]} = ${fmt(r.shape[0] * r.shape[1])} tokens`;
  const comp = Object.entries(r.composition);
  $("#b-composition").innerHTML = comp.length > 1
    ? `<div class="note">This batch mixes datasets: ${comp.map(([k, v]) => `<b>${v}</b> window${v > 1 ? "s" : ""} from ${esc(k.startsWith("http") ? shortUrl(k) : k)}`).join(", ")}. The window above is from ${esc(r.row0_source.startsWith("http") ? shortUrl(r.row0_source) : r.row0_source)}.</div>`
    : "";
  $("#b-result").classList.remove("hidden");
}

// ---------- 4. model ----------
const swigluHidden = (d) => 32 * Math.round((8 * d) / 3 / 32);
function estimateParams(L, d, T) {
  const rms = $("#m-norm")?.value === "rmsnorm";  // RMSNorm: a scale per feature, no shift
  const sw = $("#m-mlp")?.value === "swiglu", h = sw ? swigluHidden(d) : 4 * d;  // SwiGLU: three matrices, 8/3 wide
  const b = $("#m-bias")?.value === "0" ? 0 : 1;   // bias terms in linear layers and LayerNorms
  const ffn = sw ? 3 * d * h + b * (2 * h + d) : 8 * d * d + b * 5 * d;
  const norm = rms ? d : (1 + b) * d;
  const H = +$("#m-head")?.value || 1, hd = Math.round(d / H);
  const qkn = $("#m-qkn")?.value === "1" ? 2 * hd : 0;  // QK-norm scales
  const kvd = (+$("#m-kv")?.value || H) * hd;            // keys and values: n_kv_head · head_dim wide
  const attn = 2 * d * d + 2 * d * kvd + b * (2 * d + 2 * kvd);
  const E = +$("#m-exp")?.value || 1;                     // mixture of experts: E copies of the feed-forward + a router
  const blk = attn + E * ffn + (E > 1 ? d * E : 0) + 2 * norm + qkn, mtp = (+$("#m-mtp")?.value || 1) - 1;  // extra heads: one block each
  return VOCAB * d + ($("#m-pos")?.value === "rope" ? 0 : T * d) + (L + mtp) * blk + norm;
}
const modelFields = () => ({ n_layer: +$("#m-layer").value, n_head: +$("#m-head").value, n_embd: +$("#m-embd").value, dropout: +$("#m-drop").value, pos_encoding: $("#m-pos").value, norm: $("#m-norm").value, mlp: $("#m-mlp").value, bias: $("#m-bias").value !== "0", qk_norm: $("#m-qkn").value === "1", n_kv_head: +$("#m-kv").value || 0, block_repeat: +$("#m-rep").value || 1, mtp_heads: +$("#m-mtp").value || 1, n_experts: +$("#m-exp").value || 1, moe_top_k: Math.min(+$("#m-topk").value || 1, +$("#m-exp").value || 1) });
const POS_HINTS = {
  learned: "Each of the context window's positions has its own trained vector, added to the token embedding (block_size × n_embd parameters). The model only knows positions it was trained on, so a full KV cache has to be rebuilt when the window slides.",
  rope: "Rotary position embeddings: no position vectors. Each query and key is rotated by an angle that grows with its position, so attention scores depend on how far apart two tokens are. No parameters, used by Llama, Mistral and Qwen, and the KV cache can slide past the window without being rebuilt (a close approximation of recomputing the window).",
};
const NORM_HINTS = {
  layernorm: "Before attention and the feed-forward layer, each vector is normalized: subtract its mean, divide by its standard deviation, then a learned scale and shift (GPT-2).",
  rmsnorm: "Divide each vector by its root mean square, then a learned scale: no mean subtraction and no shift. Fewer operations and parameters; used by Llama, Mistral and Qwen.",
};
const MLP_HINTS = {
  gelu: "Each token's vector is widened to 4 × n_embd, passed through GELU, and narrowed back (GPT-2).",
  swiglu: "Two projections to about 8/3 × n_embd: one goes through SiLU and acts as a gate that scales the other, feature by feature, then one projection back. Three matrices instead of two, narrower, so about the same parameters. Used by Llama, Mistral and Qwen.",
};
const BIAS_HINTS = {
  1: "Every linear layer adds a learned bias vector to its output, and each LayerNorm learns a shift (GPT-2).",
  0: "No bias vectors anywhere: linear layers are pure matrix multiplications and LayerNorms only scale. A few thousand fewer parameters, and often slightly more stable training. Used by Llama, Mistral and Qwen.",
};
const QKN_HINTS = {
  0: "Queries and keys go into attention as they are. Their dot products can grow large during training, which makes attention very peaked and training less stable at high learning rates.",
  1: "Each head's query and key are RMS-normalized before attention (a learned scale per dimension), so attention scores stay in a steady range. Steadier training that tolerates a higher learning rate; used by OLMo 2, Gemma 3 and Qwen3.",
};
function renderKvOptions(want) {
  // key/value heads must divide the number of query heads
  const H = +$("#m-head").value || 1, sel = $("#m-kv"), cur = want ?? (+sel.value || H);
  const opts = Array.from({ length: H }, (_, i) => i + 1).filter((k) => H % k === 0).reverse();
  sel.innerHTML = opts.map((k) => `<option value="${k}">${k === H ? `${k} (one per head)` : k === 1 ? "1 (multi-query)" : `${k} (groups of ${H / k})`}</option>`).join("");
  sel.value = opts.includes(cur) ? cur : H;
  const k = +sel.value, hd = Math.round((+$("#m-embd").value || 0) / H);
  $("#m-kv-hint").textContent = k === H
    ? "Ordinary multi-head attention: every query head has its own key and value head, and the KV cache stores all of them."
    : `Grouped-query attention: ${H} query heads share ${k} key/value head${k > 1 ? "s" : ""} (${H / k} each). Keys and values are ${k * hd} numbers per token and layer instead of ${H * hd}, so the KV cache is ${H / k}× smaller and generation reads less memory. Used by Llama 2 70B, Llama 3, Mistral and Qwen.`;
}
const effLayers = (c) => (c?.n_layer || 0) * (c?.block_repeat || 1);  // layers the input passes through
function renderRepHint() {
  const r = +$("#m-rep").value || 1, L = +$("#m-layer").value || 0;
  $("#m-rep-hint").textContent = r === 1
    ? "Each block has its own weights and is applied once."
    : `Layer sharing (MobileLLM): each block runs ${r} times in a row with the same weights, so the model is ${L * r} layers deep with the parameters of ${L}. It costs ${r}× the compute per token (training and generation) and ${r}× the KV cache, but no extra memory for weights.`;
}
function renderMtpHint() {
  const n = +$("#m-mtp").value || 1;
  $("#m-mtp-hint").textContent = n === 1
    ? "Training teaches the model to predict the next token only."
    : `Multi-token prediction (Meta 2024, DeepSeek-V3): ${n - 1} extra head${n > 2 ? "s" : ""} (one transformer block each, sharing the output layer) also learn to predict ${n === 2 ? "the token 2 ahead" : `the tokens 2 to ${n} ahead`}. A richer training signal; the extra heads are only used in training (and could draft tokens for speculative decoding). Generating and the validation loss use the next-token head.`;
}
function renderExpHint() {
  const E = +$("#m-exp").value || 1, k = Math.min(+$("#m-topk").value || 1, E);
  $("#m-topk").disabled = E === 1;
  $("#m-exp-hint").textContent = E === 1
    ? "One feed-forward layer per block, used by every token."
    : `Mixture of experts (Switch Transformer, Mixtral, DeepSeek): ${E} feed-forward layers per block and a small router that sends each token to ${k} of them. About ${E}× the feed-forward parameters, but each token only runs ${k}, so compute per token grows ${k === 1 ? "barely" : `about ${k}× in the feed-forward part`}. A balancing loss keeps all experts busy.`;
}
const normName = (c) => c.norm === "rmsnorm" ? "RMSNorm" : "LayerNorm";

function setModelFields(c) {
  $("#m-layer").value = c.n_layer; $("#m-head").value = c.n_head; $("#m-embd").value = c.n_embd;
  if (c.dropout != null) $("#m-drop").value = c.dropout;
  else if (!$("#m-drop").value) $("#m-drop").value = 0;
  if (c.pos_encoding) $("#m-pos").value = c.pos_encoding;
  $("#m-pos-hint").textContent = POS_HINTS[$("#m-pos").value];
  if (c.norm) $("#m-norm").value = c.norm;
  $("#m-norm-hint").textContent = NORM_HINTS[$("#m-norm").value];
  if (c.mlp) $("#m-mlp").value = c.mlp;
  $("#m-mlp-hint").textContent = MLP_HINTS[$("#m-mlp").value];
  if (c.bias != null) $("#m-bias").value = c.bias ? "1" : "0";
  $("#m-bias-hint").textContent = BIAS_HINTS[$("#m-bias").value];
  renderKvOptions(c.n_kv_head || (c.n_kv_head === 0 ? c.n_head : undefined));
  if (c.qk_norm != null) $("#m-qkn").value = c.qk_norm ? "1" : "0";
  if (c.block_repeat != null) $("#m-rep").value = String(c.block_repeat || 1);
  renderRepHint();
  if (c.mtp_heads != null) $("#m-mtp").value = String(c.mtp_heads || 1);
  renderMtpHint();
  if (c.n_experts != null) $("#m-exp").value = String(c.n_experts || 1);
  if (c.moe_top_k != null) $("#m-topk").value = String(c.moe_top_k || 1);
  renderExpHint();
  $("#m-qkn-hint").textContent = QKN_HINTS[$("#m-qkn").value];
  renderEstimate();
}

function renderModelPresets() {
  const T = ui.status?.batch.block_size ?? 128, f = modelFields();
  $("#m-preset").innerHTML = Object.entries(MODEL_PRESETS).map(([k, p]) => {
    const on = p.n_layer === f.n_layer && p.n_head === f.n_head && p.n_embd === f.n_embd;
    return `<button data-mpreset="${k}" class="${on ? "on" : ""}">${p.label} <span class="muted">${short(estimateParams(p.n_layer, p.n_embd, T))}</span></button>`;
  }).join("");
}

function renderEstimate() {
  if (!ui.status) return;
  renderModelPresets();
  const T = ui.status.batch.block_size, f = modelFields(), ok = f.n_embd && f.n_head && f.n_embd % f.n_head === 0;
  const p = estimateParams(f.n_layer, f.n_embd, T), base = estimateParams(6, 384, T), mem = p * BYTES_PER_PARAM;
  const tooBig = ui.status.memory && mem > 0.6 * ui.status.memory;
  $("#m-estimate").innerHTML = `<div class="est-main"><span class="muted small">This model will have</span><b>${fmt(p)}</b><span class="muted small">parameters (${short(p)}, ${(p / base).toFixed(p / base < 10 ? 1 : 0)}× the Small model)</span></div>
    <div class="est-bar"><div style="width:${Math.min(100, (p / estimateParams(12, 768, T)) * 100)}%"></div></div>
    <div class="muted small">Training needs about <b>${bytes(mem)}</b> for weights, gradients and optimizer state, plus activations.${tooBig ? ` <span class="bad-text">That's too much for this Mac's ${bytes(ui.status.memory)}.</span>` : ""}</div>`;
  $("#m-hint").innerHTML = `Context window: <b>${T}</b> tokens (set in step 3). ` + (ok
    ? `Each of the ${f.n_head} heads works on ${f.n_embd / f.n_head} dimensions. <b>More heads don't add parameters</b>: they split the embedding into smaller pieces. More layers or a bigger embedding do.`
    : `<span class="bad-text">Embedding size must be divisible by the number of heads.</span>`);
}

function renderParams(b) {
  const head = `<thead><tr><th>Part</th><th>Formula</th><th>Calculation</th><th class="num">Parameters</th></tr></thead>`;
  const row = (r) => `<tr class="${r.count === 0 ? "zero" : ""}"><td>${esc(r.name)}${r.note ? `<span class="pnote">${esc(r.note)}</span>` : ""}</td>` +
    `<td class="f">${esc(r.formula)}</td><td class="e">${esc(r.expr)}</td><td class="num">${fmt(r.count)}</td></tr>`;
  const table = (rows, label, total) => `<div class="ptable-scroll"><table class="ptable">${head}<tbody>${rows.map(row).join("")}` +
    `<tr class="total"><td>${label}</td><td></td><td></td><td class="num">${fmt(total)}</td></tr></tbody></table></div>`;
  return table(b.parts, "Total", b.total) +
    `<details class="pblock"><summary>Inside one transformer block: ${fmt(b.block)} parameters</summary>` +
    table(b.per_block, "One block", b.block) +
    `<p class="formula-line">A linear layer with <code>a</code> inputs and <code>b</code> outputs has <code>a × b</code> weights plus <code>b</code> biases. ` +
    `A LayerNorm learns a scale and a shift for each of the <code>n_embd</code> features. The feed-forward network widens to <code>4 × n_embd</code> and back.</p></details>`;
}

function renderArchitecture(c, b) {
  const part = (name) => b.parts.find((p) => p.name === name)?.count ?? 0;
  const shownHeads = Math.min(c.n_head, 12);
  const heads = Array.from({ length: shownHeads }, (_, i) => `<span class="head">H${i + 1}</span>`).join("") + (c.n_head > shownHeads ? `<span class="head more">+${c.n_head - shownHeads}</span>` : "");
  const hs = c.n_embd / c.n_head;
  const arrow = '<div class="arr">↓</div>';
  const drop = `<div class="mini drop" title="Dropout ${c.dropout}: zeroes ${Math.round(c.dropout * 100)}% of values at random while training; off when generating">Dropout</div>`;
  const dec = '<span class="kind decoder">Decoder only</span>';
  const box = (cls, title, sub, params, badge = "") => `<div class="abox ${cls}"><div><div class="at">${title} ${badge}</div>${sub ? `<div class="as">${sub}</div>` : ""}</div>${params != null ? `<span class="ap">${short(params)}</span>` : ""}</div>`;
  $("#m-arch").innerHTML = `<div class="arch2">
    ${box("io", "Input: token IDs", `up to <b>${c.block_size}</b> tokens · shape (batch, ${c.block_size})`, null)}${arrow}
    <div class="sub-flow">${c.pos_encoding === "rope"
      ? box("emb", "Token embedding (positions: RoPE)", `each token becomes a vector of ${c.n_embd} numbers · a ${fmt(c.vocab_size)} × ${c.n_embd} table; positions are added later, by rotating queries and keys inside attention`, part("Token embeddings"))
      : box("emb", "Token embedding + position embedding", `each token becomes a vector of ${c.n_embd} numbers · ${fmt(c.vocab_size)} × ${c.n_embd} and ${c.block_size} × ${c.n_embd} tables`, part("Token embeddings") + part("Position embeddings"))}<div class="arr r">→</div>${drop}</div>${arrow}
    <div class="stackbox"><div class="stack-badge">× ${c.n_layer} layer${c.n_layer > 1 ? "s" : ""}${(c.block_repeat || 1) > 1 ? `, each applied ${c.block_repeat}× (${effLayers(c)} deep)` : ""}</div>
      <div class="stack-title">Transformer block <span class="ap">${short(b.block)} each</span></div>
      <div class="sub-flow">
        <div class="mini ln">${normName(c)}</div><div class="arr r">→</div>
        <div class="abox attn"><div><div class="at">Masked multi-head self-attention ${dec}</div><div class="as">${c.n_head} heads × ${hs} dims${c.n_kv_head && c.n_kv_head !== c.n_head ? ` · ${c.n_kv_head} shared key/value head${c.n_kv_head > 1 ? "s" : ""} (grouped-query)` : ""}${c.qk_norm ? " · queries and keys normalized (QK-norm)" : ""} · causal mask: each token sees only earlier tokens · attention dropout</div><div class="heads">${heads}</div></div></div>
        <div class="arr r">→</div>${drop}<div class="arr r">→</div><div class="resid">+ residual</div>
      </div>
      <div class="sub-flow">
        <div class="mini ln">${normName(c)}</div><div class="arr r">→</div>
        <div class="abox ffn"><div>${(c.n_experts || 1) > 1
          ? `<div class="at">Mixture of experts: ${c.n_experts} feed-forward networks</div><div class="as">a router sends each token to ${c.moe_top_k || 1} of the ${c.n_experts} experts (each ${c.mlp === "swiglu" ? `SwiGLU, ${c.n_embd} → ${swigluHidden(c.n_embd)} → ${c.n_embd}` : `GELU, ${c.n_embd} → ${4 * c.n_embd} → ${c.n_embd}`}); their outputs are mixed by the router's weights</div>`
          : c.mlp === "swiglu"
          ? `<div class="at">Feed-forward network (SwiGLU)</div><div class="as">two Linears ${c.n_embd} → ${swigluHidden(c.n_embd)} · SiLU(gate) × up · Linear ${swigluHidden(c.n_embd)} → ${c.n_embd}</div>`
          : `<div class="at">Feed-forward network (MLP)</div><div class="as">Linear ${c.n_embd} → ${4 * c.n_embd} · GELU · Linear ${4 * c.n_embd} → ${c.n_embd}</div>`}</div></div>
        <div class="arr r">→</div>${drop}<div class="arr r">→</div><div class="resid">+ residual</div>
      </div>
    </div>${arrow}
    ${box("", `Final ${normName(c)}`, "", part(`Final ${normName(c)}`))}${arrow}
    ${box("emb", "Output layer (shares the token embeddings)", `a score (logit) for each of the ${fmt(c.vocab_size)} tokens at every position · shape (batch, ${c.block_size}, ${fmt(c.vocab_size)})`, 0, dec)}${arrow}
    ${box("io", "Softmax → next-token probabilities", "sample one token, append it, repeat (autoregressive decoding)", null, dec)}
  </div>
  <div class="arch-legend"><span><span class="mini ln">${normName(c)}</span> normalizes each vector</span><span><span class="mini drop">Dropout</span> training only (p = ${c.dropout})</span><span>${dec} parts an encoder doesn't have</span><span class="muted">Not in this model: an encoder stack and cross-attention (encoder–decoder models)</span></div>`;
}

function renderModel(r) {
  const c = r.config;
  $("#m-params-total").textContent = `${fmt(r.breakdown.total)} parameters`;
  $("#m-params").innerHTML = renderParams(r.breakdown);
  renderArchitecture(c, r.breakdown);
  $("#m-stats").innerHTML = [
    stat("Parameters", short(r.params), `${fmt(r.params)} exactly`),
    stat("Input tokens", fmt(c.block_size), "context window: the most it reads at once"),
    stat("Output", fmt(c.vocab_size), "scores per position → 1 next token"),
    stat("Loss before training", fmtLoss(r.initial_loss), `random guessing: ${fmtLoss(r.random_loss)}`),
    stat("Training memory", bytes(r.train_memory), "weights, gradients, optimizer"),
  ].join("");
  $("#m-sample").textContent = r.sample;
  $("#m-result").classList.remove("hidden");
}

// ---------- 5. inside the transformer (toy walkthrough + real attention) ----------
function prng(seed) {
  return () => { seed = (seed + 0x6D2B79F5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
const r1 = (x) => Math.round(x * 10) / 10;
const r2 = (x) => Math.round(x * 100) / 100;
const randMatrix = (seed, rows, cols, scale) => { const rnd = prng(seed); return Array.from({ length: rows }, () => Array.from({ length: cols }, () => r1((rnd() * 2 - 1) * scale))); };
const W_Q = [[1.0, -0.4, 0.2, 0.6], [0.2, 0.8, -0.6, 0.4], [-0.4, 0.6, 1.0, -0.2], [0.6, 0.2, 0.4, 0.8]];
const W_K = [[0.8, 0.2, -0.4, 0.4], [-0.2, 1.0, 0.4, 0.2], [0.6, -0.4, 0.8, 0.2], [0.2, 0.4, 0.2, 1.0]];
const W_V = [[0.2, 0.5, 0.1, -0.3], [0.4, -0.1, 0.3, 0.2], [-0.3, 0.2, 0.4, 0.1], [0.1, 0.3, -0.2, 0.4]];
const W_O = randMatrix(55, 4, 4, 0.6);
const W_1 = randMatrix(99, 4, 16, 1.0);   // feed-forward: expand 4 -> 16 (4 x n_embd, like the real model)
const W_2 = randMatrix(77, 16, 4, 0.4);   // feed-forward: project 16 -> 4
const matmul = (A, B) => A.map((row) => B[0].map((_, j) => r2(row.reduce((s, a, k) => s + a * B[k][j], 0))));
const transpose = (A) => A[0].map((_, j) => A.map((r) => r[j]));
const addM = (A, B) => A.map((row, i) => row.map((v, j) => r2(v + B[i][j])));
const gelu = (x) => 0.5 * x * (1 + Math.tanh(0.7978845608 * (x + 0.044715 * x ** 3)));
const toyEmbedding = (id) => { const rnd = prng(id + 1); return [0, 1, 2, 3].map(() => r1(rnd() * 4 - 2)); };
function layerNorm(row) {
  const mu = row.reduce((a, b) => a + b, 0) / row.length;
  const v = row.reduce((a, b) => a + (b - mu) ** 2, 0) / row.length;
  return { mu: r2(mu), sd: r2(Math.sqrt(v + 1e-5)), out: row.map((x) => r2((x - mu) / Math.sqrt(v + 1e-5))) };
}
const lnRows = (M) => M.map((r) => layerNorm(r).out);

function buildWalk(tokens) {
  const T = tokens.length;
  const E = tokens.map((t) => toyEmbedding(t.id));
  const P = tokens.map((_, t) => [r1(Math.sin(t) * 0.5), r1(Math.cos(t) * 0.5), r1(Math.sin(t / 2) * 0.5), r1(Math.cos(t / 2) * 0.5)]);
  const X = E.map((e, i) => e.map((v, j) => r1(v + P[i][j])));
  const XN = lnRows(X);                                    // LayerNorm 1 (scale 1, shift 0 in this toy)
  const Q = matmul(XN, W_Q), K = matmul(XN, W_K), V = matmul(XN, W_V);
  const S = matmul(Q, transpose(K)).map((row) => row.map((v) => r2(v / 2)));  // ÷ √d_k = √4 = 2
  const A = S.map((row, i) => {
    const vis = row.slice(0, i + 1), m = Math.max(...vis);
    const ex = vis.map((v) => Math.exp(v - m)), sum = ex.reduce((a, b) => a + b, 0);
    return row.map((_, j) => (j <= i ? r2(ex[j] / sum) : 0));
  });
  const C = A.map((row) => [0, 1, 2, 3].map((k) => r2(row.reduce((s, w, j) => s + w * V[j][k], 0))));
  const O = matmul(C, W_O);                                // attention output projection
  const H = addM(X, O);                                    // residual
  const HN = lnRows(H);                                    // LayerNorm 2
  const U = matmul(HN, W_1);                               // expand to 16
  const G = U.map((r) => r.map((x) => r2(gelu(x))));       // GELU
  const F = matmul(G, W_2);                                // project back to 4
  const X2 = addM(H, F);                                   // residual
  const XF = lnRows(X2);                                   // final LayerNorm
  return { tokens, T, E, P, X, XN, Q, K, V, S, A, C, O, H, HN, U, G, F, X2, XF, stage: 0, focus: T - 1 };
}

function matrix(M, { rows, cols, title, heat = false, focus = -1, mask = false, digits = 2, clickable = false } = {}) {
  const flat = M.flat().filter((v, i) => !mask || (i % M[0].length) <= Math.floor(i / M[0].length));
  const lo = Math.min(...flat), hi = Math.max(...flat);
  const head = cols ? `<tr><th></th>${cols.map((c) => `<th>${c}</th>`).join("")}</tr>` : "";
  const body = M.map((row, i) => `<tr class="${i === focus ? "focus" : ""}" ${clickable ? `data-focus="${i}"` : ""}><th>${rows ? rows[i] : ""}</th>${row.map((v, j) => {
    if (mask && j > i) return '<td class="masked">−∞</td>';
    const h = heat ? (hi === lo ? 0.5 : (v - lo) / (hi - lo)) : 0;
    return `<td style="--h:${h.toFixed(2)}" class="${heat ? "heat" : ""}">${Number(v).toFixed(digits)}</td>`;
  }).join("")}</tr>`).join("");
  return `<div class="mat">${title ? `<div class="mat-title">${title}</div>` : ""}<table>${head}${body}</table></div>`;
}

const vec = (v, digits = 2) => `[${v.map((x) => Number(x).toFixed(digits)).join(", ")}]`;
const tokLabel = (t) => esc(t.text).replace(/ /g, "·");
const dropNote = (where) => `<div class="dropnote"><span class="tag drop">Dropout · training only</span> ${where}</div>`;

const STAGES = [
  {
    title: "Tokens become IDs", kind: "both",
    body: (w) => `<p>The prompt is tokenized. From here on the model only sees these numbers.</p>
      <div class="chips">${w.tokens.map((t, i) => `<span class="idchip">${tokenChip(t, i)}<b>${t.id}</b></span>`).join("")}</div>
      <p class="muted small">Next we follow these ${w.T} tokens through one transformer block, and then out the other end as a prediction.</p>`,
  },
  {
    title: "Embeddings: each token becomes a vector", kind: "both",
    body: (w, c) => `<p>Each ID picks a row from the <b>token embedding table</b>, and each position adds a row from the <b>position embedding table</b>. The sum <code>x = e + p</code> says both <i>what</i> the token is and <i>where</i> it is.</p>
      <div class="mats">${matrix(w.E, { rows: w.tokens.map(tokLabel), title: "token embedding e", digits: 1 })}<div class="op">+</div>${matrix(w.P, { rows: w.tokens.map((_, i) => `pos ${i}`), title: "position p", digits: 1 })}<div class="op">=</div>${matrix(w.X, { rows: w.tokens.map(tokLabel), title: "x", digits: 1, heat: true })}</div>
      ${dropNote(`is applied to <code>x</code> here: while training, ${Math.round((c.dropout ?? 0.1) * 100)}% of the numbers are zeroed at random so the model can't rely on any single one.`)}
      <p class="muted small">Toy numbers with 4 dimensions. Your model uses ${c.n_embd} dimensions from a ${fmt(c.vocab_size)} × ${c.n_embd} table that it learns during training.</p>
      ${c.pos_encoding === "rope" ? `<p class="note">This model uses <b>RoPE</b> instead: there is no position table and nothing is added here. Position enters inside attention, where each query and key is <b>rotated</b> by an angle proportional to its position (a different speed for each pair of dimensions). The dot product of a rotated query and key then depends only on how far apart the two tokens are.</p>` : ""}`,
  },
  {
    title: "LayerNorm before attention", kind: "both",
    body: (w, c) => {
      const i = w.focus, ln = layerNorm(w.X[i]);
      return `<p>Before attention, each token's vector is <b>normalized</b>: subtract its mean, divide by its standard deviation. This keeps the numbers in a steady range however deep the network gets. (The real LayerNorm then multiplies by a learned scale and adds a learned shift; here those are 1 and 0.)</p>
        <div class="calc">“${tokLabel(w.tokens[i])}”: x = ${vec(w.X[i], 1)} · mean = ${ln.mu} · std = ${ln.sd}<br>(x − mean) ÷ std = <b>${vec(ln.out)}</b></div>
        <div class="mats">${matrix(w.XN, { rows: w.tokens.map(tokLabel), title: "LayerNorm(x)", heat: true, focus: i, clickable: true })}</div>
        ${c.norm === "rmsnorm" ? `<p class="note">This model uses <b>RMSNorm</b> instead: it skips the mean and divides by the root mean square, √(mean of x²), then multiplies by a learned scale. No shift. It's cheaper and works as well, which is why Llama, Mistral and Qwen use it.</p>` : ""}
        <p class="muted small">This model uses <i>pre-norm</i>: LayerNorm comes before attention and before the feed-forward network (as in GPT-2), rather than after them as in the original Transformer.</p>`;
    },
  },
  {
    title: "Queries, keys and values", kind: "both",
    body: (w, c) => `<p>Three learned matrices turn every normalized vector into three new vectors:</p>
      <ul class="qkv"><li><b>Query (Q)</b>: what this token is looking for.</li><li><b>Key (K)</b>: what this token offers to others.</li><li><b>Value (V)</b>: the information it passes on if someone attends to it.</li></ul>
      <div class="mats">${matrix(w.XN, { rows: w.tokens.map(tokLabel), title: "LayerNorm(x)" })}<div class="op">×</div>${matrix(W_Q, { title: "W<sub>Q</sub>", digits: 1 })}<div class="op">=</div>${matrix(w.Q, { rows: w.tokens.map(tokLabel), title: "Q", heat: true })}</div>
      <div class="mats">${matrix(w.K, { rows: w.tokens.map(tokLabel), title: "K = LayerNorm(x) × W<sub>K</sub>", heat: true })}${matrix(w.V, { rows: w.tokens.map(tokLabel), title: "V = LayerNorm(x) × W<sub>V</sub>", heat: true })}</div>
      <p class="muted small">In your model each head has its own W<sub>Q</sub>, W<sub>K</sub>, W<sub>V</sub> of size ${c.n_embd} × ${c.n_embd / c.n_head}.</p>`,
  },
  {
    title: "Attention scores: how well each query matches each key", kind: "both",
    body: (w) => {
      const i = w.focus, j = 0, q = w.Q[i], k = w.K[j];
      const terms = q.map((v, n) => `${v.toFixed(2)}×${k[n].toFixed(2)}`).join(" + ");
      return `<p>Each query is compared with every key using a dot product, divided by <code>√d<sub>k</sub> = √4 = 2</code> to keep the numbers in a good range. Bigger score = better match. Click a row to follow a different token.</p>
        <div class="mats">${matrix(w.S, { rows: w.tokens.map(tokLabel), cols: w.tokens.map(tokLabel), title: "scores = Q · Kᵀ / 2", heat: true, focus: i, clickable: true })}</div>
        <div class="calc">score(“${tokLabel(w.tokens[i])}” → “${tokLabel(w.tokens[j])}”) = (${terms}) / 2 = <b>${w.S[i][j].toFixed(2)}</b></div>`;
    },
  },
  {
    title: "Causal mask: no peeking at the future", kind: "decoder",
    body: (w) => `<p>When predicting what comes next, a token must not see tokens after it, or it could cheat by reading the answer. Scores for future positions are set to <code>−∞</code>, which becomes 0 after softmax.</p>
      <div class="mats">${matrix(w.S, { rows: w.tokens.map(tokLabel), cols: w.tokens.map(tokLabel), title: "masked scores", heat: true, mask: true, focus: w.focus, clickable: true })}</div>
      <p class="muted small"><b>This is what makes a decoder.</b> An encoder (like BERT) skips the mask, so every token sees the whole sentence in both directions; that suits understanding text, but not writing it one token at a time.</p>`,
  },
  {
    title: "Softmax: scores become attention weights", kind: "both",
    body: (w, c) => {
      const i = w.focus;
      const bars = w.A[i].slice(0, i + 1).map((v, j) => `<div class="wbar"><span class="wl">${tokLabel(w.tokens[j])}</span><div class="wtrack"><div style="width:${v * 100}%"></div></div><span class="wv">${(v * 100).toFixed(0)}%</span></div>`).join("");
      return `<p>Softmax turns each row into percentages that add up to 100%: how much attention a token pays to each earlier token (and itself).</p>
        <div class="mats">${matrix(w.A, { rows: w.tokens.map(tokLabel), cols: w.tokens.map(tokLabel), title: "attention weights", heat: true, mask: true, focus: i, clickable: true })}
        <div class="wbars"><div class="mat-title">“${tokLabel(w.tokens[i])}” attends to</div>${bars}</div></div>
        ${dropNote(`is applied to these attention weights while training (${Math.round((c.dropout ?? 0.1) * 100)}%), so a token can't always lean on the same neighbour.`)}`;
    },
  },
  {
    title: "The context vector: a weighted mix of values", kind: "both",
    body: (w) => {
      const i = w.focus;
      const terms = w.A[i].slice(0, i + 1).map((a, j) => `<div class="term"><span class="wv">${a.toFixed(2)}</span>×<span class="vec">${vec(w.V[j])}</span><span class="muted small">V of “${tokLabel(w.tokens[j])}”</span></div>`).join("");
      return `<p>Each token's <b>context vector</b> is the sum of all value vectors, weighted by its attention. This is how information moves between tokens: “${tokLabel(w.tokens[i])}” now carries a blend of what came before it.</p>
        <div class="terms">${terms}<div class="term total">= <span class="vec"><b>${vec(w.C[i])}</b></span><span class="muted small">context vector of “${tokLabel(w.tokens[i])}”</span></div></div>
        <div class="mats">${matrix(w.C, { rows: w.tokens.map(tokLabel), title: "all context vectors = weights × V", heat: true, focus: i, clickable: true })}</div>`;
    },
  },
  {
    title: "Output projection, dropout and the residual connection", kind: "both",
    body: (w, c) => {
      const i = w.focus, o = w.O[i];
      const p = c.dropout ?? 0.1, keep = 1 / (1 - p), dropIdx = (i + 1) % 4;
      const dropped = o.map((v, n) => (n === dropIdx ? 0 : r2(v * keep)));
      return `<p>The context vector is mixed by one more learned matrix, <b>W<sub>O</sub></b> (the output projection). Then it is <b>added back</b> to the token's original vector <code>x</code>: the <b>residual connection</b>. The block only has to learn a <i>change</i> to <code>x</code>, which makes deep networks much easier to train.</p>
        <div class="calc">“${tokLabel(w.tokens[i])}”: context ${vec(w.C[i])} × W<sub>O</sub> = <b>${vec(o)}</b></div>
        <div class="dropdemo"><div class="mat-title">Dropout while training (one possible random draw, p = ${p})</div>
          <div class="dd-row"><span class="muted small">before</span><span class="vec">${vec(o)}</span></div>
          <div class="dd-row"><span class="muted small">after</span><span class="vec">[${dropped.map((v, n) => (n === dropIdx ? '<s class="zeroed">0.00</s>' : v.toFixed(2))).join(", ")}]</span></div>
          <p class="muted small">Each number has a ${Math.round(p * 100)}% chance of being zeroed; the survivors are multiplied by 1 / (1 − ${p}) = ${keep.toFixed(2)} so the total stays about the same. When generating text, dropout is switched off and the vector passes through unchanged.</p></div>
        <div class="calc">residual: x ${vec(w.X[i], 1)} + ${vec(o)} = <b>${vec(w.H[i])}</b></div>
        <div class="mats">${matrix(w.H, { rows: w.tokens.map(tokLabel), title: "after attention: h = x + context × W<sub>O</sub>", heat: true, focus: i, clickable: true })}</div>`;
    },
  },
  {
    title: "LayerNorm, then the feed-forward network", kind: "both",
    body: (w, c) => {
      const i = w.focus, ln = layerNorm(w.H[i]);
      const cells = (v, cls = "") => `<div class="vcells ${cls}">${v.map((x) => `<span style="--h:${Math.min(1, Math.abs(x) / 2).toFixed(2)}" class="${x < 0 ? "neg" : ""}">${x.toFixed(2)}</span>`).join("")}</div>`;
      return `<p>Now each token is processed <b>on its own</b> by the feed-forward network (FFN). First another LayerNorm, then it is <b>widened</b> to 4× its size, passed through <b>GELU</b>, and <b>narrowed</b> back. Attention moved information <i>between</i> tokens; the FFN works on what each token now holds.</p>
        <div class="ffn-flow">
          <div><div class="mat-title">1. LayerNorm(h) · mean ${ln.mu}, std ${ln.sd}</div>${cells(ln.out)}</div>
          <div><div class="mat-title">2. × W<sub>1</sub> (4 → 16): widen</div>${cells(w.U[i])}</div>
          <div><div class="mat-title">3. GELU ${geluSvg()} keeps positives, squashes negatives toward 0</div>${cells(w.G[i])}</div>
          <div><div class="mat-title">4. × W<sub>2</sub> (16 → 4): narrow back</div>${cells(w.F[i])}</div>
        </div>
        ${dropNote("is applied to the FFN output while training.")}
        <div class="calc">residual: h ${vec(w.H[i])} + FFN ${vec(w.F[i])} = <b>${vec(w.X2[i])}</b> → the block's output for “${tokLabel(w.tokens[i])}”</div>
        <p class="muted small">Toy sizes 4 → 16 → 4. Your model goes ${c.n_embd} → ${c.mlp === "swiglu" ? swigluHidden(c.n_embd) : 4 * c.n_embd} → ${c.n_embd}; the FFN holds about two thirds of each block's parameters.</p>
        ${c.mlp === "swiglu" ? `<p class="note">This model uses <b>SwiGLU</b> instead: the token is projected twice, to ${swigluHidden(c.n_embd)} numbers each. One copy goes through SiLU (a smooth GELU-like curve) and <b>multiplies</b> the other, feature by feature, like a learned gate, before projecting back. Three matrices instead of two, each narrower, so the parameter count stays about the same.</p>` : ""}`;
    },
  },
  {
    title: "Many heads, many layers", kind: "both",
    body: (w, c) => `<p>That was <b>one head</b> in <b>one block</b>. Your model runs <b>${c.n_head} heads</b> side by side in every block, each with its own Q, K and V, so different heads can track different things (the previous word, the subject of the sentence, punctuation…). Their context vectors are concatenated before W<sub>O</sub>.</p>
      <div class="pipeline">
        <div class="pp">${c.n_head} heads × ${c.n_embd / c.n_head} dims</div><span>→</span>
        <div class="pp">concatenate → ${c.n_embd}</div><span>→</span>
        <div class="pp">W<sub>O</sub> ${c.n_embd} × ${c.n_embd}</div><span>→</span>
        <div class="pp acc">+ residual</div><span>→</span>
        <div class="pp">LayerNorm → FFN</div><span>→</span><div class="pp acc">+ residual</div>
      </div>
      <p>The output of one block is the input of the next. The whole block repeats <b>${c.n_layer} times</b>, each time refining every token's vector with more context.</p>`,
  },
  {
    title: "Final LayerNorm, output layer, softmax: picking the next token", kind: "decoder",
    body: (w) => {
      const r = ui.real;
      const last = w.T - 1, final = w.XF[last];
      const cands = (r ? r.next.slice(0, 6) : w.tokens.slice(0, 6).map((t) => ({ id: t.id, text: t.text })));
      const logits = cands.map((cd) => r2(toyEmbedding(cd.id).reduce((s, e, k) => s + e * final[k], 0)));
      const m = Math.max(...logits), ex = logits.map((l) => Math.exp(l - m)), sum = ex.reduce((a, b) => a + b, 0);
      const probs = ex.map((e) => e / sum);
      const best = probs.indexOf(Math.max(...probs));
      const rows = cands.map((cd, n) => `<tr class="${n === best ? "focus" : ""}"><td class="wl">${esc(cd.text).replace(/ /g, "·")}</td><td class="mono">${vec(toyEmbedding(cd.id), 1)}</td><td class="num mono">${logits[n].toFixed(2)}</td><td><div class="wtrack"><div style="width:${probs[n] * 100}%"></div></div></td><td class="num mono">${(probs[n] * 100).toFixed(0)}%</td></tr>`).join("");
      return `<p>Only the <b>last</b> token's vector (“${tokLabel(w.tokens[last])}”) is used to predict what comes next. It goes through a final LayerNorm, and is then compared with <b>every row of the token embedding table</b>: the same table used at the start (<i>weight tying</i>). Each dot product is a <b>logit</b>: a score for that word being next. Softmax turns the logits into probabilities.</p>
        <div class="calc">final vector = LayerNorm(${vec(w.X2[last])}) = <b>${vec(final)}</b></div>
        <div class="ptable-scroll"><table class="ptable logit-table"><thead><tr><th>Candidate</th><th>Its embedding</th><th class="num">Logit = final · embedding</th><th>Probability</th><th></th></tr></thead><tbody>${rows}</tbody></table></div>
        <p class="muted small">Toy numbers, ${cands.length} candidates${r ? " (your model's real top candidates, with toy embeddings, so the toy may pick a different winner than the real model: see the next stage)" : ""}.${r && r.tokens.length > w.T ? ` The walkthrough follows only the first ${w.T} tokens; the candidates come from the full prompt.` : ""} The real model computes ${fmt(VOCAB)} logits at once. Then it picks: <b>greedy</b> takes the highest; <b>sampling</b> draws at random by probability, after <b>temperature</b> (divides the logits: below 1 sharpens, above 1 flattens) and <b>top-k</b> (keeps only the k best).</p>`;
    },
  },
  {
    title: "Your model's real prediction", kind: "decoder",
    body: (w, c) => {
      const r = ui.real;
      if (!r) return `<p>Build or train a model to see real predictions here.</p>`;
      const last = r.tokens.map((t) => t.text).join("");
      const bars = r.next.map((n) => `<div class="wbar pick"><span class="wl">${esc(n.text).replace(/ /g, "·").replace(/\n/g, "↵")}</span><div class="wtrack"><div style="width:${(n.prob / r.next[0].prob) * 100}%"></div></div><span class="wv">${(n.prob * 100).toFixed(1)}%</span><button class="btn small ghost" data-append="${esc(n.text)}">Add</button></div>`).join("");
      return `<p>The same steps, for real: ${c.n_layer} blocks of ${c.n_head} heads with ${c.n_embd}-dimensional vectors, then ${fmt(c.vocab_size)} logits. These are your ${r.source === "trained" ? "trained" : "untrained"} model's top 10 for <code>${esc(last)}</code>:</p>
        <div class="wbars wide">${bars}</div>
        <p class="muted small">Click <b>Add</b> to append a token and run the walkthrough again: repeating this loop, one token at a time, is how text is generated. This is <b>autoregressive</b> decoding, the defining job of a decoder.</p>`;
    },
  },
];

function geluSvg() {
  const pts = []; for (let x = -3; x <= 3; x += 0.25) { const g = gelu(x); pts.push(`${(x + 3) * 8},${24 - (g + 0.5) * 6}`); }
  return `<svg viewBox="0 0 48 28" width="40" height="24" class="gelu-svg"><polyline points="${pts.join(" ")}" fill="none" stroke="currentColor" stroke-width="2"/></svg>`;
}

const KIND_BADGE = {
  both: '<span class="kind both" title="Encoders do this too">Encoder &amp; decoder</span>',
  decoder: '<span class="kind decoder" title="Only decoder (GPT-style) models do this">Decoder only</span>',
};

function renderWalk() {
  const w = ui.walk;
  if (!w) return;
  const c = ui.realConfig || ui.status?.model?.config || ui.status?.active?.model || { n_embd: 384, n_head: 6, n_layer: 6, vocab_size: VOCAB };
  $("#a-dots").innerHTML = STAGES.map((st, i) => `<button class="sdot ${i === w.stage ? "on" : i < w.stage ? "past" : ""} ${st.kind}" data-stage="${i}" title="${esc(st.title)}${st.kind === "decoder" ? " (decoder only)" : ""}">${i + 1}</button>`).join("");
  $("#a-stage").innerHTML = `<div class="stage-head"><h3 class="stage-title">${w.stage + 1}. ${STAGES[w.stage].title}</h3>${KIND_BADGE[STAGES[w.stage].kind]}</div>${STAGES[w.stage].body(w, c)}`;
  $("#a-prev").disabled = w.stage === 0;
  $("#a-next").disabled = w.stage === STAGES.length - 1;
}

async function runWalk() {
  const prompt = $("#a-prompt").value;
  const prev = await api("/api/tokenize/preview", { text: prompt });
  if (!prev.tokens.length) throw new Error("Type a prompt first.");
  const keepStage = ui.walk?.stage ?? 0;
  ui.walk = buildWalk(prev.tokens.slice(0, 6));
  ui.walk.stage = keepStage;
  ui.real = null;
  try {
    ui.real = await api("/api/inspect", { prompt, layer: +($("#a-layer").value || 0) });
    ui.realConfig = { ...(ui.status?.active?.model || ui.status?.model?.config || {}), n_head: ui.real.n_head, n_layer: ui.real.n_layer };
  } catch (e) { ui.real = null; }
  ui.inspected = true;
  $("#a-walk").classList.remove("hidden");
  renderWalk();
  renderReal();
  render();
}

function renderReal() {
  const r = ui.real, card = $("#a-real");
  card.classList.toggle("hidden", !r);
  if (!r) return;
  const layerOpts = Array.from({ length: r.n_layer }, (_, i) => `<option value="${i}">${i + 1} of ${r.n_layer}</option>`).join("");
  if ($("#a-layer").innerHTML !== layerOpts) $("#a-layer").innerHTML = layerOpts;
  $("#a-layer").value = r.layer;
  const heads = ["avg", ...Array.from({ length: r.n_head }, (_, i) => i)];
  if (ui.realHead !== "avg" && ui.realHead >= r.n_head) ui.realHead = "avg";
  $("#a-heads").innerHTML = heads.map((h) => `<button data-head="${h}" class="${String(h) === String(ui.realHead) ? "on" : ""}">${h === "avg" ? "Average" : `H${h + 1}`}</button>`).join("");
  const T = r.tokens.length;
  const M = ui.realHead === "avg"
    ? Array.from({ length: T }, (_, i) => Array.from({ length: T }, (_, j) => r.attention.reduce((s, h) => s + h[i][j], 0) / r.n_head))
    : r.attention[ui.realHead];
  $("#a-real-note").innerHTML = `Attention weights from your <b>${r.source}</b> model for the prompt, in layer ${r.layer + 1}${ui.realHead === "avg" ? ", averaged over all heads" : `, head ${ui.realHead + 1}`}. Each row is a token; the colors show which earlier tokens it attends to. ${r.source === "untrained" ? "An untrained model spreads attention almost evenly." : "Different heads learn different patterns: try a few."}`;
  $("#a-heatmap").innerHTML = `<div class="table-scroll">${matrix(M, { rows: r.tokens.map(tokLabel), cols: r.tokens.map(tokLabel), heat: true, mask: true })}</div>`;
}

// ---------- 6. train ----------
function setTrainPreset(name, values) {
  const v = values || TRAIN_PRESETS[name];
  if (v) TRAIN_FIELDS.forEach((f) => { $(`#t-${f}`).value = v[f]; });
  highlightPreset(name);
}
function highlightPreset(name) {
  $$("#t-preset button").forEach((b) => b.classList.toggle("on", b.dataset.preset === name));
  renderTrainLearn(true);
}
function matchPreset(cfg) {
  return Object.keys(TRAIN_PRESETS).find((p) => TRAIN_FIELDS.every((f) => Number(TRAIN_PRESETS[p][f]) === Number(cfg[f]))) || "custom";
}
function trainValues() {
  const v = {};
  TRAIN_FIELDS.forEach((f) => { v[f] = Number($(`#t-${f}`).value); });
  return v;
}

function lrAt(step, v) {
  if (step < v.warmup_steps) return v.learning_rate * (1 / 3 + (2 / 3) * (step / v.warmup_steps));  // PyTorch LinearLR starts at 1/3
  const t = (step - v.warmup_steps) / Math.max(1, v.max_iters - v.warmup_steps);
  return v.min_lr + (v.learning_rate - v.min_lr) * (1 + Math.cos(Math.PI * t)) / 2;
}

function renderTrainLearn(force = false) {
  const s = ui.status;
  if (!s) return;
  if (!$("#t-loss-examples").innerHTML) {
    const ex = [[1 / VOCAB, "1 in 50,257 (random guess)"], [0.01, "1%"], [0.1, "10%"], [0.5, "50%"], [0.9, "90%"]];
    $("#t-loss-examples").innerHTML = ex.map(([p, label]) => `<tr><td>${label}</td><td class="num">${(-Math.log(p)).toFixed(2)}</td><td class="num">${1 / p < 10 ? (1 / p).toFixed(1) : fmt(Math.round(1 / p))}</td></tr>`).join("");
  }
  const v = trainValues();
  const key = JSON.stringify([v, s.model?.params, s.selection?.tokens, s.batch]);
  if (!force && key === ui.trainLearnKey) return;
  ui.trainLearnKey = key;
  if (!v.max_iters || v.max_iters < 1) return;
  // learning-rate chart
  const W = 360, H = 120, L = 6, R = 6, T = 8, B = 18, n = 60;
  const pts = Array.from({ length: n + 1 }, (_, i) => { const st = (v.max_iters * i) / n; return [st, lrAt(st, v)]; });
  const sx = (x) => L + (x / v.max_iters) * (W - L - R), sy = (y) => T + (1 - y / v.learning_rate) * (H - T - B);
  $("#t-lr-chart").innerHTML = `<svg viewBox="0 0 ${W} ${H}"><path d="${pts.map((p, i) => `${i ? "L" : "M"}${sx(p[0]).toFixed(1)},${sy(p[1]).toFixed(1)}`).join("")}" fill="none" stroke="var(--accent)" stroke-width="2"/>
    <line x1="${sx(v.warmup_steps)}" x2="${sx(v.warmup_steps)}" y1="${T}" y2="${H - B}" stroke="var(--border)" stroke-dasharray="3 3"/>
    <text x="${L}" y="${H - 4}">0</text><text x="${sx(v.warmup_steps) + 5}" y="${H - B - 6}">warmup ends</text><text x="${W - R}" y="${H - 4}" text-anchor="end">${fmt(v.max_iters)} steps</text></svg>`;
  // the plan for this run, with real numbers
  const params = s.model?.params ?? s.selection?.checkpoint?.params?.total ?? estimateParams(6, 384, s.batch.block_size);
  const updates = Math.ceil(v.max_iters / v.gradient_accumulation_steps);
  const tokens = v.max_iters * s.batch.batch_size * s.batch.block_size;
  const pct = s.selection?.tokens ? (tokens / s.selection.tokens.train) * 100 : null;
  $("#t-plan").innerHTML = `<b>With these settings:</b> ${fmt(v.max_iters)} steps ÷ ${v.gradient_accumulation_steps} = <b>${fmt(updates)} optimizer updates</b>. Each one changes all ${fmt(params)} parameters, so <b>${short(params * updates)} parameter updates</b> in total. ` +
    `The model reads ${fmt(v.max_iters)} × ${s.batch.batch_size} × ${s.batch.block_size} = <b>${short(tokens)} tokens</b>${pct != null ? ` (${pctText(pct)} of your training data${pct > 100 ? ", so it sees some text more than once" : ""})` : ""}.`;
}

function duration(sec) {
  sec = Math.max(0, Math.round(sec));
  const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
  return h ? `${h}h ${m}m` : m ? `${m}m ${s}s` : `${s}s`;
}

function renderResume(s) {
  const r = s.selection?.resume, card = $("#t-resume");
  card.classList.toggle("hidden", !r);
  if (!r) return;
  const finished = r.step >= r.max_iters, m = r.model;
  $("#t-resume-info").innerHTML = `Saved run: <b>step ${fmt(r.step)}</b> of ${fmt(r.max_iters)}${finished ? " (finished)" : " (stopped early)"}` +
    `${m ? ` · ${m.n_layer} layers, ${m.n_embd}-dim` : ""} · best validation loss <b>${fmtLoss(r.best_val)}</b> · learning rate ${r.lr != null ? r.lr.toExponential(1) : "–"}`;
  const to = $("#t-resume-to");
  if (document.activeElement !== to && (!to.value || ui.resumeKey !== `${r.step}/${r.max_iters}`)) {
    ui.resumeKey = `${r.step}/${r.max_iters}`;
    to.value = finished ? r.max_iters + Math.max(1000, Math.round(r.max_iters / 2)) : r.max_iters;
  }
  $("#t-resume-run").disabled = training_running(s) || s.job.status === "running";
}

function renderTrain(s) {
  renderResume(s);
  const tr = s.train, runs = s.selection?.runs || [];
  $("#t-runs").classList.toggle("hidden", !runs.length);
  $("#t-runs-body").innerHTML = runs.map((r) => `<tr title="${esc((r.datasets || []).join(" + "))}${r.mix ? ` (mixed ${r.mix === "equal" ? "equally" : "by size"})` : ""}">
    <td>${esc(r.trained_at)}${r.stopped_early ? ' <span class="tag">stopped</span>' : ""}</td><td>${r.model ? `${r.model.n_layer}L · ${r.model.n_head}H · ${r.model.n_embd}` : ""} <span class="muted">(${short(r.params)})</span></td>
    <td class="num">${fmt(r.steps)}</td><td class="num">${fmt(r.updates)}</td><td class="num">${short(r.param_updates)}</td><td class="num">${short(r.tokens)}</td>
    <td class="num">${fmtLoss(r.best_val)}</td><td class="num">${duration(r.seconds)}</td></tr>`).join("");
  $("#t-live").classList.toggle("hidden", !tr);
  if (!tr) return;
  const running = training_running(s);
  const eta = running && tr.it_per_sec ? (tr.max_iters - tr.step) / tr.it_per_sec : null;
  const last = tr.evals[tr.evals.length - 1];
  $("#t-stats").innerHTML = [
    stat("Step", `${fmt(tr.step)}`, `of ${fmt(tr.max_iters)}`),
    stat("Speed", tr.it_per_sec ? `${tr.it_per_sec.toFixed(1)}` : "–", "steps / second"),
    stat(running ? "Time left" : "Status", running ? (eta != null ? duration(eta) : "–") : cap(s.job.kind === "train" ? s.job.status : "done"), running ? "estimated" : ""),
    stat("Validation loss", fmtLoss(last?.val), tr.best_val != null ? `best ${fmtLoss(tr.best_val)} · perplexity ${fmt(Math.round(Math.exp(tr.best_val)))}` : "after first eval"),
    stat("Learning rate", tr.lr != null ? Number(tr.lr).toExponential(1) : "–"),
  ].join("");
  const pct = s.selection?.tokens ? (tr.tokens / s.selection.tokens.train) * 100 : null;
  $("#t-updates").innerHTML = [
    stat("Optimizer updates", fmt(tr.updates), `of ${fmt(tr.total_updates)} · one every ${tr.config.gradient_accumulation_steps} steps`),
    stat("Parameters updated", short(tr.params * tr.updates), `${short(tr.params)} parameters × ${fmt(tr.updates)} updates`),
    stat("Tokens processed", short(tr.tokens), pct != null ? `${pctText(pct)} of the training data` : ""),
    stat("Training data", esc(tr.datasets.map((n) => (n.startsWith("http") ? shortUrl(n) : n.split("/").pop())).join(" + ")), tr.datasets.length > 1 ? `mixed ${s.selection?.mix_mode === "equal" ? "equally" : "by size"}` : "", true),
  ].join("");
  drawChart(tr);
}

const training_running = (s) => s.job.kind === "train" && s.job.status === "running";

function drawChart(tr) {
  const W = 760, H = 300, L = 46, R = 14, T = 12, B = 30;
  const pts = tr.points, ev = tr.evals;
  if (!pts.length && !ev.length) { $("#t-chart").innerHTML = '<p class="muted small">Waiting for the first steps…</p>'; return; }
  let ema = null;
  const smooth = pts.map(([x, y]) => { ema = ema == null ? y : ema * 0.9 + y * 0.1; return [x, ema]; });
  const ys = [...smooth.map((p) => p[1]), ...ev.flatMap((e) => [e.train, e.val])];
  let y0 = Math.min(...ys), y1 = Math.max(...ys);
  if (y1 - y0 < 0.2) { y0 -= 0.1; y1 += 0.1; }
  const pad = (y1 - y0) * 0.08; y0 -= pad; y1 += pad;
  const xMax = Math.max(tr.max_iters, 1);
  const sx = (x) => L + (x / xMax) * (W - L - R);
  const sy = (y) => T + (1 - (y - y0) / (y1 - y0)) * (H - T - B);
  const path = (arr) => arr.map((p, i) => `${i ? "L" : "M"}${sx(p[0]).toFixed(1)},${sy(p[1]).toFixed(1)}`).join("");
  let grid = "";
  for (let i = 0; i <= 4; i++) {
    const y = y0 + ((y1 - y0) * i) / 4;
    grid += `<line x1="${L}" x2="${W - R}" y1="${sy(y)}" y2="${sy(y)}"/><text x="${L - 8}" y="${sy(y) + 4}" text-anchor="end">${y.toFixed(1)}</text>`;
  }
  for (let i = 0; i <= 4; i++) {
    const x = Math.round((xMax * i) / 4);
    grid += `<text x="${sx(x)}" y="${H - 8}" text-anchor="${i === 0 ? "start" : i === 4 ? "end" : "middle"}">${fmt(x)}</text>`;
  }
  const dots = (key, color) => ev.map((e) => `<circle cx="${sx(e.step)}" cy="${sy(e[key])}" r="3.5" fill="${color}"><title>step ${fmt(e.step)}: ${key} ${e[key]}</title></circle>`).join("");
  $("#t-chart").innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Loss over training steps">
    <g class="grid">${grid}</g>
    <path d="${path(smooth)}" fill="none" stroke="var(--series-step)" stroke-width="1.6"/>
    <path d="${path(ev.map((e) => [e.step, e.train]))}" fill="none" stroke="var(--series-train)" stroke-width="2.2"/>
    <path d="${path(ev.map((e) => [e.step, e.val]))}" fill="none" stroke="var(--series-val)" stroke-width="2.2"/>
    ${dots("train", "var(--series-train)")}${dots("val", "var(--series-val)")}
  </svg>`;
}

// ---------- 7. generate ----------
function renderGenerateCard(s) {
  const c = s.active, el = $("#g-model");
  $("#g-params-wrap").classList.toggle("hidden", !c?.params);
  if (!c) { el.className = "note warn"; el.innerHTML = 'No trained model for this training data yet. Train one in the Train step, or <a href="#" data-go="models">pick a saved model</a>.'; return; }
  const key = JSON.stringify([c.id || s.selection?.path, c.model, c.trained_at]);
  if (c.params && ui.genParamsKey !== key) {
    ui.genParamsKey = key;
    $("#g-params-total").textContent = `${short(c.params.total)} (${fmt(c.params.total)})`;
    $("#g-params").innerHTML = renderParams(c.params);
    $$(".ctx-size").forEach((e) => { e.textContent = c.model.block_size; });
    $$(".dev-name").forEach((e) => { e.textContent = { mps: "Apple GPU (Metal)", cuda: "Nvidia GPU", cpu: "CPU" }[s.device] || s.device; });
  }
  el.className = "note";
  if (ui.kvKey !== key) { ui.kvKey = key; loadKvTokens().then(renderKv); renderKvMem(); fillCompare(); refreshDist(); }
  const m = c.model, names = (c.datasets || s.datasets.map((d) => d.name)).map((n) => (n.startsWith("http") ? shortUrl(n) : n));
  el.innerHTML = (c.source === "library" ? `Using the saved model <b>${esc(c.name)}</b> (<a href="#" data-go="models">change</a>), trained on ${esc(names.join(" + "))}: ` : `Using the model trained on <b>${esc(names.join(" + "))}</b> (<a href="#" data-go="models">pick another</a>): `) +
    `${m.n_layer} layers, ${m.n_head} heads, ${m.n_embd}-dim, ` +
    `${fmt(c.steps ?? 0)} steps${c.updates ? ` (${fmt(c.updates)} updates)` : ""}${c.stopped_early ? ", stopped early" : ""}, best validation loss <b>${fmtLoss(c.best_val)}</b>` +
    (c.trained_at ? ` · ${esc(c.trained_at)}` : "");
}

// ---------- sampling settings ----------
const SAMPLING_PRESETS = {
  greedy: { label: "Greedy (temperature 0)", temperature: 0, top_k: 0, top_p: 1, min_p: 0, repetition_penalty: 1, frequency_penalty: 0, presence_penalty: 0 },
  greedydry: { label: "Greedy + DRY 0.8: deterministic, without loops", temperature: 0, top_k: 0, top_p: 1, min_p: 0, repetition_penalty: 1, frequency_penalty: 0, presence_penalty: 0, dry_multiplier: 0.8 },
  balanced: { label: "Balanced: T 0.8, top-p 0.9, repetition 1.15", temperature: 0.8, top_k: 0, top_p: 0.9, min_p: 0, repetition_penalty: 1.15, frequency_penalty: 0, presence_penalty: 0 },
  creative: { label: "Creative: T 1.1, min-p 0.05", temperature: 1.1, top_k: 0, top_p: 1, min_p: 0.05, repetition_penalty: 1.05, frequency_penalty: 0, presence_penalty: 0 },
  reset: { label: "Old default: T 0.8, top-k 50, no penalty", temperature: 0.8, top_k: 50, top_p: 1, min_p: 0, repetition_penalty: 1, frequency_penalty: 0, presence_penalty: 0 },
  topk: { label: "Top-k 10, T 1.0", temperature: 1.0, top_k: 10, top_p: 1, min_p: 0, repetition_penalty: 1, frequency_penalty: 0, presence_penalty: 0 },
  penalty: { label: "Greedy + repetition penalty 1.3", temperature: 0, top_k: 0, top_p: 1, min_p: 0, repetition_penalty: 1.3, frequency_penalty: 0, presence_penalty: 0 },
  wild: { label: "Wild: temperature 1.6, no filters", temperature: 1.6, top_k: 0, top_p: 1, min_p: 0, repetition_penalty: 1, frequency_penalty: 0, presence_penalty: 0 },
};
const SAMPLING_FIELDS = [["temperature", "#g-temp"], ["top_k", "#g-topk"], ["top_p", "#g-topp"], ["min_p", "#g-minp"],
  ["repetition_penalty", "#g-rep"], ["frequency_penalty", "#g-freq"], ["presence_penalty", "#g-pres"],
  ["typical_p", "#g-typ"], ["dry_multiplier", "#g-dry"]];
const SAMPLING_OFF = { typical_p: 1, dry_multiplier: 0 };  // presets that don't mention these leave them off
const samplingBody = () => Object.fromEntries(SAMPLING_FIELDS.map(([k, sel]) => [k, +$(sel).value]));
function setSampling(p) {
  SAMPLING_FIELDS.forEach(([k, sel]) => { $(sel).value = p[k] ?? SAMPLING_OFF[k]; });
  syncSliders();
  refreshDist();
}

function syncSliders() {
  const t = +$("#g-temp").value;
  $("#g-temp-v").textContent = t === 0 ? "0 (greedy)" : t.toFixed(1);
  $("#g-topp-v").textContent = +$("#g-topp").value >= 1 ? "off" : Number($("#g-topp").value).toFixed(2);
  $("#g-minp-v").textContent = +$("#g-minp").value === 0 ? "off" : Number($("#g-minp").value).toFixed(2);
  $("#g-rep-v").textContent = +$("#g-rep").value === 1 ? "off" : Number($("#g-rep").value).toFixed(2);
  $("#g-freq-v").textContent = +$("#g-freq").value === 0 ? "off" : Number($("#g-freq").value).toFixed(1);
  $("#g-pres-v").textContent = +$("#g-pres").value === 0 ? "off" : Number($("#g-pres").value).toFixed(1);
  $("#g-typ-v").textContent = +$("#g-typ").value >= 1 ? "off" : Number($("#g-typ").value).toFixed(2);
  $("#g-dry-v").textContent = +$("#g-dry").value === 0 ? "off" : Number($("#g-dry").value).toFixed(1);
  $("#g-topk-v").textContent = $("#g-topk").value === "0" ? "off" : $("#g-topk").value;
  $("#g-len-v").textContent = `${$("#g-len").value} tokens`;
}

const confClass = (p) => (p >= 0.4 ? "hi" : p >= 0.1 ? "mid" : "lo");

function renderGenMetrics(m, live = false) {
  const mem = m.memory;
  $("#g-metrics").innerHTML = [
    stat("Prompt tokens", fmt(m.prompt_tokens), m.prompt_truncated ? "longer than the context window" : "input"),
    stat("Generated tokens", fmt(m.output_tokens), live ? "so far" : "output"),
    stat("Context used", `${fmt(m.context_used ?? Math.min(m.prompt_tokens + m.output_tokens, m.context_window))} / ${fmt(m.context_window)}`, "tokens the model could see at the end"),
    stat("Time to first token", m.ttft_ms != null ? `${Math.round(m.ttft_ms)} ms` : "–", m.prefill_tokens_per_sec ? `prefill at ${fmt(Math.round(m.prefill_tokens_per_sec))} tok/s` : "prefill + first pick"),
    stat("Total time", m.total_ms != null ? `${(m.total_ms / 1000).toFixed(2)} s` : "–", live ? "running…" : "for all tokens"),
    stat("Decode speed", m.tokens_per_sec != null ? `${m.tokens_per_sec} tok/s` : "–", m.ms_per_token != null ? `${m.ms_per_token} ms per token after the first` : ""),
    stat("KV cache", m.use_kv_cache === false ? "off" : m.kv_cache_bytes != null ? bytes(m.kv_cache_bytes) : "on",
      m.kv_bytes_per_token ? `${bytes(m.kv_bytes_per_token)} per token${m.recomputes ? ` · rebuilt ${fmt(m.recomputes)}×` : ""}` : ""),
    stat("Memory", m.weights_bytes ? bytes(m.weights_bytes) : "–", mem?.allocated != null ? `weights · ${bytes(mem.allocated)} in use on the GPU${mem.peak ? `, peak ${bytes(mem.peak)}` : ""}` : "weights"),
    stat("Average confidence", m.avg_prob != null ? `${Math.round(m.avg_prob * 100)}%` : "–", "model's probability for each chosen token"),
    ...(m.avg_kept != null ? [
      stat("Candidates per token", fmt(m.avg_kept), "tokens left after top-k / top-p / min-p, on average"),
      stat("Repetition", m.repeated_4grams != null ? `${Math.round(m.repeated_4grams * 100)}%` : "–", m.distinct_2 != null ? `of 4-token runs repeat earlier ones · ${Math.round(m.distinct_2 * 100)}% distinct pairs` : "of 4-token runs repeat earlier ones"),
    ] : []),
  ].join("");
}

async function runBenchmark() {
  $("#g-bench-busy").classList.remove("hidden");
  try {
    const sel = $("#g-bench-n").value;
    const r = await api("/api/generate/benchmark", { prompt: $("#g-prompt").value, n_tokens: +sel || 100, length: +sel ? null : sel });
    const a = r.no_cache, b = r.kv_cache;
    const row = (label, f, better) => `<tr><td>${label}</td><td class="num">${f(a)}</td><td class="num">${f(b)}</td><td class="num">${better ? better(a, b) : ""}</td></tr>`;
    const x = (u, v) => { if (!u || !v) return ""; const k = u / v; return Math.abs(k - 1) < 0.05 ? "about the same" : `${k.toFixed(2)}× ${k > 1 ? "faster" : "slower"}`; };
    $("#g-bench-out").innerHTML = `<div class="ptable-scroll"><table class="ptable bench"><thead><tr><th></th><th class="num">Without cache</th><th class="num">With KV cache</th><th class="num">Difference</th></tr></thead><tbody>
      ${row("Time to first token", (m) => `${m.ttft_ms} ms`)}
      ${row("Decode speed", (m) => `${m.tokens_per_sec} tok/s`, (u, v) => x(v.tokens_per_sec, u.tokens_per_sec))}
      ${row("Time per token", (m) => `${m.ms_per_token} ms`)}
      ${row(`Total for ${r.n_tokens} tokens`, (m) => `${(m.total_ms / 1000).toFixed(2)} s`, () => `<b>${x(a.total_ms, b.total_ms)}</b>`)}
      ${row("KV cache size", (m) => (m.kv_cache_bytes ? bytes(m.kv_cache_bytes) : "–"))}
      ${row("Cache rebuilds (window full)", (m) => fmt(m.recomputes))}
      ${row("GPU memory in use", (m) => (m.memory?.allocated != null ? bytes(m.memory.allocated) : "–"))}
      </tbody></table></div>
      <p class="hint">${r.same_output ? "✓ Both runs produced exactly the same tokens: the cache changes speed, not results." : "The runs differ slightly: tiny floating-point differences changed a sampled token."}
      The cache stores 2 (key and value) × ${r.kv_bytes_per_token / (2 * 4) / (effLayers(ui.status?.active?.model) || 6)} numbers × ${effLayers(ui.status?.active?.model) || 6} layers × 4 bytes = ${bytes(r.kv_bytes_per_token)} per token, up to ${bytes(r.kv_bytes_per_token * r.context_window)} for a full ${r.context_window}-token window.
      ${r.prompt_tokens + r.n_tokens > r.context_window ? (ui.status?.active?.model?.pos_encoding === "rope"
        ? `Past ${r.context_window} tokens the window slides. This RoPE model's cache slides with it (the oldest token is overwritten, nothing is rebuilt), so the advantage stays. Past the window the cached run is a close approximation of a full recompute, not an exact copy: deeper layers' cached keys still remember tokens that have left the window.`
        : `Past ${r.context_window} tokens the window slides, so this model rebuilds its cache every step and the advantage shrinks.`) : ""}</p>
      <p class="hint"><b>Why the gain is modest here:</b> with a small model and at most ${r.context_window} tokens, each step is dominated by the fixed cost of launching work on the GPU, not by the arithmetic the cache saves. On an M5 the cache measured about 1.1× faster for the Small model and 1.3× for Large (run it a few times: GPU timings vary). The benefit grows with model size and context length: for large language models reading thousands of tokens, the KV cache is what makes generation practical.</p>`;
    drawBenchCharts(r);
  } finally { $("#g-bench-busy").classList.add("hidden"); }
}

// A small line chart: series = [{values, color, label}], x values start at x0.
function lineChart(series, { x0 = 1, xLabel = "", yLabel = "", yMax = null, marks = [], W = 380, H = 190 } = {}) {
  const L = 44, R = 10, T = 10, Bm = 34, n = Math.max(...series.map((s) => s.values.length));
  const ymax = yMax ?? (Math.max(...series.flatMap((s) => s.values)) * 1.08 || 1);
  const X = (i) => L + (n > 1 ? (i / (n - 1)) * (W - L - R) : 0), Y = (v) => T + (1 - Math.min(v, ymax) / ymax) * (H - T - Bm);
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => f * ymax);
  const nice = (v) => (ymax >= 100 ? Math.round(v) : ymax >= 10 ? v.toFixed(0) : v.toFixed(1));
  const grid = ticks.map((v) => `<line x1="${L}" x2="${W - R}" y1="${Y(v)}" y2="${Y(v)}"/><text x="${L - 6}" y="${Y(v) + 4}" text-anchor="end">${nice(v)}</text>`).join("");
  const markEls = marks.map((m) => `<line x1="${X(m.i)}" x2="${X(m.i)}" y1="${T}" y2="${H - Bm}" stroke="var(--muted)" stroke-dasharray="3 3"/>` +
    (m.shade ? `<rect x="${X(m.i)}" y="${T}" width="${W - R - X(m.i)}" height="${H - T - Bm}" fill="var(--muted)" opacity=".07"/>` : "") +
    (X(m.i) > W * 0.6 ? `<text x="${X(m.i) - 4}" y="${T + 11}" text-anchor="end">${esc(m.label)}</text>` : `<text x="${X(m.i) + 4}" y="${T + 11}">${esc(m.label)}</text>`)).join("");
  const pts = (vs) => vs.map((v, i) => `${X(i).toFixed(1)},${Y(v).toFixed(1)}`).join(" ");
  const avg = (vs, k) => vs.map((_, i) => { const w = vs.slice(Math.max(0, i - k), i + k + 1); return w.reduce((a, b) => a + b, 0) / w.length; });
  const lines = series.map((s) => (s.smooth
    ? `<polyline fill="none" stroke="${s.color}" stroke-width="1" opacity=".22" points="${pts(s.values)}"/>` +
      `<polyline fill="none" stroke="${s.color}" stroke-width="2.2" stroke-linejoin="round" points="${pts(avg(s.values, s.smooth))}"><title>${esc(s.label)} (moving average)</title></polyline>`
    : `<polyline fill="none" stroke="${s.color}" stroke-width="1.8" stroke-linejoin="round" points="${pts(s.values)}"><title>${esc(s.label)}</title></polyline>`)).join("");
  return `<svg viewBox="0 0 ${W} ${H}"><g class="grid">${grid}</g>${markEls}${lines}
    <text x="${L}" y="${H - 18}">${x0}</text><text x="${W - R}" y="${H - 18}" text-anchor="end">${x0 + n - 1}</text>
    <text x="${(L + W - R) / 2}" y="${H - 4}" text-anchor="middle">${esc(xLabel)}</text>
    <text transform="translate(11 ${(T + H - Bm) / 2}) rotate(-90)" text-anchor="middle">${esc(yLabel)}</text></svg>`;
}

function drawBenchCharts(r) {
  const a = r.no_cache.token_ms, b = r.kv_cache.token_ms;
  if (!a?.length || !b?.length) return;
  $("#g-bench-charts").classList.remove("hidden");
  // per-token time, skipping the first token (it includes the prompt's prefill and would flatten the chart)
  const pa = a.slice(1), pb = b.slice(1);
  const sorted = [...pa, ...pb].sort((x, y) => x - y), p98 = sorted[Math.floor(sorted.length * 0.98)] || 1;
  const fullAt = r.context_window - r.prompt_tokens;   // tokens generated when the window is full
  const marks = fullAt > 1 && fullAt < pa.length + 1 ? [{ i: fullAt - 1, label: "window full →", shade: true }] : [];
  const k = Math.max(2, Math.round(pa.length / 40));
  $("#g-bench-per").innerHTML = lineChart([
    { values: pa, color: "var(--series-val)", label: "without cache", smooth: k },
    { values: pb, color: "var(--accent)", label: "with KV cache", smooth: k },
  ], { x0: 2, xLabel: "token number", yLabel: "ms per token", yMax: p98 * 1.1, marks });
  const cum = (xs) => { let s = 0; return xs.map((v) => (s += v) / 1000); };
  const ca = cum(a), cb = cum(b);
  $("#g-bench-cum").innerHTML = lineChart([
    { values: ca, color: "var(--series-val)", label: "without cache" },
    { values: cb, color: "var(--accent)", label: "with KV cache" },
  ], { x0: 1, xLabel: "tokens generated", yLabel: "seconds", marks: marks.map((m) => ({ ...m, i: m.i + 1, label: "window full", shade: false })) });
  const saved = ca[ca.length - 1] - cb[cb.length - 1];
  $("#g-bench-saved").textContent = saved > 0 ? `cache saved ${saved.toFixed(2)} s (${Math.round((saved / ca[ca.length - 1]) * 100)}%)` : "no time saved on this run";
}

// ---------- sampling explorer: how the next token is picked ----------
let distTimer = null, distSeq = 0;
function refreshDist() {
  if (!ui.status?.active) return;
  clearTimeout(distTimer);
  distTimer = setTimeout(async () => {
    const seq = ++distSeq;
    try {
      const r = await api("/api/generate/distribution", { prompt: $("#g-prompt").value, ...samplingBody(), n: 16 });
      if (seq === distSeq) renderDist(r);
    } catch (e) { $("#sd-chart").innerHTML = `<p class="muted small">${esc(e.message)}</p>`; }
  }, 250);
}

function renderDist(r) {
  const max = Math.max(...r.rows.map((x) => Math.max(x.raw, x.final)), 1e-9);
  const label = (t) => esc(JSON.stringify(t).slice(1, -1)).replace(/ /g, "␣");
  $("#sd-summary").textContent = r.greedy ? "greedy: the top token is always picked"
    : `${fmt(r.n_kept)} of ${fmt(r.vocab)} tokens kept · they held ${Math.round(r.kept_mass * 100)}% of the probability`;
  $("#sd-chart").innerHTML = r.rows.map((x) => `
    <div class="sd-row ${x.cut ? "cut" : ""}">
      <code class="sd-tok" title="token id ${x.id}">${label(x.text)}</code>
      <div class="sd-bars">
        <div class="sd-bar raw" style="width:${(x.raw / max) * 100}%"></div>
        <div class="sd-bar fin" style="width:${(x.final / max) * 100}%"></div>
      </div>
      <span class="sd-num">${(x.raw * 100).toFixed(1)}% → <b>${(x.final * 100).toFixed(1)}%</b></span>
      <span class="sd-tag">${x.cut ? `cut by ${x.cut}` : ""}${x.seen ? `${x.cut ? " · " : ""}in context ×${x.seen}` : ""}</span>
    </div>`).join("") +
    `<p class="hint">Uncertainty (entropy): ${r.entropy_raw.toFixed(2)} nats for the model's own distribution, ${r.entropy_final.toFixed(2)} after sampling settings. Higher means more tokens are in play. <i>in context ×n</i> marks tokens the penalties push down.</p>`;
}

// ---------- compare sampling settings ----------
function fillCompare() {
  const opts = [["current", "Your settings (sliders above)"], ...Object.entries(SAMPLING_PRESETS).map(([k, p]) => [k, p.label])];
  const defaults = ["greedy", "balanced", "wild"];
  $$(".cmp-sel").forEach((s, i) => { if (!s.options.length) { s.innerHTML = opts.map(([k, l]) => `<option value="${k}">${esc(l)}</option>`).join(""); s.value = defaults[i]; } });
}

async function runCompare() {
  $("#cmp-busy").classList.remove("hidden");
  try {
    const keys = $$(".cmp-sel").map((s) => s.value);
    const settings = keys.map((k) => (k === "current" ? samplingBody() : (({ label, ...p }) => p)(SAMPLING_PRESETS[k])));
    const r = await api("/api/generate/compare", { prompt: $("#g-prompt").value, max_new_tokens: +$("#cmp-n").value, settings });
    const names = $$(".cmp-sel").map((s) => s.options[s.selectedIndex].text);
    $("#cmp-out").innerHTML = r.results.map((x, i) => `
      <div class="cmp-col">
        <div class="cmp-name">${"ABC"[i]} · ${esc(names[i])}</div>
        <div class="doc cmp-text"><span class="prompt">${esc(r.prompt)}</span>${esc(x.text)}${x.stopped ? '<span class="story-end"> ⏹ end of text</span>' : ""}</div>
        <div class="cmp-stats">
          <span><b>${x.distinct_2 != null ? Math.round(x.distinct_2 * 100) + "%" : "–"}</b> distinct pairs</span>
          <span><b>${x.repeated_4grams != null ? Math.round(x.repeated_4grams * 100) + "%" : "–"}</b> repeated 4-grams</span>
          <span><b>${x.avg_prob != null ? Math.round(x.avg_prob * 100) + "%" : "–"}</b> avg confidence</span>
          <span><b>${x.avg_kept ?? "–"}</b> candidates per token</span>
          <span><b>${fmt(x.tokens)}</b> tokens · ${x.seconds} s</span>
        </div>
      </div>`).join("");
  } finally { $("#cmp-busy").classList.add("hidden"); }
}

// ---------- KV cache: interactive diagram ----------
const kv = { tokens: null, key: null, step: 1, timer: null };

async function loadKvTokens() {
  const prompt = ($("#g-prompt").value || "The water cycle is the process by which").slice(0, 300);
  if (kv.key === prompt && kv.tokens) return;
  kv.key = prompt;
  try { kv.tokens = (await api("/api/tokenize/preview", { text: prompt })).tokens.slice(0, 12); }
  catch (e) { kv.tokens = prompt.split(/(?=\s)/).slice(0, 12).map((t) => ({ text: t })); }
  $("#kv-step").max = kv.tokens.length;
  kv.step = Math.min(kv.step, kv.tokens.length);
  $("#kv-step").value = kv.step;
}

function kvModel() {
  const m = ui.status?.active?.model || ui.status?.model?.config || { n_layer: 6, n_embd: 384, block_size: 256 };
  const kvw = ((m.n_kv_head || m.n_head || 1) * m.n_embd) / (m.n_head || 1);  // stored key/value width
  const L = effLayers(m) || m.n_layer;  // with layer sharing, every pass through a block keeps its own cache
  return { n_layer: L, n_embd: kvw, block: m.block_size, perToken: 2 * L * kvw * 4 };
}

function renderKv() {
  if (!kv.tokens?.length) return;
  const m = kvModel(), on = $("#kv-on").checked, t = kv.step, toks = kv.tokens, n = toks.length;
  const layersShown = Math.min(m.n_layer, 4), cw = 46, ch = 13, L = 86, T = 64;
  const W = L + n * cw + 12, rowH = 2 * ch + 12, H = T + layersShown * rowH + (m.n_layer > layersShown ? 18 : 0) + 8;
  const cell = (i) => (i > t - 1 ? "kv-empty" : !on ? "kv-recompute" : i === t - 1 ? "kv-new" : "kv-reused");
  let g = "";
  // tokens along the top
  toks.forEach((tk, i) => {
    const x = L + i * cw;
    g += `<rect class="kv-tok ${i < t ? (i === t - 1 ? "now" : "seen") : ""}" x="${x + 2}" y="${T - 30}" width="${cw - 4}" height="20" rx="5"/>` +
      `<text class="kv-toktext" x="${x + cw / 2}" y="${T - 16}" text-anchor="middle">${esc(JSON.stringify(tk.text).slice(1, -1).slice(0, 7))}</text>`;
  });
  // attention from the newest token to every token so far (including itself)
  const xNew = L + (t - 1) * cw + cw / 2;
  for (let i = 0; i < t; i++) {
    const x = L + i * cw + cw / 2;
    if (i === t - 1) continue;
    g += `<path class="kv-attn" d="M${xNew},${T - 32} Q${(x + xNew) / 2},${T - 32 - Math.min(26, 6 + (t - i) * 3)} ${x},${T - 32}"/>`;
  }
  g += `<text x="8" y="${T - 16}" class="kv-label">tokens</text>`;
  for (let l = 0; l < layersShown; l++) {
    const y = T + l * rowH;
    g += `<text x="8" y="${y + ch - 2}" class="kv-label">layer ${l + 1} K</text><text x="8" y="${y + 2 * ch + 2}" class="kv-label">layer ${l + 1} V</text>`;
    for (let i = 0; i < n; i++) {
      const x = L + i * cw, c = cell(i);
      g += `<rect class="${c}" x="${x + 3}" y="${y}" width="${cw - 6}" height="${ch}" rx="3"/><rect class="${c}" x="${x + 3}" y="${y + ch + 3}" width="${cw - 6}" height="${ch}" rx="3"/>`;
    }
  }
  if (m.n_layer > layersShown) g += `<text x="${L}" y="${T + layersShown * rowH + 10}" class="kv-label">… and ${m.n_layer - layersShown} more layers, each the same</text>`;
  $("#kv-diagram").innerHTML = `<svg viewBox="0 0 ${W} ${H}">${g}</svg>
    <div class="legend"><span><i class="sw kv-key kv-new"></i>computed now</span><span><i class="sw kv-key kv-reused"></i>read from the cache</span><span><i class="sw kv-key kv-recompute"></i>recomputed (no cache)</span></div>`;
  const doneNo = (t * (t + 1)) / 2, doneKv = t;
  $("#kv-step-v").textContent = `${t} of ${n}`;
  $("#kv-explain").innerHTML = [
    stat("This step", `${on ? 1 : t} token${(on ? 1 : t) > 1 ? "s" : ""}`, on ? "only the new token runs through the layers" : `all ${t} tokens run through every layer again`),
    stat("Tokens processed so far", fmt(on ? doneKv : doneNo), on ? `without the cache: ${fmt(doneNo)}, growing with the square of the length` : `with the cache: ${fmt(doneKv)}, growing in a straight line`),
    stat("Cache held", on ? bytes(t * m.perToken) : "none", on ? `${fmt(2 * m.n_layer * t)} key and value vectors of ${m.n_embd} numbers (float32)` : `all ${fmt(2 * m.n_layer * t)} keys and values are computed again`),
    stat("Attention reads", `${t} key${t > 1 ? "s" : ""}`, "the new token's query is compared with every key so far, both ways"),
  ].join("");
}

function kvPlay() {
  if (kv.timer) { clearInterval(kv.timer); kv.timer = null; $("#kv-play").textContent = "▶ Play"; return; }
  if (kv.step >= kv.tokens.length) kv.step = 0;
  $("#kv-play").textContent = "⏸ Pause";
  kv.timer = setInterval(() => {
    kv.step++;
    $("#kv-step").value = kv.step;
    renderKv();
    if (kv.step >= kv.tokens.length) kvPlay();
  }, 800);
}

// ---------- KV cache: memory calculator ----------
function kvModels() {
  const c = ui.status?.active, m = c?.model || ui.status?.model?.config;
  const kvh = m ? m.n_kv_head || m.n_head : 0, grouped = m && m.n_kv_head && m.n_kv_head !== m.n_head;
  const own = m ? [{ name: `This model (${short(c?.params?.total || ui.status?.model?.params || 0)} parameters)`, layers: effLayers(m) || m.n_layer, width: (kvh * m.n_embd) / m.n_head, params: c?.params?.total || ui.status?.model?.params || 0, max: m.block_size,
    gqa: grouped ? `${m.n_head} query heads share ${kvh} key/value heads, so it stores ${(kvh * m.n_embd) / m.n_head} numbers per layer instead of ${m.n_embd}` : "" }] : [];
  return own.concat([
    { name: "GPT-2 small (124M)", layers: 12, width: 768, params: 124e6, max: 1024 },
    { name: "GPT-2 XL (1.5B)", layers: 48, width: 1600, params: 1.56e9, max: 1024 },
    { name: "Llama 2 7B", layers: 32, width: 4096, params: 6.7e9, max: 4096 },
    { name: "Llama 3.1 8B (grouped-query attention)", layers: 32, width: 1024, params: 8.0e9, max: 131072, gqa: "32 query heads share 8 key/value heads, so it stores 1,024 numbers per layer instead of 4,096" },
    { name: "Llama 3.1 70B (grouped-query attention)", layers: 80, width: 1024, params: 70.6e9, max: 131072, gqa: "64 query heads share 8 key/value heads" },
  ]);
}

function renderKvMem() {
  const models = kvModels(), sel = $("#kvm-model");
  const names = models.map((m) => m.name).join("|");
  if (sel.dataset.names !== names) {
    const prev = sel.value;
    sel.innerHTML = models.map((m, i) => `<option value="${i}">${esc(m.name)}</option>`).join("");
    sel.dataset.names = names;
    if (prev && +prev < models.length) sel.value = prev;
  }
  const m = models[+sel.value || 0], b = +$("#kvm-dtype").value;
  const ctx = 2 ** +$("#kvm-ctx").value, batch = 2 ** +$("#kvm-batch").value;
  $("#kvm-ctx-v").textContent = `${fmt(ctx)} tokens`;
  $("#kvm-batch-v").textContent = fmt(batch);
  const perTok = 2 * m.layers * m.width * b, cache = perTok * ctx * batch, weights = m.params * b;
  const big = Math.max(cache, weights), pct = (v) => Math.max(0.5, (v / big) * 100);
  $("#kvm-out").innerHTML = `
    <div class="kvm-bars">
      <div class="kvm-row"><span>Weights</span><div class="kvm-track"><div class="kvm-bar w" style="width:${pct(weights)}%"></div></div><b>${bytes(weights)}</b></div>
      <div class="kvm-row"><span>KV cache</span><div class="kvm-track"><div class="kvm-bar c" style="width:${pct(cache)}%"></div></div><b>${bytes(cache)}</b></div>
    </div>
    <p class="hint">${bytes(perTok)} per token = 2 × ${m.layers} layers × ${fmt(m.width)} numbers × ${b} byte${b > 1 ? "s" : ""}, × ${fmt(ctx)} tokens × ${fmt(batch)} sequence${batch > 1 ? "s" : ""}.
    ${cache > weights ? `<b>The cache is ${(cache / weights).toFixed(1)}× bigger than the model itself.</b> This is why long contexts and many users at once are limited by memory, and why servers use tricks like grouped-query attention, 8-bit caches and paged caches.` : `The cache is ${Math.round((cache / weights) * 100)}% of the weights' size.`}
    ${ctx > m.max ? ` Note: this model can only see ${fmt(m.max)} tokens.` : ""} ${m.gqa ? ` ${esc(m.gqa)}.` : ""}</p>`;
}

function drawLatency(ms) {
  const W = 360, H = 120, B = 16, n = ms.length, max = Math.max(...ms);
  const bw = (W - 8) / n;
  const bars = ms.map((v, i) => `<rect x="${4 + i * bw}" y="${H - B - (v / max) * (H - B - 8)}" width="${Math.max(1, bw - 1)}" height="${(v / max) * (H - B - 8)}" fill="${i === 0 ? "var(--series-val)" : "var(--accent)"}"><title>token ${i + 1}: ${v} ms</title></rect>`).join("");
  $("#g-latency").innerHTML = `<svg viewBox="0 0 ${W} ${H}">${bars}<text x="4" y="${H - 3}">1</text><text x="${W - 4}" y="${H - 3}" text-anchor="end">${n}</text></svg>`;
  const rest = ms.slice(1), avg = rest.length ? rest.reduce((a, b) => a + b, 0) / rest.length : 0;
  $("#g-lat-note").textContent = `first token ${Math.round(ms[0])} ms (orange), then about ${avg.toFixed(0)} ms each`;
}

async function generate() {
  const body = { prompt: $("#g-prompt").value, ...samplingBody(), max_new_tokens: +$("#g-len").value, use_kv_cache: $("#g-kv").checked };
  const res = await fetch("/api/generate/stream", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  if (!res.ok) { const j = await res.json().catch(() => ({})); throw new Error(j.detail || res.statusText); }
  // keep the previous output in the history
  const prevText = $("#g-text").innerHTML;
  if (prevText && ui.lastGen) {
    const item = document.createElement("div");
    item.className = "card gen-item";
    item.innerHTML = `<div class="doc">${ui.lastGen}</div>`;
    $("#g-history").prepend(item);
    while ($("#g-history").children.length > 4) $("#g-history").lastChild.remove();
  }
  $("#g-live").classList.remove("hidden");
  const out = $("#g-text");
  out.classList.toggle("plain", !$("#g-color").checked);
  out.innerHTML = `<span class="prompt">${esc(body.prompt)}</span>`;
  let meta = null, count = 0, probSum = 0, lastMs = 0, nlRun = 0;
  const reader = res.body.getReader(), dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let nl;
    while ((nl = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, nl); buf = buf.slice(nl + 1);
      if (!line.trim()) continue;
      const ev = JSON.parse(line);
      if (ev.type === "start") { meta = ev; renderGenMetrics({ ...ev, output_tokens: 0 }, true); }
      else if (ev.type === "token") {
        count++; probSum += ev.prob; lastMs = ev.ms;
        const alts = ev.alternatives.map((a) => `${JSON.stringify(a.text)} ${(a.prob * 100).toFixed(1)}%`).join("\n");
        const span = document.createElement("span");
        span.className = `gt ${confClass(ev.prob)}`;
        span.title = `picked ${JSON.stringify(ev.text)} (${(ev.prob * 100).toFixed(1)}%), drawn from ${fmt(ev.kept)} candidate${ev.kept === 1 ? "" : "s"}\nmodel's top choices:\n${alts}`;
        // undertrained models emit long runs of blank lines: show at most one empty line in a row
        let shown = ev.text;
        if (/^\n+$/.test(shown)) { shown = "\n".repeat(Math.max(0, Math.min(shown.length, 2 - nlRun))); nlRun += ev.text.length; }
        else { const m = shown.match(/\n+$/); nlRun = m ? m[0].length : 0; }
        span.textContent = shown;
        out.appendChild(span);
        if (count === 1 || count % 5 === 0) renderGenMetrics({ ...meta, output_tokens: count, ttft_ms: count === 1 ? ev.ms : undefined, total_ms: ev.ms, avg_prob: probSum / count }, true);
      } else if (ev.type === "done") {
        renderGenMetrics({ ...meta, ...ev });
        if (ev.token_ms.length) drawLatency(ev.token_ms);
        if (ev.stopped) { const end = document.createElement("span"); end.className = "story-end"; end.textContent = " ⏹ end of story"; out.appendChild(end); }
        ui.lastGen = out.innerHTML;
        ui.generated = true;
      }
    }
  }
  render();
}

// ---------- 8. models & evaluation ----------
const MODEL_COLORS = ["var(--accent)", "var(--series-val)", "var(--series-train)", "#9b5cf6", "#c2410c", "#0e7490", "#be185d", "#4d7c0f"];
const lib = { data: null, picked: new Set() };

async function loadModels() {
  lib.data = await (await fetch("/api/models")).json();
  const ids = new Set(lib.data.models.map((m) => m.id));
  lib.picked = new Set([...lib.picked].filter((i) => ids.has(i)));
  renderLibrary();
  renderEvaluation();
}

const shortData = (ds) => (ds || []).map((n) => (n.startsWith("http") ? shortUrl(n) : n.split("/").pop())).join(" + ") || "–";
const when = (t) => (t ? esc(t.slice(0, 16)) : "–");

function renderLibrary() {
  const r = lib.data, models = r.models;
  const total = models.reduce((a, m) => a + (m.bytes || 0), 0);
  $("#lib-summary").textContent = models.length ? `${models.length} model${models.length > 1 ? "s" : ""} · ${bytes(total)} on disk` : "";
  if (!models.length) {
    $("#lib-table").innerHTML = '<p class="muted">No saved models yet. Train one, import one from Colab, or press <b>Find models on this Mac</b>.</p>';
    return;
  }
  $("#lib-table").innerHTML = `<div class="ptable-scroll"><table class="ptable lib">
    <thead><tr><th><input type="checkbox" id="lib-all" title="Select all"></th><th>Model</th><th>Trained on</th><th class="num">Best val loss</th><th class="num">Evaluated</th></tr></thead>
    <tbody>${models.map((m) => `<tr class="${m.active ? "in-use" : ""}">
      <td><input type="checkbox" data-pick="${esc(m.id)}" ${lib.picked.has(m.id) ? "checked" : ""}></td>
      <td><div class="lib-name" title="${esc(m.id)}">${esc(m.name)} ${m.active ? '<span class="tag good">in use</span>' : ""}</div>
        <div class="muted small">${short(m.params || 0)} parameters${m.model ? ` · ${m.model.n_layer} layers · ${m.model.n_embd}-dim · context ${m.model.block_size} · ${m.model.pos_encoding === "rope" ? "RoPE" : "learned positions"}${m.model.norm === "rmsnorm" ? " · RMSNorm" : ""}${m.model.mlp === "swiglu" ? " · SwiGLU" : ""}${m.model.bias === false ? " · no biases" : ""}${m.model.qk_norm ? " · QK-norm" : ""}${m.model.n_kv_head && m.model.n_kv_head !== m.model.n_head ? ` · ${m.model.n_kv_head} KV heads` : ""}${(m.model.block_repeat || 1) > 1 ? ` · blocks shared ×${m.model.block_repeat}` : ""}${(m.model.mtp_heads || 1) > 1 ? ` · trained to predict ${m.model.mtp_heads} tokens` : ""}${(m.model.n_experts || 1) > 1 ? ` · ${m.model.n_experts} experts, ${m.model.moe_top_k || 1} per token` : ""}` : ""} · ${bytes(m.bytes || 0)} · ${esc(m.model_type || "autoregressive")} · ${esc(m.source)}</div>
        <div class="lib-actions">${m.active ? "" : `<button class="btn small" data-use="${esc(m.id)}">Use for generation</button>`}<button class="btn small ghost" data-rename="${esc(m.id)}">Rename</button><button class="btn small ghost danger" data-delete="${esc(m.id)}">Delete</button></div></td>
      <td>${esc(shortData(m.datasets))}<div class="muted small">${m.steps ? `${fmt(m.steps)} steps` : ""}${m.chip ? ` · ${esc(m.chip)}` : ""}${m.platform ? ` (${esc(m.platform)})` : ""}</div><div class="muted small">${when(m.trained_at)}</div></td>
      <td class="num">${fmtLoss(m.best_val)}<div class="muted small">on its own data</div></td>
      <td class="num">${m.eval ? `<span class="tag">${when(m.eval.evaluated_at)}</span>` : '<span class="muted small">not yet</span>'}</td>
    </tr>`).join("")}</tbody></table></div>`;
  $("#lib-all").checked = models.every((m) => lib.picked.has(m.id));
}

function renderEvaluation() {
  const r = lib.data, s = r.suite;
  $("#ev-suite").innerHTML = `<ol class="flow small-flow">
      <li><b>Held-out loss</b> on ${r.eval_datasets.length ? r.eval_datasets.map((d) => `<b>${esc(d.title)}</b>`).join(", ") : "<i>no tokenized datasets yet</i>"}: ${s.windows} fixed windows of ${s.context} tokens from each validation split, the same positions for every model (models with a shorter context window read shorter windows). Datasets you add and tokenize in step 1 join the suite.</li>
      <li><b>Generation</b>: ${s.prompts.length} fixed prompts, ${s.tokens} tokens each, with a fixed seed, decoded <b>greedy</b> and with the <b>Balanced</b> preset. Measures repetition (4-token runs seen before), variety (distinct word pairs), confidence, speed with the KV cache, and time to first token.</li>
      <li><b>Size</b>: parameters, weights in memory, KV cache per token.</li></ol>`;
  const evald = r.models.filter((m) => m.eval && m.eval.suite_version === s.version);
  if (!evald.length) {
    $("#ev-table").innerHTML = '<p class="muted">No results yet. Tick some models above and press <b>Evaluate selected models</b>.</p>';
    $("#ev-chart").innerHTML = ""; $("#ev-samples").innerHTML = ""; return;
  }
  const dsKeys = [], titles = {};
  evald.forEach((m) => Object.entries(m.eval.heldout || {}).forEach(([k, v]) => { if (!dsKeys.includes(k)) { dsKeys.push(k); titles[k] = v.title; } }));
  const cols = [
    ...dsKeys.map((k) => ({ label: `${titles[k]} loss`, get: (m) => m.eval.heldout?.[k]?.loss, fmt: (v, m) => `${v.toFixed(3)}<div class="muted small">ppl ${fmt(m.eval.heldout[k].perplexity >= 100 ? Math.round(m.eval.heldout[k].perplexity) : m.eval.heldout[k].perplexity)}</div>`, best: "min" })),
    { label: "Greedy repetition", tip: "share of 4-token runs repeating earlier ones, greedy decoding", get: (m) => m.eval.generation.greedy_repeated_4grams, fmt: (v) => `${Math.round(v * 100)}%`, best: "min" },
    { label: "Distinct pairs", tip: "Balanced preset: share of word pairs that are unique", get: (m) => m.eval.generation.balanced_distinct_2, fmt: (v) => `${Math.round(v * 100)}%`, best: "max" },
    { label: "Confidence", get: (m) => m.eval.generation.balanced_avg_prob, fmt: (v) => `${Math.round(v * 100)}%` },
    { label: "Speed", get: (m) => m.eval.generation.tokens_per_sec, fmt: (v) => `${Math.round(v)} tok/s`, best: "max" },
    { label: "First token", get: (m) => m.eval.generation.ttft_ms, fmt: (v) => `${Math.round(v)} ms`, best: "min" },
    { label: "Weights", get: (m) => m.eval.size.weights_bytes, fmt: (v, m) => `${bytes(v)}<div class="muted small">${short(m.eval.size.params)} params</div>` },
    { label: "KV cache / token", get: (m) => m.eval.size.kv_bytes_per_token, fmt: (v) => bytes(v) },
  ];
  const bestOf = cols.map((c) => { const vals = evald.map(c.get).filter((v) => v != null); return !c.best || vals.length < 2 ? null : c.best === "min" ? Math.min(...vals) : Math.max(...vals); });
  $("#ev-table").innerHTML = `<div class="ptable-scroll"><table class="ptable ev">
    <thead><tr><th>Model</th>${cols.map((c) => `<th class="num" title="${esc(c.tip || "")}">${esc(c.label)}</th>`).join("")}</tr></thead>
    <tbody>${evald.map((m, i) => `<tr><td><span class="sw-dot" style="background:${MODEL_COLORS[i % MODEL_COLORS.length]}"></span>${esc(m.name)}</td>${cols.map((c, j) => { const v = c.get(m); return `<td class="num">${v == null ? "–" : `<span class="${v === bestOf[j] ? "best" : ""}">${c.fmt(v, m)}</span>`}</td>`; }).join("")}</tr>`).join("")}</tbody></table></div>
    <p class="hint">A model scores best on the data it was trained on: compare models <i>within</i> a column. Loss is the average surprise per token (lower is better); perplexity is e<sup>loss</sup>, roughly how many tokens the model is choosing between. Read repetition and variety together with loss: a model that barely trained shows 100% distinct pairs because its output is noise.</p>`;
  // grouped bars: held-out loss per dataset
  const maxL = Math.max(...evald.flatMap((m) => dsKeys.map((k) => m.eval.heldout?.[k]?.loss || 0)));
  $("#ev-chart").innerHTML = `<div class="ev-bars">${dsKeys.map((k) => `<div class="ev-group"><div class="ev-gtitle">${esc(titles[k])}</div>
    ${evald.map((m, i) => { const v = m.eval.heldout?.[k]?.loss; return `<div class="kvm-row"><span class="muted small" title="${esc(m.name)}">${esc(m.name.split(" · ").slice(0, 2).join(" · "))}</span><div class="kvm-track"><div class="kvm-bar" style="width:${v ? (v / maxL) * 100 : 0}%;background:${MODEL_COLORS[i % MODEL_COLORS.length]}"></div></div><b>${v ? v.toFixed(2) : "–"}</b></div>`; }).join("")}</div>`).join("")}</div>`;
  // samples
  const sel = $("#ev-prompt");
  if (sel.options.length !== s.prompts.length) sel.innerHTML = s.prompts.map((p, i) => `<option value="${i}">${esc(p)}</option>`).join("");
  renderSamples();
}

function renderSamples() {
  if (!lib.data) return;
  const s = lib.data.suite, i = +$("#ev-prompt").value || 0, mode = $("#ev-mode").value;
  const evald = lib.data.models.filter((m) => m.eval && m.eval.suite_version === s.version);
  $("#ev-samples").innerHTML = evald.map((m, j) => {
    const smp = m.eval.samples?.[i];
    return `<div class="cmp-col"><div class="cmp-name"><span class="sw-dot" style="background:${MODEL_COLORS[j % MODEL_COLORS.length]}"></span>${esc(m.name)}</div>
      <div class="doc cmp-text"><span class="prompt">${esc(smp?.prompt || s.prompts[i])}</span>${esc(smp?.[mode] || "")}</div></div>`;
  }).join("");
}

async function libAction(fn) { await fn(); await loadModels(); await poll(); }

// ---------- events ----------
document.addEventListener("click", (e) => {
  const t = e.target;
  const g = t.closest("[data-go]"); if (g) go(g.dataset.go);
  const sc = t.closest("[data-scroll]"); if (sc) { e.preventDefault(); $(`#${sc.dataset.scroll}`).scrollIntoView({ behavior: "smooth" }); }
  const add = t.closest("[data-add]");
  if (add) { const c = catalog[+add.dataset.add]; action(add, () => addDataset({ name: c.dataset, config_name: c.config, text_column: c.text_column })); }
  const rm = t.closest("[data-remove]");
  if (rm) action(rm, async () => { await api("/api/dataset/remove", { key: rm.dataset.remove }); await poll(); loadCatalog(); });
  const tag = t.closest("[data-tag]"); if (tag) { galleryTag = tag.dataset.tag; renderGallery(); }
  const cfg = t.closest("[data-config]"); if (cfg) { $("#ds-config").value = cfg.dataset.config; action(null, checkDataset); }
  const ex = t.closest("[data-example]"); if (ex) { $("#tok-input").value = ex.dataset.example; previewTokens(); $("#tok-input").scrollIntoView({ behavior: "smooth", block: "center" }); }
  const mp = t.closest("[data-mpreset]"); if (mp) setModelFields(MODEL_PRESETS[mp.dataset.mpreset]);
  const st = t.closest("[data-stage]"); if (st && ui.walk) { ui.walk.stage = +st.dataset.stage; renderWalk(); }
  const fr = t.closest("[data-focus]"); if (fr && ui.walk) { ui.walk.focus = +fr.dataset.focus; renderWalk(); }
  const ap = t.closest("[data-append]"); if (ap) { $("#a-prompt").value += ap.dataset.append; action(null, runWalk); }
  const hd = t.closest("[data-head]"); if (hd) { ui.realHead = hd.dataset.head === "avg" ? "avg" : +hd.dataset.head; renderReal(); }
  const mix = t.closest("[data-mix]"); if (mix) action(null, async () => { await api("/api/mix", { mode: mix.dataset.mix }); await poll(); });
});

$$("#ds-name, #ds-config, #ds-column").forEach((i) => i.addEventListener("input", () => { $("#ds-check-result").className = "check hidden"; }));
$("#ds-check").addEventListener("click", (e) => action(e.target, checkDataset));
$("#ds-load").addEventListener("click", (e) => action(e.target, () => addDataset(dsBody())));
$$("#ds-split button").forEach((b) => b.addEventListener("click", () => { ui.sample.split = b.dataset.split; loadSample(0); }));
$("#ds-browse-which").addEventListener("change", (e) => { ui.sample = { key: e.target.value, split: "train", index: 0, loadedFor: e.target.value }; loadSample(0); });
$("#ds-prev").addEventListener("click", () => loadSample(Math.max(0, ui.sample.index - 1)));
$("#ds-next").addEventListener("click", () => loadSample(ui.sample.index + 1));
$("#ds-rand").addEventListener("click", () => loadSample(-1));

$("#tok-input").addEventListener("input", () => { clearTimeout(previewTimer); previewTimer = setTimeout(previewTokens, 150); });
$("#tok-run").addEventListener("click", (e) => action(e.target, async () => { await api("/api/tokenize", {}); await poll(); }));

$("#b-run").addEventListener("click", (e) => action(e.target, async () => {
  const r = await api("/api/batch", { block_size: +$("#b-block").value, batch_size: +$("#b-batch").value, seed: +$("#b-seed").value });
  ui.batchSampled = true;
  renderBatch(r);
  await poll();
}));

$$("#m-layer, #m-head, #m-embd").forEach((i) => i.addEventListener("input", renderEstimate));
$("#m-exp").addEventListener("change", () => { renderExpHint(); renderEstimate(); renderModelPresets(); });
$("#m-topk").addEventListener("change", () => { renderExpHint(); renderEstimate(); });
$("#m-mtp").addEventListener("change", () => { renderMtpHint(); renderEstimate(); renderModelPresets(); });
$("#m-rep").addEventListener("change", () => { renderRepHint(); renderEstimate(); });
$("#m-layer").addEventListener("input", () => renderRepHint());
$("#m-kv").addEventListener("change", () => { renderKvOptions(); renderEstimate(); renderModelPresets(); });
$$("#m-head, #m-embd").forEach((i) => i.addEventListener("input", () => { renderKvOptions(); renderEstimate(); }));
$("#m-qkn").addEventListener("change", () => { $("#m-qkn-hint").textContent = QKN_HINTS[$("#m-qkn").value]; renderEstimate(); renderModelPresets(); });
$("#m-bias").addEventListener("change", () => { $("#m-bias-hint").textContent = BIAS_HINTS[$("#m-bias").value]; renderEstimate(); renderModelPresets(); });
$("#m-mlp").addEventListener("change", () => { $("#m-mlp-hint").textContent = MLP_HINTS[$("#m-mlp").value]; renderEstimate(); renderModelPresets(); });
$("#m-norm").addEventListener("change", () => { $("#m-norm-hint").textContent = NORM_HINTS[$("#m-norm").value]; renderEstimate(); renderModelPresets(); });
$("#m-pos").addEventListener("change", () => { $("#m-pos-hint").textContent = POS_HINTS[$("#m-pos").value]; renderEstimate(); renderModelPresets(); });
$("#m-run").addEventListener("click", (e) => action(e.target, async () => {
  const r = await api("/api/model", modelFields());
  renderModel(r);
  await poll();
}));

$("#a-run").addEventListener("click", (e) => action(e.target, runWalk));
$("#a-prompt").addEventListener("keydown", (e) => { if (e.key === "Enter") action($("#a-run"), runWalk); });
$("#a-prev").addEventListener("click", () => { if (ui.walk && ui.walk.stage > 0) { ui.walk.stage--; renderWalk(); } });
$("#a-next").addEventListener("click", () => { if (ui.walk && ui.walk.stage < STAGES.length - 1) { ui.walk.stage++; renderWalk(); } });
$("#a-layer").addEventListener("change", () => action(null, async () => {
  ui.real = await api("/api/inspect", { prompt: $("#a-prompt").value, layer: +$("#a-layer").value });
  renderReal();
}));

$$("#t-preset button").forEach((b) => b.addEventListener("click", () => {
  if (b.dataset.preset === "custom") highlightPreset("custom");
  else setTrainPreset(b.dataset.preset);
}));
TRAIN_FIELDS.forEach((f) => $(`#t-${f}`).addEventListener("input", () => highlightPreset(matchPreset(trainValues()))));
$("#t-start").addEventListener("click", (e) => {
  if (ui.status?.selection?.checkpoint && !confirm("This trains a new model for this training data. The current one is kept in the Models step's library. Continue?")) return;
  action(e.target, async () => { await api("/api/train", trainValues()); await poll(); });
});
$("#t-stop").addEventListener("click", (e) => action(e.target, () => api("/api/train/stop", {})));
$("#t-resume-run").addEventListener("click", (e) => action(e.target, async () => {
  await api("/api/train/resume", { max_iters: +$("#t-resume-to").value });
  await poll();
}));

$$("#g-temp, #g-topk, #g-len, #g-topp, #g-minp, #g-rep, #g-freq, #g-pres, #g-typ, #g-dry").forEach((i) => i.addEventListener("input", () => { syncSliders(); if (i.id !== "g-len") refreshDist(); }));
$("#g-prompt").addEventListener("input", refreshDist);
$("#cmp-run").addEventListener("click", (e) => action(e.target, runCompare));
$("#lib-scan").addEventListener("click", (e) => action(e.target, () => libAction(async () => {
  const r = await api("/api/models/scan", {});
  toast(r.added.length ? `Added ${r.added.length}: ${r.added.join("; ")}` : "No new models found.");
})));
$("#lib-save").addEventListener("click", (e) => action(e.target, () => libAction(async () => {
  const r = await api("/api/models/save-current", {});
  toast(r.added ? `Saved as "${r.name}".` : `Already saved as "${r.name}".`);
})));
$("#lib-default").addEventListener("click", (e) => action(e.target, () => libAction(() => api("/api/models/use", { id: null }))));
$("#ev-run").addEventListener("click", (e) => action(e.target, async () => {
  if (!lib.picked.size) throw new Error("Tick at least one model in the table first.");
  await api("/api/eval/run", { ids: [...lib.picked] });
  ui.evalRunning = true;
  await poll();
}));
$("#ev-prompt").addEventListener("change", renderSamples);
$("#ev-mode").addEventListener("change", renderSamples);
document.addEventListener("change", (e) => {
  const p = e.target.closest("[data-pick]");
  if (p) { p.checked ? lib.picked.add(p.dataset.pick) : lib.picked.delete(p.dataset.pick); $("#lib-all").checked = lib.data.models.every((m) => lib.picked.has(m.id)); }
  if (e.target.id === "lib-all") { lib.picked = e.target.checked ? new Set(lib.data.models.map((m) => m.id)) : new Set(); renderLibrary(); }
});
document.addEventListener("click", (e) => {
  const u = e.target.closest("[data-use]");
  if (u) action(u, () => libAction(async () => { await api("/api/models/use", { id: u.dataset.use }); toast("Generate and Inside the transformer now use this model."); }));
  const rn = e.target.closest("[data-rename]");
  if (rn) {
    const m = lib.data.models.find((x) => x.id === rn.dataset.rename);
    const name = prompt("New name for this model:", m?.name || "");
    if (name && name.trim()) action(rn, () => libAction(() => api("/api/models/rename", { id: rn.dataset.rename, name })));
  }
  const del = e.target.closest("[data-delete]");
  if (del) {
    const m = lib.data.models.find((x) => x.id === del.dataset.delete);
    if (confirm(`Delete "${m?.name}" from the library? Its saved weights and evaluation results (${bytes(m?.bytes || 0)}) are removed for good. Models in data/ aren't touched.`)) {
      action(del, () => libAction(() => api("/api/models/delete", { id: del.dataset.delete })));
    }
  }
});
document.addEventListener("click", (e) => { const b = e.target.closest("[data-spreset]"); if (b) setSampling(SAMPLING_PRESETS[b.dataset.spreset]); });
$("#g-color").addEventListener("change", (e) => $("#g-text").classList.toggle("plain", !e.target.checked));
$("#g-run").addEventListener("click", (e) => action(e.target, generate));
$("#g-bench").addEventListener("click", (e) => action(e.target, runBenchmark));
$("#kv-play").addEventListener("click", kvPlay);
$("#kv-step").addEventListener("input", (e) => { kv.step = +e.target.value; renderKv(); });
$("#kv-on").addEventListener("change", renderKv);
$("#g-prompt").addEventListener("change", () => loadKvTokens().then(renderKv));
$$("#kvm-model, #kvm-dtype, #kvm-ctx, #kvm-batch").forEach((i) => i.addEventListener("input", renderKvMem));

// ---------- polling ----------
async function poll() {
  try { ui.status = await api("/api/status"); } catch (e) { return; }
  const j = ui.status.job, prev = ui.prevJob;
  if (prev && prev.status === "running" && j.status !== "running") {
    if (j.status === "error") toast(j.error || "Something went wrong.", true);
    else if (j.kind === "dataset") toast(j.message || "Dataset added.");
    else if (j.kind === "tokenize") toast("Training data tokenized.");
    else if (j.kind === "train") toast(j.message);
    loadCatalog();  // refresh Downloaded / Tokenized / Trained badges
  }
  ui.prevJob = j;
  render();
}

const hashStep = location.hash.slice(1);
if (STEPS.some((s) => s.id === hashStep)) ui.step = hashStep;  // e.g. http://127.0.0.1:8000/#train
window.addEventListener("hashchange", () => {
  const id = location.hash.slice(1);
  if (STEPS.some((s) => s.id === id) && id !== ui.step) go(id);
});
syncSliders();
previewTokens();
loadCatalog();
poll().then(() => go(STEPS.some((s) => s.id === ui.step) ? ui.step : "dataset"));
setInterval(poll, 1000);

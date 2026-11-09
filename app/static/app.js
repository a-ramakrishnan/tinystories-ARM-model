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
  { id: "generate", title: "Generate", sub: "Write stories" },
];

const TRAIN_PRESETS = {
  quick: { max_iters: 2000, learning_rate: 1e-4, min_lr: 1e-5, warmup_steps: 200, eval_interval: 250, eval_iters: 50, gradient_accumulation_steps: 8 },
  full: { max_iters: 20000, learning_rate: 1e-4, min_lr: 1e-5, warmup_steps: 1000, eval_interval: 500, eval_iters: 500, gradient_accumulation_steps: 32 },
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
  }
}

function stepBlocker(id) {
  const s = ui.status, sel = s?.selection;
  if (id === "dataset") return null;
  if (!sel) return ["Add a dataset first.", "dataset"];
  if (id === "tokenize") return null;
  if (!sel.tokenized) return ["Tokenize your training data first.", "tokenize"];
  if (id === "batches" || id === "model") return null;
  if (id === "attention") return s.model || sel.checkpoint ? null : ["Build a model first.", "model"];
  if (id === "train") return s.model || sel.checkpoint ? null : ["Build a model first.", "model"];
  if (id === "generate") return sel.checkpoint ? null : ["Train a model first.", "train"];
  return null;
}

function go(id) {
  ui.step = id;
  try { localStorage.setItem("lmstudio.step", id); } catch (e) { /* ignore */ }
  history.replaceState(null, "", `#${id}`);
  $$(".panel").forEach((p) => p.classList.toggle("active", p.dataset.step === id));
  window.scrollTo({ top: 0, behavior: "smooth" });
  render();
  if (id === "batches") renderBatchExample();
  if (id === "tokenize") renderTokenExamples();
}

// ---------- rendering ----------
function render() {
  const s = ui.status;
  if (!s) return;
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
    const busy = busyStep === st.id;
    const cls = ["step", st.id === ui.step && "active", stepDone(st.id) && "done", busy && "busy"].filter(Boolean).join(" ");
    return `<li><button class="${cls}" data-go="${st.id}"><span class="num">${stepDone(st.id) && !busy ? "✓" : i + 1}</span><span><div class="t">${st.title}</div><div class="s">${st.sub}</div></span></button></li>`;
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
function estimateParams(L, d, T) { return VOCAB * d + T * d + L * (12 * d * d + 13 * d) + 2 * d; }
const modelFields = () => ({ n_layer: +$("#m-layer").value, n_head: +$("#m-head").value, n_embd: +$("#m-embd").value, dropout: +$("#m-drop").value });

function setModelFields(c) {
  $("#m-layer").value = c.n_layer; $("#m-head").value = c.n_head; $("#m-embd").value = c.n_embd;
  if (c.dropout != null) $("#m-drop").value = c.dropout;
  else if (!$("#m-drop").value) $("#m-drop").value = 0.1;
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
  const box = (cls, title, sub, params) => `<div class="abox ${cls}"><div><div class="at">${title}</div>${sub ? `<div class="as">${sub}</div>` : ""}</div>${params != null ? `<span class="ap">${short(params)}</span>` : ""}</div>`;
  $("#m-arch").innerHTML = `<div class="arch2">
    ${box("io", "Input: token IDs", `up to <b>${c.block_size}</b> tokens · shape (batch, ${c.block_size})`, null)}${arrow}
    ${box("emb", "Token embedding + position embedding", `each token becomes a vector of ${c.n_embd} numbers · ${fmt(c.vocab_size)} × ${c.n_embd} and ${c.block_size} × ${c.n_embd} tables`, part("Token embeddings") + part("Position embeddings"))}${arrow}
    <div class="stackbox"><div class="stack-badge">× ${c.n_layer} layer${c.n_layer > 1 ? "s" : ""}</div>
      <div class="stack-title">Transformer block <span class="ap">${short(b.block)} each</span></div>
      <div class="sub-flow">
        <div class="mini">LayerNorm</div><div class="arr r">→</div>
        <div class="abox attn"><div><div class="at">Multi-head self-attention</div><div class="as">${c.n_head} heads × ${hs} dims · causal mask (no peeking ahead)</div><div class="heads">${heads}</div></div></div>
        <div class="arr r">→</div><div class="resid">+ residual</div>
      </div>
      <div class="sub-flow">
        <div class="mini">LayerNorm</div><div class="arr r">→</div>
        <div class="abox ffn"><div><div class="at">Feed-forward network (MLP)</div><div class="as">Linear ${c.n_embd} → ${4 * c.n_embd} · GELU · Linear ${4 * c.n_embd} → ${c.n_embd}</div></div></div>
        <div class="arr r">→</div><div class="resid">+ residual</div>
      </div>
    </div>${arrow}
    ${box("", "Final LayerNorm", "", part("Final LayerNorm"))}${arrow}
    ${box("emb", "Output layer (shares the token embeddings)", `a score (logit) for each of the ${fmt(c.vocab_size)} tokens at every position · shape (batch, ${c.block_size}, ${fmt(c.vocab_size)})`, 0)}${arrow}
    ${box("io", "Softmax → next-token probabilities", "sample one token, append it, repeat", null)}
  </div>`;
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
const W_Q = [[1.0, -0.4, 0.2, 0.6], [0.2, 0.8, -0.6, 0.4], [-0.4, 0.6, 1.0, -0.2], [0.6, 0.2, 0.4, 0.8]];
const W_K = [[0.8, 0.2, -0.4, 0.4], [-0.2, 1.0, 0.4, 0.2], [0.6, -0.4, 0.8, 0.2], [0.2, 0.4, 0.2, 1.0]];
const W_V = [[0.2, 0.5, 0.1, -0.3], [0.4, -0.1, 0.3, 0.2], [-0.3, 0.2, 0.4, 0.1], [0.1, 0.3, -0.2, 0.4]];
const matmul = (A, B) => A.map((row) => B[0].map((_, j) => r2(row.reduce((s, a, k) => s + a * B[k][j], 0))));
const transpose = (A) => A[0].map((_, j) => A.map((r) => r[j]));

function buildWalk(tokens) {
  const T = tokens.length;
  const E = tokens.map((t) => { const rnd = prng(t.id + 1); return [0, 1, 2, 3].map(() => r1(rnd() * 4 - 2)); });
  const P = tokens.map((_, t) => [r1(Math.sin(t) * 0.5), r1(Math.cos(t) * 0.5), r1(Math.sin(t / 2) * 0.5), r1(Math.cos(t / 2) * 0.5)]);
  const X = E.map((e, i) => e.map((v, j) => r1(v + P[i][j])));
  const Q = matmul(X, W_Q), K = matmul(X, W_K), V = matmul(X, W_V);
  const S = matmul(Q, transpose(K)).map((row) => row.map((v) => r2(v / 2)));  // ÷ √d_k = √4 = 2
  const A = S.map((row, i) => {
    const vis = row.slice(0, i + 1), m = Math.max(...vis);
    const ex = vis.map((v) => Math.exp(v - m)), sum = ex.reduce((a, b) => a + b, 0);
    return row.map((_, j) => (j <= i ? r2(ex[j] / sum) : 0));
  });
  const C = A.map((row) => [0, 1, 2, 3].map((k) => r2(row.reduce((s, w, j) => s + w * V[j][k], 0))));
  return { tokens, T, E, P, X, Q, K, V, S, A, C, stage: 0, focus: T - 1 };
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

const tokLabel = (t) => esc(t.text).replace(/ /g, "·");

const STAGES = [
  {
    title: "1. Tokens become IDs",
    body: (w) => `<p>The prompt is tokenized. From here on the model only sees these numbers.</p>
      <div class="chips">${w.tokens.map((t, i) => `<span class="idchip">${tokenChip(t, i)}<b>${t.id}</b></span>`).join("")}</div>
      <p class="muted small">Next we follow these ${w.T} tokens through one attention head.</p>`,
  },
  {
    title: "2. Embeddings: each token becomes a vector",
    body: (w, c) => `<p>Each ID picks a row from the <b>token embedding table</b>, and each position adds a row from the <b>position embedding table</b>. The sum <code>x = e + p</code> says both <i>what</i> the token is and <i>where</i> it is.</p>
      <div class="mats">${matrix(w.E, { rows: w.tokens.map(tokLabel), title: "token embedding e", digits: 1 })}<div class="op">+</div>${matrix(w.P, { rows: w.tokens.map((_, i) => `pos ${i}`), title: "position p", digits: 1 })}<div class="op">=</div>${matrix(w.X, { rows: w.tokens.map(tokLabel), title: "x", digits: 1, heat: true })}</div>
      <p class="muted small">Toy numbers with 4 dimensions. Your model uses ${c.n_embd} dimensions, looked up from a ${fmt(c.vocab_size)} × ${c.n_embd} table that it learns during training.</p>`,
  },
  {
    title: "3. Queries, keys and values",
    body: (w, c) => `<p>Three learned matrices turn every <code>x</code> into three vectors:</p>
      <ul class="qkv"><li><b>Query (Q)</b>: what this token is looking for.</li><li><b>Key (K)</b>: what this token offers to others.</li><li><b>Value (V)</b>: the information it passes on if someone attends to it.</li></ul>
      <div class="mats">${matrix(w.X, { rows: w.tokens.map(tokLabel), title: "x", digits: 1 })}<div class="op">×</div>${matrix(W_Q, { title: "W<sub>Q</sub>", digits: 1 })}<div class="op">=</div>${matrix(w.Q, { rows: w.tokens.map(tokLabel), title: "Q", heat: true })}</div>
      <div class="mats">${matrix(w.K, { rows: w.tokens.map(tokLabel), title: "K = x × W<sub>K</sub>", heat: true })}${matrix(w.V, { rows: w.tokens.map(tokLabel), title: "V = x × W<sub>V</sub>", heat: true })}</div>
      <p class="muted small">In your model each head has its own W<sub>Q</sub>, W<sub>K</sub>, W<sub>V</sub> of size ${c.n_embd} × ${c.n_embd / c.n_head}.</p>`,
  },
  {
    title: "4. Attention scores: how well each query matches each key",
    body: (w) => {
      const i = w.focus, j = 0, q = w.Q[i], k = w.K[j];
      const terms = q.map((v, n) => `${v.toFixed(2)}×${k[n].toFixed(2)}`).join(" + ");
      return `<p>Each query is compared with every key using a dot product, divided by <code>√d<sub>k</sub> = √4 = 2</code> to keep the numbers in a good range. Bigger score = better match. Click a row to follow a different token.</p>
        <div class="mats">${matrix(w.S, { rows: w.tokens.map(tokLabel), cols: w.tokens.map(tokLabel), title: "scores = Q · Kᵀ / 2", heat: true, focus: i, clickable: true })}</div>
        <div class="calc">score(“${tokLabel(w.tokens[i])}” → “${tokLabel(w.tokens[j])}”) = (${terms}) / 2 = <b>${w.S[i][j].toFixed(2)}</b></div>`;
    },
  },
  {
    title: "5. Causal mask: no peeking at the future",
    body: (w) => `<p>When predicting what comes next, a token must not see tokens after it, or it could cheat by reading the answer. Scores for future positions are set to <code>−∞</code>, which becomes 0 after softmax.</p>
      <div class="mats">${matrix(w.S, { rows: w.tokens.map(tokLabel), cols: w.tokens.map(tokLabel), title: "masked scores", heat: true, mask: true, focus: w.focus, clickable: true })}</div>
      <p class="muted small">This is what makes it a <i>decoder-only</i> or <i>causal</i> language model.</p>`,
  },
  {
    title: "6. Softmax: scores become attention weights",
    body: (w) => {
      const i = w.focus;
      const bars = w.A[i].slice(0, i + 1).map((v, j) => `<div class="wbar"><span class="wl">${tokLabel(w.tokens[j])}</span><div class="wtrack"><div style="width:${v * 100}%"></div></div><span class="wv">${(v * 100).toFixed(0)}%</span></div>`).join("");
      return `<p>Softmax turns each row into percentages that add up to 100%: how much attention a token pays to each earlier token (and itself).</p>
        <div class="mats">${matrix(w.A, { rows: w.tokens.map(tokLabel), cols: w.tokens.map(tokLabel), title: "attention weights", heat: true, mask: true, focus: i, clickable: true })}
        <div class="wbars"><div class="mat-title">“${tokLabel(w.tokens[i])}” attends to</div>${bars}</div></div>`;
    },
  },
  {
    title: "7. The context vector: a weighted mix of values",
    body: (w) => {
      const i = w.focus;
      const terms = w.A[i].slice(0, i + 1).map((a, j) => `<div class="term"><span class="wv">${a.toFixed(2)}</span>×<span class="vec">[${w.V[j].map((v) => v.toFixed(2)).join(", ")}]</span><span class="muted small">V of “${tokLabel(w.tokens[j])}”</span></div>`).join("");
      return `<p>Each token's <b>context vector</b> is the sum of all value vectors, weighted by its attention. This is how information moves between tokens: “${tokLabel(w.tokens[i])}” now carries a blend of what came before it.</p>
        <div class="terms">${terms}<div class="term total">= <span class="vec"><b>[${w.C[i].map((v) => v.toFixed(2)).join(", ")}]</b></span><span class="muted small">context vector of “${tokLabel(w.tokens[i])}”</span></div></div>
        <div class="mats">${matrix(w.C, { rows: w.tokens.map(tokLabel), title: "all context vectors = weights × V", heat: true, focus: i, clickable: true })}</div>`;
    },
  },
  {
    title: "8. Many heads, then the feed-forward network",
    body: (w, c) => `<p>That was <b>one head</b>. Your model runs <b>${c.n_head} heads</b> side by side, each with its own Q, K, V, so different heads can track different things (the previous word, the subject of the sentence, punctuation…).</p>
      <div class="pipeline">
        <div class="pp">${c.n_head} heads × ${c.n_embd / c.n_head} dims</div><span>→</span>
        <div class="pp">concatenate → ${c.n_embd}</div><span>→</span>
        <div class="pp">output projection ${c.n_embd} × ${c.n_embd}</div><span>→</span>
        <div class="pp acc">+ x (residual)</div>
      </div>
      <p>Then each token goes through the <b>feed-forward network</b> on its own: it widens to ${4 * c.n_embd} numbers, applies the GELU curve (keeping positive signals, damping negative ones), and narrows back.</p>
      <div class="pipeline">
        <div class="pp">LayerNorm</div><span>→</span><div class="pp">Linear ${c.n_embd} → ${4 * c.n_embd}</div><span>→</span>
        <div class="pp gelu">${geluSvg()} GELU</div><span>→</span><div class="pp">Linear ${4 * c.n_embd} → ${c.n_embd}</div><span>→</span><div class="pp acc">+ residual</div>
      </div>
      <p class="muted small">Attention moves information <i>between</i> tokens; the feed-forward network processes it <i>within</i> each token. This pair is one transformer block, and it repeats <b>${c.n_layer} times</b>.</p>`,
  },
  {
    title: "9. Predicting the next token",
    body: (w, c) => {
      const r = ui.real;
      const last = r ? r.tokens.map((t) => t.text).join("") : "";
      if (!r) return `<p>After the last block, the final vector of the <b>last</b> token is compared with every row of the token embedding table, giving ${fmt(c.vocab_size)} scores. Softmax turns them into probabilities. Build or train a model to see real predictions here.</p>`;
      const bars = r.next.map((n) => `<div class="wbar pick"><span class="wl">${esc(n.text).replace(/ /g, "·").replace(/\n/g, "↵")}</span><div class="wtrack"><div style="width:${(n.prob / r.next[0].prob) * 100}%"></div></div><span class="wv">${(n.prob * 100).toFixed(1)}%</span><button class="btn small ghost" data-append="${esc(n.text)}">Add</button></div>`).join("");
      return `<p>After the last block, the final vector of the <b>last</b> token (“${tokLabel(r.tokens[r.tokens.length - 1])}”) goes through LayerNorm and is compared with every row of the token embedding table: ${fmt(c.vocab_size)} scores (logits). Softmax turns them into probabilities. These are your ${r.source === "trained" ? "trained" : "untrained"} model's real top 10 for <code>${esc(last)}</code>:</p>
        <div class="wbars wide">${bars}</div>
        <p class="muted small">Greedy decoding would take the top one; sampling picks at random in proportion to these probabilities (shaped by temperature and top-k). Click <b>Add</b> to append a token and run the walkthrough again: this loop is how text is generated.</p>`;
    },
  },
];

function geluSvg() {
  const pts = []; for (let x = -3; x <= 3; x += 0.25) { const g = 0.5 * x * (1 + Math.tanh(0.7978845608 * (x + 0.044715 * x ** 3))); pts.push(`${(x + 3) * 8},${24 - (g + 0.5) * 6}`); }
  return `<svg viewBox="0 0 48 28" width="40" height="24"><polyline points="${pts.join(" ")}" fill="none" stroke="currentColor" stroke-width="2"/></svg>`;
}

function renderWalk() {
  const w = ui.walk;
  if (!w) return;
  const c = ui.realConfig || ui.status?.model?.config || ui.status?.selection?.checkpoint?.model || { n_embd: 384, n_head: 6, n_layer: 6, vocab_size: VOCAB };
  $("#a-dots").innerHTML = STAGES.map((st, i) => `<button class="sdot ${i === w.stage ? "on" : i < w.stage ? "past" : ""}" data-stage="${i}" title="${esc(st.title)}">${i + 1}</button>`).join("");
  $("#a-stage").innerHTML = `<h3 class="stage-title">${STAGES[w.stage].title}</h3>${STAGES[w.stage].body(w, c)}`;
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
    ui.realConfig = { ...(ui.status?.selection?.checkpoint?.model || ui.status?.model?.config || {}), n_head: ui.real.n_head, n_layer: ui.real.n_layer };
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

function renderTrain(s) {
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
  const c = s.selection?.checkpoint, el = $("#g-model");
  $("#g-params-wrap").classList.toggle("hidden", !c?.params);
  if (!c) { el.className = "note warn"; el.textContent = "No trained model for this training data yet. Train one in the Train step."; return; }
  const key = JSON.stringify([s.selection.path, c.model, c.trained_at]);
  if (c.params && ui.genParamsKey !== key) {
    ui.genParamsKey = key;
    $("#g-params-total").textContent = `${short(c.params.total)} (${fmt(c.params.total)})`;
    $("#g-params").innerHTML = renderParams(c.params);
    $$(".ctx-size").forEach((e) => { e.textContent = c.model.block_size; });
  }
  el.className = "note";
  const m = c.model, names = (c.datasets || s.datasets.map((d) => d.name)).map((n) => (n.startsWith("http") ? shortUrl(n) : n));
  el.innerHTML = `Using the model trained on <b>${esc(names.join(" + "))}</b>: ${m.n_layer} layers, ${m.n_head} heads, ${m.n_embd}-dim, ` +
    `${fmt(c.steps ?? 0)} steps${c.updates ? ` (${fmt(c.updates)} updates)` : ""}${c.stopped_early ? ", stopped early" : ""}, best validation loss <b>${fmtLoss(c.best_val)}</b>` +
    (c.trained_at ? ` · ${esc(c.trained_at)}` : "");
}

function syncSliders() {
  $("#g-temp-v").textContent = Number($("#g-temp").value).toFixed(1);
  $("#g-topk-v").textContent = $("#g-topk").value === "0" ? "off" : $("#g-topk").value;
  $("#g-len-v").textContent = `${$("#g-len").value} tokens`;
}

const confClass = (p) => (p >= 0.4 ? "hi" : p >= 0.1 ? "mid" : "lo");

function renderGenMetrics(m, live = false) {
  $("#g-metrics").innerHTML = [
    stat("Prompt tokens", fmt(m.prompt_tokens), m.prompt_truncated ? "longer than the context window" : "input"),
    stat("Generated tokens", fmt(m.output_tokens), live ? "so far" : "output"),
    stat("Context used", `${fmt(m.context_used ?? Math.min(m.prompt_tokens + m.output_tokens, m.context_window))} / ${fmt(m.context_window)}`, "tokens the model could see at the end"),
    stat("Time to first token", m.ttft_ms != null ? `${Math.round(m.ttft_ms)} ms` : "–", "prefill + first pick"),
    stat("Total time", m.total_ms != null ? `${(m.total_ms / 1000).toFixed(2)} s` : "–", live ? "running…" : "for all tokens"),
    stat("Speed", m.tokens_per_sec != null ? `${m.tokens_per_sec} tok/s` : "–", m.ms_per_token != null ? `${m.ms_per_token} ms per token after the first` : ""),
    stat("Average confidence", m.avg_prob != null ? `${Math.round(m.avg_prob * 100)}%` : "–", "model's probability for each chosen token"),
  ].join("");
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
  const body = { prompt: $("#g-prompt").value, temperature: +$("#g-temp").value, top_k: +$("#g-topk").value, max_new_tokens: +$("#g-len").value };
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
        span.title = `picked ${JSON.stringify(ev.text)} (${(ev.prob * 100).toFixed(1)}%)\nmodel's top choices:\n${alts}`;
        // undertrained models emit long runs of blank lines: show at most one empty line in a row
        let shown = ev.text;
        if (/^\n+$/.test(shown)) { shown = "\n".repeat(Math.max(0, Math.min(shown.length, 2 - nlRun))); nlRun += ev.text.length; }
        else { const m = shown.match(/\n+$/); nlRun = m ? m[0].length : 0; }
        span.textContent = shown;
        out.appendChild(span);
        if (count === 1 || count % 5 === 0) renderGenMetrics({ ...meta, output_tokens: count, ttft_ms: count === 1 ? ev.ms : undefined, total_ms: ev.ms, avg_prob: probSum / count }, true);
      } else if (ev.type === "done") {
        renderGenMetrics({ ...meta, ...ev });
        drawLatency(ev.token_ms);
        ui.lastGen = out.innerHTML;
        ui.generated = true;
      }
    }
  }
  render();
}

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
  if (ui.status?.selection?.checkpoint && !confirm("This replaces the model already trained on this training data. Continue?")) return;
  action(e.target, async () => { await api("/api/train", trainValues()); await poll(); });
});
$("#t-stop").addEventListener("click", (e) => action(e.target, () => api("/api/train/stop", {})));

$$("#g-temp, #g-topk, #g-len").forEach((i) => i.addEventListener("input", syncSliders));
$("#g-color").addEventListener("change", (e) => $("#g-text").classList.toggle("plain", !e.target.checked));
$("#g-run").addEventListener("click", (e) => action(e.target, generate));

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

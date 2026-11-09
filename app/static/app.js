// TinyStories LM Studio: front end for app/server.py
// Polls /api/status once a second and renders each step from it.

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (n) => Number(n).toLocaleString();
const fmtLoss = (x) => (x == null ? "–" : Number(x).toFixed(3));

const STEPS = [
  { id: "dataset", title: "Dataset", sub: "Download once, keep locally" },
  { id: "tokenize", title: "Tokenize", sub: "Text → numbers" },
  { id: "batches", title: "Batches", sub: "Training windows" },
  { id: "model", title: "Model", sub: "Build the transformer" },
  { id: "train", title: "Train", sub: "Watch it learn" },
  { id: "generate", title: "Generate", sub: "Write stories" },
];

const TRAIN_PRESETS = {
  quick: { max_iters: 2000, learning_rate: 1e-4, min_lr: 1e-5, warmup_steps: 200, eval_interval: 250, eval_iters: 50, gradient_accumulation_steps: 8 },
  full: { max_iters: 20000, learning_rate: 1e-4, min_lr: 1e-5, warmup_steps: 1000, eval_interval: 500, eval_iters: 500, gradient_accumulation_steps: 32 },
};
const TRAIN_FIELDS = Object.keys(TRAIN_PRESETS.quick);

const ui = {
  step: "dataset",
  status: null,
  prevJob: null,
  filled: false,
  batchSampled: false,
  generated: false,
  sample: { split: "train", index: 0, loadedFor: null },
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
  const s = ui.status, d = s?.dataset;
  switch (id) {
    case "dataset": return !!d;
    case "tokenize": return !!d?.tokenized;
    case "batches": return ui.batchSampled || !!s?.model;
    case "model": return !!s?.model;
    case "train": return !!d?.checkpoint;
    case "generate": return ui.generated;
  }
}

function stepBlocker(id) {
  const s = ui.status, d = s?.dataset;
  if (id === "dataset") return null;
  if (!d) return ["Load a dataset first.", "dataset"];
  if (id === "tokenize") return null;
  if (!d.tokenized) return ["Tokenize the dataset first.", "tokenize"];
  if (id === "batches" || id === "model") return null;
  if (id === "train") return s.model || d.checkpoint ? null : ["Build a model first.", "model"];
  if (id === "generate") return d.checkpoint ? null : ["Train a model first.", "train"];
  return null;
}

function go(id) {
  ui.step = id;
  try { localStorage.setItem("lmstudio.step", id); } catch (e) { /* ignore */ }
  history.replaceState(null, "", `#${id}`);
  $$(".panel").forEach((p) => p.classList.toggle("active", p.dataset.step === id));
  window.scrollTo({ top: 0, behavior: "smooth" });
  render();
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
  renderDataset(s);
  renderTokenize(s);
  renderModelHint(s);
  renderTrain(s);
  renderGenerate(s);
}

function renderTopbar(s) {
  const names = { mps: "Apple GPU · Metal", cuda: "NVIDIA GPU · CUDA", cpu: "CPU only" };
  $("#device-pill").innerHTML = `<span class="dot" style="background:${s.device === "cpu" ? "var(--warn)" : "var(--good)"}"></span>${esc(s.chip)} · ${names[s.device] || s.device}`;
  $("#device-pill").title = `PyTorch ${s.torch} · ${s.dtype}`;
  const j = s.job, pill = $("#job-pill");
  if (j.status === "running") {
    const pct = j.progress != null ? ` ${Math.round(j.progress * 100)}%` : "";
    pill.className = "pill running";
    pill.innerHTML = `<span class="dot"></span>${esc(cap(j.kind))}${pct}`;
  } else pill.className = "pill hidden";
}

const cap = (x) => (x ? x[0].toUpperCase() + x.slice(1) : "");

function renderSidebar(s) {
  const busyKind = s.job.status === "running" ? s.job.kind : null;
  const busyStep = { dataset: "dataset", tokenize: "tokenize", train: "train" }[busyKind];
  $("#steps").innerHTML = STEPS.map((st, i) => {
    const cls = ["step", st.id === ui.step && "active", stepDone(st.id) && "done", busyStep === st.id && "busy"].filter(Boolean).join(" ");
    const mark = stepDone(st.id) && busyStep !== st.id ? "✓" : i + 1;
    return `<li><button class="${cls}" data-go="${st.id}"><span class="num">${mark}</span><span><div class="t">${st.title}</div><div class="s">${st.sub}</div></span></button></li>`;
  }).join("");
}

function renderBlocker() {
  const b = stepBlocker(ui.step), el = $("#blocked");
  if (!b) { el.classList.add("hidden"); return; }
  const target = STEPS.find((x) => x.id === b[1]);
  el.innerHTML = `<span>${esc(b[0])}</span><button class="btn small" data-go="${b[1]}">Go to ${target.title}</button>`;
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
  const d = s.dataset;
  $("#ds-name").value = d?.name || s.default_dataset;
  $("#ds-config").value = d?.config_name || "";
  $("#ds-column").value = d?.text_column || "text";
  $("#b-block").value = s.batch.block_size;
  $("#b-batch").value = s.batch.batch_size;
  $("#b-seed").value = s.batch.seed;
  const m = s.model?.config || { n_layer: 6, n_head: 6, n_embd: 384, dropout: 0.1 };
  $("#m-layer").value = m.n_layer; $("#m-head").value = m.n_head; $("#m-embd").value = m.n_embd; $("#m-drop").value = m.dropout;
  setTrainPreset(s.train?.config ? matchPreset(s.train.config) : "quick", s.train?.config);
  ui.filled = true;
}

// --- dataset ---
function renderDataset(s) {
  const d = s.dataset;
  $("#ds-info").classList.toggle("hidden", !d);
  if (!d) return;
  const key = d.path;
  $("#ds-stats").innerHTML = [
    stat("Training documents", fmt(d.splits.train)),
    stat("Validation documents", fmt(d.splits.validation)),
    stat("Stored at", `<span class="mono">${esc(d.path)}</span>`, d.was_local ? "Loaded from local copy" : "Downloaded this session", true),
    d.tokenized ? stat("Training tokens", fmt(d.tokens.train), `${fmt(d.tokens.validation)} validation`) : stat("Tokens", "Not yet", "Tokenize in step 2", true),
  ].join("");
  if (ui.sample.loadedFor !== key) { ui.sample = { split: "train", index: 0, loadedFor: key }; loadSample(); }
}

function stat(k, v, d = "", small = false) {
  return `<div class="stat"><div class="k">${k}</div><div class="v ${small ? "sm" : ""}">${v}</div>${d ? `<div class="d">${d}</div>` : ""}</div>`;
}

async function loadSample(index = ui.sample.index) {
  try {
    const r = await api(`/api/dataset/sample?split=${ui.sample.split}&index=${index}`);
    ui.sample.index = r.index;
    $("#ds-text").textContent = r.text;
    $("#ds-pos").textContent = `${fmt(r.index + 1)} / ${fmt(r.total)}`;
    $$("#ds-split button").forEach((b) => b.classList.toggle("on", b.dataset.split === r.split));
  } catch (e) { toast(e.message, true); }
}

// --- tokenize ---
function tokenChip(t, colorKey) {
  const shown = esc(t.text).replace(/ /g, '<span class="sp">·</span>').replace(/\n/g, '<span class="nl">↵</span>');
  return `<span class="tok c${colorKey % 6}" title="token id ${t.id}">${shown || "&nbsp;"}</span>`;
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
  const d = s.dataset;
  const st = $("#tok-status"), btn = $("#tok-run");
  if (!d) { st.textContent = "Load a dataset in step 1 first."; btn.classList.add("hidden"); return; }
  if (d.tokenized) {
    st.innerHTML = `Done. <b>${fmt(d.tokens.train)}</b> training tokens and <b>${fmt(d.tokens.validation)}</b> validation tokens are saved in <code>${esc(d.path)}/train.bin</code> and <code>validation.bin</code>. This step is skipped from now on.`;
    btn.classList.add("hidden");
  } else {
    st.innerHTML = `Every document in <b>${esc(d.name)}</b> is converted to token IDs and written to two files. This is done once per dataset and can take a while for large ones.`;
    btn.classList.remove("hidden");
  }
}

// --- batches ---
function renderBatch(r) {
  const pos = r.input.map((_, i) => `<td>${i}</td>`).join("");
  const row = (label, toks) => `<tr><th>${label}</th>${toks.map((t) => `<td>${tokenChip(t, t.id)}</td>`).join("")}</tr>`;
  $("#b-table").innerHTML = `<tr class="pos"><th></th>${pos}</tr>` + row("Input", r.input) + row("Target", r.target);
  $("#b-shape").textContent = `${r.shape[0]} × ${r.shape[1]}`;
  $("#b-result").classList.remove("hidden");
}

// --- model ---
function renderModelHint(s) {
  const embd = +$("#m-embd").value, head = +$("#m-head").value;
  const ok = embd && head && embd % head === 0;
  $("#m-hint").innerHTML = `Context window: <b>${s.batch.block_size}</b> tokens (set in step 3). ` +
    (ok ? `Each head works on ${embd / head} dimensions.` : `<span style="color:var(--bad)">Embedding size must be divisible by the number of heads.</span>`);
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
    `A LayerNorm learns a scale and a shift for each of the <code>n_embd</code> features. The MLP widens to <code>4 × n_embd</code> and back.</p></details>`;
}

const paramsShort = (n) => (n >= 1e9 ? `${(n / 1e9).toFixed(2)}B` : n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : `${(n / 1e3).toFixed(0)}K`);

function renderModel(r) {
  const c = r.config;
  $("#m-params-total").textContent = `${fmt(r.breakdown.total)} parameters`;
  $("#m-params").innerHTML = renderParams(r.breakdown);
  $("#m-stats").innerHTML = [
    stat("Parameters", paramsShort(r.params), `${fmt(r.params)} exactly`),
    stat("Loss before training", fmtLoss(r.initial_loss), "on one batch"),
    stat("Random guessing", fmtLoss(r.random_loss), `ln(${fmt(c.vocab_size)})`),
  ].join("");
  const shown = Math.min(c.n_layer, 3);
  let blocks = "";
  for (let i = 1; i <= shown; i++) blocks += `<div class="layer block"><b>Block ${i}</b><span>${c.n_head}-head attention + MLP</span></div>`;
  if (c.n_layer > shown) blocks += `<div class="more">… ${c.n_layer - shown} more block${c.n_layer - shown > 1 ? "s" : ""} …</div>`;
  $("#m-arch").innerHTML =
    `<div class="layer"><b>Token + position embeddings</b><span>${fmt(c.vocab_size)} × ${c.n_embd}</span></div>${blocks}` +
    `<div class="layer"><b>LayerNorm → next-token scores</b><span>${c.n_embd} → ${fmt(c.vocab_size)}</span></div>`;
  $("#m-sample").textContent = r.sample;
  $("#m-result").classList.remove("hidden");
}

// --- train ---
function setTrainPreset(name, values) {
  const v = values || TRAIN_PRESETS[name];
  if (v) TRAIN_FIELDS.forEach((f) => { $(`#t-${f}`).value = v[f]; });
  highlightPreset(name);
}

function highlightPreset(name) {
  $$("#t-preset button").forEach((b) => b.classList.toggle("on", b.dataset.preset === name));
  updateTrainHint();
}

function matchPreset(cfg) {
  return Object.keys(TRAIN_PRESETS).find((p) => TRAIN_FIELDS.every((f) => Number(TRAIN_PRESETS[p][f]) === Number(cfg[f]))) || "custom";
}

function trainValues() {
  const v = {};
  TRAIN_FIELDS.forEach((f) => { v[f] = Number($(`#t-${f}`).value); });
  return v;
}

function updateTrainHint() {
  const v = trainValues(), bs = ui.status?.batch.batch_size ?? 32, bl = ui.status?.batch.block_size ?? 128;
  $("#t-hint").innerHTML = `Each optimizer update averages <b>${v.gradient_accumulation_steps}</b> batches of <b>${bs}</b> windows ` +
    `(${fmt(bs * v.gradient_accumulation_steps * bl)} tokens). The learning rate warms up for ${fmt(v.warmup_steps)} steps, then decays to ${v.min_lr}.`;
}

function renderTrain(s) {
  const tr = s.train, d = s.dataset;
  $("#t-live").classList.toggle("hidden", !tr);
  if (!tr) return;
  const running = s.job.kind === "train" && s.job.status === "running";
  const eta = running && tr.it_per_sec ? (tr.max_iters - tr.step) / tr.it_per_sec : null;
  const last = tr.evals[tr.evals.length - 1];
  $("#t-stats").innerHTML = [
    stat("Step", `${fmt(tr.step)}`, `of ${fmt(tr.max_iters)}`),
    stat("Speed", tr.it_per_sec ? `${tr.it_per_sec.toFixed(1)}` : "–", "steps / second"),
    stat(running ? "Time left" : "Status", running ? (eta != null ? duration(eta) : "–") : cap(s.job.kind === "train" ? s.job.status : "done"), running ? "estimated" : ""),
    stat("Validation loss", fmtLoss(last?.val), tr.best_val != null ? `best ${fmtLoss(tr.best_val)}` : "after first eval"),
    stat("Learning rate", tr.lr != null ? Number(tr.lr).toExponential(1) : "–"),
  ].join("");
  drawChart(tr);
}

function duration(sec) {
  sec = Math.max(0, Math.round(sec));
  const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
  return h ? `${h}h ${m}m` : m ? `${m}m ${s}s` : `${s}s`;
}

function drawChart(tr) {
  const W = 760, H = 300, L = 46, R = 14, T = 12, B = 30;
  const pts = tr.points, ev = tr.evals;
  if (!pts.length && !ev.length) { $("#t-chart").innerHTML = '<p class="muted small">Waiting for the first steps…</p>'; return; }
  // Smooth the noisy per-batch loss with an exponential moving average
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

// --- generate ---
function renderGenerate(s) {
  const c = s.dataset?.checkpoint, el = $("#g-model");
  $("#g-params-wrap").classList.toggle("hidden", !c?.params);
  if (!c) { el.className = "note warn"; el.textContent = "No trained model for this dataset yet. Train one in step 5."; return; }
  const key = JSON.stringify([s.dataset.path, c.model, c.trained_at]);
  if (c.params && ui.genParamsKey !== key) {  // only re-render when the model changes, so an open <details> stays open
    ui.genParamsKey = key;
    $("#g-params-total").textContent = `${paramsShort(c.params.total)} (${fmt(c.params.total)})`;
    $("#g-params").innerHTML = renderParams(c.params);
  }
  el.className = "note";
  const m = c.model;
  el.innerHTML = `Using the model trained on <b>${esc(s.dataset.name)}</b>: ${m.n_layer} layers, ${m.n_embd}-dim, ` +
    `${fmt(c.steps ?? 0)} steps${c.stopped_early ? " (stopped early)" : ""}, best validation loss <b>${fmtLoss(c.best_val)}</b>` +
    (c.trained_at ? ` · ${esc(c.trained_at)}` : "");
}

function syncSliders() {
  $("#g-temp-v").textContent = Number($("#g-temp").value).toFixed(1);
  $("#g-topk-v").textContent = $("#g-topk").value === "0" ? "off" : $("#g-topk").value;
  $("#g-len-v").textContent = `${$("#g-len").value} tokens`;
}

// ---------- events ----------
document.addEventListener("click", (e) => {
  const g = e.target.closest("[data-go]");
  if (g) go(g.dataset.go);
});

// --- dataset check + gallery ---
const dsBody = () => ({ name: $("#ds-name").value.trim(), config_name: $("#ds-config").value.trim() || null, text_column: $("#ds-column").value.trim() || "text" });
const bytes = (n) => (n >= 1e9 ? `${(n / 1e9).toFixed(1)} GB` : n >= 1e6 ? `${(n / 1e6).toFixed(1)} MB` : `${Math.max(1, Math.round(n / 1e3))} KB`);

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
  el.innerHTML = `<div class="title">✓ Text dataset${r.kind === "url" ? " (file link)" : ""}: ready to load</div>` +
    `<div class="row"><span>Columns:</span>${cols}</div>` +
    `<div class="row">${r.download_size ? `<span>Download <b>${bytes(r.download_size)}</b></span>` : ""}${rows ? `<span>${rows}</span>` : ""}` +
    `<span>${r.downloaded ? "Already downloaded" : `Will be saved to <code>${esc(r.path)}</code>`}</span></div>`;
  return r;
}

const shortUrl = (u) => { try { const x = new URL(u); return `${x.hostname} / ${x.pathname.split("/").pop()}`; } catch (e) { return u; } };

let catalog = [], galleryTag = "All", galleryKey = null;
async function loadCatalog() {
  try { catalog = await api("/api/catalog"); galleryKey = null; renderGallery(); } catch (e) { /* server not ready */ }
}

function renderGallery() {
  const cur = ui.status?.dataset;
  const key = JSON.stringify([galleryTag, cur?.path, catalog.length]);
  if (key === galleryKey) return;
  galleryKey = key;
  const tags = ["All", ...new Set(catalog.flatMap((c) => c.tags))];
  $("#gallery-tags").innerHTML = tags.map((t) => `<button class="chip-btn ${t === galleryTag ? "on" : ""}" data-tag="${esc(t)}">${esc(t)}</button>`).join("");
  $("#gallery-grid").innerHTML = catalog.map((c, i) => {
    if (galleryTag !== "All" && !c.tags.includes(galleryTag)) return "";
    const id = c.dataset.startsWith("http") ? shortUrl(c.dataset) : c.dataset + (c.config ? ` · ${c.config}` : "");
    const status = c.trained ? "Trained" : c.tokenized ? "Tokenized" : c.downloaded ? "Downloaded" : "";
    const isCur = cur && cur.name === c.dataset && (cur.config_name || null) === (c.config || null);
    return `<div class="ds-card ${isCur ? "current" : ""}">
      <div class="top"><div><h4>${esc(c.title)}</h4><div class="id">${esc(id)}</div></div>${isCur ? '<span class="tag status">In use</span>' : status ? `<span class="tag status">${status}</span>` : ""}</div>
      <div class="tags">${c.tags.map((t) => `<span class="tag">${esc(t)}</span>`).join("")}</div>
      <p>${esc(c.description)}</p>
      <div class="meta"><span><b>${esc(c.rows)}</b></span><span><b>${esc(c.size)}</b></span>${c.text_column !== "text" ? `<span>column <b>${esc(c.text_column)}</b></span>` : ""}</div>
      <div class="btns"><button class="btn primary" data-use="${i}">Use</button><a class="btn" href="${esc(c.url)}" target="_blank" rel="noopener">View ↗</a></div>
    </div>`;
  }).join("");
}

document.addEventListener("click", (e) => {
  const use = e.target.closest("[data-use]");
  if (use) {
    const c = catalog[+use.dataset.use];
    $("#ds-name").value = c.dataset; $("#ds-config").value = c.config || ""; $("#ds-column").value = c.text_column;
    window.scrollTo({ top: 0, behavior: "smooth" });
    action(null, checkDataset);
  }
  const tag = e.target.closest("[data-tag]");
  if (tag) { galleryTag = tag.dataset.tag; renderGallery(); }
  const cfg = e.target.closest("[data-config]");
  if (cfg) { $("#ds-config").value = cfg.dataset.config; action(null, checkDataset); }
});

$$("#ds-name, #ds-config, #ds-column").forEach((i) => i.addEventListener("input", () => $("#ds-check-result").className = "check hidden"));
$("#ds-check").addEventListener("click", (e) => action(e.target, checkDataset));
$("#ds-load").addEventListener("click", (e) => action(e.target, async () => {
  await api("/api/dataset", dsBody());
  await poll();
}));
$$("#ds-split button").forEach((b) => b.addEventListener("click", () => { ui.sample.split = b.dataset.split; loadSample(0); }));
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

$$("#m-layer, #m-head, #m-embd").forEach((i) => i.addEventListener("input", () => ui.status && renderModelHint(ui.status)));
$("#m-run").addEventListener("click", (e) => action(e.target, async () => {
  const r = await api("/api/model", { n_layer: +$("#m-layer").value, n_head: +$("#m-head").value, n_embd: +$("#m-embd").value, dropout: +$("#m-drop").value });
  renderModel(r);
  await poll();
}));

$$("#t-preset button").forEach((b) => b.addEventListener("click", () => {
  if (b.dataset.preset === "custom") highlightPreset("custom");
  else setTrainPreset(b.dataset.preset);
}));
TRAIN_FIELDS.forEach((f) => $(`#t-${f}`).addEventListener("input", () => highlightPreset(matchPreset(trainValues()))));
$("#t-start").addEventListener("click", (e) => {
  if (ui.status?.dataset?.checkpoint && !confirm("This replaces the model already trained on this dataset. Continue?")) return;
  action(e.target, async () => { await api("/api/train", trainValues()); await poll(); });
});
$("#t-stop").addEventListener("click", (e) => action(e.target, () => api("/api/train/stop", {})));

$$("#g-temp, #g-topk, #g-len").forEach((i) => i.addEventListener("input", syncSliders));
$("#g-run").addEventListener("click", (e) => action(e.target, async () => {
  $("#g-busy").classList.remove("hidden");
  try {
    const r = await api("/api/generate", {
      prompt: $("#g-prompt").value, temperature: +$("#g-temp").value, top_k: +$("#g-topk").value, max_new_tokens: +$("#g-len").value,
    });
    ui.generated = true;
    const item = document.createElement("div");
    item.className = "gen-item";
    const completion = r.completion.replace(/\n{3,}/g, "\n\n");  // undertrained models emit long runs of blank lines
    item.innerHTML = `<div class="doc"><span class="prompt">${esc(r.prompt)}</span>${esc(completion)}</div>` +
      `<div class="gen-meta"><span>temperature ${$("#g-temp").value} · top-k ${$("#g-topk").value === "0" ? "off" : $("#g-topk").value}</span><span>${r.seconds}s</span></div>`;
    $("#g-out").prepend(item);
    while ($("#g-out").children.length > 6) $("#g-out").lastChild.remove();
    render();
  } finally { $("#g-busy").classList.add("hidden"); }
}));

// ---------- polling ----------
async function poll() {
  try {
    ui.status = await api("/api/status");
  } catch (e) { return; }
  const j = ui.status.job, prev = ui.prevJob;
  if (prev && prev.status === "running" && j.status !== "running") {
    if (j.status === "error") toast(j.error || "Something went wrong.", true);
    else if (j.kind === "dataset") toast("Dataset ready.");
    else if (j.kind === "tokenize") toast("Dataset tokenized.");
    else if (j.kind === "train") toast(j.message);
    loadCatalog();  // refresh Downloaded / Tokenized / Trained badges
  }
  ui.prevJob = j;
  render();
  renderGallery();
}

const hashStep = location.hash.slice(1);
if (STEPS.some((s) => s.id === hashStep)) ui.step = hashStep;  // e.g. http://127.0.0.1:8000/#train
window.addEventListener("hashchange", () => {
  const id = location.hash.slice(1);
  if (STEPS.some((s) => s.id === id) && id !== ui.step) go(id);
});
go(STEPS.some((s) => s.id === ui.step) ? ui.step : "dataset");
syncSliders();
previewTokens();
loadCatalog();
poll();
setInterval(poll, 1000);

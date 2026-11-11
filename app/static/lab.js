// Lab: pick an architecture (a known design or every switch), one or more datasets, train, and compare the
// result with every model trained so far. Uses the helpers in app.js ($, api, esc, fmt, short, bytes, toast…).

const LAB_GPT2 = { n_layer: 6, n_head: 6, n_embd: 384, dropout: 0, pos_encoding: "learned", norm: "layernorm", mlp: "gelu",
  bias: true, qk_norm: false, n_kv_head: 0, block_repeat: 1, mtp_heads: 1, n_experts: 1, moe_top_k: 1 };
const LAB_BEST = { ...LAB_GPT2, pos_encoding: "rope", mlp: "swiglu", bias: false, qk_norm: true };

const LAB_PRESETS = [
  { id: "gpt2", title: "GPT-2 style", tag: "2019", lr: 6e-4, arch: { ...LAB_GPT2 },
    note: "Learned positions, LayerNorm, GELU, biases. Where the architecture work started.",
    measured: [["1,000 updates", "2.631"], ["5,000 updates", "1.903"]] },
  { id: "llama", title: "Llama style", tag: "2023", lr: 1.8e-3, arch: { ...LAB_GPT2, pos_encoding: "rope", norm: "rmsnorm", mlp: "swiglu", bias: false, n_kv_head: 2 },
    note: "RoPE, RMSNorm, SwiGLU, no biases, grouped-query attention. Each part was tested on its own here; never as a whole.",
    measured: [["as a whole", "not trained yet"]] },
  { id: "best", title: "Our best, dense", tag: "this project", lr: 1.8e-3, arch: { ...LAB_BEST },
    note: "RoPE, SwiGLU, no biases, QK-norm, learning rate 1.8e-3: every change that won its A/B test.",
    measured: [["1,000 updates", "2.251"], ["5,000 updates", "1.782"]] },
  { id: "moe", title: "Our best + experts", tag: "best so far", best: true, lr: 1.8e-3, arch: { ...LAB_BEST, n_experts: 4, moe_top_k: 2 },
    note: "The dense best with 4 feed-forward experts per block, 2 per token: 51M parameters, 37M used per token.",
    measured: [["1,000 updates", "2.200"], ["5,000 updates", "1.732"]] },
  { id: "fast", title: "Fast and lean", tag: "for speed", lr: 1.8e-3, arch: { ...LAB_BEST, n_layer: 3, n_head: 7, n_embd: 448 },
    note: "3 wide layers instead of 6: generates 1.6× faster with a 42% smaller KV cache, for 0.02 more loss.",
    measured: [["1,000 updates", "2.271"], ["generation", "215 tokens/s"]] },
];

const LAB_SWITCHES = [
  { group: "Shape", note: "At ~30M parameters and 1,000 updates, 6 × 384 beat 3 × 448, 11 × 320 and 22 × 256 (A7). Keep the head size (embedding ÷ heads) at 64.",
    items: [
      { key: "n_layer", label: "Layers", type: "number", min: 1, max: 24 },
      { key: "n_head", label: "Attention heads", type: "number", min: 1, max: 16 },
      { key: "n_embd", label: "Embedding size", type: "number", min: 64, max: 1024, step: 32 },
    ] },
  { group: "Attention", items: [
      { key: "pos_encoding", label: "Positions", options: [["learned", "Learned (GPT-2)"], ["rope", "RoPE, rotary (Llama)"]], measured: "RoPE: 2.631 → 2.475, and a KV cache that can slide (A1)" },
      { key: "qk_norm", label: "QK-norm", options: [[false, "Off"], [true, "On"]], measured: "2.369 → 2.336; at a 7× higher learning rate, 2.55 instead of 3.33 (A5, sweep)" },
      { key: "n_kv_head", label: "Key/value heads", options: "kv", measured: "Half as many: KV cache halved, loss unchanged (A6)" },
    ] },
  { group: "Each block", items: [
      { key: "norm", label: "Normalization", options: [["layernorm", "LayerNorm (GPT-2)"], ["rmsnorm", "RMSNorm (Llama)"]], measured: "A tie: 2.475 vs 2.483 (A2)" },
      { key: "mlp", label: "Feed-forward", options: [["gelu", "GELU, 4× wide (GPT-2)"], ["swiglu", "SwiGLU, gated (Llama)"]], measured: "SwiGLU: 2.475 → 2.366 at equal size (A3)" },
      { key: "bias", label: "Biases", options: [[true, "Yes (GPT-2)"], [false, "No (Llama)"]], measured: "A tie, with 29k fewer parameters (A4)" },
      { key: "n_experts", label: "Experts", options: [[1, "1 (dense)"], [4, "4"], [8, "8"]], measured: "4 experts, 2 per token: 2.251 → 2.200; 1 per token was worse (A10)" },
      { key: "moe_top_k", label: "Experts per token", options: [[1, "1"], [2, "2"]], when: (a) => a.n_experts > 1 },
      { key: "block_repeat", label: "Repeat each block", options: [[1, "Once"], [2, "2× (shared)"], [3, "3× (shared)"]], measured: "Worse at this budget: 2.251 → 2.346 at 2× (A8)" },
    ] },
  { group: "Training objective", items: [
      { key: "mtp_heads", label: "Tokens predicted", options: [[1, "1, the next"], [2, "2"], [3, "3"], [4, "4"]], measured: "Worse at 30M: 2.251 → 2.336 with 2 (A9)" },
    ] },
];

const LAB_LABELS = { pos_encoding: "positions", norm: "normalization", mlp: "feed-forward", bias: "biases", qk_norm: "QK-norm",
  n_kv_head: "KV heads", block_repeat: "block sharing", mtp_heads: "tokens predicted", n_experts: "experts", moe_top_k: "experts per token",
  n_layer: "layers", n_head: "heads", n_embd: "embedding" };

const labState = { preset: "moe", arch: { ...LAB_PRESETS[3].arch }, lr: 1.8e-3, datasets: null, picked: new Set(),
  models: null, est: null, estTimer: null, info: null, opened: false, lastPhase: null, search: "" };
try { const saved = JSON.parse(localStorage.getItem("lab.picked") || "null"); if (Array.isArray(saved)) labState.picked = new Set(saved); } catch (e) { /* ignore */ }
const savePicked = () => { try { localStorage.setItem("lab.picked", JSON.stringify([...labState.picked])); } catch (e) { /* ignore */ } };

async function labOpen() {
  if (!labState.opened) {
    labState.opened = true;
    renderLabPresets();
    renderLabSwitches();
    labEstimate();
  }
  await Promise.all([labLoadDatasets(), labLoadModels()]).catch((e) => toast(e.message, true));
}

// ---------- 1. presets ----------
function renderLabPresets() {
  $("#lab-presets").innerHTML = LAB_PRESETS.map((p) => `<button class="lab-preset ${p.id === labState.preset ? "on" : ""} ${p.best ? "best" : ""}" data-lab-preset="${p.id}">
      <div class="lp-top"><b>${esc(p.title)}</b><span class="tag">${esc(p.tag)}</span></div>
      <div class="lp-note">${esc(p.note)}</div>
      <div class="lp-measured">${p.measured.map(([k, v]) => `<span><span class="muted">${esc(k)}</span> <b>${esc(v)}</b></span>`).join("")}</div>
    </button>`).join("");
}

function labApplyPreset(id) {
  const p = LAB_PRESETS.find((x) => x.id === id);
  labState.preset = id;
  labState.arch = { ...p.arch };
  labState.lr = p.lr;
  $("#lab-lr").value = String(p.lr);
  renderLabPresets();
  renderLabSwitches();
  labEstimate();
}

// ---------- 2. every switch ----------
function kvOptions(a) {
  const H = a.n_head || 1;
  return Array.from({ length: H }, (_, i) => i + 1).filter((k) => H % k === 0).reverse()
    .map((k) => [k === H ? 0 : k, k === H ? `${k} (one per head)` : k === 1 ? "1 (multi-query)" : `${k} (groups of ${H / k})`]);
}

function renderLabSwitches() {
  const a = labState.arch;
  $("#lab-switches").innerHTML = LAB_SWITCHES.map((g) => `<div class="lab-group">
      <div class="lab-group-title">${esc(g.group)}</div>
      <div class="lab-items">${g.items.filter((it) => !it.when || it.when(a)).map((it) => {
        const diff = JSON.stringify(a[it.key]) !== JSON.stringify(LAB_GPT2[it.key]);
        let input;
        if (it.type === "number") {
          input = `<input type="number" data-lab-key="${it.key}" min="${it.min}" max="${it.max}" step="${it.step || 1}" value="${a[it.key]}">`;
        } else {
          const opts = it.options === "kv" ? kvOptions(a) : it.options;
          input = `<select class="select" data-lab-key="${it.key}">${opts.map(([v, t]) => `<option value='${JSON.stringify(v)}' ${JSON.stringify(v) === JSON.stringify(a[it.key]) ? "selected" : ""}>${esc(t)}</option>`).join("")}</select>`;
        }
        return `<label class="lab-item ${diff ? "changed" : ""}"><span class="li-label">${esc(it.label)}${diff ? '<span class="li-dot" title="differs from GPT-2 style"></span>' : ""}</span>${input}${it.measured ? `<span class="li-measured">${esc(it.measured)}</span>` : ""}</label>`;
      }).join("")}</div>
      ${g.note ? `<p class="hint">${esc(g.note)}</p>` : ""}
    </div>`).join("");
}

function labSetKey(key, raw) {
  const a = labState.arch;
  a[key] = raw;
  if (key === "n_head" && a.n_kv_head && a.n_head % a.n_kv_head) a.n_kv_head = 0;
  if (key === "n_experts" && a.n_experts === 1) a.moe_top_k = 1;
  const match = LAB_PRESETS.find((p) => JSON.stringify(p.arch) === JSON.stringify(a));
  labState.preset = match ? match.id : null;
  renderLabPresets();
  renderLabSwitches();
  labEstimate();
}

function labEstimate() {
  clearTimeout(labState.estTimer);
  labState.estTimer = setTimeout(async () => {
    try {
      labState.est = await api("/api/lab/estimate", labState.arch);
      labState.estError = null;
    } catch (e) { labState.est = null; labState.estError = e.message; }
    renderLabSummary();
    renderLabCalc();
  }, 150);
}

function renderLabSummary() {
  const e = labState.est, a = labState.arch, el = $("#lab-summary");
  if (!e) { el.innerHTML = `<p class="bad-text small">${esc(labState.estError || "…")}</p>`; return; }
  const emb = e.breakdown.parts.find((p) => p.name === "Token embeddings")?.count || 0;
  const changes = Object.keys(LAB_LABELS).filter((k) => JSON.stringify(a[k]) !== JSON.stringify(LAB_GPT2[k]));
  const ctx = e.config.block_size;
  const stat = (label, value, sub = "") => `<div class="ls-stat"><div class="muted small">${label}</div><b>${value}</b>${sub ? `<div class="muted small">${sub}</div>` : ""}</div>`;
  el.innerHTML = `<div class="ls-stats">
      ${stat("Parameters", short(e.params), fmt(e.params))}
      ${stat("Used per token", short(e.active), e.active !== e.params ? `${Math.round((e.active / e.params) * 100)}% of the total` : "all of them")}
      ${stat("KV cache per token", bytes(e.kv_bytes_per_token), `${bytes(e.kv_bytes_per_token * ctx)} for a full ${ctx}-token window`)}
      ${stat("Memory to train", bytes(e.train_memory), "16 bytes per parameter + activations")}
    </div>
    <div class="ls-split"><div style="width:${(emb / e.params) * 100}%" title="token embeddings"></div></div>
    <div class="muted small">${Math.round((emb / e.params) * 100)}% of the parameters are the vocabulary embedding (${fmt(VOCAB)} × ${a.n_embd}); the rest are ${a.n_layer} transformer blocks.</div>
    <div class="ls-diff"><b>${changes.length ? `Differs from GPT-2 style in ${changes.length} way${changes.length > 1 ? "s" : ""}:` : "Same as GPT-2 style."}</b>
      ${changes.map((k) => `<span class="tag">${esc(LAB_LABELS[k])}</span>`).join(" ")}</div>`;
}

// ---------- 3. datasets ----------
async function labLoadDatasets() {
  labState.info = await api("/api/lab/datasets");
  const ds = labState.info.datasets;
  if (labState.datasets === null) {
    const def = ds.find((d) => d.key === "roneneldan__tinystories") || ds[0];
    labState.datasets = new Set(def ? [def.key] : []);
  }
  renderLabDatasets();
  renderLabCalc();
}

function renderLabDatasets() {
  const ds = labState.info?.datasets || [];
  $("#lab-data-note").textContent = ds.length ? `${ds.length} tokenized on this Mac` : "";
  if (!ds.length) { $("#lab-datasets").innerHTML = '<p class="muted">No tokenized datasets yet. Add and tokenize one in steps 1 and 2.</p>'; return; }
  const several = labState.datasets.size > 1;
  $("#lab-datasets").innerHTML = `<div class="lab-ds">${ds.map((d) => `<label class="lab-ds-item ${labState.datasets.has(d.key) ? "on" : ""}">
      <input type="checkbox" data-lab-ds="${esc(d.key)}" ${labState.datasets.has(d.key) ? "checked" : ""}>
      <span><b>${esc(d.title)}</b><span class="muted small">${short(d.train_tokens)} training tokens · ${short(d.val_tokens)} validation</span></span></label>`).join("")}</div>
    ${several ? `<label class="inline small">Mix <select id="lab-mix" class="select"><option value="size">In proportion to their size</option><option value="equal">Equally</option></select></label>` : ""}
    <p class="hint">Every model is also evaluated on the other datasets here (held-out loss), so you can see how it does on text it never saw.</p>`;
  if (several && labState.mix) $("#lab-mix").value = labState.mix;
}

// ---------- 4. train ----------
function renderLabCalc() {
  const e = labState.est, ds = (labState.info?.datasets || []).filter((d) => labState.datasets?.has(d.key));
  const updates = +$("#lab-updates").value, tokens = updates * 32 * 256;
  const dataTokens = ds.reduce((x, d) => x + d.train_tokens, 0);
  const dev = labState.info?.device;
  // measured: the 30M model took ~1.5 s per update on the M5 (micro-batches of 8) and ~0.15 s on a Colab L4
  const perUpdate = (dev === "cuda" ? 0.15 : dev === "mps" ? 1.5 : 12) * Math.max(0.3, (e?.active || 3e7) / 3e7) * (labState.arch.block_repeat || 1);
  const mins = (updates * perUpdate) / 60;
  $("#lab-cost").textContent = e ? `about ${mins < 1.5 ? "a minute" : `${Math.round(mins)} minutes`} on this ${dev === "mps" ? "Mac" : dev === "cuda" ? "GPU" : "computer"}, plus about a minute to evaluate` : "";
  $("#lab-calc").innerHTML = e ? `<div class="calc-line"><span>Tokens seen</span><code>${fmt(updates)} updates × 32 sequences × 256 tokens = <b>${short(tokens)}</b></code></div>
    ${dataTokens ? `<div class="calc-line"><span>Passes over the data</span><code>${short(tokens)} ÷ ${short(dataTokens)} = <b>${(tokens / dataTokens).toFixed(tokens / dataTokens < 0.1 ? 3 : 2)}</b></code></div>` : ""}
    <div class="calc-line"><span>Learning rate</span><code>warm up over ${fmt(Math.max(1, Math.floor(updates / 10)))} updates to <b>${(+$("#lab-lr").value).toExponential(1)}</b>, cosine decay to a tenth</code></div>
    ${dev === "mps" ? `<div class="calc-line"><span>On this Mac</span><code>each update = 4 micro-batches of 8 (same gradient as 32, a quarter of the memory)</code></div>` : ""}` : "";
}

async function labTrain(btn) {
  const ds = [...(labState.datasets || [])];
  if (!ds.length) { toast("Pick at least one dataset.", true); return; }
  const preset = LAB_PRESETS.find((p) => p.id === labState.preset);
  const name = $("#lab-name").value.trim() || `${preset ? preset.title : "Custom"} · ${(labState.info.datasets.filter((d) => labState.datasets.has(d.key)).map((d) => d.title)).join(" + ")} · ${fmt(+$("#lab-updates").value)} updates`;
  await action(btn, async () => {
    await api("/api/lab/train", { name, arch: labState.arch, datasets: ds, mix: $("#lab-mix")?.value || "size",
      updates: +$("#lab-updates").value, learning_rate: +$("#lab-lr").value });
    toast("Training started.");
  });
}

function labRender(s) {
  if (!labState.opened) return;
  const j = s.job, L = s.lab, running = j.status === "running";
  const mine = j.kind === "lab";
  $("#lab-train").disabled = running;
  $("#lab-stop").classList.toggle("hidden", !(mine && running));
  $("#lab-busy").textContent = running && !mine ? `Waiting: ${j.kind} is running.` : "";
  const live = $("#lab-live");
  if (!L) { live.classList.add("hidden"); return; }
  live.classList.remove("hidden");
  const pct = Math.round((L.update / L.updates) * 100);
  const chart = L.points.length > 1 ? xyChart([
    { label: "training loss", color: "var(--series-train)", points: L.points, smooth: 5 },
    { label: "validation loss", color: "var(--series-val)", points: L.evals.map((e) => [e.update, e.val]), dots: true },
  ], { xLabel: "update", yLabel: "loss", W: 560, H: 200 }) : "";
  const phase = { training: `Training: update ${fmt(L.update)} of ${fmt(L.updates)}`, evaluating: "Evaluating on every dataset and saving to the library…",
    done: `Done: best validation loss <b>${fmtLoss(L.best_val)}</b>. Saved to the library and ticked below.`, stopped: "Stopped. Nothing was saved." }[L.phase] || "";
  live.innerHTML = `<div class="lab-live-head"><b>${esc(L.name)}</b> <span class="muted small">on ${esc(L.datasets.join(" + "))}</span></div>
    <div class="bar"><div style="width:${L.phase === "training" ? pct : 100}%"></div></div>
    <div class="small">${phase}${L.evals.length ? ` · validation ${L.evals.map((e) => fmtLoss(e.val)).join(" → ")}` : ""}</div>
    <div class="chart">${chart}</div>`;
  if (L.phase !== labState.lastPhase) {
    if (L.phase === "done" && labState.lastPhase) {
      if (L.model_id) labState.picked.add(L.model_id);
      savePicked();
      labLoadModels();
    }
    labState.lastPhase = L.phase;
  }
}

// ---------- 5. compare ----------
async function labLoadModels() {
  labState.models = (await api("/api/models")).models;
  const ids = new Set(labState.models.map((m) => m.id));
  labState.picked = new Set([...labState.picked].filter((i) => ids.has(i)));
  if (!labState.picked.size) labPick("final");
  renderLabModels();
  renderLabCompare();
}

const tr = (m) => m.training || {};
const groupOf = (m) => (m.training?.experiment && m.training?.experiment !== "Lab" ? m.training.experiment : m.training?.experiment === "Lab" || m.platform === "lab" ? "Lab runs" : "Earlier runs");
const labelOf = (m) => (tr(m).experiment && tr(m).experiment !== "Lab" && tr(m).variant ? tr(m).variant : m.name);

function archTags(c) {
  if (!c) return [];
  return [c.pos_encoding === "rope" ? "RoPE" : "learned pos.", c.norm === "rmsnorm" ? "RMSNorm" : null, c.mlp === "swiglu" ? "SwiGLU" : "GELU",
    c.bias === false ? "no bias" : null, c.qk_norm ? "QK-norm" : null, c.n_kv_head && c.n_kv_head !== c.n_head ? `KV ${c.n_kv_head}` : null,
    (c.n_experts || 1) > 1 ? `${c.n_experts} experts top-${c.moe_top_k || 1}` : null, (c.block_repeat || 1) > 1 ? `shared ×${c.block_repeat}` : null,
    (c.mtp_heads || 1) > 1 ? `MTP ${c.mtp_heads}` : null, `${c.n_layer}×${c.n_embd}`].filter(Boolean);
}

function labGroups() {
  const q = labState.search.toLowerCase();
  const groups = new Map();
  for (const m of labState.models || []) {
    if (q && !`${m.name} ${groupOf(m)} ${archTags(m.model).join(" ")}`.toLowerCase().includes(q)) continue;
    const g = groupOf(m);
    if (!groups.has(g)) groups.set(g, []);
    groups.get(g).push(m);
  }
  // Lab runs, then the experiments newest first (A10 … A1; the sweep came between A5 and A6), then earlier runs
  const num = (g) => (g === "Lab runs" ? 1000 : g.startsWith("Final") ? 999 : g === "Earlier runs" ? -1
    : g.startsWith("LR sweep") ? 5.5 : +(g.match(/^A(\d+)/)?.[1] || 0) - (g.includes("(Mac)") ? 0.1 : 0));
  return [...groups.entries()].sort((a, b) => num(b[0]) - num(a[0]));
}

function renderLabModels() {
  const groups = labGroups();
  $("#lab-cmp-count").textContent = `${labState.picked.size} of ${labState.models.length} models selected`;
  $("#lab-models").innerHTML = groups.map(([g, ms]) => `<div class="lab-mgroup"><div class="lab-mgroup-title">${esc(g)} <span class="muted small">${ms.length}</span></div>
    ${ms.map((m) => `<label class="lab-mrow ${labState.picked.has(m.id) ? "on" : ""}" title="${esc(m.name)}"><input type="checkbox" data-lab-pick="${esc(m.id)}" ${labState.picked.has(m.id) ? "checked" : ""}>
      <span class="lab-mname">${esc(labelOf(m))}</span><span class="lab-mtags">${archTags(m.model).slice(0, 6).map((t) => `<span>${esc(t)}</span>`).join("")}</span>
      <span class="lab-mval">${fmtLoss(m.training?.best_val ?? m.best_val)}</span></label>`).join("")}</div>`).join("") ||
    '<p class="muted">No models match.</p>';
}

function labPick(which) {
  const ms = labState.models || [];
  if (which === "none") labState.picked = new Set();
  if (which === "final") labState.picked = new Set(ms.filter((m) => tr(m).experiment?.startsWith("Final")).map((m) => m.id));
  if (which === "ladder") {
    const want = [["A1 RoPE (Mac)", "learned"], ["A1 RoPE (Mac)", "rope"], ["A3 SwiGLU (Mac)", "swiglu"], ["A4 no biases (Mac)", "nobias"],
      ["A5 QK-norm", "qk-norm"], ["A10 mixture of experts", "baseline"], ["A10 mixture of experts", "moe4-top2"]];
    labState.picked = new Set(want.map(([e, v]) => ms.find((m) => tr(m).experiment === e && tr(m).variant === v)?.id).filter(Boolean));
    labState.order = want.map(([e, v]) => ms.find((m) => tr(m).experiment === e && tr(m).variant === v)?.id).filter(Boolean);
  } else labState.order = null;
  savePicked();
}

function labCompareRows(models) {
  const heldKeys = [], titles = {};
  models.forEach((m) => Object.entries(m.eval?.heldout || {}).forEach(([k, v]) => { if (!heldKeys.includes(k)) { heldKeys.push(k); titles[k] = v.title; } }));
  const c = (m) => m.model || {}, t = tr, g = (m) => m.eval?.generation || {};
  const yesno = (v) => (v ? "yes" : "no");
  const updatesOf = (m) => t(m).updates_done ?? t(m).steps ?? m.updates ?? m.steps;
  const batchOf = (m) => t(m).batch || null, blockOf = (m) => t(m).block || c(m).block_size;
  return [
    { section: "Architecture" },
    { label: "Shape", get: (m) => `${c(m).n_layer} layers × ${c(m).n_embd}`, sub: (m) => `${c(m).n_head} heads of ${c(m).n_embd / c(m).n_head}` },
    { label: "Positions", get: (m) => (c(m).pos_encoding === "rope" ? "RoPE" : "learned") },
    { label: "Normalization", get: (m) => (c(m).norm === "rmsnorm" ? "RMSNorm" : "LayerNorm") },
    { label: "Feed-forward", get: (m) => (c(m).mlp === "swiglu" ? "SwiGLU" : "GELU") },
    { label: "Biases", get: (m) => yesno(c(m).bias !== false) },
    { label: "QK-norm", get: (m) => yesno(c(m).qk_norm) },
    { label: "Key/value heads", get: (m) => `${c(m).n_kv_head || c(m).n_head} of ${c(m).n_head}` },
    { label: "Experts", get: (m) => ((c(m).n_experts || 1) > 1 ? `${c(m).n_experts}, ${c(m).moe_top_k || 1} per token` : "dense") },
    { label: "Each block applied", get: (m) => `${c(m).block_repeat || 1}×` },
    { label: "Tokens predicted (training)", get: (m) => `${c(m).mtp_heads || 1}` },
    { label: "Context window", get: (m) => `${blockOf(m)} tokens` },
    { section: "Size and cost" },
    { label: "Parameters", num: (m) => m.params, fmt: (v) => short(v), sub: (m) => fmt(m.params) },
    { label: "Used per token", num: (m) => t(m).active_params ?? m.params, fmt: (v) => short(v) },
    { label: "Weights on disk", num: (m) => m.bytes, fmt: (v) => bytes(v) },
    { label: "KV cache per token", num: (m) => m.eval?.size?.kv_bytes_per_token ?? t(m).kv_bytes_per_token, fmt: (v) => bytes(v), best: "min" },
    { label: "Training data", get: (m) => shortData(t(m).datasets || m.datasets) },
    { label: "Weight updates", num: updatesOf, fmt: (v) => fmt(v) },
    { label: "Tokens seen", num: (m) => (updatesOf(m) && batchOf(m) ? updatesOf(m) * batchOf(m) * blockOf(m) : null), fmt: (v) => short(v),
      sub: (m) => (batchOf(m) ? `${fmt(updatesOf(m))} × ${batchOf(m)} × ${blockOf(m)}` : "") },
    { label: "Peak learning rate", num: (m) => t(m).lr, fmt: (v) => Number(v).toExponential(1) },
    { label: "Hardware", get: (m) => t(m).chip || m.chip || "–" },
    { label: "Training time", num: (m) => t(m).seconds ?? m.seconds, fmt: (v) => (v >= 3600 ? `${(v / 3600).toFixed(1)} h` : `${Math.round(v / 60)} min`) },
    { section: "Quality", note: "lower loss is better; compare on the same data" },
    { label: "Validation loss (own data)", num: (m) => t(m).best_val ?? m.best_val, fmt: (v) => v.toFixed(3), best: "min" },
    ...heldKeys.map((k) => ({ label: `Held-out: ${titles[k]}`, num: (m) => m.eval?.heldout?.[k]?.loss, fmt: (v) => v.toFixed(3), best: "min" })),
    { label: "Next-token accuracy", num: (m) => t(m).head_accuracy?.[0], fmt: (v) => `${(v * 100).toFixed(1)}%`, best: "max" },
    { label: "Greedy repetition", num: (m) => g(m).greedy_repeated_4grams, fmt: (v) => `${Math.round(v * 100)}%`, best: "min" },
    { label: "Variety (Balanced)", num: (m) => g(m).balanced_distinct_2, fmt: (v) => `${Math.round(v * 100)}%`, best: "max" },
    { section: "Speed", note: "only comparable on the same hardware" },
    { label: "Training, updates/s", num: (m) => t(m).updates_per_sec, fmt: (v) => v.toFixed(2), best: "max" },
    { label: "Generation, tokens/s", num: (m) => g(m).tokens_per_sec ?? t(m).tokens_per_sec, fmt: (v) => Math.round(v), best: "max" },
    { label: "First token", num: (m) => g(m).ttft_ms, fmt: (v) => `${Math.round(v)} ms`, best: "min" },
  ];
}

function renderLabCompare() {
  let models = (labState.models || []).filter((m) => labState.picked.has(m.id));
  // columns run from the highest validation loss to the lowest: the first is the reference, the best comes last
  const loss = (m) => tr(m).best_val ?? m.best_val ?? Infinity;
  if (labState.order) models.sort((a, b) => labState.order.indexOf(a.id) - labState.order.indexOf(b.id));
  else models.sort((a, b) => loss(b) - loss(a));
  if (!models.length) { $("#lab-compare").innerHTML = '<p class="muted">Tick some models above to compare them.</p>'; $("#lab-curves").innerHTML = ""; return; }
  const rows = labCompareRows(models);
  const colors = (i) => MODEL_COLORS[i % MODEL_COLORS.length];
  const cells = (r) => models.map((m) => {
    if (r.num) { const v = r.num(m); return v == null || Number.isNaN(v) ? null : v; }
    return r.get(m);
  });
  const body = rows.map((r) => {
    if (r.section) return `<tr class="sec"><th colspan="${models.length + 1}">${esc(r.section)}${r.note ? ` <span class="muted small">· ${esc(r.note)}</span>` : ""}</th></tr>`;
    const vals = cells(r);
    if (vals.every((v) => v == null || v === "–")) return "";
    const nums = vals.filter((v) => typeof v === "number");
    const best = r.best && nums.length > 1 ? (r.best === "min" ? Math.min(...nums) : Math.max(...nums)) : null;
    const shown = vals.map((v) => (v == null ? "–" : r.num ? r.fmt(v) : v));
    return `<tr><th>${esc(r.label)}</th>${vals.map((v, i) => {
      const differs = i > 0 && shown[i] !== shown[0];
      const sub = r.sub && v != null ? r.sub(models[i]) : "";
      return `<td class="${differs ? "diff" : ""}">${v === best && best != null ? `<b>${shown[i]}</b>` : esc(shown[i])}${sub ? `<div class="muted small">${esc(sub)}</div>` : ""}</td>`;
    }).join("")}</tr>`;
  }).join("");
  $("#lab-compare").innerHTML = `<div class="cmp-table-wrap"><table class="cmp-table">
    <thead><tr><th></th>${models.map((m, i) => `<th title="${esc(m.name)}"><span class="sw-dot" style="background:${colors(i)}"></span>${esc(labelOf(m))}<div class="muted small">${esc(groupOf(m))}</div></th>`).join("")}</tr></thead>
    <tbody>${body}</tbody></table></div>`;
  // validation curves on a shared update axis, and training loss where it was recorded
  const val = models.map((m, i) => ({ label: labelOf(m), color: colors(i), dots: true,
    points: (tr(m).val_curve || []).map((v, k) => [tr(m).val_steps?.[k] ?? (k + 1) * (tr(m).eval_every || 1), v]) })).filter((s) => s.points.length);
  const train = models.map((m, i) => ({ label: labelOf(m), color: colors(i), smooth: 8,
    points: (tr(m).step_losses || []).map((v, k) => [(k + 1) * (tr(m).step_loss_stride || 1), v]) })).filter((s) => s.points.length > 5);
  $("#lab-curves").innerHTML = (val.length ? `<div><div class="muted small">Validation loss</div><div class="chart">${xyChart(val, { xLabel: "update", yLabel: "loss", W: 520, H: 230 })}</div></div>` : "") +
    (train.length ? `<div><div class="muted small">Training loss (moving average)</div><div class="chart">${xyChart(train, { xLabel: "update", yLabel: "loss", W: 520, H: 230,
      yMin: Math.min(...train.flatMap((s) => s.points.map((p) => p[1]))) * 0.97,
      // skip the first 10% of training (loss falls from ~11) so the differences that matter are visible
      yMax: Math.max(...train.map((s) => s.points[Math.floor(s.points.length / 10)]?.[1] ?? 0)) * 1.05 })}</div></div>` : "") +
    (models.some((m) => !tr(m).val_curve) ? '<p class="hint">Models from earlier runs don\'t have a saved loss curve; their table columns use what their saved config records.</p>' : "");
}

// a line chart with a numeric x axis (updates), for series of different lengths
function xyChart(series, { xLabel = "", yLabel = "", W = 520, H = 220, yMin = null, yMax = null } = {}) {
  const pts = series.flatMap((s) => s.points);
  if (!pts.length) return "";
  const L = 46, R = 12, T = 10, B = 34;
  const xmax = Math.max(...pts.map((p) => p[0])), ys = pts.map((p) => p[1]).filter(Number.isFinite);
  const lo = yMin ?? Math.min(...ys) * 0.97, hi = yMax ?? Math.max(...ys) * 1.02;
  const X = (x) => L + (x / xmax) * (W - L - R), Y = (y) => T + (1 - (Math.min(Math.max(y, lo), hi) - lo) / (hi - lo || 1)) * (H - T - B);
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => lo + f * (hi - lo));
  const grid = ticks.map((v) => `<line x1="${L}" x2="${W - R}" y1="${Y(v)}" y2="${Y(v)}"/><text x="${L - 6}" y="${Y(v) + 4}" text-anchor="end">${v.toFixed(2)}</text>`).join("");
  const avg = (p, k) => p.map((_, i) => { const w = p.slice(Math.max(0, i - k), i + k + 1); return [p[i][0], w.reduce((a, b) => a + b[1], 0) / w.length]; });
  const line = (p) => p.map(([x, y]) => `${X(x).toFixed(1)},${Y(y).toFixed(1)}`).join(" ");
  const lines = series.map((s) => {
    const p = s.smooth ? avg(s.points, s.smooth) : s.points;
    return `<polyline fill="none" stroke="${s.color}" stroke-width="2" stroke-linejoin="round" points="${line(p)}"><title>${esc(s.label)}</title></polyline>` +
      (s.dots ? s.points.map(([x, y]) => `<circle cx="${X(x)}" cy="${Y(y)}" r="3" fill="${s.color}"><title>${esc(s.label)}: ${y.toFixed(3)} at update ${fmt(x)}</title></circle>`).join("") : "");
  }).join("");
  const xt = [0, 0.5, 1].map((f) => `<text x="${X(f * xmax)}" y="${H - 18}" text-anchor="${f === 0 ? "start" : f === 1 ? "end" : "middle"}">${fmt(Math.round(f * xmax))}</text>`).join("");
  const legend = series.length > 1 ? `<div class="xy-legend">${series.map((s) => `<span><i style="background:${s.color}"></i>${esc(s.label)}</span>`).join("")}</div>` : "";
  return `<svg viewBox="0 0 ${W} ${H}"><g class="grid">${grid}</g>${lines}${xt}
    <text x="${(L + W - R) / 2}" y="${H - 3}" text-anchor="middle">${esc(xLabel)}</text>
    <text transform="translate(11 ${(T + H - B) / 2}) rotate(-90)" text-anchor="middle">${esc(yLabel)}</text></svg>${legend}`;
}

// ---------- events ----------
document.addEventListener("click", (e) => {
  const p = e.target.closest("[data-lab-preset]");
  if (p) labApplyPreset(p.dataset.labPreset);
  if (e.target.closest("#lab-reset")) labApplyPreset(labState.preset || "gpt2");
  if (e.target.closest("#lab-train")) labTrain(e.target.closest("#lab-train"));
  if (e.target.closest("#lab-stop")) action(e.target.closest("#lab-stop"), () => api("/api/train/stop", {}));
  const pk = e.target.closest("#lab-pick-final, #lab-pick-ladder, #lab-pick-none");
  if (pk) { labPick(pk.id.replace("lab-pick-", "")); renderLabModels(); renderLabCompare(); }
});
document.addEventListener("change", (e) => {
  const k = e.target.dataset?.labKey;
  if (k) {
    const raw = e.target.type === "number" ? Math.max(+e.target.min, Math.min(+e.target.max, Math.round(+e.target.value || 0))) : JSON.parse(e.target.value);
    labSetKey(k, raw);
  }
  const d = e.target.dataset?.labDs;
  if (d) { e.target.checked ? labState.datasets.add(d) : labState.datasets.delete(d); renderLabDatasets(); renderLabCalc(); }
  if (e.target.id === "lab-mix") labState.mix = e.target.value;
  if (e.target.id === "lab-updates" || e.target.id === "lab-lr") renderLabCalc();
  const pick = e.target.dataset?.labPick;
  if (pick) { e.target.checked ? labState.picked.add(pick) : labState.picked.delete(pick); labState.order = null; savePicked(); renderLabModels(); renderLabCompare(); }
});
document.addEventListener("input", (e) => {
  if (e.target.id === "lab-search") { labState.search = e.target.value; renderLabModels(); }
});

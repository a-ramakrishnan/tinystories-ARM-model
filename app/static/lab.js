// Lab: pick an architecture (a known design or every switch), one or more datasets, train, and compare the
// result with every model trained so far. Uses the helpers in app.js ($, api, esc, fmt, short, bytes, toast…).

const LAB_GPT2 = { n_layer: 6, n_head: 6, n_embd: 384, dropout: 0, pos_encoding: "learned", norm: "layernorm", mlp: "gelu",
  bias: true, qk_norm: false, n_kv_head: 0, block_repeat: 1, mtp_heads: 1, n_experts: 1, moe_top_k: 1,
  rope_theta: 10000, moe_aux_weight: 0.01 };
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
      { key: "pos_encoding", label: "Positions", options: [["learned", "Learned (GPT-2)"], ["rope", "RoPE, rotary (Llama)"]], measured: "RoPE: 2.631 → 2.475, and a KV cache that can slide (A1)",
        params: (a) => (a.pos_encoding === "rope" ? [{ key: "rope_theta", label: "Rotation base θ", min: 100, max: 10000000, step: 100, def: 10000,
          hint: "sets the slowest rotation; 10,000 in GPT-NeoX and Llama 2, 500,000 in Llama 3 for long contexts" }] : []) },
      { key: "qk_norm", label: "QK-norm", options: [[false, "Off"], [true, "On"]], measured: "2.369 → 2.336; at a 7× higher learning rate, 2.55 instead of 3.33 (A5, sweep)" },
      { key: "n_kv_head", label: "Key/value heads", options: "kv", measured: "Half as many: KV cache halved, loss unchanged (A6)" },
    ] },
  { group: "Each block", items: [
      { key: "norm", label: "Normalization", options: [["layernorm", "LayerNorm (GPT-2)"], ["rmsnorm", "RMSNorm (Llama)"]], measured: "A tie: 2.475 vs 2.483 (A2)" },
      { key: "mlp", label: "Feed-forward", options: [["gelu", "GELU, 4× wide (GPT-2)"], ["swiglu", "SwiGLU, gated (Llama)"]], measured: "SwiGLU: 2.475 → 2.366 at equal size (A3)" },
      { key: "bias", label: "Biases", options: [[true, "Yes (GPT-2)"], [false, "No (Llama)"]], measured: "A tie, with 29k fewer parameters (A4)" },
      { key: "n_experts", label: "Experts", options: [[1, "1 (dense)"], [4, "4"], [8, "8"]], measured: "4 experts, 2 per token: 2.251 → 2.200; 1 per token was worse (A10)",
        params: (a) => (a.n_experts > 1 ? [{ key: "moe_aux_weight", label: "Balancing loss weight", min: 0, max: 0.2, step: 0.005, def: 0.01,
          hint: "how hard the router is pushed to spread tokens evenly; 0.01 as in Switch Transformer" }] : []) },
      { key: "moe_top_k", label: "Experts per token", options: [[1, "1"], [2, "2"]], when: (a) => a.n_experts > 1 },
      { key: "block_repeat", label: "Repeat each block", options: [[1, "Once"], [2, "2× (shared)"], [3, "3× (shared)"]], measured: "Worse at this budget: 2.251 → 2.346 at 2× (A8)" },
    ] },
  { group: "Training objective", items: [
      { key: "mtp_heads", label: "Tokens predicted", options: [[1, "1, the next"], [2, "2"], [3, "3"], [4, "4"]], measured: "Worse at 30M: 2.251 → 2.336 with 2 (A9)" },
    ] },
];

const LAB_LABELS = { pos_encoding: "positions", norm: "normalization", mlp: "feed-forward", bias: "biases", qk_norm: "QK-norm",
  n_kv_head: "KV heads", block_repeat: "block sharing", mtp_heads: "tokens predicted", n_experts: "experts", moe_top_k: "experts per token",
  n_layer: "layers", n_head: "heads", n_embd: "embedding", rope_theta: "RoPE θ", moe_aux_weight: "balancing weight", dropout: "dropout" };

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
  await Promise.all([labLoadDatasets(), labLoadModels(), labLoadZips()]).catch((e) => toast(e.message, true));
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
  labState.reco = null;
  renderLabReco();
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
        const why = labState.reco && labState.reco.why[it.key];
        const params = (it.params ? it.params(a) : []).map((q) => `<label class="li-param"><span>${esc(q.label)}${a[q.key] !== q.def ? ` <button class="li-def" data-lab-def="${q.key}" data-def="${q.def}" title="back to the default">default ${fmt(q.def)}</button>` : ""}</span>
          <input type="number" data-lab-key="${q.key}" data-float="1" min="${q.min}" max="${q.max}" step="${q.step}" value="${a[q.key]}"><span class="li-measured">${esc(q.hint)}</span></label>`).join("");
        return `<div class="lab-item ${diff ? "changed" : ""} ${why ? "reco" : ""}"><label class="li-main"><span class="li-label">${esc(it.label)}${diff ? '<span class="li-dot" title="differs from GPT-2 style"></span>' : ""}</span>${input}</label>${why ? `<span class="li-why">✓ ${esc(why)}</span>` : it.measured ? `<span class="li-measured">${esc(it.measured)}</span>` : ""}${params}</div>`;
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
  renderLabReco();
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
  $("#lab-data-note").textContent = ds.length ? `${ds.filter((d) => d.local).length} ready on this Mac · ${ds.filter((d) => !d.local).length} more for Colab or Kaggle` : "";
  if (!ds.length) { $("#lab-datasets").innerHTML = '<p class="muted">No datasets yet. Add and tokenize one in steps 1 and 2.</p>'; return; }
  const several = labState.datasets.size > 1;
  const sorted = [...ds].sort((a, b) => (b.local ? 1 : 0) - (a.local ? 1 : 0));
  $("#lab-datasets").innerHTML = `<div class="lab-ds">${sorted.map((d) => `<label class="lab-ds-item ${labState.datasets.has(d.key) ? "on" : ""}">
      <input type="checkbox" data-lab-ds="${esc(d.key)}" ${labState.datasets.has(d.key) ? "checked" : ""}>
      <span><b>${esc(d.title)}</b>${d.local
        ? `<span class="muted small">${short(d.train_tokens)} training tokens · ${short(d.val_tokens)} validation</span><span class="tag good">ready on this Mac</span>`
        : `<span class="muted small">${esc([d.rows, d.size].filter(Boolean).join(" · "))}</span><span class="tag ghost">Colab or Kaggle</span>`}</span></label>`).join("")}</div>
    ${several ? `<label class="inline small">Mix <select id="lab-mix" class="select"><option value="size">In proportion to their size</option><option value="equal">Equally</option></select></label>` : ""}
    <p class="hint">Datasets marked <b>Colab or Kaggle</b> aren't tokenized on this Mac: a cloud run downloads and tokenizes them there. To train on one here, add and tokenize it in steps 1 and 2. Every model is also evaluated on WikiText-2 and Tiny Shakespeare (held-out loss), to see how it does on text it never saw.</p>`;
  if (several && labState.mix) $("#lab-mix").value = labState.mix;
}

// ---------- 4. train ----------
function renderLabCalc() {
  const e = labState.est, ds = (labState.info?.datasets || []).filter((d) => labState.datasets?.has(d.key));
  const updates = +$("#lab-updates").value, tokens = updates * 32 * 256;
  const dataTokens = ds.every((d) => d.train_tokens) ? ds.reduce((x, d) => x + d.train_tokens, 0) : 0;
  const dev = labState.info?.device;
  // measured: the 30M model took ~1.5 s per update on the M5 (micro-batches of 8) and ~0.15 s on a Colab L4
  // experts run one after another in this implementation: measured ~1.8× slower per update than the active size suggests
  const perUpdate = (dev === "cuda" ? 0.15 : dev === "mps" ? 1.5 : 12) * Math.max(0.3, (e?.active || 3e7) / 3e7) * (labState.arch.block_repeat || 1)
    * ((labState.arch.n_experts || 1) > 1 ? 1.8 : 1) * (labState.arch.mtp_heads || 1);
  const mins = (updates * perUpdate) / 60;
  $("#lab-cost").textContent = e ? `about ${mins < 1.5 ? "a minute" : `${Math.round(mins)} minutes`} on this ${dev === "mps" ? "Mac" : dev === "cuda" ? "GPU" : "computer"}, plus about a minute to evaluate` : "";
  $("#lab-calc").innerHTML = e ? `<div class="calc-line"><span>Tokens seen</span><code>${fmt(updates)} updates × 32 sequences × 256 tokens = <b>${short(tokens)}</b></code></div>
    ${dataTokens ? `<div class="calc-line"><span>Passes over the data</span><code>${short(tokens)} ÷ ${short(dataTokens)} = <b>${(tokens / dataTokens).toFixed(tokens / dataTokens < 0.1 ? 3 : 2)}</b></code></div>` : ""}
    <div class="calc-line"><span>Learning rate</span><code>warm up over ${fmt(Math.round(updates * +$("#lab-warmup").value / 100))} updates to <b>${(+$("#lab-lr").value).toExponential(1)}</b>, cosine decay to ${(+$("#lab-lr").value * +$("#lab-minlr").value / 100).toExponential(1)}</code></div>
    ${dev === "mps" ? `<div class="calc-line"><span>On this Mac</span><code>each update = 4 micro-batches of 8 (same gradient as 32, a quarter of the memory)</code></div>` : ""}` : "";
}

// What the Train button will train: a name and every setting, for this Mac's API or a cloud spec.
function labPlan() {
  const all = labState.info?.datasets || [], ds = all.filter((d) => labState.datasets?.has(d.key));
  const preset = LAB_PRESETS.find((p) => p.id === labState.preset), t = labGetTrain();
  const dataTitle = ds.map((d) => d.title).join(" + ");
  const name = $("#lab-name").value.trim() || `${preset ? preset.title : labState.reco ? LAB_GOALS[labState.reco.goal].title : "Custom"} · ${dataTitle} · ${fmt(t.updates)} updates`;
  return { name, ds, t, dataTitle, arch: { ...labState.arch, dropout: t.dropout }, allLocal: ds.every((d) => d.local) };
}

function labTrain() {
  const p = labPlan();
  if (!p.ds.length) { toast("Pick at least one dataset.", true); return; }
  if (!labState.est) { toast(labState.estError || "Fix the model settings first.", true); return; }
  const dev = labState.info?.device, here = dev === "mps" ? "This Mac" : dev === "cuda" ? "This computer's GPU" : "This computer (CPU)";
  const missing = p.ds.filter((d) => !d.local).map((d) => d.title);
  const same = (x) => JSON.stringify(x.arch) === JSON.stringify(labState.arch);
  $("#lab-where-body").innerHTML = `
    <p class="muted small"><b>${esc(p.name)}</b><br>${short(labState.est.params)} parameters · ${fmt(p.t.updates)} updates on ${esc(p.dataTitle)}</p>
    <div class="where-opts">
      <button class="where-opt" data-where="local" ${p.allLocal ? "" : "disabled"}><b>💻 ${here}</b>
        <span>${p.allLocal ? esc($("#lab-cost").textContent || "Trains here, in this page.") : `${esc(missing.join(", "))} ${missing.length > 1 ? "aren't" : "isn't"} tokenized here. Add ${missing.length > 1 ? "them" : "it"} in steps 1–2 first, or use the cloud.`}</span></button>
      <button class="where-opt" data-where="colab"><b>☁️ Google Colab</b><span>A free T4 GPU, or L4 / A100 with Colab Pro. About 10× faster than the Mac. Needs a Google account.</span></button>
      <button class="where-opt" data-where="kaggle"><b>☁️ Kaggle</b><span>Free GPUs (30 hours a week). Needs a Kaggle account with a verified phone number.</span></button>
    </div>
    <div id="lab-cloud" class="hidden">
      <div class="where-also"><b>Also train, for comparison</b> <span class="muted small">same data and updates, in the same run</span>
        ${LAB_PRESETS.filter((x) => ["gpt2", "best", "moe"].includes(x.id)).map((x) => `<label class="inline small"><input type="checkbox" data-also="${x.id}" ${same(x) ? "disabled" : ""}> ${esc(x.title)}${same(x) ? " (this model)" : ""}</label>`).join("")}</div>
      <ol class="where-steps">
        <li><b>Copy the run spec.</b> It holds every setting chosen here. <button class="btn small primary" id="lab-copy-spec">Copy spec</button> <span id="lab-copied" class="good-text small"></span>
          <textarea id="lab-spec" readonly rows="2" spellcheck="false"></textarea></li>
        <li><b>Open the notebook.</b> <a id="lab-open-nb" class="btn small" target="_blank" rel="noopener">Open</a> <span id="lab-nb-note" class="muted small"></span></li>
        <li id="lab-paste-step"></li>
        <li><b>Bring the results home.</b> Save the zip in the project folder or Downloads, then press <b>Import</b> under “Results from Colab or Kaggle” below. The models join the comparison.</li>
      </ol>
    </div>`;
  $("#lab-where").showModal();
}

const LAB_NB = "a-ramakrishnan/tinystories-ARM-model/blob/main/lab_run.ipynb";
function labSpec() {
  const p = labPlan();
  const variant = (name, display, arch, lr) => ({ name, display_name: display, arch, lr, warmup_frac: p.t.warmup / 100, min_lr_frac: p.t.minlr / 100,
    weight_decay: p.t.wd, grad_clip: p.t.clip });
  const variants = [variant("model", p.name, p.arch, p.t.lr)];
  $$("[data-also]:checked").forEach((c) => {
    const x = LAB_PRESETS.find((q) => q.id === c.dataset.also);
    variants.push(variant(x.id, `${x.title} · ${p.dataTitle} · ${fmt(p.t.updates)} updates`, { ...x.arch, dropout: p.t.dropout }, x.lr));
  });
  const spec = { title: `Lab: ${p.name}`.slice(0, 90), updates: p.t.updates, mix: $("#lab-mix")?.value || "size",
    datasets: p.ds.map((d) => ({ name: d.name, config: d.config || null, text_column: d.text_column || "text", title: d.title })), variants };
  return btoa(unescape(encodeURIComponent(JSON.stringify(spec))));  // base64: safe to paste into a notebook form field
}

function labCloud(where) {
  labState.where = where;
  $$(".where-opt").forEach((b) => b.classList.toggle("on", b.dataset.where === where));
  $("#lab-cloud").classList.remove("hidden");
  const colab = where === "colab";
  $("#lab-open-nb").textContent = colab ? "Open in Colab" : "Open in Kaggle";
  $("#lab-open-nb").href = colab ? `https://colab.research.google.com/github/${LAB_NB}` : `https://kaggle.com/kernels/welcome?src=https://github.com/${LAB_NB}`;
  $("#lab-nb-note").textContent = colab ? "Then Runtime → Change runtime type → a GPU." : "Then Settings → Accelerator → GPU, and turn Internet on.";
  $("#lab-paste-step").innerHTML = colab
    ? "<b>Paste and run.</b> Paste the spec into the <code>spec</code> field of the last cell, then <i>Runtime → Run all</i>. The zip downloads when it finishes."
    : "<b>Paste and run.</b> Paste the spec between the quotes of <code>spec = \"\"</code> in the last cell, then <i>Run all</i>. The zip appears in the Output panel.";
  labRefreshSpec();
  labCopySpec();
}
function labRefreshSpec() { $("#lab-spec").value = labSpec(); $("#lab-copied").textContent = ""; }
async function labCopySpec() {
  const el = $("#lab-spec");
  try { await navigator.clipboard.writeText(el.value); $("#lab-copied").textContent = "Copied to the clipboard."; }
  catch (e) { el.focus(); el.select(); $("#lab-copied").textContent = "Select the text below and copy it."; }
}

async function labTrainLocal(btn) {
  const p = labPlan();
  await action(btn, async () => {
    await api("/api/lab/train", { name: p.name, arch: p.arch, datasets: p.ds.map((d) => d.key), mix: $("#lab-mix")?.value || "size",
      updates: p.t.updates, learning_rate: p.t.lr, warmup_frac: p.t.warmup / 100, min_lr_frac: p.t.minlr / 100, weight_decay: p.t.wd, grad_clip: p.t.clip });
    $("#lab-where").close();
    toast("Training started on this Mac.");
  });
}

// ---------- results from Colab or Kaggle ----------
async function labLoadZips() {
  const r = await api("/api/lab/zips");
  const when = (t) => new Date(t * 1000).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
  $("#lab-zips").innerHTML = r.zips.length ? `<div class="lab-zips">${r.zips.map((z) => `<div class="lab-zip"><span class="lab-mname" title="${esc(z.path)}">${esc(z.file)}</span>
      <span class="muted small">${esc(z.folder)} · ${bytes(z.bytes)} · ${when(z.modified)}</span>
      ${z.imported ? '<span class="tag good">imported</span>' : ""}<button class="btn small ${z.imported ? "ghost" : "primary"}" data-lab-import="${esc(z.path)}">${z.imported ? "Import again" : "Import"}</button></div>`).join("")}</div>`
    : '<p class="muted small">No results zips in the project folder or Downloads yet.</p>';
}

function labRender(s) {
  if (!labState.opened) return;
  const j = s.job, L = s.lab, running = j.status === "running";
  if (j.kind === "import") {
    if (labState.importing && !running) { labLoadModels(); labLoadZips(); toast(j.status === "error" ? j.error : j.message, j.status === "error"); }
    labState.importing = running;
  }
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
// A/B variants go by their short variant name; Lab runs (here or in the cloud) by the name given in the Lab
const labelOf = (m) => (tr(m).experiment?.startsWith("Lab") ? (tr(m).description || m.name)
  : tr(m).experiment === "Reference" ? m.name : tr(m).experiment && tr(m).variant ? tr(m).variant : m.name);

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
  const num = (g) => (g === "Reference" ? 1001 : g === "Lab runs" ? 1000 : g.startsWith("Lab") ? 1000 : g.startsWith("Final") ? 999 : g === "Earlier runs" ? -1
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
  if (which === "gpt2") labState.picked = new Set(ms.filter((m) => tr(m).experiment === "Reference"
    || (m.datasets || []).some((d) => /fineweb/i.test(d))).map((m) => m.id));
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
    { label: "RoPE base θ", get: (m) => (c(m).pos_encoding === "rope" ? fmt(c(m).rope_theta || 10000) : "–") },
    { label: "Balancing loss weight", get: (m) => ((c(m).n_experts || 1) > 1 ? String(c(m).moe_aux_weight ?? 0.01) : "–") },
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
    // A/B and Lab runs before these were editable all used the defaults: 10% warmup, decay to 10%, 0.1, 1.0, no dropout
    { label: "Warmup / final learning rate", get: (m) => (t(m).warmup_frac != null ? `${Math.round(t(m).warmup_frac * 100)}% / ${Math.round(t(m).min_lr_frac * 100)}% of peak`
      : t(m).batch ? "10% / 10% of peak" : "–") },
    { label: "Weight decay / clipping / dropout", get: (m) => (t(m).weight_decay != null ? `${t(m).weight_decay} / ${t(m).grad_clip} / ${t(m).dropout ?? 0}`
      : t(m).batch ? "0.1 / 1.0 / 0" : "–") },
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
  if (e.target.closest("#lab-recommend")) labRecommend($("#lab-goal").value);
  if (e.target.closest("#lab-reco-restore")) labRecommend(labState.reco.goal);
  if (e.target.closest("#lab-train-reset")) { labSetTrain(labState.reco ? labState.reco.train : LAB_TRAIN_DEFAULTS); renderLabReco(); }
  const def = e.target.closest("[data-lab-def]");
  if (def) { e.preventDefault(); labSetKey(def.dataset.labDef, +def.dataset.def); }
  if (e.target.closest("#lab-train")) labTrain();
  const w = e.target.closest("[data-where]");
  if (w && !w.disabled) { if (w.dataset.where === "local") labTrainLocal(w); else labCloud(w.dataset.where); }
  if (e.target.closest("#lab-copy-spec")) labCopySpec();
  if (e.target.closest("#lab-where-close")) $("#lab-where").close();
  if (e.target.closest("#lab-zips-refresh")) labLoadZips().catch((err) => toast(err.message, true));
  const imp = e.target.closest("[data-lab-import]");
  if (imp) action(imp, async () => { await api("/api/lab/import", { path: imp.dataset.labImport }); labState.importing = true; toast("Importing…"); });
  if (e.target.closest("#lab-stop")) action(e.target.closest("#lab-stop"), () => api("/api/train/stop", {}));
  const pk = e.target.closest("#lab-pick-final, #lab-pick-ladder, #lab-pick-gpt2, #lab-pick-none");
  if (pk) { labPick(pk.id.replace("lab-pick-", "")); renderLabModels(); renderLabCompare(); }
});
document.addEventListener("change", (e) => {
  const k = e.target.dataset?.labKey;
  if (k) {
    const num = Math.max(+e.target.min, Math.min(+e.target.max, +e.target.value || 0));
    const raw = e.target.type === "number" ? (e.target.dataset.float ? num : Math.round(num)) : JSON.parse(e.target.value);
    labSetKey(k, raw);
  }
  if (e.target.dataset?.also) labRefreshSpec();
  const d = e.target.dataset?.labDs;
  if (d) { e.target.checked ? labState.datasets.add(d) : labState.datasets.delete(d); renderLabDatasets(); renderLabCalc(); }
  if (e.target.id === "lab-mix") labState.mix = e.target.value;
  if (Object.values(LAB_TRAIN_FIELDS).includes(`#${e.target.id}`)) {
    const el = e.target, v = Math.max(+el.min, Math.min(+el.max, +el.value || 0));
    if (String(v) !== el.value) el.value = String(v);
    if (el.id === "lab-dropout") labState.arch.dropout = v;
    renderLabCalc();
    renderLabReco();
  }
  const pick = e.target.dataset?.labPick;
  if (pick) { e.target.checked ? labState.picked.add(pick) : labState.picked.delete(pick); labState.order = null; savePicked(); renderLabModels(); renderLabCompare(); }
});
document.addEventListener("input", (e) => {
  if (e.target.id === "lab-search") { labState.search = e.target.value; renderLabModels(); }
});

// ---------- recommendations, from everything measured in this project ----------
const LAB_TRAIN_DEFAULTS = { updates: 1000, lr: 1.8e-3, warmup: 10, minlr: 10, wd: 0.1, clip: 1, dropout: 0 };
const LAB_GOALS = {
  quality: { title: "Best quality", arch: { ...LAB_BEST, n_experts: 4, moe_top_k: 2 }, train: { updates: 5000 },
    expect: "Validation loss 1.732 after 5,000 updates and 2.200 after 1,000 (both measured on TinyStories).",
    cost: "51M parameters (37M used per token); generation about half the dense model's speed in this implementation (66 vs 132 tokens/s).",
    why: { pos_encoding: "RoPE: 2.631 → 2.475", mlp: "SwiGLU: 2.475 → 2.366", bias: "no biases: free (a tie, fewer parameters)", qk_norm: "QK-norm: 2.369 → 2.336, safe at high learning rates",
      n_experts: "4 experts, 2 per token: 2.251 → 2.200", moe_top_k: "2 per token: 1 per token was worse", n_layer: "6 × 384 beat 3, 11 and 22 layers", n_kv_head: "all heads: the cache isn't the bottleneck here",
      block_repeat: "no sharing: it was worse (2.346)", mtp_heads: "next token only: multi-token was worse at 30M", norm: "either: RMSNorm tied" } },
  balanced: { title: "Quality per compute", arch: { ...LAB_BEST, n_kv_head: 3 }, train: { updates: 5000 },
    expect: "About 1.78 after 5,000 updates (the dense best measured 1.782; 3 key/value heads alone measured 2.252 vs 2.251 after 1,000).",
    cost: "Every part was measured, but not this exact combination. Generates at the dense speed with half the KV cache.",
    why: { pos_encoding: "RoPE: 2.631 → 2.475", mlp: "SwiGLU: 2.475 → 2.366", bias: "no biases: free", qk_norm: "QK-norm: 2.369 → 2.336",
      n_kv_head: "3 key/value heads: KV cache halved, loss unchanged", n_experts: "dense: experts halve generation speed here", n_layer: "6 × 384: best shape at this size",
      block_repeat: "no sharing: it was worse", mtp_heads: "next token only", norm: "either: RMSNorm tied" } },
  speed: { title: "Fastest generation", arch: { ...LAB_BEST, n_layer: 3, n_head: 7, n_embd: 448 }, train: { updates: 5000 },
    expect: "Validation loss 2.271 after 1,000 updates (0.02 behind the best dense shape), 215 tokens/s on an L4 vs 132.",
    cost: "Fewer, wider layers: tokens are generated 1.6× faster and the KV cache is 42% smaller.",
    why: { n_layer: "3 layers: each layer runs in turn, so fewer layers = faster tokens", n_embd: "448 wide keeps ~30M parameters", n_head: "7 heads of 64",
      pos_encoding: "RoPE: 2.631 → 2.475", mlp: "SwiGLU: 2.475 → 2.366", bias: "no biases: free", qk_norm: "QK-norm: small win, stable",
      n_experts: "dense: experts halve generation speed here", block_repeat: "no sharing: it doubles the work per token", mtp_heads: "next token only" } },
  memory: { title: "Smallest KV cache", arch: { ...LAB_BEST, n_kv_head: 2 }, train: { updates: 5000 },
    expect: "Validation loss 2.264 after 1,000 updates (+0.013 over the best dense model); a 3× smaller KV cache (6,144 bytes per token).",
    cost: "For long contexts or many users at once. 1 key/value head would make it 6× smaller for +0.033.",
    why: { n_kv_head: "2 key/value heads: cache 3× smaller for +0.013", pos_encoding: "RoPE: 2.631 → 2.475", mlp: "SwiGLU: 2.475 → 2.366", bias: "no biases: free",
      qk_norm: "QK-norm: small win, stable", n_experts: "dense: experts add parameters to store", n_layer: "6 layers: each layer adds its own cache",
      block_repeat: "no sharing: each pass adds its own cache", mtp_heads: "next token only" } },
};

function labRecommend(goal) {
  const g = LAB_GOALS[goal];
  labState.goal = goal;
  labState.arch = { ...g.arch };
  labState.preset = (LAB_PRESETS.find((p) => JSON.stringify(p.arch) === JSON.stringify(labState.arch)) || {}).id || null;
  labState.reco = { goal, arch: { ...g.arch }, train: { ...LAB_TRAIN_DEFAULTS, ...g.train }, why: g.why };
  labSetTrain(labState.reco.train);
  renderLabPresets();
  renderLabSwitches();
  renderLabReco();
  labEstimate();
  toast(`Recommended settings for ${g.title.toLowerCase()} applied. Change anything you like.`);
}

function renderLabReco() {
  const r = labState.reco, el = $("#lab-reco");
  if (!r) { el.innerHTML = ""; return; }
  const g = LAB_GOALS[r.goal];
  const changed = Object.keys(r.arch).filter((k) => JSON.stringify(labState.arch[k]) !== JSON.stringify(r.arch[k]));
  const t = labGetTrain(), tChanged = Object.keys(r.train).filter((k) => +t[k] !== +r.train[k]);
  el.innerHTML = `<div class="lab-reco"><div><b>Recommended for ${esc(g.title.toLowerCase())}.</b> ${esc(g.expect)}</div>
    <div class="muted small">${esc(g.cost)}</div>
    <div class="small">${changed.length || tChanged.length
      ? `You changed ${[...changed.map((k) => LAB_LABELS[k] || k), ...tChanged.map((k) => LAB_TRAIN_LABELS[k])].map((x) => `<span class="tag">${esc(x)}</span>`).join(" ")} from the recommendation. <button class="btn small ghost" id="lab-reco-restore">Restore the recommendation</button>`
      : '<span class="good-text">Using the recommendation as is.</span> Every field below can still be changed.'}</div></div>`;
}

// ---------- training settings (editable) ----------
const LAB_TRAIN_LABELS = { updates: "updates", lr: "learning rate", warmup: "warmup", minlr: "final learning rate", wd: "weight decay", clip: "clipping", dropout: "dropout" };
const LAB_TRAIN_FIELDS = { updates: "#lab-updates", lr: "#lab-lr", warmup: "#lab-warmup", minlr: "#lab-minlr", wd: "#lab-wd", clip: "#lab-clip", dropout: "#lab-dropout" };
function labGetTrain() { return Object.fromEntries(Object.entries(LAB_TRAIN_FIELDS).map(([k, sel]) => [k, +$(sel).value])); }
function labSetTrain(t) {
  Object.entries(LAB_TRAIN_FIELDS).forEach(([k, sel]) => { if (t[k] != null) $(sel).value = String(t[k]); });
  labState.arch.dropout = +$("#lab-dropout").value;
  renderLabCalc();
}

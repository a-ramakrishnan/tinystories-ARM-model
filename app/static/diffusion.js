// "Diffusion" page: the masked diffusion models (diffusion_models/) writing by denoising. The continuation starts
// as all [MASK] (█); every step is one pass through the model, which predicts every position at once, and a few
// of the masked positions are revealed. A player replays the steps; the grid shows when each position was revealed.

const df = { info: null, run: null, k: 0, timer: null };
const DF_PRESETS = {
  default: { label: "confident first + repetition penalty 1.3 (default)", sampler: "keep", repetition_penalty: 1.3, block_size: 0, temperature: 1.0 },
  keep: { label: "confident first, no penalty (the old default)", sampler: "keep", repetition_penalty: 1.0, block_size: 0, temperature: 1.0 },
  blocks: { label: "blocks of 8, left to right, penalty 1.2", sampler: "keep", repetition_penalty: 1.2, block_size: 8, temperature: 1.0 },
  random: { label: "random order, temperature 0.8", sampler: "random", repetition_penalty: 1.0, block_size: 0, temperature: 0.8 },
  v1: { label: "v1: re-sample every position every step", sampler: "v1", repetition_penalty: 1.0, block_size: 0, temperature: 1.0 },
};
const dfShade = (p) => `color-mix(in srgb, var(--accent) ${Math.round(12 + 70 * p)}%, transparent)`;
const dfFmt = (v, d = 3) => (v == null ? "–" : (+v).toFixed(d));

async function diffusionOpen() {
  df.info = await (await fetch("/api/diffusion")).json();
  const runs = df.info.runs;
  const ms = $("#df-model"), prev = ms.value;
  ms.innerHTML = runs.map((r) => `<option value="${esc(r.id)}">${esc(r.name)}</option>`).join("");
  if (prev && runs.some((r) => r.id === prev)) ms.value = prev;
  const ps = $("#df-prompt"), pp = ps.value;
  ps.innerHTML = df.info.prompts.map((p) => `<option value="${esc(p.key)}">${esc(p.source)}: ${esc(p.prompt.slice(0, 48).replace(/\s+/g, " "))}…</option>`).join("") +
    '<option value="">Your own prompt (no real continuation)</option>';
  if (ps.dataset.set) ps.value = pp;
  else diffusionMatchPrompt();
  ps.dataset.set = "1";
  $("#df-own-wrap").classList.toggle("hidden", ps.value !== "");
  $("#df-preset").innerHTML = Object.entries(DF_PRESETS).map(([k, v]) => `<option value="${k}">${esc(v.label)}</option>`).join("");
  $("#df-empty").classList.toggle("hidden", runs.length > 0);
  $("#df-main").classList.toggle("hidden", runs.length === 0);
  diffusionModels();
}

// pick a prompt from the model's own dataset
function diffusionMatchPrompt() {
  const r = df.info?.runs.find((x) => x.id === $("#df-model").value);
  const p = r && df.info.prompts.find((x) => x.source === r.source);
  if (p) $("#df-prompt").value = p.key;
  $("#df-own-wrap").classList.toggle("hidden", $("#df-prompt").value !== "");
}

// ---------- one generation ----------
async function diffusionRun() {
  diffusionStop();
  $("#df-busy").classList.remove("hidden");
  try {
    const key = $("#df-prompt").value, pr = DF_PRESETS[$("#df-preset").value];
    df.run = await api("/api/diffusion/generate", { run: $("#df-model").value, prompt_key: key || null,
      prompt: key ? "" : $("#df-own").value, tokens: +$("#df-tokens").value, steps: +$("#df-steps").value,
      sampler: pr.sampler, repetition_penalty: pr.repetition_penalty, block_size: pr.block_size,
      temperature: pr.temperature, top_k: 50, seed: +$("#df-seed").value });
  } finally { $("#df-busy").classList.add("hidden"); }
  const r = df.run, n = r.trace.length;
  // for every position: the step that revealed it (last change), and its confidence then
  r.revealedAt = Array(r.gen_len).fill(null);
  r.revealConf = Array(r.gen_len).fill(0);
  r.trace.forEach((st, k) => st.ids.forEach((id, i) => {
    const before = k ? r.trace[k - 1].ids[i] : r.mask_id;
    if (id !== r.mask_id && id !== before) { r.revealedAt[i] = k + 1; r.revealConf[i] = st.conf[i]; }
    if (id === r.mask_id) r.revealedAt[i] = null;
  }));
  $("#df-out").classList.remove("hidden");
  $("#df-step").max = n;
  $("#df-model-note").textContent = r.device;
  diffusionMetrics();
  $("#df-masked").innerHTML = lineChart([{ label: "still masked", color: "var(--accent)",
    values: r.trace.map((st) => st.ids.filter((x) => x === r.mask_id).length) }], { xLabel: "step", yLabel: "masked positions", yMax: r.gen_len });
  $("#df-ref").innerHTML = r.reference ? `<div class="card-head"><h4>The real continuation</h4></div><div class="tg-text">${tgTok(r.reference)}</div>` : "";
  diffusionShow(0);
  diffusionPlay();
}

function diffusionShow(k) {
  const r = df.run;
  if (!r) return;
  const n = r.trace.length;
  df.k = Math.max(0, Math.min(k, n));
  $("#df-step").value = df.k;
  $("#df-step-label").textContent = `step ${df.k} of ${n}`;
  const ids = df.k ? r.trace[df.k - 1].ids : Array(r.gen_len).fill(r.mask_id);
  const prev = df.k > 1 ? r.trace[df.k - 2].ids : Array(r.gen_len).fill(r.mask_id);
  const conf = df.k ? r.trace[df.k - 1].conf : [];
  const fresh = [];
  const slots = ids.map((id, i) => {
    if (id === r.mask_id) return '<span class="df-mask">█</span>';
    const isNew = id !== prev[i];
    if (isNew) fresh.push({ i, id, p: conf[i] });
    return `<span class="tg-tok${isNew ? " cur" : ""}" style="background:${dfShade(isNew ? conf[i] : r.revealConf[i] || .5)}" title="position ${i + 1}">${tgTok(r.text_of[id] ?? "")}</span>`;
  }).join("");
  $("#df-text").innerHTML = `<span class="tg-prompt">${tgTok(r.prompt)}</span>${slots}`;
  const left = ids.filter((x) => x === r.mask_id).length;
  const st = r.trace[df.k - 1];
  $("#df-stepcard").innerHTML = df.k
    ? `<div class="muted small">Step ${df.k}: one pass through the model, reading all ${r.prompt_tokens + r.gen_len} positions at once (noise level t = ${st.t})</div>
       <p><b>${fresh.length}</b> position${fresh.length === 1 ? "" : "s"} revealed${r.mode === "v1" ? " or changed" : ""} · <b>${left}</b> still masked</p>
       <div class="kvm-bars">${fresh.slice(0, 8).map((f) => `<div class="kvm-row tg-alt"><span class="mono">${tgTok(r.text_of[f.id]) || "∅"}</span><div class="kvm-track"><div class="kvm-bar c" style="width:${Math.max(1, f.p * 100)}%"></div></div><b>pos ${f.i + 1}</b></div>`).join("")}</div>
       <div class="muted small">bars: the model's probability for each revealed token</div>`
    : `<div class="muted small">Step 0</div><p>All ${r.gen_len} positions after the prompt are masked. They are filled in ${n} steps, about ${(r.gen_len / n).toFixed(1)} per step, in whatever order the sampler picks.</p>`;
  diffusionGrid();
}

function diffusionGrid() {
  // a cell per (step, position): filled from the step that revealed the position on; the cells revealed at the
  // current step are outlined. An autoregressive model would draw a staircase here.
  const r = df.run, n = r.trace.length, L = r.gen_len, W = 360, cw = W / L, ch = Math.max(2, W / n), H = ch * n;
  let cells = "";
  for (let i = 0; i < L; i++) {
    const at = r.revealedAt[i];
    if (at == null || at > df.k) continue;
    cells += `<rect x="${i * cw}" y="${(at - 1) * ch}" width="${cw}" height="${(df.k - at + 1) * ch}" fill="var(--accent)" opacity=".13"/>`;
    cells += `<rect x="${i * cw}" y="${(at - 1) * ch}" width="${cw}" height="${ch}" fill="${dfShade(r.revealConf[i])}"${at === df.k ? ' stroke="var(--accent)" stroke-width="1.5"' : ""}><title>position ${i + 1} revealed at step ${at}: ${esc(r.text_of[r.final[i]] ?? "")}</title></rect>`;
  }
  $("#df-grid").innerHTML = `<svg viewBox="-34 -16 ${W + 40} ${H + 22}">
    <rect x="0" y="0" width="${W}" height="${H}" fill="none" stroke="var(--border)"/>${cells}
    <text x="0" y="-5">position 1</text><text x="${W}" y="-5" text-anchor="end">${L}</text>
    <text x="-5" y="${ch}" text-anchor="end">step 1</text><text x="-5" y="${H}" text-anchor="end">${n}</text></svg>
    <p class="hint">Each column is a position, each row a step; a position's cell starts at the step that revealed it. Writing left to right would make a staircase; here the order is chosen by the sampler${$("#df-preset").value === "blocks" ? " (block by block, left to right)" : ""}.</p>`;
}

function diffusionMetrics() {
  const r = df.run, stat = (k, v, d = "") => `<div class="stat"><div class="k">${k}</div><div class="v">${v}</div><div class="d">${d}</div></div>`;
  $("#df-metrics").innerHTML = [
    stat("Steps (model passes)", r.steps, `for ${r.gen_len} positions`),
    stat("Time", `${r.seconds.toFixed(2)} s`, `${(r.seconds * 1000 / r.steps).toFixed(1)} ms per step`),
    stat("Speed", `${r.tokens_per_sec ?? "–"} tok/s`, "positions filled per second"),
    stat("Variety", tgPct(r.repetition.distinct_2), `distinct pairs · ${tgPct(r.repetition.repeated_4grams)} repeated 4-token runs`),
    stat("Length", r.length, r.length < r.gen_len ? "the model ended the document early" : "tokens"),
  ].join("");
}

// ---------- player ----------
function diffusionPlay() {
  diffusionStop();
  const r = df.run;
  if (!r) return;
  if (df.k >= r.trace.length) diffusionShow(0);
  $("#df-play").textContent = "⏸";
  const tick = () => {
    if (df.k >= r.trace.length) { diffusionStop(); return; }
    diffusionShow(df.k + 1);
    df.timer = setTimeout(tick, +$("#df-speed").value);
  };
  df.timer = setTimeout(tick, 150);
}
function diffusionStop() { clearTimeout(df.timer); df.timer = null; $("#df-play").textContent = "▶"; }

// ---------- the trained models ----------
function diffusionModels() {
  const runs = df.info.runs;
  if (!runs.length) { $("#df-models").innerHTML = ""; return; }
  const q = (r, k) => r.quality?.["keep, repetition 1.3"]?.[k] ?? null;
  const qOld = (r, k) => r.quality?.["keep (confident first), top-k 50"]?.[k] ?? null;
  const rows = [["Parameters", (r) => r.params ? `${(r.params / 1e6).toFixed(1)}M` : "–"],
    ["Training tokens", (r) => r.tokens ? `${(r.tokens / 1e6).toFixed(0)}M` : "–"],
    ["Training time", (r) => r.seconds ? `${(r.seconds / 60).toFixed(0)} min on ${esc(r.gpu || "?")}` : "–"],
    ["Likelihood bound (nats/token)", (r) => dfFmt(r.best_bound)],
    ["Judge loss of its text (default sampler)", (r) => dfFmt(q(r, "judge_loss"))],
    ["Repeated 4-token runs (default / no penalty)", (r) => q(r, "repeated_4grams") == null ? "–" : `${tgPct(q(r, "repeated_4grams"))} / ${tgPct(qOld(r, "repeated_4grams"))}`],
    ["Real text's judge loss", (r) => dfFmt(r.quality?.["real continuation"]?.judge_loss)]];
  $("#df-models").innerHTML = `<div class="ptable-scroll"><table class="ptable"><tr><th></th>${runs.map((r) => `<th class="num">${esc(r.name)}<div class="muted small">${esc(r.id)}${r.finished_at ? " · " + esc(r.finished_at) : ""}</div></th>`).join("")}</tr>
    ${rows.map(([label, f]) => `<tr><td>${label}</td>${runs.map((r) => `<td class="num">${f(r)}</td>`).join("")}</tr>`).join("")}</table></div>
    <div class="card-head"><h4>Likelihood bound while training</h4><span class="muted small">an upper bound on the loss per token, comparable with an autoregressive model's loss</span></div>
    <div class="chart">${lineChart(runs.filter((r) => r.val).map((r, i) => ({ label: r.source, color: i ? "var(--series-val)" : "var(--accent)", values: r.val.map((v) => v[2]) })),
      { x0: 1, xLabel: "evaluation (every 1,000 or 2,000 updates)", yLabel: "nats/token" })}</div>
    <p class="hint">The judge loss is our base autoregressive model's loss on the generated text (diffusion_models/quality.py); repeating one word also scores low, so read it with the repetition.</p>`;
}

$("#df-prompt").addEventListener("change", (e) => $("#df-own-wrap").classList.toggle("hidden", e.target.value !== ""));
$("#df-model").addEventListener("change", diffusionMatchPrompt);
$("#df-run").addEventListener("click", (e) => action(e.target, diffusionRun));
$("#df-play").addEventListener("click", () => (df.timer ? diffusionStop() : diffusionPlay()));
$("#df-first").addEventListener("click", () => { diffusionStop(); diffusionShow(0); });
$("#df-next").addEventListener("click", () => { diffusionStop(); diffusionShow(df.k + 1); });
$("#df-step").addEventListener("input", (e) => { diffusionStop(); diffusionShow(+e.target.value); });

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
  denoiserFlow();
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

// ---------- H2: adding noise ----------
// One real text, one random number u per position: at level t a position is hidden when u < t/T, so raising t only
// adds masks. The server runs all T levels through the model in one batch and returns its guesses and the loss on
// the hidden positions (the training loss ignores the visible ones).
const dn = { r: null };

async function noiseRun() {
  const key = $("#df-prompt").value;
  dn.r = await api("/api/diffusion/noise", { run: $("#df-model").value, prompt_key: key || null,
    prompt: key ? "" : $("#df-own").value, tokens: 48, seed: +$("#df-seed").value });
  const r = dn.r, T = r.T;
  $("#dn-out").classList.remove("hidden");
  $("#dn-t").max = T;
  if (+$("#dn-t").value > T) $("#dn-t").value = Math.round(T / 4);
  $("#dn-note").textContent = `${r.ids.length} tokens · T = ${T}`;
  noiseShow();
}

function noiseShow() {
  const r = dn.r;
  if (!r) return;
  const T = r.T, t = +$("#dn-t").value, lv = t ? r.levels[t - 1] : null, show = $("#dn-guess").checked;
  const hidden = new Set(lv ? lv.masked : []);
  const tok = (id) => tgTok(r.text_of[id] ?? "");
  let right = 0;
  const html = r.ids.map((id, i) => {
    const fresh = t && r.u[i] >= (t - 1) / T && r.u[i] < t / T;
    if (!hidden.has(i)) return `<span class="tg-tok">${tok(id)}</span>`;
    const g = lv.guess[i], ok = g === id;
    if (ok) right++;
    const title = `position ${i + 1}: hidden "${esc(r.text_of[id] ?? "")}" · model's guess "${esc(r.text_of[g] ?? "")}" (${tgPct(lv.guess_p[i])}) · p(real token) ${tgPct(lv.p_true[i])}`;
    if (!show) return `<span class="df-mask${fresh ? " dn-fresh" : ""}" title="${title}">█</span>`;
    return `<span class="tg-tok dn-guess ${ok ? "ok" : "bad"}${fresh ? " dn-fresh" : ""}" title="${title}">${tok(g) || "∅"}</span>`;
  }).join("");
  $("#dn-text").innerHTML = html;
  $("#dn-t-label").textContent = `t = ${t} of ${T} · ${tgPct(t / T)} masked`;
  $("#dn-legend").innerHTML = show
    ? `Hidden positions show the model's guess: <span class="dn-guess ok">right</span> or <span class="dn-guess bad">wrong</span>; outlined ones were hidden at this level. Hover a token for the real one.`
    : `█ = hidden ([MASK]); outlined ones were hidden at this level. Tick “show the model's guesses” to see what it predicts.`;
  const stat = (k, v, d = "") => `<div class="stat"><div class="k">${k}</div><div class="v">${v}</div><div class="d">${d}</div></div>`;
  $("#dn-stats").innerHTML = [
    stat("Noise level", `t = ${t}`, `mask probability t/T = ${(t / T).toFixed(2)}`),
    stat("Hidden", `${hidden.size} of ${r.ids.length}`, "the loss is computed only on these"),
    stat("Loss on hidden tokens", lv?.loss == null ? "–" : lv.loss.toFixed(2), "cross-entropy, nats per hidden token"),
    stat("Guessed right", lv?.correct == null ? "–" : tgPct(lv.correct), lv ? `${right} of ${hidden.size} top guesses` : ""),
  ].join("");
  const marks = t ? [{ i: t - 1, label: `t = ${t}` }] : [];
  $("#dn-loss").innerHTML = lineChart([{ label: "loss on hidden tokens", color: "var(--accent)", values: r.levels.map((l) => l.loss ?? 0) }],
    { xLabel: "noise level t", yLabel: "nats per hidden token", marks }) +
    `<p class="hint">One text, so the line is noisy; it rises with t because there is less left to read. Training averages this over random t and many texts.</p>`;
  $("#dn-sched").innerHTML = lineChart([{ label: "t/T", color: "var(--series-val)", values: r.levels.map((l) => l.ratio) },
    { label: "hidden in this text", color: "var(--accent)", values: r.levels.map((l) => l.masked.length / r.ids.length) }],
  { xLabel: "noise level t", yLabel: "share hidden", yMax: 1, marks }) +
    `<p class="hint">Orange: the linear schedule t/T. Purple: what this text actually got (each token is a coin flip).</p>`;
}

// ---------- H2: fewer steps or more ----------
const dq = { r: null };

async function stepsRun() {
  $("#dq-busy").classList.remove("hidden");
  try {
    const key = $("#df-prompt").value, pr = DF_PRESETS[$("#df-preset").value];
    dq.r = await api("/api/diffusion/steps", { run: $("#df-model").value, prompt_key: key || null,
      prompt: key ? "" : $("#df-own").value, tokens: 64, steps: 64, step_counts: [4, 8, 16, 32, 64],
      sampler: pr.sampler, repetition_penalty: pr.repetition_penalty, block_size: pr.block_size,
      temperature: pr.temperature, top_k: 50, seed: +$("#df-seed").value });
  } finally { $("#dq-busy").classList.add("hidden"); }
  const r = dq.r;
  $("#dq-out").classList.remove("hidden");
  $("#dq-k").max = r.rows.length - 1;
  $("#dq-k").value = r.rows.length - 1;
  $("#dq-note").textContent = `${r.tokens} tokens · judge: ${r.judge}`;
  stepsShow();
}

function stepsShow() {
  const r = dq.r;
  if (!r) return;
  const k = +$("#dq-k").value, row = r.rows[k];
  $("#dq-k-label").textContent = `${row.steps} steps · ${row.per_step} token${row.per_step === 1 ? "" : "s"} per pass`;
  $("#dq-text").innerHTML = `<span class="tg-prompt">${tgTok(r.prompt)}</span>${tgTok(row.completion)}`;
  const stat = (k, v, d = "") => `<div class="stat"><div class="k">${k}</div><div class="v">${v}</div><div class="d">${d}</div></div>`;
  $("#dq-stats").innerHTML = [
    stat("Steps", row.steps, `${row.per_step} token${row.per_step === 1 ? "" : "s"} revealed per pass`),
    stat("Time", `${row.seconds.toFixed(2)} s`, `${(row.seconds * 1000 / row.steps).toFixed(1)} ms per pass`),
    stat("Judge loss", dfFmt(row.judge_loss, 2), r.reference ? `real text: ${dfFmt(r.reference.judge_loss, 2)}` : "lower reads more like real text"),
    stat("Variety", tgPct(row.repetition.distinct_2), `distinct pairs · ${tgPct(row.repetition.repeated_4grams)} repeated 4-token runs`),
  ].join("");
  const labels = r.rows.map((x) => x.steps).join(" · ");
  const marks = [{ i: k, label: `${row.steps} steps` }];
  const series = [{ label: "judge loss", color: "var(--accent)", values: r.rows.map((x) => x.judge_loss ?? 0) }];
  if (r.reference) series.push({ label: "real text", color: "var(--series-val)", values: r.rows.map(() => r.reference.judge_loss) });
  $("#dq-loss").innerHTML = lineChart(series, { x0: 1, xLabel: `steps: ${labels}`, yLabel: "nats per token", marks }) +
    `<p class="hint">${r.reference ? "Orange: the judge loss of the real continuation. " : ""}One prompt and seed, so differences of a few tenths can be luck; a very low loss with heavy repetition is not good text.</p>`;
  $("#dq-time").innerHTML = lineChart([{ label: "seconds", color: "var(--series-train)", values: r.rows.map((x) => x.seconds) }],
    { x0: 1, xLabel: `steps: ${labels}`, yLabel: "seconds", marks }) +
    `<p class="hint">Time grows with the number of passes: each pass reads the whole text, however many tokens it reveals.</p>`;
}

// ---------- H2: inside the denoiser ----------
function denoiserFlow() {
  const run = df.info?.runs.find((x) => x.id === $("#df-model").value) || df.info?.runs[0];
  if (!run) return;
  const c = run.model || {}, gpt = c.arch === "gpt", T = c.diffusion_steps || 64;
  $("#da-note").textContent = `${run.name}: ${c.n_layers ?? "?"} blocks × ${c.d_model ?? "?"} wide, T = ${T}`;
  const box = (x, y, w, h, title, sub, hl) => `<g><rect x="${x}" y="${y}" width="${w}" height="${h}" rx="8" fill="${hl ? "var(--accent-soft)" : "var(--panel-2)"}" stroke="${hl ? "var(--accent)" : "var(--border)"}"/>
    <text x="${x + w / 2}" y="${y + h / 2 - (sub ? 4 : -4)}" text-anchor="middle" class="da-t">${title}</text>${sub ? `<text x="${x + w / 2}" y="${y + h / 2 + 12}" text-anchor="middle" class="da-s">${sub}</text>` : ""}</g>`;
  const arrow = (x1, y1, x2, y2) => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="var(--muted)" stroke-width="1.4" marker-end="url(#da-arr)"/>`;
  // the input row: a few tokens, some masked
  const toks = ["The", "█", "sat", "█", "the", "█"], tw = 52;
  const tokRow = toks.map((w, i) => `<rect x="${20 + i * (tw + 6)}" y="14" width="${tw}" height="26" rx="5" fill="${w === "█" ? "var(--panel-2)" : "var(--panel)"}" stroke="var(--border)"/>
    <text x="${20 + i * (tw + 6) + tw / 2}" y="32" text-anchor="middle" class="${w === "█" ? "da-m" : "da-t"}">${w}</text>`).join("");
  const outRow = ["The", "cat", "sat", "on", "the", "mat"].map((w, i) => `<rect x="${20 + i * (tw + 6)}" y="452" width="${tw}" height="26" rx="5" fill="${toks[i] === "█" ? "var(--accent-soft)" : "var(--panel)"}" stroke="${toks[i] === "█" ? "var(--accent)" : "var(--border)"}"/>
    <text x="${20 + i * (tw + 6) + tw / 2}" y="470" text-anchor="middle" class="da-t">${w}</text>`).join("");
  const W = 760, midX = 186;
  $("#da-flow").innerHTML = `<svg viewBox="0 0 ${W} 520" class="da-svg">
    <defs><marker id="da-arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="var(--muted)"/></marker></defs>
    ${tokRow}<text x="${20 + 6 * (tw + 6) + 6}" y="32" class="da-s">noisy text at level t (█ = [MASK], a token id of its own)</text>
    ${arrow(midX, 42, midX, 66)}
    ${box(12, 68, 110, 46, "token embedding", `${c.vocab_size ?? 50304} × ${c.d_model ?? "d"}`)}
    ${box(130, 68, 104, 46, "+ position", gpt ? "RoPE, in attention" : `learned, ${c.seq_len ?? "L"} slots`)}
    ${box(242, 68, 118, 46, "+ time embedding", `one per t (${T + 1})`, true)}
    ${arrow(midX, 116, midX, 140)}
    <rect x="12" y="142" width="348" height="186" rx="10" fill="none" stroke="var(--border)" stroke-dasharray="4 4"/>
    <text x="22" y="160" class="da-s">× ${c.n_layers ?? "N"} blocks</text>
    ${box(32, 168, 308, 58, "self-attention, bidirectional", `${c.n_heads ?? "h"} heads · each position reads all the others`, true)}
    ${arrow(midX, 228, midX, 250)}
    ${box(32, 252, 308, 58, gpt ? "SwiGLU MLP" : "MLP (GELU)", `${c.d_model ?? "d"} → ${c.d_ff ?? (gpt ? "SwiGLU" : "4d")} → ${c.d_model ?? "d"}, with residual connections`)}
    ${arrow(midX, 330, midX, 354)}
    ${box(32, 356, 308, 50, "final norm → output layer", "tied to the token embedding: a score per token")}
    ${arrow(midX, 408, midX, 448)}
    <text x="${midX + 8}" y="432" class="da-s">a distribution per position</text>
    ${outRow}
    ${box(400, 356, 340, 50, "loss: cross-entropy on the hidden positions only", "visible tokens get label −100 and add nothing", true)}
    <line x1="372" y1="465" x2="398" y2="408" stroke="var(--accent)" stroke-width="1.4" stroke-dasharray="3 3"/>
    <text x="410" y="428" class="da-s">only the highlighted outputs (cat, on, mat) are scored;</text>
    <text x="410" y="444" class="da-s">the bound weights each level's loss by 1/(t/T) and averages</text>
    <text x="410" y="460" class="da-s">over all tokens (diffusion_models/noising.py)</text>
    <text x="400" y="100" class="da-s">What differs from the autoregressive GPT:</text>
    <text x="400" y="122" class="da-s">• the input has [MASK] tokens instead of a cut-off text</text>
    <text x="400" y="142" class="da-s">• a time embedding says how much is hidden (t)</text>
    <text x="400" y="162" class="da-s">• attention has no causal mask: it reads both ways</text>
    <text x="400" y="182" class="da-s">• it predicts the hidden tokens themselves, not the next one</text>
    <text x="400" y="202" class="da-s">• generation runs it once per step, revealing a few tokens</text>
    <text x="400" y="222" class="da-s">  each time, instead of once per token</text>
  </svg>`;
  // causal vs bidirectional attention, 8 positions
  const n = 8, s = 18, grid = (full) => {
    let cells = "";
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      const on = full || j <= i;
      cells += `<rect x="${j * s}" y="${i * s}" width="${s - 2}" height="${s - 2}" rx="2" fill="${on ? "var(--accent)" : "var(--panel-2)"}" opacity="${on ? .7 : 1}" stroke="var(--border)"/>`;
    }
    return `<svg viewBox="-4 -4 ${n * s + 8} ${n * s + 8}" class="da-grid">${cells}</svg>`;
  };
  $("#da-attn").innerHTML = `<div class="two-col">
    <div><div class="card-head"><h4>Autoregressive GPT: causal attention</h4></div>${grid(false)}
      <p class="hint">Row = the position reading, column = the position read. Each position sees only itself and the past, so the model can only predict what comes next.</p></div>
    <div><div class="card-head"><h4>Diffusion denoiser: bidirectional attention</h4></div>${grid(true)}
      <p class="hint">Every position sees the whole text, masked or not. A hidden word in the middle can use the words after it, which is why the model can fill gaps in any order.${gpt ? " This model is the base GPT's own block code with the causal mask turned off (causal=False)." : ""}</p></div>
  </div>`;
}

$("#dn-run").addEventListener("click", (e) => action(e.target, noiseRun));
$("#dn-t").addEventListener("input", noiseShow);
$("#dn-guess").addEventListener("change", noiseShow);
$("#dq-run").addEventListener("click", (e) => action(e.target, stepsRun));
$("#dq-k").addEventListener("input", stepsShow);
$("#df-model").addEventListener("change", () => { denoiserFlow(); dn.r = null; dq.r = null; $("#dn-out").classList.add("hidden"); $("#dq-out").classList.add("hidden"); });

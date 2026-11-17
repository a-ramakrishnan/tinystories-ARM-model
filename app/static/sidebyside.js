// "Side by side" page (H1): the same prompt written by an autoregressive model (one token per pass, left to right)
// and a masked diffusion model (every position at once, a few revealed per pass), replayed together, with both
// scored the same way. Uses tgTok / tgShade / tgPct from textgen.js.

const sbs = { info: null, r: null, s: 0, timer: null };
const SBS_ORDERS = {
  blocks: { label: "blocks of 16, left to right, penalty 1.2 (recommended)", sampler: "keep", repetition_penalty: 1.2, block_size: 16 },
  whole: { label: "the whole text at once, confident first, penalty 1.2", sampler: "keep", repetition_penalty: 1.2, block_size: 0 },
  random: { label: "the whole text at once, random order", sampler: "random", repetition_penalty: 1.0, block_size: 0 },
};
const SBS_TICKS = 240;            // slider positions when lined up by clock time
const sbsM = (v) => (v == null ? "–" : v >= 1e9 ? `${(v / 1e9).toFixed(1)}B` : `${Math.round(v / 1e6)}M`);
const sbsF = (v, d = 3) => (v == null ? "–" : (+v).toFixed(d));

async function sbsOpen() {
  sbs.info = await (await fetch("/api/sidebyside")).json();
  const i = sbs.info, keep = (sel, opts, fallback) => {
    const prev = sel.value;
    sel.innerHTML = opts;
    sel.value = prev && [...sel.options].some((o) => o.value === prev) ? prev : fallback;
  };
  keep($("#sbs-ar"), i.models.map((m) => `<option value="${esc(m.id)}">${m.id === i.base ? "Base model: " : ""}${esc(m.name)}</option>`).join(""), i.base);
  keep($("#sbs-run"), i.runs.map((r) => `<option value="${esc(r.id)}">${esc(r.name)}${r.best_bound ? ` · bound ${r.best_bound.toFixed(3)}` : ""}</option>`).join(""), i.default_run);
  keep($("#sbs-prompt"), i.prompts.map((p) => `<option value="${esc(p.key)}">${esc(p.source)}: ${esc(p.prompt.slice(0, 48).replace(/\s+/g, " "))}…</option>`).join("") +
    '<option value="">Your own prompt (no real continuation)</option>', i.prompts.find((p) => p.source !== "TinyStories")?.key || "");
  keep($("#sbs-order"), Object.entries(SBS_ORDERS).map(([k, v]) => `<option value="${k}">${esc(v.label)}</option>`).join(""), "blocks");
  $("#sbs-own-wrap").classList.toggle("hidden", $("#sbs-prompt").value !== "");
  $("#sbs-empty").classList.toggle("hidden", i.runs.length > 0 && i.models.length > 0);
  $("#sbs-main").classList.toggle("hidden", !(i.runs.length > 0 && i.models.length > 0));
  sbsHow();
}

async function sbsRun() {
  sbsStop();
  $("#sbs-busy").classList.remove("hidden");
  try {
    const key = $("#sbs-prompt").value, o = SBS_ORDERS[$("#sbs-order").value];
    sbs.r = await api("/api/sidebyside", { ar_model: $("#sbs-ar").value || null, run: $("#sbs-run").value || null,
      prompt_key: key || null, prompt: key ? "" : $("#sbs-own").value, tokens: +$("#sbs-tokens").value,
      steps: +$("#sbs-steps").value, ar_temperature: +$("#sbs-ardecode").value, sampler: o.sampler,
      repetition_penalty: o.repetition_penalty, block_size: o.block_size, seed: 1 });
  } finally { $("#sbs-busy").classList.add("hidden"); }
  sbsShowRun();
}

// draw a result (from the server, or saved: the website replays saved runs with this same code)
function sbsShowRun() {
  const d = sbs.r.diffusion;
  // each diffusion position: the pass that revealed it (its last change) and the model's confidence then
  d.revealedAt = Array(d.gen_len).fill(null);
  d.revealConf = Array(d.gen_len).fill(0);
  d.trace.forEach((st, k) => st.ids.forEach((id, i) => {
    const before = k ? d.trace[k - 1].ids[i] : d.mask_id;
    if (id !== d.mask_id && id !== before) { d.revealedAt[i] = k + 1; d.revealConf[i] = st.conf[i]; }
    if (id === d.mask_id) d.revealedAt[i] = null;
  }));
  $("#sbs-out").classList.remove("hidden");
  $("#sbs-note").textContent = `${sbs.r.device} · ${sbs.r.prompt_tokens}-token prompt · ${sbs.r.tokens} new tokens`;
  sbsNumbers();
  sbsHow();
  $("#sbs-ref").innerHTML = sbs.r.reference ? `<div class="card-head"><h4>The real continuation</h4><span class="muted small">${esc(sbs.r.source)}</span></div><div class="tg-text small">${tgTok(sbs.r.reference)}</div>` : "";
  sbsSlider();
  sbsShow(0);
  sbsPlay();
}

// ---------- the shared player: by model pass, or by clock time ----------
function sbsByTime() { return $("#sbs-align").value === "time"; }
function sbsSlider() {
  const r = sbs.r;
  $("#sbs-step").max = sbsByTime() ? SBS_TICKS : Math.max(r.ar.passes, r.diffusion.passes);
}
function sbsAt(s) {
  // how many passes each model has done at slider position s
  const r = sbs.r, a = r.ar, d = r.diffusion;
  if (!sbsByTime()) return { ar: Math.min(s, a.passes), df: Math.min(s, d.passes), label: `pass ${s}` };
  const tau = (s / SBS_TICKS) * Math.max(a.ms, d.ms);
  return { ar: a.steps.filter((x) => x.ms <= tau).length, df: d.trace.filter((x) => x.ms <= tau).length,
    label: `${(tau / 1000).toFixed(2)} s` };
}

function sbsShow(s) {
  const r = sbs.r;
  if (!r) return;
  sbs.s = Math.max(0, Math.min(s, +$("#sbs-step").max));
  $("#sbs-step").value = sbs.s;
  const at = sbsAt(sbs.s), a = r.ar, d = r.diffusion;
  $("#sbs-step-label").textContent = at.label;
  // autoregressive lane: the tokens written so far, then empty slots
  $("#sbs-ar-text").innerHTML = `<span class="tg-prompt">${tgTok(r.prompt)}</span>` + a.steps.map((x, i) => i < at.ar
    ? `<span class="tg-tok${i === at.ar - 1 ? " cur" : ""}" style="background:${tgShade(x.prob)}" title="pass ${i + 1} · p = ${x.prob}">${tgTok(x.text)}</span>`
    : '<span class="tg-tok empty"> </span>').join("");
  $("#sbs-ar-state").innerHTML = `<b>${at.ar}</b> of ${a.passes} passes · <b>${at.ar}</b> token${at.ar === 1 ? "" : "s"} written`;
  // diffusion lane: every position, masked (█) until revealed
  const ids = at.df ? d.trace[at.df - 1].ids : Array(d.gen_len).fill(d.mask_id);
  const prev = at.df > 1 ? d.trace[at.df - 2].ids : Array(d.gen_len).fill(d.mask_id);
  const conf = at.df ? d.trace[at.df - 1].conf : [];
  const left = ids.filter((x) => x === d.mask_id).length;
  $("#sbs-df-text").innerHTML = `<span class="tg-prompt">${tgTok(r.prompt)}</span>` + ids.map((id, i) => {
    if (id === d.mask_id) return '<span class="df-mask">█</span>';
    const fresh = id !== prev[i];
    return `<span class="tg-tok${fresh ? " cur" : ""}" style="background:${tgShade(fresh ? conf[i] : d.revealConf[i] || .5)}" title="position ${i + 1}">${tgTok(d.text_of[id] ?? "")}</span>`;
  }).join("");
  $("#sbs-df-state").innerHTML = `<b>${at.df}</b> of ${d.passes} passes · <b>${d.gen_len - left}</b> of ${d.gen_len} positions revealed`;
  sbsGrids(at);
}

function sbsGrids(at) {
  // a cell per (pass, position), the same row height in both lanes, so the heights compare the number of passes
  const r = sbs.r, a = r.ar, d = r.diffusion, W = 300, rows = Math.max(a.passes, d.passes), rh = Math.max(1.5, 300 / rows);
  const svg = (n, passes, cells, k) => `<svg viewBox="-40 -16 ${W + 46} ${passes * rh + 22}">
    <rect x="0" y="0" width="${W}" height="${passes * rh}" fill="none" stroke="var(--border)"/>${cells}
    ${k ? `<line x1="0" x2="${W}" y1="${k * rh}" y2="${k * rh}" stroke="var(--accent)" stroke-dasharray="3 3"/>` : ""}
    <text x="0" y="-5">position 1</text><text x="${W}" y="-5" text-anchor="end">${n}</text>
    <text x="-5" y="${Math.min(rh, passes * rh) + 3}" text-anchor="end">pass 1</text><text x="-5" y="${passes * rh}" text-anchor="end">${passes}</text></svg>`;
  let ca = "";
  const cw = W / a.passes;
  for (let i = 0; i < at.ar; i++) {
    ca += `<rect x="0" y="${i * rh}" width="${(i + 1) * cw}" height="${rh}" fill="var(--accent)" opacity=".13"/>`;
    ca += `<rect x="${i * cw}" y="${i * rh}" width="${cw}" height="${rh}" fill="${tgShade(a.steps[i].prob)}"><title>pass ${i + 1} writes position ${i + 1}: ${esc(a.steps[i].text)}</title></rect>`;
  }
  $("#sbs-ar-grid").innerHTML = svg(a.passes, a.passes, ca, at.ar);
  let cd = "";
  const dw = W / d.gen_len;
  for (let i = 0; i < d.gen_len; i++) {
    const t = d.revealedAt[i];
    if (t == null || t > at.df) continue;
    cd += `<rect x="${i * dw}" y="${(t - 1) * rh}" width="${dw}" height="${(at.df - t + 1) * rh}" fill="var(--accent)" opacity=".13"/>`;
    cd += `<rect x="${i * dw}" y="${(t - 1) * rh}" width="${dw}" height="${rh}" fill="${tgShade(d.revealConf[i])}"><title>position ${i + 1} revealed at pass ${t}</title></rect>`;
  }
  $("#sbs-df-grid").innerHTML = svg(d.gen_len, d.passes, cd, at.df);
}

function sbsPlay() {
  sbsStop();
  if (!sbs.r) return;
  if (sbs.s >= +$("#sbs-step").max) sbsShow(0);
  $("#sbs-play").textContent = "⏸";
  const tick = () => {
    if (sbs.s >= +$("#sbs-step").max) { sbsStop(); return; }
    sbsShow(sbs.s + 1);
    sbs.timer = setTimeout(tick, +$("#sbs-speed").value * (sbsByTime() ? 0.5 : 1));
  };
  sbs.timer = setTimeout(tick, 150);
}
function sbsStop() { clearTimeout(sbs.timer); sbs.timer = null; $("#sbs-play").textContent = "▶"; }

// ---------- the numbers ----------
function sbsNumbers() {
  const r = sbs.r, a = r.ar, d = r.diffusion;
  const better = (x, y, low = true) => (x == null || y == null || x === y ? ["", ""] : (low ? x < y : x > y) ? ["sbs-win", ""] : ["", "sbs-win"]);
  const row = (label, x, y, fmt, low = true, hint = "") => {
    const [cx, cy] = low === null ? ["", ""] : better(x, y, low);
    return `<tr><td>${label}${hint ? `<div class="muted small">${hint}</div>` : ""}</td><td class="num ${cx}">${x == null ? "–" : fmt(x)}</td><td class="num ${cy}">${y == null ? "–" : fmt(y)}</td></tr>`;
  };
  const ms = (v) => `${(v / 1000).toFixed(2)} s`;
  $("#sbs-numbers").innerHTML = `<div class="ptable-scroll"><table class="ptable">
    <tr><th></th><th class="num">Autoregressive<div class="muted small">${esc(a.name.split(" · ").slice(0, 2).join(" · "))}</div></th>
      <th class="num">Diffusion<div class="muted small">${esc(d.name)}</div></th></tr>
    <tr><td colspan="3" class="sbs-sec">This text</td></tr>
    ${row("Model passes", a.passes, d.passes, (v) => v, true, `for ${r.tokens} tokens`)}
    ${row("Time", a.ms, d.ms, ms, true, "on this Mac, including the steps' bookkeeping")}
    ${row("Tokens per second", a.tokens_per_sec, d.tokens_per_sec, (v) => v.toFixed(0), false)}
    ${row("Time per pass", a.ms / a.passes, d.ms / d.passes, (v) => `${v.toFixed(1)} ms`, true, "the autoregressive model reuses its KV cache; the diffusion model re-reads every position")}
    ${row("Judge loss", a.judge_loss, d.judge_loss, sbsF, true, `the base model's loss on each generated text${r.reference_judge_loss != null ? ` (real text: ${sbsF(r.reference_judge_loss)})` : ""}${a.id === r.judge ? "; the judge is the autoregressive model itself, so its own text scores low" : ""}; looping text scores low too, so read it with repetition`)}
    ${row("Repeated 4-token runs", a.repeated_4grams, d.repeated_4grams, tgPct, true)}
    ${row("Distinct pairs", a.distinct_2, d.distinct_2, tgPct, false)}
    ${row("How it chose", a.decoding, `${d.decoding}, ${d.passes} passes`, (v) => esc(v), null)}
    ${r.reference ? `<tr><td colspan="3" class="sbs-sec">The real continuation</td></tr>
    ${row("Loss on it (nats per token)", a.ref_loss, d.ref_bound, sbsF, true, "exact for the autoregressive model; for diffusion the likelihood bound (32 maskings), an upper limit on its real loss")}` : ""}
    <tr><td colspan="3" class="sbs-sec">The models</td></tr>
    ${row("Parameters", a.params, d.params, sbsM, null)}
    ${row("Training tokens", a.tokens_trained, d.tokens_trained, sbsM, null)}
    ${row("Layers × width · context", `${a.layers} × ${a.width} · ${a.context}`, `${d.layers} × ${d.width} · ${d.context}`, (v) => v, null)}
    ${row("Validation, training mix", a.best_val, d.best_bound, sbsF, r.same_data ? true : null, r.same_data ? "the same validation text: loss vs likelihood bound" : "different training data: not comparable")}
  </table></div>`;
}

// ---------- how the two differ ----------
function sbsHow() {
  const r = sbs.r, n = 8, s = 16;
  const grid = (full) => {
    let c = "";
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      const on = full || j <= i;
      c += `<rect x="${j * s}" y="${i * s}" width="${s - 2}" height="${s - 2}" rx="2" fill="${on ? "var(--accent)" : "var(--panel-2)"}" opacity="${on ? .7 : 1}" stroke="var(--border)"/>`;
    }
    return `<svg viewBox="-4 -4 ${n * s + 8} ${n * s + 8}" class="sbs-attn">${c}</svg>`;
  };
  const a = r?.ar, d = r?.diffusion;
  const L = r ? r.tokens : 64, K = d ? d.passes : 32;
  $("#sbs-how").innerHTML = `<div class="ptable-scroll"><table class="ptable sbs-how">
    <tr><th></th><th>Autoregressive</th><th>Masked diffusion</th></tr>
    <tr><td>Attention</td><td>${grid(false)}<div class="muted small">causal: each position reads only itself and the past</div></td>
      <td>${grid(true)}<div class="muted small">bidirectional: every position reads the whole text, masked or not</div></td></tr>
    <tr><td>Training</td><td>predict the <b>next token</b> at every position: loss on all of them</td>
      <td>mask a random share t of the tokens, predict the <b>hidden ones</b> from everything else: loss on the masked positions only, t drawn anew for every text</td></tr>
    <tr><td>Writing ${L} tokens</td><td>${L} passes, one token each, in reading order; a token, once written, never changes</td>
      <td>${K} passes; each predicts every masked position and reveals about ${(L / K).toFixed(1)}, the most confident first; fewer passes are faster but decide more at once</td></tr>
    <tr><td>Cost of a pass</td><td>small: the KV cache keeps what was read, so a pass reads just one new token</td>
      <td>large: no cache, every pass reads the prompt and all ${L} positions again</td></tr>
    <tr><td>Its loss</td><td>exact: the probability of the text, token by token</td>
      <td>a bound: averaged over masking levels, an upper limit on the real loss, so the gap to the autoregressive loss is at most what it shows</td></tr>
    ${a && d ? `<tr><td>These two</td><td>${a.layers} layers × ${a.width}, ${sbsM(a.params)} parameters, ${sbsM(a.tokens_trained)} tokens</td>
      <td>${d.layers} layers × ${d.width}, ${sbsM(d.params)} parameters, ${sbsM(d.tokens_trained)} tokens; the same blocks with the causal mask off, plus a noise-level embedding</td></tr>` : ""}
  </table></div>`;
}

$("#sbs-prompt")?.addEventListener("change", (e) => $("#sbs-own-wrap").classList.toggle("hidden", e.target.value !== ""));
$("#sbs-go")?.addEventListener("click", (e) => action(e.target, sbsRun));
$("#sbs-play").addEventListener("click", () => (sbs.timer ? sbsStop() : sbsPlay()));
$("#sbs-first").addEventListener("click", () => { sbsStop(); sbsShow(0); });
$("#sbs-next").addEventListener("click", () => { sbsStop(); sbsShow(sbs.s + 1); });
$("#sbs-step").addEventListener("input", (e) => { sbsStop(); sbsShow(+e.target.value); });
$("#sbs-align").addEventListener("change", () => { if (!sbs.r) return; sbsStop(); sbsSlider(); sbsShow(0); });

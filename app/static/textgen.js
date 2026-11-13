// "Writing text" page: how the model writes a continuation, one token per step, left to right, on fixed held-out
// prompts (textgen.py). A player steps through the generation; the grid shows which position each step writes.

const tg = { info: null, run: null, k: 0, timer: null };
const tgShade = (p) => `color-mix(in srgb, var(--accent) ${Math.round(12 + 70 * p)}%, transparent)`;
const tgHeat = (nats) => `color-mix(in srgb, var(--bad) ${Math.round(Math.min(1, nats / 8) * 70)}%, transparent)`;
const tgTok = (t) => esc(t).replace(/\n/g, "↵\n");
const tgPct = (v) => (v == null ? "–" : `${(v * 100).toFixed(1)}%`);

async function textgenOpen() {
  tg.info = await (await fetch("/api/textgen")).json();
  const ms = $("#tg-model"), prev = ms.value;
  ms.innerHTML = tg.info.models.map((m) => `<option value="${esc(m.id)}">${m.id === tg.info.base ? "Base model: " : ""}${esc(m.name)}</option>`).join("");
  ms.value = prev && tg.info.models.some((m) => m.id === prev) ? prev : tg.info.base || "";
  const ps = $("#tg-prompt"), pp = ps.value;
  ps.innerHTML = tg.info.prompts.map((p) => `<option value="${esc(p.key)}">${esc(p.source)}: ${esc(p.prompt.slice(0, 48).replace(/\s+/g, " "))}…</option>`).join("") +
    '<option value="">Your own prompt (no real continuation)</option>';
  if (ps.dataset.set) ps.value = pp;  // keep the choice when the page is reopened (the first option otherwise)
  ps.dataset.set = "1";
  $("#tg-own-wrap").classList.toggle("hidden", ps.value !== "");
  textgenSummary();
}

// ---------- one generation ----------
async function textgenRun() {
  textgenStop();
  $("#tg-busy").classList.remove("hidden");
  try {
    const key = $("#tg-prompt").value;
    tg.run = await api("/api/textgen/trace", { model_id: $("#tg-model").value || null, prompt_key: key || null,
      prompt: key ? "" : $("#tg-own").value, tokens: +$("#tg-tokens").value, temperature: +$("#tg-decode").value });
  } finally { $("#tg-busy").classList.add("hidden"); }
  const r = tg.run, n = r.steps.length;
  $("#tg-out").classList.remove("hidden");
  $("#tg-step").max = n;
  $("#tg-model-note").textContent = `${r.model_name.split(" · ")[0]} · ${r.device}`;
  textgenMetrics();
  $("#tg-prob").innerHTML = lineChart([{ label: "probability", color: "var(--accent)", values: r.steps.map((s) => s.prob) }], { xLabel: "step", yLabel: "probability", yMax: 1 });
  const per = r.steps.map((s, i) => s.ms - (i ? r.steps[i - 1].ms : 0));
  $("#tg-ms").innerHTML = lineChart([{ label: "ms", color: "var(--series-val)", values: per }], { xLabel: "step", yLabel: "ms" });
  textgenRef();
  textgenShow(0);
  textgenPlay();
}

function textgenShow(k) {
  const r = tg.run;
  if (!r) return;
  const n = r.steps.length;
  tg.k = Math.max(0, Math.min(k, n));
  $("#tg-step").value = tg.k;
  $("#tg-step-label").textContent = `step ${tg.k} of ${n}`;
  // the text: the prompt, the tokens written so far (shaded by confidence), and empty slots still to write
  const slots = r.steps.map((s, i) => i < tg.k
    ? `<span class="tg-tok${i === tg.k - 1 ? " cur" : ""}" style="background:${tgShade(s.prob)}" title="step ${s.step} · p = ${s.prob}">${tgTok(s.text)}</span>`
    : '<span class="tg-tok empty"> </span>').join("");
  $("#tg-text").innerHTML = `<span class="tg-prompt">${tgTok(r.prompt)}</span>${slots}`;
  // the current step
  const s = r.steps[tg.k - 1];
  $("#tg-stepcard").innerHTML = s
    ? `<div class="muted small">Step ${s.step} writes position ${s.pos + 1}</div>
       <div class="tg-chosen"><span class="tg-tok cur" style="background:${tgShade(s.prob)}">${tgTok(s.text)}</span></div>
       <div class="muted small">one pass through the model, reading ${r.prompt_tokens + s.pos} tokens · ${(s.ms - (s.step > 1 ? r.steps[s.step - 2].ms : 0)).toFixed(1)} ms</div>
       <div class="kvm-bars">${s.alternatives.map((a) => `<div class="kvm-row tg-alt"><span class="mono" title="${esc(a.text)}">${tgTok(a.text) || "∅"}</span><div class="kvm-track"><div class="kvm-bar ${a.text === s.text ? "c" : ""}" style="width:${Math.max(1, a.prob * 100)}%;${a.text === s.text ? "" : "background:var(--series-step)"}"></div></div><b>${tgPct(a.prob)}</b></div>`).join("")}</div>
       <div class="muted small">the model's five most likely next tokens${+$("#tg-decode").value ? "; sampling may pick another" : "; greedy takes the top one"}</div>`
    : `<div class="muted small">Step 0</div><p>Nothing written yet: ${n} positions to fill, so ${n} steps.</p>`;
  textgenGrid();
}

function textgenGrid() {
  // a cell per (step, position): filled once that position is written; the cell written at each step is outlined
  const r = tg.run, n = r.steps.length, W = 360, c = W / n;
  let cells = "";
  for (let st = 0; st < Math.min(tg.k, n); st++) {
    cells += `<rect x="0" y="${st * c}" width="${(st + 1) * c}" height="${c}" fill="var(--accent)" opacity=".13"/>`;
    cells += `<rect x="${st * c}" y="${st * c}" width="${c}" height="${c}" fill="${tgShade(r.steps[st].prob)}"${st === tg.k - 1 ? ' stroke="var(--accent)" stroke-width="1.5"' : ""}><title>step ${st + 1} writes position ${st + 1}: ${esc(r.steps[st].text)}</title></rect>`;
  }
  $("#tg-grid").innerHTML = `<svg viewBox="-34 -16 ${W + 40} ${W + 22}">
    <rect x="0" y="0" width="${W}" height="${W}" fill="none" stroke="var(--border)"/>${cells}
    <text x="0" y="-5">position 1</text><text x="${W}" y="-5" text-anchor="end">${n}</text>
    <text x="-5" y="${c}" text-anchor="end">step 1</text><text x="-5" y="${W}" text-anchor="end">${n}</text></svg>
    <p class="hint">Left to right: step <i>k</i> always writes position <i>k</i>, so the filled cells form a staircase and ${n} tokens take ${n} steps.</p>`;
}

function textgenMetrics() {
  const r = tg.run, stat = (k, v, d = "") => `<div class="stat"><div class="k">${k}</div><div class="v">${v}</div><div class="d">${d}</div></div>`;
  $("#tg-metrics").innerHTML = [
    stat("Steps (model passes)", r.model_passes, `for ${r.tokens} tokens`),
    stat("Time", `${(r.total_ms / 1000).toFixed(2)} s`, `${r.ms_per_token ?? "–"} ms per token after the first`),
    stat("Speed", r.tokens_per_sec ? `${r.tokens_per_sec} tok/s` : "–", "writing, after reading the prompt"),
    stat("Confidence", tgPct(r.avg_prob), "average probability of the chosen tokens"),
    stat("Variety", r.distinct_2 == null ? "–" : tgPct(r.distinct_2), `distinct pairs · ${tgPct(r.repeated_4grams)} repeated 4-token runs`),
    r.ref_loss != null ? stat("Loss on the real continuation", r.ref_loss.toFixed(3), `perplexity ${r.ref_ppl} · top-1 ${tgPct(r.ref_top1)}`) : "",
  ].join("");
}

function textgenRef() {
  const r = tg.run;
  if (!r.reference) { $("#tg-ref").innerHTML = ""; return; }
  const toks = r.reference_tokens || [];
  $("#tg-ref").innerHTML = `<div class="card-head"><h4>The real continuation</h4><span class="muted small">shaded by how surprised the model was by each real token</span></div>
    <div class="tg-text">${toks.map((t, i) => `<span class="tg-tok" style="background:${tgHeat(r.ref_token_loss[i])}" title="loss ${r.ref_token_loss[i]} nats">${tgTok(t)}</span>`).join("")}</div>
    <p class="hint">The darker the token, the less the model expected it. Averaged, that's the loss above (${r.ref_loss.toFixed(3)} nats).</p>`;
}

// ---------- player ----------
function textgenPlay() {
  textgenStop();
  const r = tg.run;
  if (!r) return;
  if (tg.k >= r.steps.length) textgenShow(0);
  $("#tg-play").textContent = "⏸";
  const tick = () => {
    if (tg.k >= r.steps.length) { textgenStop(); return; }
    textgenShow(tg.k + 1);
    const sp = +$("#tg-speed").value;
    // real time: wait as long as that step actually took
    const wait = sp || Math.max(4, (r.steps[tg.k]?.ms ?? 0) - (r.steps[tg.k - 1]?.ms ?? 0));
    tg.timer = setTimeout(tick, wait);
  };
  tg.timer = setTimeout(tick, 150);
}
function textgenStop() { clearTimeout(tg.timer); tg.timer = null; $("#tg-play").textContent = "▶"; }

// ---------- the fixed prompt set ----------
function textgenSummary() {
  const done = tg.info.models.filter((m) => m.results);
  if (!done.length) {
    $("#tg-summary").innerHTML = '<p class="muted">No results yet. Pick a model above and press <b>Run the prompt set</b>.</p>';
    $("#tg-prompts").innerHTML = "";
    return;
  }
  const rows = [["Steps (model passes) per text", "model_passes", (v) => v.toFixed(0)],
    ["Loss on the real continuation", "ref_loss", (v) => v.toFixed(3)], ["Perplexity", "ref_ppl", (v) => v.toFixed(1)],
    ["Top-1 on the real text", "ref_top1", tgPct], ["Confidence", "avg_prob", tgPct],
    ["Speed (tokens/s)", "tokens_per_sec", (v) => v.toFixed(0)], ["Distinct pairs", "distinct_2", tgPct],
    ["Repeated 4-token runs", "repeated_4grams", tgPct]];
  $("#tg-summary").innerHTML = `<div class="ptable-scroll"><table class="ptable"><tr><th>Average over ${done[0].results.prompts.length} prompts</th>${done.map((m) => `<th class="num">${esc(m.name.split(" · ")[0])}<div class="muted small">${m.id === tg.info.base ? "base model · " : ""}${esc(m.results.measured_at)}</div></th>`).join("")}</tr>
    ${rows.map(([label, k, f]) => `<tr><td>${label}</td>${done.map((m) => { const v = m.results.summary[k]; return `<td class="num">${v == null ? "–" : f(v)}</td>`; }).join("")}</tr>`).join("")}</table></div>`;
  const m = done.find((x) => x.id === $("#tg-model").value) || done[0];
  $("#tg-prompts").innerHTML = `<details class="tg-set"><summary class="muted small">The ${m.results.prompts.length} prompts, with ${esc(m.name.split(" · ")[0])}'s continuation and the real one</summary>
    ${m.results.prompts.map((p) => `<div class="tg-setrow"><div class="muted small">${esc(p.source)} · loss ${p.ref_loss?.toFixed(3) ?? "–"} · repeated 4-token runs ${tgPct(p.repeated_4grams)}</div>
      <div class="tg-text small"><span class="tg-prompt">${tgTok(p.prompt)}</span><span class="tg-gen">${tgTok(p.text)}</span></div>
      <div class="tg-text small muted"><b>Real:</b> ${tgTok(p.reference)}</div></div>`).join("")}</details>`;
}

$("#tg-prompt").addEventListener("change", (e) => $("#tg-own-wrap").classList.toggle("hidden", e.target.value !== ""));
$("#tg-model").addEventListener("change", () => tg.info && textgenSummary());
$("#tg-run").addEventListener("click", (e) => action(e.target, textgenRun));
$("#tg-play").addEventListener("click", () => (tg.timer ? textgenStop() : textgenPlay()));
$("#tg-first").addEventListener("click", () => { textgenStop(); textgenShow(0); });
$("#tg-next").addEventListener("click", () => { textgenStop(); textgenShow(tg.k + 1); });
$("#tg-step").addEventListener("input", (e) => { textgenStop(); textgenShow(+e.target.value); });
$("#tg-runset").addEventListener("click", (e) => action(e.target, async () => {
  await api("/api/textgen/run", { model_id: $("#tg-model").value || null });
  tg.waiting = true;
}));
setInterval(() => {
  const j = ui.status?.job;
  if (tg.waiting && j && j.kind === "textgen" && j.status !== "running") { tg.waiting = false; textgenOpen(); }
}, 1500);

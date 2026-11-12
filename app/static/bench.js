// Benchmarks page: standard zero-shot tests (HellaSwag, LAMBADA, ARC-Easy, PIQA, Winogrande) for saved models,
// against OpenAI GPT-2 small's published scores. Data from /api/benchmarks (benchmarks.py).

const bench = { data: null, picked: new Set(), tasks: new Set() };
const bpct = (v) => (v == null ? "–" : `${(v * 100).toFixed(1)}%`);
const BENCH_COLORS = ["#5b5bf0", "#e0662f", "#2f9e6b", "#c0399a", "#3f7be0", "#b8862c"];

async function benchOpen() {
  bench.data = await (await fetch("/api/benchmarks")).json();
  if (!bench.tasks.size) bench.data.tasks.forEach((t) => bench.tasks.add(t.key));
  benchRender();
}

function benchMetricOf(task, res) {
  // the number to compare on: the task's headline metric if present, else accuracy
  if (!res) return null;
  return res[task.metric] ?? res.acc ?? null;
}

function benchRender() {
  const d = bench.data;
  if (!d) return;
  const tasks = d.tasks;
  const done = d.models.filter((m) => m.benchmarks?.results && Object.keys(m.benchmarks.results).length);
  // ---- the tests ----
  $("#bn-tasks").innerHTML = tasks.map((t) => `<div class="bn-task">
      <div class="bn-ttitle">${esc(t.title)}</div>
      <p>${esc(t.what)}</p>
      <div class="muted small">${esc(t.data)}</div>
      <div class="muted small">Chance: <b>${t.key === "lambada" ? "≈0%" : bpct(t.chance)}</b> · OpenAI GPT-2 small (published): <b>${Object.entries(t.gpt2).map(([k, v]) => `${k === "ppl" ? `perplexity ${v}` : `${k} ${bpct(v)}`}`).join(", ")}</b></div>
    </div>`).join("");
  // ---- results table: a row per task, a column per benchmarked model, plus GPT-2's published numbers ----
  if (!done.length) {
    $("#bn-results").innerHTML = '<p class="muted">No results yet. Pick a model below and press <b>Run benchmarks</b>.</p>';
    $("#bn-chart").innerHTML = "";
  } else {
    const head = `<tr><th>Benchmark</th><th class="num">Chance</th><th class="num">GPT-2 small<div class="muted small">published</div></th>${done.map((m, i) => `<th class="num"><span class="sw-dot" style="background:${BENCH_COLORS[i % BENCH_COLORS.length]}"></span>${esc(m.name.split(" · ")[0])}<div class="muted small">${short(m.params || 0)} · ${esc(m.benchmarks.measured_at || "")}</div></th>`).join("")}</tr>`;
    const rows = tasks.flatMap((t) => {
      const metrics = t.key === "lambada" ? ["acc", "ppl"] : t.key === "winogrande" ? ["acc"] : ["acc", "acc_norm"];
      return metrics.map((k, mi) => {
        const vals = done.map((m) => m.benchmarks.results[t.key]?.[k]);
        const valid = vals.filter((v) => v != null);
        const best = valid.length > 1 ? (k === "ppl" ? Math.min(...valid) : Math.max(...valid)) : null;
        const fmtV = (v) => (v == null ? "–" : k === "ppl" ? v.toFixed(1) : bpct(v));
        const label = k === "acc" ? "accuracy" : k === "acc_norm" ? "accuracy, length-normalized" : "perplexity of the right word (lower is better)";
        return `<tr class="${mi === 0 ? "bn-first" : ""}"><td>${mi === 0 ? `<b>${esc(t.title)}</b> ` : ""}<span class="muted small">${label}${k === t.metric ? " ★" : ""}</span></td>
          <td class="num muted">${k === "ppl" ? "–" : t.key === "lambada" ? "≈0%" : bpct(t.chance)}</td>
          <td class="num muted">${t.gpt2[k] != null ? fmtV(t.gpt2[k]) : "–"}</td>
          ${done.map((m) => { const r = m.benchmarks.results[t.key], v = r?.[k];
            const se = k === t.metric && r?.stderr ? `<div class="muted small">± ${(r.stderr * 100).toFixed(1)}</div>` : "";
            return `<td class="num"><span class="${v != null && v === best ? "best" : ""}">${fmtV(v)}</span>${se}</td>`; }).join("")}</tr>`;
      });
    }).join("");
    // the average of each model's headline metric over the multiple-choice tasks
    const avgRow = `<tr class="bn-first"><td><b>Average</b> <span class="muted small">★ metrics of HellaSwag, ARC-Easy, PIQA, Winogrande, and LAMBADA accuracy</span></td><td class="num muted">${bpct(tasks.reduce((a, t) => a + (t.key === "lambada" ? 0 : t.chance), 0) / tasks.length)}</td>
      <td class="num muted">${bpct(tasks.reduce((a, t) => a + (t.gpt2[t.metric] ?? t.gpt2.acc), 0) / tasks.length)}</td>
      ${done.map((m) => { const vs = tasks.map((t) => benchMetricOf(t, m.benchmarks.results[t.key])); return `<td class="num"><b>${vs.some((v) => v == null) ? "–" : bpct(vs.reduce((a, b) => a + b, 0) / vs.length)}</b></td>`; }).join("")}</tr>`;
    $("#bn-results").innerHTML = `<div class="ptable-scroll"><table class="ptable bn">${head}${rows}${avgRow}</table></div>
      <p class="hint">★ is the number usually quoted for each test. ± is one standard error: differences smaller than about twice that are within the noise. Zero-shot: no examples are shown to the model; each answer is scored by how likely the model finds it. The context is cut to the model's window (256 tokens for ours, 1,024 for GPT-2).</p>`;
    // chart: per task, how far above chance each model gets (headline metric)
    $("#bn-chart").innerHTML = `<div class="ev-bars">${tasks.map((t) => {
      const lo = t.key === "lambada" ? 0 : t.chance;
      const vals = [["GPT-2 small (published)", t.gpt2[t.metric] ?? t.gpt2.acc, "#9aa3b8"], ...done.map((m, i) => [m.name.split(" · ")[0], benchMetricOf(t, m.benchmarks.results[t.key]), BENCH_COLORS[i % BENCH_COLORS.length]])];
      const hi = Math.max(lo + 0.05, ...vals.map((v) => v[1] || 0)) * 1.05;
      return `<div class="ev-group"><div class="ev-gtitle">${esc(t.title)} <span class="muted small">${t.metric === "acc_norm" ? "acc_norm" : "accuracy"} · chance ${t.key === "lambada" ? "0%" : bpct(t.chance)}</span></div>
        ${vals.map(([n, v, col]) => `<div class="kvm-row"><span class="muted small" title="${esc(n)}">${esc(n)}</span><div class="kvm-track"><div class="kvm-bar" style="width:${v == null ? 0 : Math.max(1, ((v - lo) / (hi - lo)) * 100)}%;background:${col}"></div></div><b>${bpct(v)}</b></div>`).join("")}</div>`;
    }).join("")}</div><p class="hint">Bars start at chance: the length is how much better than guessing each model does.</p>`;
  }
  // ---- run panel ----
  // models worth testing first: ones with results, and ones trained on web text (story-only models score at chance)
  const web = (m) => (m.datasets || []).some((x) => /fineweb|webtext/i.test(x)) || m.id.startsWith("webtext");
  const pick = (m) => `<label class="bn-pick"><input type="checkbox" data-bpick="${esc(m.id)}" ${bench.picked.has(m.id) ? "checked" : ""}><span>${esc(m.name)}</span><span class="muted small">${short(m.params || 0)}${m.benchmarks ? " · has results" : ""}</span></label>`;
  const first = d.models.filter((m) => m.benchmarks || web(m)), rest = d.models.filter((m) => !(m.benchmarks || web(m)));
  $("#bn-models").innerHTML = `<div class="bn-modelgrid">${first.map(pick).join("")}</div>` +
    (rest.length ? `<details class="bn-more"><summary class="muted small">Show all ${d.models.length} models (the others are trained on stories or small text sets)</summary><div class="bn-modelgrid">${rest.map(pick).join("")}</div></details>` : "");
  $("#bn-taskpick").innerHTML = tasks.map((t) => `<label class="inline"><input type="checkbox" data-btask="${t.key}" ${bench.tasks.has(t.key) ? "checked" : ""}> ${esc(t.title)}</label>`).join("");
}

document.addEventListener("change", (e) => {
  const p = e.target.closest?.("[data-bpick]"), t = e.target.closest?.("[data-btask]");
  if (p) { p.checked ? bench.picked.add(p.dataset.bpick) : bench.picked.delete(p.dataset.bpick); }
  if (t) { t.checked ? bench.tasks.add(t.dataset.btask) : bench.tasks.delete(t.dataset.btask); }
});
$("#bn-run").addEventListener("click", (e) => action(e.target, async () => {
  if (!bench.picked.size) throw new Error("Tick at least one model.");
  await api("/api/benchmarks/run", { ids: [...bench.picked], tasks: [...bench.tasks] });
  bench.waiting = true;
}));
// refresh the page when a benchmark job finishes
setInterval(() => {
  const j = ui.status?.job;
  if (bench.waiting && j && j.kind === "benchmarks" && j.status !== "running") { bench.waiting = false; benchOpen(); if (typeof loadModels === "function") loadModels(); }
}, 1500);

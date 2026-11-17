// Quantize page: a saved model, a recipe (round to nearest 8/4-bit, or GPTQ through llm-compressor), the quantized copy,
// and both compared: sizes, held-out perplexity, the same prompt continued. Data from /api/quantize (quantization/).

const quant = { data: null, model: null, method: "rtn8", copy: null };

const QZ_WHAT = {
  rtn8: "Every weight is rounded to the nearest of 255 levels (−127…127 times a scale per row). No data needed; a quarter of the memory. Section 27 of the journey: almost free for our models.",
  rtn4: "Every weight is rounded to the nearest of 15 levels (−7…7 times a scale per group of 64 weights). No data needed; about a sixth of the memory, for a little loss. The shared embedding / output layer stays at 8 bits.",
  gptq4: "GPTQ (Frantar et al., 2022) quantizes each matrix one input column at a time and, after each column, shifts the columns still to come to cancel the error it just made, using how the layer's inputs vary on calibration text. Blocks are done in order, each calibrated on the already-quantized blocks' outputs. 4-bit weights (−8…7 times a scale per group of 128), the same size as round to nearest; it runs in llm-compressor's own environment.",
};

async function quantOpen(keepCopy = false) {
  const q = new URLSearchParams();
  if (quant.model) q.set("model", quant.model);
  q.set("samples", $("#qz-samples").value || 256);
  q.set("seq_len", $("#qz-seqlen").value || 512);
  quant.data = await api(`/api/quantize?${q}`);
  if (!quant.data.model) { $("#qz-model-info").innerHTML = '<p class="muted">No saved models yet: train one or import one first.</p>'; return; }
  quant.model = quant.data.model.id;
  const ids = quant.data.copies.map((c) => c.id);
  if (!keepCopy || !ids.includes(quant.copy)) quant.copy = ids[0] || null;
  quantRender();
}

function quantRender() {
  const d = quant.data, m = d.model, c = m.model;
  // ---- 1. the model ----
  $("#qz-model").innerHTML = d.models.map((x) => `<option value="${esc(x.id)}" ${x.id === m.id ? "selected" : ""}>${esc(x.name)}${x.active ? " (in use)" : ""}</option>`).join("");
  $("#qz-model-sum").textContent = `${short(m.params || 0)} parameters · ${bytes(m.bytes || 0)} on disk (32-bit floats)`;
  const all = [...m.layers, m.embedding];
  const total = all.reduce((a, r) => a + r.params, 0);
  $("#qz-model-info").innerHTML = `<p class="muted small">${c.n_layer} blocks · ${c.n_embd}-dim · ${c.n_head} heads · context ${c.block_size} · vocabulary ${fmt(c.vocab_size)}${m.chat ? " · chat model" : ""}. Its weight matrices, which quantization stores as integers:</p>
    <div class="ptable-scroll"><table class="ptable qz-table"><thead><tr><th>Matrix</th><th>What it does</th><th class="num">Shape</th><th class="num">How many</th><th class="num">Weights</th><th class="num">Share</th><th>GPTQ</th></tr></thead><tbody>
    ${all.map((r) => `<tr><td><code>${esc(r.name)}</code></td><td>${esc(r.what)}</td><td class="num">${r.shape[0]} × ${r.shape[1]}</td><td class="num">${r.count}</td><td class="num">${short(r.params)}</td><td class="num">${((r.params / total) * 100).toFixed(0)}%</td>
      <td>${r.name.startsWith("lm_head") ? '<span class="muted small">kept at 8 bits</span>' : r.gptq ? '<span class="tag good">yes</span>' : '<span class="tag drop" title="its width is not a multiple of 128: rounded to nearest instead">rounded</span>'}</td></tr>`).join("")}
    </tbody></table></div>
    <p class="hint">The norms (a few thousand numbers) stay as floats. The embedding is shared with the output layer, so it's stored once.</p>`;
  // ---- 2. the recipe ----
  $("#qz-methods").innerHTML = d.methods.map((x) => `<button data-qmethod="${x.key}" class="${x.key === quant.method ? "on" : ""}" ${x.available ? "" : 'title="needs llm-compressor\'s environment"'}>${esc(x.title)}</button>`).join("");
  const meth = d.methods.find((x) => x.key === quant.method) || d.methods[0];
  $("#qz-method-what").textContent = QZ_WHAT[meth.key];
  $("#qz-recipe").textContent = meth.recipe;
  $("#qz-calib").classList.toggle("hidden", meth.key !== "gptq4");
  $("#qz-calib-data").innerHTML = d.calibration_data ? `from <code>data/${esc(d.calibration_data)}/train.bin</code> (training text; perplexity below uses the held-out split)` : '<span class="bad-text">no tokenized calibration data (data/…/train.bin) on this Mac</span>';
  $("#qz-setup").classList.toggle("hidden", meth.available);
  if (!meth.available) $("#qz-setup").innerHTML = `GPTQ runs in llm-compressor's own environment, which isn't set up yet. In a terminal, in the project folder:<pre class="qz-recipe">${esc(d.gptq_setup)}</pre>`;
  $("#qz-run").disabled = !meth.available || (ui.status?.job?.status === "running");
  // ---- 3–5. the copies ----
  $("#qz-copy").innerHTML = d.copies.length ? d.copies.map((x) => `<option value="${esc(x.id)}" ${x.id === quant.copy ? "selected" : ""}>${esc(x.name.split(" · ").slice(-1)[0])} · ${bytes(x.bytes)} · ${esc((x.added_at || "").slice(5, 16))}</option>`).join("") : "<option>none yet</option>";
  quantRenderCopy();
}

function quantRenderCopy() {
  const d = quant.data, copy = d.copies.find((x) => x.id === quant.copy), r = copy?.comparison;
  if (!copy) {
    $("#qz-sizes").innerHTML = '<p class="muted">No quantized copy of this model yet: pick a recipe above and press <b>Quantize and compare</b>.</p>';
    $("#qz-ppl").innerHTML = ""; $("#qz-ppl-when").textContent = ""; $("#qz-gen-out").innerHTML = ""; return;
  }
  const rec = copy.recipe || {}, q = copy.quantized || {};
  const how = q.method === "gptq" ? `GPTQ ${rec.scheme || "W4A16"} (llm-compressor), calibrated on ${rec.calibration?.samples} × ${rec.calibration?.seq_len} tokens in ${rec.seconds ?? "?"} s` : `round to nearest, ${q.bits}-bit`;
  if (!r) {
    $("#qz-sizes").innerHTML = `<p class="muted">${esc(copy.name)}: ${how}. Not measured yet. <button class="btn small" data-qmeasure="${esc(copy.id)}">Measure</button></p>`;
    $("#qz-ppl").innerHTML = ""; return;
  }
  const o = r.original_model, z = r.quantized_model;
  const bar = (label, v, max, cls) => `<div class="kvm-row"><span class="muted small">${label}</span><div class="kvm-track"><div class="kvm-bar ${cls}" style="width:${(v / max) * 100}%"></div></div><b>${bytes(v)}</b></div>`;
  const maxB = Math.max(o.file_bytes, o.memory_bytes);
  $("#qz-sizes").innerHTML = `<p class="muted small">${esc(z.name)}<br>${how}${q.embed_bits ? ` · shared embedding ${q.embed_bits}-bit` : ""}${rec.rounded_instead?.length ? ` · ${rec.rounded_instead.length} matrices rounded instead (width not a multiple of 128)` : ""}</p>
    <div class="stats">${stat("Original", bytes(o.file_bytes), "file on disk · 32-bit floats")}${stat("Quantized", bytes(z.file_bytes), `file on disk · ${(o.file_bytes / z.file_bytes).toFixed(1)}× smaller`)}${stat("Weights in memory", `${bytes(o.memory_bytes)} → ${bytes(z.memory_bytes)}`, "what generation reads per token", true)}</div>
    <div class="kvm-bars qz-bars">${bar("Original, file", o.file_bytes, maxB, "w")}${bar("Quantized, file", z.file_bytes, maxB, "c")}</div>
    <p class="hint">Small file differences between copies of the same bit width come from the scales (one per 64 or 128 weights) and the layers kept at 8 bits.</p>`;
  // ---- 4. perplexity ----
  $("#qz-ppl-when").textContent = `measured ${r.measured_at} on ${r.device}`;
  const keys = Object.keys(o.perplexity);
  $("#qz-ppl").innerHTML = keys.length ? `<div class="ptable-scroll"><table class="ptable qz-table"><thead><tr><th>Held-out text</th><th class="num">Perplexity: original → quantized</th><th class="num">Change</th><th class="num" title="mean next-token loss in nats; perplexity is e^loss">Loss: original → quantized</th></tr></thead><tbody>
    ${keys.map((k) => { const a = o.perplexity[k], b = z.perplexity[k]; if (!b) return ""; const dl = b.loss - a.loss, dp = (b.perplexity / a.perplexity - 1) * 100;
      return `<tr><td>${esc(qzTitle(k, a.title))}</td><td class="num">${fmt(a.perplexity)} → <b>${fmt(b.perplexity)}</b></td><td class="num"><span class="qz-delta ${dp > 0.05 ? "up" : dp < -0.05 ? "down" : ""}">${dp >= 0 ? "+" : ""}${dp.toFixed(1)}%</span></td><td class="num">${a.loss.toFixed(3)} → ${b.loss.toFixed(3)} <span class="qz-delta ${dl > 0.0005 ? "up" : dl < -0.0005 ? "down" : ""}">(${dl >= 0 ? "+" : ""}${dl.toFixed(3)})</span></td></tr>`; }).join("")}
    </tbody></table></div>
    <p class="hint">The same ${fmt(Object.values(o.perplexity)[0].tokens)} tokens per data set (fixed windows of each validation split, as in the evaluation suite). Perplexity is e<sup>loss</sup>: roughly how many tokens the model is choosing between. A model is best on the data it was trained on; compare within a row.</p>` : '<p class="muted">No tokenized held-out data on this Mac.</p>';
  // ---- 5. the saved sample, until the user generates their own ----
  if (!$("#qz-gen-out").dataset.live) quantShowGen([{ ...o.sample, label: "Original", name: o.name }, { ...z.sample, label: "Quantized", name: z.name }], r.prompt);
}

function qzTitle(key, title) {     // results measured before titles were stored for every set
  if (title && title !== key) return title;
  const rest = key.split("__").pop();
  return rest.startsWith("cosmopedia-") ? "Cosmopedia " + rest.slice(11) : rest.startsWith("mini-fineweb-edu-") ? "Mini FineWeb-Edu " + rest.slice(17) : rest;
}

function quantShowGen(rows, prompt) {
  $("#qz-gen-out").innerHTML = rows.map((g) => `<div class="cmp-col"><div class="cmp-name"><b>${esc(g.label)}</b> <span class="muted small">${g.tokens_per_sec ? `${g.tokens_per_sec} tokens/s · ` : ""}confidence ${Math.round((g.avg_prob || 0) * 100)}% · repeated 4-token runs ${Math.round((g.repeated_4grams || 0) * 100)}%</span></div>
    <div class="doc cmp-text"><span class="prompt">${esc(prompt)}</span>${esc(g.text)}</div></div>`).join("");
}

async function quantGenerate() {
  const copy = quant.data?.copies.find((x) => x.id === quant.copy);
  if (!copy) throw new Error("Quantize the model first.");
  const prompt = $("#qz-prompt").value;
  $("#qz-gen-out").innerHTML = '<p class="muted"><span class="spinner"></span> Generating with both models…</p>';
  const r = await api("/api/quantize/generate", { ids: [quant.model, copy.id], prompt, tokens: +$("#qz-tokens").value, temperature: +$("#qz-temp").value });
  $("#qz-gen-out").dataset.live = "1";
  quantShowGen([{ ...r.results[0], label: "Original" }, { ...r.results[1], label: "Quantized" }], prompt);
}

function quantJobDone(j) {
  if (ui.step !== "quantize") return;
  if (j.result?.copy) quant.copy = j.result.copy;
  delete $("#qz-gen-out").dataset.live;
  quantOpen(true).catch((e) => toast(e.message, true));
}

$("#qz-model").addEventListener("change", (e) => { quant.model = e.target.value; quant.copy = null; delete $("#qz-gen-out").dataset.live; action(null, quantOpen); });
$("#qz-copy").addEventListener("change", (e) => { quant.copy = e.target.value; delete $("#qz-gen-out").dataset.live; quantRenderCopy(); });
$$("#qz-samples, #qz-seqlen").forEach((i) => i.addEventListener("change", () => action(null, () => quantOpen(true))));
$("#qz-run").addEventListener("click", (e) => action(e.target, async () => {
  await api("/api/quantize/run", { id: quant.model, method: quant.method, samples: +$("#qz-samples").value, seq_len: +$("#qz-seqlen").value });
  await poll();
}));
$("#qz-gen").addEventListener("click", (e) => action(e.target, quantGenerate));
document.addEventListener("click", (e) => {
  const b = e.target.closest("[data-qmethod]");
  if (b && quant.data) { quant.method = b.dataset.qmethod; quantRender(); }
  const ms = e.target.closest("[data-qmeasure]");
  if (ms) action(ms, async () => { await api("/api/quantize/measure", { id: ms.dataset.qmeasure }); await poll(); });
});

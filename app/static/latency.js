// Prefill vs decode: a feel-it simulator for time to first token (prefill, compute bound)
// and tokens per second (decode, memory-bandwidth bound). No model runs here: the text is
// streamed on a timer so the two numbers can be felt separately.
(() => {
  const $ = (s) => document.querySelector(s);
  if (!$("#lat-out")) return;
  const TEXT = "Prefill processes your whole prompt at once, which is compute bound. Decode then emits one token per forward pass, " +
    "reading the entire model from memory each time, which is bandwidth bound.";
  const words = TEXT.split(" ");   // one word stands in for one token (GPT-2 averages ~1.3 tokens a word)
  let timer = null, t0 = 0;

  const vals = () => ({ ttft: +$("#lat-ttft").value, tps: +$("#lat-tps").value });
  const secs = (ms) => (ms >= 10000 ? (ms / 1000).toFixed(0) : (ms / 1000).toFixed(1));
  const statBox = (k, v, unit, d) =>
    `<div class="stat"><div class="k">${k}</div><div class="v">${v}<span class="muted small"> ${unit}</span></div><div class="d">${d}</div></div>`;

  function renderStats() {
    const { ttft, tps } = vals();
    const total = ttft + (words.length / tps) * 1000;
    $("#lat-ttft-v").textContent = `${ttft} ms`;
    $("#lat-tps-v").textContent = `${tps}/s`;
    $("#lat-gap").textContent = `${Math.round(1000 / tps)} ms between tokens`;
    $("#lat-stats").innerHTML = [
      statBox("TTFT", ttft, "ms", "Compute bound, grows with prompt length"),
      statBox("TPS", tps, "/s", "Bandwidth bound: the whole model is read for every token"),
      statBox("Total", secs(total), "s", `${words.length} words, end to end`),
    ].join("");
    return total;
  }

  function idle() {
    $("#lat-out").innerHTML = `<span class="lat-wait">Press Run to stream the reply.</span>`;
    $("#lat-bar-pre").style.width = $("#lat-bar-dec").style.width = "0";
    $("#lat-clock").textContent = "";
  }

  function stop() { if (timer) clearTimeout(timer); timer = null; $("#lat-run").textContent = "▶ Run"; }

  function frame() {
    const { ttft, tps } = vals(), total = ttft + (words.length / tps) * 1000;
    const t = performance.now() - t0;
    const shown = t < ttft ? 0 : Math.min(words.length, Math.floor(((t - ttft) / 1000) * tps) + 1);
    const done = t >= total;
    $("#lat-out").innerHTML = shown === 0
      ? `<span class="lat-wait">Prefill: reading the prompt…</span><span class="lat-caret"></span>`
      : `${words.slice(0, shown).join(" ")}${done ? "" : '<span class="lat-caret"></span>'}`;
    $("#lat-bar-pre").style.width = `${(Math.min(t, ttft) / total) * 100}%`;
    $("#lat-bar-dec").style.width = `${(Math.max(0, Math.min(t, total) - ttft) / total) * 100}%`;
    $("#lat-clock").textContent = `${(Math.min(t, total) / 1000).toFixed(2)} s · ${shown}/${words.length} tokens`;
    if (done) return stop();
    timer = setTimeout(frame, 16);
  }

  $("#lat-run").addEventListener("click", () => {
    if (timer) { stop(); return; }
    renderStats();
    t0 = performance.now();
    $("#lat-run").textContent = "■ Stop";
    timer = setTimeout(frame, 16);
  });

  $("#lat-mine").addEventListener("click", () => {
    const m = window.lastGenMetrics;
    if (!m) { $("#lat-note").textContent = "Generate some text above first; this uses that run's numbers."; return; }
    $("#lat-ttft").value = Math.min(4000, Math.round(m.ttft_ms / 50) * 50);
    $("#lat-tps").value = Math.max(1, Math.min(200, Math.round(m.tokens_per_sec)));
    const gbps = m.weights_bytes ? (m.weights_bytes * m.tokens_per_sec) / 1e9 : null;
    $("#lat-note").innerHTML = `From your last generation: ${Math.round(m.ttft_ms)} ms to the first token, ${m.tokens_per_sec} tok/s` +
      (gbps ? ` · decoding read the weights at about <b>${gbps.toFixed(1)} GB/s</b> (${(m.weights_bytes / 1e6).toFixed(0)} MB × ${m.tokens_per_sec} tokens/s)` : "") +
      (m.tokens_per_sec > 200 ? " · TPS capped at 200 on the slider" : "");
    stop(); renderStats(); idle();
  });

  ["#lat-ttft", "#lat-tps"].forEach((s) => $(s).addEventListener("input", () => { renderStats(); }));
  renderStats(); idle();
})();

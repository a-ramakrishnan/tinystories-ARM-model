// Explorer → Flow → "Play the data flow": one token through a whole transformer block, stage by stage.
// Each step lights only the parts of the Flow that are working on that token, sends particles along its own
// ribbons, and explains the step with the model's real numbers. Uses ex / ex.fg (explorer.js, flow.js).

const fp = { on: false, auto: false, step: 0, t: null, timer: null, raf: null };
const FP_DUR = 4200;   // ms per step when playing

const fpTok = (i) => `<code>${esc(ex.data.tokens[i].text)}</code>`;
const fpPct = (p) => `${(p * 100).toFixed(p < 0.1 ? 1 : 0)}%`;

function fpSteps(t) {
  // The whole forward pass for one token: the embedding, block 1 in full detail, then every later block as two
  // steps (attention, feed-forward) with that block's real numbers, then the prediction. Each step names the
  // block it happens in; the player switches the diagram to that block.
  const d = ex.data, c = d.config, h = ex.head, T = d.tokens.length, N = c.passes;
  const H = c.n_head, D = c.n_embd, hd = c.head_dim, norm = c.norm === "rmsnorm" ? "RMSNorm" : "LayerNorm";
  const rope = c.pos_encoding === "rope";
  const row = (sel) => `${sel}[data-row="${t}"]`;
  const upto = (sel) => Array.from({ length: t + 1 }, (_, j) => `${sel}[data-row="${j}"]`);
  const card = [...upto(".fl-cardrow"), ".fl-cardtitle", 'text[data-focus="attention"]'];
  const sizeIn = (b) => (b === 0 ? d.embedding.token_norm[t] : d.layers[b - 1].resid_norm[t]);
  const weights = (b) => d.layers[b].attention[h][t].slice(0, t + 1).map((w, j) => `${fpTok(j)} ${fpPct(w)}`).join(" · ");
  const bestOf = (b) => { const a = d.layers[b].attention[h][t].slice(0, t + 1); return a.indexOf(Math.max(...a)); };
  const sharpest = (b) => {   // the head that focuses hardest for this token in block b
    let hb = 0, v = -1, to = 0;
    d.layers[b].attention.forEach((head, k) => head[t].forEach((w, j) => { if (j <= t && w > v) { v = w; hb = k; to = j; } }));
    return `head ${hb + 1} on ${fpTok(to)} (${fpPct(v)})`;
  };
  const lens = (b) => { const x = d.layers[b].lens[t][0]; return `<code>${esc(x.text)}</code> (${fpPct(x.p)})`; };
  const ffn = (b) => {
    const L = d.layers[b];
    return c.n_experts > 1
      ? `${norm}, then the router scores the ${c.n_experts} experts (${L.router.probs[t].map((p, e) => `${e + 1}: ${fpPct(p)}`).join(", ")}) and sends the token to expert${c.moe_top_k > 1 ? "s" : ""} ${L.router.chosen[t].map((e) => e + 1).join(" and ")}, mixing their outputs by those weights.`
      : `${norm}, then the feed-forward layer works on this token alone: ${D} → <b>${L.mlp.width}</b> numbers (${c.mlp === "swiglu" ? "silu(gate) × up" : "GELU"}), ${fpPct(L.mlp.active[t])} of them strongly active, → ${D}.`;
  };
  const ffnLit = [row('[data-p="dip"]'), ".fl-hidden", row(".fl-act"), row(".fl-mlprow"), ".fl-resid", 'text[data-focus="mlp"]'];
  const steps = [];
  steps.push({ layer: 0, name: "Embedding", lit: [row(".fl-embrow"), 'text[data-focus="embedding"]'], paths: [],
    text: `${fpTok(t)} is token ${fmt(d.tokens[t].id)} in the vocabulary. That id picks row ${fmt(d.tokens[t].id)} of a learned table: a vector of <b>${D}</b> numbers that stands for this token. ` +
      (rope ? `Its position (${t + 1} of ${T}) isn't added here: RoPE rotates the query and key by an angle for this position, inside attention.` : `A learned vector for position ${t} is added, so the same word at another place looks different.`) +
      ` Now it goes through ${N} blocks, one after another.` });
  for (let b = 0; b < N; b++) {
    const L = d.layers[b], sc = L.scores[h][t], tag = `Block ${b + 1} of ${N} · `;
    if (b === 0) {
      steps.push(
        { layer: b, name: tag + "Query, key, value", lit: [row(".fl-embrow"), row('[data-p="fan"]'), `.fl-tab[data-headpick="${h}"]`, 'text[data-focus="qkv"]'], paths: [row('[data-p="fan"]')],
          text: `${norm}, then one matrix turns the ${D} numbers into a <b>query</b>, a <b>key</b> and a <b>value</b> for each of the ${H} heads, ${hd} numbers each${c.n_kv_head < H ? ` (${c.n_kv_head} key/value heads, shared)` : ""}.` +
            `${c.qk_norm ? " QK-norm normalizes each query and key." : ""}${rope ? " RoPE rotates them by the token's position." : ""} We follow head ${h + 1}.` },
        { layer: b, name: tag + "Into the head", lit: [`.fl-tab[data-headpick="${h}"]`, row('[data-p="list"]'), row(".fl-cardrow"), ".fl-cardtitle"], paths: [row('[data-p="list"]')],
          text: `In head ${h + 1}, ${fpTok(t)} now has a <b>query</b> (what it is looking for), a <b>key</b> (what it offers to others) and a <b>value</b> (what it will pass on if someone attends to it).` },
        { layer: b, name: tag + "Scores", stage: "scores", lit: [...card, row('[data-p="qline"]'), ...upto('[data-p="kline"]'), `.ex-ag[data-i="${t}"]`], paths: [row('[data-p="qline"]'), ...upto('[data-p="kline"]')],
          text: `Its query is compared with every key: the dot product of two ${hd}-number vectors, divided by √${hd} = ${Math.sqrt(hd).toFixed(1)}. ` +
            sc.map((v, j) => `${j <= t ? "" : "<span class='muted'>"}${fpTok(j)} ${v.toFixed(1)}${j <= t ? "" : "</span>"}`).join(" · ") },
        { layer: b, name: tag + "Mask", stage: "masked", lit: [...card, row('[data-p="qline"]'), `.ex-ag[data-i="${t}"]`], paths: [],
          text: t < T - 1 ? `A decoder can't look ahead: the scores for ${Array.from({ length: T - 1 - t }, (_, k) => fpTok(t + 1 + k)).join(", ")} are set to <b>−∞</b>, so they'll get no weight. When the model is generating, those tokens don't exist yet.`
            : `${fpTok(t)} is the last token, so there is nothing after it to hide. For every earlier token, the later positions are set to <b>−∞</b>.` },
        { layer: b, name: tag + "Softmax", stage: "softmax", lit: [...card, row('[data-p="qline"]'), `.ex-ag[data-i="${t}"]`], paths: [row('[data-p="qline"]')],
          text: `Softmax turns the scores into weights that add up to 100%: ${weights(b)}. Head ${h + 1} looks mostly at ${fpTok(bestOf(b))}.` },
        { layer: b, name: tag + "Weighted values → Out", values: true, lit: [...card, ".fl-vband"], paths: [],
          text: `The head's output for ${fpTok(t)} is the weighted sum of the values: ${L.attention[h][t].slice(0, t + 1).map((w, j) => (w >= 0.01 ? `${w.toFixed(2)} × v(${fpTok(j)})` : null)).filter(Boolean).join(" + ")}. ` +
            `All ${H} heads do this at once; their outputs are put side by side (${H} × ${hd} = ${D}) and mixed by one more matrix.` },
        { layer: b, name: tag + "Residual", lit: [row(".fl-cardrow"), row('[data-p="outline"]'), row(".fl-mlprow")], paths: [row('[data-p="outline"]')],
          text: `That result is <b>added</b> to the token's own vector (the residual connection): attention adds information, it never replaces what was there. Its size goes from ${sizeIn(b).toFixed(1)} to ${L.resid_norm_mid[t].toFixed(1)}.` },
        { layer: b, name: tag + (c.n_experts > 1 ? "Experts" : "Feed-forward"), lit: ffnLit, paths: [row('[data-p="dip"]')],
          text: `${ffn(b)} That is added to the vector too: size ${L.resid_norm_mid[t].toFixed(1)} → ${L.resid_norm[t].toFixed(1)}. If the model stopped after this block, it would predict ${lens(b)} next.` },
      );
    } else {
      steps.push(
        { layer: b, name: tag + "Attention", stage: "softmax", values: true, dur: 3600,
          lit: [row(".fl-embrow"), row('[data-p="fan"]'), `.fl-tab[data-headpick="${h}"]`, row('[data-p="list"]'), ...card, row('[data-p="qline"]'), ...upto('[data-p="kline"]'), `.ex-ag[data-i="${t}"]`, ".fl-vband"],
          paths: [row('[data-p="fan"]'), row('[data-p="list"]')],
          text: `The vector that left block ${b} (size ${sizeIn(b).toFixed(1)}) is the input to block ${b + 1}, which has its own weights. Same steps: query, key and value; scores; mask; softmax. Head ${h + 1} here: ${weights(b)}, mostly ${fpTok(bestOf(b))}. ` +
            `Of all ${H} heads, the sharpest is ${sharpest(b)}. The weighted values are added back: size → ${L.resid_norm_mid[t].toFixed(1)}.` },
        { layer: b, name: tag + (c.n_experts > 1 ? "Experts" : "Feed-forward"), dur: 3600, lit: [...ffnLit, row('[data-p="outline"]')], paths: [row('[data-p="outline"]'), row('[data-p="dip"]')],
          text: `${ffn(b)} Added back: size ${L.resid_norm_mid[t].toFixed(1)} → ${L.resid_norm[t].toFixed(1)}. If the model stopped after block ${b + 1}, it would predict ${lens(b)} next.` },
      );
    }
  }
  const last = d.layers[N - 1].lens[t][0];
  if (t === T - 1) {
    const ps = exProbs(), kept = ps.filter((p) => !p.cut).length;
    steps.push({ layer: N - 1, name: "Next token", lit: [".fl-pband", '[data-p="pline"]', ".fl-probs", ".fl-last", row('[data-p="blockline"]'), 'text[data-focus="probs"]'], paths: [row('[data-p="blockline"]'), '[data-p="pline"]'],
      text: `After the last block: a final ${norm} and the output layer (the embedding table again) score all ${fmt(c.vocab_size)} tokens, and softmax turns the scores into probabilities: ${ps.slice(0, 4).map((p) => `<code>${esc(p.text)}</code> ${fpPct(p.p)}`).join(" · ")}. With temperature ${ex.temp}, top-k ${ex.topk || "off"} and top-p ${ex.topp >= 1 ? "off" : ex.topp}, the next token is drawn from ${kept} candidate${kept === 1 ? "" : "s"}.` });
  } else {
    const real = exRealNext(t);
    steps.push({ layer: N - 1, name: "Its prediction", lit: [row(".fl-mlprow"), row('[data-p="blockline"]'), row(".fl-blk"), ".fl-pband", '[data-p="pline"]', ".fl-probs", ".fl-last", 'text[data-focus="probs"]', ".fl-realtxt", ".fl-realbg"],
      paths: [row('[data-p="blockline"]'), '[data-p="pline"]'],
      text: `After all ${N} blocks, the final ${norm} and output layer turn ${fpTok(t)}'s vector into a prediction for the token after it (on the right): <code>${esc(last.text)}</code> is its top guess, ${fpPct(last.p)} as the model computes it` +
        ` (${fpPct(exProbs(t)[0].p)} on the right, after temperature ${ex.temp}${ex.topk ? `, top-k ${ex.topk}` : ""}${ex.topp < 1 ? ` and top-p ${ex.topp}` : ""}). ` +
        (real ? `The real next token is ${fpTok(t + 1)}: the model gave it ${(real.p * 100).toFixed(real.p < 0.01 ? 2 : 1)}%, its choice no. ${real.rank}${real.rank === 1 ? " (a correct guess)" : ""}. ` : "") +
        `In training, the loss is how surprised the model is by the real token (−log of that probability: ${real ? (-Math.log(Math.max(real.p, 1e-9))).toFixed(2) : "–"}); when generating, only the last token's prediction is used.` });
  }
  return steps;
}

// ---------- drawing a step ----------
function fpApply() {
  const svg = $("#ex-svg svg");
  if (!svg || !fp.on || !ex.data) return;
  const steps = fpSteps(fp.t), st = steps[Math.min(fp.step, steps.length - 1)];
  if (st.layer !== ex.layer) {            // the token has moved on to another block: redraw that block
    ex.layer = st.layer;
    $("#ex-layer-label").textContent = `Block ${ex.layer + 1} of ${ex.data.config.passes}`;
    exRender();                           // calls flAfterRender → fpApply again, now on the right block
    return;
  }
  svg.classList.add("fl-play");
  $$(".fl-lit", svg).forEach((e) => e.classList.remove("fl-lit"));
  st.lit.forEach((sel) => $$(sel, svg).forEach((e) => e.classList.add("fl-lit")));
  const ov = $("#fl-overlay", svg);
  ov.innerHTML = "";
  const g = ex.fg, d = ex.data, L = d.layers[ex.layer], t = fp.t, h = ex.head;
  // numbers next to the token's row of attention dots
  if (st.stage) {
    for (let j = 0; j < d.tokens.length; j++) {
      const v = st.stage === "scores" ? L.scores[h][t][j].toFixed(1)
        : j > t ? "−∞" : st.stage === "masked" ? L.scores[h][t][j].toFixed(1) : Math.round(L.attention[h][t][j] * 100);
      ov.insertAdjacentHTML("beforeend", `<text x="${g.colX(j)}" y="${g.yQ(t) - g.S * 0.55}" class="fl-ovnum${j > t && st.stage !== "scores" ? " masked" : ""}" text-anchor="middle">${v}</text>`);
    }
  }
  // the weighted values flowing to this token's Out
  const extra = [];
  if (st.values) {
    L.attention[h][t].forEach((w, j) => {
      if (j > t || w < 0.005) return;
      const p = `M${g.xTick + 14},${g.yV(j)} C${g.xTick + 160},${g.yV(j)} ${g.xOut - 160},${g.yQ(t)} ${g.xOut},${g.yQ(t)}`;
      ov.insertAdjacentHTML("beforeend", `<path d="${p}" class="fl-ovpath" stroke-width="${1 + 7 * w}" data-w="${w}"/>`);
    });
    extra.push(...$$(".fl-ovpath", ov));
    $$(`.fl-cardrow[data-row="${t}"]`, svg).forEach((e) => e.classList.add("fl-lit"));
  }
  const paths = [...st.paths.flatMap((sel) => $$(sel, svg)), ...extra];
  fpParticles(paths);
  // the caption and the step dots
  $("#fl-caption").classList.remove("hidden");
  $("#fl-caption").innerHTML = `<div class="fl-capname">Step ${fp.step + 1} of ${steps.length} · ${esc(st.name)} <span class="muted small">· following ${fpTok(t)}, head ${h + 1}</span></div><div>${st.text}</div>`;
  $("#fl-dots").innerHTML = steps.map((s, i) => `${i > 0 && s.layer !== steps[i - 1].layer ? '<span class="fl-dot-sep"></span>' : ""}<button class="fl-dot-btn${i === fp.step ? " on" : ""}${i < fp.step ? " done" : ""}" data-fstep="${i}" title="${esc(s.name)}"></button>`).join("");
  $("#fl-play").textContent = fp.auto ? "❚❚ Pause" : fp.step >= steps.length - 1 ? "↺ Replay" : "▶ Play";
}

function fpParticles(paths) {
  cancelAnimationFrame(fp.raf);
  const ov = $("#fl-overlay");
  if (!ov || !paths.length) return;
  const items = paths.map((p) => {
    const w = +(p.dataset.w || 0.6);
    const color = p.getAttribute("stroke") || getComputedStyle(p).stroke || "#7a5cf0";
    ov.insertAdjacentHTML("beforeend", `<circle r="${(3 + 4 * Math.min(1, w)).toFixed(1)}" class="fl-particle" fill="${color}"/>`);
    return { p, len: p.getTotalLength(), dot: ov.lastElementChild };
  });
  const start = performance.now(), period = 1700;
  const ease = (x) => (x < 0.5 ? 2 * x * x : 1 - (-2 * x + 2) ** 2 / 2);
  const frame = (now) => {
    const k = ((now - start) % period) / period;
    items.forEach((it) => {
      const pt = it.p.getPointAtLength(it.len * ease(k));
      it.dot.setAttribute("cx", pt.x);
      it.dot.setAttribute("cy", pt.y);
      it.dot.setAttribute("opacity", k < 0.08 ? k / 0.08 : k > 0.92 ? (1 - k) / 0.08 : 1);
    });
    fp.raf = requestAnimationFrame(frame);
  };
  fp.raf = requestAnimationFrame(frame);
}

// ---------- controls ----------
function fpSchedule() {
  clearTimeout(fp.timer);
  if (!fp.auto) return;
  const st = fpSteps(fp.t)[fp.step];
  fp.timer = setTimeout(() => {
    const n = fpSteps(fp.t).length;
    if (fp.step < n - 1) { fp.step++; fpApply(); fpSchedule(); } else { fp.auto = false; fpApply(); }
  }, st?.dur || FP_DUR);
}

function fpStart(fromStep = 0) {
  if (!ex.data) return;
  const T = ex.data.tokens.length;
  fp.t = Math.min(+($("#fl-tok").value || (ex.tok ?? T - 1)), T - 1);
  fp.on = true; fp.auto = true; fp.step = fromStep;
  ex.layer = fpSteps(fp.t)[fromStep].layer;
  ex.focus = "overview"; ex.cell = null;
  exRender();          // redraws, then flAfterRender() applies the step
  fpSchedule();
}

function fpStop() {
  fp.on = false; fp.auto = false;
  clearTimeout(fp.timer); cancelAnimationFrame(fp.raf);
  $("#fl-caption").classList.add("hidden");
  $("#fl-dots").innerHTML = "";
  $("#fl-play").textContent = "▶ Play the data flow";
  const svg = $("#ex-svg svg");
  if (svg) { svg.classList.remove("fl-play"); $$(".fl-lit", svg).forEach((e) => e.classList.remove("fl-lit")); const ov = $("#fl-overlay", svg); if (ov) ov.innerHTML = ""; }
}

function fpGo(delta) {
  if (!fp.on) { fpStart(0); fp.auto = false; clearTimeout(fp.timer); fpApply(); return; }
  const n = fpSteps(fp.t).length;
  fp.step = Math.max(0, Math.min(n - 1, fp.step + delta));
  fp.auto = false; clearTimeout(fp.timer);
  fpApply();
}

function fpFillTokens() {
  const d = ex.data;
  if (!d) return;
  const want = ex.tok ?? fp.t ?? d.tokens.length - 1;
  $("#fl-tok").innerHTML = d.tokens.map((t, i) => `<option value="${i}"${i === want ? " selected" : ""}>${esc(t.text.trim() || "␠")} (${i + 1})</option>`).join("");
}

// called by exRenderFlow after every redraw
function flAfterRender() {
  const T = ex.data.tokens.length;
  $("#fl-player").classList.remove("hidden");
  $("#fl-blockno").textContent = `${ex.layer + 1} of ${ex.data.config.passes}`;
  if ($("#fl-tok").options.length !== T) fpFillTokens();
  if (fp.on && fp.t >= T) fpStop();
  if (!fp.on) $("#fl-tok").value = String(ex.tok ?? T - 1);   // follows the token clicked in the diagram
  if (fp.on) fpApply();
}

$("#fl-play").addEventListener("click", () => {
  if (!fp.on) return fpStart(0);
  const n = fpSteps(fp.t).length;
  if (fp.auto) { fp.auto = false; clearTimeout(fp.timer); fpApply(); return; }
  if (fp.step >= n - 1) fp.step = 0;
  fp.auto = true; fpApply(); fpSchedule();
});
$("#fl-prev").addEventListener("click", () => fpGo(-1));
$("#fl-next").addEventListener("click", () => fpGo(1));
$("#fl-stop").addEventListener("click", () => { fpStop(); exRender(); });   // back to the clicked / last token's prediction
$("#fl-tok").addEventListener("change", () => { if (fp.on) fpStart(0); });
$("#fl-dots").addEventListener("click", (e) => {
  const b = e.target.closest("[data-fstep]");
  if (!b) return;
  fp.step = +b.dataset.fstep; fp.auto = false; clearTimeout(fp.timer); fpApply();
});
document.addEventListener("keydown", (e) => {
  if (!fp.on || ["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement?.tagName)) return;
  if (e.key === "ArrowRight") { e.preventDefault(); fpGo(1); }
  if (e.key === "ArrowLeft") { e.preventDefault(); fpGo(-1); }
  if (e.key === " ") { e.preventDefault(); $("#fl-play").click(); }
});

// the block buttons (← →) during playback jump the player to that block's first step
const fpSetLayerOrig = exSetLayer;
exSetLayer = async function (n) {
  if (!fp.on || !ex.data) return fpSetLayerOrig(n);
  const N = ex.data.config.passes, b = (n + N) % N;
  const i = fpSteps(fp.t).findIndex((st) => st.layer === b);
  fp.step = Math.max(0, i); fp.auto = false; clearTimeout(fp.timer);
  fpApply();
};

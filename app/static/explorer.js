// Explorer: one prompt flowing through the real model, drawn as a diagram (Inside the transformer → Explorer).
// Data comes from /api/explain (explain.py). Uses $, $$, esc, fmt, api, action and toast from app.js.

const ex = { data: null, layer: 0, head: 0, stage: "softmax", hover: null, cell: null, focus: "overview", tok: null, view: "flow",
  temp: 0.8, topk: 0, topp: 0.9 };

const EX_ROW = 40;           // height of one token row; attention cells are square, so rows line up everywhere
const EX_TOP = 132;          // room for the band title, column titles and rotated key labels above the first row

// ---------- colours: a value's sign picks the hue, its size the opacity (works on light and dark panels) ----------
function exCell(x, y, w, h, v, scale, cls = "") {
  const a = Math.min(1, Math.abs(v) / (scale || 1));
  const fill = v >= 0 ? "#f08c2e" : "#3f7be0";
  return `<rect class="${cls}" x="${x}" y="${y}" width="${w}" height="${h}" fill="${fill}" fill-opacity="${(0.08 + 0.92 * a).toFixed(3)}"/>`;
}
const exMaxAbs = (rows) => Math.max(1e-6, ...rows.flat(3).map((v) => Math.abs(v)));
// a row's colour scale: its 90th-percentile size, so one or two huge dimensions (GPT-2 has some) don't wash out the rest
const exRowScale = (row) => { const a = row.map(Math.abs).sort((x, y) => x - y); return Math.max(1e-6, a[Math.floor(0.9 * (a.length - 1))]); };

function exStrip(x, y, vals, cw, h, scale, cls) {
  return vals.map((v, i) => exCell(x + i * cw, y, cw, h, v, scale, cls)).join("");
}

function exRibbon(x1, y1, x2, y2, color, row, w = 6) {
  const mx = (x1 + x2) / 2;
  return `<path class="ex-rib" data-row="${row}" d="M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}" stroke="${color}" stroke-width="${w}" fill="none"/>`;
}

// ---------- explanations: one per part of the diagram, filled with this model's real sizes ----------
function exExplain(key) {
  const d = ex.data, c = d?.config;
  if (!c) return "";
  const H = c.n_head, D = c.n_embd, hd = c.head_dim, N = c.passes;
  const ff = c.mlp === "swiglu" ? `SwiGLU: two projections to ${d.layers[0].mlp?.width ?? "?"} numbers, one gating the other (silu(gate) × up), then back to ${D}`
    : `${D} → ${4 * D} numbers, a GELU, then back to ${D}`;
  const texts = {
    overview: `<b>One prompt through the whole model.</b> Each token becomes a vector of ${D} numbers, passes through ${N} transformer blocks that mix information between tokens (attention) and refine each token on its own (feed-forward), and the last token's vector becomes a probability for each of the ${fmt(c.vocab_size)} tokens in the vocabulary. Everything below is your model's real numbers for this prompt. Hover or click any part.`,
    embedding: `<b>Embedding.</b> Each token id picks one row of a learned table: a vector of ${D} numbers (only every ${Math.round(D / 48)}th is drawn; orange is positive, blue negative, each row scaled to its own typical size). ${c.pos_encoding === "rope" ? "This model uses <b>RoPE</b>: no position vector is added here. Instead, each query and key is <i>rotated</i> by an angle that depends on its position, inside attention, so attention sees how far apart two tokens are." : "This model adds a learned <b>position</b> vector (one per position, 0 to " + (c.block_size - 1) + ") so the same word at different places looks different."}`,
    qkv: `<b>Query, key, value.</b> After ${c.norm === "rmsnorm" ? "RMSNorm" : "LayerNorm"}, one matrix turns each token's vector into a query, a key and a value, split into ${H} heads of ${hd} numbers${c.n_kv_head < H ? ` (here ${c.n_kv_head} key/value heads are shared by the ${H} query heads: grouped-query attention)` : ""}. The query asks "what am I looking for?", the key says "what do I contain?", the value is what gets passed on.${c.qk_norm ? " This model normalizes each head's query and key first (QK-norm), so their dot products can't grow without bound." : ""} Showing head ${ex.head + 1} of block ${ex.layer + 1}.`,
    attention: {
      scores: `<b>Scores.</b> Every query is compared with every key: the dot product of the two ${hd}-number vectors, divided by √${hd} = ${Math.sqrt(hd).toFixed(1)} so the numbers stay in a range softmax handles well. Row = the token asking, column = the token being looked at. Click a cell to see the calculation.`,
      masked: `<b>Mask.</b> A decoder must not see the future: when it reads "${esc(d.tokens[Math.min(1, d.tokens.length - 1)].text)}", the next token doesn't exist yet. Scores above the diagonal are set to −∞, so they'll get zero weight.`,
      softmax: `<b>Softmax.</b> Each row is turned into weights that are positive and add up to 100%: how much the token in that row takes from each earlier token. The head's output for a token is the weighted sum of the values. Different heads learn different habits: one looks at the previous token, one at the first token, one at a name.`,
    }[ex.stage],
    out: `<b>Combine the heads.</b> The ${H} heads' outputs (${hd} numbers each) are put side by side (${H} × ${hd} = ${D}) and mixed by one more matrix. The result is <i>added</i> to the token's vector (the residual connection): attention adds information, it never replaces what was there.`,
    mlp: c.n_experts > 1 ? `<b>Mixture of experts.</b> This block has ${c.n_experts} feed-forward experts; a small router scores them for each token and sends it to the top ${c.moe_top_k}, mixing their outputs by the router's weights. The bars show the router's probabilities; the chosen experts are outlined.`
      : `<b>Feed-forward.</b> Each token on its own, no mixing between tokens: ${ff}. It's where most of the parameters are, and where much of the model's knowledge is stored. The strip shows ${d.layers[ex.layer].mlp?.hidden?.[0]?.length ?? 64} of the ${d.layers[ex.layer].mlp?.width ?? "?"} hidden values; the number on the right is the share that's strongly active. Its output is added back to the token's vector too.`,
    blocks: `<b>${N} blocks.</b> The same structure repeats ${N} times${c.block_repeat > 1 ? ` (${c.n_layer} blocks, each applied ${c.block_repeat} times with the same weights)` : ""}, each with its own weights. Use ← → to step through them: every block's attention looks different. The logit lens below shows what the model would predict if it stopped after each block.`,
    probs: `<b>Probabilities.</b> After the last block, a final norm and the output layer (the embedding table again, shared) give each of the ${fmt(c.vocab_size)} tokens a score (a logit). Softmax turns scores into probabilities. Temperature divides the scores first (below 1 sharper, above 1 flatter); top-k and top-p then keep only the most likely tokens. The next token is drawn from what's left.`,
    lens: `<b>Logit lens.</b> Apply the final norm and output layer early, after each block, as if the model stopped there. Early blocks predict generic, frequent tokens; the prediction sharpens block by block. Each cell is the top guess for the token <i>after</i> that column's token, with its probability.`,
  };
  return texts[key] || texts.overview;
}

// ---------- the diagram ----------
function exRender() {
  const d = ex.data;
  if (!d) return;
  const c = d.config, T = d.tokens.length, L = d.layers[ex.layer];
  const det = L.detail;
  const H = c.n_head;
  ex.head = Math.min(ex.head, H - 1);
  $$("#ex-viewsel .seg-btn").forEach((b) => b.classList.toggle("on", b.dataset.exview === ex.view));
  if (ex.view === "flow" && typeof exRenderFlow === "function") {
    exRenderFlow(d, c, T, L, H);
    exTokenPanel(); exProbBox(); exFocusClass(); exPanel(); exLens();
    return;
  }
  if (typeof fpStop === "function" && fp.on) fpStop();          // the player belongs to the Flow view
  $("#fl-player")?.classList.add("hidden");
  if (!det) { exRun(); return; }       // this block's details weren't loaded (the Flow moved here): fetch them
  const R = EX_ROW, pad = 7, sh = R - 2 * pad;     // row height; strips leave a gap above and below
  // Two bands, one per half of a transformer block, so each fits the width at full size:
  //   band A (attention): tokens | embedding | query | key | value | attention matrix
  //   band B (feed-forward): tokens | out + residual | feed-forward or experts | the other blocks
  const yA = EX_TOP, yB = EX_TOP + T * R + 150;
  const rowA = (i) => yA + i * R, rowB = (i) => yB + i * R;
  const xTok = 128, xEmb = 142, embW = 132;
  const vecW = 96, xQ = xEmb + embW + 70, xK = xQ + vecW + 18, xV = xK + vecW + 18;
  const xAtt = xV + vecW + 72, C = R, attW = T * C;
  const xOut = xEmb, outW = 132, xMlp = xOut + outW + 90;
  const mlpW = c.n_experts > 1 ? c.n_experts * 30 : 300, xStack = xMlp + mlpW + (c.n_experts > 1 ? 70 : 110);
  const W = Math.max(xAtt + attW + 30, xStack + 140), Hh = yB + T * R + 40;
  let s = "";
  const title = (x, y, t, sub, key, color) => `<g class="ex-title" data-focus="${key}"><text x="${x}" y="${y}" class="ex-h"${color ? ` style="fill:${color}"` : ""}>${t}</text>${sub ? `<text x="${x}" y="${y + 18}" class="ex-sub">${sub}</text>` : ""}</g>`;
  const band = (y, n, text) => `<text x="8" y="${y}" class="ex-band"><tspan class="ex-bandn">${n}</tspan> ${text}</text>`;
  const tokLabel = (i, y) => `<g class="ex-tokg" data-tok="${i}"><rect x="4" y="${y + 3}" width="${xTok + 6}" height="${R - 6}" rx="6" class="ex-tokbg"/><text x="${xTok}" y="${y + R / 2 + 5}" class="ex-tok" text-anchor="end">${esc(d.tokens[i].text.replace(/\n/g, "↵")) || "␠"}</text></g>`;
  const norm = c.norm === "rmsnorm" ? "RMSNorm" : "LayerNorm";

  // ---- band A: attention ----
  s += band(yA - 98, "1", `Attention: tokens share information · block ${ex.layer + 1} of ${c.passes}, head ${ex.head + 1} of ${H}`);
  s += title(xEmb, yA - 66, "Embedding", c.pos_encoding === "rope" ? `${c.n_embd} numbers · position: RoPE` : `token + position · ${c.n_embd} numbers`, "embedding");
  s += title(xQ, yA - 66, "Query", c.qk_norm ? "QK-norm applied" : `${c.head_dim} numbers`, "qkv", "#4f7fe0");
  s += title(xK, yA - 66, "Key", `${c.head_dim} numbers`, "qkv", "#d65a5a");
  s += title(xV, yA - 66, "Value", c.n_kv_head < H ? `shared by ${H / c.n_kv_head} heads` : `${c.head_dim} numbers`, "qkv", "#2f9e66");
  s += title(xAtt, yA - 66, "Attention", { scores: "scores: q·k / √d", masked: "future masked (−∞)", softmax: "softmax: each row adds to 100%" }[ex.stage], "attention", "#6d4fe0");
  let bg = `<rect class="ex-block" x="${xQ - 26}" y="${yA - 78}" width="${xAtt + attW + 18 - (xQ - 26)}" height="${T * R + 90}" rx="14"/>`;

  // ---- band B: feed-forward ----
  s += band(yB - 98, "2", `Feed-forward: each token on its own · block ${ex.layer + 1} of ${c.passes}`);
  s += title(xOut, yB - 66, "Out + residual", `${H} heads combined, + residual`, "out");
  s += title(xMlp, yB - 66, c.n_experts > 1 ? "Experts" : "Feed-forward",
    c.n_experts > 1 ? `${norm} → router picks ${c.moe_top_k} of ${c.n_experts}` : `${norm} → ${c.mlp === "swiglu" ? "SwiGLU" : "GELU"} · ${L.mlp?.width ?? ""} hidden numbers (${L.mlp?.hidden?.[0]?.length ?? 64} shown)`, "mlp", "#b8862c");
  s += title(xStack, yB - 66, "Next block", `${c.passes - 1 - ex.layer} more after this one`, "blocks");
  bg += `<rect class="ex-block" x="${xMlp - 26}" y="${yB - 78}" width="${mlpW + (c.n_experts > 1 ? 52 : 96)}" height="${T * R + 90}" rx="14"/>`;
  // the hand-over from band A to band B
  s += `<path d="M${xAtt + attW / 2},${yA + T * R + 14} C${xAtt + attW / 2},${yB - 140} ${xOut + outW / 2},${yA + T * R + 40} ${xOut + outW / 2},${yB - 128}" class="ex-handover"/>`;
  s += `<text x="${(xAtt + attW / 2 + xOut + outW / 2) / 2}" y="${yA + T * R + 54}" class="ex-sub" text-anchor="middle">each token's attention output continues below</text>`;

  const qs = det.q[ex.head], ks = det.k[ex.head], vs = det.v[ex.head];
  const qkvScale = exRowScale([qs, ks, vs].flat(2));
  const step = Math.max(1, Math.round(c.head_dim / 32));
  const sub = (v) => v.filter((_, i) => i % step === 0).slice(0, 32);
  for (let i = 0; i < T; i++) {
    const ya = rowA(i), yb = rowB(i), picked = ex.tok === i ? " ex-picked" : "";
    // band A row
    s += `<g class="ex-row${picked}" data-row="${i}">${tokLabel(i, ya)}`;
    s += `<g data-focus="embedding">${exStrip(xEmb, ya + pad, d.embedding.sum[i], embW / 48, sh, exRowScale(d.embedding.sum[i]))}</g>`;
    s += exRibbon(xEmb + embW, ya + R / 2, xQ, ya + R / 2, "#8a93ab", i, 6);
    s += `<g data-focus="qkv">${exStrip(xQ, ya + pad, sub(qs[i]), vecW / 32, sh, qkvScale)}${exStrip(xK, ya + pad, sub(ks[i]), vecW / 32, sh, qkvScale)}${exStrip(xV, ya + pad, sub(vs[i]), vecW / 32, sh, qkvScale)}</g>`;
    s += exRibbon(xV + vecW, ya + R / 2, xAtt, ya + R / 2, "#8f86e8", i, 7);
    s += `</g>`;
    // band B row
    s += `<g class="ex-row${picked}" data-row="${i}">${tokLabel(i, yb)}`;
    s += `<g data-focus="out">${exStrip(xOut, yb + pad, det.attn_out[i], outW / det.attn_out[i].length, sh, exRowScale(det.attn_out[i]))}</g>`;
    s += exRibbon(xOut + outW, yb + R / 2, xMlp, yb + R / 2, "#c89b3c", i, 7);
    if (c.n_experts > 1) {
      const pr = L.router.probs[i], chosen = L.router.chosen[i];
      s += `<g data-focus="mlp">${pr.map((p, e) => `<rect x="${xMlp + e * 30}" y="${yb + R - 5 - p * (R - 10)}" width="24" height="${Math.max(1.5, p * (R - 10))}" class="ex-router${chosen.includes(e) ? " on" : ""}"/>`).join("")}</g>`;
    } else {
      const hv = L.mlp.hidden[i];
      s += `<g data-focus="mlp">${exStrip(xMlp, yb + pad, hv, mlpW / hv.length, sh, exRowScale(hv))}<text x="${xMlp + mlpW + 10}" y="${yb + R / 2 + 4}" class="ex-num">${Math.round(L.mlp.active[i] * 100)}% active</text></g>`;
    }
    s += exRibbon(xMlp + mlpW + (c.n_experts > 1 ? 0 : 76), yb + R / 2, xStack, yb + R / 2, "#9aa3b8", i, 6);
    s += `</g>`;
  }
  if (c.n_experts > 1) {
    for (let e = 0; e < c.n_experts; e++) s += `<text x="${xMlp + e * 30 + 12}" y="${yB - 8}" class="ex-key" text-anchor="middle">expert ${e + 1}</text>`;
  }
  // key labels above the attention matrix; query labels are the token rows themselves
  for (let j = 0; j < T; j++) {
    s += `<text x="${xAtt + j * C + C / 2 - 2}" y="${yA - 8}" class="ex-key" text-anchor="start" transform="rotate(-55 ${xAtt + j * C + C / 2 - 2} ${yA - 8})">${esc(d.tokens[j].text.trim() || "␠")}</text>`;
  }
  // attention matrix of the chosen head, at the chosen stage
  const att = L.attention[ex.head], sc = det.scores[ex.head];
  const scScale = exMaxAbs(sc.map((row, i) => row.filter((_, j) => j <= i)));
  s += `<g data-focus="attention">`;
  for (let i = 0; i < T; i++) {
    for (let j = 0; j < T; j++) {
      const x = xAtt + j * C, y = rowA(i), future = j > i, sel = ex.cell && ex.cell[0] === i && ex.cell[1] === j;
      let cell, label = "";
      if (ex.stage === "scores") { cell = exCell(x + 1.5, y + 1.5, C - 3, C - 3, sc[i][j], scScale, "ex-acell"); label = sc[i][j].toFixed(1); }
      else if (future) { cell = `<rect class="ex-acell ex-masked" x="${x + 1.5}" y="${y + 1.5}" width="${C - 3}" height="${C - 3}" rx="3"/>`; label = ex.stage === "masked" ? "−∞" : ""; }
      else if (ex.stage === "masked") { cell = exCell(x + 1.5, y + 1.5, C - 3, C - 3, sc[i][j], scScale, "ex-acell"); label = sc[i][j].toFixed(1); }
      else { cell = `<rect class="ex-acell" x="${x + 1.5}" y="${y + 1.5}" width="${C - 3}" height="${C - 3}" rx="3" fill="#7a5cf0" fill-opacity="${(0.06 + 0.94 * att[i][j]).toFixed(3)}"/>`; label = att[i][j] >= 0.05 ? Math.round(att[i][j] * 100) : ""; }
      s += `<g class="ex-ag" data-i="${i}" data-j="${j}">${cell}${label !== "" ? `<text x="${x + C / 2}" y="${y + C / 2 + 4}" class="ex-anum${ex.stage === "softmax" && att[i][j] > 0.55 ? " inv" : ""}" text-anchor="middle">${label}</text>` : ""}${sel ? `<rect x="${x + 1.5}" y="${y + 1.5}" width="${C - 3}" height="${C - 3}" rx="3" class="ex-sel"/>` : ""}</g>`;
    }
  }
  s += `</g>`;
  // the other blocks, as a stack
  const remaining = c.passes - 1;
  s += `<g data-focus="blocks">`;
  for (let k = Math.min(4, remaining); k >= 0; k--) {
    s += `<rect class="ex-stack${k === 0 ? " front" : ""}" x="${xStack + k * 7}" y="${yB - 12 + k * 7}" width="96" height="${T * R + 12}" rx="10"/>`;
  }
  s += `<text x="${xStack + 48}" y="${yB + T * R / 2}" class="ex-stacktxt" text-anchor="middle">×${c.passes}</text>`;
  s += `<text x="${xStack + 48}" y="${yB + T * R / 2 + 20}" class="ex-sub" text-anchor="middle">blocks, then</text>`;
  s += `<text x="${xStack + 48}" y="${yB + T * R / 2 + 36}" class="ex-sub" text-anchor="middle">next token ↓</text></g>`;

  // natural size: scrolls sideways only for long prompts (full screen gives it the whole window)
  $("#ex-svg").innerHTML = `<svg viewBox="0 0 ${W} ${Hh}" class="ex-svgel" style="width:100%;max-width:${Math.round(W * ($("#ex-card").classList.contains("ex-fullscreen") ? 1.35 : 1))}px;min-width:${Math.round(W * 0.92)}px">${bg}${s}</svg>`;
  exTokenPanel();
  exProbBox();
  exFocusClass();
  exPanel();
  exLens();
}

function exTokenPanel() {
  const d = ex.data, box = $("#ex-token");
  if (ex.tok == null || !d || ex.tok >= d.tokens.length) { box.classList.add("hidden"); return; }
  const t = ex.tok, toks = d.tokens.map((x) => x.text.trim() || "␠"), H = d.config.n_head;
  const norms = d.layers.map((L) => L.resid_norm[t]);
  const maxN = Math.max(...norms, d.embedding.token_norm[t]);
  const rows = d.layers.map((L, b) => {
    // which earlier token this token attends to most, averaged over heads, and the single most focused head
    const avg = toks.map((_, j) => L.attention.reduce((a, head) => a + head[t][j], 0) / H);
    const jBest = avg.indexOf(Math.max(...avg));
    let hBest = 0, hVal = -1, hTo = 0;
    L.attention.forEach((head, h) => head[t].forEach((v, j) => { if (v > hVal) { hVal = v; hBest = h; hTo = j; } }));
    const lens = L.lens[t];
    return `<tr class="${b === ex.layer ? "on" : ""}" data-block="${b}"><td class="muted">${b + 1}</td>
      <td><code>${esc(toks[jBest])}</code> <span class="muted small">${Math.round(avg[jBest] * 100)}% avg</span></td>
      <td>head ${hBest + 1} → <code>${esc(toks[hTo])}</code> <span class="muted small">${Math.round(hVal * 100)}%</span></td>
      <td><span class="ex-nbar"><i style="width:${(norms[b] / maxN * 100).toFixed(1)}%"></i></span> <span class="muted small">${norms[b].toFixed(1)}</span></td>
      <td>${lens.map((x, k) => `<code class="${k ? "muted" : ""}">${esc(x.text.replace(/\n/g, "↵")) || "␠"}</code> <span class="muted small">${Math.round(x.p * 100)}%</span>`).join(" &nbsp;")}</td></tr>`;
  }).join("");
  box.classList.remove("hidden");
  box.innerHTML = `<div class="card-head"><h3>Following <code>${esc(d.tokens[t].text)}</code> (token ${t + 1}) through the ${d.config.passes} blocks</h3><button class="btn small" id="ex-token-close">Close</button></div>
    <p class="muted small">Each block, this token's attention (averaged over the ${H} heads, and the head that focuses hardest), the size of its vector (it grows as blocks add to it through the residual connections), and what the model would predict after it if it stopped at that block. Click a row to show that block above.</p>
    <table class="ex-ttab"><tr><th>block</th><th>attends most to</th><th>sharpest head</th><th>vector size</th><th>would predict next (top 3)</th></tr>${rows}</table>`;
  $("#ex-token-close").addEventListener("click", () => { ex.tok = null; exRender(); });
  $$("#ex-token tr[data-block]").forEach((tr) => tr.addEventListener("click", () => action(null, () => exSetLayer(+tr.dataset.block))));
}

function exProbBox() {
  const d = ex.data, last = d.tokens[d.tokens.length - 1].text;
  const probs = exProbs().slice(0, 12);
  const kept = exProbs().filter((p) => !p.cut).length;
  $("#ex-probs").innerHTML = `<div class="ex-ptitle" data-focus="probs">Next token after <code>${esc(last)}</code> <span class="muted small">${kept} of the top ${d.logits.length} kept</span></div>` +
    probs.map((p) => `<div class="ex-prow${p.cut ? " cut" : ""}"><code>${esc(p.text.replace(/\n/g, "↵")) || "␠"}</code><span class="ex-pb"><i style="width:${(p.p * 100).toFixed(1)}%"></i></span><span class="muted small">${p.cut ? "cut" : (p.p * 100).toFixed(1) + "%"}</span></div>`).join("");
}

function exProbs() {
  const lg = ex.data.logits;
  const t = Math.max(0.05, ex.temp);
  const m = Math.max(...lg.map((x) => x.logit / t));
  let ps = lg.map((x) => ({ ...x, p: Math.exp(x.logit / t - m) }));
  const z = ps.reduce((a, b) => a + b.p, 0);
  ps.forEach((x) => { x.p /= z; });  // over the top 60 tokens: the rest hold almost nothing at these settings
  let cum = 0;
  ps.forEach((x, i) => {
    x.cut = (ex.topk > 0 && i >= ex.topk) || (ex.topp < 1 && cum >= ex.topp);
    cum += x.p;
  });
  const kept = ps.filter((x) => !x.cut).reduce((a, b) => a + b.p, 0);
  ps.forEach((x) => { x.p = x.cut ? 0 : x.p / kept; });
  return ps;
}

function exFocusClass() {
  $$("#ex-svg [data-focus]").forEach((g) => g.classList.toggle("ex-dim", ex.focus !== "overview" && ex.focus !== "lens" && g.dataset.focus !== ex.focus));
  $$("#ex-svg .ex-row").forEach((g) => g.classList.toggle("ex-hl", ex.hover != null && +g.dataset.row === ex.hover));
  $$("#ex-svg .ex-rib").forEach((p) => p.classList.toggle("ex-faint", ex.hover != null && +p.dataset.row !== ex.hover));
}

function exPanel() {
  const d = ex.data;
  let html = exExplain(ex.focus);
  if (ex.focus === "attention" && ex.cell) {
    const [i, j] = ex.cell, det = d.layers[ex.layer].detail;
    const q = det.q[ex.head][i], k = det.k[ex.head][j];
    const terms = q.map((v, n) => v * k[n]);
    const dot = terms.reduce((a, b) => a + b, 0), hd = d.config.head_dim;
    const first = terms.slice(0, 6).map((_, n) => `${q[n].toFixed(2)}×${k[n].toFixed(2)}`).join(" + ");
    const w = d.layers[ex.layer].attention[ex.head][i][j];
    html += `<div class="ex-calc"><b>"${esc(d.tokens[i].text)}" looking at "${esc(d.tokens[j].text)}"</b> (head ${ex.head + 1}, block ${ex.layer + 1})<br>
      q · k = ${first} + … (${hd} terms) = <b>${dot.toFixed(2)}</b><br>
      score = ${dot.toFixed(2)} / √${hd} = <b>${(dot / Math.sqrt(hd)).toFixed(2)}</b>${j > i ? " → masked: a future token" : `<br>after softmax over the row: <b>${(w * 100).toFixed(1)}%</b> of this token's attention`}</div>`;
  }
  $("#ex-explain").innerHTML = html;
  $$(".ex-tab").forEach((b) => b.classList.toggle("on", b.dataset.focus === ex.focus));
}

function exLens() {
  const d = ex.data, T = d.tokens.length;
  const head = `<tr><th>after block</th>${d.tokens.map((t) => `<th>${esc(t.text.trim() || "␠")}</th>`).join("")}</tr>`;
  const rows = d.layers.map((L, b) => `<tr class="${b === ex.layer ? "on" : ""}"><td class="muted">${b + 1}</td>${L.lens.map((cands, t) => {
    const top = cands[0];
    return `<td title="${cands.map((x) => `${x.text} ${(x.p * 100).toFixed(1)}%`).join("  ·  ")}" style="--p:${top.p.toFixed(3)}" class="${t === T - 1 ? "last" : ""}">${esc(top.text.replace(/\n/g, "↵")) || "␠"}<span>${Math.round(top.p * 100)}%</span></td>`;
  }).join("")}</tr>`).join("");
  $("#ex-lens").innerHTML = `<table class="ex-lenstab">${head}${rows}</table>`;
}

// ---------- loading and controls ----------
async function exRun() {
  const prompt = $("#a-prompt").value;
  const d = await api("/api/explain", { prompt, layer: ex.layer });
  ex.data = d;
  ex.layer = d.layer;
  ex.cell = null;
  $("#ex-wrap").classList.remove("hidden");
  $("#ex-heads").innerHTML = Array.from({ length: d.config.n_head }, (_, h) => `<button class="seg-btn${h === ex.head ? " on" : ""}" data-head="${h}">${h + 1}</button>`).join("");
  $("#ex-layer-label").textContent = `Block ${ex.layer + 1} of ${d.config.passes}`;
  const name = ui.status?.active?.name;
  $("#ex-source").textContent = d.source === "untrained" ? "Model: untrained (random weights)" : `Model: ${name || "the trained model"}`;
  exRender();
}

async function exSetLayer(n) {
  const d = ex.data;
  if (!d) return;
  const passes = d.config.passes;
  ex.layer = (n + passes) % passes;
  if (!d.layers[ex.layer].detail && ex.view !== "flow") await exRun(); else { $("#ex-layer-label").textContent = `Block ${ex.layer + 1} of ${passes}`; exRender(); }
}

function exInit() {
  $("#ex-svg").addEventListener("mouseover", (e) => {
    const row = e.target.closest(".ex-row"), ag = e.target.closest(".ex-ag");
    const r = row ? +row.dataset.row : ag ? +ag.dataset.i : null;
    if (r !== ex.hover) { ex.hover = r; exFocusClass(); }
  });
  $("#ex-svg").addEventListener("mouseleave", () => { ex.hover = null; exFocusClass(); });
  $("#ex-svg").addEventListener("click", (e) => {
    const ag = e.target.closest(".ex-ag");
    const f = e.target.closest("[data-focus]");
    const tk = e.target.closest(".ex-tokg");
    if (tk) { ex.tok = ex.tok === +tk.dataset.tok ? null : +tk.dataset.tok; exRender(); if (ex.tok != null) $("#ex-token").scrollIntoView({ behavior: "smooth", block: "nearest" }); return; }
    if (ag) { ex.cell = [+ag.dataset.i, +ag.dataset.j]; ex.focus = "attention"; exRender(); return; }
    if (f) { ex.focus = f.dataset.focus; ex.cell = null; exRender(); }
  });
  $("#ex-heads").addEventListener("click", (e) => {
    const b = e.target.closest("[data-head]");
    if (!b) return;
    ex.head = +b.dataset.head;
    $$("#ex-heads .seg-btn").forEach((x) => x.classList.toggle("on", x === b));
    exRender();
  });
  $$("#ex-stage .seg-btn").forEach((b) => b.addEventListener("click", () => {
    ex.stage = b.dataset.stage; ex.focus = "attention";
    $$("#ex-stage .seg-btn").forEach((x) => x.classList.toggle("on", x === b));
    exRender();
  }));
  try { ex.view = localStorage.getItem("lmstudio.exview") || "flow"; } catch (e) { /* ignore */ }
  $$("#ex-viewsel .seg-btn").forEach((b) => b.addEventListener("click", () => {
    ex.view = b.dataset.exview;
    try { localStorage.setItem("lmstudio.exview", ex.view); } catch (e) { /* ignore */ }
    exRender();
  }));
  $("#ex-full").addEventListener("click", () => exFull(!$("#ex-card").classList.contains("ex-fullscreen")));
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && $("#ex-card").classList.contains("ex-fullscreen")) exFull(false); });
  $("#ex-prev").addEventListener("click", () => action($("#ex-prev"), () => exSetLayer(ex.layer - 1)));
  $("#ex-next").addEventListener("click", () => action($("#ex-next"), () => exSetLayer(ex.layer + 1)));
  $$(".ex-tab").forEach((b) => b.addEventListener("click", () => { ex.focus = b.dataset.focus; ex.cell = null; exRender(); }));
  const sl = [["#ex-temp", "temp", (v) => v.toFixed(1)], ["#ex-topk", "topk", (v) => (v === 0 ? "off" : v)], ["#ex-topp", "topp", (v) => (v >= 1 ? "off" : v.toFixed(2))]];
  sl.forEach(([sel, key, show]) => {
    const el = $(sel), out = $(`${sel}-v`);
    const sync = () => { ex[key] = +el.value; out.textContent = show(+el.value); if (ex.data) { ex.focus = "probs"; exRender(); } };
    el.addEventListener("input", sync);
    out.textContent = show(+el.value);
  });
  $$(".a-view").forEach((b) => b.addEventListener("click", () => {
    const v = b.dataset.view;
    $$(".a-view").forEach((x) => x.classList.toggle("on", x === b));
    $("#a-walk-view").classList.toggle("hidden", v !== "walk");
    $("#a-explorer-view").classList.toggle("hidden", v !== "explorer");
    $("#a-run").textContent = v === "explorer" ? "Explore" : "Walk through";
    document.body.classList.toggle("ex-wide", v === "explorer");
    try { localStorage.setItem("lmstudio.aview", v); } catch (e) { /* ignore */ }
    if (v === "explorer" && !ex.data && $("#a-prompt").value.trim()) action($("#a-run"), exRun);
  }));
  try { if (localStorage.getItem("lmstudio.aview") === "explorer") $('.a-view[data-view="explorer"]').click(); } catch (e) { /* ignore */ }
}

function exFull(on) {
  // moved to <body> while full screen: inside the step panel, its fade-in animation would pin a fixed
  // element to the panel instead of the window
  const card = $("#ex-card");
  if (on && !ex.placeholder) {
    ex.placeholder = document.createComment("ex-card");
    card.before(ex.placeholder);
    document.body.appendChild(card);
  } else if (!on && ex.placeholder) {
    ex.placeholder.replaceWith(card);
    ex.placeholder = null;
  }
  card.classList.toggle("ex-fullscreen", on);
  document.body.classList.toggle("ex-noscroll", on);
  $("#ex-full").textContent = on ? "Exit full screen (Esc)" : "Full screen";
  if (ex.data) exRender();
}

function exActive() { return !$("#a-explorer-view").classList.contains("hidden"); }

exInit();

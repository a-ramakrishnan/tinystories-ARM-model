// Explorer → Flow: the same real data as the Diagram, drawn as ribbons that carry each token from stage to stage.
// Every head's Q/K/V tabs → the chosen head's card (keys, queries, values, the attention dots, out) → the
// feed-forward layer as a dip into a wider hidden layer → the other blocks → the next-token probabilities.
// Uses the Explorer's state (ex), helpers and event handlers (explorer.js): rows carry data-row, attention dots
// are .ex-ag with data-i / data-j, parts carry data-focus, so hover, clicks and explanations work the same.

const FL = { Q: "#4f7fe0", K: "#d65a5a", V: "#2f9e66", A: "#7a5cf0", M: "#7d9be6" };

function flCurve(x1, y1, x2, y2, k = 0.5) {
  const dx = (x2 - x1) * k;
  return `M${x1},${y1} C${x1 + dx},${y1} ${x2 - dx},${y2} ${x2},${y2}`;
}
// a filled ribbon whose two ends have their own heights: (x1, top1..bot1) → (x2, top2..bot2)
function flBand(x1, t1, b1, x2, t2, b2, k = 0.5) {
  const dx = (x2 - x1) * k;
  return `M${x1},${t1} C${x1 + dx},${t1} ${x2 - dx},${t2} ${x2},${t2} L${x2},${b2} C${x2 - dx},${b2} ${x1 + dx},${b1} ${x1},${b1} Z`;
}
const flText = (x, y, t, cls, anchor = "start", extra = "") => `<text x="${x}" y="${y}" class="${cls}" text-anchor="${anchor}" ${extra}>${t}</text>`;
const flTok = (t) => esc(t.replace(/\n/g, "↵")) || "␠";

function exRenderFlow(d, c, T, L, H) {
  const att = L.attention[ex.head];
  const TOP = 86;
  // ---------- sizes ----------
  const tab = 15, tabGap = 2, groupGap = 9, groupH = 3 * tab + 2 * tabGap;
  const headsH = H * (groupH + groupGap) - groupGap;
  const S = Math.max(18, Math.min(27, 320 / T));        // spacing of the card's lists = attention dot pitch
  const listH = T * S;
  const xTokR = 118, xEmb = 124, xTab = 214;
  const x0 = 262, cardW = 520, y0 = TOP;                 // the head card
  const yKey = y0 + 64, yQry = yKey + listH + 70, yVal = yQry + listH + 70;
  const cardH = yVal + listH + 64 - y0;
  const yK = (j) => yKey + j * S + S / 2, yQ = (i) => yQry + i * S + S / 2, yV = (j) => yVal + j * S + S / 2;
  const xLab = x0 + 140, xTick = x0 + 146, xM0 = x0 + 222, colX = (j) => xM0 + j * S + S / 2, xOut = x0 + cardW - 60;
  const span = Math.max(headsH, cardH);
  const yT = (i) => TOP + span * (i + 0.5) / T;          // token rows on the far left, spread over the height
  // feed-forward section
  const MR = Math.max(26, Math.min(48, 560 / T));        // row pitch
  const xm = x0 + cardW + 50 + 120, yM = (i) => y0 + 40 + i * MR;
  const ratio = (L.mlp?.width || 4 * c.n_embd) / c.n_embd;
  const xh = xm + 180, yH0 = yM(0), yH1 = Math.min(yM(T - 1) + T * MR * (ratio - 1) + 40, y0 + span + 260);
  const ybot = (i) => yM(T - 1) + 24 + (yH1 - yM(T - 1) - 24) * (i + 1) / T;
  const xRes2 = xh + 180, xBlk = xRes2 + 36, xLast = xBlk + 96, xProb = xLast + 110;
  const probs = exProbs().slice(0, 18), PR = 26;
  const W = xProb + 300, Hh = Math.max(TOP + span, yH1, y0 + probs.length * PR) + 60;
  let s = "";

  // ---------- titles ----------
  s += flText(xEmb - 20, 26, "Embedding", "ex-h", "start", 'data-focus="embedding"');
  s += flText(xTab - 10, 26, "Q · K · V", "ex-h", "start", 'data-focus="qkv"') + flText(xTab - 10, 44, `${H} heads`, "ex-sub");
  s += flText(x0 + 10, 26, "Multi-head self-attention", "ex-h", "start", 'data-focus="attention"') + flText(x0 + 10, 44, `block ${ex.layer + 1} of ${c.passes}`, "ex-sub");
  s += flText(xm + 100, 26, c.n_experts > 1 ? "Experts" : "Feed-forward (MLP)", "ex-h", "start", 'data-focus="mlp"') +
    flText(xm + 100, 44, c.n_experts > 1 ? `each token to ${c.moe_top_k} of ${c.n_experts} experts` : `${c.n_embd} → ${L.mlp.width} → ${c.n_embd} numbers per token`, "ex-sub");
  s += flText(xBlk - 6, 26, `×${c.passes} blocks`, "ex-h", "start", 'data-focus="blocks"');
  s += flText(xProb, 26, "Probabilities", "ex-h", "start", 'data-focus="probs"') + flText(xProb, 44, `next token after "${esc(d.tokens[T - 1].text.trim())}"`, "ex-sub");

  // ---------- tokens and the fan into every head's Q/K/V ----------
  const tabY = (h, k) => TOP + h * (groupH + groupGap) + k * (tab + tabGap);
  let fan = "";
  for (let i = 0; i < T; i++) {
    for (let h = 0; h < H; h++) {
      for (let k = 0; k < 3; k++) {
        const col = [FL.Q, FL.K, FL.V][k];
        fan += `<path class="ex-rib fl-fan${h === ex.head ? " on" : ""}" data-row="${i}" d="${flCurve(xEmb + 14, yT(i), xTab, tabY(h, k) + tab / 2, 0.6)}" stroke="${col}" stroke-width="${h === ex.head ? 1.6 : 1}" fill="none"/>`;
      }
    }
  }
  s += `<g data-focus="qkv">${fan}</g>`;
  for (let i = 0; i < T; i++) {
    s += `<g class="ex-row" data-row="${i}"><g class="ex-tokg" data-tok="${i}" data-focus="embedding"><rect x="4" y="${yT(i) - 13}" width="${xEmb + 18}" height="26" rx="6" class="ex-tokbg"/>` +
      flText(xTokR, yT(i) + 5, flTok(d.tokens[i].text), "ex-tok", "end") + `<rect x="${xEmb}" y="${yT(i) - 10}" width="14" height="20" rx="3" class="fl-emb"/></g></g>`;
  }
  s += `<g data-focus="qkv">`;
  for (let h = 0; h < H; h++) {
    ["Q", "K", "V"].forEach((lab, k) => {
      s += `<g class="fl-tab${h === ex.head ? " on" : ""}" data-headpick="${h}"><rect x="${xTab}" y="${tabY(h, k)}" width="24" height="${tab}" rx="3" fill="${FL[lab]}"/>` +
        flText(xTab + 12, tabY(h, k) + tab - 3.5, lab, "fl-tabtxt", "middle") + `</g>`;
    });
  }
  s += `</g>`;

  // ---------- the head card (other heads stacked behind) ----------
  for (let k = Math.min(5, H - 1); k >= 1; k--) {
    s += `<rect class="fl-card back" x="${x0 + k * 7}" y="${y0 + k * 7}" width="${cardW}" height="${cardH}" rx="10"/>`;
  }
  s += `<rect class="fl-card" x="${x0}" y="${y0}" width="${cardW}" height="${cardH}" rx="10"/>`;
  // lines from the chosen head's tabs to the card's lists
  const tk = (k) => tabY(ex.head, k) + tab / 2;
  let lines = "";
  for (let i = 0; i < T; i++) {
    lines += `<path class="ex-rib fl-line" data-row="${i}" d="${flCurve(xTab + 24, tk(1), xTick, yK(i), 0.55)}" stroke="${FL.K}"/>`;
    lines += `<path class="ex-rib fl-line" data-row="${i}" d="${flCurve(xTab + 24, tk(0), xTick, yQ(i), 0.55)}" stroke="${FL.Q}"/>`;
    lines += `<path class="ex-rib fl-line" data-row="${i}" d="${flCurve(xTab + 24, tk(2), xTick, yV(i), 0.55)}" stroke="${FL.V}"/>`;
  }
  s += `<g data-focus="qkv">${lines}</g>`;
  s += `<g data-focus="attention">`;
  s += flText(xLab + 10, yKey - 14, "Key", "fl-lt", "middle", `style="fill:${FL.K}"`);
  s += flText(xLab + 10, yQry - 14, "Query", "fl-lt", "middle", `style="fill:${FL.Q}"`);
  s += flText(xLab + 10, yVal - 14, "Value", "fl-lt", "middle", `style="fill:${FL.V}"`);
  // key lines run right to their column, then down into the attention dots
  for (let j = 0; j < T; j++) {
    const x = colX(j), y = yK(j), r = 7;
    s += `<path class="ex-rib fl-kline" data-row="${j}" d="M${xTick + 14},${y} H${x - r} Q${x},${y} ${x},${y + r} V${yQ(0) - S / 2}" stroke="${FL.K}"/>`;
  }
  // query lines run straight into their row of dots
  for (let i = 0; i < T; i++) s += `<path class="ex-rib fl-kline" data-row="${i}" d="M${xTick + 14},${yQ(i)} H${xM0}" stroke="${FL.Q}"/>`;
  // value ribbon up to Out, and the fan from the dots to Out
  s += `<path class="fl-vband" d="${flBand(xTick + 14, yV(0) - S / 2, yV(T - 1) + S / 2, xOut - 6, yQ(0) - S / 2, yQ(T - 1) + S / 2, 0.55)}"/>`;
  s += `<path class="fl-oband" d="${flBand(xM0 + T * S + 4, yQ(0) - S / 2, yQ(T - 1) + S / 2, xOut - 6, yQ(0) - S / 2, yQ(T - 1) + S / 2, 0.5)}"/>`;
  for (let i = 0; i < T; i++) {
    const label = flTok(d.tokens[i].text);
    s += `<g class="ex-row" data-row="${i}">` +
      flText(xLab, yK(i) + 4, label, "fl-lab", "end") + `<rect x="${xTick}" y="${yK(i) - 1.5}" width="14" height="3" fill="${FL.K}"/>` +
      flText(xLab, yQ(i) + 4, label, "fl-lab", "end") + `<rect x="${xTick}" y="${yQ(i) - 1.5}" width="14" height="3" fill="${FL.Q}"/>` +
      flText(xLab, yV(i) + 4, label, "fl-lab", "end") + `<rect x="${xTick}" y="${yV(i) - 1.5}" width="14" height="3" fill="${FL.V}"/>` +
      `<rect x="${xOut}" y="${yQ(i) - 2}" width="18" height="4" rx="1" fill="${FL.A}"/></g>`;
  }
  s += flText(xOut + 9, yQry - 14, "Out", "fl-lt", "middle", `style="fill:${FL.A}"`);
  // the attention dots: size and colour follow the weight; future positions are masked
  for (let i = 0; i < T; i++) {
    for (let j = 0; j < T; j++) {
      const sel = ex.cell && ex.cell[0] === i && ex.cell[1] === j, w = att[i][j], cx = colX(j), cy = yQ(i);
      const dot = j > i ? `<circle cx="${cx}" cy="${cy}" r="${S * 0.36}" class="fl-dot masked"/>`
        : `<circle cx="${cx}" cy="${cy}" r="${S * 0.36}" fill="${FL.A}" fill-opacity="${(0.1 + 0.9 * w).toFixed(3)}" class="fl-dot"/>`;
      s += `<g class="ex-ag" data-i="${i}" data-j="${j}"><title>${esc(d.tokens[i].text)} → ${esc(d.tokens[j].text)}: ${j > i ? "masked" : (w * 100).toFixed(1) + "%"}</title>${dot}${sel ? `<circle cx="${cx}" cy="${cy}" r="${S * 0.46}" class="ex-sel"/>` : ""}</g>`;
    }
  }
  s += flText(xM0 + T * S / 2, yQ(T - 1) + S + 18, "Attention", "fl-lt", "middle", `style="fill:${FL.A}"`);
  // head navigation inside the card
  const nx = x0 + cardW - 150, ny = y0 + cardH - 22;
  s += flText(nx, ny, `Head ${ex.head + 1} of ${H}`, "fl-nav") +
    `<g class="fl-navbtn" data-headnav="-1"><rect x="${nx + 98}" y="${ny - 16}" width="22" height="22" rx="5"/>${flText(nx + 109, ny, "‹", "fl-navtxt", "middle")}</g>` +
    `<g class="fl-navbtn" data-headnav="1"><rect x="${nx + 124}" y="${ny - 16}" width="22" height="22" rx="5"/>${flText(nx + 135, ny, "›", "fl-navtxt", "middle")}</g>`;
  s += `</g>`;

  // ---------- out → feed-forward rows ----------
  s += `<g data-focus="out">`;
  for (let i = 0; i < T; i++) {
    s += `<path class="ex-rib fl-outline" data-row="${i}" d="${flCurve(xOut + 18, yQ(i), xm + 8, yM(i), 0.5)}" stroke="${FL.A}"/>`;
  }
  s += `</g><g data-focus="mlp">`;
  // each token's ribbon dips into the wider hidden layer and comes back up (the residual runs alongside)
  const bw = Math.max(8, MR - 16);
  for (let i = T - 1; i >= 0; i--) {
    const y = yM(i), yb = ybot(i);
    const path = `M${xm + 24},${y} C${xm + 100},${y} ${xh - 60},${yb} ${xh},${yb} C${xh + 60},${yb} ${xRes2 - 80},${y} ${xRes2},${y}`;
    s += `<path d="${path}" class="fl-dipgap" stroke-width="${bw + 5}"/>`;
    s += `<path d="${path}" class="ex-rib fl-dip" data-row="${i}" stroke-width="${bw}"/>`;
  }
  s += `<rect x="${xh - 7}" y="${ybot(0) - bw / 2 - 6}" width="14" height="${ybot(T - 1) - ybot(0) + bw + 12}" rx="5" class="fl-hidden"/>`;
  for (let i = 0; i < T; i++) {
    const a = c.n_experts > 1 ? Math.max(...L.router.probs[i]) : L.mlp.active[i];
    s += `<circle cx="${xh}" cy="${ybot(i)}" r="4.5" class="fl-act" fill-opacity="${(0.25 + 0.75 * Math.min(1, a * (c.n_experts > 1 ? 1 : 4))).toFixed(2)}"><title>${esc(d.tokens[i].text)}: ${c.n_experts > 1 ? `router's top expert ${(a * 100).toFixed(0)}%` : `${(a * 100).toFixed(0)}% of the hidden layer strongly active`}</title></circle>`;
  }
  s += flText(xh, ybot(T - 1) + bw / 2 + 26, c.n_experts > 1 ? "experts" : `hidden layer: ${L.mlp.width} numbers (${ratio.toFixed(1)}× wider)`, "ex-sub", "middle");
  // residual brackets and bars
  s += `<path d="M${xm + 8},${yM(0) - MR / 2 - 4} q0,-14 18,-14 H${xRes2 - 18} q18,0 18,14" class="fl-resid"/>` + flText((xm + xRes2) / 2, yM(0) - MR / 2 - 22, "residual: added back", "ex-sub", "middle");
  for (let i = 0; i < T; i++) {
    s += `<g class="ex-row" data-row="${i}">${flText(xm - 4, yM(i) + 5, flTok(d.tokens[i].text), "ex-tok", "end")}` +
      `<rect x="${xm + 8}" y="${yM(i) - bw / 2}" width="16" height="${bw}" rx="3" class="fl-res"/>` +
      `<rect x="${xRes2}" y="${yM(i) - bw / 2}" width="16" height="${bw}" rx="3" class="fl-res"/></g>`;
  }
  s += `</g>`;

  // ---------- the other blocks, then the probabilities ----------
  s += `<g data-focus="blocks">`;
  for (let i = 0; i < T; i++) {
    s += `<path class="ex-rib fl-outline" data-row="${i}" d="${flCurve(xRes2 + 16, yM(i), xBlk, yM(i), 0.5)}" stroke="${FL.M}"/>`;
    for (let k = 0; k < 4; k++) s += `<rect x="${xBlk + k * 14}" y="${yM(i) - bw / 2}" width="9" height="${bw}" rx="2" class="fl-blk" fill-opacity="${0.35 + k * 0.15}"/>`;
  }
  s += flText(xBlk + 28, yM(T - 1) + bw / 2 + 20, `${c.passes - 1 - ex.layer} more`, "ex-sub", "middle");
  s += `</g><g data-focus="probs">`;
  const yl = yM(T - 1);
  s += `<path class="fl-pband" d="${flBand(xBlk + 60, yl - bw / 2, yl + bw / 2, xProb + 10, y0 + 6, y0 + probs.length * PR, 0.5)}"/>`;
  s += flText(xLast + 10, yl + 5, flTok(d.tokens[T - 1].text), "ex-tok", "middle");
  const top = probs.reduce((a, b) => (b.p > a.p ? b : a), probs[0]);
  probs.forEach((p, k) => {
    const y = y0 + 20 + k * PR, on = p === top;
    s += flText(xProb + 120, y + 5, flTok(p.text), `fl-ptok${p.cut ? " cut" : ""}${on ? " top" : ""}`, "end");
    s += `<rect x="${xProb + 130}" y="${y - 2}" width="${Math.max(2, p.p * 110)}" height="${on ? 5 : 4}" rx="2" class="fl-pbar${p.cut ? " cut" : ""}${on ? " top" : ""}"/>`;
    s += flText(xProb + 136 + Math.max(2, p.p * 110), y + 5, p.cut ? "0%" : `${(p.p * 100).toFixed(2)}%`, `fl-ppct${on ? " top" : ""}`);
  });
  s += `</g>`;

  $("#ex-svg").innerHTML = `<svg viewBox="0 0 ${W} ${Hh}" class="ex-svgel" style="width:100%;max-width:${Math.round(W * ($("#ex-card").classList.contains("ex-fullscreen") ? 1.15 : 1))}px;min-width:${Math.round(W * 0.8)}px">${s}</svg>`;
}

// head picks inside the Flow (a Q/K/V tab, or the ‹ › in the card)
document.addEventListener("click", (e) => {
  if (!ex.data || ex.view !== "flow") return;
  const nav = e.target.closest?.("[data-headnav]"), pick = e.target.closest?.("[data-headpick]");
  if (!nav && !pick) return;
  const H = ex.data.config.n_head;
  ex.head = pick ? +pick.dataset.headpick : (ex.head + +nav.dataset.headnav + H) % H;
  $$("#ex-heads .seg-btn").forEach((b) => b.classList.toggle("on", +b.dataset.head === ex.head));
  ex.focus = "attention";
  exRender();
});

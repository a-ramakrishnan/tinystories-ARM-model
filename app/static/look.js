// The look shared by the app and the website (pages/build.py copies this file): the light/dark switch and the dot
// pattern behind the page heads.
//
// Theme: the page follows the system until the reader presses #theme-btn; the choice is kept in localStorage and set
// as data-theme on <html> (a one-line script in each page's <head> applies it before the first paint).
//
// Tokens: a grid of small rounded squares, one per token, showing the project's two ways to write. Bright bands sweep
// left to right along the rows, each row at its own pace (a GPT writing one token after another), and scattered
// tokens light up and fade (a diffusion model revealing positions in any order). On the home page's hero the middle
// stays clear, behind the words; in a page head (.panel-head) they gather on the right and fade out towards the text.
(() => {
  const root = document.documentElement;
  const btn = document.getElementById("theme-btn");
  if (btn) {
    const dark = () => (root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches);
    btn.addEventListener("click", () => {
      const t = dark() ? "light" : "dark";
      root.dataset.theme = t;
      try { localStorage.setItem("site-theme", t); } catch (e) {}
      window.dispatchEvent(new Event("themechange"));
    });
  }

  const still = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const fields = [];
  document.querySelectorAll(".home-hero, .panel-head").forEach((host) => {
    const cv = document.createElement("canvas");
    cv.className = "dots";
    cv.setAttribute("aria-hidden", "true");
    host.prepend(cv);
    fields.push({ cv, ctx: cv.getContext("2d"), centre: host.classList.contains("home-hero"), visible: true, w: 0, h: 0, colour: "" });
  });
  if (!fields.length) return;

  const size = (f) => {
    const r = f.cv.getBoundingClientRect(), d = Math.min(2, devicePixelRatio || 1);
    f.w = r.width; f.h = r.height;
    f.cv.width = Math.max(1, f.w * d); f.cv.height = Math.max(1, f.h * d);
    f.ctx.setTransform(d, 0, 0, d, 0, 0);
    f.colour = getComputedStyle(f.cv).color;
  };
  // a fixed pseudo-random number per grid cell, so each token keeps its own rhythm
  const hash = (i, j) => { const v = Math.sin(i * 127.1 + j * 311.7) * 43758.5453; return v - Math.floor(v); };
  const draw = (f, t) => {
    const { ctx, w, h } = f;
    if (!w || !h) return;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = f.colour;
    const gap = w < 600 ? 18 : 22;
    let fadeAt;
    if (f.centre) {
      const cx = w / 2, cy = h * 0.47, clear = Math.min(w, h) * 0.38, edge = Math.hypot(w, h) * 0.5;
      const squash = w > 700 ? 0.55 : 0.8;                             // the words are wide: keep a wide oval clear
      fadeAt = (x, y) => {
        const d = Math.hypot((x - cx) * squash, y - cy);
        return Math.min(1, Math.max(0, (d - clear) / (clear * 0.9))) * Math.min(1, (edge * 1.1 - d) / (edge * 0.5));
      };
    } else {
      const from = w * 0.72;
      fadeAt = (x) => Math.min(1, Math.max(0, ((x - from) / (w - from)) * 1.3));
    }
    const span = w + 240;
    for (let j = 0, y = gap / 2; y < h; j++, y += gap) {
      const speed = 0.045 + 0.04 * hash(j, 7);                       // px per ms: each row writes at its own pace
      const head = ((t * speed + hash(j, 3) * span) % span) - 120;     // where this row's writing has got to
      for (let i = 0, x = gap / 2; x < w; i++, x += gap) {
        const fade = fadeAt(x, y);
        if (fade <= 0.01) continue;
        const behind = head - x;                                        // tokens just written glow, then settle
        const sweep = behind >= 0 ? Math.exp(-behind / 70) : Math.exp(-((behind / 14) ** 2));
        const phase = hash(i, j), period = 5200 + 4200 * hash(j, i);
        const reveal = Math.max(0, Math.sin((t / period + phase) * 6.2832)) ** 10;   // brief, now and then
        const lit = Math.min(1, Math.max(sweep, reveal));
        ctx.globalAlpha = fade * (0.1 + 0.72 * lit);
        const s = (1.6 + 2.2 * lit) * (0.6 + 0.4 * fade);
        ctx.beginPath();
        if (ctx.roundRect) ctx.roundRect(x - s, y - s, 2 * s, 2 * s, s * 0.45); else ctx.rect(x - s, y - s, 2 * s, 2 * s);
        ctx.fill();
      }
    }
    ctx.globalAlpha = 1;
  };

  let raf = 0, last = 0;
  const loop = (t) => {
    last = t;
    fields.forEach((f) => f.visible && draw(f, t));
    raf = still ? 0 : requestAnimationFrame(loop);
  };
  const recolour = () => setTimeout(() => fields.forEach((f) => { f.colour = getComputedStyle(f.cv).color; draw(f, last); }), 30);
  addEventListener("themechange", recolour);
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", recolour);
  // a page head can be hidden (the app's steps) or change size: size each canvas when its box changes
  const ro = new ResizeObserver((es) => es.forEach((e) => {
    const f = fields.find((x) => x.cv === e.target);
    if (f) { size(f); draw(f, last); }
  }));
  const io = new IntersectionObserver((es) => es.forEach((e) => {
    const f = fields.find((x) => x.cv === e.target);
    if (f) f.visible = e.isIntersecting;
  }));
  fields.forEach((f) => { size(f); ro.observe(f.cv); io.observe(f.cv); });
  loop(0);
})();

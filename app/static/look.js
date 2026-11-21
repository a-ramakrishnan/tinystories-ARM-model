// The look shared by the app and the website (pages/build.py copies this file): the light/dark switch and the dot
// pattern behind the page heads.
//
// Theme: the page follows the system until the reader presses #theme-btn; the choice is kept in localStorage and set
// as data-theme on <html> (a one-line script in each page's <head> applies it before the first paint).
//
// Dots: a grid of dots with a slow ripple. On the home page's hero they're clear in the middle, behind the words;
// in a page head (.panel-head) they're dense on the right and fade out towards the text.
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
  const draw = (f, t) => {
    const { ctx, w, h } = f;
    if (!w || !h) return;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = f.colour;
    const gap = w < 600 ? 18 : 22;
    let cx, cy, fadeAt;
    if (f.centre) {
      cx = w / 2; cy = h * 0.46;
      const clear = Math.min(w, h) * 0.34, edge = Math.hypot(w, h) * 0.5;
      fadeAt = (x, d) => Math.min(1, Math.max(0, (d - clear) / (clear * 0.9))) * Math.min(1, (edge * 1.1 - d) / (edge * 0.5));
    } else {
      cx = w * 0.92; cy = h * 0.5;
      const from = w * 0.72;
      fadeAt = (x) => Math.min(1, Math.max(0, ((x - from) / (w - from)) * 1.3));
    }
    for (let y = gap / 2; y < h; y += gap) {
      for (let x = gap / 2; x < w; x += gap) {
        const d = Math.hypot((x - cx) * 0.8, y - cy);
        const fade = fadeAt(x, d);
        if (fade <= 0.01) continue;
        const wave = 0.5 + 0.5 * Math.sin(d / 26 - t / 900);
        ctx.globalAlpha = fade * (0.12 + 0.6 * wave * wave);
        ctx.beginPath();
        ctx.arc(x, y, 0.8 + 1.5 * wave * fade, 0, 6.2832);
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

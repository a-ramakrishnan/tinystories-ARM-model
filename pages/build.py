# The project's website for GitHub Pages (plan G): the documentation as pages, plus the app's Journey timeline and
# a Side by side replay that run in the browser from saved data (no server, no model).
#
#   python3 pages/build.py [--out _site]        then open _site/index.html, or: python3 -m http.server -d _site
#
# Inputs: the Markdown docs (README, docs/, diffusion_models/README.md, the model cards), the results store
# (results/results.jsonl) for the headline numbers, app/static (styles, journey.js, sidebyside.js) and
# pages/recorded/sidebyside.json (recorded by pages/record.py), and outputs/f4/*.json (diffusion_models/compare.py)
# for the AR vs diffusion page. Links between docs become links between pages; links
# to code go to GitHub. .github/workflows/pages.yml builds and deploys it on every push to main.
import argparse
import html
import json
import os
import re
import shutil
import time

import markdown

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REPO = "https://github.com/a-ramakrishnan/tinystories-ARM-model"
HF = "https://huggingface.co/aramkris"

# (source, page, title, group)
DOCS = [
    ("README.md", "overview.html", "Overview", "Docs"),
    ("docs/usage.md", "usage.html", "How to use it", "Docs"),
    ("docs/components.md", "components.html", "Understanding the components", "Docs"),
    ("docs/journey.md", "story.html", "The story", "Docs"),
    ("docs/analysis.md", "analysis.html", "Analysis", "Docs"),
    ("docs/results.md", "results.html", "Results", "Docs"),
    ("docs/plan.md", "plan.html", "The plan", "Docs"),
    ("diffusion_models/README.md", "diffusion.html", "The diffusion model", "Docs"),
    ("docs/model_cards/fineweb-edu-124m-base.md", "card-base.html", "Base model (124M)", "Models"),
    ("docs/model_cards/fineweb-edu-124m-chat-sft.md", "card-chat-sft.html", "Chat model, SFT", "Models"),
    ("docs/model_cards/fineweb-edu-124m-chat-dpo.md", "card-chat-dpo.html", "Chat model, SFT + DPO", "Models"),
    ("docs/model_cards/fineweb-edu-124m-diffusion.md", "card-diffusion.html", "Diffusion model (124M)", "Models"),
    ("docs/model_cards/fineweb-edu-124m-base-gptq-4bit.md", "card-gptq.html", "Base model, GPTQ 4-bit", "Models"),
]
APPS = [("try.html", "Try it"), ("journey.html", "Journey"), ("explorer.html", "Explorer"), ("sidebyside.html", "Side by side"),
        ("compare.html", "AR vs diffusion")]
PAGE_OF = {src: page for src, page, _, _ in DOCS}
TASKS = [("hellaswag", "HellaSwag", "acc_norm"), ("lambada", "LAMBADA", "acc"), ("arc_easy", "ARC-Easy", "acc"),
         ("piqa", "PIQA", "acc"), ("winogrande", "Winogrande", "acc")]
GPT2 = {"hellaswag": 0.3114, "lambada": 0.3256, "arc_easy": 0.4381, "piqa": 0.6289, "winogrande": 0.5162}


def slug(text, sep="-"):
    """GitHub's heading anchors, so links like journey.md#37-gptq-... keep working."""
    text = re.sub(r"<[^>]+>", "", text).strip().lower()
    return re.sub(r"[^\w\- ]", "", text).replace(" ", sep)


# ---------- links ----------

def rewrite_links(body, src, assets):
    """Docs → pages, images → copied assets, anything else in the repo → GitHub."""
    base = os.path.dirname(src)

    def fix(m):
        attr, url = m.group(1), html.unescape(m.group(2))
        if re.match(r"^[a-z]+:|^#|^//", url):
            return m.group(0)
        path, _, frag = url.partition("#")
        target = os.path.normpath(os.path.join(base, path)) if path else src
        if target in PAGE_OF:
            new = PAGE_OF[target] + (f"#{frag}" if frag else "")
        elif target.startswith("docs/screenshots/") or re.search(r"\.(png|jpe?g|gif|svg)$", target, re.I):
            assets.add(target)
            new = "assets/" + target.replace("/", "_")
        else:
            kind = "tree" if os.path.isdir(os.path.join(ROOT, target)) else "blob"
            new = f"{REPO}/{kind}/main/{target}" + (f"#{frag}" if frag else "")
        return f'{attr}="{html.escape(new, quote=True)}"'
    return re.sub(r'\b(href|src)="([^"]*)"', fix, body)


def md_to_html(text):
    md = markdown.Markdown(extensions=["tables", "fenced_code", "toc", "sane_lists"],
                           extension_configs={"toc": {"slugify": slug, "permalink": False}})
    body = md.convert(text)
    body = re.sub(r"<table>", '<div class="ptable-scroll"><table class="ptable doc-table">', body)
    body = body.replace("</table>", "</table></div>")
    return body, md.toc_tokens


# ---------- the page frame ----------

TOP_LINKS = [("try.html", "Try it"), ("compare.html", "AR vs diffusion"), ("journey.html", "Journey"), ("usage.html", "Docs")]
ICON_SEARCH = '<svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true"><circle cx="11" cy="11" r="6.5" fill="none" stroke="currentColor" stroke-width="2"/><path d="M16 16l4.5 4.5" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>'
ICON_THEME = ('<svg class="i-sun" viewBox="0 0 24 24" width="18" height="18" aria-hidden="true"><circle cx="12" cy="12" r="4.5" fill="currentColor"/>'
              '<g stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M12 2.5v2M12 19.5v2M2.5 12h2M19.5 12h2M5.3 5.3l1.4 1.4M17.3 17.3l1.4 1.4M5.3 18.7l1.4-1.4M17.3 6.7l1.4-1.4"/></g></svg>'
              '<svg class="i-moon" viewBox="0 0 24 24" width="18" height="18" aria-hidden="true"><path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5z" fill="currentColor"/></svg>')
ICON_MENU = '<svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true"><path d="M4 7h16M4 12h16M4 17h16" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>'
# the mark: a 3 x 3 grid of tokens, the middle one lit (one token, attending to its neighbours)
LOGO = ('<svg class="logo" viewBox="0 0 28 28" width="28" height="28" aria-hidden="true">' +
        "".join(f'<circle cx="{6 + 8 * (i % 3)}" cy="{6 + 8 * (i // 3)}" r="{3.4 if i == 4 else 2.2}" class="{"lit" if i == 4 else "dim"}"/>' for i in range(9)) +
        '</svg>')


def nav(current):
    def link(page, title):
        on = ' class="on"' if page == current else ""
        return f'<a href="{page}"{on}>{html.escape(title)}</a>'
    docs = "".join(link(p, t) for _, p, t, g in DOCS if g == "Docs")
    cards = "".join(link(p, t) for _, p, t, g in DOCS if g == "Models")
    apps = "".join(link(p, t) for p, t in APPS)
    home_on = ' class="on"' if current == "index.html" else ""
    doc_pages = {p for _, p, _, g in DOCS if g == "Docs"}
    top = "".join(f'<a href="{p}"{" class=on" if p == current or (p == "usage.html" and current in doc_pages) else ""}>{t}</a>'
                  for p, t in TOP_LINKS)
    return f'''<header class="site-top">
  <button class="site-menu icon-only" aria-label="Menu" onclick="document.body.classList.toggle('nav-open')">{ICON_MENU}</button>
  <a class="brand" href="index.html">{LOGO}<span>TinyStories ARM</span></a>
  <nav class="top-links">{top}</nav>
  <span class="top-gap"></span>
  <button class="search-pill" id="search-open" aria-label="Search the pages">{ICON_SEARCH}<span>Search</span><kbd>⌘K</kbd></button>
  <button class="icon-only theme-btn" id="theme-btn" aria-label="Switch between light and dark">{ICON_THEME}</button>
  <a class="top-gh" href="{REPO}">GitHub</a>
</header>
<nav class="site-nav">
  <div class="grp"><a href="index.html"{home_on}>Home</a></div>
  <div class="grp"><div class="lbl">Explore</div>{apps}</div>
  <div class="grp"><div class="lbl">Docs</div>{docs}</div>
  <div class="grp"><div class="lbl">Models</div>{cards}</div>
  <div class="grp"><div class="lbl">Elsewhere</div><a href="{REPO}">GitHub</a><a href="{HF}">Hugging Face</a></div>
</nav>
<div class="search-dlg" id="search-dlg" hidden><div class="search-box" role="dialog" aria-label="Search the pages">
  <div class="search-in">{ICON_SEARCH}<input id="search-q" placeholder="Go to a page…" autocomplete="off" spellcheck="false"><kbd>esc</kbd></div>
  <div class="search-list" id="search-list"></div></div></div>'''


# Before the first paint: the theme the reader chose last time (light or dark; none = follow the system).
THEME_HEAD = '''<script>try{var t=localStorage.getItem("site-theme");if(t)document.documentElement.dataset.theme=t}catch(e){}</script>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap">'''

# Every page: the ⌘K page search (over the side menu's links). The theme switch and the dots are app/static/look.js.
SITE_JS = r'''<script>
(() => {
  const dlg = document.getElementById("search-dlg"), q = document.getElementById("search-q"), list = document.getElementById("search-list");
  const seen = new Set(), pages = [];
  document.querySelectorAll(".site-nav a").forEach((a) => {
    const href = a.getAttribute("href");
    if (seen.has(href)) return;
    seen.add(href);
    pages.push({ href, title: a.textContent, group: a.closest(".grp").querySelector(".lbl")?.textContent || "" });
  });
  let sel = 0, shown = [];
  const draw = () => {
    const s = q.value.trim().toLowerCase();
    shown = pages.filter((p) => !s || (p.title + " " + p.group).toLowerCase().includes(s));
    sel = Math.min(sel, Math.max(0, shown.length - 1));
    list.innerHTML = shown.map((p, i) => `<a href="${p.href}" class="${i === sel ? "sel" : ""}"><span>${p.title}</span><small>${p.group}</small></a>`).join("")
      || '<div class="search-none">No page by that name</div>';
  };
  const open = () => { dlg.hidden = false; q.value = ""; sel = 0; draw(); q.focus(); };
  const close = () => { dlg.hidden = true; };
  document.getElementById("search-open").addEventListener("click", open);
  dlg.addEventListener("click", (e) => { if (e.target === dlg) close(); });
  q.addEventListener("input", () => { sel = 0; draw(); });
  document.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") { e.preventDefault(); dlg.hidden ? open() : close(); return; }
    if (dlg.hidden) return;
    if (e.key === "Escape") close();
    else if (e.key === "ArrowDown") { sel = Math.min(sel + 1, shown.length - 1); draw(); e.preventDefault(); }
    else if (e.key === "ArrowUp") { sel = Math.max(sel - 1, 0); draw(); e.preventDefault(); }
    else if (e.key === "Enter" && shown[sel]) location.href = shown[sel].href;
  });
})();
</script>'''


def page(title, current, body, head="", scripts="", wide=False, description=""):
    cls = "site" + (" wide" if wide else "") + (" home" if current == "index.html" else "")
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<meta name="description" content="{html.escape(description or 'A GPT language model and a masked diffusion model, built from scratch and trained on Apple Silicon.')}">
{THEME_HEAD}
<link rel="icon" href="assets/favicon.svg" type="image/svg+xml"><link rel="apple-touch-icon" href="assets/apple-touch-icon.png">
<meta name="theme-color" content="#0f7a6c">
<link rel="stylesheet" href="assets/styles.css"><link rel="stylesheet" href="assets/site.css">{head}</head>
<body class="{cls}">
{nav(current)}
<main class="site-main">{body}
<footer class="site-foot"><span>Built from <a href="{REPO}">the repository</a> on {time.strftime("%d %B %Y")}. MIT licence.</span>
<span><a href="{REPO}">GitHub</a> · <a href="{HF}">Hugging Face</a></span></footer>
</main><script src="assets/look.js"></script>{SITE_JS}{scripts}</body></html>'''


def toc_html(tokens):
    items = []
    for t in tokens:
        if t["level"] > 3:
            continue
        items.append(f'<a href="#{t["id"]}" class="l{t["level"]}">{t["name"]}</a>')
        for c in t.get("children", []):
            if c["level"] <= 3:
                items.append(f'<a href="#{c["id"]}" class="l{c["level"]}">{c["name"]}</a>')
    return f'<aside class="doc-toc"><div class="lbl">On this page</div>{"".join(items)}</aside>' if len(items) > 2 else ""


# ---------- the landing page ----------

def headline_numbers():
    """Benchmark scores of the base model and the 124M diffusion models, from the results store."""
    rows = {}
    path = os.path.join(ROOT, "results", "results.jsonl")
    if not os.path.exists(path):
        return rows
    want = {"mini-fineweb-edu-ful-202611151103-bb160c": "Autoregressive 124M (the base model)",
            "diffusion:f6-124m-stage2-2p5b": "Diffusion 124M, 2.5B tokens",
            "diffusion:f6-124m-stage2": "Diffusion 124M, 1B tokens"}
    for line in open(path):
        r = json.loads(line)
        if r.get("kind") == "benchmark" and r.get("model") in want:
            metric = dict((k, m) for k, _, m in TASKS).get(r.get("key"))
            if metric and metric in r["numbers"]:
                rows.setdefault(want[r["model"]], {})[r["key"]] = r["numbers"][metric]
    return {k: rows[k] for k in want.values() if k in rows}


LEVEL = 1.0      # points: differences this small are within a benchmark's margin of error (±1–1.4 points)


def verdict(d):
    """How a score compares with GPT-2 small's: (css class, label)."""
    if d is None:
        return "", ""
    if abs(d) < LEVEL:
        return "level", "≈ level"
    return ("up", f"+{d:.1f}") if d > 0 else ("down", f"−{-d:.1f}")


def landing(numbers):
    def cell(v):
        return f"{100 * v:.1f}" if v is not None else "–"
    table = ""
    if numbers:
        g = [GPT2[k] for k, _, _ in TASKS]
        g_avg = sum(g) / len(g)
        head = "".join(f'<th class="num">{t}</th>' for _, t, _ in TASKS)
        body = f'<tr class="ref"><td>GPT-2 small (OpenAI, published)<div class="muted small">the yardstick</div></td>' + \
            "".join(f"<td class=num>{cell(v)}</td>" for v in g) + f'<td class="num"><b>{cell(g_avg)}</b></td></tr>'
        notes = []
        for name, sc in numbers.items():
            vals = [sc.get(k) for k, _, _ in TASKS]
            full = all(v is not None for v in vals)
            avg = sum(vals) / len(vals) if full else None
            cells, ahead, level, behind = "", [], [], []      # (task, difference) in each
            for (k, title, _), v, ref in zip(TASKS, vals, g):
                d = None if v is None else 100 * (v - ref)
                cls, lab = verdict(d)
                if d is not None:
                    (ahead if cls == "up" else level if cls == "level" else behind).append((title, lab))
                cells += f'<td class="num">{cell(v)}<div class="vs {cls}">{lab}</div></td>'
            da = None if avg is None else 100 * (avg - g_avg)
            cls, lab = verdict(da)
            if da is None:
                summary = "not all five measured yet"
            elif cls == "level":
                summary = "<b>level with GPT-2 small</b> on average"
            else:
                summary = f"<b class='{cls}'>{abs(da):.1f} points {'ahead of' if da > 0 else 'behind'} GPT-2 small</b> on average"
            def names(xs, signed=True):
                return ", ".join(f"{t} ({lab})" if signed else t for t, lab in xs)
            parts = [f"ahead on {names(ahead)}" if ahead else "", f"level on {names(level, False)}" if level else "",
                     f"behind on {names(behind)}" if behind else ""]
            detail = "; ".join(p for p in parts if p)
            body += f'<tr><td>{html.escape(name)}</td>{cells}<td class="num"><b>{cell(avg)}</b><div class="vs {cls}">{lab}</div></td></tr>'
            notes.append(f"<li><b>{html.escape(name)}</b>: {summary}. {detail[:1].upper() + detail[1:]}.</li>")
        table = f'''<div class="card"><div class="card-head"><h3>Where the models stand</h3><span class="muted small">zero-shot accuracy, %; from the results store</span></div>
<div class="ptable-scroll"><table class="ptable home-bench"><tr><th></th>{head}<th class="num">Average</th></tr>{body}</table></div>
<div class="vs-notes"><div class="card-head"><h4>Against GPT-2 small</h4></div><ul>{"".join(notes)}</ul></div>
<p class="hint">Under each score, the difference from GPT-2 small in points: <span class="vs up">green</span> ahead, <span class="vs down">red</span> behind,
<span class="vs level">≈ level</span> within {LEVEL:.0f} point, about each benchmark's margin of error. GPT-2 small is OpenAI's 124M model from 2019, trained on 40 GB of web text;
our models have the same size. Scored the way EleutherAI's lm-evaluation-harness scores them; the diffusion models score each answer by its likelihood bound.
All of it: <a href="results.html">Results</a>.</p></div>'''
    tiles = [("try.html", "Try it", "Chat with the 124M model, or let the base model continue your text, running in your browser: nothing is sent anywhere."),
             ("journey.html", "Journey", "Every decision from the first commit to today: what changed, why, what it measured and what it cost."),
             ("explorer.html", "Explorer", "One prompt through the real 124M model: every block's attention, the feed-forward layer and the logit lens."),
             ("sidebyside.html", "Side by side", "The same prompt written left to right by the GPT and all at once by the diffusion model, replayed in your browser."),
             ("compare.html", "AR vs diffusion", "The two 124M models compared on held-out text, benchmarks, text quality and speed: what next-token prediction buys."),
             ("story.html", "The story", "From a 356-line training script to a 124M model that beats GPT-2 small on held-out web text."),
             ("usage.html", "How to use it", "Install, run the local web app, train on your Mac's GPU or on Colab, evaluate and compare."),
             ("components.html", "Understanding the components", "Data, tokenization, batching, the transformer, attention, training and inference, one at a time."),
             ("plan.html", "The plan", "The final goals and the roadmap: what's done, what's next and who does it.")]
    tiles_html = "".join(f'<a class="tile" href="{p}"><b>{t}</b><span>{d}</span><i class="arr">→</i></a>' for p, t, d in tiles[:5])
    reads_html = "".join(f'<a class="tile" href="{p}"><b>{t}</b><span>{d}</span><i class="arr">→</i></a>' for p, t, d in tiles[5:])
    questions = ["What is the water cycle?", "Why is the sky blue?", "How do plants make food?", "What causes the seasons?",
                 "Explain gravity to a ten-year-old.", "What is a prime number?", "Give me three tips for studying."]
    body = f'''<section class="home-hero">
<div class="hero-in">
<span class="hero-chip">124M parameters · 2.5B tokens · trained from scratch</span>
<h1>TinyStories ARM</h1>
<p class="tagline">A GPT and a diffusion language model, built from scratch on a Mac and compared like for like.</p>
<form class="ask" id="ask" action="try.html">
  <input id="ask-q" name="q" autocomplete="off" spellcheck="false" aria-label="Ask the chat model" placeholder="{questions[0]}">
  <button class="ask-go" type="submit">Ask <span aria-hidden="true">↵</span></button>
</form>
<p class="ask-note">Runs in your browser: the 174 MB chat model loads once, and nothing you type leaves the page.</p>
<div class="actions"><a class="btn primary" href="usage.html">Get started <span aria-hidden="true">→</span></a><a class="btn" href="{REPO}">Code on GitHub</a><a class="btn" href="{HF}">Models on Hugging Face</a></div>
</div>
</section>
<div class="home-facts">
  <div><b>3.296</b><span>loss on held-out web text<br>GPT-2 small: 3.401</span></div>
  <div><b>42.3</b><span>benchmark average, zero-shot<br>GPT-2 small: 44.4</span></div>
  <div><b>0.85</b><span>nats between the GPT and<br>the diffusion model</span></div>
  <div><b>2.3%</b><span>repetition in chat answers,<br>after DPO with a reward model</span></div>
</div>
<section class="home-sec"><div class="sec-head"><h2>Explore</h2><p>Interactive pages that run in your browser.</p></div>
<div class="tiles explore">{tiles_html}</div></section>
<section class="home-sec"><div class="sec-head"><h2>Read</h2><p>How it was built, how to run it, and what's next.</p></div>
<div class="tiles">{reads_html}</div></section>
<section class="home-sec">{table}</section>
<script>
(() => {{
  const qs = {json.dumps(questions)};
  const box = document.getElementById("ask-q");
  let i = 0;
  setInterval(() => {{
    if (box.value || document.activeElement === box) return;
    box.classList.add("swap");
    setTimeout(() => {{ i = (i + 1) % qs.length; box.placeholder = qs[i]; box.classList.remove("swap"); }}, 250);
  }}, 3200);
  document.getElementById("ask").addEventListener("submit", (e) => {{
    e.preventDefault();
    location.href = "try.html?q=" + encodeURIComponent(box.value.trim() || box.placeholder);
  }});
}})();
</script>'''
    return body



# ---------- the browser-only app pages ----------

SHIM = """<script>
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const tgShade = (p) => `color-mix(in srgb, var(--accent) ${Math.round(12 + 70 * p)}%, transparent)`;
const tgTok = (t) => esc(t).replace(/\\n/g, "↵\\n");
const tgPct = (v) => (v == null ? "–" : `${(v * 100).toFixed(1)}%`);
</script>"""


def journey_page():
    body = '''<header class="panel-head"><span class="eyebrow">Journey</span><h1>How the model got better</h1>
<p>Every decision from the first commit to today: what we changed, why, what it measured, the arithmetic behind it, and
what it cost. The best result is at the top; scroll down to go back in time. The full story is in <a href="story.html">The story</a>.</p></header>
<div id="journey"></div>'''
    scripts = SHIM + '<script src="assets/journey.js"></script><script>journeyOpen();</script>'
    return page("Journey · TinyStories ARM", "journey.html", body, scripts=scripts, wide=True,
                description="Every decision in building a GPT from scratch, with what it measured and what it cost.")


def sidebyside_page(data):
    runs = data["runs"]
    options = "".join(f'<option value="{i}">{html.escape(r["source"])}: {html.escape(r["prompt"][:56].strip())}…</option>' for i, r in enumerate(runs))
    body = f'''<header class="panel-head"><span class="eyebrow">Explore</span><h1>Side by side: left to right, or all at once</h1>
<p>The same prompt written by the two kinds of model. The <b>autoregressive</b> GPT writes one token per pass, left to right.
The <b>masked diffusion</b> model starts from a continuation that is all [MASK] (█) and, at every pass, predicts every masked
position at once and reveals a few. Both have 124M parameters and were trained on the same 2.5B tokens with the same
tokenizer: only the training objective differs. These runs were recorded on an Apple M5 ({html.escape(data["recorded_at"])}); the app
writes new ones live.</p></header>
<div class="card">
  <div class="card-head"><h3>One prompt, two models</h3><span id="sbs-note" class="muted small"></span></div>
  <label>Recorded run <select id="sbs-saved" class="select">{options}</select></label>
  <div id="sbs-out">
    <div class="tg-player">
      <button id="sbs-first" class="icon-btn" title="Back to the start">⏮</button>
      <button id="sbs-play" class="icon-btn" title="Play">▶</button>
      <button id="sbs-next" class="icon-btn" title="One step">⏭</button>
      <input id="sbs-step" type="range" min="0" max="1" value="0">
      <span id="sbs-step-label" class="muted small mono"></span>
      <select id="sbs-align" class="select"><option value="pass">line up by model pass</option><option value="time">line up by clock time</option></select>
      <select id="sbs-speed" class="select"><option value="250">slow</option><option value="90" selected>medium</option><option value="30">fast</option></select>
    </div>
    <div class="sbs-lanes">
      <div class="sbs-lane"><div class="card-head"><h4>Autoregressive: one token per pass</h4><span id="sbs-ar-state" class="muted small"></span></div>
        <div id="sbs-ar-text" class="tg-text"></div>
        <div class="card-head"><h4>Which position each pass writes</h4><span class="muted small">rows: passes · columns: positions</span></div>
        <div id="sbs-ar-grid" class="chart"></div></div>
      <div class="sbs-lane"><div class="card-head"><h4>Diffusion: every position, every pass</h4><span id="sbs-df-state" class="muted small"></span></div>
        <div id="sbs-df-text" class="tg-text"></div>
        <div class="card-head"><h4>When each position is revealed</h4><span class="muted small">rows: passes · columns: positions</span></div>
        <div id="sbs-df-grid" class="chart"></div></div>
    </div>
    <p class="hint">Both grids use the same height per pass, so the taller one took more passes. Left to right draws a staircase; the
    diffusion model fills its positions block by block, the most confident first. The colour is the model's probability for the token it
    chose. Lined up by clock time, each lane moves at its recorded speed.</p>
    <div class="card-head"><h4>The numbers</h4><span class="muted small">the better of the two in bold where lower or higher is better</span></div>
    <div id="sbs-numbers"></div>
    <div id="sbs-ref"></div>
  </div>
</div>
<div class="card"><div class="card-head"><h3>What differs, and what doesn't</h3><span class="muted small">same tokenizer, same kind of transformer blocks</span></div><div id="sbs-how"></div></div>'''
    scripts = SHIM + '''<script src="assets/sidebyside.js"></script><script>
fetch("assets/sidebyside.json").then((r) => r.json()).then((data) => {
  const show = (i) => {
    sbsStop();
    sbs.r = JSON.parse(JSON.stringify(data.runs[i]));
    sbsShowRun();
  };
  $("#sbs-saved").addEventListener("change", (e) => show(+e.target.value));
  show(0);
});
</script>'''
    return page("Side by side · TinyStories ARM", "sidebyside.html", body, scripts=scripts, wide=True,
                description="A GPT and a masked diffusion model of the same size writing the same prompt, replayed side by side.")


def explorer_markup():
    """The app's Explorer (Inside the transformer → Explorer), taken from app/static/index.html so the two stay the same."""
    text = open(os.path.join(ROOT, "app", "static", "index.html"), encoding="utf-8").read()
    start = text.index('<div id="a-explorer-view"')
    end = text.index('<footer class="panel-foot">', start)
    return text[start:end].replace('<div id="a-explorer-view" class="hidden">', '<div id="a-explorer-view">', 1)


def explorer_page(index):
    options = "".join(f'<option value="{i}">{html.escape(p["prompt"])}</option>' for i, p in enumerate(index["prompts"]))
    body = f'''<header class="panel-head"><span class="eyebrow">Explore</span><h1>Explorer: one prompt through the real model</h1>
<p>Every number below is the real model's: <b>{html.escape(index["model"])}</b>, the project's base model. A prompt's tokens become
vectors, go through every transformer block (attention mixes the tokens, the feed-forward layer refines each one) and come out as a
probability for every token in the vocabulary. Hover a row to follow a token, click a part for its explanation, press
<b>▶ Play the data flow</b> to watch one token go through a block stage by stage, or switch to <b>Diagram</b> for the queries, keys
and values. Recorded on an Apple M5 ({html.escape(index["recorded_at"])}); the app explores any prompt live.</p></header>
<div class="card"><label>Recorded prompt <select id="ex-saved" class="select">{options}</select></label>
<input id="a-prompt" type="hidden"></div>
{explorer_markup()}'''
    scripts = SHIM + '''<script>
// what explorer.js needs from the app (app.js), and /api/explain answered from the recorded files
const fmt = (n) => Number(n).toLocaleString();
const toast = (m) => console.warn(m);
const action = async (el, fn) => { try { await fn(); } catch (e) { console.error(e); } };
const ui = { status: { active: { name: EX_MODEL } } };
const exCache = {};
let exPick = 0;
async function api(path, body) {
  const get = async (k, f) => (exCache[k] ??= await (await fetch(`assets/explorer/${f}`)).json());
  const base = await get(exPick, `p${exPick}.json`);
  const L = Math.max(0, Math.min(body.layer || 0, base.config.passes - 1));
  const d = structuredClone(base);
  d.layers[L].detail = await get(`${exPick}-${L}`, `p${exPick}-block${L}.json`);
  d.layer = L;
  return d;
}
</script><script src="assets/explorer.js"></script><script src="assets/flow.js"></script><script src="assets/flowplay.js"></script><script>
document.body.classList.add("ex-wide");
const exShow = (i) => { exPick = i; $("#a-prompt").value = EX_PROMPTS[i]; ex.tok = null; exRun(); };
$("#ex-saved").addEventListener("change", (e) => exShow(+e.target.value));
exShow(0);
</script>'''
    scripts = scripts.replace("EX_MODEL", json.dumps(index["model"])).replace("EX_PROMPTS", json.dumps([p["prompt"] for p in index["prompts"]]))
    return page("Explorer · TinyStories ARM", "explorer.html", body, scripts=scripts, wide=True,
                description="One prompt through the real 124M model: embeddings, attention, feed-forward and the logit lens, in the browser.")


WLLAMA = "https://cdn.jsdelivr.net/npm/@wllama/wllama@3.8.1/esm"    # llama.cpp compiled to WebAssembly (G3)
TRY_MODELS = {   # the published GGUF copies (serving/publish.py): 8-bit, 174 MB each
    "chat": ("aramkris/fineweb-edu-124m-chat-dpo-qwen3", "fineweb-edu-124m-chat-dpo-Q8_0.gguf"),
    "base": ("aramkris/fineweb-edu-124m-base-qwen3", "fineweb-edu-124m-base-Q8_0.gguf"),
}


def try_page():
    body = '''<header class="panel-head"><span class="eyebrow">Try it</span><h1>Talk to the model, in your browser</h1>
<p>The project's 124M model runs right here: <a href="https://github.com/ngxson/wllama">wllama</a> (llama.cpp compiled to
WebAssembly, on the GPU through WebGPU when your browser has it) downloads the 8-bit copy from Hugging Face once
(174 MB, then cached) and generates on your own machine. Nothing you type leaves the page. It's a 124M model trained on
2.5B tokens: it writes fluently and is often wrong on facts, and it repeats itself on long answers.</p></header>
<div class="card try-card">
  <div class="try-row">
    <label>Model <select id="try-model" class="select">
      <option value="chat">Chat (instruction-tuned + DPO): answers questions</option>
      <option value="base">Base: continues your text</option></select></label>
    <label>Run on <select id="try-device" class="select">
      <option value="gpu">GPU (WebGPU)</option>
      <option value="cpu">CPU (WebAssembly)</option></select></label>
    <button id="try-load" class="btn primary">Load the model (174 MB)</button>
    <span id="try-status" class="muted small"></span>
  </div>
  <div id="try-bar" class="try-bar hidden"><i></i></div>
  <label class="block">Your text <textarea id="try-prompt" rows="3" spellcheck="false">What is the water cycle?</textarea></label>
  <div class="try-row">
    <label>Temperature <output id="try-temp-v">0.7</output><input id="try-temp" type="range" min="0" max="1.5" step="0.1" value="0.7"></label>
    <label>Length <output id="try-len-v">160</output><input id="try-len" type="range" min="20" max="400" step="20" value="160"></label>
    <button id="try-run" class="btn primary" disabled>Generate</button>
    <button id="try-stop" class="btn hidden">Stop</button>
  </div>
  <div id="try-out" class="try-out hidden"></div>
</div>
<div id="try-metrics" class="card hidden"><div class="card-head"><h3>Inside this run</h3>
  <span class="muted small">llama.cpp's own timings, the same numbers llama-server reports</span></div>
  <div id="try-tiles" class="stats try-stats"></div>
  <div class="try-chart-head"><b>Time for each token</b>
    <span class="try-legend"><i class="sw pre"></i>first token (reads the whole prompt: prefill) <i class="sw dec"></i>each next token (decode)</span></div>
  <div id="try-chart" class="try-chart"></div>
  <p class="muted small">The first bar is tall because the model reads every prompt token before it can write: that is
  the time to first token. After that each token costs about the same, because each one is a full pass that reads all
  the weights from memory for very little arithmetic. That is why decoding is limited by memory bandwidth, and prefill
  (many tokens per pass) by compute. The bars creep up as the answer grows, because each new token attends to every
  token before it in the KV cache.</p>
</div>
<div id="try-env" class="card hidden"><div class="card-head"><h3>Where the model is running</h3>
  <span class="muted small">from llama.cpp's load log</span></div>
  <div id="try-env-tiles" class="stats try-stats"></div>
</div>
<div id="try-history" class="card hidden"><div class="card-head"><h3>Your runs</h3>
  <span class="muted small">switch between GPU and CPU, or change the prompt length, and compare</span></div>
  <div class="try-table"><table><thead><tr><th>Model</th><th>On</th><th>Prompt tokens</th><th>Reused</th><th>First token</th>
  <th>Prefill</th><th>Decode</th><th>Tokens</th><th>Ended</th></tr></thead><tbody id="try-runs"></tbody></table></div>
</div>
<div class="card"><div class="card-head"><h3>How this works</h3></div><ul class="small-flow">
<li>The model was exported exactly to the standard Qwen3 format and converted to GGUF (plan steps S1 and S7: the
same scores as the original on every benchmark). The files are public:
<a href="https://huggingface.co/aramkris/fineweb-edu-124m-chat-dpo-qwen3">chat</a>,
<a href="https://huggingface.co/aramkris/fineweb-edu-124m-base-qwen3">base</a>.</li>
<li>The chat model writes in a plain <code>User: …</code> / <code>Assistant: …</code> format; the GGUF carries that chat template, so
this page sends messages the same way an OpenAI-style client would.</li>
<li>On the GPU, all 12 blocks and the output layer run as WebGPU shaders (Metal on a Mac, Vulkan or Direct3D elsewhere);
only the token embedding table (50,257 × 768, 39 MB) stays on the CPU, since each step reads just one row of it. On the
CPU the WebAssembly build runs on one thread: several threads need the page to be cross-origin isolated, which GitHub
Pages can't switch on.</li>
<li>To run it on your own computer instead: <code>ollama run hf.co/aramkris/fineweb-edu-124m-chat-dpo-qwen3:Q8_0</code>, or
<code>llama-server -hf aramkris/fineweb-edu-124m-chat-dpo-qwen3:Q8_0 --jinja -c 512</code>.</li></ul></div>'''
    script = TRY_JS.replace("__WLLAMA__", WLLAMA).replace("__MODELS__", json.dumps(TRY_MODELS))
    return page("Try it · TinyStories ARM", "try.html", body, scripts=SHIM + script,
                description="Chat with a 124M GPT trained from scratch, running in your browser with llama.cpp (WebAssembly).")


TRY_JS = r'''<script type="module">
import { Wllama } from "__WLLAMA__/index.js";
const MODELS = __MODELS__;
const st = { w: null, loaded: null, abort: null, env: null, runs: [] };
const $v = (id) => document.getElementById(id);
const status = (t) => { $v("try-status").textContent = t; };
const key = () => `${$v("try-model").value}|${$v("try-device").value}`;
const f1 = (x) => (x == null || !isFinite(x) ? "–" : x >= 100 ? x.toFixed(0) : x.toFixed(1));
const ms = (x) => (x == null ? "–" : x >= 1000 ? `${(x / 1000).toFixed(2)} s` : `${x.toFixed(0)} ms`);
const stat = (k, v, d = "") => `<div class="stat"><div class="k">${k}</div><div class="v">${v}</div><div class="d">${d}</div></div>`;

if (!navigator.gpu) {
  $v("try-device").value = "cpu";
  $v("try-device").querySelector('[value="gpu"]').disabled = true;
  $v("try-device").querySelector('[value="gpu"]').textContent = "GPU (no WebGPU in this browser)";
}
for (const [id, fmt] of [["try-temp", (v) => (+v).toFixed(1)], ["try-len", (v) => v]])
  $v(id).addEventListener("input", (e) => { $v(id + "-v").textContent = fmt(e.target.value); });
const syncButtons = () => {
  $v("try-run").disabled = st.loaded !== key();
  $v("try-load").disabled = st.loaded === key();
  $v("try-load").textContent = st.loaded && st.loaded.split("|")[0] === $v("try-model").value
    ? "Reload on this device" : "Load the model (174 MB)";
};
{ const q = new URLSearchParams(location.search).get("q");
  if (q) { $v("try-model").value = "chat"; $v("try-prompt").value = q.slice(0, 500); } }
$v("try-model").addEventListener("change", () => {
  const chat = $v("try-model").value === "chat";
  $v("try-prompt").value = chat ? "What is the water cycle?" : "The water cycle is the process by which";
  syncButtons();
});
$v("try-device").addEventListener("change", syncButtons);

// llama.cpp's load log says where every buffer went; keep it to show the set-up
function parseEnv(lines, info, device, loadMs) {
  const e = { device, loadMs, buffers: {}, kv: 0, compute: 0, layers: null, adapter: null };
  for (const l of lines) {
    let m;
    if ((m = l.match(/adapter_info:.*vendor: (\S+) \| architecture: (\S+)/))) e.adapter = `${m[1]} ${m[2]}`;
    if ((m = l.match(/offloaded (\d+)\/(\d+) layers/))) e.layers = `${m[1]} of ${m[2]}`;
    if ((m = l.match(/(\S+) model buffer size =\s*([\d.]+) MiB/))) e.buffers[m[1]] = +m[2];
    if ((m = l.match(/KV buffer size =\s*([\d.]+) MiB/))) e.kv += +m[1];
    if ((m = l.match(/compute buffer size =\s*([\d.]+) MiB/))) e.compute += +m[1];
  }
  const md = info.metadata || {};
  e.n_layer = info.n_layer; e.n_embd = info.n_embd; e.n_vocab = info.n_vocab; e.n_ctx = info.n_ctx;
  e.ff = +md["qwen3.feed_forward_length"] || 4 * info.n_embd;
  // parameters: per block 4·d² (attention) + 3·d·ff (SwiGLU); plus the output layer (vocab × d)
  e.blockParams = info.n_layer * (4 * e.n_embd ** 2 + 3 * e.n_embd * e.ff);
  e.flopsPerToken = 2 * (e.blockParams + info.n_vocab * e.n_embd);
  const total = Object.values(e.buffers).reduce((a, b) => a + b, 0);
  const embed = info.n_vocab * info.n_embd * 34 / 32 / 2 ** 20;   // Q8_0: 34 bytes per 32 weights
  e.totalMiB = total; e.readMiB = total - embed;                    // a step reads one row of the embedding table
  return e;
}

function showEnv(e) {
  const gpuMiB = Object.entries(e.buffers).filter(([k]) => k !== "CPU").reduce((a, [, v]) => a + v, 0);
  const kvFormula = `2 (K and V) × ${e.n_layer} layers × ${e.n_ctx} positions × ${e.n_embd} × 2 bytes`;
  $v("try-env-tiles").innerHTML = [
    stat("Runs on", e.device === "gpu" ? "WebGPU" : "CPU", e.device === "gpu" ? (e.adapter || "the browser's GPU") : "WebAssembly, 1 thread"),
    stat("Layers on the GPU", e.layers || "none", "12 blocks + the output layer"),
    e.device === "gpu" ? stat("Weights in GPU memory", `${f1(gpuMiB)} MB`, `${f1(e.buffers.CPU || 0)} MB on the CPU: the embedding table`)
      : stat("Weights in memory", `${f1(e.totalMiB)} MB`, "all on the CPU, in the page's WebAssembly memory"),
    stat("KV cache", `${f1(e.kv)} MB`, kvFormula),
    stat("Scratch for one pass", `${f1(e.compute)} MB`, "intermediate activations"),
    stat("Load time", ms(e.loadMs), "download (or cache) + set-up"),
    stat("Model", `${e.n_layer} × ${e.n_embd}`, `${e.n_layer} blocks, width ${e.n_embd}, ${e.ff} SwiGLU, ${e.n_ctx}-token context, 8-bit (Q8_0)`),
    stat("Work per token", `${(e.flopsPerToken / 1e9).toFixed(2)} GFLOP`, `≈ 2 × ${(e.flopsPerToken / 2 / 1e6).toFixed(0)}M parameters; ${f1(e.readMiB)} MB of weights read`),
  ].join("");
  $v("try-env").classList.remove("hidden");
}

$v("try-load").addEventListener("click", async () => {
  const [which, device] = key().split("|"), [repo, file] = MODELS[which];
  $v("try-load").disabled = true; $v("try-run").disabled = true;
  $v("try-bar").classList.remove("hidden");
  const lines = [];
  const keep = (...a) => lines.push(a.join(" "));
  const logger = { debug: keep, log: keep, warn: (...a) => { keep(...a); console.warn(...a); }, error: (...a) => { keep(...a); console.error(...a); } };
  try {
    if (st.w) { await st.w.exit(); st.w = null; st.loaded = null; }
    st.w = new Wllama({ default: "__WLLAMA__/wasm/wllama.wasm" }, { logger });
    status("Downloading…");
    const t0 = performance.now();
    await st.w.loadModelFromHF({ repo, file }, { n_ctx: 512, n_gpu_layers: device === "gpu" ? 99999 : 0,
      progressCallback: ({ loaded, total }) => {
        const p = total ? loaded / total : 0;
        $v("try-bar").firstElementChild.style.width = `${Math.round(p * 100)}%`;
        status(`Downloading… ${(loaded / 1e6).toFixed(0)} of ${(total / 1e6).toFixed(0)} MB`);
      } });
    st.env = parseEnv(lines, await st.w.getLoadedContextInfo(), device, performance.now() - t0);
    st.loaded = key();
    status(`Loaded: the ${which} model, on the ${device === "gpu" ? "GPU" : "CPU"}.`);
    showEnv(st.env);
  } catch (e) {
    status(`Couldn't load the model: ${e.message || e}`); console.error(e);
  } finally { $v("try-bar").classList.add("hidden"); syncButtons(); }
});

function showRun(r, live) {
  const t = r.timings, e = st.env;
  const tiles = [stat("Time to first token", ms(r.ttft), live ? "waiting for prefill…" : `${t ? t.prompt_n : "?"} prompt tokens read first`)];
  if (t) {
    const bw = e.readMiB * 1.048576 * t.predicted_per_second / 1000;   // GB/s
    const tok = (k) => `${k} token${k === 1 ? "" : "s"}`, few = t.prompt_n < 8;   // too few new tokens to time prefill
    const fewNote = `only ${tok(t.prompt_n)} new${t.cache_n ? `, ${t.cache_n} reused from the KV cache` : ""}: too few to measure; try a longer or a new prompt`;
    tiles[0] = stat("Time to first token", ms(r.ttft), `${tok(t.prompt_n)} of prompt read first`);
    tiles.push(
      stat("Prefill", few ? "–" : `${f1(t.prompt_per_second)} tok/s`, few ? fewNote : `${tok(t.prompt_n)} in ${ms(t.prompt_ms)}, ${t.prompt_per_token_ms.toFixed(1)} ms each`),
      stat("Decode", `${f1(t.predicted_per_second)} tok/s`, `${tok(t.predicted_n)} in ${ms(t.predicted_ms)}, ${t.predicted_per_token_ms.toFixed(1)} ms each`),
      stat("Prefill vs decode", few ? "–" : `${f1(t.prompt_per_second / t.predicted_per_second)}×`, "faster per token: many tokens share one read of the weights"),
      stat("Compute rate", few ? `${f1(e.flopsPerToken * t.predicted_per_second / 1e9)} GFLOP/s` : `${f1(e.flopsPerToken * t.prompt_per_second / 1e9)} GFLOP/s`,
        few ? "while decoding" : `prefill; decode only ${f1(e.flopsPerToken * t.predicted_per_second / 1e9)}`),
      stat("Memory read while decoding", `${f1(bw)} GB/s`, `${f1(e.readMiB)} MB of weights × ${f1(t.predicted_per_second)} tokens/s`),
      stat("Context used", `${t.prompt_n + t.predicted_n} / ${e.n_ctx}`, `${Math.round(100 * (t.prompt_n + t.predicted_n) / e.n_ctx)}% of the KV cache`),
      stat("Reused from the KV cache", `${t.cache_n}`, t.cache_n ? "prompt tokens kept from the last run: not read again" : "nothing in common with the last prompt"),
      stat("Ended by", r.finish === "stop" ? "end of text" : r.finish === "length" ? "length limit" : (r.finish || "stop button"), `${r.tokens} tokens arrived`));
  } else {
    tiles.push(stat("Tokens so far", `${r.tokens}`, live ? "streaming" : "stopped: llama.cpp sends its timings only at the end"));
  }
  $v("try-tiles").innerHTML = tiles.join("");
  drawChart(r.gaps);
  $v("try-metrics").classList.remove("hidden");
}

function drawChart(gaps) {
  const box = $v("try-chart");
  if (!gaps.length) { box.innerHTML = ""; return; }
  const W = Math.max(box.clientWidth, 280), H = 150, padL = 44, padB = 18, padT = 8;
  // the first bar (prefill) can dwarf the rest: cap the scale at 4× the typical decode time and label the cut bar
  const dec = gaps.slice(1).sort((a, b) => a - b), med = dec.length ? dec[Math.floor(dec.length / 2)] : 0;
  const n = gaps.length, max = med && Math.max(...gaps) > 5 * med ? 4 * med : Math.max(...gaps), step = (W - padL) / n, bw = Math.max(1, step - (step > 4 ? 2 : 0));
  const y = (v) => padT + (H - padT - padB) * (1 - v / max);
  const ticks = [0, max / 2, max];
  let s = `<svg viewBox="0 0 ${W} ${H}" width="100%" height="${H}" role="img" aria-label="Milliseconds for each generated token">`;
  for (const t of ticks) s += `<line x1="${padL}" x2="${W}" y1="${y(t)}" y2="${y(t)}" class="grid"/><text x="${padL - 6}" y="${y(t) + 4}" text-anchor="end">${t.toFixed(0)} ms</text>`;
  gaps.forEach((g, i) => {
    const h = Math.max(1, H - padB - y(Math.min(g, max)));
    if (g > max) s += `<text x="${padL + i * step + 3}" y="${padT + 10}" class="cut">↑ ${g.toFixed(0)} ms</text>`;
    s += `<rect x="${padL + i * step}" y="${H - padB - h}" width="${bw}" height="${h}" rx="${bw > 6 ? 2 : 0}" class="${i ? "dec" : "pre"}" data-i="${i}"/>`;
  });
  s += `<text x="${padL}" y="${H - 3}">token 1</text><text x="${W}" y="${H - 3}" text-anchor="end">token ${n}</text></svg><div class="try-tip hidden"></div>`;
  box.innerHTML = s;
  const tip = box.querySelector(".try-tip"), svg = box.querySelector("svg");
  svg.addEventListener("mousemove", (ev) => {
    const r = svg.getBoundingClientRect(), x = (ev.clientX - r.left) * W / r.width;
    const i = Math.min(n - 1, Math.max(0, Math.floor((x - padL) / step)));
    if (x < padL) { tip.classList.add("hidden"); return; }
    tip.innerHTML = `<b>token ${i + 1}</b>: ${gaps[i].toFixed(1)} ms${i ? "" : " (prefill: reading the prompt)"}`;
    tip.style.left = `${Math.min(ev.clientX - r.left + 10, r.width - 190)}px`; tip.classList.remove("hidden");
  });
  svg.addEventListener("mouseleave", () => tip.classList.add("hidden"));
}

function addHistory(r) {
  const t = r.timings, short = (x) => (x == null ? "–" : `${f1(x)} tok/s`);
  st.runs.unshift(`<tr><td>${r.model}</td><td>${r.device === "gpu" ? "GPU" : "CPU"}</td><td>${t ? t.prompt_n : "–"}</td><td>${t ? t.cache_n : "–"}</td>
    <td>${ms(r.ttft)}</td><td>${t && t.prompt_n < 8 ? "–" : short(t?.prompt_per_second)}</td><td>${short(t?.predicted_per_second)}</td><td>${r.tokens}</td>
    <td>${r.finish === "stop" ? "end of text" : r.finish === "length" ? "length" : "stopped"}</td></tr>`);
  st.runs = st.runs.slice(0, 10);
  $v("try-runs").innerHTML = st.runs.join("");
  $v("try-history").classList.remove("hidden");
}

$v("try-run").addEventListener("click", async () => {
  const [model, device] = st.loaded.split("|"), chat = model === "chat", text = $v("try-prompt").value;
  const opts = { max_tokens: +$v("try-len").value, temperature: +$v("try-temp").value, top_p: 0.9,
    repeat_penalty: 1.1, stream: true };
  const out = $v("try-out");
  out.classList.remove("hidden");
  out.innerHTML = chat ? `<div class="try-q">${esc(text)}</div><div class="try-a"></div>` : `<span class="try-p">${esc(text)}</span><span class="try-a"></span>`;
  const ans = out.querySelector(".try-a");
  $v("try-run").disabled = true; $v("try-stop").classList.remove("hidden");
  st.abort = new AbortController();
  const r = { model, device, ttft: null, tokens: 0, gaps: [], timings: null, finish: null };
  const t0 = performance.now(); let prev = t0, lastDraw = 0;
  showRun(r, true);
  try {
    const stream = chat
      ? await st.w.createChatCompletion({ messages: [{ role: "user", content: text }], ...opts, abortSignal: st.abort.signal })
      : await st.w.createCompletion({ prompt: text, ...opts, abortSignal: st.abort.signal });
    for await (const ch of stream) {
      if (st.abort.signal.aborted) break;
      if (ch.timings) r.timings = ch.timings;
      const c = ch.choices?.[0];
      if (c?.finish_reason) r.finish = c.finish_reason;
      const piece = chat ? c?.delta?.content : c?.text;
      if (!piece) continue;
      const now = performance.now();
      r.ttft ??= now - t0;
      r.gaps.push(now - prev); prev = now; r.tokens++;
      ans.textContent = chat && !ans.textContent ? piece.trimStart() : ans.textContent + piece;
      if (now - lastDraw > 150) { showRun(r, true); lastDraw = now; }
    }
  } catch (e) {
    if (!st.abort.signal.aborted) { ans.textContent += ` [error: ${e.message || e}]`; console.error(e); }
  } finally {
    showRun(r, false); addHistory(r);
    $v("try-run").disabled = false; $v("try-stop").classList.add("hidden");
  }
});
$v("try-stop").addEventListener("click", () => st.abort?.abort());
</script>'''


# ---------- build ----------

# ---------- goal 2: the two kinds of model compared (F4) ----------

F4_DIR = os.path.join(ROOT, "outputs", "f4")
HELDOUT_NAMES = {"baominhtran__mini-fineweb-edu-small": "Mini FineWeb-Edu", "baominhtran__mini-fineweb-edu-tiny": None,
                 "huggingfacetb__cosmopedia-khanacademy": "Cosmopedia: Khan Academy",
                 "huggingfacetb__cosmopedia-openstax": "Cosmopedia: OpenStax",
                 "huggingfacetb__cosmopedia-stanford-first100000": "Cosmopedia: Stanford",
                 "huggingfacetb__cosmopedia-stories-first300000": "Cosmopedia: stories",
                 "huggingfacetb__cosmopedia-wikihow": "Cosmopedia: WikiHow", "roneneldan__tinystories": "TinyStories",
                 "salesforce__wikitext-wikitext-2-raw-v1": "WikiText-2",
                 "url-raw.githubusercontent.com-input.txt-4c649141": "Tiny Shakespeare"}


def bar_rows(rows, lo, hi, fmt, legend):
    """Horizontal paired bars in plain HTML (they reflow on a phone): rows = [(label, [(value, css class), ...], note)]."""
    span = hi - lo
    out = []
    for label, vals, note in rows:
        bars = "".join(f'<div class="cmp-bar {c}" style="width:{max(1.5, 100 * (v - lo) / span):.1f}%"><span>{fmt(v)}</span></div>'
                       for v, c in vals)
        out.append(f'<div class="cmp-row"><div class="cmp-lab">{label}<span class="muted small">{note}</span></div>'
                   f'<div class="cmp-bars">{bars}</div></div>')
    return f'<div class="cmp-legend">{legend}</div><div class="cmp-chart">{"".join(out)}</div>'


def speed_quality_svg(settings, real):
    """GPT-2's judge loss (lower = more like real text) against tokens per second, one point per way of writing."""
    pts = []
    for name, v in settings.items():
        kind = "ar" if name.startswith("AR") else "df"
        label = {"AR, sampled (top-k 50)": "AR, sampled", "AR, greedy": "AR, greedy"}.get(name, name)
        pts.append((v["tokens_per_sec"], v["judge_loss"], v["repeated_4grams"], kind, label))
    W, H, L, R, T, B = 520, 330, 52, 18, 16, 46
    x0, x1, y0, y1 = 0, 130, 1.0, 5.0
    X = lambda x: L + (x - x0) / (x1 - x0) * (W - L - R)
    Y = lambda y: T + (y - y0) / (y1 - y0) * (H - T - B)
    g = []
    for yv in (1, 2, 3, 4, 5):
        g.append(f'<line x1="{L}" x2="{W - R}" y1="{Y(yv):.1f}" y2="{Y(yv):.1f}" class="grid"/>'
                 f'<text x="{L - 8}" y="{Y(yv) + 4:.1f}" class="tick" text-anchor="end">{yv}</text>')
    for xv in (0, 25, 50, 75, 100, 125):
        g.append(f'<text x="{X(xv):.1f}" y="{H - B + 18}" class="tick" text-anchor="middle">{xv}</text>')
    g.append(f'<line x1="{L}" x2="{W - R}" y1="{Y(real):.1f}" y2="{Y(real):.1f}" class="ref"/>'
             f'<text x="{W - R}" y="{Y(real) + 17:.1f}" class="tick" text-anchor="end">real text {real:.2f}</text>')
    df = sorted(p for p in pts if p[3] == "df")
    g.append('<polyline class="df-line" points="' + " ".join(f"{X(p[0]):.1f},{Y(p[1]):.1f}" for p in df) + '"/>')
    for x, y, rep, kind, label in pts:
        anchor, dx, dy = ("end", -10, 4) if x > 90 else ("start", 10, 4)
        if label == "AR, greedy":
            dy = 18
        elif label == "AR, sampled":
            dy = -11
        elif label == "diffusion, 64 passes":
            dy = 22
        g.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="6" class="pt {kind}"><title>{label}: {x:.0f} tokens/s, '
                 f'judge loss {y:.2f}, repetition {100 * rep:.1f}%</title></circle>'
                 f'<text x="{X(x) + dx:.1f}" y="{Y(y) + dy:.1f}" class="lbl" text-anchor="{anchor}">{label}</text>')
    mid = (T + H - B) / 2
    g.append(f'<text x="{(L + W - R) / 2:.0f}" y="{H - 8}" class="axis" text-anchor="middle">tokens per second on an Apple M5 (faster →)</text>'
             f'<text x="14" y="{mid:.0f}" class="axis" text-anchor="middle" transform="rotate(-90 14 {mid:.0f})">GPT-2 judge loss (better ↓)</text>')
    return f'<svg class="cmp-svg" viewBox="0 0 {W} {H}" role="img" aria-label="Text quality against writing speed">{"".join(g)}</svg>'


CMP_JS = """<script>
const CMP = __DATA__;
const cmpShow = (i) => {
  document.getElementById("cmp-prompt").textContent = CMP[i].prompt;
  CMP[i].texts.forEach((t, j) => { document.getElementById("cmp-t" + j).textContent = t.trimStart(); });
};
document.getElementById("cmp-ex").addEventListener("change", (e) => cmpShow(+e.target.value));
cmpShow(0);
</script>"""


def compare_page(d, d51):
    bm = d["benchmarks"]
    avg = lambda key: 100 * sum(v[key] for v in bm.values()) / len(bm)
    ar_avg, df_avg, g_avg = avg("ar"), avg("diffusion"), avg("gpt2")
    ho = d["heldout"]
    web = ho["baominhtran__mini-fineweb-edu-small"]
    st = d["writing"]["settings"]
    ar_s, df16, df64 = st["AR, sampled (top-k 50)"], st["diffusion, 16 passes"], st["diffusion, 64 passes"]
    real = d["writing"]["real continuation"]["judge_loss"]
    gap51 = d51["heldout"]["roneneldan__tinystories"]["gap"]
    slower = ar_s["tokens_per_sec"] / df64["tokens_per_sec"]
    stats = [("Held-out web text", f"+{web['gap']:.2f} nats",
              f"per token: {web['ar_loss']:.3f} against a bound of {web['diffusion_bound']:.3f}"),
             ("Benchmark average", f"{ar_avg:.1f} vs {df_avg:.1f}", f"autoregressive vs diffusion (GPT-2 small {g_avg:.1f})"),
             ("At the same speed", f"{ar_s['judge_loss']:.2f} vs {df16['judge_loss']:.2f}",
              f"GPT-2's judge loss at about {ar_s['tokens_per_sec']:.0f} tokens/s each (lower is better)"),
             ("At the same quality", f"{slower:.1f}× slower", "the diffusion model needs one pass per token to catch up")]
    tiles = "".join(f'<div class="stat"><div class="k">{k}</div><div class="v">{v}</div><div class="d">{t}</div></div>'
                    for k, v, t in stats)
    rows = []
    for key, v in sorted(ho.items(), key=lambda kv: -kv[1]["gap"]):
        name = HELDOUT_NAMES.get(key, v["title"])
        if name:
            rows.append((html.escape(name), [(v["ar_loss"], "ar"), (v["diffusion_bound"], "df")], f"gap +{v['gap']:.2f}"))
    held = bar_rows(rows, 0, 6.2, lambda x: f"{x:.2f}",
                    '<span><i class="sw ar"></i>autoregressive: loss</span><span><i class="sw df"></i>diffusion: likelihood bound</span>'
                    '<span class="muted">nats per token, lower is better</span>')
    rows = []
    for key, title, _ in TASKS:
        v = bm[key]
        rows.append((title, [(100 * v["ar"], "ar"), (100 * v["diffusion"], "df"), (100 * v["gpt2"], "g2")],
                     f"chance {100 * v['chance']:.0f}%" if v["chance"] else "chance 0%"))
    rows.append(("<b>Average</b>", [(ar_avg, "ar"), (df_avg, "df"), (g_avg, "g2")], ""))
    bench = bar_rows(rows, 0, 70, lambda x: f"{x:.1f}",
                     '<span><i class="sw ar"></i>autoregressive 124M</span><span><i class="sw df"></i>diffusion 124M</span>'
                     '<span><i class="sw g2"></i>GPT-2 small</span><span class="muted">zero-shot accuracy, %</span>')
    wrows = "".join(f'<tr><td>{html.escape(n)}</td><td class="num">{v["passes"]}</td><td class="num">{v["judge_loss"]:.2f}</td>'
                    f'<td class="num">{100 * v["repeated_4grams"]:.1f}%</td><td class="num">{v["tokens_per_sec"]:.0f}</td></tr>'
                    for n, v in st.items())
    show = [("AR, sampled (top-k 50)", "Autoregressive, sampled"), ("diffusion, 16 passes", "Diffusion, 16 passes: the same speed"),
            ("diffusion, 64 passes", "Diffusion, 64 passes: one per token")]
    exs = st[show[0][0]]["examples"]
    examples = [{"source": exs[i]["source"], "prompt": exs[i]["prompt"],
                 "texts": [st[k]["examples"][i]["text"] for k, _ in show]} for i in range(0, len(exs), 2)]
    opts = "".join(f'<option value="{i}">{html.escape(e["source"])}: {html.escape(" ".join(e["prompt"].split())[:60])}…</option>'
                   for i, e in enumerate(examples))
    lanes = "".join(f'<div class="cmp-lane"><h4>{t}</h4><p class="cmp-text" id="cmp-t{j}"></p></div>' for j, (_, t) in enumerate(show))
    body = (
        '<header class="panel-head"><span class="eyebrow">Goal 2</span><h1>Two ways to write, compared</h1>'
        '<p>An <b>autoregressive</b> GPT writes one token at a time, left to right. A <b>masked diffusion</b> model starts from a '
        'continuation that is all blanks and fills in every position over a few passes, the most confident first. Here both have '
        '124M parameters and were trained on the same 2.5B tokens (FineWeb-Edu and Cosmopedia), with the same tokenizer and the same '
        'transformer blocks: only the training objective differs. Measured on an Apple M5 '
        f'({html.escape(d["measured_at"][:10])}); <a href="story.html#38-two-ways-to-write-compared-f4">the full write-up</a>.</p></header>'
        f'<div class="stats cmp-stats">{tiles}</div>'
        '<div class="card"><div class="card-head"><h3>Held-out text: how surprised each model is</h3>'
        '<span class="muted small">64 windows of 256 tokens per set</span></div>'
        f'{held}'
        '<p class="hint">The autoregressive number is its exact loss; the diffusion number is its likelihood bound (16 masking levels '
        'per window), an upper limit on its real loss, so the true gap may be a little smaller. The gap is about the same at both '
        f'sizes: +{gap51:.2f} on TinyStories for the 51M pair (164M tokens each), +{web["gap"]:.2f} on FineWeb-Edu for this 124M pair. '
        'It is largest on Cosmopedia\'s synthetic textbooks, which are very predictable from the left, and smallest on text far '
        'from the training data (WikiText, Shakespeare), where both models are unsure anyway.</p></div>'
        '<div class="card"><div class="card-head"><h3>Benchmarks</h3>'
        '<span class="muted small">scored like EleutherAI\'s lm-evaluation-harness</span></div>'
        f'{bench}'
        '<p class="hint">The diffusion model scores each answer by its likelihood bound. It is furthest behind on ARC-Easy (science '
        'facts, barely above chance) and LAMBADA (the last word of a passage), and level on Winogrande, where both models are at '
        'chance. The autoregressive model is ahead of GPT-2 small on HellaSwag and ARC-Easy.</p></div>'
        '<div class="card"><div class="card-head"><h3>Writing: quality against speed</h3>'
        '<span class="muted small">8 held-out prompts × 2 seeds, 64 new tokens each</span></div>'
        f'{speed_quality_svg(st, real)}'
        '<p class="hint">The judge is OpenAI\'s GPT-2 small, which neither model was trained on: lower means more like text it would '
        'write. With 16 passes the diffusion model writes as fast as the autoregressive one, but its text is much less likely; it '
        f'catches up only with 64 passes, one per token, and is then {slower:.1f}× slower, because every pass re-reads every position '
        'while the autoregressive model, with its KV cache, reads only the new token. Greedy autoregressive text scores best and reads '
        'worst: a judge loss rewards repetition, so read it next to the repetition column.</p>'
        '<div class="ptable-scroll"><table class="ptable"><tr><th>124M, 64 tokens</th><th class="num">Passes</th>'
        '<th class="num">Judge loss</th><th class="num">Repeated 4-token runs</th><th class="num">Tokens/s</th></tr>'
        f'{wrows}<tr class="ref"><td>The real continuation</td><td class="num">–</td><td class="num">{real:.2f}</td>'
        '<td class="num">0%</td><td class="num">–</td></tr></table></div></div>'
        '<div class="card"><div class="card-head"><h3>Read them yourself</h3><span class="muted small">the same prompt, the first seed</span></div>'
        f'<label>Prompt <select id="cmp-ex" class="select">{opts}</select></label>'
        '<p class="cmp-prompt" id="cmp-prompt"></p>'
        f'<div class="cmp-lanes">{lanes}</div>'
        '<p class="hint">To watch the two models write step by step, see <a href="sidebyside.html">Side by side</a>.</p></div>'
        '<div class="card"><div class="card-head"><h3>What it means</h3></div><div class="prose"><ul>'
        '<li><b>At this size and budget, the masked diffusion objective costs about 0.85 nats per token and 7 benchmark points</b> '
        'against next-token prediction, and the gap didn\'t shrink from 51M to 124M parameters.</li>'
        '<li><b>Its promised speed costs quality.</b> Writing many tokens per pass means fewer, less careful decisions; matching the '
        'autoregressive model\'s text takes as many passes as tokens.</li>'
        '<li><b>Where it could still win</b> is work that needs the right-hand side of the text: filling a gap, planning, meeting '
        'constraints, or revising earlier tokens. The reasoning comparison (plan step I7) will test that.</li></ul></div></div>')
    scripts = CMP_JS.replace("__DATA__", json.dumps(examples).replace("</", "<\\/"))
    return page("Two ways to write, compared · TinyStories ARM", "compare.html", body, scripts=scripts, wide=True,
                description="A 124M GPT and a 124M masked diffusion model trained on the same data, compared on loss, "
                            "benchmarks, text quality and speed.")


def build(out):
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(os.path.join(out, "assets"))
    assets = set()
    for src, dst, title, _ in DOCS:
        path = os.path.join(ROOT, src)
        if not os.path.exists(path):
            continue
        text = open(path, encoding="utf-8").read()
        body, toc = md_to_html(text)
        body = rewrite_links(body, src, assets)
        edit = f'<p class="doc-src">From <a href="{REPO}/blob/main/{src}">{src}</a></p>'
        group = dict((p, g) for _, p, _, g in DOCS).get(dst, "Docs")
        m = re.match(r"\s*(<h1[^>]*>.*?</h1>)\s*(<p>.*?</p>)?", body, re.S)
        if m:          # the title (and the opening paragraph, if short) into the page head
            lead = m.group(2) if m.group(2) and len(re.sub("<[^>]+>", "", m.group(2))) < 420 else ""
            head = f'<header class="panel-head doc-hero"><span class="eyebrow">{group}</span>{m.group(1)}{lead}{edit}</header>'
            body = body[m.end(1) + (m.end(2) - m.end(1) if lead else 0):]
        else:
            head = f'<header class="panel-head doc-hero"><span class="eyebrow">{group}</span><h1>{html.escape(title)}</h1>{edit}</header>'
        content = f'{head}<div class="doc-page">{toc_html(toc)}<article class="prose">{body}</article></div>'
        with open(os.path.join(out, dst), "w", encoding="utf-8") as f:
            f.write(page(f"{title} · TinyStories ARM", dst, content, wide=True))
    for a in sorted(assets):
        if os.path.exists(os.path.join(ROOT, a)):
            shutil.copy(os.path.join(ROOT, a), os.path.join(out, "assets", a.replace("/", "_")))
    for name in ("styles.css", "look.js", "favicon.svg", "apple-touch-icon.png", "journey.js", "sidebyside.js", "explorer.js", "flow.js", "flowplay.js"):
        shutil.copy(os.path.join(ROOT, "app", "static", name), os.path.join(out, "assets", name))
    j = os.path.join(out, "assets", "journey.js")          # its "full story" links: the site's own page
    text = open(j, encoding="utf-8").read()
    text = re.sub(r'const J_DOC = "[^"]*";', 'const J_DOC = "story.html";', text)
    open(j, "w", encoding="utf-8").write(text)
    shutil.copy(os.path.join(HERE, "site.css"), os.path.join(out, "assets", "site.css"))
    with open(os.path.join(out, "index.html"), "w", encoding="utf-8") as f:
        f.write(page("TinyStories ARM: a GPT and a diffusion model from scratch", "index.html", landing(headline_numbers())))
    with open(os.path.join(out, "journey.html"), "w", encoding="utf-8") as f:
        f.write(journey_page())
    with open(os.path.join(out, "try.html"), "w", encoding="utf-8") as f:
        f.write(try_page())
    if os.path.exists(os.path.join(F4_DIR, "124m.json")):
        with open(os.path.join(out, "compare.html"), "w", encoding="utf-8") as f:
            f.write(compare_page(json.load(open(os.path.join(F4_DIR, "124m.json"))),
                                 json.load(open(os.path.join(F4_DIR, "51m-stories.json")))))
    data_path = os.path.join(HERE, "recorded", "sidebyside.json")
    if os.path.exists(data_path):
        shutil.copy(data_path, os.path.join(out, "assets", "sidebyside.json"))
        with open(os.path.join(out, "sidebyside.html"), "w", encoding="utf-8") as f:
            f.write(sidebyside_page(json.load(open(data_path))))
    ex_dir = os.path.join(HERE, "recorded", "explorer")
    if os.path.exists(os.path.join(ex_dir, "index.json")):
        shutil.copytree(ex_dir, os.path.join(out, "assets", "explorer"))
        with open(os.path.join(out, "explorer.html"), "w", encoding="utf-8") as f:
            f.write(explorer_page(json.load(open(os.path.join(ex_dir, "index.json")))))
    open(os.path.join(out, ".nojekyll"), "w").close()
    n = sum(len(fs) for _, _, fs in os.walk(out))
    size = sum(os.path.getsize(os.path.join(d, x)) for d, _, fs in os.walk(out) for x in fs)
    print(f"built {out}: {n} files, {size / 1e6:.1f} MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build the project's website (GitHub Pages).")
    ap.add_argument("--out", default=os.path.join(ROOT, "_site"))
    build(ap.parse_args().out)

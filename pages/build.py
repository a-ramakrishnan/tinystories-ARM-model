# The project's website for GitHub Pages (plan G): the documentation as pages, plus the app's Journey timeline and
# a Side by side replay that run in the browser from saved data (no server, no model).
#
#   python3 pages/build.py [--out _site]        then open _site/index.html, or: python3 -m http.server -d _site
#
# Inputs: the Markdown docs (README, docs/, diffusion_models/README.md, the model cards), the results store
# (results/results.jsonl) for the headline numbers, app/static (styles, journey.js, sidebyside.js) and
# pages/recorded/sidebyside.json (recorded by pages/record.py). Links between docs become links between pages; links
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
APPS = [("journey.html", "Journey"), ("sidebyside.html", "Side by side")]
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

def nav(current):
    def link(page, title):
        on = ' class="on"' if page == current else ""
        return f'<a href="{page}"{on}>{html.escape(title)}</a>'
    docs = "".join(link(p, t) for _, p, t, g in DOCS if g == "Docs")
    cards = "".join(link(p, t) for _, p, t, g in DOCS if g == "Models")
    apps = "".join(link(p, t) for p, t in APPS)
    home_on = ' class="on"' if current == "index.html" else ""
    return f'''<header class="site-top"><a class="brand" href="index.html"><span class="logo">M</span>TinyStories ARM</a>
  <button class="site-menu" aria-label="Menu" onclick="document.body.classList.toggle('nav-open')">☰</button></header>
<nav class="site-nav">
  <div class="grp"><a href="index.html"{home_on}>Home</a></div>
  <div class="grp"><div class="lbl">Explore</div>{apps}</div>
  <div class="grp"><div class="lbl">Docs</div>{docs}</div>
  <div class="grp"><div class="lbl">Models</div>{cards}</div>
  <div class="grp"><div class="lbl">Elsewhere</div><a href="{REPO}">GitHub</a><a href="{HF}">Hugging Face</a></div>
</nav>'''


def page(title, current, body, head="", scripts="", wide=False, description=""):
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<meta name="description" content="{html.escape(description or 'A GPT language model and a masked diffusion model, built from scratch and trained on Apple Silicon.')}">
<link rel="stylesheet" href="assets/styles.css"><link rel="stylesheet" href="assets/site.css">{head}</head>
<body class="site{' wide' if wide else ''}">
{nav(current)}
<main class="site-main">{body}
<footer class="site-foot">Built from <a href="{REPO}">the repository</a> on {time.strftime("%d %B %Y")}. MIT licence.</footer>
</main>{scripts}</body></html>'''


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
    tiles = [("journey.html", "Journey", "Every decision from the first commit to today: what changed, why, what it measured and what it cost."),
             ("sidebyside.html", "Side by side", "The same prompt written left to right by the GPT and all at once by the diffusion model, replayed in your browser."),
             ("story.html", "The story", "From a 356-line training script to a 124M model that beats GPT-2 small on held-out web text."),
             ("usage.html", "How to use it", "Install, run the local web app, train on your Mac's GPU or on Colab, evaluate and compare."),
             ("components.html", "Understanding the components", "Data, tokenization, batching, the transformer, attention, training and inference, one at a time."),
             ("plan.html", "The plan", "The final goals and the roadmap: what's done, what's next and who does it.")]
    tiles_html = "".join(f'<a class="tile" href="{p}"><b>{t}</b><span>{d}</span></a>' for p, t, d in tiles)
    body = f'''<section class="hero">
<span class="eyebrow">Built from scratch, trained on a Mac</span>
<h1>A GPT and a diffusion language model, from scratch</h1>
<p>A small GPT-style language model written in PyTorch and trained on Apple Silicon (and on Colab when it gets big), with
a local web app that shows every stage: data, tokens, attention, training, generation and evaluation. Next to it is a
masked diffusion language model of the same size, trained on the same 2.5B tokens. The two are compared on the
same footing.</p>
<div class="actions"><a class="btn primary" href="{REPO}">Code on GitHub</a><a class="btn" href="usage.html">Get started</a><a class="btn" href="{HF}">Models on Hugging Face</a></div>
</section>
<div class="tiles">{tiles_html}</div>
{table}'''
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


# ---------- build ----------

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
        edit = f'<p class="doc-src muted small">From <a href="{REPO}/blob/main/{src}">{src}</a></p>'
        content = f'<div class="doc-page">{toc_html(toc)}<article class="prose">{edit}{body}</article></div>'
        with open(os.path.join(out, dst), "w", encoding="utf-8") as f:
            f.write(page(f"{title} · TinyStories ARM", dst, content, wide=True))
    for a in sorted(assets):
        if os.path.exists(os.path.join(ROOT, a)):
            shutil.copy(os.path.join(ROOT, a), os.path.join(out, "assets", a.replace("/", "_")))
    for name in ("styles.css", "journey.js", "sidebyside.js"):
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
    data_path = os.path.join(HERE, "recorded", "sidebyside.json")
    if os.path.exists(data_path):
        shutil.copy(data_path, os.path.join(out, "assets", "sidebyside.json"))
        with open(os.path.join(out, "sidebyside.html"), "w", encoding="utf-8") as f:
            f.write(sidebyside_page(json.load(open(data_path))))
    open(os.path.join(out, ".nojekyll"), "w").close()
    n = sum(len(fs) for _, _, fs in os.walk(out))
    size = sum(os.path.getsize(os.path.join(d, x)) for d, _, fs in os.walk(out) for x in fs)
    print(f"built {out}: {n} files, {size / 1e6:.1f} MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build the project's website (GitHub Pages).")
    ap.add_argument("--out", default=os.path.join(ROOT, "_site"))
    build(ap.parse_args().out)

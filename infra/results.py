# J5: one results store. Every evaluation adds records to results/results.jsonl (committed), one JSON object per
# line, and docs/results.md is generated from it, so numbers are no longer copied into documents by hand.
#
#   python3 -m infra.results scan                    # add what's in models/, diffusion_models/runs/, outputs/ (idempotent)
#   python3 -m infra.results scan <folder> ...       # one folder, e.g. a run folder just pulled from Hugging Face
#   python3 -m infra.results render [--check]        # write docs/results.md (--check: fail if it's out of date)
#   python3 -m infra.results show [model]            # the latest numbers, as text
#
# A record:
#   {"kind": "benchmark", "model": "<id>", "name": "...", "model_type": "autoregressive", "params": 123552768,
#    "key": "hellaswag", "settings": {"context": 512}, "numbers": {"acc": 0.293, "acc_norm": 0.323, ...},
#    "measured_at": "2026-11-15 16:51", "recorded_at": "...", "commit": "675d4e4", "from": "models/<id>/benchmarks.json"}
#
# kinds: heldout (key = dataset), generation, benchmark (key = task), chat (sft.py compare), chat_heldout (dpo.py eval),
# bound (diffusion training's best likelihood bound), quality (key = sampler config, diffusion_models/quality.py),
# facts (key = chat / plain, facts.py).
# Diffusion runs are the models "diffusion:<run folder>". (kind, model, key, measured_at) identifies a measurement, so
# scanning the same files twice adds nothing; the latest measured_at wins when tables are built.
#
# The evaluation scripts still write their own JSON files next to each model (the app reads those); they also call
# record_file() after saving, which never lets a problem here stop an evaluation. Plain Python, no torch: the app's
# server (Python 3.9), CI and Colab all import it.
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORE = os.path.join(ROOT, "results", "results.jsonl")
DOC = os.path.join(ROOT, "docs", "results.md")
LIB = os.path.join(ROOT, "models")
RUNS = os.path.join(ROOT, "diffusion_models", "runs")
OUTPUTS = os.path.join(ROOT, "outputs")

# the benchmark tasks, their quoted metric (as on the Benchmarks page), chance, and OpenAI GPT-2 small's published score
TASKS = [("hellaswag", "HellaSwag", "acc_norm", 0.25, 0.3114), ("lambada", "LAMBADA", "acc", 0.0, 0.3256),
         ("arc_easy", "ARC-Easy", "acc", 0.25, 0.4381), ("piqa", "PIQA", "acc", 0.5, 0.6289),
         ("winogrande", "Winogrande", "acc", 0.5, 0.5162)]
SOURCES = {"roneneldan/TinyStories": "TinyStories", "BaoMinhTran/mini-fineweb-edu@small": "Mini FineWeb-Edu",
           "BaoMinhTran/mini-fineweb-edu@full": "FineWeb-Edu full + Cosmopedia"}


# ---------- reading and writing the store ----------

def load(path=None):
    path = path or STORE
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def ident(r):
    return (r["kind"], r["model"], r.get("key"), r.get("measured_at"))


def commit():
    """The checked-out commit ('-dirty' with uncommitted changes), or None outside a git checkout."""
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                             timeout=5).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                               capture_output=True, text=True, timeout=5).stdout.strip()
        return (sha + "-dirty" if dirty else sha) or None
    except Exception:
        return None


def add(records, path=None, with_commit=True):
    """Append the records that aren't in the store yet; returns how many were added."""
    path = path or STORE
    have = {ident(r) for r in load(path)}
    new, sha = [], commit() if with_commit else None
    now = time.strftime("%Y-%m-%d %H:%M")
    for r in records:
        if ident(r) in have:
            continue
        have.add(ident(r))
        new.append({**r, "recorded_at": now, "commit": r.get("commit", sha)})
    if new:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a") as f:
            for r in new:
                f.write(json.dumps(r, sort_keys=True) + "\n")
    return len(new)


def latest(records, kind=None, model=None):
    """{(kind, model, key): record} keeping the most recent measurement of each."""
    out = {}
    for r in records:
        if (kind and r["kind"] != kind) or (model and r["model"] != model):
            continue
        k = (r["kind"], r["model"], r.get("key"))
        if k not in out or (r.get("measured_at") or "") >= (out[k].get("measured_at") or ""):
            out[k] = r
    return out


# ---------- turning result files into records ----------

def _json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _mtime(path):
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(path)))


def _rel(path):
    p = os.path.abspath(path)
    return os.path.relpath(p, ROOT) if p.startswith(ROOT + os.sep) else os.path.basename(os.path.dirname(p)) + "/" + os.path.basename(p)


def describe(model_id, folder=None):
    """name, model_type and params of a library model or a diffusion run, from its own files when they're here."""
    if model_id.startswith("diffusion:"):
        d = folder or os.path.join(RUNS, model_id.split(":", 1)[1])
        m = _json(os.path.join(d, "metrics.json")) or {}
        datasets = m.get("datasets") or (_json(os.path.join(d, "config.json")) or {}).get("datasets") or []
        source = SOURCES.get(datasets[0], datasets[0]) if datasets else None
        params, tokens = m.get("params"), m.get("tokens")
        name = " · ".join(x for x in [f"Diffusion {round(params / 1e6)}M" if params else "Diffusion", source,
                                       f"{round(tokens / 1e6):,}M tokens" if tokens else None] if x)
        return {"name": name, "model_type": "masked-diffusion", "params": params}
    meta = _json(os.path.join(folder or os.path.join(LIB, model_id), "meta.json")) or {}
    return {"name": meta.get("name") or model_id, "model_type": meta.get("model_type") or "autoregressive",
            "params": meta.get("params")}


def _rec(kind, model, numbers, measured_at, src, key=None, settings=None, info=None):
    info = info or describe(model)
    return {"kind": kind, "model": model, "key": key, "numbers": numbers, "settings": settings or {},
            "measured_at": measured_at, "from": _rel(src), **info}


def _round(d):
    return {k: (round(v, 5) if isinstance(v, float) else v) for k, v in d.items()}


def from_file(path, model=None, info=None):
    """Records for one result file. model: the id it belongs to (eval.json, benchmarks.json, quality.json, metrics.json);
    chat_eval / dpo_eval / held_out_eval files name their models inside."""
    d = _json(path)
    if not isinstance(d, dict):
        return []
    name = os.path.basename(path)
    out = []
    if name == "eval.json" and model and "heldout" in d:
        when = d.get("evaluated_at") or _mtime(path)
        for key, v in d["heldout"].items():
            out.append(_rec("heldout", model, _round({"loss": v.get("loss"), "perplexity": v.get("perplexity")}), when,
                            path, key, {"context": v.get("context"), "tokens": v.get("tokens"), "title": v.get("title"),
                                        "suite_version": d.get("suite_version")}, info))
        g = dict(d.get("generation") or {})
        settings = g.pop("settings", {})
        if g:
            out.append(_rec("generation", model, _round(g), when, path, None, {**settings, "device": d.get("device")}, info))
    elif name == "benchmarks.json" and model:
        when = d.get("measured_at") or _mtime(path)
        for task, v in (d.get("results") or {}).items():
            out.append(_rec("benchmark", model, _round(v), when, path, task,
                            {"context": d.get("context"), "samples": d.get("samples"), "harness": d.get("harness")}, info))
    elif name == "quality.json" and model:
        when = d.get("measured_at") or _mtime(path)
        for cfg, v in (d.get("results") or {}).items():
            out.append(_rec("quality", model, _round(v.get("summary") or {}), when, path, cfg,
                            {**(v.get("settings") or {}), "n_tokens": d.get("n_tokens"), "seeds": d.get("seeds"),
                             "judge": d.get("judge"), "source": d.get("source")}, info))
    elif name == "metrics.json" and model and d.get("model_type") == "masked-diffusion" and d.get("best_bound"):
        out.append(_rec("bound", model, {"bound": d["best_bound"]}, d.get("finished_at") or _mtime(path), path, None,
                        {"tokens": d.get("tokens"), "updates": d.get("updates"), "gpu": d.get("gpu"),
                         "seconds": d.get("seconds")}, info))
    elif name in ("facts.json", "facts_plain.json") and model and "summary" in d:      # facts.py
        s = d["summary"]
        out.append(_rec("facts", model, _round({k: s.get(k) for k in ("acc", "acc_norm", "right_lp", "generated")}),
                        d.get("measured_at") or _mtime(path), path, d.get("format"),
                        {"n": s.get("n"), "facts_version": d.get("facts_version"), "by_topic": s.get("by_topic")}, info))
    elif "results" in d and isinstance(d["results"], dict):     # chat_eval.json (sft.py) / dpo_eval.json, held_out_eval.json (dpo.py)
        when = d.get("measured_at") or _mtime(path)
        for mid, v in d["results"].items():
            if not isinstance(v, dict):
                continue
            if "answer_loss" in v:
                ended = v.get("ended_by_itself", "")
                num, _, den = ended.partition("/")
                out.append(_rec("chat", mid, {"answer_loss": v["answer_loss"],
                                              "greedy_repeated_4grams": v.get("greedy_repeated_4grams"),
                                              "ended": round(int(num) / int(den), 3) if den else None},
                                when, path, None, {"questions": int(den) if den else None, "format": d.get("format")}))
            elif "reference_loss" in v:
                out.append(_rec("chat_heldout", mid, {k: v.get(k) for k in ("repeated_4grams", "ended", "mean_length",
                                                                           "reference_loss")},
                                when, path, None, {"n": d.get("n")}))
    return out


def from_folder(folder):
    """Every result file in a model library folder, a diffusion run folder, or an outputs folder."""
    folder = os.path.abspath(folder.rstrip("/"))
    meta = _json(os.path.join(folder, "meta.json"))
    metrics = _json(os.path.join(folder, "metrics.json")) or {}
    if meta and meta.get("id"):
        model = meta["id"]
    elif metrics.get("model_type") == "masked-diffusion" or os.path.exists(os.path.join(folder, "model.pt")):
        model = "diffusion:" + os.path.basename(folder)
    else:
        model = None
    info = describe(model, folder) if model else None
    out = []
    for name in ("eval.json", "benchmarks.json", "quality.json", "metrics.json", "chat_eval.json", "dpo_eval.json",
                 "held_out_eval.json", "facts.json", "facts_plain.json"):
        p = os.path.join(folder, name)
        if os.path.exists(p):
            out += from_file(p, model, info if name not in ("chat_eval.json", "dpo_eval.json", "held_out_eval.json") else None)
    return out


def scan(folders=None, path=None):
    if not folders:
        folders = [os.path.join(base, n) for base in (LIB, RUNS, OUTPUTS) if os.path.isdir(base)
                   for n in sorted(os.listdir(base)) if os.path.isdir(os.path.join(base, n))]
    return add([r for f in folders for r in from_folder(f)], path)


def record_file(path, model=None, say=None):
    """Called by the evaluation scripts right after they save a result file. Never raises."""
    try:
        if model and not model.startswith("diffusion:") and os.path.basename(path) in ("quality.json", "metrics.json",
                                                                                        "benchmarks.json") \
                and os.path.exists(os.path.join(os.path.dirname(path), "model.pt")):
            model = "diffusion:" + model
        info = describe(model, os.path.dirname(os.path.abspath(path))) if model else None
        n = add(from_file(path, model, info))
        if n and os.path.exists(DOC):
            write_doc()
        if say and n:
            say(f"results store: {n} new record{'s' if n != 1 else ''} in {_rel(STORE)}")
        return n
    except Exception as e:      # the evaluation already saved its own file; the store can be rebuilt with `scan`
        if say:
            say(f"results store: not updated ({e}); run `python3 -m infra.results scan` later")
        return 0


# ---------- the generated page ----------

def _pct(x):
    return "" if x is None else f"{100 * x:.1f}"


def _order(models, recs):
    first = {}
    for r in recs:
        if r["model"] in models:
            first[r["model"]] = min(first.get(r["model"], "9"), r.get("measured_at") or "9")
    return sorted(models, key=lambda m: (first.get(m, ""), m))


def _names(recs):
    out = {}
    for r in sorted(recs, key=lambda r: r.get("measured_at") or ""):
        if r.get("name") and r["name"] != r["model"]:
            out[r["model"]] = r["name"]
    return out


def _tokens(n):
    return "" if not n else f"{n / 1e9:.1f}B" if n >= 1e9 else f"{round(n / 1e6)}M"


def _title(key, title=None):
    """'huggingfacetb__cosmopedia-khanacademy' -> 'Cosmopedia khanacademy' when the suite didn't store a title."""
    if title and title != key:
        return title
    rest = key.split("__", 1)[-1]
    for prefix, nice in (("cosmopedia-", "Cosmopedia "), ("mini-fineweb-edu-", "Mini FineWeb-Edu ")):
        if rest.startswith(prefix):
            return nice + rest[len(prefix):]
    return rest


def _cell(text):
    return str(text).replace("|", "\\|")


def render(records=None):
    recs = load() if records is None else records
    names = _names(recs)
    name = lambda m: _cell(names.get(m, m))
    L = ["# Results", "",
         "Generated by `python3 -m infra.results render` from [results/results.jsonl](../results/results.jsonl), the "
         "one results store: every evaluation adds its numbers there. Don't edit this page by hand. The latest "
         "measurement of each model is shown; the store keeps the earlier ones, with when and at which commit each "
         "was measured.", ""]

    # benchmarks
    b = latest(recs, "benchmark")
    models = _order({m for (_, m, _) in b}, recs)
    if models:
        L += ["## Benchmarks", "",
              "Zero-shot, lm-evaluation-harness style, in percent: " +
              ", ".join(f"{t} {m}" for _, t, m, _, _ in TASKS) +
              ". Diffusion models score each choice by its likelihood bound. Average over the five tasks when all were run.", "",
              "| Model | " + " | ".join(t for _, t, _, _, _ in TASKS) + " | Average | Measured |",
              "|---|" + "---|" * (len(TASKS) + 2)]
        L.append("| *Chance* | " + " | ".join(_pct(c) for _, _, _, c, _ in TASKS) +
                 f" | {_pct(sum(c for *_, c, _ in TASKS) / len(TASKS))} | |")
        L.append("| *OpenAI GPT-2 small (published)* | " + " | ".join(_pct(g) for *_, g in TASKS) +
                 f" | {_pct(sum(g for *_, g in TASKS) / len(TASKS))} | |")
        for m in models:
            vals = [(b.get(("benchmark", m, k)) or {}).get("numbers", {}).get(metric) for k, _, metric, _, _ in TASKS]
            when = max((b[("benchmark", m, k)].get("measured_at") or "") for k, *_ in TASKS if ("benchmark", m, k) in b)
            avg = _pct(sum(vals) / len(vals)) if all(v is not None for v in vals) else ""
            L.append(f"| {name(m)} | " + " | ".join(_pct(v) for v in vals) + f" | {avg} | {when[:10]} |")
        L.append("")

    # held-out loss
    h = latest(recs, "heldout")
    if h:
        titles = {key: _title(key, (r.get("settings") or {}).get("title")) for (_, _, key), r in h.items()}
        groups = {}                     # models measured on the same data sets share a table
        for m in {m for (_, m, _) in h}:
            groups.setdefault(tuple(sorted(k for (_, mm, k) in h if mm == m)), []).append(m)
        when_of = lambda m: max((r.get("measured_at") or "") for (_, mm, _), r in h.items() if mm == m)
        L += ["## Held-out loss", "",
              "Cross-entropy (nats per token) on held-out text from `evaluate.py`; lower is better. One table per set of "
              "data sets the models were measured on (the suite grew as new data was added), newest first.", ""]
        for keys, models in sorted(groups.items(), key=lambda g: max(when_of(m) for m in g[1]), reverse=True):
            keys = sorted(keys, key=lambda k: titles[k])
            L += ["| Model | " + " | ".join(_cell(titles[k]) for k in keys) + " | Measured |",
                  "|---|" + "---|" * (len(keys) + 1)]
            for m in _order(set(models), recs):
                vals = [h[("heldout", m, k)]["numbers"].get("loss") for k in keys]
                L.append(f"| {name(m)} | " + " | ".join("" if v is None else f"{v:.3f}" for v in vals) +
                         f" | {when_of(m)[:10]} |")
            L.append("")

    # chat
    c, ch = latest(recs, "chat"), latest(recs, "chat_heldout")
    models = _order({m for (_, m, _) in c} | {m for (_, m, _) in ch}, recs)
    if models:
        L += ["## Chat", "",
              "`sft.py compare`: loss on held-out smol-smoltalk answers, and greedy answers to the fixed questions "
              "(repeated 4-grams, share that end by themselves). `dpo.py eval`: the same on 100 held-out questions, "
              "with the loss on their reference answers.", "",
              "| Model | Answer loss | Repetition | Ended | Held-out repetition | Held-out ended | Reference loss | Measured |",
              "|---|---|---|---|---|---|---|---|"]
        for m in models:
            a = (c.get(("chat", m, None)) or {}).get("numbers", {})
            o = (ch.get(("chat_heldout", m, None)) or {}).get("numbers", {})
            when = max((x.get("measured_at") or "") for x in (c.get(("chat", m, None)), ch.get(("chat_heldout", m, None))) if x)
            f3 = lambda v: "" if v is None else f"{v:.3f}"
            L.append(f"| {name(m)} | {f3(a.get('answer_loss'))} | {_pct(a.get('greedy_repeated_4grams'))} | "
                     f"{_pct(a.get('ended'))} | {_pct(o.get('repeated_4grams'))} | {_pct(o.get('ended'))} | "
                     f"{f3(o.get('reference_loss'))} | {when[:10]} |")
        L.append("")

    # diffusion
    bd, q = latest(recs, "bound"), latest(recs, "quality")
    models = _order({m for (_, m, _) in bd} | {m for (_, m, _) in q}, recs)
    if models:
        L += ["## Diffusion models", "",
              "Best likelihood bound on held-out text during training (nats per token, an upper bound on the loss), and "
              "`diffusion_models/quality.py`: 64-token continuations judged by an autoregressive model (its loss on the "
              "text; lower reads as more natural) and the share of repeated 4-grams, per sampler.", "",
              "| Model | Bound | Tokens | Sampler | Judge loss | Repetition | Distinct 2-grams |", "|---|---|---|---|---|---|---|"]
        for m in models:
            r = bd.get(("bound", m, None))
            refs = ("real continuation", "base AR model")      # the yardsticks first, then the samplers
            rows = sorted((k for (_, mm, k) in q if mm == m),
                          key=lambda k: (not any((k or "").startswith(x) for x in refs), k or ""))
            first = True
            for k in rows or [None]:
                n = (q.get(("quality", m, k)) or {}).get("numbers", {})
                head = (f"| {name(m)} | {r['numbers']['bound']:.3f} | {_tokens(r['settings'].get('tokens'))} |" if r
                        else f"| {name(m)} | | |") if first else "| | | |"
                L.append(f"{head} {_cell(k or '')} | {'' if n.get('judge_loss') is None else n['judge_loss']} | "
                         f"{_pct(n.get('repeated_4grams'))} | {_pct(n.get('distinct_2'))} |")
                first = False
        L.append("")
    return "\n".join(L)


def write_doc(path=None, check=False):
    path = path or DOC
    text = render()
    old = open(path).read() if os.path.exists(path) else None
    if check:
        return old == text
    if old != text:
        with open(path, "w") as f:
            f.write(text)
    return True


def show(model=None):
    lines = []
    for (kind, m, key), r in sorted(latest(load(), model=model).items(), key=lambda x: (x[0][1], x[0][0], x[0][2] or "")):
        nums = ", ".join(f"{k} {v}" for k, v in r["numbers"].items() if v is not None)
        lines.append(f"{m}  {kind}{' ' + key if key else ''}: {nums}  ({r.get('measured_at')})")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="The results store: results/results.jsonl and docs/results.md.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan", help="add the result files of these folders (default: every model, run and output)")
    s.add_argument("folders", nargs="*")
    r = sub.add_parser("render", help="write docs/results.md")
    r.add_argument("--check", action="store_true", help="exit 1 if docs/results.md is out of date")
    w = sub.add_parser("show")
    w.add_argument("model", nargs="?")
    a = ap.parse_args(argv)
    if a.cmd == "scan":
        n = scan(a.folders)
        print(f"{n} new record{'s' if n != 1 else ''} in {_rel(STORE)} ({len(load())} in all)")
        write_doc()
    elif a.cmd == "render":
        ok = write_doc(check=a.check)
        if a.check and not ok:
            print("docs/results.md is out of date: run python3 -m infra.results render")
            return 1
    elif a.cmd == "show":
        print(show(a.model))
    return 0


if __name__ == "__main__":
    sys.exit(main())

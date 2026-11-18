# Save runs for the website (G2). The website can't run models, so it replays runs recorded here; the app must be
# running (python3 app/server.py).
#   Side by side: each prompt goes through POST /api/sidebyside with the page's defaults → pages/recorded/sidebyside.json
#   Explorer: a few prompts through POST /api/explain on the model in use (pick the base model on the Models page
#   first), every block in detail → pages/recorded/explorer/ (one file per prompt, plus one per block's detail, so the
#   page only downloads the blocks you look at)
#
#   python3 pages/record.py [--only sidebyside|explorer] [--server http://localhost:8000] [--per-source 2]
import argparse
import json
import os
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "recorded", "sidebyside.json")
EX_DIR = os.path.join(HERE, "recorded", "explorer")
# short prompts (the Explorer reads up to 16 tokens): a fact to finish, a process, and a name to copy from earlier
EX_PROMPTS = ["The capital of France is Paris, and the capital of Italy is",
              "The water cycle is the process by which water moves",
              "Anna gave the book to Ben because Ben wanted to"]


def call(server, path, body=None):
    req = urllib.request.Request(server + path, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def record_explorer(server):
    os.makedirs(EX_DIR, exist_ok=True)
    model = (call(server, "/api/status").get("active") or {}).get("name") or "the model in use"
    index = []
    for n, prompt in enumerate(EX_PROMPTS):
        print(f"explorer: {prompt!r}", flush=True)
        base = call(server, "/api/explain", {"prompt": prompt, "layer": 0})
        passes = base["config"]["passes"]
        for b in range(passes):
            d = base if b == 0 else call(server, "/api/explain", {"prompt": prompt, "layer": b})
            with open(os.path.join(EX_DIR, f"p{n}-block{b}.json"), "w") as f:
                json.dump(d["layers"][b]["detail"], f, separators=(",", ":"))
        for L in base["layers"]:
            L.pop("detail", None)
        with open(os.path.join(EX_DIR, f"p{n}.json"), "w") as f:
            json.dump(base, f, separators=(",", ":"))
        index.append({"prompt": prompt, "tokens": len(base["tokens"]), "passes": passes})
    with open(os.path.join(EX_DIR, "index.json"), "w") as f:
        json.dump({"recorded_at": time.strftime("%Y-%m-%d %H:%M"), "model": model, "prompts": index}, f, indent=1)
    size = sum(os.path.getsize(os.path.join(EX_DIR, x)) for x in os.listdir(EX_DIR))
    print(f"saved {len(index)} prompts to {os.path.relpath(EX_DIR)} ({size / 1e6:.1f} MB) from {model}")


def main():
    ap = argparse.ArgumentParser(description="Record runs for the website.")
    ap.add_argument("--server", default="http://localhost:8000")
    ap.add_argument("--only", choices=["sidebyside", "explorer"])
    ap.add_argument("--per-source", type=int, default=2, help="side by side: prompts per held-out source (FineWeb-Edu, TinyStories)")
    a = ap.parse_args()
    if a.only != "sidebyside":
        record_explorer(a.server)
    if a.only == "explorer":
        return
    info = call(a.server, "/api/sidebyside")
    picked, count = [], {}
    for p in info["prompts"]:
        if count.get(p["source"], 0) < a.per_source:
            picked.append(p)
            count[p["source"]] = count.get(p["source"], 0) + 1
    runs = []
    for p in picked:
        print(f"{p['source']}: {p['prompt'][:60]!r}", flush=True)
        r = call(a.server, "/api/sidebyside", {"prompt_key": p["key"], "tokens": 64, "steps": 32, "ar_temperature": 1.0,
                                               "sampler": "keep", "repetition_penalty": 1.2, "block_size": 16, "seed": 1})
        runs.append(r)
    with open(OUT, "w") as f:
        json.dump({"recorded_at": time.strftime("%Y-%m-%d %H:%M"), "runs": runs}, f, separators=(",", ":"))
    print(f"saved {len(runs)} runs to {os.path.relpath(OUT)} ({os.path.getsize(OUT) / 1e3:.0f} KB)")


if __name__ == "__main__":
    main()

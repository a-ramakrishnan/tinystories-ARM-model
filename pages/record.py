# Save side-by-side runs for the website (G2). The website can't run models, so it replays runs recorded here:
# the app must be running (python3 app/server.py); each prompt goes through POST /api/sidebyside as on the
# Side by side page, with the page's defaults, and the results land in pages/data/sidebyside.json.
#
#   python3 pages/record.py [--server http://localhost:8000] [--per-source 2]
import argparse
import json
import os
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "sidebyside.json")


def call(server, path, body=None):
    req = urllib.request.Request(server + path, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser(description="Record side-by-side runs for the website.")
    ap.add_argument("--server", default="http://localhost:8000")
    ap.add_argument("--per-source", type=int, default=2, help="prompts per held-out source (FineWeb-Edu, TinyStories)")
    a = ap.parse_args()
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

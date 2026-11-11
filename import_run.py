# Bring a model trained on a cloud GPU (Colab or Kaggle) back to this Mac.
#
#   python3 import_run.py ~/Downloads/tinystories-run-roneneldan__tinystories-20261110-1430.zip
#
# The zip is made by Step 10 of train_step_by_step.ipynb. This script:
#   1. puts best_model_params.pt and model_config.json in data/<dataset>/, after backing up the ones there,
#   2. adds the run to data/<dataset>/runs.jsonl, so it shows in the web app's training history,
#   3. saves a copy in the model library (models/, see registry.py), and
#   4. keeps the run's extras (loss chart, metrics, sample story) in data/<dataset>/imported/<zip name>/.
# Then `python3 app/server.py` uses the imported model for Generate and Inside the transformer.
#
# It also takes the zip from an A/B test (ab_test.py, or the A/B section of the notebook): every variant goes
# into the model library with its evaluation, and the comparison into outputs/ab/<stamp>/.

import argparse
import json
import os
import shutil
import sys
import time
import zipfile

import registry

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(PROJECT_DIR, "data")
REQUIRED = ("run.json", "best_model_params.pt", "model_config.json")


def import_ab(zip_path, z):
    """An A/B test zip: one folder per variant (weights, config, eval.json) plus ab.json, results.md, losses.png."""
    summary = json.loads(z.read("ab.json"))
    stamp = os.path.basename(summary["stamp"])
    out_dir = os.path.join(PROJECT_DIR, "outputs", "ab", stamp)
    os.makedirs(out_dir, exist_ok=True)
    for name in z.namelist():
        dest = os.path.realpath(os.path.join(out_dir, name))
        if not dest.startswith(os.path.realpath(out_dir) + os.sep) or name.endswith("/"):
            continue  # never write outside the folder
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with z.open(name) as src, open(dest, "wb") as out:
            shutil.copyfileobj(src, out)
    title = summary.get("title") or "A/B"
    print(f"A/B test \"{title}\" from {summary.get('chip')} ({summary.get('steps', 0):,} updates per variant):")
    for v in summary["variants"]:
        folder = os.path.join(out_dir, os.path.basename(v["name"]))
        weights, cfg = os.path.join(folder, "best_model_params.pt"), os.path.join(folder, "model_config.json")
        if not (os.path.exists(weights) and os.path.exists(cfg)):
            print(f"  {v['name']}: no weights (diverged?), skipped")
            continue
        meta, added = registry.add_model(weights, cfg, source="imported", origin=os.path.basename(zip_path),
                                         name=f"{title} · {v['name']} · {v['description']} · {v['steps']:,} updates · {summary.get('chip')}")
        ev_path = os.path.join(folder, "eval.json")
        if os.path.exists(ev_path):
            with open(ev_path) as f:
                registry.save_eval(meta["id"], json.load(f))
        print(f"  {v['name']}: best validation loss {v['best_val']}  ->  {'saved' if added else 'already'} in the "
              f"library as \"{meta['name']}\"")
    print(f"\nThe comparison is in {os.path.relpath(out_dir, PROJECT_DIR)}/results.md (chart: losses.png);"
          f" the Models step of the web app compares the variants.")
    return out_dir


def import_run(zip_path, data_root=DATA_ROOT):
    with zipfile.ZipFile(zip_path) as z:
        names = set(z.namelist())
        if "ab.json" in names:
            return import_ab(zip_path, z)
        missing = [n for n in REQUIRED if n not in names]
        if missing:
            raise SystemExit(f"{zip_path} isn't a results zip from the notebook (missing {', '.join(missing)}).")
        run = json.loads(z.read("run.json"))
        slug = os.path.basename(run["slug"])  # basename: never write outside data/
        target = os.path.join(data_root, slug)
        os.makedirs(target, exist_ok=True)

        # 1. back up the current model, then install the new one
        stamp = time.strftime("%Y%m%d-%H%M%S")
        for name in ("best_model_params.pt", "model_config.json"):
            dest = os.path.join(target, name)
            if os.path.exists(dest):
                backup = os.path.join(target, "backups", stamp)
                os.makedirs(backup, exist_ok=True)
                shutil.move(dest, os.path.join(backup, name))
                print(f"Backed up the previous {name} to {os.path.relpath(backup, PROJECT_DIR)}/")
            with z.open(name) as src, open(dest, "wb") as out:
                shutil.copyfileobj(src, out)

        # 1b. the training state, if the zip has it, so the run can be continued (full-model.py --resume)
        if "training_state.pt" in names:
            for name in ("training_state.pt", "training_state.json"):
                if name in names:
                    with z.open(name) as src, open(os.path.join(target, name), "wb") as out:
                        shutil.copyfileobj(src, out)
            print("Installed the training state: continue this run with Continue training in the web app's Train step,")
            print("or  python3 full-model.py --resume --max-iters <total steps>")

        # 2. keep a copy in the model library (models/), where it's never overwritten
        meta, added = registry.add_model(os.path.join(target, "best_model_params.pt"), os.path.join(target, "model_config.json"),
                                         source="imported", origin=os.path.basename(zip_path))
        print(f"{'Saved' if added else 'Already'} in the model library as \"{meta['name']}\" (models/{meta['id']})")

        # 3. add the run to the training history
        if run.get("summary"):
            with open(os.path.join(target, "runs.jsonl"), "a") as f:
                f.write(json.dumps({**run["summary"], "imported_from": os.path.basename(zip_path)}) + "\n")

        # 4. keep the extras
        extras_dir = os.path.join(target, "imported", os.path.splitext(os.path.basename(zip_path))[0])
        extras = [n for n in names if n not in ("best_model_params.pt", "model_config.json", "training_state.pt",
                                                 "training_state.json") and "/" not in n]
        if extras:
            os.makedirs(extras_dir, exist_ok=True)
            for name in extras:
                with z.open(name) as src, open(os.path.join(extras_dir, name), "wb") as out:
                    shutil.copyfileobj(src, out)

    s = run.get("summary", {})
    print(f"\nImported into {os.path.relpath(target, PROJECT_DIR)}/")
    if s:
        best = f"{s['best_val']:.3f}" if s.get("best_val") is not None else "n/a"
        print(f"  trained on {s.get('chip', '?')} ({s.get('platform', '?')}): {s.get('steps', 0):,} steps, "
              f"{s.get('params', 0) / 1e6:.1f}M parameters, best validation loss {best}")
    if extras:
        print(f"  extras ({', '.join(sorted(extras))}) in {os.path.relpath(extras_dir, PROJECT_DIR)}/")
    if not os.path.isdir(os.path.join(target, "dataset")):
        print(f"\nNote: {run.get('dataset')} isn't downloaded on this Mac yet. Add it in the web app's Dataset step "
              f"(it's validated and downloaded once) to use this model there.")
    print("\nStart the app with  python3 app/server.py  and the Generate step will use this model.")
    return target


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Import a model trained on Colab or Kaggle (the zip from Step 10), or an A/B test zip.")
    ap.add_argument("zip", help="path to tinystories-run-*.zip or ab-*.zip")
    args = ap.parse_args()
    if not os.path.exists(args.zip):
        sys.exit(f"Not found: {args.zip}")
    import_run(args.zip)

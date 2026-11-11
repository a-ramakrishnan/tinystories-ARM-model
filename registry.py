# Model library: every trained or imported model, kept under its own name so none is ever overwritten.
#
#   models/<id>/best_model_params.pt   the weights (a copy, so later training can't change them)
#   models/<id>/model_config.json      model config + training summary (as written by save_model_config)
#   models/<id>/meta.json              name, type, source, fingerprint, and a summary for listing
#   models/<id>/eval.json              results of the evaluation suite (see evaluate.py), if run
#
# Command line:
#   python3 registry.py list           show the library
#   python3 registry.py scan           add every model found under data/ (current ones and backups)

import hashlib
import json
import os
import re
import shutil
import sys
import time

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
LIB_ROOT = os.path.join(PROJECT_DIR, "models")
DATA_ROOT = os.path.join(PROJECT_DIR, "data")
WEIGHTS = "best_model_params.pt"
CONFIG = "model_config.json"
META = "meta.json"
EVAL = "eval.json"
ACTIVE = "active.json"   # the model the web app uses for generation

SIZE_NAMES = {(4, 256): "Tiny", (6, 384): "Small", (8, 512): "Medium", (12, 768): "Large"}
DATASET_TITLES = {"roneneldan/TinyStories": "TinyStories", "BaoMinhTran/mini-fineweb-edu": "Mini FineWeb-Edu",
                  "Salesforce/wikitext": "WikiText"}


def fingerprint(path):
    """Fast identity for a weights file: its size plus a hash of the first and last megabyte."""
    size = os.path.getsize(path)
    h = hashlib.sha1(str(size).encode())
    with open(path, "rb") as f:
        h.update(f.read(1 << 20))
        if size > 2 << 20:
            f.seek(-(1 << 20), os.SEEK_END)
            h.update(f.read(1 << 20))
    return h.hexdigest()[:16]


def _short_dataset(name):
    if not name:
        return "unknown data"
    if name in DATASET_TITLES:
        return DATASET_TITLES[name]
    if name.startswith("http"):
        if "tinyshakespeare" in name:
            return "Tiny Shakespeare"
        return os.path.basename(name.split("?")[0]) or name
    return name.split("/")[-1]


def default_name(info):
    m = info["model"]
    size = SIZE_NAMES.get((m["n_layer"], m["n_embd"]), f"{m['n_layer']}L·{m['n_embd']}")
    params = info.get("params")
    data = " + ".join(_short_dataset(d) for d in (info.get("datasets") or [])) or "unknown data"
    steps = info.get("steps")
    steps = f"{steps / 1000:g}k steps" if steps and steps >= 1000 else f"{steps} steps" if steps else ""
    parts = [f"{size} {params / 1e6:.0f}M" + (" RoPE" if m.get("pos_encoding") == "rope" else "")
                                       + (" RMSNorm" if m.get("norm") == "rmsnorm" else "")
                                       + (" SwiGLU" if m.get("mlp") == "swiglu" else "")
                                       + (" no-bias" if m.get("bias") is False else "")
                                       + (" QK-norm" if m.get("qk_norm") else "")
                                       + (f" GQA-{m['n_kv_head']}" if m.get("n_kv_head") and m["n_kv_head"] != m.get("n_head") else "") if params else size, data, steps]
    return " · ".join(p for p in parts if p)


def _model_params(cfg):
    """Parameter count from the config (weight-tied GPT), without building the model."""
    from model import GPTConfig, param_breakdown
    return param_breakdown(GPTConfig(**cfg))["total"]


def _read_config(path):
    with open(path) as f:
        return json.load(f)


# Old checkpoints don't record their datasets; their folder name tells us
FOLDER_DATASETS = {"roneneldan__tinystories": "roneneldan/TinyStories",
                   "baominhtran__mini-fineweb-edu-small": "BaoMinhTran/mini-fineweb-edu",
                   "salesforce__wikitext-wikitext-2-raw-v1": "Salesforce/wikitext"}


def _datasets_from_path(weights):
    parts = os.path.normpath(os.path.abspath(weights)).split(os.sep)
    for part in parts:
        if part in FOLDER_DATASETS:
            return [FOLDER_DATASETS[part]]
    return None


def _summary(info, weights):
    cfg = info["model"]
    return {
        "model_type": info.get("model_type", "autoregressive"),
        "model": cfg,
        "params": info.get("params") or _model_params(cfg),
        "datasets": info.get("datasets") or _datasets_from_path(weights),
        "dataset_config": info.get("dataset_config"),
        "steps": info.get("steps"), "updates": info.get("updates"), "tokens": info.get("tokens"),
        "best_val": info.get("best_val"), "seconds": info.get("seconds"),
        "trained_at": info.get("trained_at"), "chip": info.get("chip"), "platform": info.get("platform"),
        "device": info.get("device"), "dtype": info.get("dtype"),
        "bytes": os.path.getsize(weights),
    }


def _new_id(summary, fp):
    when = re.sub(r"\D", "", summary.get("trained_at") or "")[:12] or time.strftime("%Y%m%d%H%M")
    data = re.sub(r"[^a-z0-9]+", "-", _short_dataset((summary.get("datasets") or [None])[0]).lower()).strip("-")
    return f"{data[:20] or 'model'}-{when}-{fp[:6]}"


def list_models(root=LIB_ROOT):
    """Every model in the library, newest first. Each entry is its meta.json plus 'eval' if evaluated."""
    if not os.path.isdir(root):
        return []
    out = []
    for d in os.listdir(root):
        meta_path = os.path.join(root, d, META)
        if not os.path.exists(meta_path):
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        ev = os.path.join(root, d, EVAL)
        if os.path.exists(ev):
            with open(ev) as f:
                meta["eval"] = json.load(f)
        out.append(meta)
    return sorted(out, key=lambda m: (m.get("trained_at") or "", m.get("added_at") or ""), reverse=True)


def model_dir(model_id, root=LIB_ROOT):
    if not re.fullmatch(r"[A-Za-z0-9._-]+", model_id or ""):
        raise ValueError("Bad model id")
    path = os.path.join(root, model_id)
    if not os.path.exists(os.path.join(path, META)):
        raise KeyError(f"No saved model '{model_id}'")
    return path


def get_model(model_id, root=LIB_ROOT):
    with open(os.path.join(model_dir(model_id, root), META)) as f:
        return json.load(f)


def paths_of(model_id, root=LIB_ROOT):
    d = model_dir(model_id, root)
    return os.path.join(d, WEIGHTS), os.path.join(d, CONFIG)


def find_by_fingerprint(fp, root=LIB_ROOT):
    return next((m for m in list_models(root) if m.get("fingerprint") == fp), None)


def add_model(weights, config_path, source="trained", name=None, root=LIB_ROOT, origin=None):
    """Copy a checkpoint into the library. Returns (meta, added): an identical model is only stored once."""
    fp = fingerprint(weights)
    existing = find_by_fingerprint(fp, root)
    if existing:
        return existing, False
    info = _read_config(config_path)
    summary = _summary(info, weights)
    model_id = _new_id(summary, fp)
    path = os.path.join(root, model_id)
    os.makedirs(path, exist_ok=True)
    shutil.copy2(weights, os.path.join(path, WEIGHTS))
    shutil.copy2(config_path, os.path.join(path, CONFIG))
    meta = {"id": model_id, "name": name or default_name(summary), "source": source, "fingerprint": fp,
            "origin": origin or os.path.relpath(os.path.dirname(os.path.abspath(weights)), PROJECT_DIR), "added_at": time.strftime("%Y-%m-%d %H:%M"),
            **summary}
    with open(os.path.join(path, META), "w") as f:
        json.dump(meta, f, indent=2)
    return meta, True


def rename_model(model_id, name, root=LIB_ROOT):
    meta = get_model(model_id, root)
    meta["name"] = (name or "").strip()[:120] or meta["name"]
    with open(os.path.join(model_dir(model_id, root), META), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def delete_model(model_id, root=LIB_ROOT):
    shutil.rmtree(model_dir(model_id, root))


def save_eval(model_id, results, root=LIB_ROOT):
    with open(os.path.join(model_dir(model_id, root), EVAL), "w") as f:
        json.dump(results, f, indent=2)


def load_eval(model_id, root=LIB_ROOT):
    path = os.path.join(model_dir(model_id, root), EVAL)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_active(root=LIB_ROOT):
    """The saved model last chosen for generation in the web app, if it still exists."""
    try:
        with open(os.path.join(root, ACTIVE)) as f:
            model_id = json.load(f).get("id")
        model_dir(model_id, root)
        return model_id
    except (OSError, ValueError, KeyError, AttributeError):
        return None


def save_active(model_id, root=LIB_ROOT):
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, ACTIVE), "w") as f:
        json.dump({"id": model_id}, f)


def scan(data_root=DATA_ROOT, root=LIB_ROOT):
    """Add every checkpoint found under data_root (each dataset's current model and its backups)."""
    found = []
    for dirpath, _, files in os.walk(data_root):
        if WEIGHTS in files and CONFIG in files:
            source = "backup" if f"{os.sep}backups{os.sep}" in dirpath + os.sep else "found"
            try:
                meta, added = add_model(os.path.join(dirpath, WEIGHTS), os.path.join(dirpath, CONFIG), source=source, root=root)
                found.append((meta, added))
            except Exception as e:  # an unreadable config shouldn't stop the scan
                print(f"Skipped {dirpath}: {e}")
    return found


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"
    if cmd == "scan":
        for meta, added in scan():
            print(f"{'added   ' if added else 'existing'}  {meta['id']}  {meta['name']}")
    for m in list_models():
        val = f"{m['best_val']:.3f}" if m.get("best_val") is not None else "–"
        print(f"{m['id']:45s} {m['name']:55s} val {val}  {m['bytes'] / 1e6:6.0f} MB  ({m['source']})")

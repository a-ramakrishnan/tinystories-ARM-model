# Local web front end for training the model step by step.
# Run from the project folder:  python3 app/server.py   (then open http://127.0.0.1:8000)
#
# The server wraps the same modules full-model.py uses. Long-running work (downloading,
# tokenizing, training) runs in a background thread; the page polls /api/status for progress.
# Generation streams tokens to the page as newline-delimited JSON.

import argparse
import json
import math
import os
import platform
import random
import subprocess
import sys
import threading
import time
import traceback
import webbrowser
from dataclasses import asdict
from typing import List, Optional

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

import torch  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse, StreamingResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from catalog import DATASET_CATALOG  # noqa: E402
from data import (DEFAULT_CONFIG, DEFAULT_DATASET, BatchLoader, SubsetNeeded, dataset_paths, get_dataset,  # noqa: E402
                  inspect_source, mix_paths, resolve_source, token_count)
from inference import (attention_maps, benchmark_generation, device_memory, generate_stream,  # noqa: E402
                       kv_bytes_per_token, load_model, load_model_config, next_token_probs, sampling_distribution,
                       save_model_config)
from model import GPT, GPTConfig, param_breakdown  # noqa: E402
import evaluate  # noqa: E402
import registry  # noqa: E402
from pretrain import pretrain, read_state_meta  # noqa: E402
from tokenization import bin_files_exist, get_tokenizer, tokenize_and_save  # noqa: E402
from train_config import TrainConfig, setup_device  # noqa: E402

DATA_ROOT = os.path.join(PROJECT_DIR, "data")
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
BYTES_PER_PARAM_TRAINING = 16  # float32 weights + gradients + AdamW's two moment buffers


def chip_name():
    if sys.platform == "darwin":
        try:
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        except Exception:
            pass
    return platform.processor() or platform.machine()


def system_memory():
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None


class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.dev = setup_device()
        self.enc = get_tokenizer()
        self.chip = chip_name()
        self.memory = system_memory()
        self.datasets = {}             # root path -> {name, config_name, text_column, paths, ds, was_local}
        self.mix_mode = "size"         # how batches are drawn from several datasets: "size" or "equal"
        self.batch = {"block_size": 256, "batch_size": 32, "seed": 42}
        self.model_config = None
        self.model = None
        self.job = {"kind": None, "status": "idle", "message": "", "progress": None, "error": None}
        self.train = None              # live training progress
        self.stop_requested = False
        self.inference_cache = None    # (checkpoint path, mtime, model, config)
        self.active_model = registry.load_active()  # saved model (models/) to use instead of the datasets' model


S = State()
app = FastAPI(title="TinyStories LM Studio")


# ---------- helpers ----------

def start_job(kind, fn):
    with S.lock:
        if S.job["status"] == "running":
            raise HTTPException(409, f"Please wait: {S.job['kind']} is still running.")
        S.job = {"kind": kind, "status": "running", "message": "Starting…", "progress": None,
                 "error": None, "started": time.time()}

    def run():
        try:
            fn()
            if S.job["status"] == "running":
                S.job.update(status="done", progress=1.0)
        except Exception as e:  # surface any failure to the page
            traceback.print_exc()
            S.job.update(status="error", error=str(e), message=str(e))

    threading.Thread(target=run, daemon=True).start()
    return {"ok": True}


def training_running():
    return S.job["kind"] == "train" and S.job["status"] == "running"


def require_idle(action):
    if S.job["status"] == "running":
        raise HTTPException(409, f"Can't {action} while {S.job['kind']} is running.")


def selected():
    return list(S.datasets.values())


def require_datasets():
    if not S.datasets:
        raise HTTPException(400, "Add at least one dataset first (step 1).")
    return selected()


def require_tokenized():
    ds = require_datasets()
    missing = [d["name"] for d in ds if not bin_files_exist(d["paths"].root)]
    if missing:
        raise HTTPException(400, f"Tokenize first (step 2): {', '.join(missing)}")
    return ds


def selection_paths():
    return mix_paths([d["paths"].root for d in selected()], DATA_ROOT) if S.datasets else None


def selection_loader(block_size, batch_size, device):
    return BatchLoader([d["paths"].root for d in selected()], block_size, batch_size, device, mix=S.mix_mode)


def tokens_of(ids):
    return [{"id": int(i), "text": S.enc.decode([int(i)])} for i in ids]


def checkpoint_info(paths):
    if paths is None or not (os.path.exists(paths.checkpoint) and os.path.exists(paths.model_config)):
        return None
    try:
        cfg, info = load_model_config(paths.model_config)
    except Exception:
        return None
    return {"model": asdict(cfg), "params": param_breakdown(cfg), "steps": info.get("steps"),
            "updates": info.get("updates"), "best_val": info.get("best_val"), "trained_at": info.get("trained_at"),
            "stopped_early": info.get("stopped_early", False), "datasets": info.get("datasets")}


def read_runs(paths, n=10):
    f = os.path.join(paths.root, "runs.jsonl") if paths else None
    if not f or not os.path.exists(f):
        return []
    with open(f) as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    return rows[-n:][::-1]


def dataset_info(d):
    p = d["paths"]
    tokenized = bin_files_exist(p.root)
    return {
        "key": p.root, "name": d["name"], "config_name": d["config_name"], "text_column": d["text_column"],
        "path": os.path.relpath(p.root, PROJECT_DIR), "was_local": d["was_local"],
        "splits": {k: len(v) for k, v in d["ds"].items()}, "tokenized": tokenized,
        "tokens": {"train": token_count(p.train_bin), "validation": token_count(p.val_bin)} if tokenized else None,
    }


def _resume_info(paths):
    st = read_state_meta(paths.root) if paths else None
    if not st:
        return None
    return {k: st.get(k) for k in ("step", "max_iters", "best_val", "last_val", "lr", "eval_interval")} | \
        {"model": st.get("model_config"), "batch_size": st["train_config"]["batch_size"]}


def selection_info():
    if not S.datasets:
        return None
    infos = [dataset_info(d) for d in selected()]
    paths = selection_paths()
    tokenized = all(i["tokenized"] for i in infos)
    shares = None
    if tokenized:
        sizes = [i["tokens"]["train"] for i in infos]
        w = [1.0] * len(sizes) if S.mix_mode == "equal" else [float(s) for s in sizes]
        shares = [x / sum(w) for x in w]
    return {
        "count": len(infos), "path": os.path.relpath(paths.root, PROJECT_DIR), "mix_mode": S.mix_mode,
        "tokenized": tokenized, "shares": shares,
        "tokens": {k: sum(i["tokens"][k] for i in infos) for k in ("train", "validation")} if tokenized else None,
        "checkpoint": checkpoint_info(paths), "runs": read_runs(paths),
        "resume": _resume_info(paths),
    }


def _load_cached(weights, config_path):
    mtime = os.path.getmtime(weights)
    if not (S.inference_cache and S.inference_cache[0] == weights and S.inference_cache[1] == mtime):
        S.inference_cache = None  # free the old model before loading the next one
        config, _ = load_model_config(config_path)
        S.inference_cache = (weights, mtime, load_model(config, weights, S.dev.device), config)
    return S.inference_cache[2]


def set_active(model_id):
    """Choose the saved model Generate uses (None: the datasets' own model), and remember it across restarts."""
    S.active_model = model_id
    registry.save_active(model_id)


def active_library_model():
    """The saved model chosen in the Models step, if it still exists."""
    if S.active_model:
        try:
            return registry.get_model(S.active_model)
        except (KeyError, ValueError):
            set_active(None)
    return None


def active_info():
    """The model Generate and Inside the transformer use: a saved model, or the datasets' own model."""
    meta = active_library_model()
    if meta:
        cfg = GPTConfig(**meta["model"])
        return {"source": "library", "id": meta["id"], "name": meta["name"], "model": meta["model"],
                "params": param_breakdown(cfg), "steps": meta.get("steps"), "updates": meta.get("updates"),
                "best_val": meta.get("best_val"), "trained_at": meta.get("trained_at"), "stopped_early": False,
                "datasets": meta.get("datasets"), "chip": meta.get("chip"), "platform": meta.get("platform")}
    ck = checkpoint_info(selection_paths())
    return {"source": "datasets", **ck} if ck else None


def require_model():
    """Generation needs either a chosen saved model or datasets with a trained model."""
    if not active_library_model():
        require_datasets()


def inference_model(allow_untrained=False):
    """The model to run (cached): the chosen saved model, else the current datasets' trained model, else
    the built untrained model if allowed."""
    meta = active_library_model()
    if meta:
        return _load_cached(*registry.paths_of(meta["id"])), "trained"
    paths = selection_paths()
    if checkpoint_info(paths) is not None:
        return _load_cached(paths.checkpoint, paths.model_config), "trained"
    if allow_untrained and S.model is not None:
        return S.model, "untrained"
    raise HTTPException(400, "No trained model for these datasets yet. Train one in the Train step.")


# ---------- request bodies ----------

class DatasetReq(BaseModel):
    name: str = DEFAULT_DATASET
    config_name: Optional[str] = DEFAULT_CONFIG
    text_column: str = "text"


class KeyReq(BaseModel):
    key: str


class MixReq(BaseModel):
    mode: str = "size"


class TextReq(BaseModel):
    text: str


class TextsReq(BaseModel):
    texts: List[str]


class BatchReq(BaseModel):
    block_size: int = 256
    batch_size: int = 32
    seed: int = 42


class ModelReq(BaseModel):
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0


class TrainReq(BaseModel):
    max_iters: int = 2000
    learning_rate: float = 6e-4
    min_lr: float = 6e-5
    warmup_steps: int = 200
    eval_interval: int = 250
    eval_iters: int = 50
    gradient_accumulation_steps: int = 1


class InspectReq(BaseModel):
    prompt: str = "The little cat sat on the"
    layer: int = 0


class SamplingReq(BaseModel):
    temperature: float = 0.8          # 0 = greedy (always the most likely token)
    top_k: int = 0                    # 0 = off
    top_p: float = 0.9                # 1 = off   (defaults: the Balanced preset)
    min_p: float = 0.0                # 0 = off
    repetition_penalty: float = 1.15  # 1 = off
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

    def clean(self):
        """(temperature, top_k, extras for generate_stream), each clamped to a sensible range."""
        t = 0.0 if self.temperature <= 0.001 else max(0.05, min(self.temperature, 3.0))
        extras = {"top_p": max(0.01, min(self.top_p, 1.0)), "min_p": max(0.0, min(self.min_p, 1.0)),
                  "repetition_penalty": max(1.0, min(self.repetition_penalty, 3.0)),
                  "frequency_penalty": max(0.0, min(self.frequency_penalty, 2.0)),
                  "presence_penalty": max(0.0, min(self.presence_penalty, 2.0))}
        return t, (self.top_k if self.top_k > 0 else None), extras


class GenerateReq(SamplingReq):
    prompt: str = "Once upon a time"
    max_new_tokens: int = 200
    use_kv_cache: bool = True


class DistributionReq(SamplingReq):
    prompt: str = "The water cycle is the process by which"
    n: int = 24


class CompareReq(BaseModel):
    prompt: str = "The water cycle is the process by which"
    max_new_tokens: int = 120
    seed: int = 1234
    settings: List[SamplingReq]


class BenchmarkReq(BaseModel):
    prompt: str = "The water cycle is the process by which"
    n_tokens: int = 100
    length: Optional[str] = None   # "fill": exactly fill the context window; "past": go 80 tokens beyond it


# ---------- status & datasets ----------

@app.get("/api/status")
def status():
    return {
        "device": S.dev.device, "dtype": S.dev.dtype, "chip": S.chip, "memory": S.memory,
        "torch": torch.__version__, "default_dataset": DEFAULT_DATASET, "default_config": DEFAULT_CONFIG,
        "datasets": [dataset_info(d) for d in selected()],
        "selection": selection_info(),
        "active": active_info(),
        "library_count": len(registry.list_models()),
        "batch": S.batch,
        "model": {"config": asdict(S.model_config), "params": sum(p.numel() for p in S.model.parameters()),
                  "breakdown": param_breakdown(S.model_config)} if S.model is not None else None,
        "job": S.job,
        "train": S.train,
    }


@app.get("/api/catalog")
def catalog():
    """Public text datasets known to work, with whether each is already downloaded, tokenized, trained or selected."""
    out = []
    for item in DATASET_CATALOG:
        p = dataset_paths(item["dataset"], item["config"], DATA_ROOT)
        out.append({**item, "key": p.root, "downloaded": os.path.isdir(p.hf_dir), "tokenized": bin_files_exist(p.root),
                    "trained": checkpoint_info(p) is not None, "selected": p.root in S.datasets})
    return out


def _clean_request(req):
    name = req.name.strip()
    config_name = (req.config_name or "").strip() or None
    if not name:
        name, config_name = DEFAULT_DATASET, config_name or DEFAULT_CONFIG
    text_column = req.text_column.strip() or "text"
    try:
        src = resolve_source(name)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if src["kind"] == "url" and src["builder"] == "text":
        text_column = "text"  # plain text files have a single 'text' column
    return name, config_name, text_column


@app.post("/api/dataset/check")
def check_dataset(req: DatasetReq):
    """Validate a dataset (name or link) without downloading it."""
    name, config_name, text_column = _clean_request(req)
    try:
        info = inspect_source(name, config_name, text_column)
    except SubsetNeeded as e:
        return {"ok": False, "error": str(e), "configs": e.configs}
    except ValueError as e:
        return {"ok": False, "error": str(e)}
    p = dataset_paths(name, config_name, DATA_ROOT)
    return {"ok": True, **info, "downloaded": os.path.isdir(p.hf_dir), "selected": p.root in S.datasets,
            "path": os.path.relpath(p.root, PROJECT_DIR)}


@app.post("/api/dataset")
def add_dataset(req: DatasetReq):
    """Add a dataset to the training data (downloading it the first time)."""
    name, config_name, text_column = _clean_request(req)
    paths = dataset_paths(name, config_name, DATA_ROOT)

    def job():
        was_local = os.path.isdir(paths.hf_dir)
        S.job["message"] = "Loading the local copy…" if was_local else f"Downloading {name} (first time only)…"
        ds = get_dataset(paths.hf_dir, name, config_name=config_name, text_column=text_column)
        S.datasets[paths.root] = {"name": name, "config_name": config_name, "text_column": text_column,
                                  "paths": paths, "ds": ds, "was_local": was_local}
        S.train, S.inference_cache = None, None
        S.job["message"] = f"Added {name}."

    return start_job("dataset", job)


@app.post("/api/dataset/remove")
def remove_dataset(req: KeyReq):
    require_idle("change the datasets")
    S.datasets.pop(req.key, None)
    S.train, S.inference_cache = None, None
    return {"ok": True}


@app.post("/api/mix")
def set_mix(req: MixReq):
    require_idle("change the mix")
    if req.mode not in ("size", "equal"):
        raise HTTPException(400, "mode must be 'size' or 'equal'.")
    S.mix_mode = req.mode
    return {"ok": True}


@app.get("/api/dataset/sample")
def dataset_sample(key: str = "", split: str = "train", index: int = -1):
    ds = require_datasets()
    d = S.datasets.get(key) or ds[0]
    if split not in d["ds"]:
        raise HTTPException(400, f"Unknown split '{split}'.")
    total = len(d["ds"][split])
    if index < 0 or index >= total:
        index = random.randrange(total)
    return {"key": d["paths"].root, "split": split, "index": index, "total": total, "text": d["ds"][split][index]["text"]}


# ---------- tokenize ----------

@app.post("/api/tokenize/preview")
def tokenize_preview(req: TextReq):
    ids = S.enc.encode_ordinary(req.text[:5000])
    return {"chars": len(req.text), "tokens": tokens_of(ids)}


@app.post("/api/tokenize/many")
def tokenize_many(req: TextsReq):
    return [{"text": t, "tokens": tokens_of(S.enc.encode_ordinary(t[:500]))} for t in req.texts[:20]]


@app.post("/api/tokenize")
def tokenize_endpoint():
    todo = [d for d in require_datasets() if not bin_files_exist(d["paths"].root)]
    if not todo:
        raise HTTPException(400, "Everything is already tokenized.")

    def job():
        for n, d in enumerate(todo, 1):
            label = f"{d['name']} ({n} of {len(todo)})" if len(todo) > 1 else d["name"]
            S.job.update(message=f"Tokenizing {label}. This is the slow part…", progress=None)

            def progress(split, done, total):
                S.job.update(message=f"{label}: writing {split}.bin", progress=done / total)

            tokenize_and_save(d["ds"], d["paths"].root, num_proc=os.cpu_count() or 4, progress=progress)
        S.job["message"] = "Tokenized."

    return start_job("tokenize", job)


# ---------- batches ----------

@app.post("/api/batch")
def batch_endpoint(req: BatchReq):
    require_idle("sample a batch")
    ds = require_tokenized()
    if not (8 <= req.block_size <= 1024 and 1 <= req.batch_size <= 512):
        raise HTTPException(400, "block_size must be 8–1024 and batch_size 1–512.")
    for d in ds:
        for f in (d["paths"].train_bin, d["paths"].val_bin):
            if token_count(f) <= req.block_size + 1:
                raise HTTPException(400, f"{d['name']}: {os.path.basename(f)} has only {token_count(f):,} tokens, fewer "
                                         f"than one {req.block_size}-token window. Use a smaller context window.")
    if S.model_config is not None and S.model_config.block_size != req.block_size:
        S.model, S.model_config = None, None  # model was built for another context size
    S.batch = req.model_dump()
    torch.manual_seed(req.seed)
    loader = selection_loader(req.block_size, req.batch_size, "cpu")
    X, Y = loader.get_batch("train")
    names = [d["name"] for d in ds]
    counts = {}
    for s in loader.last_sources:
        counts[names[s]] = counts.get(names[s], 0) + 1
    n = min(24, req.block_size)
    return {"shape": list(X.shape), "input": tokens_of(X[0, :n].tolist()), "target": tokens_of(Y[0, :n].tolist()),
            "row0_source": names[loader.last_sources[0]], "composition": counts}


# ---------- model ----------

@app.post("/api/model")
def model_endpoint(req: ModelReq):
    require_idle("build a model")
    if req.n_embd % req.n_head != 0:
        raise HTTPException(400, f"Embedding size ({req.n_embd}) must be divisible by the number of heads ({req.n_head}).")
    if not (1 <= req.n_layer <= 48 and 1 <= req.n_head <= 32 and 16 <= req.n_embd <= 2048 and 0 <= req.dropout < 1):
        raise HTTPException(400, "Model settings are out of range (layers 1–48, heads 1–32, embedding 16–2048).")
    config = GPTConfig(vocab_size=S.enc.n_vocab, block_size=S.batch["block_size"], n_layer=req.n_layer,
                       n_head=req.n_head, n_embd=req.n_embd, dropout=req.dropout, bias=True)
    breakdown = param_breakdown(config)
    need = breakdown["total"] * BYTES_PER_PARAM_TRAINING
    if S.memory and need > 0.6 * S.memory:
        raise HTTPException(400, f"Too big for this Mac: training {breakdown['total'] / 1e6:,.0f}M parameters needs about "
                                 f"{need / 1e9:.1f} GB for weights, gradients and optimizer state, but it has "
                                 f"{S.memory / 1e9:.0f} GB of memory. Use fewer layers or a smaller embedding size.")
    torch.manual_seed(S.batch["seed"])
    S.model, S.model_config, S.train = None, None, None
    model = GPT(config).to(S.dev.device)
    S.model, S.model_config = model, config

    result = {"params": sum(p.numel() for p in model.parameters()), "config": asdict(config), "breakdown": breakdown,
              "random_loss": math.log(S.enc.n_vocab), "initial_loss": None, "train_memory": need}
    model.eval()
    if S.datasets and all(bin_files_exist(d["paths"].root) for d in selected()):
        X, Y = selection_loader(config.block_size, S.batch["batch_size"], S.dev.device).get_batch("train")
        with torch.no_grad():
            _, loss = model(X, Y)
        result["initial_loss"] = loss.item()
    ids = S.enc.encode_ordinary("Once upon a time")
    out = model.generate(torch.tensor([ids], device=S.dev.device), 30)
    result["sample"] = S.enc.decode(out[0].tolist())
    model.train()
    return result


# ---------- inside the transformer ----------

@app.post("/api/inspect")
def inspect_endpoint(req: InspectReq):
    """Real attention weights (one layer, every head) and next-token probabilities for a short prompt."""
    if training_running():
        raise HTTPException(409, "Training is running. Try again when it finishes.")
    ids = S.enc.encode_ordinary(req.prompt)[:16]
    if not ids:
        raise HTTPException(400, "Type a prompt first.")
    model, source = inference_model(allow_untrained=True)
    was_training = model.training
    model.eval()
    try:
        layer = max(0, min(req.layer, model.config.n_layer - 1))
        att = attention_maps(model, ids, S.dev.device, layer=layer)
        top = next_token_probs(model, ids, S.dev.device, k=10)
    finally:
        if was_training:
            model.train()
    return {"tokens": tokens_of(ids), "source": source, "layer": layer, "n_layer": model.config.n_layer,
            "n_head": model.config.n_head, "attention": [[[round(v, 3) for v in row] for row in h] for h in att.tolist()],
            "next": [{"id": i, "text": S.enc.decode([i]), "prob": round(p, 4)} for i, p in top]}


# ---------- training ----------

@app.post("/api/train")
def train_endpoint(req: TrainReq):
    ds = require_tokenized()
    if S.model is None:
        raise HTTPException(400, "Build the model first (Model step).")
    if not (1 <= req.max_iters <= 1_000_000) or not (0 <= req.warmup_steps < req.max_iters):
        raise HTTPException(400, "Need max_iters ≥ 1 and warmup_steps below max_iters.")
    if req.eval_interval < 1 or req.eval_iters < 1 or req.gradient_accumulation_steps < 1:
        raise HTTPException(400, "eval_interval, eval_iters and gradient_accumulation_steps must be ≥ 1.")
    if not (0 < req.min_lr <= req.learning_rate):
        raise HTTPException(400, "min_lr must be positive and no higher than learning_rate.")

    paths = selection_paths()
    if checkpoint_info(paths) is not None:  # the run overwrites this checkpoint: keep a copy in the library first
        try:
            registry.add_model(paths.checkpoint, paths.model_config, source="trained")
        except Exception:
            traceback.print_exc()
    cfg = TrainConfig(batch_size=S.batch["batch_size"], seed=S.batch["seed"],
                      best_model_path=paths.checkpoint, **req.model_dump())
    return _run_training(ds, paths, S.model, S.model_config, cfg, resume=None)


def _run_training(ds, paths, model, config, cfg, resume=None):
    """Start the training job. resume: the saved state's summary (read_state_meta) to continue from, or None."""
    n_params = sum(p.numel() for p in model.parameters())
    tokens_per_step = cfg.batch_size * config.block_size
    names = [d["name"] for d in ds]

    def job():
        S.stop_requested = False
        os.makedirs(paths.root, exist_ok=True)
        if len(ds) > 1:
            with open(os.path.join(paths.root, "mix.json"), "w") as f:
                json.dump({"datasets": [{"name": d["name"], "config_name": d["config_name"], "path": d["paths"].root}
                                        for d in ds]}, f, indent=2)
        if not resume:
            for f in (paths.checkpoint, paths.model_config):  # replace any earlier model for these datasets
                if os.path.exists(f):
                    os.remove(f)
        S.inference_cache = None
        start = resume["step"] if resume else 0
        earlier = resume.get("val_losses") or [] if resume else []
        tr = S.train = {"max_iters": cfg.max_iters, "step": start, "lr": None, "points": [],
                        "evals": [{"step": (i + 1) * resume["eval_interval"], "train": round(t, 4), "val": round(v, 4)}
                                  for i, (t, v) in enumerate(zip(resume.get("train_losses") or [], earlier))] if resume else [],
                        "best_val": round(resume["best_val"], 4) if resume and resume.get("best_val") else None,
                        "resumed_from": start or None,
                        "started": time.time(), "it_per_sec": None, "config": asdict(cfg),
                        "params": n_params, "updates": 0, "total_updates": math.ceil(cfg.max_iters / cfg.gradient_accumulation_steps),
                        "tokens": 0, "tokens_per_step": tokens_per_step, "datasets": names}
        stride = max(1, cfg.max_iters // 1000)
        t0 = time.time()

        def callback(ev):
            if ev["type"] == "step":
                step = ev["step"]
                tr["step"], tr["lr"] = step, ev["lr"]
                tr["updates"] = step // cfg.gradient_accumulation_steps + (1 if step == cfg.max_iters and step % cfg.gradient_accumulation_steps else 0)
                tr["tokens"] = step * tokens_per_step
                if step == 1 or step % stride == 0:
                    tr["points"].append([step, round(ev["loss"], 4)])
                tr["it_per_sec"] = (step - start) / max(time.time() - t0, 1e-6)
                S.job.update(progress=step / cfg.max_iters, message=f"Step {step:,} of {cfg.max_iters:,}")
            else:
                tr["evals"].append({"step": ev["step"], "train": round(ev["train"], 4), "val": round(ev["val"], 4)})
                if tr["best_val"] is None or ev["val"] < tr["best_val"]:
                    tr["best_val"] = round(ev["val"], 4)
            return not S.stop_requested

        S.job["message"] = f"Resuming at step {start:,}…" if resume else "Training…"
        loader = selection_loader(config.block_size, cfg.batch_size, S.dev.device)
        pretrain(model, loader, cfg, S.dev, callback, resume=bool(resume), model_config=asdict(config))

        if not os.path.exists(paths.checkpoint):  # stopped (or finished) before the first evaluation
            torch.save(model.state_dict(), paths.checkpoint)
        seconds = round(time.time() - t0, 1)
        summary = {"datasets": names, "mix": S.mix_mode if len(ds) > 1 else None, "steps": tr["step"],
                   "updates": tr["updates"], "tokens": tr["tokens"], "params": n_params,
                   "param_updates": n_params * tr["updates"], "best_val": tr["best_val"], "seconds": seconds,
                   "trained_at": time.strftime("%Y-%m-%d %H:%M"), "stopped_early": S.stop_requested,
                   "device": S.dev.device, "chip": S.chip, **({"resumed_from": start} if resume else {})}
        if resume and not tr["evals"][len(earlier):]:  # no new evaluation: keep the earlier best for the summary
            summary["best_val"] = tr["best_val"]
        save_model_config(paths.model_config, config, {**summary, "train_config": asdict(cfg)})
        try:  # keep a copy in the model library, where later training can't overwrite it
            registry.add_model(paths.checkpoint, paths.model_config, source="trained")
        except Exception:
            traceback.print_exc()
        set_active(None)  # Generate switches to the model just trained
        with open(os.path.join(paths.root, "runs.jsonl"), "a") as f:
            f.write(json.dumps({**summary, "model": {k: getattr(config, k) for k in ("n_layer", "n_head", "n_embd", "block_size")},
                                "max_iters": cfg.max_iters}) + "\n")
        if S.stop_requested:
            S.job.update(status="stopped", message=f"Stopped at step {tr['step']:,}. Model saved.")
        else:
            S.job["message"] = "Training complete. Model saved."

    return start_job("train", job)


class ResumeReq(BaseModel):
    max_iters: Optional[int] = None   # total steps; more than the saved run's extends it


@app.post("/api/train/resume")
def train_resume(req: ResumeReq):
    """Continue the saved training run for the current datasets, from its last saved step."""
    ds = require_tokenized()
    paths = selection_paths()
    state = read_state_meta(paths.root)
    if state is None:
        raise HTTPException(400, "There's no saved training run to continue for this training data.")
    max_iters = req.max_iters or state["max_iters"]
    if max_iters <= state["step"]:
        raise HTTPException(400, f"The saved run is at step {state['step']:,}: ask for more steps than that.")
    if max_iters > 1_000_000:
        raise HTTPException(400, "That's too many steps.")
    if state.get("model_config"):
        config = GPTConfig(**state["model_config"])
    else:
        config, _ = load_model_config(paths.model_config)
    saved = {**state["train_config"], "betas": tuple(state["train_config"]["betas"])}
    cfg = TrainConfig(**{**saved, "best_model_path": paths.checkpoint, "max_iters": max_iters})
    if checkpoint_info(paths) is not None:  # keep the current best model in the library first
        try:
            registry.add_model(paths.checkpoint, paths.model_config, source="trained")
        except Exception:
            traceback.print_exc()
    model = GPT(config).to(S.dev.device)
    S.model, S.model_config = model, config
    return _run_training(ds, paths, model, config, cfg, resume=state)


@app.post("/api/train/stop")
def train_stop():
    if training_running():
        S.stop_requested = True
        S.job["message"] = "Stopping after this step…"
    return {"ok": True}


# ---------- generation ----------

@app.post("/api/generate/stream")
def generate_stream_endpoint(req: GenerateReq):
    """Stream generated tokens as newline-delimited JSON: a 'start' line, one 'token' line each, then 'done'."""
    require_model()
    if training_running():
        raise HTTPException(409, "Training is running. Wait for it to finish, or stop it.")
    model, _ = inference_model()
    prompt = req.prompt[:4000]
    ids = S.enc.encode_ordinary(prompt)
    if not ids:
        raise HTTPException(400, "Type a prompt first.")
    block = model.config.block_size
    max_new = max(1, min(req.max_new_tokens, 1000))
    temperature, top_k, extras = req.clean()

    def line(obj):
        return json.dumps(obj) + "\n"

    def gen():
        n_params = sum(p.numel() for p in model.parameters())
        yield line({"type": "start", "prompt_tokens": len(ids), "context_window": block,
                    "prompt_truncated": len(ids) > block, "params": n_params, "device": S.dev.device,
                    "use_kv_cache": req.use_kv_cache})
        out, times, probs, kept, stats = [], [], [], [], {}
        for ev in generate_stream(model, ids, S.dev.device, max_new, temperature, top_k,
                                  use_kv_cache=req.use_kv_cache, stats=stats, stop_token=S.enc.eot_token, sampling=extras):
            out.append(ev["id"])
            times.append(ev["ms"])
            probs.append(ev["prob"])
            kept.append(ev["kept"])
            yield line({"type": "token", "text": S.enc.decode([ev["id"]]), "id": ev["id"], "prob": round(ev["prob"], 4),
                        "ms": round(ev["ms"], 1), "kept": ev["kept"],
                        "alternatives": [{"text": S.enc.decode([i]), "prob": p} for i, p in ev["alternatives"]]})
        if not out:  # the model ended the story straight away
            yield line({"type": "done", "completion": "", "prompt_tokens": len(ids), "output_tokens": 0,
                        "total_tokens": len(ids), "context_window": block, "stopped": True, "token_ms": [],
                        "use_kv_cache": req.use_kv_cache, "memory": device_memory(S.dev.device)})
            return
        total_ms, first_ms = times[-1], times[0]
        decode_ms = total_ms - first_ms
        yield line({"type": "done", "completion": S.enc.decode(out), "prompt_tokens": len(ids),
                    "output_tokens": len(out), "total_tokens": len(ids) + len(out),
                    "context_used": min(len(ids) + len(out) - 1, block), "context_window": block,
                    "ttft_ms": round(first_ms, 1), "total_ms": round(total_ms, 1),
                    "tokens_per_sec": round((len(out) - 1) / (decode_ms / 1000), 1) if len(out) > 1 and decode_ms > 0 else None,
                    "ms_per_token": round(decode_ms / (len(out) - 1), 1) if len(out) > 1 else None,
                    "token_ms": [round(b - a, 1) for a, b in zip([0.0] + times[:-1], times)],
                    "avg_prob": round(sum(probs) / len(probs), 4),
                    "prefill_tokens_per_sec": round(min(len(ids), block) / (first_ms / 1000), 1) if first_ms > 0 else None,
                    "avg_kept": round(sum(kept) / len(kept), 1), **repetition_stats(out),
                    "use_kv_cache": req.use_kv_cache, "kv_cache_bytes": stats.get("kv_cache_bytes", 0),
                    "kv_bytes_per_token": kv_bytes_per_token(model.config), "recomputes": stats.get("recomputes", 0),
                    "weights_bytes": n_params * 4, "memory": device_memory(S.dev.device),
                    "stopped": stats.get("stopped", False)})

    return StreamingResponse(gen(), media_type="application/x-ndjson")


def repetition_stats(ids):
    """How repetitive a generation is: share of distinct word pairs, and share of 4-token runs seen before."""
    pairs = list(zip(ids, ids[1:]))
    grams = [tuple(ids[i:i + 4]) for i in range(len(ids) - 3)]
    seen, repeats = set(), 0
    for g in grams:
        repeats += g in seen
        seen.add(g)
    return {"distinct_2": round(len(set(pairs)) / len(pairs), 3) if pairs else None,
            "repeated_4grams": round(repeats / len(grams), 3) if grams else None}


@app.post("/api/generate/distribution")
def generate_distribution(req: DistributionReq):
    """The model's next-token distribution for a prompt, step by step through penalties and the sampling filters."""
    require_model()
    if training_running():
        raise HTTPException(409, "Training is running. Wait for it to finish, or stop it.")
    model, _ = inference_model()
    ids = S.enc.encode_ordinary(req.prompt[:4000])[-model.config.block_size:]
    if not ids:
        raise HTTPException(400, "Type a prompt first.")
    temperature, top_k, extras = req.clean()
    with torch.no_grad():
        x = torch.tensor(ids, dtype=torch.long, device=S.dev.device).unsqueeze(0)
        logits, _ = model(x)
        st = sampling_distribution(logits[0, -1], x[0], temperature, top_k, stages=True, **extras)
    n = max(5, min(req.n, 60))
    order = torch.topk(st["raw"], n).indices.tolist()
    order += [i for i in torch.topk(st["final"], n).indices.tolist() if i not in order and st["final"][i] > 0][:10]
    ent = lambda p: float(-(p[p > 0] * p[p > 0].log()).sum())  # noqa: E731
    rows = []
    for i in order:
        cut = ("greedy" if st["greedy"] else "top-k") if st["cut_k"][i] else "top-p" if st["cut_p"][i] else "min-p" if st["cut_min"][i] else None
        rows.append({"id": i, "text": S.enc.decode([i]), "raw": round(float(st["raw"][i]), 5),
                     "penalized": round(float(st["penalized"][i]), 5), "scaled": round(float(st["scaled"][i]), 5),
                     "final": round(float(st["final"][i]), 5), "cut": cut, "seen": int(st["counts"][i])})
    return {"rows": rows, "n_kept": st["n_kept"], "vocab": int(st["raw"].numel()), "kept_mass": round(st["kept_mass"], 4),
            "entropy_raw": round(ent(st["raw"]), 3), "entropy_final": round(ent(st["final"]), 3),
            "prompt_tokens": len(ids), "greedy": st["greedy"]}


@app.post("/api/generate/compare")
def generate_compare(req: CompareReq):
    """Continue the same prompt with up to four sampling settings (same seed) and compare the results."""
    require_model()
    if training_running():
        raise HTTPException(409, "Training is running. Wait for it to finish, or stop it.")
    model, _ = inference_model()
    ids = S.enc.encode_ordinary(req.prompt[:2000])
    if not ids:
        raise HTTPException(400, "Type a prompt first.")
    n = max(10, min(req.max_new_tokens, 400))
    results = []
    for s in req.settings[:4]:
        temperature, top_k, extras = s.clean()
        torch.manual_seed(req.seed)
        out, probs, kept, stats, t0 = [], [], [], {}, time.time()
        for ev in generate_stream(model, ids, S.dev.device, n, temperature, top_k, stats=stats,
                                  stop_token=S.enc.eot_token, sampling=extras):
            out.append(ev["id"])
            probs.append(ev["prob"])
            kept.append(ev["kept"])
        results.append({"text": S.enc.decode(out), "tokens": len(out), "stopped": stats.get("stopped", False),
                        "avg_prob": round(sum(probs) / len(probs), 4) if probs else None,
                        "avg_kept": round(sum(kept) / len(kept), 1) if kept else None,
                        "seconds": round(time.time() - t0, 2), **repetition_stats(out)})
    return {"prompt": req.prompt, "results": results}


@app.post("/api/generate/benchmark")
def generate_benchmark(req: BenchmarkReq):
    """Generate the same tokens with and without the KV cache and compare speed and memory."""
    require_model()
    if training_running():
        raise HTTPException(409, "Training is running. Wait for it to finish, or stop it.")
    model, _ = inference_model()
    ids = S.enc.encode_ordinary(req.prompt[:2000]) or [S.enc.eot_token]
    ids = ids[-model.config.block_size:]
    room = model.config.block_size - len(ids)
    n = {"fill": room, "past": room + 80}.get(req.length, req.n_tokens)
    n = max(10, min(n, 500))
    r = benchmark_generation(model, ids, S.dev.device, n_tokens=n)
    return {**r, "n_tokens": n, "prompt_tokens": len(ids), "context_window": model.config.block_size,
            "kv_bytes_per_token": kv_bytes_per_token(model.config), "device": S.dev.device, "chip": S.chip,
            "n_layer": model.config.n_layer, "n_embd": model.config.n_embd,
            "params": sum(p.numel() for p in model.parameters())}


# ---------- model library and evaluation ----------

class ModelIdReq(BaseModel):
    id: Optional[str] = None


class RenameReq(BaseModel):
    id: str
    name: str


class EvalReq(BaseModel):
    ids: List[str]
    datasets: Optional[List[str]] = None   # dataset keys (folder names); default: every one available


def _model_row(m):
    return {k: m.get(k) for k in ("id", "name", "source", "origin", "added_at", "model_type", "model", "params",
                                  "datasets", "steps", "tokens", "best_val", "seconds", "trained_at", "chip",
                                  "platform", "bytes")} | {"eval": m.get("eval"), "active": m["id"] == S.active_model}


@app.get("/api/models")
def models_list():
    return {"models": [_model_row(m) for m in registry.list_models()], "active": S.active_model,
            "eval_datasets": [{k: d[k] for k in ("key", "title", "tokens")} for d in evaluate.eval_datasets()],
            "suite": {"version": evaluate.SUITE_VERSION, "windows": evaluate.EVAL_WINDOWS, "context": evaluate.EVAL_CONTEXT,
                      "prompts": evaluate.PROMPTS, "tokens": evaluate.GEN_TOKENS, "balanced": evaluate.BALANCED}}


@app.post("/api/models/scan")
def models_scan():
    require_idle("look for models")
    found = registry.scan(DATA_ROOT)
    return {"added": [m["name"] for m, added in found if added], "found": len(found)}


@app.post("/api/models/save-current")
def models_save_current():
    paths = selection_paths()
    if checkpoint_info(paths) is None:
        raise HTTPException(400, "There's no trained model for the current datasets.")
    meta, added = registry.add_model(paths.checkpoint, paths.model_config, source="saved")
    return {"id": meta["id"], "name": meta["name"], "added": added}


@app.post("/api/models/use")
def models_use(req: ModelIdReq):
    require_idle("switch models")
    if req.id:
        try:
            registry.get_model(req.id)
        except (KeyError, ValueError) as e:
            raise HTTPException(404, str(e))
    set_active(req.id or None)
    return {"active": S.active_model}


@app.post("/api/models/rename")
def models_rename(req: RenameReq):
    try:
        return {"name": registry.rename_model(req.id, req.name)["name"]}
    except (KeyError, ValueError) as e:
        raise HTTPException(404, str(e))


@app.post("/api/models/delete")
def models_delete(req: ModelIdReq):
    require_idle("delete a model")
    try:
        registry.delete_model(req.id)
    except (KeyError, ValueError) as e:
        raise HTTPException(404, str(e))
    if S.active_model == req.id:
        set_active(None)
    if S.inference_cache and f"{os.sep}{req.id}{os.sep}" in S.inference_cache[0]:
        S.inference_cache = None
    return {"ok": True}


@app.post("/api/eval/run")
def eval_run(req: EvalReq):
    ids = [i for i in req.ids if i][:20]
    if not ids:
        raise HTTPException(400, "Pick at least one model to evaluate.")
    datasets = [d for d in evaluate.eval_datasets() if req.datasets is None or d["key"] in req.datasets]

    def job():
        S.inference_cache = None  # make room: models are loaded one at a time
        for n, mid in enumerate(ids):
            name = registry.get_model(mid)["name"]

            def say(msg, n=n, name=name):
                S.job.update(message=f"{name} ({n + 1} of {len(ids)}): {msg}…",
                             progress=n / len(ids))
            evaluate.evaluate_model(mid, S.dev.device, datasets=datasets, progress=say)
        S.job["message"] = f"Evaluated {len(ids)} model{'s' if len(ids) > 1 else ''}."

    return start_job("evaluate", job)


# ---------- static front end ----------

@app.middleware("http")
async def no_stale_frontend(request, call_next):
    """Make browsers recheck the page and its files, so a `git pull` shows up on the next reload."""
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Web front end for training the model step by step.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true", help="Don't open the browser automatically")
    args = ap.parse_args()
    url = f"http://{args.host}:{args.port}"
    print(f"\n  TinyStories LM Studio → {url}\n  Device: {S.dev.device} ({S.dev.dtype}) on {S.chip}\n")
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")

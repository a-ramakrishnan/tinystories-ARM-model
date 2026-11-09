# Dataset download and batch sampling
# Default: Tiny Stories Data set -> https://huggingface.co/datasets/roneneldan/TinyStories
# A dataset can be given as:
#   - a Hugging Face dataset name:   roneneldan/TinyStories
#   - a Hugging Face dataset link:   https://huggingface.co/datasets/roneneldan/TinyStories
#   - a link to a text data file:    https://example.com/stories.txt  (.txt, .jsonl, .json, .csv, .parquet)
# Only text datasets are accepted: the chosen column must hold strings, and image/audio/video datasets are rejected.

import hashlib
import os
import re
import urllib.request
from dataclasses import dataclass
from urllib.parse import urlparse

import numpy as np
import torch
from datasets import (Dataset, DatasetDict, Value, get_dataset_config_names, load_dataset, load_dataset_builder,
                      load_from_disk)

DEFAULT_DATASET = "roneneldan/TinyStories"
VALIDATION_SPLIT_NAMES = ("validation", "valid", "val", "test")

HF_DATASET_URL = re.compile(r"^https?://(?:www\.)?huggingface\.co/datasets/([^/?#]+/[^/?#]+)/?(?:$|[?#]|/(?!resolve/))", re.I)
GITHUB_BLOB_URL = re.compile(r"^https?://github\.com/([^/]+)/([^/]+)/blob/(.+)$", re.I)
FILE_BUILDERS = {".txt": "text", ".text": "text", ".md": "text", ".json": "json", ".jsonl": "json",
                 ".csv": "csv", ".parquet": "parquet"}
COMPRESSION_EXTS = (".gz", ".bz2", ".xz", ".zst")
NON_TEXT_FEATURES = ("Image", "Audio", "Video", "Pdf")


class SubsetNeeded(ValueError):
    """The dataset has several subsets (configs) and one must be chosen. `configs` lists them."""

    def __init__(self, message, configs):
        super().__init__(message)
        self.configs = configs


def resolve_source(name):
    """Work out what the user gave us.

    Returns {"kind": "hub", "name": "owner/dataset"} for Hugging Face datasets (name or page link), or
    {"kind": "url", "name": <file url>, "builder": "text" | "json" | "csv" | "parquet"} for a data file link.
    """
    s = (name or "").strip()
    s = re.sub(r"^(https?://(?:www\.)?huggingface\.co/datasets/[^/]+/[^/]+)/blob/", r"\1/resolve/", s, flags=re.I)
    m = HF_DATASET_URL.match(s)
    if m:
        return {"kind": "hub", "name": m.group(1)}
    if re.match(r"^https?://", s, re.I):
        gh = GITHUB_BLOB_URL.match(s)
        if gh:  # GitHub page link -> raw file link
            s = f"https://raw.githubusercontent.com/{gh.group(1)}/{gh.group(2)}/{gh.group(3)}"
        path = urlparse(s).path.lower()
        for ext in COMPRESSION_EXTS:
            if path.endswith(ext):
                path = path[: -len(ext)]
        ext = os.path.splitext(path)[1]
        if ext not in FILE_BUILDERS:
            raise ValueError(f"Unsupported link ({ext or 'no file extension'}). Use a Hugging Face dataset page, "
                             "or a direct link to a .txt, .jsonl, .json, .csv or .parquet file.")
        return {"kind": "url", "name": s, "builder": FILE_BUILDERS[ext]}
    if not re.match(r"^[\w.-]+(/[\w.-]+)?$", s):
        raise ValueError(f"'{s}' is not a Hugging Face dataset name (owner/name) or a link.")
    return {"kind": "hub", "name": s}


def dataset_slug(name, config_name=None):
    """Folder-safe name for a dataset, e.g. 'roneneldan/TinyStories' -> 'roneneldan__tinystories'."""
    src = resolve_source(name)
    if src["kind"] == "url":
        u = urlparse(src["name"])
        base = os.path.basename(u.path) or "data"
        full = f"url-{u.netloc}-{base}-{hashlib.sha1(src['name'].encode()).hexdigest()[:8]}"
    else:
        full = f"{src['name']}-{config_name}" if config_name else src["name"]
    return re.sub(r"[^A-Za-z0-9._-]+", "__", full).lower()


@dataclass
class DatasetPaths:
    """Where everything for one dataset lives: data/<slug>/..."""
    root: str

    @property
    def hf_dir(self):          # local copy of the downloaded dataset
        return os.path.join(self.root, "dataset")

    @property
    def train_bin(self):
        return os.path.join(self.root, "train.bin")

    @property
    def val_bin(self):
        return os.path.join(self.root, "validation.bin")

    @property
    def checkpoint(self):      # best weights from training on this dataset
        return os.path.join(self.root, "best_model_params.pt")

    @property
    def model_config(self):    # model config + training summary for the checkpoint
        return os.path.join(self.root, "model_config.json")


def dataset_paths(name=DEFAULT_DATASET, config_name=None, data_root="data"):
    return DatasetPaths(os.path.join(data_root, dataset_slug(name, config_name)))


def _normalize_splits(raw, text_column, val_fraction, seed):
    """Return a DatasetDict with exactly 'train' and 'validation' splits and a single 'text' column."""
    if isinstance(raw, Dataset):
        raw = DatasetDict(train=raw)
    if "train" not in raw:
        raise ValueError(f"Dataset has no 'train' split (found: {', '.join(raw)})")

    train = raw["train"]
    check_features(train.features, text_column)
    val_name = next((s for s in VALIDATION_SPLIT_NAMES if s in raw), None)
    if val_name:
        val = raw[val_name]
    else:
        # No validation split: hold out a small slice of train (at least a few rows for small files)
        if len(train) < 20:
            raise ValueError(f"Only {len(train)} documents: too small to train on.")
        n_val = max(int(len(train) * val_fraction), min(50, len(train) // 10))
        parts = train.train_test_split(test_size=n_val, seed=seed)
        train, val = parts["train"], parts["test"]

    def keep_text(d):
        d = d.select_columns([text_column])
        return d.rename_column(text_column, "text") if text_column != "text" else d

    return DatasetDict(train=keep_text(train), validation=keep_text(val))


def _is_text_feature(f):
    return isinstance(f, Value) and f.dtype in ("string", "large_string")


def check_features(features, text_column):
    """Reject datasets that aren't text-only, or whose chosen column doesn't hold text."""
    if features is None:
        return
    non_text = [k for k, f in features.items() if type(f).__name__ in NON_TEXT_FEATURES]
    if non_text:
        cols = ", ".join(f"'{c}'" for c in non_text)
        raise ValueError(f"Not a text dataset: the {cols} column{'s' if len(non_text) > 1 else ''} hold"
                         f"{'' if len(non_text) > 1 else 's'} images, audio or video. Pick a text-only dataset.")
    text_cols = [k for k, f in features.items() if _is_text_feature(f)]
    if text_column not in features:
        hint = f" Text columns: {', '.join(text_cols)}." if text_cols else " It has no text columns."
        raise ValueError(f"Column '{text_column}' not found.{hint}")
    if not _is_text_feature(features[text_column]):
        raise ValueError(f"Column '{text_column}' holds {features[text_column]}, not text."
                         + (f" Text columns: {', '.join(text_cols)}." if text_cols else ""))


def check_text_content(ds, n=500):
    """Make sure the training split actually contains readable text."""
    sample = ds["train"].select(range(min(n, len(ds["train"]))))["text"]
    nonempty = [t for t in sample if t and t.strip()]
    if not nonempty:
        raise ValueError("The text column is empty in the first rows: is this the right column?")
    joined = "".join(nonempty)[:200_000]
    readable = sum(ch.isprintable() or ch in "\n\r\t" for ch in joined) / len(joined)
    if readable < 0.95:
        raise ValueError("This doesn't look like readable text (too many control or binary characters).")


def _check_url(url):
    """Make sure a file link is reachable and isn't a web page. Returns its size in bytes (or None)."""
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "tinystories-lm-studio"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            ctype = r.headers.get("Content-Type", "")
            size = r.headers.get("Content-Length")
    except Exception as e:
        raise ValueError(f"Couldn't reach that link: {e}")
    if "text/html" in ctype:
        raise ValueError("That link opens a web page, not a data file. Use the link to the raw file.")
    return int(size) if size and size.isdigit() else None


def _builder(src, config_name):
    if src["kind"] == "url":
        kwargs = {"sample_by": "paragraph"} if src["builder"] == "text" else {}
        return load_dataset_builder(src["builder"], data_files=src["name"], **kwargs)
    return load_dataset_builder(src["name"], config_name)


def inspect_source(name, config_name=None, text_column="text"):
    """Check a dataset before downloading it. Raises ValueError with a readable message if it won't work.

    Returns what was found: kind, resolved name, columns and their types, size if known.
    """
    src = resolve_source(name)
    size = _check_url(src["name"]) if src["kind"] == "url" else None
    try:
        info = _builder(src, config_name).info
    except Exception as e:  # dataset not found, gated, needs a subset, ...
        msg = str(e)
        if "Config name is missing" in msg:
            configs = get_dataset_config_names(src["name"])
            raise SubsetNeeded(f"This dataset has several subsets. Pick one: {', '.join(configs)}", configs)
        if "BuilderConfig" in msg and "not found" in msg:
            configs = get_dataset_config_names(src["name"])
            raise SubsetNeeded(f"Unknown subset '{config_name}'. Available: {', '.join(configs)}", configs)
        raise ValueError(msg.split("\n")[0])
    if src["kind"] == "url" and src["builder"] == "text":
        info_features = {"text": Value("string")}
    else:
        info_features = info.features
    if info_features is not None:
        check_features(info_features, text_column)
    return {
        "kind": src["kind"], "name": src["name"], "config_name": config_name,
        "columns": {k: str(v) for k, v in info_features.items()} if info_features is not None else None,
        "text_column": text_column,
        "download_size": size or info.download_size,
        "rows": {k: v.num_examples for k, v in (info.splits or {}).items()} or None,
    }


def get_dataset(local_dir, name=DEFAULT_DATASET, config_name=None, text_column="text", val_fraction=0.005, seed=42):
    """Return the dataset as {'train', 'validation'} splits with a 'text' column, downloading it only once.

    `name` is a Hugging Face dataset name, a Hugging Face dataset link, or a link to a text data file.
    The first run downloads it and saves a copy to `local_dir` (git-ignored). Later runs load that local
    copy without touching the network. `config_name` is the dataset's subset, if it has one
    (e.g. 'wikitext-2-raw-v1'). `text_column` is renamed to 'text'. If there is no validation split,
    `val_fraction` of train is held out. Plain .txt files are split into documents at blank lines.
    """
    if os.path.isdir(local_dir):
        print(f"Loading dataset from local copy: {local_dir}")
        return load_from_disk(local_dir)
    src = resolve_source(name)
    if src["kind"] == "url" and src["builder"] == "text":
        text_column = "text"
    inspect_source(name, config_name, text_column)  # validate before downloading everything
    print(f"Downloading {src['name']}{f' ({config_name})' if config_name else ''} and saving it to {local_dir}")
    if src["kind"] == "url":
        kwargs = {"sample_by": "paragraph"} if src["builder"] == "text" else {}
        raw = load_dataset(src["builder"], data_files=src["name"], **kwargs)
    else:
        raw = load_dataset(src["name"], config_name)
    ds = _normalize_splits(raw, text_column, val_fraction, seed)
    check_text_content(ds)
    ds.save_to_disk(local_dir)
    return ds


def parse_dataset_spec(spec):
    """Parse 'name[@subset][#column]' into (name, subset, column), e.g.
    'Salesforce/wikitext@wikitext-2-raw-v1' or 'euclaise/writingprompts#story'. Links may end in '#column' too."""
    column = None
    if "#" in spec:
        spec, column = spec.rsplit("#", 1)
    config = None
    if "@" in spec and not re.match(r"^https?://", spec, re.I):
        spec, config = spec.rsplit("@", 1)
    return spec.strip(), (config or None), (column or "text")


def mix_paths(dataset_roots, data_root="data"):
    """Where the trained model for a set of datasets lives.

    One dataset: its own folder. Several: data/mix-<names>-<hash>/ (holds only the checkpoint and its config;
    the tokens stay in each dataset's folder).
    """
    roots = sorted(dict.fromkeys(dataset_roots))
    if len(roots) == 1:
        return DatasetPaths(roots[0])
    names = "+".join(os.path.basename(r)[:18] for r in roots)
    digest = hashlib.sha1("|".join(os.path.basename(r) for r in roots).encode()).hexdigest()[:8]
    return DatasetPaths(os.path.join(data_root, re.sub(r"[^A-Za-z0-9._+-]+", "_", f"mix-{names}-{digest}")))


def token_count(path):
    return os.path.getsize(path) // 2 if os.path.exists(path) else 0


class BatchLoader:
    """Samples random (input, target) blocks from the tokenized train/validation .bin files.

    Some functions from https://github.com/karpathy/nanoGPT/blob/master/train.py with slight modifications.
    block_size = context window.

    data_dir can be one dataset folder or a list of them. With several, each window in a batch is drawn from
    one dataset, picked at random with probability `weights` (default: in proportion to each dataset's size;
    mix="equal" gives every dataset the same share).
    """

    def __init__(self, data_dir, block_size, batch_size, device, mix="size"):
        dirs = [data_dir] if isinstance(data_dir, str) else list(data_dir)
        self.sources = [{"train": os.path.join(d, "train.bin"), "val": os.path.join(d, "validation.bin")} for d in dirs]
        self.paths = self.sources[0]
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self.device_type = "cuda" if "cuda" in str(device) else str(device)
        self.weights = {}
        for split in ("train", "val"):
            sizes = torch.tensor([float(token_count(s[split])) for s in self.sources])
            w = torch.ones(len(dirs)) if mix == "equal" else sizes
            self.weights[split] = w / w.sum()
        self.last_sources = None  # which dataset each row of the last batch came from

    def _windows(self, split):
        """Pick (source index, start offset) for every row of a batch."""
        src = torch.multinomial(self.weights[split], self.batch_size, replacement=True) if len(self.sources) > 1 \
            else torch.zeros(self.batch_size, dtype=torch.long)
        data = {}
        rows = []
        for s in src.tolist():
            if s not in data:
                # We recreate np.memmap every batch to avoid a memory leak, as per
                # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
                data[s] = np.memmap(self.sources[s][split], dtype=np.uint16, mode="r")
            rows.append((s, int(torch.randint(len(data[s]) - self.block_size, (1,)))))
        return data, rows

    def get_batch(self, split):
        data, rows = self._windows(split)
        self.last_sources = [s for s, _ in rows]
        x = torch.stack([torch.from_numpy((data[s][i:i + self.block_size]).astype(np.int64)) for s, i in rows])
        y = torch.stack([torch.from_numpy((data[s][i + 1:i + 1 + self.block_size]).astype(np.int64)) for s, i in rows])
        if self.device_type == "cuda":
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y

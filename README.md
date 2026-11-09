# TinyStories ARM: train a language model on your Apple Silicon Mac

![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-M1%20to%20M5-black?logo=apple)
![PyTorch MPS](https://img.shields.io/badge/PyTorch-Metal%20(MPS)-ee4c2c?logo=pytorch&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-blue)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/a-ramakrishnan/tinystories-ARM-model/blob/main/train_step_by_step.ipynb)

**Built for Apple Silicon.** This project builds a small GPT-style language model (~30M parameters) from
scratch in PyTorch and trains it on your Mac's own GPU. No cloud, no Nvidia card. By default it learns from
[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories), about 2.1 million simple short
stories, and then writes stories of its own. You can switch to any Hugging Face text dataset.

There are three ways to use it:

| | What it is | Best for |
|---|---|---|
| **Web app** · `python3 app/server.py` | A local, step-by-step studio in your browser with live training charts | Learning how it works, experimenting |
| **Command line** · `python3 full-model.py` | Runs the whole pipeline in one go | Long training runs |
| **Colab notebook** · badge above | The same steps as a notebook on Colab's free GPU | Trying it without a Mac |

## Made for Apple Silicon

- **Trains on the Mac GPU.** The device is picked automatically, and on M-series Macs that's the GPU through
  PyTorch's Metal backend (MPS). The app and the script both show it at startup:
  `Training on device: mps (float32)`.
- **No extra setup.** The standard `pip install torch` already includes Metal support on macOS.
- **Full precision on MPS.** Training uses float32 on the Mac. Mixed precision is only switched on for CUDA,
  where it's well supported.
- **Unified memory.** The default model and batch size train comfortably on a 16 GB Mac.
- **Measured speed.** On an M5 with 16 GB, the default model trains at about **2 steps per second**
  (32 × 128 tokens per step). The *Quick* preset (2,000 steps) takes roughly 20 minutes. The full run
  (20,000 steps) takes several hours.

It also runs on an Nvidia GPU (CUDA, with mixed precision) or the CPU, but the Mac is what it's built and
tested on.

Check that PyTorch can see your Mac's GPU:

```bash
python3 -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

## Quick start (Mac)

Requires an Apple Silicon Mac (M1 or later) and Python 3.9+.

```bash
git clone https://github.com/a-ramakrishnan/tinystories-ARM-model.git
cd tinystories-ARM-model
pip install -r requirements.txt
```

### Web app (recommended)

```bash
python3 app/server.py
```

It opens http://127.0.0.1:8000 in your browser. The sidebar walks through six steps:

1. **Dataset**: pick a preset (TinyStories, WikiText-2) or type any Hugging Face dataset name, then browse
   its documents.
2. **Tokenize**: type text and see it split into tokens live, then tokenize the whole dataset with a
   progress bar.
3. **Batches**: sample a batch and see how each target is the input shifted by one token.
4. **Model**: choose layers, heads and embedding size; see the parameter count, the architecture and the
   untrained model's gibberish.
5. **Train**: pick *Quick*, *Full* or custom settings and watch loss, speed, time left and learning rate
   update live. You can stop at any time and keep what's been learned.
6. **Generate**: write a prompt and tune temperature, top-k and length.

Options: `--port 8080` to use another port, `--no-browser` to skip opening a browser tab.

### Command line

```bash
python3 full-model.py
```

Training takes hours, so keep the Mac awake and save a log:

```bash
caffeinate -i python3 full-model.py 2>&1 | tee train.log
```

The loss plot window **pauses the script** until you close it; generation runs after that.

## Choosing a dataset

TinyStories is the default. Any Hugging Face dataset with a `train` split and a text column works.

From the command line:

```bash
python3 full-model.py --dataset Salesforce/wikitext --dataset-config wikitext-2-raw-v1
```

| Option | Meaning | Default |
|---|---|---|
| `--dataset` | Hugging Face dataset name | `roneneldan/TinyStories` |
| `--dataset-config` | Subset name, for datasets that have them | none |
| `--text-column` | Column holding the text | `text` |

In the web app, enter the same three values in step 1. In the notebook, they're the form fields in Step 1.

How datasets are handled:

- **Downloaded once.** Each dataset is saved to its own folder under `data/` and loaded from there afterwards,
  without touching the network.
- **Splits.** A `validation`, `valid`, `val` or `test` split is used for validation. If there isn't one,
  0.5% of `train` is held out.
- **Checked first.** The text column is checked before anything is downloaded.
- **Kept separate.** Every dataset keeps its own tokens and trained model, so switching back and forth
  doesn't redo any work:

```
data/
└── roneneldan__tinystories/
    ├── dataset/               local copy of the downloaded dataset (~1.8 GB for TinyStories)
    ├── train.bin              token IDs for training (~900 MB for TinyStories)
    ├── validation.bin         token IDs for validation
    ├── best_model_params.pt   best weights from training
    └── model_config.json      model settings and training summary for those weights
```

`data/` is git-ignored, so none of this is committed.

## What happens on each run

| Step | Module | First run | Later runs |
|---|---|---|---|
| 1. Get the dataset | `data.py` | Downloads it and saves a local copy | Loads the local copy |
| 2. Tokenize | `tokenization.py` | Converts every document to GPT-2 token IDs | Skipped |
| 3. Batches | `data.py` | Samples random 128-token windows | Same (fresh batches every run) |
| 4. Train | `pretrain.py` | Trains and saves the best weights | Same |
| 5. Plot | `plots.py` | Training and validation loss | Same |
| 6. Generate | `inference.py` | Writes text from the prompts | Same |

## Project structure

```
full-model.py              Command-line entry point: model config, --dataset options, runs every step
app/
  server.py                Web app backend (FastAPI): runs the steps and reports live progress
  static/                  Web app front end (HTML, CSS, JavaScript; no build step)
data.py                    Dataset download/caching per dataset, validation split, BatchLoader
tokenization.py            GPT-2 tokenizer and writing train.bin / validation.bin
model.py                   GPT architecture (attention, MLP, blocks) and estimate_loss()
train_config.py            TrainConfig hyperparameters and setup_device() (MPS / CUDA / CPU)
pretrain.py                Training loop: AdamW, warmup + cosine LR schedule, gradient accumulation
plots.py                   Loss curves
inference.py               Load a checkpoint and generate text
train_step_by_step.ipynb   Notebook version for Google Colab
```

## Configuration

### Model size: `full-model.py`

```python
config = GPTConfig(
    vocab_size=50257,   # GPT-2 tokenizer
    block_size=128,     # context window (tokens)
    n_layer=6,          # transformer blocks
    n_head=6,           # attention heads per block
    n_embd=384,         # embedding size
    dropout=0.1,
    bias=True
)
```

### Training: `train_config.py`

| Setting | Default | Meaning |
|---|---|---|
| `max_iters` | 20000 | Training steps |
| `learning_rate` | 1e-4 | Peak learning rate |
| `min_lr` | 1e-5 | Learning rate at the end of the cosine decay |
| `warmup_steps` | 1000 | Steps of linear warmup |
| `batch_size` | 32 | Sequences per batch |
| `gradient_accumulation_steps` | 32 | Batches averaged per optimizer update |
| `eval_interval` | 500 | Evaluate every N steps |
| `eval_iters` | 500 | Batches per evaluation |
| `weight_decay` | 0.1 | AdamW weight decay |
| `grad_clip` | 0.5 | Gradient norm clipping |

The web app's *Quick* preset uses 2,000 steps, 200 warmup, evaluation every 250 steps over 50 batches, and
gradient accumulation of 8. *Full* uses the defaults above.

## Generating text from a trained model

The web app's Generate step does this for you. From Python:

```python
from data import dataset_paths
from inference import generate_text, load_model, load_model_config
from tokenization import get_tokenizer
from train_config import setup_device

paths = dataset_paths("roneneldan/TinyStories")          # same name (and subset) you trained on
config, info = load_model_config(paths.model_config)
dev = setup_device()
model = load_model(config, paths.checkpoint, dev.device)

print(generate_text(model, get_tokenizer(), "Once upon a time there was a dragon who",
                    dev.device, max_new_tokens=200, temperature=0.8, top_k=50))
```

- `temperature`: lower is more predictable, higher is more varied.
- `top_k`: only sample from the `k` most likely next tokens (`None` for no limit).

## No Mac? Use Colab

Click the **Open in Colab** badge, choose *Runtime → Change runtime type → T4 GPU*, and run the cells from
top to bottom. The notebook clones this repository, installs the dependencies, and walks through the same
steps as the web app, with form fields for the settings and the dataset.

## Credits

- Data loading and the model are adapted from Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT).
- Default dataset: [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) by Ronen Eldan and
  Yuanzhi Li.

## License

[MIT](LICENSE)

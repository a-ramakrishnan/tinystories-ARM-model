# TinyStories Small Language Model

A small GPT-style language model (~30M parameters), built from scratch in PyTorch and trained on
[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories), a dataset of about 2.1 million
simple short stories. After training, it writes short children's stories from a prompt.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/a-ramakrishnan/tinystories-ARM-model/blob/main/train_step_by_step.ipynb)

There are two ways to use it:

- **Step by step in a notebook**: [`train_step_by_step.ipynb`](train_step_by_step.ipynb) walks through every
  stage with explanations, editable settings and outputs to look at. Runs on Google Colab or locally.
- **All at once from the command line**: `python3 full-model.py` runs the whole pipeline.

## Quick start

### Option 1: Google Colab (no setup)

1. Click the **Open in Colab** badge above.
2. Switch to a GPU: *Runtime → Change runtime type → T4 GPU*.
3. Run the cells from top to bottom.

The first cell clones this repository and installs the dependencies.

### Option 2: Run locally

Requires Python 3.9+.

```bash
git clone https://github.com/a-ramakrishnan/tinystories-ARM-model.git
cd tinystories-ARM-model
pip install -r requirements.txt
```

Run the full pipeline:

```bash
python3 full-model.py
```

Or open the notebook (needs Jupyter: `pip install notebook`):

```bash
jupyter notebook train_step_by_step.ipynb
```

Training takes hours, so on a Mac it helps to stop the machine sleeping and keep a log:

```bash
caffeinate -i python3 full-model.py 2>&1 | tee train.log
```

## What happens when you run it

| Step | Module | First run | Later runs |
|---|---|---|---|
| 1. Get the dataset | `data.py` | Downloads TinyStories and saves it to `data/tinystories/` (~1.8 GB) | Loads the local copy, no download |
| 2. Tokenize | `tokenization.py` | Converts every story to GPT-2 token IDs in `train.bin` / `validation.bin` (~910 MB) | Skipped |
| 3. Batches | `data.py` | Samples random 128-token windows from `train.bin` | Same (fresh batches every run) |
| 4. Train | `pretrain.py` | 20,000 steps; saves the best weights to `best_model_params.pt` | Same |
| 5. Plot | `plots.py` | Shows training and validation loss | Same |
| 6. Generate | `inference.py` | Writes stories from the prompts in `full-model.py` | Same |

The plot window **pauses the script** until you close it; generation runs after that.

The dataset folder, the `.bin` files and the checkpoint are all git-ignored.

## Hardware

The device is picked automatically:

| Device | Used when | Precision |
|---|---|---|
| `cuda` | An Nvidia GPU is available (e.g. Colab T4) | bfloat16 or float16 (mixed precision) |
| `mps` | An Apple Silicon Mac (M1 or later) | float32 |
| `cpu` | Neither | float32 (very slow) |

The script prints which one it uses at startup, e.g. `Training on device: mps (float32)`.

## Project structure

```
full-model.py          Entry point: model config, prompts, runs every step
data.py                get_dataset() downloads once and caches locally; BatchLoader samples batches
tokenization.py        GPT-2 tokenizer and writing train.bin / validation.bin
model.py               GPT architecture (attention, MLP, blocks) and estimate_loss()
train_config.py        TrainConfig hyperparameters and setup_device()
pretrain.py            Training loop: AdamW, warmup + cosine LR schedule, gradient accumulation
plots.py               Loss curves
inference.py           load_model() and generate_text()
train_step_by_step.ipynb   Step-by-step notebook (Colab or local)
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

To try different settings without editing files, call `main()` with your own `TrainConfig`. For example, a
short test run:

```python
import importlib.util
from train_config import TrainConfig

spec = importlib.util.spec_from_file_location("full_model", "full-model.py")
full_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(full_model)

full_model.main(TrainConfig(max_iters=1000, eval_interval=100, eval_iters=50))
```

(`full-model.py` has a hyphen in its name, so it can't be imported with a plain `import`.)

## Generating text from a trained model

Once `best_model_params.pt` exists you can generate stories without retraining:

```python
from model import GPTConfig
from inference import load_model, generate_text
from tokenization import get_tokenizer
from train_config import setup_device

config = GPTConfig(vocab_size=50257, block_size=128, n_layer=6, n_head=6, n_embd=384, dropout=0.1, bias=True)
dev = setup_device()
model = load_model(config, "best_model_params.pt", dev.device)

print(generate_text(model, get_tokenizer(), "Once upon a time there was a dragon who",
                    dev.device, max_new_tokens=200, temperature=0.8, top_k=50))
```

The config must match the one used for training.

- `temperature`: lower is more predictable, higher is more varied.
- `top_k`: only sample from the `k` most likely next tokens (`None` for no limit).

## Credits

- Data loading and the model are adapted from Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT).
- Dataset: [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) by Ronen Eldan and Yuanzhi Li.

## License

[MIT](LICENSE)

# TinyStories ARM: train a language model on your Apple Silicon Mac

![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-M1%20to%20M5-black?logo=apple)
![PyTorch MPS](https://img.shields.io/badge/PyTorch-Metal%20(MPS)-ee4c2c?logo=pytorch&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-blue)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/a-ramakrishnan/tinystories-ARM-model/blob/main/train_step_by_step.ipynb)

**Built for Apple Silicon.** Build a small GPT-style language model from scratch in PyTorch and train it on
your Mac's own GPU, step by step, in a local web app. See inside the transformer as it works: attention
weights, context vectors and next-token probabilities. No cloud, no Nvidia card.

![Training a language model on an Apple M5 in the TinyStories LM Studio web app](docs/screenshots/train.png)

> **New here?** [From a training script to a working language model](docs/journey.md) tells the story of this
> project: where it started, every change made to the model and the inference, why each one worked, and what
> it measured. It's the best place to learn from.

## About

**TinyStories ARM** is an open-source, beginner-friendly project for **training a small language model (SLM)
from scratch on a Mac with Apple Silicon** (M1, M2, M3, M4 and M5). It implements a **GPT-style decoder-only
transformer** in **PyTorch**, trains it on the Mac's GPU through **Metal Performance Shaders (MPS)**, and
explains every stage of the **LLM training pipeline** in a local web app:

1. **Data**: pick one or more text datasets from **Hugging Face** (or any text file link) and keep them locally.
2. **Tokenization**: convert text to token IDs with the **GPT-2 BPE tokenizer** (`tiktoken`), with examples.
3. **Batching**: cut the token stream into context windows for **next-token prediction**, with a worked example.
4. **Model**: a **transformer** with **multi-head causal self-attention**, feed-forward (MLP) blocks, LayerNorm
   and weight tying, from 16M to 124M parameters, with an architecture diagram and a line-by-line
   **parameter count**.
5. **Inside the transformer**: a 13-stage **transformer visualization** with **Q, K and V matrices**:
   **LayerNorm**, attention scores, the causal mask, softmax, **context vectors**, the output projection,
   **dropout**, **residual connections**, the **feed-forward network** (GELU), and the final **logits and
   softmax** that pick the next token. Each stage is labelled **encoder & decoder** or **decoder only**, and
   heatmaps show the trained model's real attention.
6. **Pre-training**: **AdamW**, warmup and **cosine learning-rate decay**, gradient accumulation and clipping,
   with live **loss curves**, optimizer updates, parameter updates and tokens processed.
7. **Inference**: streamed **text generation** with **temperature, top-k, top-p, min-p, greedy and
   repetition / frequency / presence penalties**, a next-token explorer, side-by-side comparison of sampling
   settings, and a **KV cache**, showing
   tokens used, **time to first token**, decode speed, KV cache size, GPU memory and the model's confidence in
   every token, plus a with/without **KV cache benchmark** with per-token timing charts, an interactive
   **KV cache diagram** and a **KV cache memory calculator**.
8. **Models & evaluation**: every trained or imported model kept in a **model library**, switchable for
   generation, and compared on a fixed **evaluation suite** (held-out loss and perplexity per dataset,
   repetition, variety, speed and size) with side-by-side samples.

The default dataset is [Mini FineWeb-Edu](https://huggingface.co/datasets/BaoMinhTran/mini-fineweb-edu)
(its `small` subset, ~500M tokens), educational web pages filtered from Common Crawl for teaching value: the
same kind of data modern small language models are pre-trained on. The project started with
[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories), 2.1 million short stories in simple
English, which is still one click away. The research behind it showed that even very small models can learn
to write coherent English when trained on simple text, which makes it ideal for **on-device AI training** on
a laptop. The code is adapted from Andrej Karpathy's **nanoGPT** and kept small and readable, so it
doubles as a hands-on **machine learning tutorial**: learn how **large language models (LLMs)** work by
building and training one yourself, locally and privately on your own **MacBook, Mac mini, iMac or
Mac Studio**.

**Keywords:** small language model · train LLM from scratch · GPT from scratch · transformer attention
visualization · PyTorch MPS · Apple Silicon machine learning · M1/M2/M3/M4/M5 GPU training · local LLM training
on Mac · TinyStories · nanoGPT · tokenization · Hugging Face datasets · deep learning on macOS

## Three ways to use it

| | What it is | Best for |
|---|---|---|
| **Web app** · `python3 app/server.py` | A local, step-by-step studio in your browser with explanations and live charts | Learning how it works, experimenting |
| **Command line** · `python3 full-model.py` | Runs the whole pipeline in one go | Long training runs |
| **Notebook on Colab or Kaggle** · badge above | The same steps on a free cloud GPU; brings the model home as a zip | Bigger or longer training runs, or no Mac |

## Made for Apple Silicon

- **Trains on the Mac GPU.** The device is picked automatically, and on M-series Macs that's the GPU through
  PyTorch's Metal backend (MPS). The app and the script both show it at startup:
  `Training on device: mps (float32)`.
- **No extra setup.** The standard `pip install torch` already includes Metal support on macOS.
- **Full precision on MPS.** Training uses float32 on the Mac. Mixed precision is only switched on for CUDA,
  where it's well supported.
- **Unified memory.** Even the Large preset (124M parameters, about 2 GB to train) fits comfortably on a
  16 GB Mac. The app checks the memory before building a model that wouldn't fit.
- **Measured speed.** On an M5 with 16 GB, the default model trains at about **2 steps per second**
  (32 × 128 tokens per step, measured with the original 128-token window). The *Quick* preset (2,000 steps) takes roughly 20 minutes and brings the
  validation loss from 10.8 down to about 4.6. The full run (20,000 steps) takes several hours. Generation
  with the KV cache runs at about 130–150 tokens per second for the default model.

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

It opens http://127.0.0.1:8000 in your browser. The sidebar walks through seven steps:

1. **Dataset**: add one or more datasets by name or link, or from the gallery of public datasets. **Check**
   validates a dataset before anything is downloaded. With several, choose whether batches mix them by size
   or equally.
2. **Tokenize**: see how BPE works through examples (common words, rare words, numbers, capitals, emoji),
   try your own text, then tokenize your training data.
3. **Batches**: a worked example shows how one window becomes many next-token exercises; then sample a real
   batch (and see which datasets it came from).
4. **Model**: pick a size preset (Tiny 16M, Small 30M, Medium 51M, Large 124M) or set layers, heads and
   embedding size yourself, with a live parameter estimate. Building shows the architecture diagram, the
   input and output token counts, and how every parameter adds up.
5. **Inside the transformer**: compare encoder-only, decoder-only and encoder–decoder models, then walk a
   prompt through a whole transformer block and out to the next token in 13 stages, and explore your
   model's real attention heatmaps by layer and head.
6. **Train**: see what one training step does, how loss relates to probability, and the learning-rate
   schedule; then train and watch loss, speed, optimizer updates, parameters updated and tokens processed.
   Every run is kept in a history table.
7. **Generate**: text streams in token by token, colored by the model's confidence, with tokens used, time
   to first token, decode speed, KV cache size, GPU memory and a per-token timing chart. Switch the KV cache
   on or off, or run a side-by-side comparison.

Options: `--port 8080` to use another port, `--no-browser` to skip opening a browser tab. A step can be
opened directly with a link such as http://127.0.0.1:8000/#attention.

### Command line

```bash
python3 full-model.py
```

Training takes hours, so keep the Mac awake and save a log:

```bash
caffeinate -i python3 full-model.py 2>&1 | tee train.log
```

The loss plot window **pauses the script** until you close it; generation runs after that.

If a run is interrupted, or you want to train an existing run for longer, continue it from its saved state:

```bash
python3 full-model.py --resume                     # finish an interrupted run
python3 full-model.py --resume --max-iters 30000   # extend a finished run to 30,000 steps in total
```

`--max-iters` also sets the number of steps for a new run. See [Resuming and extending a run](#resuming-and-extending-a-run).

## Screenshots

| Choose one or more datasets | Tokenization, explained with examples |
|---|---|
| ![Two datasets in the training data, mixed equally](docs/screenshots/dataset.png) | ![Tokenizer examples with colored tokens](docs/screenshots/tokenize.png) |
| **Batches: a worked example** | **Pick a model size and see its parameter count** |
| ![One window of tokens turned into next-token training examples](docs/screenshots/batches.png) | ![Model size presets with a live parameter estimate of 123,751,680](docs/screenshots/model-size.png) |
| **The architecture, drawn from your settings** | **How every parameter adds up** |
| ![Transformer architecture diagram with 12 attention heads and feed-forward network](docs/screenshots/model-architecture.png) | ![Parameter breakdown table with formulas](docs/screenshots/model-parameters.png) |
| **Encoder, decoder, or both?** | **Attention weights after softmax, with the real model's heatmap** |
| ![Encoder-only, decoder-only and encoder–decoder models compared, with stage labels](docs/screenshots/encoder-decoder.png) | ![Toy attention weights and real attention heatmap for one layer and head](docs/screenshots/attention-weights.png) |
| **How a context vector is computed** | **Output projection, dropout and the residual connection** |
| ![Context vector as a weighted sum of value vectors](docs/screenshots/attention-context.png) | ![Output projection, a dropout example and the residual add](docs/screenshots/attention-residual.png) |
| **The context vector goes through LayerNorm and the feed-forward network** | **Final LayerNorm, logits and softmax pick the next token** |
| ![LayerNorm, widen to 16, GELU, narrow back, residual](docs/screenshots/attention-ffn.png) | ![Logits from the final vector and the token embeddings, softmax probabilities](docs/screenshots/attention-logits.png) |
| **Your real model's next-token probabilities** | **Generation with inference metrics** |
| ![Top 10 next-token probabilities from the trained model](docs/screenshots/attention-next-token.png) | ![Streamed generation with time to first token, KV cache size, memory and confidence colors](docs/screenshots/generate.png) |
| **KV cache: with vs without** | |
| ![Side-by-side generation benchmark with and without the KV cache](docs/screenshots/kv-cache.png) | |
| **What one training step does** | **Training on a mix: updates, parameters updated, tokens** |
| ![Loss and perplexity examples, learning-rate schedule and the plan for a run](docs/screenshots/train-explained.png) | ![Live training stats on two datasets mixed equally](docs/screenshots/train.png) |

## Inside the transformer

The web app's fifth step follows a short prompt through a whole transformer block, and out the other end as
a prediction, in 13 stages. It uses **small made-up numbers** (4 dimensions instead of 384) so every value
can be checked by hand, and every number on screen is actually computed from the ones before it. Each stage
is labelled **Encoder & decoder** (an encoder like BERT does the same) or **Decoder only**:

| # | Stage | Shown | |
|---|---|---|---|
| 1 | Tokens become IDs | the real token IDs | Encoder & decoder |
| 2 | Embeddings | token vector + position vector = `x`, where dropout applies | Encoder & decoder |
| 3 | **LayerNorm** | mean, standard deviation and the normalized vector (pre-norm, as in GPT-2) | Encoder & decoder |
| 4 | Queries, keys, values | `Q = LN(x)·W_Q`, `K = LN(x)·W_K`, `V = LN(x)·W_V` | Encoder & decoder |
| 5 | Attention scores | `Q·Kᵀ / √d_k`, one worked out term by term | Encoder & decoder |
| 6 | Causal mask | future positions set to `−∞` | **Decoder only** |
| 7 | Softmax | attention weights per token, where attention dropout applies | Encoder & decoder |
| 8 | Context vector | each token's weighted sum of value vectors, term by term | Encoder & decoder |
| 9 | Output projection, **dropout**, residual | `context·W_O`, a dropout example (zero + rescale by 1/(1−p)), `h = x + …` | Encoder & decoder |
| 10 | LayerNorm and the **feed-forward network** | `LN(h)` → widen to 16 → GELU → narrow to 4 → dropout → `+ h` | Encoder & decoder |
| 11 | Many heads, many layers | how heads are concatenated and blocks repeat, with your model's sizes | Encoder & decoder |
| 12 | Final LayerNorm, **logits**, softmax | the last token's vector · each candidate's embedding (weight tying) → probabilities | **Decoder only** |
| 13 | Your model's real prediction | top 10 next tokens from the **trained model**; click one to append it and continue | **Decoder only** |

Above the walkthrough, a card compares **encoder-only** (BERT: unmasked attention, for understanding text),
**decoder-only** (GPT and this model: masked attention, for generating text) and **encoder–decoder**
(original Transformer, T5: adds **cross-attention**, for translation). This model has no encoder and no
cross-attention. The Model step's architecture diagram marks the same decoder-only parts, plus every
LayerNorm and dropout.

Below the walkthrough, a heatmap shows your trained model's **real attention weights** for the same
prompt, for any layer and head (or averaged over heads).

## Choosing datasets

TinyStories is the default, but any text dataset works, and you can **train on several at once**. A dataset
can be given as:

| Form | Example |
|---|---|
| Hugging Face name | `Salesforce/wikitext` (plus subset `wikitext-2-raw-v1`) |
| Hugging Face link | `https://huggingface.co/datasets/roneneldan/TinyStories` |
| Link to a text file | `https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt` |

File links can point to `.txt`, `.jsonl`, `.json`, `.csv` or `.parquet` files. GitHub page links are
converted to the raw file automatically. Plain `.txt` files are split into documents at blank lines.

From the command line:

```bash
python3 full-model.py --dataset Salesforce/wikitext --dataset-config wikitext-2-raw-v1
python3 full-model.py --dataset https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
python3 full-model.py --list-datasets
```

| Option | Meaning | Default |
|---|---|---|
| `--dataset` | Hugging Face name or link, or a link to a text file. Repeat it to train on several | `roneneldan/TinyStories` |
| `--dataset-config` | Subset name, for datasets that have them | none |
| `--text-column` | Column holding the text | `text` |
| `--mix` | With several datasets: `size` (in proportion to their tokens) or `equal` | `size` |
| `--list-datasets` | Print the public datasets below with ready-to-run commands | |

In the web app, enter the same values in step 1. In the notebook, they're the form fields in Step 1.

### Training on several datasets

Each window in a batch is drawn from one dataset. **By size** picks datasets in proportion to their
tokens; **equally** gives each the same share, so a small dataset isn't drowned out (mixed by size, Tiny
Shakespeare would get only 0.07% of the batches next to TinyStories). In the command line, write a
dataset's subset and column as `name@subset#column`:

```bash
python3 full-model.py --mix equal \
  --dataset roneneldan/TinyStories \
  --dataset Salesforce/wikitext@wikitext-2-raw-v1 \
  --dataset https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

### Text-only validation

Datasets are checked before anything is downloaded, using the dataset's metadata:

- Datasets with **image, audio or video** columns are rejected.
- The chosen column must exist and hold **text** (not numbers or labels). If it doesn't, the message lists
  the text columns that do exist.
- Datasets with several **subsets** list them so you can pick one (in the web app, as clickable buttons).
- **Links** must be reachable and point to a data file, not a web page.
- After downloading, the first rows are checked to make sure they hold **readable text**.

### Public datasets that work out of the box

Each of these was checked with the validation above. The web app shows them as a gallery with **Add** and
**View** buttons.

| Dataset | What it is | Size | Use with |
|---|---|---|---|
| [Mini FineWeb-Edu](https://huggingface.co/datasets/BaoMinhTran/mini-fineweb-edu) | Educational web pages (the default, `small` subset; also `debug`, `tiny`, `base`, `full`) | 595K pages · 1.3 GB | `--dataset BaoMinhTran/mini-fineweb-edu@small` |
| [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) | Simple short stories | 2.1M stories · 1.0 GB | `--dataset roneneldan/TinyStories` |
| [Tiny Shakespeare](https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare) | A 1 MB selection of Shakespeare's plays | 1 text file · 1.1 MB | `--dataset https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt` |
| [WikiText-2](https://huggingface.co/datasets/Salesforce/wikitext) | Good and featured Wikipedia articles | 37K lines · 7.7 MB | `--dataset Salesforce/wikitext@wikitext-2-raw-v1` |
| [WikiText-103](https://huggingface.co/datasets/Salesforce/wikitext) | Over 100M tokens of Wikipedia | 1.8M lines · 315 MB | `--dataset Salesforce/wikitext@wikitext-103-raw-v1` |
| [Simple English Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) | Every Simple English Wikipedia article | 242K articles · 157 MB | `--dataset wikimedia/wikipedia@20231101.simple` |
| [WritingPrompts](https://huggingface.co/datasets/euclaise/writingprompts) | Stories written for Reddit writing prompts | 273K stories · 605 MB | `--dataset euclaise/writingprompts#story` |
| [TinyStories Instruct](https://huggingface.co/datasets/roneneldan/TinyStoriesInstruct) | TinyStories with instructions | 21.8M lines · 2.7 GB | `--dataset roneneldan/TinyStoriesInstruct` |
| [CNN / DailyMail](https://huggingface.co/datasets/abisee/cnn_dailymail) | News articles | 287K articles · 837 MB | `--dataset abisee/cnn_dailymail@3.0.0#article` |
| [AG News](https://huggingface.co/datasets/fancyzhx/ag_news) | News headlines and descriptions | 120K articles · 20 MB | `--dataset fancyzhx/ag_news` |
| [IMDB Reviews](https://huggingface.co/datasets/stanfordnlp/imdb) | Movie reviews | 25K reviews · 83 MB | `--dataset stanfordnlp/imdb` |
| [Rotten Tomatoes](https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes) | One-sentence review snippets | 8.5K sentences · 0.5 MB | `--dataset cornell-movie-review-data/rotten_tomatoes` |

Sizes are download sizes reported by Hugging Face. Find more on the
[Hugging Face datasets hub](https://huggingface.co/datasets?modality=modality:text&sort=trending).

### Where datasets are stored

Each dataset is downloaded once and kept in its own folder, with its own tokens and trained model, so
switching back and forth doesn't redo any work. A `validation`, `valid`, `val` or `test` split is used for
validation; if there isn't one, a small slice of `train` is held out.

```
data/
├── roneneldan__tinystories/
│   ├── dataset/               local copy of the downloaded dataset (~1.8 GB for TinyStories)
│   ├── train.bin              token IDs for training (~900 MB for TinyStories)
│   ├── validation.bin         token IDs for validation
│   ├── best_model_params.pt   best weights from training on this dataset
│   ├── model_config.json      model settings and training summary for those weights
│   └── runs.jsonl             history of every training run
└── mix-<datasets>-<hash>/     a model trained on several datasets (their tokens stay in their own folders)
```

`data/` is git-ignored, so none of this is committed.

## How many parameters?

The default (Small) model, with its 256-token context window, has **30,044,544** parameters. The web app
shows this table for whatever size you build:

| Part | Formula | Small model |
|---|---|---|
| Token embeddings | `vocab_size × n_embd` | 50,257 × 384 = 19,298,688 |
| Position embeddings | `block_size × n_embd` | 256 × 384 = 98,304 |
| Transformer blocks | `n_layer × (12·n_embd² + 13·n_embd)` | 6 × 1,774,464 = 10,646,784 |
| Final LayerNorm | `2 × n_embd` | 768 |
| Output layer | shares the token embeddings (weight tying) | 0 |
| **Total** | | **30,044,544** |

Inside each block: query/key/value projection `n_embd × 3·n_embd + 3·n_embd`, attention output
`n_embd² + n_embd`, feed-forward expand `n_embd × 4·n_embd + 4·n_embd`, feed-forward project back
`4·n_embd × n_embd + n_embd`, and two LayerNorms `2 × 2·n_embd`. The calculation is in `param_breakdown()`
in [model.py](model.py), and it matches PyTorch's own count.

### Bigger models

Add layers or widen the embedding to get a bigger model. **Adding attention heads doesn't add
parameters**: the heads split the embedding into smaller pieces (`n_embd / n_head` each).

| Preset | Layers | Heads | Embedding | Parameters | Memory to train* |
|---|---|---|---|---|---|
| Tiny | 4 | 4 | 256 | 16,090,880 | 0.26 GB |
| Small (default) | 6 | 6 | 384 | 30,044,544 | 0.48 GB |
| Medium | 8 | 8 | 512 | 51,082,752 | 0.82 GB |
| Large (GPT-2 small size) | 12 | 12 | 768 | 123,849,984 | 1.98 GB |

Counts are for the default 256-token context window; the position table adds `block_size × n_embd`, so a
128-token window has slightly fewer (e.g. 29,995,392 for Small). \*Weights, gradients and AdamW state at
16 bytes per parameter, plus activations. Any combination up to 48
layers, 32 heads and an embedding size of 2,048 can be set by hand, as long as it fits in memory.

## Getting better stories

The first versions of this project produced stories like *"She was a big house. She wanted to go to a big
her mommy"*: grammatical fragments that never hold together. The model wasn't broken; it was badly
undertrained, and a few settings made every step count for much less than it should. These changes fixed it:

| Change | Before | Now | Why it matters |
|---|---|---|---|
| Separator between stories | none | `<|endoftext|>` after every story | Without it, stories ran straight into each other, so the model never learned where one ends and training windows spliced unrelated stories together. |
| Learning rate | `1e-4` → `1e-5` | `6e-4` → `6e-5` | `1e-4` is too low for small GPTs; nanoGPT uses `6e-4` for the 124M GPT-2. |
| Gradient accumulation | 8 (Quick), 32 (Full) | 1 | With accumulation, 2,000 steps gave only 250 weight updates (625 for 20,000). Now every step is an update. |
| Context window | 128 tokens | 256 tokens | Many stories are longer than 128 tokens, so the model couldn't see the beginning of the story it was writing. |
| Dropout | 0.1 | 0.0 | Train and validation loss were equal: the model was underfitting, not memorizing, so dropout only slowed it down. |
| Gradient clipping | 0.5 | 1.0 | The usual setting; 0.5 clipped normal updates. |
| Generation | ran to the length limit | stops at `<|endoftext|>` | A finished story ends instead of starting a new one mid-sentence. |

Token files made before the separator was added are detected (they have no `tokens.json`) and re-tokenized
automatically the next time a dataset is used.

### Measured: before and after

Both runs used TinyStories on a Colab **Tesla T4** (float16).

| | Old settings | New settings |
|---|---|---|
| Model | Large (124M) | Medium (51M) |
| Steps / optimizer updates | 2,000 / 250 | 20,000 / 20,000 |
| Validation loss at step 1,000 | 4.905 | **2.649** |
| Validation loss at step 2,000 | 4.527 | **2.214** |
| Best validation loss | 4.527 | **1.583** (step 19,000) |
| Training time | 11.2 min | 101.2 min (26,991 tokens/s, 8.56 GB peak GPU memory) |

The new settings beat the old ones at the same step even with a model less than half the size. Training
and validation loss stayed equal the whole way, so the model is still learning rather than memorizing, and
a longer run or a larger model would go lower still.

![Training and validation loss for the 20,000-step Medium run on a T4](docs/screenshots/colab-loss-full.png)

A story from the new model, generated on the Mac (MPS) from the prompt *"Once upon a time"*. It stopped by
itself at `<|endoftext|>`:

> Once upon a time, there was a little boy named Timmy. Timmy loved to play with his toys all day long. One
> day, he accidentally broke his favorite toy car. He was very sad and didn't know what to do.
>
> Timmy's mom saw him crying and asked him what was wrong. Timmy told his mom that he broke his toy car and
> he couldn't fix it. [...] But then, Timmy's mom made a present for him. She gave him a toy car to play
> with instead. Timmy was happy again and played with his new toy car all day long.

The stories now hold one plot and one set of characters from start to finish. Small slips remain, such as a
character changing name mid-story or a sentence that doesn't follow.

### Measured: Mini FineWeb-Edu, the new default

The **Large** model (123,849,984 parameters) trained on Mini FineWeb-Edu's `small` subset for **30,000
steps** on a Colab **NVIDIA L4** (bfloat16): 245.8 million tokens in **161.7 minutes** at 25,330 tokens/s,
with 11.1 GB of the L4's 23.7 GB at the peak.

| Step | 1,000 | 2,000 | 5,000 | 10,000 | 15,000 | 20,000 | 25,000 | 29,000 |
|---|---|---|---|---|---|---|---|---|
| Validation loss | 5.428 | 4.870 | 4.287 | 4.006 | 3.847 | 3.734 | 3.628 | **3.603** |

The best validation loss is **3.603** (perplexity about 37). That's much higher than TinyStories' 1.58, as
expected: educational web pages use a far bigger vocabulary and range of topics than children's stories.
Training and validation loss stayed equal throughout, and the loss was still falling at the end, so more
data, more steps or a bigger model would all help.

![Training and validation loss for the Large model on Mini FineWeb-Edu, 30,000 steps on an L4](docs/screenshots/fineweb-loss.png)

**Generated text, and why sampling settings matter.** The same model on the Mac (MPS), with the same seed:

> **Greedy** (69% of 4-token runs repeat earlier ones): *Photosynthesis is the process by which plants convert
> sunlight into energy. It is the process by which plants convert sunlight into energy. The process of
> photosynthesis is a process by which plants convert sunlight into energy. This process is called
> photosynthesis. The process of photosynthesis is called photosynthesis. The process of photosynthesis is
> called photosynthesis. …*

> **Balanced** (temperature 0.8, top-p 0.9, repetition penalty 1.15; 0% repeated): *Photosynthesis is the
> process of converting sunlight into energy. This process uses a variety of substances to produce a wide
> range of goods and services, from cars to cars to industrial plants. The process is responsible for
> producing more energy than you might think. It's important to note that not all energy-intensive
> chemicals are toxic and should be treated to ensure safety and efficacy. …*

> **Greedy** (74% repeated): *The French Revolution began in 1776 when the French Revolution was finally
> broken. The French Revolution was a major turning point in the French Revolution. The French Revolution
> was a major turning point in the French Revolution. …*

> **Balanced** (0% repeated): *The French Revolution began in the 17th century when France signed the French
> Crown. It was first issued a military service at its present time, and the French capital of Paris became
> the property of France. In 1775, King Charles I of France moved to France for a military service …*

Greedy decoding gets the first sentence right and then loops. With a small amount of randomness and a
repetition penalty, the model writes fluent, varied, encyclopedia-style English, but the facts are invented:
a 124M model trained on 246M tokens has learned how educational text *sounds*, not what is true.

**The KV cache on the L4** (Step 9 of the notebook, the same text both ways): here the cache was slightly
*slower*, 0.92× for short generations and 0.98× when filling the window. A data-center GPU runs this model's
small matrix multiplications so fast that each step is mostly fixed launch overhead, and the cache adds a
little bookkeeping. On the Apple M5 the same comparison saved 37% (see [The KV cache](#the-kv-cache)). The
cache grows to 18.9 MB (73.7 KB per token in float32) when the 256-token window is full.

![Time to generate with and without the KV cache on an NVIDIA L4, and the cache size](docs/screenshots/fineweb-kv-cache-l4.png)

## Improving the model, step by step

The full story, with the reasoning and lessons behind each step, is in
[docs/journey.md](docs/journey.md).

The steps taken so far to improve the model, and the ones planned next, in order. The rule from step 6 on:
**change one thing at a time**, train it the same way as the baseline, and compare with the
[evaluation suite](#the-evaluation-suite) before keeping it.

### Done

| # | Step | What changed | Result |
|---|---|---|---|
| 1 | **Fix undertraining** | `<\|endoftext\|>` between documents, learning rate 6e-4, no gradient accumulation, 256-token context, no dropout, clipping 1.0, stop at the end of a story ([details](#getting-better-stories)) | TinyStories validation loss after 2,000 steps: 4.527 → **2.214**, with a smaller model |
| 2 | **Train longer on a cloud GPU** | 20,000 steps of the Medium model on a Colab T4 | Loss **1.583**: coherent short stories ([before and after](#measured-before-and-after)) |
| 3 | **Harder, more realistic data** | Mini FineWeb-Edu as the default; the Large model for 30,000 steps on a Colab L4 | Loss **3.603** on educational web text ([results](#measured-mini-fineweb-edu-the-new-default)) |
| 4 | **Better decoding** | Top-p, min-p, repetition / frequency / presence penalties, greedy; **Balanced** (temperature 0.8, top-p 0.9, repetition penalty 1.15) as the default ([sampling](#sampling-how-the-next-token-is-picked)) | The FineWeb model's repetition: **73% → 0%** of 4-token runs |
| 5 | **Keep every model, measure every model the same way** | A model library and a fixed evaluation suite ([saved models](#saved-models-and-evaluation)) | A baseline to beat: **3.603** on FineWeb, 5.257 on WikiText-2, 73% greedy repetition |
| 6 | **Resume training** | The full training state is saved at every evaluation; runs can be finished after an interruption or extended, in the app, on the command line and on Colab (with Google Drive) ([details](#resuming-and-extending-a-run)) | Tested on the Mac: a run stopped at step 25 resumed there and finished; a 60-step run extended to 90 kept its loss history and best checkpoint |
| 7 | **RoPE + a sliding KV cache** (A1, I1) | Rotary position embeddings instead of a learned position table (**Positions** in the Model step, `--rope`); with RoPE the KV cache is a ring buffer that slides past the window instead of being rebuilt ([the A/B](docs/journey.md#11-architecture-step-one-rotary-positions-rope)) | Small model, 1,000 steps on TinyStories: validation loss 2.631 → **2.475**, lower on WikiText-2 and Shakespeare too, with 98k fewer parameters; past the window the cache is never rebuilt (1.14× → **1.47×** faster than no cache); past the window it approximates a full recompute rather than matching it exactly ([why](docs/journey.md#11-architecture-step-one-rotary-positions-rope)) |
| 8 | **RMSNorm** (A2) | Scale-only normalization instead of LayerNorm (**Normalization** in the Model step, `--rmsnorm`) ([the A/B](docs/journey.md#12-architecture-step-two-rmsnorm)) | **A tie**, on top of RoPE: validation loss 2.475 → 2.483, held-out WikiText-2 9.461 → 9.454; 5k fewer parameters and a ~17% faster training step on the Mac. Kept as an option |
| 9 | **SwiGLU feed-forward** (A3) | A gated feed-forward layer, `silu(xW) · xV`, 8/3 × wide instead of 4 × so the parameter count stays the same (**Feed-forward** in the Model step, `--swiglu`) ([the A/B](docs/journey.md#13-architecture-step-three-a-gated-feed-forward-layer-swiglu)) | **A win**, on top of RoPE: validation loss 2.475 → **2.366**, held-out TinyStories 2.440 → **2.338**, the baseline's loss reached ~30% sooner; unseen datasets about the same; greedy repetition up (9% → 28%), Balanced 0% |
| 10 | **No biases** (A4) | No bias terms in linear layers or LayerNorms (**Biases** in the Model step, `--no-bias`) ([the A/B](docs/journey.md#14-architecture-step-four-no-biases)) | **A tie**, on top of RoPE + SwiGLU: validation loss 2.366 → 2.365 with 29k fewer parameters; greedy repetition back to 7.5%, so the jump in step 9 was noise. Kept as an option |
| 11 | **QK-norm** (A5), and A/B tests on a cloud GPU | RMS-normalize each head's queries and keys before attention (**QK-norm** in the Model step, `--qk-norm`); `ab_test.py` and the notebook's A/B section run comparisons on Colab ([the A/B](docs/journey.md#15-architecture-step-five-qk-norm-and-moving-the-ab-tests-to-a-gpu)) | **A small win**: 2.369 → 2.336 at 6e-4, 2.266 → 2.251 at 1.8e-3, less greedy repetition; no loss spikes at either rate. **Tripling the learning rate helped more** (2.369 → 2.266): 6e-4 is too low for these short runs. Each variant took 2½ min on an L4 vs ~25 on the Mac |
| 12 | **A learning-rate sweep** | 5 learning rates × QK-norm off/on, on a Colab L4 ([details](docs/journey.md#16-a-learning-rate-sweep-and-what-qk-norm-is-really-for)) | **Best at 1.8e-3** (2.251 with QK-norm). Without QK-norm, loss rises to 3.33 at 1.2e-2; with it, only 2.55. The new baseline for A/B tests: RoPE + SwiGLU, no biases, QK-norm, lr 1.8e-3 |
| 13 | **Grouped-query attention** (A6) | Query heads share key/value heads (**Key/value heads** in the Model step, `--kv-heads N`); the KV cache stores only the shared heads ([the A/B](docs/journey.md#17-architecture-step-six-grouped-query-attention)) | 3 KV heads: **cache 2× smaller for free** (2.251 → 2.252); 2: 3× smaller, +0.013; 1 (multi-query): 6× smaller, +0.033. No generation speedup at this size: the cache is tiny and the shared heads are copied out |
| 14 | **Depth vs width** (A7) | Four shapes at ~29.9M parameters: 3×448, 6×384, 11×320, 22×256 (`ab_test.py` `layers=`/`embd=`/`heads=`) ([the A/B](docs/journey.md#18-architecture-step-seven-deep-and-thin-or-wide-and-shallow)) | **6×384 is best** at 1,000 updates (2.251); 3 and 11 layers 2.27; 22 layers 2.40 (slow start, still closing). The 3-layer model generates **5× faster** than the 22-layer one, with a 4× smaller KV cache |

### Next: the architecture

The focus now is the model itself rather than the training data: each idea is added as a switch, trained
for a short run on the Mac (the Small model, 1,000 updates on TinyStories, the same data and seed; since step 12, at learning rate 1.8e-3) and kept only if the
evaluation suite says it helps.

`ab_test.py` runs these comparisons: name the variants, and it trains each the same way, evaluates it and writes a
comparison table and loss chart. The notebook's **A/B test** section does the same on a cloud GPU, and
`import_run.py` brings its zip home into the model library:

```bash
python3 ab_test.py --title "A5 QK-norm" --variant "baseline: rope swiglu no-bias" --variant "qk: rope swiglu no-bias qk-norm"
```

| # | Idea | What it is | Why it should help | How it's measured |
|---|---|---|---|---|
| ~~A1~~ | ~~**RoPE**~~ | Done: step 7 above | | |
| ~~A2~~ | ~~**RMSNorm**~~ | Done: step 8 above | | |
| ~~A3~~ | ~~**SwiGLU feed-forward**~~ | Done: step 9 above | | |
| ~~A4~~ | ~~**No biases**~~ | Done: step 10 above | | |
| ~~A5~~ | ~~**QK-norm**~~ | Done: step 11 above | | |
| ~~A6~~ | ~~**Grouped-query attention**~~ | Done: step 13 above | | |
| ~~A7~~ | ~~**Deep and thin vs wide and shallow**~~ | Done: step 14 above | | |
| A8 | **Layer sharing** | Reuse each block's weights twice in a row | More depth for the same parameters and memory | Loss, parameters, speed |
| A9 | **Multi-token prediction** | Extra heads that also predict the 2nd, 3rd… next token during training | Richer training signal; the extra heads can draft tokens for speculative decoding (I5) | Loss, and draft acceptance rate |
| A10 | **Mixture of experts** | Several feed-forward "experts", each token routed to one or two | More parameters for the same compute per token | Loss at equal compute, memory |

### Next: inference

| # | Idea | What it is | Why it should help | How it's measured |
|---|---|---|---|---|
| ~~I1~~ | ~~**Sliding KV cache**~~ | Done: step 7 above | | |
| I2 | **Half-precision inference** | Run the model in float16 / bfloat16 on the Mac's GPU | About half the memory, often faster | tokens/s, memory, same outputs within tolerance |
| I3 | **Quantized weights and KV cache** | 8-bit weights and/or 8-bit keys and values | 2–4× less memory | Loss increase, memory, speed |
| I4 | **Better decoding algorithms** | Beam search, contrastive search, typical sampling, DRY (a smarter repetition penalty) | Fewer loops and better coherence without retraining | Repetition, distinct pairs, samples in the comparison card |
| I5 | **Speculative decoding** | A small model (Tiny, or A9's extra heads) drafts several tokens; the large model checks them in one pass | 1.5–3× faster generation with identical output | Acceptance rate, tokens/s |
| I6 | **Batched generation** | Generate for several prompts at once | Much higher throughput for a little extra latency | tokens/s vs batch size |
| I7 | **Prefix caching** | Reuse the KV cache of a prompt prefix that's already been seen | Much shorter time to first token for repeated prompts | Time to first token |
| I8 | **Compiled generation** | `torch.compile` and a fixed-size cache, CUDA graphs on Nvidia | Less per-token overhead, the main cost for small models | tokens/s on the Mac and on Colab |

### Kept for later

| # | Idea | Why |
|---|---|---|
| L1 | **Faster training** | `torch.compile`, gradient accumulation to ~100K tokens per step and a learning-rate check; batches of 32 × 256 tokens leave a cloud GPU mostly idle (25,330 tokens/s on the L4) |
| L2 | **Tokenize large datasets in little memory** | Stream the tokenizer over the dataset in chunks, so Mini FineWeb-Edu fits on a 16 GB Mac and joins the held-out tests |
| L3 | **Longer context (512 tokens)** | Educational pages average ~800 tokens; at 256 the model never sees a whole page (easier with A1) |
| L4 | **Filter the data by quality** | Keep only Mini FineWeb-Edu pages with an educational score of 3 or more |
| L5 | **Mix in synthetic textbooks** | Cosmopedia: clean, explanatory text, the idea behind TinyStories applied to general knowledge |
| L6 | **One combined cloud run** | Everything that won above, with ~1 billion tokens (about 10× the current run) |
| L7 | **Instruction fine-tuning** | Question-and-answer pairs so the model answers questions; a Chat mode in Generate |
| L8 | **A masked diffusion language model** | Generate a whole block at once and refine it; reuses A1–A4 and the whole pipeline |

## Training: what gets updated

Each step runs a batch forward, measures the loss (the average of `−ln(p)`, where `p` is the probability
the model gave the correct next token), and computes gradients. Every `gradient_accumulation_steps` steps,
AdamW updates **every parameter**. The app shows, live and for every past run:

- **Optimizer updates**: `steps ÷ gradient_accumulation_steps` (the *Quick* preset: 2,000 ÷ 1 = 2,000).
- **Parameters updated**: `parameters × updates` (Small, *Quick*: 30,044,544 × 2,000 ≈ 60 billion).
- **Tokens processed**: `steps × batch_size × block_size` (*Quick*: 2,000 × 32 × 256 ≈ 16.4 million).

Every run is appended to `runs.jsonl` in the model's folder and listed in the Train step's history table.
The command line prints the same summary at the end of a run.

### Resuming and extending a run

Every evaluation saves the **full training state** next to the model, in `training_state.pt`: the latest
weights, AdamW's running averages for every parameter, the mixed-precision scaler, the step, the loss history
and the random-number state, with a small summary in `training_state.json`. A run can then carry on later:

- **Finish an interrupted run.** Continuing with the same number of steps picks up at the step where the state
  was saved, with the same weights, optimizer state and learning-rate schedule (after stopping in the app,
  closing the terminal, or a disconnected Colab session).
- **Extend a finished run.** Asking for more steps stretches the cosine learning-rate schedule to the new end.
  A finished run has decayed to the minimum learning rate, so instead of crawling on at the minimum, the
  learning rate ramps back up to the stretched curve over a short re-warmup (200 steps, or a fifth of the
  extension if that's shorter) and then decays again. Extending a 30,000-step run to 40,000, it ramps from
  6e-5 to 1.4e-4 and returns to 6e-5 at step 40,000.

Every run now also ends with a final evaluation, so the last stretch of training counts towards the best
checkpoint, and a run shorter than one evaluation interval still saves its model.

Where to continue a run:

| Where | How |
|---|---|
| Web app | **Continue training** in the Train step: shows the saved run (step, best loss, learning rate) and trains until the step you choose |
| Command line | `python3 full-model.py --resume [--max-iters N]` |
| Colab / Kaggle | Tick `resume` in Step 5. Files in `/content` disappear when a Colab session disconnects, so for long runs tick `keep_checkpoints_in_drive` to save the checkpoints and training state in Google Drive |
| Colab → Mac | Tick `include_training_state` in Step 10; `import_run.py` installs it, and the run continues on the Mac |

The training state is about three times the size of the weights (the weights plus AdamW's two running
averages), e.g. about 1.5 GB for the 124M Large model.

## Inference metrics

The Generate step streams tokens as they are produced and reports:

| Metric | What it means |
|---|---|
| Prompt tokens / generated tokens | Tokens in and out |
| Context used | How many tokens the model could see at the end (at most the context window, 256) |
| Time to first token | Reading the prompt (prefill) and picking the first token, with the prefill speed |
| Total time, decode speed | Speed of the decode loop: one forward pass per new token |
| KV cache | Size of the cache at the end, and per token (`2 × n_layer × n_embd × 4 bytes`, 18 KB for the default model) |
| Memory | Size of the weights, and GPU memory in use (plus the peak on CUDA) |
| Average confidence | The model's average probability for the tokens it picked |
| Per-token timing chart | Time for each token; the first one is the slowest |

Each generated token is colored by the model's probability for it, and hovering shows its top alternatives
and how many candidates it was drawn from.

### Sampling: how the next token is picked

The model gives a probability to every one of the 50,257 tokens; **sampling settings** decide which of them
can be picked. The Generate step has all the common ones, applied in this order:

| Setting | What it does | Off |
|---|---|---|
| Repetition penalty | Divides the score of every token already in the context by the penalty (CTRL-style) | 1 |
| Frequency / presence penalty | Subtracts penalty × times seen / a flat amount for tokens seen at all (OpenAI-style) | 0 |
| Temperature | Sharpens (below 1) or flattens (above 1) the probabilities. **0 = greedy**: always the most likely token | – |
| Top-k | Keeps the k most likely tokens | 0 |
| Top-p (nucleus) | Keeps the smallest set of tokens whose probabilities add up to p | 1 |
| Min-p | Drops tokens less likely than min-p × the top token's probability | 0 |

Presets set them in one click: **Greedy**, **Balanced** (temperature 0.8, top-p 0.9, repetition penalty
1.15), **Creative** (temperature 1.1, min-p 0.05) and **Old default** (temperature 0.8, top-k 50, no
penalty). **Balanced is the default**: with the old default, the FineWeb model often fell into loops,
repeating one sentence over and over (73% of 4-token runs repeated with greedy decoding, 0% with Balanced).

**How the next token is picked** shows the model's top candidates for the token after your prompt and updates
as you move the sliders: the model's own probability, the probability after the settings, which tokens a
filter cut (and which one), which tokens the penalties pushed down, how many tokens are left and how much of
the probability they hold.

![The next-token candidates before and after sampling settings, with tokens cut by min-p](docs/screenshots/sampling-explorer.png)

**Compare sampling settings** continues the same prompt with three settings side by side, with the same
random seed, and measures each one: distinct word pairs, repeated 4-token runs, average confidence and
candidates per token. With the TinyStories model, greedy decoding repeated 9% of its 4-token runs, the
Balanced preset 0%, and temperature 1.6 without filters drew from all 50,257 tokens and fell apart into
word salad. The Generate step also reports repetition and candidates per token for every generation.

![Greedy, Balanced and temperature 1.6 continuing the same prompt side by side](docs/screenshots/sampling-compare.png)

### The KV cache

With the **KV cache** on (the default), every layer keeps the keys and values it has already computed, so
each new token only runs itself through the model instead of re-reading the whole context. The cache is
allocated once for the whole context window and filled in place. It's exact: cached and uncached logits
match to within 2×10⁻⁶. The **KV cache: with vs without** card generates the same tokens both ways, with the
same seed, and compares them.

The gain grows with the number of tokens the model has to re-read. Measured on an Apple M5 with the 51M
Medium model (256-token window), generating 328 tokens from an 8-token prompt:

| | Without cache | With KV cache |
|---|---|---|
| Time per token | 42.1 ms | 26.7 ms |
| Decode speed | 23.7 tok/s | 37.4 tok/s |
| Total for 328 tokens | 13.8 s | **8.8 s (1.58× faster, 37% saved)** |
| KV cache | none | 8.4 MB (33 KB per token) |
| Same tokens generated | | ✓ |

GPU timings vary from run to run (another run measured 1.28×), so run it a few times. With a small model,
each step is dominated by the fixed cost of launching work on the GPU, so short generations gain little;
for large language models reading thousands of tokens, the cache is what makes generation practical. This
model's positions are learned per slot, so once the 256-token window is full the cache has to be rebuilt
every step (shown as "rebuilds"), and the advantage disappears.

**Per-token timing charts.** Pick *fill the window* or *past the window* as the length to see the whole
story. Without the cache, each token gets slower as the context grows. With the cache, the time stays flat
until the window is full, then jumps to meet the uncached line. The second chart shows the total time, and
the gap between its lines is the time saved.

![Time for each token and total time, with and without the KV cache, on an Apple M5](docs/screenshots/kv-timing.png)

**How the KV cache works.** An interactive diagram of the cache for your prompt. Step through generation, or
press Play, and see which keys and values each layer computes and which it reads back from the cache. Turn
the cache off to see every earlier token recomputed instead. The counters show the work: with the cache,
the number of tokens processed grows in a straight line; without it, with the square of the length.

![Interactive KV cache diagram: keys and values per layer, computed now vs read from the cache](docs/screenshots/kv-diagram.png)

**KV cache memory.** A calculator for `2 × layers × KV width × bytes × tokens × sequences`, for this model,
GPT-2 and Llama models, with the context length, batch size and precision as sliders. It compares the cache
with the size of the weights and shows when the cache becomes the bigger of the two: Llama 3.1 8B at 32K
tokens for 8 users needs 34 GB of cache for 16 GB of weights. It also explains grouped-query attention,
which shares one key/value head among several query heads to shrink the cache 4×.

![KV cache memory calculator comparing cache size with model weights](docs/screenshots/kv-memory.png)

The Colab and Kaggle notebook's Step 9 runs the same comparison on the cloud GPU and saves a chart of the
time taken and the cache size (`kv_cache.png`) in the results zip.

## What happens on each run

| Step | Module | First run | Later runs |
|---|---|---|---|
| 1. Get the data | `data.py` | Validates, downloads and saves a local copy of each dataset | Loads the local copies |
| 2. Tokenize | `tokenization.py` | Converts every document to GPT-2 token IDs | Skipped |
| 3. Batches | `data.py` | Samples random 256-token windows (from several datasets if mixed) | Same (fresh batches every run) |
| 4. Train | `pretrain.py` | Trains, saves the best weights and logs the run | Same |
| 5. Plot | `plots.py` | Training and validation loss | Same |
| 6. Generate | `inference.py` | Writes text from the prompts | Same |

## Project structure

```
full-model.py              Command-line entry point: model config, --dataset/--mix options, runs every step
app/
  server.py                Web app backend (FastAPI): runs the steps, streams generation, reports progress
  static/                  Web app front end (HTML, CSS, JavaScript; no build step)
data.py                    Dataset sources, text-only validation, local caching, multi-dataset BatchLoader
catalog.py                 Public text datasets shown in the gallery and by --list-datasets
tokenization.py            GPT-2 tokenizer and writing train.bin / validation.bin
model.py                   GPT architecture (with an optional KV cache), param_breakdown() and estimate_loss()
train_config.py            TrainConfig hyperparameters and setup_device() (MPS / CUDA / CPU)
pretrain.py                Training loop: AdamW, warmup + cosine LR schedule, gradient accumulation
plots.py                   Loss curves
inference.py               Checkpoints, streamed generation, KV cache benchmark, GPU memory, attention maps
train_step_by_step.ipynb   Notebook for training on Google Colab or Kaggle
import_run.py              Brings a model trained on Colab or Kaggle into data/ (from the notebook's results zip)
registry.py                Model library: every trained or imported model saved under models/<id>/
evaluate.py                Evaluation suite: the same held-out loss and generation tests for every model
docs/screenshots/          Screenshots used in this README
```

## Configuration

### Model size: `full-model.py`

```python
config = GPTConfig(
    vocab_size=50257,   # GPT-2 tokenizer
    block_size=256,     # context window (tokens)
    n_layer=6,          # transformer blocks
    n_head=6,           # attention heads per block
    n_embd=384,         # embedding size
    dropout=0.0,
    bias=True
)
```

### Training: `train_config.py`

| Setting | Default | Meaning |
|---|---|---|
| `max_iters` | 20000 | Training steps |
| `learning_rate` | 6e-4 | Peak learning rate |
| `min_lr` | 6e-5 | Learning rate at the end of the cosine decay |
| `warmup_steps` | 1000 | Steps of linear warmup |
| `batch_size` | 32 | Sequences per batch |
| `gradient_accumulation_steps` | 1 | Batches averaged per optimizer update (1 = update every step) |
| `eval_interval` | 1000 | Evaluate every N steps |
| `eval_iters` | 100 | Batches per evaluation |
| `weight_decay` | 0.1 | AdamW weight decay |
| `grad_clip` | 1.0 | Gradient norm clipping |

The web app's *Quick* preset uses 2,000 steps, 200 warmup, and evaluation every 250 steps over 50 batches,
with the same learning rate and no gradient accumulation. *Full* uses the defaults above. See
[Getting better stories](#getting-better-stories) for why these settings replaced the original ones.

## Saved models and evaluation

Training a new model on the same data used to replace the old one. Now every model you train (in the app or
with `full-model.py`) or import from Colab is also saved in a **model library**, `models/<id>/`, under its
own name, and is never overwritten. The app's **Models** step (step 8) lists them:

![The model library: every saved model with its size, training data and best validation loss](docs/screenshots/models-library.png)

- **Use for generation** switches Generate and *Inside the transformer* to that model, whatever datasets
  are selected. A model trained on Colab can be used without downloading its dataset. The choice is remembered
  across restarts (`models/active.json`).
- **Find models on this Mac** adds the models already under `data/`, including the backups `import_run.py`
  made. **Save the current datasets' model** saves the one you're working with; training also saves the
  current model before replacing it.
- **Rename** and **Delete** (which removes only the library copy; `data/` isn't touched).

On the command line:

```bash
python3 registry.py scan        # add every model under data/ to the library
python3 registry.py list        # show the library
```

### The evaluation suite

Every model gets the **same tests**, so the numbers can be compared directly (`evaluate.py`):

1. **Held-out loss** on every tokenized dataset on this Mac: 64 fixed windows of 256 tokens from each
   validation split, the same positions for every model. Loss is the average surprise per token; perplexity
   (e<sup>loss</sup>) is roughly how many tokens the model is choosing between.
2. **Generation**: 6 fixed prompts (stories and educational text), 100 tokens each with a fixed seed,
   decoded greedy and with the Balanced preset. Measures how much greedy decoding repeats itself, the variety
   of the Balanced text, the model's confidence, speed with the KV cache and time to first token.
3. **Size**: parameters, weights in memory and KV cache per token.

Results are stored with each model (`models/<id>/eval.json`). Tick models in the Models step and press
**Evaluate selected models** (15–40 seconds each on an M5), or:

```bash
python3 evaluate.py              # evaluate every saved model that hasn't been evaluated yet
python3 evaluate.py --table      # print the comparison table
```

All six models on this Mac, evaluated on an Apple M5:

![Comparison of six models: held-out loss per dataset, repetition, variety, speed and size](docs/screenshots/models-compare.png)

| Model | TinyStories | WikiText-2 | Tiny Shakespeare | Greedy repetition | Speed |
|---|---|---|---|---|---|
| Large 124M · Mini FineWeb-Edu · 30k steps | 3.872 | **5.257** | **5.664** | 73% | 49 tok/s |
| Medium 51M · TinyStories · 20k steps | **1.561** | 8.693 | 7.632 | **3%** | 99 tok/s |
| Large 124M · TinyStories · 2k steps (old settings) | 4.510 | 9.842 | 8.495 | 69% | 60 tok/s |
| Small 30M · TinyStories · 2k steps (old settings) | 4.620 | 9.765 | 8.100 | 73% | 97 tok/s |
| Small 30M · TinyStories + Tiny Shakespeare · 1k steps | 6.382 | 9.375 | 7.047 | 89% | 98 tok/s |
| Tiny 16M · WikiText · 40 steps (a quick test) | 10.701 | 10.547 | 10.783 | 98% | 118 tok/s |

What it shows: each model is best on the kind of text it was trained on. The TinyStories model is
excellent on stories (perplexity 4.8) and lost on Wikipedia, while the FineWeb model, trained on varied
educational text, is the best on WikiText-2 and Shakespeare, neither of which it ever saw. The well-trained
TinyStories model barely repeats itself even with greedy decoding (3%); the FineWeb model still loops
greedily (73%), which is why the Balanced preset matters for it. The undertrained models, including the
original 2,000-step runs, score far worse everywhere. (Mini FineWeb-Edu joins the held-out tests once it's
downloaded and tokenized on this Mac.)

The **Samples** card shows every model's continuation of the same prompt side by side:

![Six models continuing "Photosynthesis is the process" with the same seed](docs/screenshots/models-samples.png)

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

For a model trained on several datasets, use `mix_paths([...dataset folders...])` from `data.py` instead of
`dataset_paths`.

## Train on a cloud GPU: Google Colab or Kaggle

Everything runs on your Mac except, if you like, the training itself. The notebook
[`train_step_by_step.ipynb`](train_step_by_step.ipynb) runs on **Google Colab** or **Kaggle**, trains on their
Nvidia GPU, and packages the result into one zip that you bring back with `import_run.py`.

On a GPU the notebook also shows:

- the GPU's name, memory and utilization (`nvidia-smi`);
- a **model size** dropdown up to **Large** (124M parameters, GPT-2 small size), which fits easily on a T4;
- training throughput (tokens per second) and **peak GPU memory** after training;
- **Step 9**: a KV cache benchmark at several lengths (tokens per second with and without the cache, cache
  size, rebuilds once the window is full, GPU memory);
- **Step 10**: everything packaged into `tinystories-run-<dataset>-<date>.zip`.

CUDA GPUs train with mixed precision: bfloat16 on GPUs with native support (compute capability 8.0+, such
as A100 and L4), float16 with a gradient scaler otherwise (such as the T4).

### Google Colab

1. Click the **Open in Colab** badge at the top of this page.
2. *Runtime → Change runtime type → T4 GPU* (free), or L4 / A100 with Colab Pro or compute units.
3. Run the cells from top to bottom. Step 0 clones this repository and installs the dependencies.
4. Set the run with the form fields: the dataset (Step 1), `model_size` (Step 4, e.g. `large`) and `mode`
   (Step 5: `quick` or `full`).
5. Step 10 downloads the results zip to your computer (allow the download if the browser asks). It's also
   in the Files panel.

Things to know on Colab:

- **"This notebook was not authored by Google".** Colab shows this warning for any notebook opened from
  GitHub. The notebook doesn't mount Google Drive or ask for your Google data; choose *Run anyway*.
- **Running everything except the download.** To run the whole notebook but decide later whether to download
  the zip, select the Step 10 cell and use *Runtime → Run before*. Run Step 10 when you're ready.
- **Changing code mid-session.** Step 0 runs `git pull` when the repository is already cloned. After pulling
  new code, use *Runtime → Restart session* (files in `/content` are kept) so Python loads it.
- **Precision.** GPUs with native bfloat16 (compute capability 8.0+: A100, L4) train in bfloat16. Older GPUs
  such as the T4 train in float16 with a gradient scaler: the T4 reports bfloat16 as supported, but only
  through slow emulation (1.4 s per step for the Large model, versus about 3.9 steps per second in float16).
- **Sessions.** Free Colab sessions can disconnect when idle or when usage limits are reached, and files in
  `/content` are lost when they do. On Colab Pro, GPU time uses compute units.

### Measured on a Colab T4

A real run of this notebook on Colab's free-tier **Tesla T4** (15.6 GB): TinyStories, the **Large** model
(123,751,680 parameters) and the *Quick* preset (2,000 steps).

| Stage | Time |
|---|---|
| Clone, install and download TinyStories | about 2 minutes |
| Tokenize 2.1 million stories (2 CPU processes) | about 5 minutes |
| Write `train.bin` and `validation.bin` | seconds (it took 16 minutes before the write was rewritten to use Arrow chunks) |
| Train 2,000 steps | **11.2 minutes** |

| Result | Value |
|---|---|
| Training speed | 3.9 steps/s, **12,202 tokens/s** (including evaluations) |
| Optimizer updates × parameters | 250 × 123,751,680 = **30.9 billion parameter updates** |
| Peak GPU memory | **7.03 GB** of 15.6 GB |
| Best validation loss | **4.527** (the Small model on an M5 reaches 4.61 after the same 2,000 steps) |
| Training tokens | 471,872,517 (identical to the Mac's `train.bin`) |
| Results zip | 495 MB (almost all of it the weights) |

| Training output | Loss curves (from the results zip) |
|---|---|
| ![Training output on a Colab T4: 2,000 steps in 11.2 minutes, 12,202 tokens/s, 7.03 GB peak GPU memory](docs/screenshots/colab-training.png) | ![Training and validation loss for the Large model on a T4](docs/screenshots/colab-loss.png) |

**Inference on the T4** (Step 9, the same text generated with and without the KV cache):

| Tokens | Without cache | With KV cache | Ratio | Cache rebuilds |
|---|---|---|---|---|
| 25 | 145 tok/s | 149 tok/s | 1.02× | 0 |
| 60 | 145 tok/s | 147 tok/s | 1.01× | 0 |
| 110 | 123 tok/s | 111 tok/s | 0.90× | 0 |
| 250 | 108 tok/s | 114 tok/s | 1.06× | 131 |

On the T4 the cache made no real difference for this model: the GPU is fast enough that each step is
dominated by Python and kernel-launch overhead rather than by the attention arithmetic the cache saves. Both
ways produced exactly the same text.

**Back on the Mac.** The zip was imported with `import_run.py`, and the web app now uses the Large model.
Its training history shows the T4 run, and it generates on the M5 at about 58 tokens per second with an
average confidence of 19% (the 30M Small model from the Mac scores about 14%). That run used the original
settings; a later 20,000-step Medium run with the new ones reached a validation loss of 1.583 and replaced
it (see [Measured: before and after](#measured-before-and-after)). Using the built-in browser
of the Claude desktop app, the download landed in the project folder rather than `~/Downloads`; result
zips are git-ignored wherever they land.

### Kaggle

1. On [kaggle.com](https://www.kaggle.com), *Create → New Notebook*, then *File → Import Notebook* and paste
   the link to [`train_step_by_step.ipynb`](https://github.com/a-ramakrishnan/tinystories-ARM-model/blob/main/train_step_by_step.ipynb)
   on GitHub.
2. In the side panel, *Settings → Accelerator → GPU* (T4 or P100; one GPU is used) and switch **Internet
   on**, which Kaggle only allows for phone-verified accounts.
3. Run the cells from top to bottom. The Colab form fields appear as plain variables you can edit. The code
   and data go to `/tmp`, so only the results zip is saved as the notebook's output.
4. Download the zip from the **Output** panel (`/kaggle/working`). *Save Version → Save & Run All* runs the
   whole notebook in the background, so you can close the tab.

Kaggle sessions run for up to about 12 hours, with a weekly GPU quota (around 30 hours at the time of
writing; your account page shows the current limit).

### Bring the model home

From the project folder on your Mac:

```bash
python3 import_run.py ~/Downloads/tinystories-run-roneneldan__tinystories-20261110-1430.zip
```

It puts the weights and config in `data/<dataset>/` (backing up the model already there to
`data/<dataset>/backups/`), adds the run to the training history, and keeps the loss chart, GPU metrics and
sample story in `data/<dataset>/imported/`. Start `python3 app/server.py` and the Generate and *Inside the
transformer* steps use the GPU-trained model. If that dataset isn't on your Mac yet, add it in the Dataset
step first.

## Credits

- Data loading and the model are adapted from Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT).
- Default dataset: [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) by Ronen Eldan and
  Yuanzhi Li.
- Thanks to Claude for all the help in refactoring, UI and writing readme etc.

## License

[MIT](LICENSE)

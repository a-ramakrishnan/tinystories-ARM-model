# From a training script to a working language model: what changed and why

This page follows the project from its first commit to today, one change at a time. For each step: what was
wrong, what we changed, why it works, what it measured, and the lesson worth keeping. Every number here was
measured in this project, on an Apple M5 Mac or a Colab GPU.

If you're learning how language models are built, read it top to bottom: most of the gains came from things
that are easy to get wrong and rarely explained.

**Contents**

1. [Where we started](#1-where-we-started)
2. [Making it run well on a Mac](#2-making-it-run-well-on-a-mac)
3. [Making it understandable: the web app](#3-making-it-understandable-the-web-app)
4. [Training on a cloud GPU](#4-training-on-a-cloud-gpu)
5. [The big fix: the model was undertrained](#5-the-big-fix-the-model-was-undertrained)
6. [Training longer, then on harder data](#6-training-longer-then-on-harder-data)
7. [Inference: the KV cache, measured honestly](#7-inference-the-kv-cache-measured-honestly)
8. [Decoding: why the model repeated itself](#8-decoding-why-the-model-repeated-itself)
9. [Measuring progress: a model library and an evaluation suite](#9-measuring-progress-a-model-library-and-an-evaluation-suite)
10. [Never losing a run: resume training](#10-never-losing-a-run-resume-training)
11. [Architecture, step one: rotary positions (RoPE)](#11-architecture-step-one-rotary-positions-rope)
12. [Lessons](#12-lessons)
13. [What's next](#13-whats-next)

---

## 1. Where we started

The first commit was a single 356-line script, adapted from nanoGPT: a GPT-2-style decoder-only transformer
(token and learned position embeddings, 6 blocks of causal self-attention and an MLP, LayerNorm, weight tying)
trained on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories), short stories written with a
small child's vocabulary.

| Setting | First version |
|---|---|
| Model | 6 layers, 6 heads, 384-dim (~30M parameters) |
| Context window | 128 tokens |
| Learning rate | 1e-4, "decaying" to a minimum of **5e-4** |
| Gradient accumulation | 32 batches per weight update (8 in the later 2,000-step *Quick* preset) |
| Dropout | 0.1 |
| Gradient clipping | 0.5 |
| Documents | concatenated with nothing between them |
| Device | Nvidia GPU (`cuda`) or CPU; on a Mac, the CPU |

The architecture was fine. Almost everything around it held the model back, and the output showed it:
grammatical fragments that never held together, like *"She was a big house. She wanted to go to a big her
mommy"*.

---

## 2. Making it run well on a Mac

**The problem.** The script only knew `cuda` and `cpu`, so on a Mac it trained on the CPU. And the learning
rate's "minimum" (5e-4) was *higher* than its peak (1e-4), so the cosine schedule made the learning rate
**rise** during training instead of fall.

**What we changed.**
- Pick the Apple GPU through PyTorch's Metal backend (`mps`) when there's no Nvidia GPU. Train in float32 on
  the Mac, since mixed precision only pays off on CUDA.
- Fix the schedule so it decays from the peak down to a tenth of it.
- Split the script into modules: `data.py`, `tokenization.py`, `model.py`, `pretrain.py`, `inference.py`,
  and so on. One file per idea makes each one easier to read, test and change.

**Lesson.** Check that your hyperparameters say what you think they say. A minimum learning rate above the
maximum is easy to write and silently turns "decay" into "growth".

---

## 3. Making it understandable: the web app

A training script hides everything interesting. We built a local web app (`python3 app/server.py`) that walks
through the pipeline one step at a time, with real numbers from your own data and model:

- **Data**: any Hugging Face text dataset or text file, checked to make sure it really is text, stored
  locally, and mixable with others.
- **Tokenize** and **Batches**: the GPT-2 tokenizer on your own text, and how training windows and targets
  are cut from the token stream.
- **Model**: an architecture diagram and a line-by-line parameter count that matches PyTorch's exactly.
- **Inside the transformer**: 13 stages from token IDs to the next-token prediction, with the trained
  model's real attention weights.
- **Train**, **Generate** and later **Models**: live loss curves, streamed generation, and model comparison.

**Lesson.** You learn a model by looking inside it. Every number the app shows is computed, not illustrated.

---

## 4. Training on a cloud GPU

A Mac is great for learning and short runs; long runs need a bigger GPU. We wrote a notebook
(`train_step_by_step.ipynb`) that runs the same steps on Google Colab or Kaggle and packs the trained model
into a zip that `import_run.py` brings back to the Mac.

Two problems showed up on Colab's free T4 GPU:

| Problem | Cause | Fix | Result |
|---|---|---|---|
| 1.4 seconds per step | The T4 reports bfloat16 as "supported", but only emulates it | Use float16 (with a gradient scaler) unless the GPU is Ampere or newer | **3.86 steps/s**, about 5× faster |
| 16 minutes to write the token files | Writing 1,024 shards one by one | Write straight from the dataset's Arrow chunks | **seconds**, byte-identical output |

**Lesson.** "Supported" is not "fast". Measure the step time on the actual hardware before a long run.

---

## 5. The big fix: the model was undertrained

The first Colab run trained the 124M Large model for 2,000 steps: validation loss **4.527**, and the stories
were still word salad. A bigger GPU didn't help, because the problem wasn't the hardware. The training set-up
made every step count for far less than it should:

| Change | Before | After | Why it matters |
|---|---|---|---|
| Separator between documents | none | `<\|endoftext\|>` after every story | Without it, stories ran into each other: the model never learned where one ends, and training windows spliced unrelated stories together |
| Gradient accumulation | 8 (Quick) / 32 (Full) batches per update | 1 | 2,000 steps gave only **250** weight updates, and 20,000 steps only 625. Accumulation makes each update better informed, but a small model needs *many* updates |
| Learning rate | 1e-4 | 6e-4 → 6e-5 | 1e-4 is too timid for small GPTs; nanoGPT uses 6e-4 for the 124M GPT-2 |
| Context window | 128 tokens | 256 tokens | A typical story is ~230 tokens: at 128, the model never saw how its own story began |
| Dropout | 0.1 | 0 | Training and validation loss were equal, so the model was underfitting, not memorizing. Dropout only slowed it down |
| Gradient clipping | 0.5 | 1.0 | 0.5 clipped ordinary updates |
| Generation | ran to the length limit | stops at `<\|endoftext\|>` | A finished story ends instead of starting a new one mid-sentence |

Measured on the same T4:

| Validation loss | Step 1,000 | Step 2,000 |
|---|---|---|
| Before: Large model, 124M | 4.905 | 4.527 |
| After: Medium model, 51M | **2.649** | **2.214** |

A model less than half the size, with better settings, was far ahead at every step.

**Lessons.**
- **Count your optimizer updates**, not your steps.
- **If training loss ≈ validation loss, you're underfitting.** Remove regularization and train harder.
- **Tell the model where documents end.** One separator token fixed story structure.

---

## 6. Training longer, then on harder data

**TinyStories, 20,000 steps.** The Medium model (51M) for 20,000 steps on a T4 (101 minutes) reached
validation loss **1.583** and wrote coherent stories that end by themselves:

> Once upon a time, there was a little boy named Timmy. Timmy loved to play with his toys all day long. One
> day, he accidentally broke his favorite toy car. He was very sad and didn't know what to do. …

![TinyStories: training and validation loss over 20,000 steps](screenshots/colab-loss-full.png)

**Mini FineWeb-Edu, 30,000 steps.** Children's stories are easy. We switched the default dataset to
[Mini FineWeb-Edu](https://huggingface.co/datasets/BaoMinhTran/mini-fineweb-edu), educational web pages like
those modern small models are trained on, and trained the Large model (124M) for 30,000 steps on a Colab L4
(bfloat16, 162 minutes, 246M tokens). Validation loss: **3.603** (perplexity ~37).

![Mini FineWeb-Edu: training and validation loss over 30,000 steps](screenshots/fineweb-loss.png)

The model writes fluent, encyclopedia-style English, and invents its facts ("The French Revolution began in
1776…").

**Lessons.**
- Loss values only compare within the same data. 3.6 on web text is a much better model than 3.6 would be
  on children's stories.
- A small model learns how text *sounds* long before it learns what is *true*.

---

## 7. Inference: the KV cache, measured honestly

Generating text is one forward pass per new token. Without a cache, every pass re-reads the whole context.
The **KV cache** keeps every layer's keys and values for the tokens already seen, so each pass only processes
the new token.

We built it (preallocated once, filled in place, checked to give the same logits as no cache), then measured
it, with per-token timing charts, an interactive diagram and a memory calculator in the app:

| Hardware | Model | Result |
|---|---|---|
| Apple M5 | Medium 51M, 328 tokens | 13.8 s → **8.8 s**: 1.58× faster, 37% saved |
| Colab L4 | Large 124M | **0.92–0.98×**: slightly *slower* |

![Time per token and total time with and without the KV cache on an M5](screenshots/kv-timing.png)

Why slower on the L4? A data-center GPU runs this model's small matrix multiplications so fast that each
step is dominated by fixed overhead (launching GPU work, Python), and the cache adds bookkeeping. The savings
grow with model size and context length; for LLMs reading thousands of tokens, the cache is what makes
generation practical.

There was a catch on both: with **learned positions**, a full context window means every position shifts, so
the cache had to be **rebuilt for every token** past the window, and the advantage disappeared. Section 11
fixes that.

**Lessons.**
- Measure optimizations on the hardware you'll use; the textbook answer can be wrong at your scale.
- The KV cache trades memory for speed: `2 × layers × width × bytes` per token. For Llama 3.1 8B at 32K
  tokens for 8 users, that's 34 GB of cache for 16 GB of weights.

![How the KV cache works: keys and values computed now vs read from the cache](screenshots/kv-diagram.png)

---

## 8. Decoding: why the model repeated itself

The FineWeb model kept producing loops like *"The process of photosynthesis is called photosynthesis. The
process of photosynthesis is called photosynthesis. …"*.

**Why.** Transformers learn to copy from their context (induction heads): once a sentence has appeared,
repeating it becomes the most likely continuation, and every repeat makes the next one more likely. Greedy
decoding, or a low temperature with only top-k, can't escape.

**What we changed.** Added the standard sampling controls: temperature (0 = greedy), top-k, **top-p**
(nucleus), **min-p**, and **repetition / frequency / presence penalties**. We also added a next-token explorer
that shows which candidates each setting cuts, and a side-by-side comparison. Then we made **Balanced**
(temperature 0.8, top-p 0.9, repetition penalty 1.15) the default.

| Decoding | Repeated 4-token runs (FineWeb model) |
|---|---|
| Greedy | **73%** |
| Balanced | **0%** |

![Greedy, Balanced and high-temperature sampling continuing the same prompt](screenshots/sampling-compare.png)

**Lesson.** Decoding is half of inference quality. The same weights can loop or write fluently depending on
four numbers.

---

## 9. Measuring progress: a model library and an evaluation suite

Each new training run used to replace the last model, and every result was measured a little differently.
Two changes fixed that:

- A **model library** (`models/`): every trained or imported model is kept under its own name, and any of
  them can be switched in for generation.
- An **evaluation suite** (`evaluate.py`): the same tests for every model. That's held-out loss on fixed
  windows of each dataset, six fixed prompts decoded greedy and Balanced (repetition, variety, confidence,
  speed), and size.

![Six models compared on the same tests](screenshots/models-compare.png)

What it showed: each model is best on its own kind of text. The TinyStories model reaches perplexity 4.8 on
stories but is lost on Wikipedia. The FineWeb model, trained on varied text, is best on WikiText-2 and
Shakespeare, neither of which it saw. And every undertrained model loops (69–98% greedy repetition).

**Lesson.** Decide how you'll measure *before* you change things. Every improvement after this point is a
comparison against a fixed baseline.

---

## 10. Never losing a run: resume training

Long runs die: a closed laptop, a disconnected Colab session. Every evaluation now saves the **full training
state**, not just the weights: AdamW's running averages, the mixed-precision scaler, the step, the loss
history and the random-number state. A run can be **finished** after an interruption, or **extended**: the
cosine schedule is stretched to the new end, and the learning rate re-warms from where it stopped instead of
crawling at the minimum.

Building it found a bug: a run shorter than one evaluation interval never saved its checkpoint. Every run now
ends with a final evaluation.

**Lesson.** The weights alone aren't enough to continue training. The optimizer's state is half of what
was learned about the loss landscape.

---

## 11. Architecture, step one: rotary positions (RoPE)

From here on, the focus is the architecture, one change at a time, each tested against the baseline with the
same data, seed and settings.

**Learned positions (GPT-2).** Each of the 256 positions has a trained vector, added to the token embedding.
The model only knows positions it was trained on, and a position vector has to describe the absolute slot,
even though language mostly cares about *relative* distance.

**RoPE (Llama, Mistral, Qwen).** No position vectors. Inside attention, each query and key is **rotated** by an
angle proportional to its position, a different speed for each pair of dimensions. The dot product of a query
at position *m* and a key at position *n* then depends only on *m − n*.

Two consequences:

1. No position parameters, and attention that is naturally relative.
2. **The KV cache can slide.** Each cached key carries its position in its rotation, so when the window is
   full, the newest token simply overwrites the oldest slot (a ring buffer). Nothing is rebuilt, and the
   cached and uncached outputs stay identical past the window.

In this project it's a switch: **Positions: Learned / RoPE** in the Model step, `--rope` on the command line,
or `pos_encoding = "rope"` in the notebook.

**The A/B test.** The Small model (30M), 1,000 steps on TinyStories, seed 42, learning rate 6e-4, the
same batches. The only difference is the position encoding.

| | Learned positions | RoPE |
|---|---|---|
| Parameters | 30,044,544 | 29,946,240 (no position table) |
| Validation loss at steps 250 / 500 / 750 / 1,000 | 3.462 / 2.956 / 2.706 / 2.631 | **3.164 / 2.753 / 2.536 / 2.475** |
| Held-out loss: TinyStories | 2.602 | **2.440** |
| Held-out loss: WikiText-2 | 9.569 | **9.461** |
| Held-out loss: Tiny Shakespeare | 8.149 | **8.040** |
| Greedy repetition (4-token runs) | 14.1% | **8.9%** |
| Balanced repetition | 0.3% | 0.7% |
| KV cache past the 256-token window | rebuilt 79 times, 1.14× faster than no cache | **never rebuilt, 1.47× faster** |
| Cached output identical to uncached | yes | yes |

RoPE was ahead at every evaluation, and by about the same margin (0.16 nats at the end). It reached the
learned model's final loss (2.63) somewhere between steps 500 and 750, which is roughly **30% fewer steps**
for the same quality. The gain carries over to the two datasets neither model was trained on.

Past the window, the learned model has to rebuild its cache for every new token (its positions are absolute
slots, and the window has moved), so the cache barely helps. The RoPE cache simply slides.

A caveat: the RoPE run also trained faster (0.67 vs 0.45 steps/s), but the runs were one after the other on a
16 GB Mac that was swapping, so that difference is memory pressure, not RoPE. Compare the losses, not the times.

**Lesson.** A good inductive bias (attention that depends on relative distance) is worth more than parameters:
RoPE has *fewer* parameters and learns faster. And a design choice in the model (absolute vs relative positions)
decides what's possible at inference (a sliding cache).

---

## 12. Lessons

1. **Most of the early gains were training hygiene, not architecture.** Document separators, enough weight
   updates and a sensible learning rate took validation loss from 4.5 to 2.2 at a smaller size.
2. **Underfitting looks like train loss ≈ val loss.** Fix it with more updates, less regularization and a
   higher learning rate, not a bigger model.
3. **Check your settings mean what you think**: a "minimum" learning rate above the maximum, "supported"
   bfloat16 that's emulated.
4. **Measure on your hardware.** The KV cache saved 37% on a Mac and slowed generation on an L4.
5. **Decoding matters as much as weights** for how good a small model seems.
6. **Fix your evaluation before you optimize**, and change one thing at a time.
7. **Small models learn style before facts.** Fluent and confidently wrong is a stage, not a bug.

## 13. What's next

The [roadmap in the README](../README.md#improving-the-model-step-by-step) lists the next steps.
- **Architecture:** RMSNorm, SwiGLU, no biases, QK-norm, grouped-query attention, depth vs width, layer
  sharing, multi-token prediction, mixture of experts.
- **Inference:** half precision, quantization, better decoding algorithms, speculative decoding, batching,
  prefix caching, compiled generation.
- **Later:** data ideas, instruction tuning and a masked diffusion model.

Every step will be added here with what it measured.

## Reproduce it

```bash
git clone https://github.com/a-ramakrishnan/tinystories-ARM-model.git
cd tinystories-ARM-model
pip install -r requirements.txt
python3 app/server.py          # the step-by-step web app
python3 full-model.py --help   # or the command line
```

For the cloud runs, open the notebook in Colab with the badge in the [README](../README.md).

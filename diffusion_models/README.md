# Masked diffusion language model

A language model that writes by **denoising** instead of left to right: it starts from a fully masked text and
fills it in over T steps, the positions it's most confident about first. Written by Ramakrishnan A; the first
version is kept unchanged in [`v1_original.py`](v1_original.py), and the modules below are built from it.

| File | What it holds |
|---|---|
| [`config.py`](config.py) | The model's shape and the run modes (`smoke`, `quick`, `colab`) |
| [`tokens.py`](tokens.py) | GPT-2's tokenizer plus one `[MASK]` token |
| [`data.py`](data.py) | The project's token files (`data/<dataset>/train.bin`), random 256-token windows |
| [`model.py`](model.py) | The bidirectional transformer with a noise-level (time) embedding; save and load |
| [`noising.py`](noising.py) | Masking a fraction t/T of the tokens; the loss and the likelihood bound |
| [`train.py`](train.py) | The training loop: AdamW, warmup + cosine, mixed precision, `torch.compile`, resumable |
| [`sample.py`](sample.py) | Generation by denoising, recording every step |
| [`render.py`](render.py) | The denoising steps as a GIF |
| [`diffusion_colab.ipynb`](diffusion_colab.ipynb) | Train on Colab, watch it generate, download the results |

**Changes from v1:** GPT-2's tokenizer (the same as the autoregressive models, so both read the same data);
plain text with `<|endoftext|>` between documents instead of a chat template; the project's tokenized files
instead of tokenizing while training; GPT-2-style initialisation; plain PyTorch instead of Accelerate; and a
likelihood bound next to the loss, the number that can be compared with an autoregressive model's loss.

```bash
python3 -m diffusion_models.train --mode smoke                       # a few minutes on the Mac
python3 -m diffusion_models.sample diffusion_models/runs/<run> --prompt "Once upon a time"
```

The `colab` run mode matches the autoregressive *Medium 51M · TinyStories · 20k steps* model: 51M parameters,
164M tokens of TinyStories, the same learning-rate schedule.

## First results (TinyStories, 13 November 2026)

The `colab` run took 49 minutes on a Colab L4 (59,000 tokens/s). Its likelihood bound fell to **2.525** nats per
token; the autoregressive model of the same size and data reaches a loss of 1.583 (the bound is an upper limit, so
the gap may be smaller than it looks).

- **A long plateau first.** For about 2,500 updates (20M tokens) the model only learned how common each word is
  (5.96 nats, the loss of word frequencies alone), then it began to use the context and the loss fell quickly:
  5.78 at update 2,000, 4.44 at 3,000, 3.34 at 5,000. Masked language models such as BERT show the same plateau.
  A higher learning rate didn't shorten it (2e-3 was worse than 6e-4 over 2,000 updates).
- **The sampler matters.** With v1's sampler, which re-samples every position at every step, the middle steps
  read fluently but the final text was jumbled: the last step re-draws all the tokens at once, independently.
  Keeping each token once it's revealed (now the default) gives coherent stories from the same model:

  > Lily wanted to play with her toys. She put it in her room. She saw a new toy on the floor. It was red and
  > shiny. Lily was very happy.

  128 tokens take 64 model passes and about 0.3 s on the L4.

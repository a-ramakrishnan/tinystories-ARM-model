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

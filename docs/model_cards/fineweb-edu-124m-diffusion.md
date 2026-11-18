---
license: mit
language: en
library_name: pytorch
tags:
  - tinystories-arm
  - masked-diffusion
  - diffusion-language-model
  - from-scratch
datasets:
  - BaoMinhTran/mini-fineweb-edu
  - HuggingFaceTB/cosmopedia
---

# FineWeb-Edu 124M masked diffusion (F6)

A 124M-parameter **masked diffusion language model** trained from scratch in
[tinystories-ARM-model](https://github.com/a-ramakrishnan/tinystories-ARM-model). It writes by denoising: the
continuation starts as all `[MASK]`, and at each step a transformer that reads in both directions predicts every
masked position, and the ones it is surest of are revealed.

It has the same design, data and tokenizer as the autoregressive
[FineWeb-Edu 124M base](https://huggingface.co/aramkris/fineweb-edu-124m-base), so the two differ only in the
objective: 12 layers × 768, RoPE, SwiGLU, no biases, QK-norm, 512-token context, 64 noise levels, attention in both
directions; Mini FineWeb-Edu (full) plus Cosmopedia subsets.

| | Likelihood bound (nats/token) | HellaSwag | LAMBADA | ARC-Easy | PIQA | Winogrande |
|---|---|---|---|---|---|---|
| **This model, 2.5B tokens (uploaded)** | **3.830** | 28.0 | 14.4 | 30.1 | 55.4 | 49.1 |
| This model after 1B tokens | 3.923 | 27.6 | 11.8 | 30.7 | 54.2 | 49.6 |
| Autoregressive base, 2.5B tokens (same data) | 2.963 (loss) | 32.3 | 20.3 | 46.8 | 62.4 | 49.9 |

The uploaded weights are from **2.5B tokens** (152,600 updates, the same as the autoregressive base). They're
the best checkpoint by likelihood bound, from update 120,000; the bound drifted back up to about 3.9 over the last
30,000 updates. Benchmark average 35.4 (after 1B tokens: 34.8; the autoregressive base: 42.3). On the same held-out
FineWeb-Edu windows, its bound is 4.148 against the base's loss of 3.296. The full comparison is in
[the project's story, section 38](https://a-ramakrishnan.github.io/tinystories-ARM-model/story.html#38-two-ways-to-write-compared-f4). The likelihood bound is an upper bound on the loss per token; the benchmarks score each
answer by its likelihood bound (8 maskings per answer).

## Use

```bash
git clone https://github.com/a-ramakrishnan/tinystories-ARM-model && cd tinystories-ARM-model
python3 -m infra.model_store pull fineweb-edu-124m-diffusion     # -> diffusion_models/runs/f6-124m-stage2
```

```python
from diffusion_models.model import load
from diffusion_models.sample import diffusion_generate
model = load("diffusion_models/runs/f6-124m-stage2", "cpu")
print(diffusion_generate(model, "Photosynthesis is", max_new_tokens=64, block_size=16, repetition_penalty=1.2,
                         record_steps=False)["text"])
```

Files: `model.pt` (weights), `config.json`, `metrics.json` (loss history, speed, samples), `loss.png`,
`manifest.json` (checksums).

A research model: fluent but loosely connected text, few facts, and 64 passes for 64 tokens.

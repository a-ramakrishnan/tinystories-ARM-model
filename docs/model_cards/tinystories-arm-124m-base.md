---
license: mit
language: en
library_name: pytorch
pipeline_tag: text-generation
tags:
  - tinystories-arm
  - gpt
  - from-scratch
datasets:
  - BaoMinhTran/mini-fineweb-edu
  - HuggingFaceTB/cosmopedia
---

# tinystories-arm 124M base (stage 2)

A 124M-parameter GPT trained from scratch in
[tinystories-ARM-model](https://github.com/a-ramakrishnan/tinystories-ARM-model): written by hand, improved one
measured change at a time, trained on a Mac and on Colab GPUs. Related models: its chat versions
([SFT](https://huggingface.co/aramkris/tinystories-arm-124m-chat-sft),
[SFT + DPO](https://huggingface.co/aramkris/tinystories-arm-124m-chat-dpo)) and a masked diffusion model of the same
size and data ([tinystories-arm-124m-diffusion](https://huggingface.co/aramkris/tinystories-arm-124m-diffusion)).

**Design:** 12 layers × 768, 12 heads, RoPE, SwiGLU, no biases, QK-norm, 512-token context, GPT-2's tokenizer.
**Data:** 2.5B tokens of Mini FineWeb-Edu (full) plus Cosmopedia subsets (stories, WikiHow, OpenStax, Stanford,
Khan Academy), with the held-out documents of Mini FineWeb-Edu small removed; 152,600 updates.

| | This model | OpenAI GPT-2 small (124M) |
|---|---|---|
| Held-out Mini FineWeb-Edu (loss, lower is better) | **3.296** | 3.401 |
| TinyStories / WikiText-2 / Shakespeare (loss) | 2.751 / 5.047 / 5.446 | 2.553 / 4.180 / 4.807 |
| HellaSwag | **32.3%** | 31.1% |
| ARC-Easy | **46.8%** | 43.8% |
| PIQA | 62.4% | 62.9% |
| LAMBADA | 20.3% | 32.6% |
| Winogrande | 49.9% | — |

Zero-shot, lm-evaluation-harness style.

## Use

The architecture is the project's own (`model.py`), so load it with the repository's code:

```bash
git clone https://github.com/a-ramakrishnan/tinystories-ARM-model && cd tinystories-ARM-model
python3 -m infra.model_store pull tinystories-arm-124m-base     # -> models/mini-fineweb-edu-ful-202611151103-bb160c
```

```python
import tiktoken, torch
from inference import load_model, load_model_config
d = "models/mini-fineweb-edu-ful-202611151103-bb160c"
cfg, _ = load_model_config(f"{d}/model_config.json")
model = load_model(cfg, f"{d}/best_model_params.pt", "cpu")
enc = tiktoken.get_encoding("gpt2")
ids = torch.tensor([enc.encode("Photosynthesis is")])
print(enc.decode(model.generate(ids, 60, temperature=0.8, top_k=50)[0].tolist()))
```

Files: `best_model_params.pt` (weights), `model_config.json`, `meta.json` (how it was trained), `eval.json` and
`benchmarks.json` (results), `manifest.json` (checksums).

It writes fluent educational-style English; at 124M parameters it knows few facts reliably, and greedy decoding
repeats itself (use temperature and top-k, or a repetition penalty).

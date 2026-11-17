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
  - HuggingFaceTB/smol-smoltalk
base_model: aramkris/fineweb-edu-124m-chat-sft
---

# FineWeb-Edu 124M chat (SFT + DPO)

A chat version of [FineWeb-Edu 124M base](https://huggingface.co/aramkris/fineweb-edu-124m-base), a GPT trained
from scratch in [tinystories-ARM-model](https://github.com/a-ramakrishnan/tinystories-ARM-model). The [SFT model](https://huggingface.co/aramkris/fineweb-edu-124m-chat-sft) plus DPO against repetition: 2,000 smol-smoltalk questions, the reference answer preferred to the SFT model's own greedy one; 19 minutes on an M5 Mac.

It answers in a plain `User: …` / `Assistant: …` format (no special tokens) and ends its answer with
`<|endoftext|>`. Measured on 100 held-out questions with greedy decoding:

| Model | Loss on the reference answers | Repeated 4-grams | Answers that end | Benchmark average |
|---|---|---|---|---|
| [SFT](https://huggingface.co/aramkris/fineweb-edu-124m-chat-sft) | **1.156** | 12.4% | 70% | 38.5 |
| this model (SFT + DPO) | 1.196 | **6.5%** | **86%** | 38.4 |

The base model scores 42.3 on the same five benchmarks; instruction tuning costs about 4 points. DPO halves the repetition and lets most answers end by themselves, at a small cost in reference-answer loss.

## Use

```bash
git clone https://github.com/a-ramakrishnan/tinystories-ARM-model && cd tinystories-ARM-model
python3 -m infra.model_store pull fineweb-edu-124m-chat-dpo     # -> models/chat-mini-fineweb-ed-202611152150-2746ba
```

```python
import tiktoken
from inference import load_model, load_model_config
from sft import answer
d = "models/chat-mini-fineweb-ed-202611152150-2746ba"
cfg, _ = load_model_config(f"{d}/model_config.json")
model = load_model(cfg, f"{d}/best_model_params.pt", "cpu")
print(answer(model, tiktoken.get_encoding("gpt2"), "Why do leaves change color in autumn?", "cpu"))
```

It answers in the right shape but with the base model's knowledge: plausible-sounding and often wrong on facts.

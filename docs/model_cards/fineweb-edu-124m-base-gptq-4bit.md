---
license: mit
language: en
library_name: pytorch
pipeline_tag: text-generation
base_model: aramkris/fineweb-edu-124m-base
base_model_relation: quantized
tags:
  - tinystories-arm
  - gpt
  - from-scratch
  - gptq
  - 4-bit
  - quantized
datasets:
  - BaoMinhTran/mini-fineweb-edu
  - HuggingFaceTB/cosmopedia
---

# FineWeb-Edu 124M base, GPTQ 4-bit

The [FineWeb-Edu 124M base model](https://huggingface.co/aramkris/fineweb-edu-124m-base) with its weights
stored as 4-bit integers, quantized with **GPTQ** through llm-compressor's `oneshot`, in
[tinystories-ARM-model](https://github.com/a-ramakrishnan/tinystories-ARM-model). **84 MB instead of 494 MB.**

**How it was made** (`quantization/gptq.py`):

- Our GPT, wrapped as a Hugging Face model (`quantization/hf_wrapper.py`), went through
  `oneshot(recipe=GPTQModifier(scheme="W4A16", targets="Linear", ignore=["re:.*lm_head"]), pipeline="sequential")`.
  That means 4-bit weights (−8…7 × one scale per group of 128) in all 48 transformer-block matrices, and 16-bit
  activations.
- It was calibrated on 256 windows of 512 tokens from Mini FineWeb-Edu's training split, in 44 s on a Mac CPU.
- The integers and scales were then read into the project's own quantized layer. The shared embedding / output
  layer is stored at 8 bits, one scale per row.

Held-out loss, on the same 16,384 tokens per set (lower is better). The round-to-nearest column is a 4-bit copy with
no calibration data, shown for comparison:

| | Original (32-bit) | Round to nearest, 4-bit | **This model (GPTQ)** |
|---|---|---|---|
| Weight file | 494 MB | 87 MB | **84 MB** |
| Mini FineWeb-Edu | 3.296 | 3.326 | **3.319** |
| TinyStories | 2.751 | 2.793 | **2.774** |
| Cosmopedia (5 sets) | 2.336–2.785 | +0.035 to +0.041 | **+0.030 to +0.037** |
| WikiText-2 | 5.047 | **5.069** | 5.072 |
| Tiny Shakespeare | 5.446 | **5.472** | 5.481 |
| Mean change over 10 sets | | +0.034 | **+0.030** |

The benchmarks (HellaSwag, ARC-Easy and the rest) haven't been run on the quantized model yet; see the base model's
card for the original's.

## Use

The quantized format is the project's own, so load the model with the repository's code. llm-compressor is only
needed to *make* a GPTQ copy, not to run one:

```bash
git clone https://github.com/a-ramakrishnan/tinystories-ARM-model && cd tinystories-ARM-model
python3 -m infra.model_store pull fineweb-edu-124m-base-gptq-4bit
```

```python
import tiktoken, torch
from inference import load_model, load_model_config
d = "models/mini-fineweb-edu-ful-202611151103-842ab9"
cfg, _ = load_model_config(f"{d}/model_config.json")
model = load_model(cfg, f"{d}/best_model_params.pt", "cpu")     # recognises the quantized file
enc = tiktoken.get_encoding("gpt2")
ids = torch.tensor([enc.encode("Photosynthesis is")])
print(enc.decode(model.generate(ids, 60, temperature=0.8, top_k=50)[0].tolist()))
```

The model can also be used in the project's app (Generate, and the Quantize page, which compares it with the
original).

Files:

- `best_model_params.pt`: the 4-bit integers packed two per byte, their scales, and the norms;
- `model_config.json`, `meta.json`: the recipe;
- `quantize.json`: the comparison above, with a sample from each model;
- `manifest.json`: checksums.

**Speed.** The weights are turned back into floats on every token, so on a Mac it decodes at about 30 tokens/s
against 80–100 for the float model. It saves memory, not time.

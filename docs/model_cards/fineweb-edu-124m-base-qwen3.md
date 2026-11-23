---
license: mit
language: en
library_name: transformers
pipeline_tag: text-generation
tags:
  - tinystories-arm
  - from-scratch
  - qwen3
  - gguf
datasets:
  - BaoMinhTran/mini-fineweb-edu
  - HuggingFaceTB/cosmopedia
base_model: aramkris/fineweb-edu-124m-base
base_model_relation: quantized
---

# FineWeb-Edu 124M base (stage 2): Qwen3 format and GGUF

[aramkris/fineweb-edu-124m-base](https://huggingface.co/aramkris/fineweb-edu-124m-base), a 124M-parameter GPT trained from scratch in
[tinystories-ARM-model](https://github.com/a-ramakrishnan/tinystories-ARM-model), converted **exactly** to the
standard Hugging Face Qwen3 format, so `transformers`, llama.cpp, Ollama, LM Studio and other Qwen3-capable tools run
it with no code from the project. It is not a Qwen model: only the file format is Qwen3's.

| File | For | Size |
|---|---|---|
| `model.safetensors` + `config.json` + tokenizer | `transformers` (bfloat16) | 329 MB |
| `fineweb-edu-124m-base-bf16.gguf` | llama.cpp, Ollama, LM Studio | 326 MB |
| `fineweb-edu-124m-base-Q8_0.gguf` | the same, 8-bit | 174 MB |

## Use

A base model: it continues text (educational web text, stories, how-tos); it doesn't follow instructions. For chat
see [fineweb-edu-124m-chat-dpo-qwen3](https://huggingface.co/aramkris/fineweb-edu-124m-chat-dpo-qwen3).

```bash
llama-cli -hf aramkris/fineweb-edu-124m-base-qwen3:Q8_0 -p "The water cycle is the process by which" -n 100
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("aramkris/fineweb-edu-124m-base-qwen3")
model = AutoModelForCausalLM.from_pretrained("aramkris/fineweb-edu-124m-base-qwen3")
ids = tok("The water cycle is the process by which", return_tensors="pt").input_ids
print(tok.decode(model.generate(ids, max_new_tokens=60, do_sample=True, top_p=0.9, temperature=0.8)[0]))
```

The context window is 512 tokens.

## How this copy was made

Our design is Qwen3's except that the block norms are LayerNorm (without a bias) where Qwen3 has RMSNorm.
[`serving/export_hf.py`](https://github.com/a-ramakrishnan/tinystories-ARM-model/blob/main/serving/export_hf.py)
makes the residual stream mean-free (centering the input embeddings and every matrix that writes into it), which makes
the two norms the same function; the output layer keeps the original embeddings (so they're untied here), and the
query and key projections are scaled by √10 so Qwen3's single epsilon acts like our QK-norm's. Nothing is retrained:
it's the same model. The GGUF files come from llama.cpp's converter
([`serving/gguf.sh`](https://github.com/a-ramakrishnan/tinystories-ARM-model/blob/main/serving/gguf.sh)).

Checked ([`serving/check.py`](https://github.com/a-ramakrishnan/tinystories-ARM-model/blob/main/serving/check.py),
plan step S7) against the original PyTorch model:

| | Original (PyTorch) | llama.cpp, bfloat16 | llama.cpp, 8-bit |
|---|---|---|---|
| Held-out Mini FineWeb-Edu loss | 3.2957 | 3.2959 | 3.2963 |
| HellaSwag (acc_norm) | 32.26% | 32.26% | 32.22% |
| LAMBADA | 20.34% | 20.32% | 20.28% |
| ARC-Easy | 46.84% | 46.76% | 46.72% |
| PIQA | 62.40% | 62.30% | 62.24% |
| WinoGrande | 49.88% | 50.04% | 49.64% |
| Average of the five | 42.35 | 42.34 | 42.22 |

Held-out loss is within 0.0013 on all ten held-out sets; in float32 the largest logit difference from the original
is 9e-5 and greedy text is identical token for token. LAMBADA through llama-server's HTTP API: 20.36%.

MIT licence.

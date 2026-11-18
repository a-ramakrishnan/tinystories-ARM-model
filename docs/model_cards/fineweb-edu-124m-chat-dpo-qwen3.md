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
  - HuggingFaceTB/smol-smoltalk
base_model: aramkris/fineweb-edu-124m-chat-dpo
---

# FineWeb-Edu 124M chat (SFT + DPO): Qwen3 format and GGUF

[aramkris/fineweb-edu-124m-chat-dpo](https://huggingface.co/aramkris/fineweb-edu-124m-chat-dpo), a 124M-parameter GPT trained from scratch in
[tinystories-ARM-model](https://github.com/a-ramakrishnan/tinystories-ARM-model), converted **exactly** to the
standard Hugging Face Qwen3 format, so `transformers`, llama.cpp, Ollama, LM Studio and other Qwen3-capable tools run
it with no code from the project. It is not a Qwen model: only the file format is Qwen3's.

| File | For | Size |
|---|---|---|
| `model.safetensors` + `config.json` + tokenizer | `transformers` (bfloat16) | 329 MB |
| `fineweb-edu-124m-chat-dpo-bf16.gguf` | llama.cpp, Ollama, LM Studio | 326 MB |
| `fineweb-edu-124m-chat-dpo-Q8_0.gguf` | the same, 8-bit | 174 MB |

## Use

It answers in a plain `User: …` / `Assistant: …` format; the chat template is included, and answers end with
`<|endoftext|>`. It answers in the right shape but with a 124M model's knowledge: plausible-sounding and often wrong.

```bash
ollama run hf.co/aramkris/fineweb-edu-124m-chat-dpo-qwen3:Q8_0
llama-server -hf aramkris/fineweb-edu-124m-chat-dpo-qwen3:Q8_0 --jinja -c 512        # OpenAI API on :8080
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("aramkris/fineweb-edu-124m-chat-dpo-qwen3")
model = AutoModelForCausalLM.from_pretrained("aramkris/fineweb-edu-124m-chat-dpo-qwen3")
ids = tok.apply_chat_template([{"role": "user", "content": "What is the capital of France?"}], add_generation_prompt=True, return_tensors="pt")
print(tok.decode(model.generate(ids, max_new_tokens=100)[0, ids.shape[1]:], skip_special_tokens=True))
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

| | Original (PyTorch) | This copy, bfloat16 | This copy, 8-bit GGUF |
|---|---|---|---|
| 100 held-out questions, greedy: repeated 4-grams | 6.5% | 6.6% | 6.7% |
| answers that end by themselves | 86% | 85% | 82% |
| mean answer length (tokens) | 129.1 | 129.1 | 133.2 |
| LAMBADA through llama-server | 12.9% | 13.0% | 12.9% |

The bfloat16 copy's prompt tokens are identical on all 100 questions and 70 of its answers are identical word for
word; the rest diverge when a near-tie flips in bfloat16 and the answer drifts. In float32 the largest logit
difference from the original is 2e-4.

MIT licence.

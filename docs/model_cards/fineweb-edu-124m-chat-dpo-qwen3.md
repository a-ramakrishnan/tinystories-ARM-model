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
base_model: aramkris/fineweb-edu-124m-base
---

# FineWeb-Edu 124M chat (SFT + DPO): Qwen3 format and GGUF

A 124M-parameter GPT trained from scratch in
[tinystories-ARM-model](https://github.com/a-ramakrishnan/tinystories-ARM-model) (project id
`chat-mini-fineweb-ed-202611191448-e45919`), converted **exactly** to the standard Hugging Face Qwen3 format, so
`transformers`, llama.cpp, Ollama, LM Studio and other Qwen3-capable tools run it with no code from the project. It is
not a Qwen model: only the file format is Qwen3's.

## How the model was trained

1. **Pre-training**: [fineweb-edu-124m-base](https://huggingface.co/aramkris/fineweb-edu-124m-base), 2.5B tokens of
   mini-fineweb-edu and Cosmopedia, 512-token context.
2. **Instruction tuning** on smol-smoltalk, at learning rate 5e-5 with **replay**: a fifth of every batch is
   pre-training text. Without it, tuning erased much of what the base model knew (its score on 200 closed-book fact
   questions fell from 57.5% to 44%); with it the score stays at 50%.
3. **DPO** on 2,000 preference pairs (a reference answer preferred over the tuned model's own greedy answer), which
   cuts repetition and makes more answers end on their own.
4. **DPO on reward-model pairs**: for 2,000 new questions the model wrote seven answers (greedy and six samples at
   temperature 0.5), [Skywork-Reward-V2-Qwen3-0.6B](https://huggingface.co/Skywork/Skywork-Reward-V2-Qwen3-0.6B)
   scored them, and its best answer was preferred over its worst. This makes the answers clean without costing
   knowledge.

| | This model (19 Nov) | Before step 4 (18 Nov) | Base model |
|---|---|---|---|
| 100 held-out questions, greedy: repeated 4-grams | 2.3% | 8.5% | |
| answers that end by themselves | 99% | 87% | |
| mean answer length (tokens) | 99 | 127 | |
| closed-book facts, choice (200 questions) | 50.5% | 50.5% | 57.5% |
| closed-book facts, answered right on its own | 31.0% | 30.0% | 24% |
| benchmark average (HellaSwag, LAMBADA, ARC-Easy, PIQA, Winogrande) | 40.5 | 40.5 | 41.5 |

Earlier versions are in this repository's history: the 18 Nov one (before step 4) was here on 19 Nov, and the 15 Nov
one until then (its native PyTorch weights are at
[aramkris/fineweb-edu-124m-chat-dpo](https://huggingface.co/aramkris/fineweb-edu-124m-chat-dpo)).

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
plan step S7) against the original PyTorch model, on the same 100 held-out questions as above:

| | Original (PyTorch) | This copy, bfloat16 | This copy, 8-bit GGUF |
|---|---|---|---|
| 100 held-out questions, greedy: repeated 4-grams | 2.3% | 2.3% | 2.9% |
| answers that end by themselves | 99% | 98% | 98% |
| mean answer length (tokens) | 99.0 | 98.8 | 100.4 |
| LAMBADA through llama-server | 14.4% | 14.3% | 14.3% |

The prompt tokens are identical on all 100 questions; 76 of the bfloat16 copy's answers (39 of the 8-bit one's) are
identical word for word, and the rest diverge when a near-tie flips at the lower precision and the answer drifts,
which moves the repetition and length a little either way.

MIT licence.

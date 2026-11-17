"""GPTQ weight quantization of a standard Hugging Face model with llm-compressor's `oneshot` (plan C1).

1. Calibrate and quantize: GPTQ, W4A16 by default (4-bit weights in groups of 128, activations left in 16 bits),
   on every Linear layer except the output layer, calibrated on packed windows of WikiText-2's *train* split.
2. Compare the original and quantized model on the same things:
   size of the weight files, a greedy continuation of one prompt, and perplexity on WikiText-2's *test* split.

The same model id is used for quantizing, for the tokenizer and for the comparison. The default,
unsloth/Qwen3-0.6B, is an unchanged copy of Qwen/Qwen3-0.6B that's already in the local Hugging Face cache.

llm-compressor pins its own torch and transformers, so it lives in its own environment (not requirements.txt):
    uv venv .venv-quant --python 3.11
    VIRTUAL_ENV=.venv-quant uv pip install -r quantization/requirements.txt
    .venv-quant/bin/python -m quantization.llm_compressor             # quantize (once), then compare
    .venv-quant/bin/python -m quantization.llm_compressor --force     # quantize again

Our own model goes through the same oneshot in gptq.py.
"""

import argparse
import gc
import json
import logging
import math
import os
import pathlib
import time
import warnings

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch  # noqa: E402
from datasets import Dataset, load_dataset  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

PROMPT = "Machine learning is a branch of"
WIKITEXT = ("Salesforce/wikitext", "wikitext-2-raw-v1")


def parse_args():
    ap = argparse.ArgumentParser(description="GPTQ (llm-compressor oneshot) on a Hugging Face model, then compare.")
    ap.add_argument("--model", default="unsloth/Qwen3-0.6B", help="Hugging Face model id or local folder")
    ap.add_argument("--out", default=None,
                    help="where the quantized model goes (default models/quantized_<scheme>/<model name>)")
    ap.add_argument("--scheme", default="W4A16", help="llm-compressor preset scheme (W4A16, W8A16, ...)")
    ap.add_argument("--samples", type=int, default=256, help="calibration windows")
    ap.add_argument("--seq-len", type=int, default=512, help="tokens per calibration window (and perplexity window)")
    ap.add_argument("--eval-tokens", type=int, default=40_000, help="test tokens scored for perplexity (0 = all)")
    ap.add_argument("--stride", type=int, default=256, help="perplexity: new tokens scored per window")
    ap.add_argument("--force", action="store_true", help="quantize again even if the output already exists")
    ap.add_argument("--skip-eval", action="store_true", help="only quantize")
    return ap.parse_args()


# ---------- 1. calibrate and quantize ----------

def calibration_set(tokenizer, n, seq_len):
    """n windows of seq_len tokens, evenly spaced over WikiText-2 train packed into one stream.

    WikiText's rows are single lines and about a third are empty, so taking rows as samples would give many
    tiny or empty ones; packing gives every sample a full window."""
    train = load_dataset(*WIKITEXT, split="train")
    ids = tokenizer("\n\n".join(t for t in train["text"] if t.strip()))["input_ids"]
    n = min(n, len(ids) // seq_len)
    step = (len(ids) - seq_len) // max(1, n - 1)
    windows = [ids[i * step:i * step + seq_len] for i in range(n)]
    print(f"Calibration: {n} windows × {seq_len} tokens from WikiText-2 train ({len(ids):,} tokens)")
    return Dataset.from_dict({"input_ids": windows, "attention_mask": [[1] * seq_len for _ in windows]})


def is_quantized(folder):
    """True when `folder` holds a finished llm-compressor save (a half-written one from a crash doesn't count)."""
    cfg = os.path.join(folder, "config.json")
    if not os.path.exists(cfg):
        return False
    with open(cfg) as f:
        return "quantization_config" in json.load(f)


def quantize(args):
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier
    logging.getLogger("llmcompressor").setLevel(logging.WARNING)

    recipe = GPTQModifier(scheme=args.scheme, targets="Linear", ignore=["lm_head"])
    print(f"Recipe: {recipe}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype="auto")
    t0 = time.time()
    oneshot(
        model=model,
        tokenizer=tokenizer,                       # saved with the model, so the output folder loads on its own
        dataset=calibration_set(tokenizer, args.samples, args.seq_len),
        recipe=recipe,
        max_seq_length=args.seq_len,
        num_calibration_samples=args.samples,
        output_dir=args.out,
    )
    print(f"Quantization complete in {time.time() - t0:.0f} s. Model saved to: {args.out}")
    del model
    gc.collect()


# ---------- 2. compare ----------

def weight_files_size(path):
    """Bytes of the *.safetensors files of a local folder, or of a Hub model (from the cache)."""
    if not os.path.isdir(path):
        from huggingface_hub import snapshot_download
        path = snapshot_download(path, allow_patterns=["*.safetensors"])
    return sum(f.stat().st_size for f in pathlib.Path(path).rglob("*.safetensors"))


def format_size(nbytes):
    if nbytes < 1024 ** 2:
        return f"{nbytes / 1024:.1f} KB"
    if nbytes < 1024 ** 3:
        return f"{nbytes / 1024 ** 2:.1f} MB"
    return f"{nbytes / 1024 ** 3:.2f} GB"


def load(path, device, dtype):
    return AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(device).eval()


@torch.no_grad()
def continuation(model, tokenizer, device, max_new_tokens=60):
    inputs = tokenizer(PROMPT, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)


@torch.no_grad()
def perplexity(model, ids, window, stride, device):
    """Sliding-window perplexity (the Hugging Face recipe): each window sees up to `window` tokens and scores only
    the ones the previous window didn't, so every token is scored once with at least window − stride of context."""
    nll, count, prev_end = 0.0, 0, 0
    for begin in range(0, ids.size(1), stride):
        end = min(begin + window, ids.size(1))
        x = ids[:, begin:end].to(device)
        y = x.clone()
        y[:, :-(end - prev_end)] = -100            # context only: already scored by the previous window
        loss = model(x, labels=y).loss
        k = int((y[:, 1:] != -100).sum())          # the labels are shifted by one inside the model
        nll += float(loss) * k
        count += k
        prev_end = end
        if end == ids.size(1):
            break
    return math.exp(nll / count), count


def compare(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32   # bf16 matmuls are slow on the Mac's CPU
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    size_orig, size_q = weight_files_size(args.model), weight_files_size(args.out)
    print("\nModel size comparison (weight files)")
    print("=" * 45)
    print(f"Original:           {format_size(size_orig)}")
    print(f"Quantized ({args.scheme}): {format_size(size_q)}")
    print(f"Reduction:          {(1 - size_q / size_orig) * 100:.0f}%")

    test = load_dataset(*WIKITEXT, split="test")
    ids = tokenizer("\n\n".join(test["text"]), return_tensors="pt").input_ids
    if args.eval_tokens:
        ids = ids[:, :args.eval_tokens]

    results = {"model": args.model, "scheme": args.scheme, "device": device, "dtype": str(dtype),
               "calibration": {"samples": args.samples, "seq_len": args.seq_len},
               "size_bytes": {"original": size_orig, "quantized": size_q}}
    for label, path in (("original", args.model), ("quantized", args.out)):
        model = load(path, device, dtype)
        t0 = time.time()
        text = continuation(model, tokenizer, device)
        ppl, n = perplexity(model, ids, args.seq_len, args.stride, device)
        results[label] = {"continuation": text, "perplexity": round(ppl, 3), "seconds": round(time.time() - t0, 1)}
        print(f"\n{label.capitalize()} model ({path})")
        print(f"Prompt:     {PROMPT}")
        print(f"Response:   {text}")
        print(f"Perplexity: {ppl:.2f}  ({n:,} WikiText-2 test tokens, window {args.seq_len}, stride {args.stride})")
        del model
        gc.collect()
    results["tokens_scored"] = n

    base, quant = results["original"]["perplexity"], results["quantized"]["perplexity"]
    print("\nPerplexity comparison")
    print("=" * 40)
    print(f"Original:          {base:.2f}")
    print(f"Quantized ({args.scheme}): {quant:.2f}")
    print(f"Difference:        {quant - base:+.2f} ({(quant / base - 1) * 100:+.1f}%)")
    with open(os.path.join(args.out, "comparison.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {os.path.join(args.out, 'comparison.json')}")


def main():
    args = parse_args()
    if args.out is None:
        name = args.model.rstrip("/").split("/")[-1]
        args.out = os.path.join("models", f"quantized_{args.scheme.lower()}", name)
    print(f"Base model:      {args.model}")
    print(f"Quantized model: {args.out}")
    if args.force or not is_quantized(args.out):
        quantize(args)
    else:
        print("Already quantized (pass --force to redo).")
    if not args.skip_eval:
        compare(args)


if __name__ == "__main__":
    main()

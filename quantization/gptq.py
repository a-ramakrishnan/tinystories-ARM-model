"""GPTQ on our own model with llm-compressor's `oneshot` (plan C2), saved as a quantized copy in the model library.

GPTQ (Frantar et al., 2022) quantizes a layer one input column at a time and, after each column, adjusts the columns
not yet quantized to cancel the error just made, using the layer's inputs on calibration text (their Hessian). Blocks
are done in order ("sequential" pipeline), each calibrated on the outputs of the already-quantized blocks before it.

1. Load a library model and wrap it as a Hugging Face model (hf_wrapper.py); oneshot needs one.
2. Calibrate on evenly spaced windows of a training-token file (train.bin), never the held-out data.
3. oneshot with GPTQModifier(scheme, targets="Linear", ignore=[the output layer]): every transformer-block matrix.
4. Read back each layer's integers and scales (after GPTQ every weight is exactly integer × scale) into our own
   QuantLinear; the shared embedding / output layer is rounded to 8 bits, as in round-to-nearest copies, so the two
   methods differ only in the blocks. Save it in the library (checkpoint.add_copy): it then loads with the normal
   loader, without llm-compressor.

llm-compressor pins its own torch and transformers, so this runs in its own environment (the app starts it there):
    .venv-quant/bin/python -m quantization.gptq <model id> [--samples 256] [--seq-len 512] [--scheme W4A16]

Lines starting with "@@ " are progress for the app; the last one is "@@RESULT {json}".
"""

import argparse
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")          # GPT-2's tokenizer comes from the local cache

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SCHEMES = {"W4A16": 4, "W8A16": 8}     # weight bits; activations stay in 16 bits
GROUP = 128                            # W4A16's group size (W8A16: one scale per row)
DEFAULT_DATA = "baominhtran__mini-fineweb-edu-small"


def say(msg):
    print("@@ " + msg, flush=True)


def calibration_windows(data_key, n, seq_len, data_root=None):
    """n windows of seq_len tokens, evenly spaced over data/<key>/train.bin (training text, not held out)."""
    import registry
    path = os.path.join(data_root or registry.DATA_ROOT, data_key, "train.bin")
    data = np.memmap(path, dtype=np.uint16, mode="r")
    starts = np.linspace(0, len(data) - seq_len - 1, n).astype(np.int64)
    return [data[s:s + seq_len].astype(np.int64).tolist() for s in starts], len(data)


def recipe(scheme="W4A16", samples=256, seq_len=512, data=DEFAULT_DATA):
    """What the app shows and meta.json keeps."""
    return {"method": "gptq", "tool": "llm-compressor oneshot", "scheme": scheme, "bits": SCHEMES[scheme],
            "group": GROUP if SCHEMES[scheme] == 4 else None, "targets": "Linear", "ignore": ["re:.*lm_head"],
            "embed_bits": 8, "pipeline": "sequential", "sequential_targets": ["Block"],
            "calibration": {"data": data, "split": "train", "samples": samples, "seq_len": seq_len}}


def unfit_layers(gpt, group):
    """Block matrices whose input width isn't a multiple of the group size: GPTQ can't group them, so they're
    left out of GPTQ and rounded to nearest (one scale per row) instead."""
    return [n for n, m in gpt.named_modules()
            if isinstance(m, torch.nn.Linear) and n != "lm_head" and group and m.in_features % group]


@torch.no_grad()
def to_quantized_gpt(gpt, bits, group, skipped):
    """Replace each GPTQ'd Linear by our QuantLinear holding its integers and scales; round the rest."""
    from quantization.rtn import QuantEmbedding, QuantLinear
    worst = 0.0
    for name, module in list(gpt.named_modules()):
        for child_name, child in list(module.named_children()):
            if not isinstance(child, torch.nn.Linear):
                continue
            full = f"{name}.{child_name}" if name else child_name
            if full == "lm_head":
                new = QuantLinear(child, 8, GROUP)                       # shared embedding: 8 bits, as RTN copies
            elif full in skipped or not hasattr(child, "weight_scale"):
                new = QuantLinear(child, bits, GROUP)
            else:
                w, s = child.weight.float(), child.weight_scale.float()   # s: (rows, groups)
                rows, groups = s.shape
                r = w.reshape(rows, groups, -1) / s.unsqueeze(-1)
                q = r.round()
                worst = max(worst, float((r - q).abs().max()))
                new = QuantLinear.from_ints(q, s.unsqueeze(-1), child.bias, bits, group or w.shape[1])
            setattr(module, child_name, new)
    gpt.transformer.wte = QuantEmbedding(gpt.lm_head)
    gpt.quantized_bits, gpt.quant_method = bits, "gptq"
    if worst > 1e-3:
        raise RuntimeError(f"GPTQ weights aren't integer × scale (off by up to {worst:.3g}): unexpected format")
    return gpt


def quantize(model_id, scheme="W4A16", samples=256, seq_len=512, data=DEFAULT_DATA, root=None):
    import registry
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier
    from transformers import AutoTokenizer

    from inference import load_model, load_model_config
    from quantization.checkpoint import add_copy, original_of
    from quantization.hf_wrapper import TinyGPTForCausalLM

    root = root or registry.LIB_ROOT
    meta = original_of(model_id, root)
    weights, config_path = registry.paths_of(model_id, root)
    config, _ = load_model_config(config_path)
    gpt = load_model(config, weights, "cpu").float()
    bits = SCHEMES[scheme]
    group = GROUP if bits == 4 else None
    rec = recipe(scheme, samples, seq_len, data)
    seq_len = min(seq_len, config.block_size)
    skipped = unfit_layers(gpt, group)
    rec["calibration"]["seq_len"], rec["rounded_instead"] = seq_len, skipped

    say(f"reading {samples} calibration windows of {seq_len} tokens from {data}/train.bin")
    from datasets import Dataset
    windows, total = calibration_windows(data, samples, seq_len)
    rec["calibration"]["file_tokens"] = total
    ds = Dataset.from_dict({"input_ids": windows, "attention_mask": [[1] * seq_len] * len(windows)})

    say(f"running GPTQ ({scheme}) block by block on the CPU")
    t0 = time.time()
    oneshot(model=TinyGPTForCausalLM.from_gpt(gpt), tokenizer=AutoTokenizer.from_pretrained("gpt2"), dataset=ds,
            recipe=GPTQModifier(scheme=scheme, targets="Linear", ignore=["re:.*lm_head"] + skipped),
            max_seq_length=seq_len, num_calibration_samples=len(windows), pipeline="sequential",
            sequential_targets=["Block"])
    rec["seconds"] = round(time.time() - t0, 1)

    say("converting to the library's quantized format")
    new = add_copy(model_id, to_quantized_gpt(gpt, bits, group, skipped), rec, root)
    say(f"saved as {new['name']} ({new['bytes'] / 1e6:.0f} MB, from {meta['bytes'] / 1e6:.0f} MB)")
    return new


def main():
    ap = argparse.ArgumentParser(description="GPTQ (llm-compressor oneshot) on a library model; saves a quantized copy.")
    ap.add_argument("model_id")
    ap.add_argument("--scheme", choices=list(SCHEMES), default="W4A16")
    ap.add_argument("--samples", type=int, default=256)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--data", default=DEFAULT_DATA, help="data/<key>/train.bin to calibrate on")
    a = ap.parse_args()
    import logging
    logging.getLogger("llmcompressor").setLevel(logging.WARNING)
    new = quantize(a.model_id, a.scheme, a.samples, a.seq_len, a.data)
    print("@@RESULT " + json.dumps({"id": new["id"], "name": new["name"], "bytes": new["bytes"]}), flush=True)


if __name__ == "__main__":
    main()

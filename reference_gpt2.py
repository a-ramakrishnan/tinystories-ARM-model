# OpenAI's GPT-2 small (124M, 2019) as a reference model: its weights, loaded into this project's GPT class and
# saved to the model library, so every model trained here can be measured against it with the evaluation suite.
#
#   python3 reference_gpt2.py               download, convert, check, save to the library and evaluate
#   python3 reference_gpt2.py --fineweb     also prepare Mini FineWeb-Edu's held-out split on this Mac (a 1.3 GB
#                                           download), so the evaluation includes it
#
# GPT-2 uses the same tokenizer as this project and the GPT-2-style architecture its code already supports
# (learned positions, LayerNorm, GELU, biases, weight tying), so the weights map one to one. Two details:
# OpenAI stored its linear layers transposed (a "Conv1D"), and GPT-2's GELU is the tanh approximation.
# Its context window is 1,024 tokens; the evaluation suite reads 256-token windows, the same as our models.

import argparse
import json
import os
import sys
import time

import torch

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

import registry  # noqa: E402
from inference import save_model_config  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402

REPO = "openai-community/gpt2"
NAME = "OpenAI GPT-2 small (124M, 2019)"
FOLDER = os.path.join(PROJECT_DIR, "outputs", "reference", "gpt2")


def gpt2_config():
    return GPTConfig(block_size=1024, vocab_size=50257, n_layer=12, n_head=12, n_embd=768, dropout=0.0, bias=True,
                     pos_encoding="learned", norm="layernorm", mlp="gelu", gelu_approx="tanh")


def convert(hf):
    """OpenAI's state dict (Hugging Face names) → this project's GPT names."""
    out = {"transformer.wte.weight": hf["wte.weight"], "transformer.wpe.weight": hf["wpe.weight"],
           "transformer.ln_f.weight": hf["ln_f.weight"], "transformer.ln_f.bias": hf["ln_f.bias"]}
    names = {"ln_1": "ln1", "ln_2": "ln2", "attn.c_attn": "attn.c_attn", "attn.c_proj": "attn.c_proj",
             "mlp.c_fc": "mlp.c_fc", "mlp.c_proj": "mlp.c_proj"}
    for i in range(12):
        for theirs, ours in names.items():
            w = hf[f"h.{i}.{theirs}.weight"]
            # Conv1D stores (in, out); nn.Linear wants (out, in)
            out[f"transformer.h.{i}.{ours}.weight"] = w.t().contiguous() if theirs not in ("ln_1", "ln_2") else w
            out[f"transformer.h.{i}.{ours}.bias"] = hf[f"h.{i}.{theirs}.bias"]
    out["lm_head.weight"] = hf["wte.weight"]  # tied, as in GPT-2
    return out


def load_gpt2():
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(REPO, "pytorch_model.bin")
    hf = torch.load(path, map_location="cpu", weights_only=True)
    hf = {k.removeprefix("transformer."): v for k, v in hf.items()}
    model = GPT(gpt2_config())
    missing, unexpected = model.load_state_dict(convert(hf), strict=False)
    missing = [k for k in missing if not k.endswith(".attn.bias")]  # a causal-mask buffer, not a weight
    assert not missing and not unexpected, (missing, unexpected)
    return model.eval()


@torch.no_grad()
def check(model):
    """Two sanity checks: GPT-2's well-known greedy continuation, and a sensible loss on ordinary English."""
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    ids = enc.encode("Hello, I'm a language model,")
    x = torch.tensor([ids])
    for _ in range(12):
        logits, _ = model(x)
        x = torch.cat([x, logits[:, -1:].argmax(-1)], 1)
    text = enc.decode(x[0].tolist())
    print("Greedy:", repr(text))
    sample = enc.encode("The quick brown fox jumps over the lazy dog. " * 3 +
                        "In 1905, Albert Einstein published four papers that changed physics.")
    t = torch.tensor([sample])
    _, loss = model(t[:, :-1], t[:, 1:])
    print(f"Loss on a sample sentence: {loss.item():.3f}")
    return text


def prepare_fineweb_validation():
    """Mini FineWeb-Edu's held-out split on this Mac (the same split, seed and tokenizer as a Colab run), so the
    evaluation suite can measure every model on it. Only validation.bin is written: training needs the cloud."""
    from datasets import DatasetDict
    from data import dataset_paths, get_dataset
    from tokenization import tokenize_and_save
    paths = dataset_paths("BaoMinhTran/mini-fineweb-edu", "small", os.path.join(PROJECT_DIR, "data"))
    if os.path.exists(paths.val_bin):
        print("Mini FineWeb-Edu's validation split is already tokenized.")
        return
    ds = get_dataset(paths.hf_dir, "BaoMinhTran/mini-fineweb-edu", config_name="small")
    tokenize_and_save(DatasetDict({"validation": ds["validation"]}), paths.root, num_proc=4)
    print(f"Wrote {paths.val_bin}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fineweb", action="store_true", help="also prepare Mini FineWeb-Edu's held-out split here")
    ap.add_argument("--no-eval", action="store_true")
    a = ap.parse_args()
    if a.fineweb:
        prepare_fineweb_validation()
    print(f"Downloading {REPO} …")
    model = load_gpt2()
    n = sum(p.numel() for p in model.parameters())
    print(f"Loaded: {n:,} parameters")
    check(model)
    os.makedirs(FOLDER, exist_ok=True)
    weights, cfg_path = os.path.join(FOLDER, "best_model_params.pt"), os.path.join(FOLDER, "model_config.json")
    torch.save(model.state_dict(), weights)
    save_model_config(cfg_path, model.config, {"datasets": ["WebText (OpenAI)"], "params": n, "trained_at": "2019-02-14 00:00",
                                               "chip": "OpenAI", "platform": "reference", "model_type": "autoregressive"})
    meta, added = registry.add_model(weights, cfg_path, source="reference", name=NAME, origin=f"huggingface.co/{REPO}")
    registry.save_training(meta["id"], {
        "experiment": "Reference", "variant": "gpt2", "description": NAME, "datasets": ["WebText (OpenAI, ~40 GB of web pages)"],
        "chip": "OpenAI", "params": n, "active_params": n, "block": 1024,
        "note": "Trained by OpenAI on WebText with a 1,024-token context; evaluated here on the same 256-token windows as our models."})
    print(f"{'Saved' if added else 'Already'} in the library as {meta['name']!r} ({meta['id']})")
    if not a.no_eval:
        import evaluate
        from train_config import setup_device
        dev = setup_device()
        print("Evaluating (held-out loss on every dataset here, generation, size) …")
        t0 = time.time()
        ev = evaluate.evaluate_model(meta["id"], dev.device)
        print(json.dumps({k: round(v["loss"], 3) for k, v in ev["heldout"].items()}), f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()

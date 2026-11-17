"""Quantized checkpoints: save a quantized model (4-bit integers packed two per byte), load it back, and add a quantized
copy of a library model to the library. Used by the app (Models and Quantize pages) and inference.load_model.

    python3 -m quantization.checkpoint <model id> --bits 4     # a round-to-nearest 4-bit copy in the library
"""

import json
import os
import tempfile

import torch

from quantization.rtn import QuantLinear, quantize_model

# The file keeps the usual name (best_model_params.pt) so every loader finds it, but holds a dict:
#   {"quantized": {"format": 1, "method": "rtn", "bits", "group", "embed_bits"}, "state_dict": {...}}
# 4-bit integers are packed two per byte on disk, so the file is as small as weight_bytes says.

FORMAT = 1


def _pack4(q):
    """int8 values in −8…7 → uint8, two per byte (the last dimension must be even)."""
    u = (q.to(torch.int16) & 0xF).to(torch.uint8).reshape(*q.shape[:-1], -1, 2)
    return u[..., 0] | (u[..., 1] << 4)


def _unpack4(p, shape):
    lo, hi = (p & 0xF).to(torch.int16), (p >> 4).to(torch.int16)
    q = torch.stack([lo, hi], dim=-1).reshape(shape)
    return torch.where(q > 7, q - 16, q).to(torch.int8)


def quant_info(model):
    """What quantize_model did to this model (None if it's a float model)."""
    layers = [m for m in model.modules() if isinstance(m, QuantLinear)]
    if not layers:
        return None
    embed = model.lm_head.bits if isinstance(getattr(model, "lm_head", None), QuantLinear) else None
    return {"format": FORMAT, "method": getattr(model, "quant_method", "rtn"), "bits": model.quantized_bits,
            "group": layers[0].group, "embed_bits": embed}


def save_quantized(model, path):
    """Save a model returned by quantize_model; the 4-bit matrices packed two per byte."""
    info = quant_info(model)
    sd = {}
    for k, v in model.state_dict().items():
        if k.endswith(".q"):
            layer = model.get_submodule(k[:-2])
            if layer.bits <= 4 and v.shape[-1] % 2 == 0:
                sd[k + "4"] = _pack4(v.cpu())
                sd[k + "4_shape"] = torch.tensor(v.shape)
                continue
        sd[k] = v.cpu()
    torch.save({"quantized": info, "state_dict": sd}, path)
    return info


def is_quantized_checkpoint(obj):
    return isinstance(obj, dict) and isinstance(obj.get("quantized"), dict) and "state_dict" in obj


def load_quantized(config, obj, device):
    """Rebuild a quantized model from what save_quantized wrote (already read with torch.load)."""
    from model import GPT
    info = obj["quantized"]
    if info.get("format") != FORMAT:
        raise ValueError(f"Unknown quantized-model format {info.get('format')}")
    model = GPT(config)
    quantize_model(model, bits=info["bits"], group=info["group"], embed_bits=info.get("embed_bits") or 8)
    sd = {}
    for k, v in obj["state_dict"].items():
        if k.endswith(".q4_shape"):
            continue
        if k.endswith(".q4"):
            sd[k[:-1]] = _unpack4(v, tuple(obj["state_dict"][k + "_shape"].tolist()))
        else:
            sd[k] = v
    model.load_state_dict(sd)
    model.quant_method = info.get("method", "rtn")
    return model.to(device).eval()


def original_of(model_id, root=None):
    """The library meta of a model that can be quantized (autoregressive, not quantized already)."""
    import registry
    meta = registry.get_model(model_id, root or registry.LIB_ROOT)
    if meta.get("quantized"):
        raise ValueError("This model is already quantized: quantize the original instead.")
    if meta.get("model_type", "autoregressive") != "autoregressive":
        raise ValueError("Only autoregressive models can be quantized so far.")
    return meta


def label_of(q):
    """'4-bit, embedding 8-bit' or 'GPTQ 4-bit, embedding 8-bit'."""
    s = ("GPTQ " if q["method"] == "gptq" else "") + f"{q['bits']}-bit"
    return s + (f", embedding {q['embed_bits']}-bit" if q.get("embed_bits") and q["embed_bits"] != q["bits"] else "")


def add_copy(model_id, model, recipe=None, root=None):
    """Save a quantized model (from quantize_model, or GPTQ converted to the same layers) as a new library model
    next to its original. recipe: what made it, kept in meta.json for the Quantize page. Returns the new meta."""
    import registry
    from inference import load_model_config
    root = root or registry.LIB_ROOT
    meta = original_of(model_id, root)
    _, info = load_model_config(registry.paths_of(model_id, root)[1])
    with tempfile.TemporaryDirectory() as tmp:
        w, c = os.path.join(tmp, "best_model_params.pt"), os.path.join(tmp, "model_config.json")
        q = save_quantized(model, w)
        with open(c, "w") as f:
            json.dump({**info, "quantized": q, "quantized_from": model_id}, f, indent=2)
        new, added = registry.add_model(w, c, source="quantized", name=f"{meta['name']} · {label_of(q)}",
                                          origin=model_id, root=root)
    new.update(quantized=q, quantized_from=model_id, recipe=recipe or new.get("recipe"))
    with open(os.path.join(registry.model_dir(new["id"], root), registry.META), "w") as f:
        json.dump(new, f, indent=2)
    return new


def rtn_recipe(bits, embed_bits=8, group=64):
    return {"method": "rtn", "bits": bits, "embed_bits": min(embed_bits, 8), "group": group}


def quantize_library_model(model_id, bits=8, embed_bits=8, group=64, say=print, root=None):
    """Round-to-nearest copy of a saved model, added to the library. Returns its meta."""
    import registry
    from inference import load_model, load_model_config
    root = root or registry.LIB_ROOT
    meta = original_of(model_id, root)
    weights, config_path = registry.paths_of(model_id, root)
    config, _ = load_model_config(config_path)
    say(f"rounding every weight to {bits} bits")
    model = quantize_model(load_model(config, weights, "cpu"), bits=bits, group=group, embed_bits=min(embed_bits, 8))
    new = add_copy(model_id, model, rtn_recipe(bits, embed_bits, group), root)
    say(f"saved as {new['name']} ({new['bytes'] / 1e6:.0f} MB, from {meta['bytes'] / 1e6:.0f} MB)")
    return new


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Save a round-to-nearest quantized copy of a library model.")
    ap.add_argument("model_id")
    ap.add_argument("--bits", type=int, choices=(4, 8), default=8)
    ap.add_argument("--embed-bits", type=int, choices=(4, 8), default=8,
                    help="bits for the shared embedding / output layer (8 is safer, see quantize_model)")
    a = ap.parse_args()
    quantize_library_model(a.model_id, a.bits, a.embed_bits)

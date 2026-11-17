# Quantized copies (quantization/): packing, saving and loading back, and adding one to a library, on tiny CPU models.
import json
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from quantization import checkpoint, gptq, rtn  # noqa: E402
import registry  # noqa: E402
from inference import load_model  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402

TINY = dict(block_size=32, vocab_size=96, n_layer=2, n_head=2, n_embd=32, bias=False, pos_encoding="rope",
            norm="rmsnorm", mlp="swiglu", qk_norm=True)


def test_pack4_round_trip():
    q = torch.randint(-8, 8, (5, 3, 16), dtype=torch.int8)
    p = checkpoint._pack4(q)
    assert p.dtype == torch.uint8 and p.shape == (5, 3, 8)       # two per byte
    assert torch.equal(checkpoint._unpack4(p, q.shape), q)


@pytest.mark.parametrize("bits,embed_bits", [(8, 8), (4, 8), (4, 4)])
def test_save_and_load_give_the_same_model(tmp_path, bits, embed_bits):
    torch.manual_seed(0)
    config = GPTConfig(**TINY)
    q = rtn.quantize_model(GPT(config).eval(), bits=bits, group=16, embed_bits=embed_bits)
    path = str(tmp_path / "best_model_params.pt")
    info = checkpoint.save_quantized(q, path)
    assert info == {"format": 1, "method": "rtn", "bits": bits, "group": 16, "embed_bits": embed_bits}
    back = load_model(config, path, "cpu")                        # the usual loader recognises the file
    assert checkpoint.quant_info(back) == info
    x = torch.randint(0, 96, (2, 32))
    assert torch.equal(q(x, all_logits=True)[0], back(x, all_logits=True)[0])


def test_quantized_copy_in_a_library(tmp_path):
    torch.manual_seed(0)
    root, src = str(tmp_path / "models"), tmp_path / "run"
    src.mkdir()
    torch.save(GPT(GPTConfig(**TINY)).state_dict(), src / "best_model_params.pt")
    (src / "model_config.json").write_text(json.dumps({"model": TINY, "datasets": ["roneneldan/TinyStories"],
                                                       "trained_at": "2026-11-17 10:00"}))
    meta, _ = registry.add_model(str(src / "best_model_params.pt"), str(src / "model_config.json"), root=root)
    new = checkpoint.quantize_library_model(meta["id"], bits=4, root=root, say=lambda m: None)
    assert new["source"] == "quantized" and new["quantized_from"] == meta["id"]
    assert new["quantized"]["bits"] == 4 and new["name"].endswith("· 4-bit, embedding 8-bit")
    assert new["bytes"] < meta["bytes"] / 2.5
    assert registry.get_model(new["id"], root)["quantized"]["bits"] == 4      # kept in meta.json
    with pytest.raises(ValueError):
        checkpoint.quantize_library_model(new["id"], root=root, say=lambda m: None)   # not twice


def test_gptq_layers_convert_to_our_format(tmp_path):
    """After llm-compressor's GPTQ, each Linear holds weight = integer × scale and a weight_scale per group of 128;
    to_quantized_gpt turns that into our QuantLinear, which saves, loads and computes the same thing."""
    torch.manual_seed(0)
    config = GPTConfig(**{**TINY, "n_embd": 128, "n_head": 2})   # 128 wide: the block matrices fit groups of 128
    gpt = GPT(config).eval()
    skipped = gptq.unfit_layers(gpt, gptq.GROUP)
    assert skipped == [f"transformer.h.{i}.mlp.c_proj" for i in range(2)]      # SwiGLU's 352-wide input
    for name, m in gpt.named_modules():                      # what GPTQ leaves behind (simulated)
        if isinstance(m, torch.nn.Linear) and name != "lm_head" and name not in skipped:
            w = m.weight.data.reshape(m.out_features, -1, 128)
            s = w.abs().amax(-1) / 7
            m.weight.data = ((w / s.unsqueeze(-1)).round().clamp(-8, 7) * s.unsqueeze(-1)).reshape(m.weight.shape)
            m.weight_scale = torch.nn.Parameter(s, requires_grad=False)
    x = torch.randint(0, 96, (2, 32))
    q = gptq.to_quantized_gpt(gpt, 4, gptq.GROUP, skipped)
    assert checkpoint.quant_info(q) == {"format": 1, "method": "gptq", "bits": 4, "group": 128, "embed_bits": 8}
    path = str(tmp_path / "best_model_params.pt")
    checkpoint.save_quantized(q, path)
    back = load_model(config, path, "cpu")
    assert checkpoint.quant_info(back)["method"] == "gptq"
    assert torch.equal(q(x, all_logits=True)[0], back(x, all_logits=True)[0])

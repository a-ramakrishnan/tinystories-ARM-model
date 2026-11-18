# S1: the Qwen3 export computes the same function as our GPT (serving/export_hf.py), on tiny random models: the base
# design with LayerNorm (folded into RMSNorm by centering the residual stream) and with RMSNorm, and with grouped-query
# attention. Needs transformers; skipped without it.
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

os.environ.setdefault("USE_TF", "0")     # transformers would import TensorFlow if installed (it hangs on the Mac)
pytest.importorskip("transformers")

from model import GPT, GPTConfig  # noqa: E402
from serving.export_hf import CHAT_TEMPLATE, build_hf, check_supported  # noqa: E402

BASE = dict(block_size=32, vocab_size=96, n_layer=2, n_head=4, n_embd=64, bias=False, pos_encoding="rope",
            norm="layernorm", mlp="swiglu", qk_norm=True)


def random_gpt(**kw):
    torch.manual_seed(0)
    m = GPT(GPTConfig(**{**BASE, **kw}))
    with torch.no_grad():           # norms away from 1, so their weights (and the queries' small scale) matter
        for n, p in m.named_parameters():
            if n.endswith("norm.weight") or ".ln" in n:
                p.uniform_(0.5, 1.5)
    return m.eval()


@pytest.mark.parametrize("kw", [{}, {"norm": "rmsnorm"}, {"n_kv_head": 2}])
def test_export_matches(kw):
    m = random_gpt(**kw)
    hf = build_hf(m)
    ids = torch.randint(0, 96, (2, 24))
    with torch.no_grad():
        ours = m(ids, all_logits=True)[0]
        theirs = hf(ids).logits
    assert (ours - theirs).abs().max().item() < 1e-4
    assert hf.config.tie_word_embeddings == (m.config.norm == "rmsnorm")


def test_unsupported_designs_are_refused():
    with pytest.raises(SystemExit):
        check_supported(GPTConfig(**{**BASE, "pos_encoding": "learned"}))
    with pytest.raises(SystemExit):
        check_supported(GPTConfig(**{**BASE, "causal": False}))


def test_chat_template_matches_sft():
    """The Jinja template gives the text sft.py trains on (the token ids were checked against sft.py's on real conversations)."""
    from jinja2 import Template
    t = Template(CHAT_TEMPLATE)
    msgs = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi "},
            {"role": "assistant", "content": "Hello!"}, {"role": "user", "content": "Bye"}]
    assert t.render(messages=msgs, add_generation_prompt=True) == \
        "<|endoftext|>System: Be brief.\n\nUser: Hi\n\nAssistant: Hello!<|endoftext|>User: Bye\n\nAssistant:"

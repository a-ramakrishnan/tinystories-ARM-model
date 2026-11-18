# S7: the llama.cpp adapter (serving/check.py) gives the evaluation code what our GPT gives it. A stand-in for
# llama-cpp-python returns a tiny GPT's logits, so the benchmark scoring through the adapter must equal scoring the
# GPT directly; the answer statistics match sft.py's repetition measure.
import os
import sys
import types

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from model import GPT, GPTConfig  # noqa: E402


class FakeTokenizer:
    def encode_ordinary(self, s):
        return [ord(c) % 96 for c in s]


def tiny():
    torch.manual_seed(0)
    return GPT(GPTConfig(block_size=64, vocab_size=96, n_layer=2, n_head=2, n_embd=32, bias=False, pos_encoding="rope",
                         norm="layernorm", mlp="swiglu", qk_norm=True)).eval()


def test_adapter_scores_like_the_model(monkeypatch):
    gpt = tiny()

    class FakeLlama:                    # llama_cpp.Llama's surface that the adapter uses
        def __init__(self, **kw):
            self.scores = None

        def reset(self):
            self.scores = None

        def eval(self, tokens):
            with torch.no_grad():
                self.scores = gpt(torch.tensor([tokens]), all_logits=True)[0][0].numpy()

        def close(self):
            pass

    monkeypatch.setitem(sys.modules, "llama_cpp", types.SimpleNamespace(Llama=FakeLlama))
    from benchmarks import score_requests
    from serving.check import LlamaCppModel
    m = LlamaCppModel("fake.gguf", n_ctx=64)
    reqs = [("The cat sat on", " the mat"), ("A", " b c d"), ("Once upon a time there", " was")]
    a = score_requests(gpt, FakeTokenizer(), reqs, "cpu", batch=2)
    b = score_requests(m, FakeTokenizer(), reqs, "cpu", batch=2)
    for (la, ga, na), (lb, gb, nb) in zip(a, b):
        assert abs(la - lb) < 1e-4 and ga == gb and na == nb


def test_answer_stats():
    from serving.check import answer_stats
    s = answer_stats([[1, 2, 3, 4, 1, 2, 3, 4], [5] * 10], cap=10)
    assert s["n"] == 2 and s["ended"] == 0.5 and s["mean_length"] == 9.0
    assert np.isclose(s["repeated_4grams"], ((5 - 4) / 5 + (7 - 1) / 7) / 2, atol=1e-4)

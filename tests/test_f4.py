# F4 (diffusion_models/compare.py): the held-out bound on fixed windows, and the summary tables, on tiny CPU models.
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from diffusion_models import compare  # noqa: E402
from diffusion_models.config import DiffusionLMConfig  # noqa: E402
from diffusion_models.model import build  # noqa: E402


def tiny(tmp_path):
    torch.manual_seed(0)
    model = build(DiffusionLMConfig(seq_len=32, d_model=32, n_layers=2, n_heads=2, d_ff=64, diffusion_steps=8, arch="gpt")).eval()
    path = str(tmp_path / "validation.bin")
    np.random.RandomState(0).randint(0, 200, 5000).astype(np.uint16).tofile(path)
    return model, path


def test_heldout_bound_is_repeatable_and_near_uniform_for_an_untrained_model(tmp_path):
    model, path = tiny(tmp_path)
    a = compare.heldout_bound(model, path, "cpu", n_windows=4, context=32, levels=8, batch=8)
    b = compare.heldout_bound(model, path, "cpu", n_windows=4, context=32, levels=8, batch=5)
    assert a == b                                     # fixed windows, levels and masks: batch size doesn't matter
    assert a["context"] == 32 and a["levels"] == 8
    # an untrained model guesses roughly uniformly over the vocabulary: the bound is about log(vocab)
    assert abs(a["bound"] - np.log(model.cfg.vocab_size)) < 1.0


def test_markdown_tables():
    r = {"pair": {"title": "tiny", "ar": "a", "diffusion": "d"},
         "heldout": {"x": {"title": "X", "ar_loss": 3.0, "diffusion_bound": 3.8, "gap": 0.8}},
         "benchmarks": {t: {"title": t, "metric": "acc", "ar": .5, "diffusion": .4, "chance": .25, "gpt2": .5}
                        for t in ("a", "b", "c", "d", "e")},
         "writing": {"tokens": 64, "prompts": 8, "seeds": [1, 2],
                     "real continuation": {"judge_loss": 3.0, "repeated_4grams": 0.0, "distinct_2": 1.0},
                     "settings": {"AR": {"passes": 64, "judge_loss": 2.5, "repeated_4grams": .01, "distinct_2": .9, "tokens_per_sec": 80}}}}
    md = compare.markdown("tiny", r)
    assert "| X | 3.000 | 3.800 | +0.800 |" in md and "**Average** | **50.0** | **40.0**" in md and "| AR | 64 | 2.500 |" in md

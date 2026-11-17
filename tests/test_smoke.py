# J1: fast smoke tests, on the CPU, with tiny models: a few seconds in all. They check that the pieces still fit
# together (the model builds and learns, the KV cache doesn't change what's generated, the diffusion model trains and
# samples, the job runner reports), not how good anything is. Run with:  python3 -m pytest -q
import json
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from model import GPT, GPTConfig  # noqa: E402

torch.manual_seed(0)

# the base model's design (RoPE, RMSNorm, SwiGLU, no biases, QK-norm), tiny
TINY = dict(block_size=32, vocab_size=96, n_layer=2, n_head=2, n_embd=32, bias=False, pos_encoding="rope",
            norm="rmsnorm", mlp="swiglu", qk_norm=True)


def tiny_gpt(**kw):
    return GPT(GPTConfig(**{**TINY, **kw}))


@pytest.mark.parametrize("kw", [{}, {"pos_encoding": "learned", "norm": "layernorm", "mlp": "gelu", "bias": True,
                                     "qk_norm": False}])
def test_gpt_learns_a_pattern(kw):
    """GPT-2 style and the base model's design both drive the loss down on a repeating sequence."""
    model = tiny_gpt(**kw)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    seq = torch.arange(33) % 7
    x, y = seq[:-1][None].repeat(4, 1), seq[1:][None].repeat(4, 1)
    first = None
    for _ in range(40):
        _, loss = model(x, y)
        first = first if first is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first * 0.5, (first, loss.item())


@pytest.mark.parametrize("pos", ["rope", "learned"])
def test_kv_cache_gives_the_same_tokens(pos):
    """Greedy generation with and without the KV cache picks identical tokens, past the context window too."""
    from inference import generate_stream
    model = tiny_gpt(pos_encoding=pos).eval()
    prompt = [1, 5, 9, 2]
    run = lambda kv: [s["id"] for s in generate_stream(model, prompt, "cpu", max_new_tokens=40, temperature=0.0,
                                                       use_kv_cache=kv)]
    assert run(True) == run(False)


def test_bidirectional_attention_sees_the_future():
    """causal=False (the diffusion denoiser) lets an early position depend on a later token; causal doesn't."""
    for causal in (True, False):
        model = tiny_gpt(causal=causal).eval()
        a = torch.tensor([[3, 4, 5, 6]])
        b = a.clone()
        b[0, -1] = 7
        la, _ = model(a, a)          # with targets: logits for every position
        lb, _ = model(b, b)
        changed = not torch.allclose(la[0, 0], lb[0, 0])
        assert changed == (not causal)


@pytest.mark.parametrize("arch", ["v1", "gpt"])
def test_diffusion_model_trains_and_samples(arch):
    from diffusion_models.config import DiffusionLMConfig
    from diffusion_models.model import build
    from diffusion_models.noising import diffusion_loss
    from diffusion_models.sample import diffusion_generate
    cfg = DiffusionLMConfig(seq_len=32, d_model=32, n_layers=2, n_heads=2, d_ff=64, diffusion_steps=8, arch=arch)
    model = build(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = torch.randint(0, 200, (4, 32))
    loss, _ = diffusion_loss(model, batch, cfg.diffusion_steps, cfg.mask_id)
    opt.zero_grad()
    loss.backward()
    opt.step()
    assert torch.isfinite(loss)
    r = diffusion_generate(model, prompt_ids=[10, 11, 12], max_new_tokens=12, diffusion_steps=4, seed=1,
                           record_steps=False)
    assert len(r["ids"]) == 12
    assert cfg.mask_id not in r["ids"]


def test_job_runner_reports_and_restarts(tmp_path):
    """infra.colab_job: progress lines become status.json, a failed command is restarted, `show` summarizes."""
    from infra import colab_job
    flag = tmp_path / "ran-once"
    script = ("import sys, os\n"
              "print('update 1,000/4,000  loss 4.5000  lr 1e-04  9,000 tokens/s', flush=True)\n"
              "print('  eval @ 1,000: masked-token loss 4.6 | likelihood bound 3.9 nats/token', flush=True)\n"
              f"flag = {str(flag)!r}\n"
              "sys.exit(0 if os.path.exists(flag) else (open(flag, 'w').close() or 3))\n")
    code = colab_job.run([sys.executable, "-c", script], str(tmp_path), "test", retries=1, retry_wait=0, echo=False)
    st = json.load(open(tmp_path / "status.json"))
    assert code == 0 and st["state"] == "finished" and st["attempt"] == 2
    assert (st["step"], st["total"], st["loss"], st["tokens_per_sec"]) == (1000, 4000, 4.5, 9000)
    assert "likelihood bound 3.9" in st["last_eval"]
    assert "finished" in colab_job.show(str(tmp_path))


def test_job_runner_flags_a_stale_run(tmp_path):
    from infra import colab_job
    json.dump({"job": "x", "state": "running", "updated_ts": 1000, "updated_at": "then", "attempt": 1},
              open(tmp_path / "status.json", "w"))
    assert "stale" in colab_job.show(str(tmp_path), now=1000 + 3600)


def test_tqdm_lines_are_parsed():
    from infra.colab_job import parse
    st = {}
    assert parse(" 12%|#   | 18000/152600 [29:01<3:33:10, 10.52it/s]", st)
    assert (st["step"], st["total"], st["steps_per_sec"]) == (18000, 152600, 10.52)

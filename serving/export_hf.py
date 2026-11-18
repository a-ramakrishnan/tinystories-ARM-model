# S1: export one of our GPTs as a standard Hugging Face Qwen3 model, so llama.cpp, LM Studio, Ollama, MLX, vLLM and
# transformers serve it with no code of ours.
#
# Our base design is Qwen3's in everything but one: RoPE (rotate-half), SwiGLU, no biases, QK-norm (a per-head
# RMSNorm before RoPE), but the block norms are LayerNorm (without a bias) where Qwen3 has RMSNorm. LayerNorm is
# RMSNorm after subtracting the vector's mean, so if the residual stream always has zero mean the two are the same
# function. The export makes it so, exactly:
#   - every input embedding row is centered (its mean subtracted);
#   - every matrix that writes into the residual stream (attention out, feed-forward down) has its output centered;
#   - LayerNorm ignores a constant added to every feature, so the network computes the same thing at every layer,
#     and the residual stream it carries is the original one minus its mean;
#   - the output layer keeps the original (uncentered) embedding matrix, so the input and output embeddings are no
#     longer tied (38.6M more numbers in the file; the same model).
#   - Qwen3 has one epsilon for every norm; ours are 1e-5 (LayerNorm) and 1e-6 (QK-norm). The queries are tiny before
#     QK-norm (mean square ~1e-5), so the epsilon matters there. RMSNorm(s·x, ε) = RMSNorm(x, ε / s²): the export keeps
#     ε = 1e-5 and multiplies the query and key projections by s = √(1e-5 / 1e-6) = √10, which makes QK-norm behave
#     exactly as with 1e-6 (only the normalized direction leaves the norm, so nothing else changes).
# verify() checks it: the same logits to float32 rounding, the same greedy text.
#
#   python3 -m serving.export_hf <model id> [--out exports/<name>] [--dtype bfloat16] [--no-verify]
import argparse
import json
import os
import shutil
import time

# transformers imports TensorFlow when it's installed, and on the Mac's system Python that import hangs: PyTorch only
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")

import torch  # noqa: E402

from inference import load_model, load_model_config  # noqa: E402
from model import swiglu_hidden  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")
EOT = 50256  # GPT-2's <|endoftext|>: our documents' separator and the chat models' end of answer

# The chat models' format (sft.py, "smoltalk-plain-v1") as a Jinja chat template, so clients that send messages get
# exactly the tokens the model was fine-tuned on: <|endoftext|>, then "User: …", a blank line, "Assistant: …<|endoftext|>"
CHAT_TEMPLATE = (
    "{{ '<|endoftext|>' }}"
    "{% for m in messages %}"
    "{% set role = m['role'] if m['role'] in ['system', 'user', 'assistant'] else 'user' %}"
    "{% if not loop.first and messages[loop.index0 - 1]['role'] != 'assistant' %}{{ '\n\n' }}{% endif %}"
    "{{ role | capitalize }}:{{ ' ' + m['content'] | trim }}"
    "{% if role == 'assistant' %}{{ '<|endoftext|>' }}{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{% if messages and messages[-1]['role'] != 'assistant' %}{{ '\n\n' }}{% endif %}"
    "{{ 'Assistant:' }}"
    "{% endif %}"
)


def resolve(model):
    """A model id in models/, or a folder with best_model_params.pt and model_config.json."""
    d = model if os.path.isdir(model) else os.path.join(MODELS, model)
    if not os.path.exists(os.path.join(d, "model_config.json")):
        raise SystemExit(f"No model {model!r} (looked in {d})")
    return d


def check_supported(c):
    problems = []
    if c.pos_encoding != "rope":
        problems.append("learned positions (Qwen3 uses RoPE)")
    if c.mlp != "swiglu":
        problems.append("a GELU feed-forward layer (Qwen3 uses SwiGLU)")
    if c.bias:
        problems.append("biases")
    if not c.qk_norm:
        problems.append("no QK-norm")
    if c.block_repeat > 1 or c.n_experts > 1 or c.mtp_heads > 1 or not c.causal:
        problems.append("shared blocks, experts, extra prediction heads or bidirectional attention")
    if problems:
        raise SystemExit("This model doesn't map onto Qwen3: " + "; ".join(problems))


QK_EPS = 1e-6       # model.py's RMSNorm default, used by QK-norm
LN_EPS = 1e-5       # model.py's LayerNorm


def to_qwen3(model, eps=None):
    """(Qwen3Config, state dict) computing the same function as our GPT (see the top of this file). eps: Qwen3's one
    epsilon (default: our block norms'); the query and key projections are scaled so QK-norm still acts with 1e-6."""
    from transformers import Qwen3Config
    c = model.config
    check_supported(c)
    hd, kv = c.n_embd // c.n_head, c.n_kv_head or c.n_head
    sd = {k: v.detach().float().cpu() for k, v in model.state_dict().items()}
    layernorm = c.norm != "rmsnorm"
    eps = eps or (LN_EPS if layernorm else QK_EPS)
    qk_scale = (eps / QK_EPS) ** 0.5
    eot = EOT if c.vocab_size > EOT else c.vocab_size - 1   # tiny test vocabularies have no <|endoftext|>
    center_rows = (lambda w: w - w.mean(dim=1, keepdim=True)) if layernorm else (lambda w: w)   # each embedding vector
    center_out = (lambda w: w - w.mean(dim=0, keepdim=True)) if layernorm else (lambda w: w)    # each output column
    wte = sd["transformer.wte.weight"]
    out = {"model.embed_tokens.weight": center_rows(wte), "lm_head.weight": wte.clone(),
           "model.norm.weight": sd["transformer.ln_f.weight"]}
    for i in range(c.n_layer):
        p, q = f"transformer.h.{i}.", f"model.layers.{i}."
        wq, wk, wv = sd[p + "attn.c_attn.weight"].split([c.n_embd, kv * hd, kv * hd], dim=0)
        gate, up = sd[p + "mlp.c_fc.weight"].chunk(2, dim=0)
        out.update({
            q + "input_layernorm.weight": sd[p + "ln1.weight"],
            q + "self_attn.q_proj.weight": wq * qk_scale, q + "self_attn.k_proj.weight": wk * qk_scale,
            q + "self_attn.v_proj.weight": wv,
            q + "self_attn.q_norm.weight": sd[p + "attn.q_norm.weight"],
            q + "self_attn.k_norm.weight": sd[p + "attn.k_norm.weight"],
            q + "self_attn.o_proj.weight": center_out(sd[p + "attn.c_proj.weight"]),
            q + "post_attention_layernorm.weight": sd[p + "ln2.weight"],
            q + "mlp.gate_proj.weight": gate, q + "mlp.up_proj.weight": up,
            q + "mlp.down_proj.weight": center_out(sd[p + "mlp.c_proj.weight"]),
        })
    cfg = Qwen3Config(vocab_size=c.vocab_size, hidden_size=c.n_embd, intermediate_size=swiglu_hidden(c.n_embd),
                      num_hidden_layers=c.n_layer, num_attention_heads=c.n_head, num_key_value_heads=kv, head_dim=hd,
                      hidden_act="silu", max_position_embeddings=c.block_size, rms_norm_eps=eps,
                      rope_theta=c.rope_theta, attention_bias=False, use_sliding_window=False,
                      tie_word_embeddings=not layernorm, bos_token_id=eot, eos_token_id=eot, pad_token_id=eot,
                      torch_dtype="float32")
    if not layernorm:
        del out["lm_head.weight"]
    return cfg, out


def build_hf(model, eps=None):
    from transformers import Qwen3ForCausalLM
    cfg, sd = to_qwen3(model, eps)
    hf = Qwen3ForCausalLM(cfg)
    missing, unexpected = hf.load_state_dict(sd, strict=False)
    missing = [m for m in missing if not (cfg.tie_word_embeddings and m == "lm_head.weight")]
    assert not missing and not unexpected, (missing, unexpected)
    return hf.eval()


def gpt2_tokenizer(chat):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tok.pad_token = tok.eos_token
    tok.model_max_length = 512
    if chat:
        tok.chat_template = CHAT_TEMPLATE
    return tok


@torch.no_grad()
def verify(model, hf, texts, enc):
    """Run our GPT and the export on the same tokens (float32, CPU): the largest logit difference, whether every
    position's top token agrees, and greedy generation token for token."""
    model, hf = model.float().cpu().eval(), hf.float().cpu().eval()
    worst, agree, total = 0.0, 0, 0
    for t in texts:
        ids = torch.tensor([enc.encode_ordinary(t)[:model.config.block_size]])
        ours = model(ids, all_logits=True)[0][0]
        theirs = hf(ids).logits[0]
        worst = max(worst, (ours - theirs).abs().max().item())
        agree += (ours.argmax(-1) == theirs.argmax(-1)).sum().item()
        total += ids.shape[1]
    prompt = torch.tensor([enc.encode_ordinary(texts[0])[:32]])
    x = prompt
    for _ in range(40):
        x = torch.cat([x, model(x)[0][:, -1].argmax(-1, keepdim=True)], dim=1)
    g = hf.generate(prompt, max_new_tokens=40, do_sample=False, pad_token_id=EOT, attention_mask=torch.ones_like(prompt))
    return {"max_logit_diff": round(worst, 6), "top1_agree": f"{agree}/{total}",
            "greedy_same": x[0].tolist() == g[0].tolist(), "greedy_tokens": 40}


def sample_texts(n=4):
    """A few held-out texts (the evaluation suite's prompts with their real continuations)."""
    import textgen
    return [p["prompt"] + p["reference"] for p in textgen.heldout_prompts()[:n]]


def export(model_id, out=None, dtype="bfloat16", run_verify=True, say=print):
    d = resolve(model_id)
    config, info = load_model_config(os.path.join(d, "model_config.json"))
    model = load_model(config, os.path.join(d, "best_model_params.pt"), "cpu").float()
    name = os.path.basename(os.path.normpath(d))
    out = out or os.path.join(ROOT, "exports", f"{name}-qwen3")
    chat = bool(info.get("chat"))
    t0 = time.time()
    hf = build_hf(model)
    report = None
    if run_verify:
        from tokenization import get_tokenizer
        report = verify(model, hf, sample_texts(), get_tokenizer())
        say(f"  checked on {report['top1_agree'].split('/')[1]} held-out tokens: largest logit difference "
            f"{report['max_logit_diff']}, top token agrees {report['top1_agree']}, "
            f"greedy text {'identical' if report['greedy_same'] else 'DIFFERENT'}")
    if os.path.isdir(out):
        shutil.rmtree(out)
    hf.to(getattr(torch, dtype))
    hf.config.torch_dtype = dtype
    hf.generation_config.update(bos_token_id=EOT, eos_token_id=EOT, pad_token_id=EOT, do_sample=True,
                                temperature=0.8, top_p=0.9, max_length=config.block_size)
    hf.save_pretrained(out, safe_serialization=True)
    gpt2_tokenizer(chat).save_pretrained(out)
    meta = {"source_model": name, "source_config": info["model"], "chat": chat, "chat_format": info.get("chat_format"),
            "exported_at": time.strftime("%Y-%m-%d %H:%M"), "dtype": dtype, "verify": report,
            "how": "serving/export_hf.py: LayerNorm folded into RMSNorm by centering the residual stream; Q and K scaled by sqrt(10) for QK-norm's epsilon"}
    with open(os.path.join(out, "export_info.json"), "w") as f:
        json.dump(meta, f, indent=1)
    size = sum(os.path.getsize(os.path.join(out, x)) for x in os.listdir(out))
    say(f"exported {name} → {os.path.relpath(out, ROOT)} ({size / 1e6:.0f} MB, {dtype}, {time.time() - t0:.0f} s)"
        + ("; chat template included" if chat else ""))
    return out, meta


def main():
    ap = argparse.ArgumentParser(description="Export a model as a standard Hugging Face Qwen3 model.")
    ap.add_argument("model", help="a model id in models/ or a folder")
    ap.add_argument("--out")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--no-verify", action="store_true")
    a = ap.parse_args()
    export(a.model, a.out, a.dtype, not a.no_verify)


if __name__ == "__main__":
    main()

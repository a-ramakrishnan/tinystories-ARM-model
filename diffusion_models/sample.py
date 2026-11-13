# Generation by denoising (v1's diffusion_generate): start with the prompt followed by [MASK]s, then for
# t = T, T-1, ..., 1:
#   1. the model predicts every position; a token is sampled at each position not fixed by the prompt;
#   2. the least confident ceil(n * (t-1)/T) of the generated positions are masked again.
# So the text starts fully masked and ends fully written, the most confident tokens settling first. As in v1,
# already-written positions are re-sampled at each step too, so a token can still change until the end.
#
# Every step is recorded (tokens, which are masked, the model's confidence), for the GIF and the visualizations.
#
#   python3 -m diffusion_models.sample diffusion_models/runs/<run> --prompt "Once upon a time"

import argparse
import math
import time

import torch
import torch.nn.functional as F

from .config import GPT2_VOCAB
from .tokens import EOT_ID, MASK_ID, decode, encode, token_texts


@torch.no_grad()
def diffusion_generate(model, prompt_text="", max_new_tokens=128, diffusion_steps=None, temperature=1.0, top_k=0,
                       record_steps=True, seed=None):
    model.eval()
    device = next(model.parameters()).device
    cfg = model.cfg
    T = diffusion_steps or cfg.diffusion_steps
    if seed is not None:
        torch.manual_seed(seed)
    prompt_ids = [EOT_ID] + encode(prompt_text)          # <|endoftext|> marks the start of a document
    prompt_ids = prompt_ids[-(cfg.seq_len - 1):]
    Lp = len(prompt_ids)
    L = min(cfg.seq_len, Lp + max_new_tokens)
    gen_len = L - Lp

    x = torch.full((1, L), MASK_ID, dtype=torch.long, device=device)
    x[0, :Lp] = torch.tensor(prompt_ids, device=device)
    fixed = torch.zeros((1, L), dtype=torch.bool, device=device)
    fixed[0, :Lp] = True
    gen = torch.arange(L, device=device) >= Lp

    frames, trace = [], []
    t0 = time.perf_counter()
    for s in range(T, 0, -1):
        t = torch.tensor([s], device=device, dtype=torch.long)
        logits = model(x, timesteps=t).float()
        logits[..., GPT2_VOCAB:] = float("-inf")          # never write [MASK] or the unused padding ids
        if temperature != 1.0:
            logits = logits / max(temperature, 1e-5)
        if top_k and top_k > 0:
            v, _ = torch.topk(logits, top_k, dim=-1)
            logits = logits.masked_fill(logits < v[..., -1:], float("-inf"))
        probs = F.softmax(logits, dim=-1)
        sampled = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(1, L)
        conf = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)   # [1, L]
        x[~fixed] = sampled[~fixed]

        # mask the least confident generated positions again, keeping ceil(n * (s-1)/T) masked
        target_masks = int(math.ceil(gen_len * (s - 1) / T))
        cand = torch.where(gen & ~fixed[0])[0]
        remasked = []
        if target_masks > 0 and cand.numel() > 0:
            _, low = torch.topk(conf[0, cand], k=min(target_masks, cand.numel()), largest=False)
            remasked = cand[low]
            x[0, remasked] = MASK_ID

        if record_steps:
            ids = x[0, Lp:].tolist()
            frames.append((s, decode(x[0].tolist()[1:])))
            trace.append({"t": s, "ids": ids, "masked": [i == MASK_ID for i in ids],
                          "conf": [round(c, 4) for c in conf[0, Lp:].tolist()],
                          "ms": round((time.perf_counter() - t0) * 1000, 1)})
    if device.type == "mps":
        torch.mps.synchronize()
    out = x[0, Lp:].tolist()
    if EOT_ID in out:                                     # the document ended: drop what follows
        out = out[:out.index(EOT_ID)]
    return {"prompt": prompt_text, "text": prompt_text + decode(out), "completion": decode(out),
            "tokens": token_texts(x[0, Lp:].tolist()), "steps": T, "gen_len": gen_len,
            "seconds": round(time.perf_counter() - t0, 3), "frames": frames, "trace": trace}


def main():
    from train_config import setup_device

    from .model import load

    ap = argparse.ArgumentParser(description="Generate text with a trained diffusion model.")
    ap.add_argument("run_dir")
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--steps", type=int, default=None, help="denoising steps (default: the model's T)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--gif", default=None, help="also save the denoising steps as a GIF")
    a = ap.parse_args()
    model = load(a.run_dir, setup_device().device)
    r = diffusion_generate(model, a.prompt, a.tokens, a.steps, a.temperature, a.top_k, seed=a.seed)
    print(r["text"])
    print(f"\n({r['gen_len']} tokens in {r['steps']} steps, {r['seconds']} s)")
    if a.gif:
        from .render import make_gif
        print("Saved", make_gif(r["frames"], a.gif, prompt=a.prompt, total=r["steps"]))


if __name__ == "__main__":
    main()

# Generation by denoising: start with the prompt followed by [MASK]s, then for t = T, T-1, ..., 1 the model
# predicts every position and a token is sampled at each one; after the step, ceil(n * (t-1)/T) positions stay
# masked. So the text starts fully masked and ends fully written, the most confident tokens settling first.
#
# Two ways of choosing which tokens to keep (`sampler`):
#   "keep" (default): a revealed token stays fixed; each step reveals the most confident of the still-masked
#       positions (the usual masked-diffusion sampler, as in MDLM / LLaDA).
#   "random": a revealed token stays fixed; each step reveals masked positions chosen at random (MDLM's
#       ancestral sampler). On the FineWeb-Edu model, "keep" collapsed into one repeated topic word ("fish, fish,
#       fish, ..."): the most confident guess at every masked position is the same frequent word. Random order
#       with temperature 0.8 gives varied text there; on TinyStories "keep" reads better.
#   "v1": v1's diffusion_generate: every generated position is re-sampled at every step, then the least
#       confident are masked again. Tokens can change until the end, but the last step re-draws all of them
#       independently, which breaks the text up: on the 51M TinyStories model the middle steps read fluently and
#       the final text didn't; with "keep" the final text is coherent stories.
#
# The defaults (confident first, top-k 50, repetition penalty 1.3) come from diffusion_models/quality.py, which
# scores the text on held-out prompts (14 Nov 2026): without the penalty, confident-first sampling copies words
# into neighbouring positions (on FineWeb-Edu 43% of 4-token runs were repeats; 4% with it, real text 0%). Blocks
# (semi-autoregressive), re-masking and left-to-right order were tried too; see the README.
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


def _filter(logits, temperature, top_k, top_p):
    if temperature != 1.0:
        logits = logits / max(temperature, 1e-5)
    if top_k and top_k > 0:
        v, _ = torch.topk(logits, top_k, dim=-1)
        logits = logits.masked_fill(logits < v[..., -1:], float("-inf"))
    if top_p and top_p < 1.0:
        sorted_logits, idx = torch.sort(logits, descending=True, dim=-1)
        cum = torch.softmax(sorted_logits, -1).cumsum(-1)
        drop = cum - torch.softmax(sorted_logits, -1) > top_p        # keep the smallest set reaching top_p
        sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, idx, sorted_logits)
    return logits


@torch.no_grad()
def diffusion_generate(model, prompt_text="", max_new_tokens=128, diffusion_steps=None, temperature=1.0, top_k=50,
                       record_steps=True, seed=None, sampler="keep", top_p=1.0, block_size=0,
                       repetition_penalty=1.3, remask=0.0, time_from="schedule", prompt_ids=None,
                       distinct=False):
    """Denoise `max_new_tokens` after the prompt in `diffusion_steps` steps (default: the model's T).

    sampler: "keep" (reveal the most confident masked positions), "random" (reveal at random) or "v1".
    block_size: 0 fills the whole continuation at once; N fills it N tokens at a time, left to right, each block
        getting an equal share of the steps (semi-autoregressive, as in LLaDA).
    top_p: nucleus sampling on top of top_k / temperature.
    repetition_penalty: > 1 makes tokens already in the text less likely at every masked position (as in CTRL).
        1.3 by default: without it, confident-first sampling keeps copying a word next to it ("water, water, ...").
    remask: at every step, this share of the block's revealed tokens with the lowest probability under the current
        prediction are masked again and re-predicted (ReMDM-style); revealed tokens can then still change.
    time_from: the noise level given to the model: "schedule" (the step's place in the schedule, as in training's
        t/T) or "mask" (the share of the whole input still masked).
    distinct: within a step, a token id is revealed at most at one position (its most confident one); the other
        positions that picked it wait for a later step. Confident-first sampling otherwise reveals the same
        frequent word at many positions at once ("water, water, water").
    prompt_ids: token ids instead of prompt_text (<|endoftext|> is put in front either way).
    """
    model.eval()
    device = next(model.parameters()).device
    cfg = model.cfg
    T = diffusion_steps or cfg.diffusion_steps
    if seed is not None:
        torch.manual_seed(seed)
    ids_in = list(prompt_ids) if prompt_ids is not None else encode(prompt_text)
    prompt_text = prompt_text if prompt_ids is None else decode(ids_in)
    prompt_ids = ([EOT_ID] + ids_in)[-(cfg.seq_len - 1):]   # <|endoftext|> marks the start of a document
    Lp = len(prompt_ids)
    L = min(cfg.seq_len, Lp + max_new_tokens)
    gen_len = L - Lp

    x = torch.full((1, L), MASK_ID, dtype=torch.long, device=device)
    x[0, :Lp] = torch.tensor(prompt_ids, device=device)
    pos = torch.arange(L, device=device)

    # the schedule: (block start, block end, steps for the block)
    bs = block_size if block_size and block_size < gen_len else gen_len
    blocks = [(Lp + i, min(L, Lp + i + bs)) for i in range(0, gen_len, bs)]
    per = [max(1, T // len(blocks))] * len(blocks)
    for i in range(T - sum(per)):
        per[i % len(blocks)] += 1
    total = sum(per)

    frames, trace = [], []
    t0 = time.perf_counter()
    step = total
    for (b0, b1), k in zip(blocks, per):
        in_block = (pos >= b0) & (pos < b1)
        blen = b1 - b0
        for j in range(k, 0, -1):
            n_masked = int((x[0] == MASK_ID).sum())
            if time_from == "mask":
                level = math.ceil(cfg.diffusion_steps * n_masked / L)
            else:
                level = round(step * cfg.diffusion_steps / total)
            t = torch.tensor([max(1, min(cfg.diffusion_steps, level))], device=device, dtype=torch.long)
            logits = model(x, timesteps=t).float()
            logits[..., GPT2_VOCAB:] = float("-inf")      # never write [MASK] or the unused padding ids
            if repetition_penalty != 1.0:
                present = torch.unique(x[0, 1:][x[0, 1:] != MASK_ID])
                sub = logits[0][:, present]
                logits[0][:, present] = torch.where(sub > 0, sub / repetition_penalty, sub * repetition_penalty)
            if remask > 0 or sampler == "v1":
                cur_p = torch.softmax(logits, -1).gather(-1, x.clamp(max=GPT2_VOCAB - 1).unsqueeze(-1)).squeeze(-1)
            probs = F.softmax(_filter(logits, temperature, top_k, top_p), dim=-1)
            sampled = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(1, L)
            conf = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)   # [1, L]
            target = int(math.ceil(blen * (j - 1) / k))     # the block's positions still masked after this step
            if sampler == "v1":
                # re-sample every generated position, then mask the least confident again
                gen = pos >= Lp
                x[0, gen] = sampled[0, gen]
                cand = torch.where(gen & in_block)[0]
                if target > 0:
                    _, low = torch.topk(conf[0, cand], k=min(target, cand.numel()), largest=False)
                    x[0, cand[low]] = MASK_ID
            else:
                if remask > 0 and j > 1:
                    done = torch.where(in_block & (x[0] != MASK_ID))[0]
                    r = int(remask * done.numel())
                    if r > 0:
                        low = done[torch.topk(cur_p[0, done], k=r, largest=False).indices]
                        x[0, low] = MASK_ID
                cand = torch.where(in_block & (x[0] == MASK_ID))[0]
                reveal = cand.numel() - target
                if reveal > 0:
                    if sampler == "random":
                        top = cand[torch.randperm(cand.numel(), device=device)[:reveal]]
                    elif distinct:
                        order = cand[torch.argsort(conf[0, cand], descending=True)]
                        seen, top = set(), []
                        for i, tok in zip(order.tolist(), sampled[0, order].tolist()):
                            if tok not in seen:
                                seen.add(tok)
                                top.append(i)
                                if len(top) == reveal:
                                    break
                        if len(top) < reveal and j == 1:      # the last step must fill everything
                            top = cand.tolist()
                        top = torch.tensor(top, device=device)
                    else:
                        top = cand[torch.topk(conf[0, cand], k=reveal).indices]
                    x[0, top] = sampled[0, top]
            if record_steps:
                ids = x[0, Lp:].tolist()
                frames.append((step, decode(x[0].tolist()[1:])))
                trace.append({"t": step, "ids": ids, "masked": [i == MASK_ID for i in ids],
                              "conf": [round(c, 4) for c in conf[0, Lp:].tolist()],
                              "ms": round((time.perf_counter() - t0) * 1000, 1)})
            step -= 1
    if device.type == "mps":
        torch.mps.synchronize()
    out = x[0, Lp:].tolist()
    ids_out = out[:out.index(EOT_ID)] if EOT_ID in out else out   # the document ended: drop what follows
    return {"sampler": sampler, "prompt": prompt_text, "text": prompt_text + decode(ids_out),
            "completion": decode(ids_out), "ids": ids_out, "tokens": token_texts(out), "steps": total,
            "gen_len": gen_len, "seconds": round(time.perf_counter() - t0, 3), "frames": frames, "trace": trace}


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
    ap.add_argument("--repetition-penalty", type=float, default=1.3)
    ap.add_argument("--block-size", type=int, default=0, help="fill N tokens at a time, left to right (0: all at once)")
    ap.add_argument("--sampler", default="keep", choices=["keep", "random", "v1"],
                    help="keep: reveal the most confident first (default); random: reveal in random order; v1: v1's re-sampling")
    ap.add_argument("--gif", default=None, help="also save the denoising steps as a GIF")
    a = ap.parse_args()
    model = load(a.run_dir, setup_device().device)
    r = diffusion_generate(model, a.prompt, a.tokens, a.steps, a.temperature, a.top_k, seed=a.seed, sampler=a.sampler,
                           repetition_penalty=a.repetition_penalty, block_size=a.block_size)
    print(r["text"])
    print(f"\n({r['gen_len']} tokens in {r['steps']} steps, {r['seconds']} s)")
    if a.gif:
        from .render import make_gif
        print("Saved", make_gif(r["frames"], a.gif, prompt=a.prompt, total=r["steps"]))


if __name__ == "__main__":
    main()

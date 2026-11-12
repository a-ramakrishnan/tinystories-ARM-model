# Inference: load trained weights and generate text from a prompt

import json
import math
import time
from dataclasses import asdict

import torch
import torch.nn.functional as F

from model import GPT, GPTConfig, apply_rope, rope_cos_sin, effective_layers, truncate_kv_caches


def save_model_config(path, config, extra=None):
    """Write the model config (plus any training summary in `extra`) next to the checkpoint."""
    with open(path, "w") as f:
        json.dump({"model": asdict(config), **(extra or {})}, f, indent=2)


def load_model_config(path):
    """Read a file written by save_model_config. Returns (GPTConfig, full dict)."""
    with open(path) as f:
        info = json.load(f)
    return GPTConfig(**info["model"]), info


def load_model(config, weights_path, device):
    """Rebuild the GPT from `config`, load saved weights, and put it in eval mode on `device`."""
    model = GPT(config)
    model.load_state_dict(torch.load(weights_path, map_location=torch.device(device)))
    return model.to(device).eval()


def generate_text(model, enc, prompt, device, max_new_tokens=200, temperature=1.0, top_k=None):
    """Continue `prompt` with up to `max_new_tokens` tokens and return the decoded text."""
    context = torch.tensor(enc.encode_ordinary(prompt), dtype=torch.long, device=device).unsqueeze(0)
    out = model.generate(context, max_new_tokens, temperature=temperature, top_k=top_k)
    ids = out[0].tolist()
    new = ids[context.size(1):]
    if enc.eot_token in new:  # the story ended: drop <|endoftext|> and anything after it
        ids = ids[:context.size(1) + new.index(enc.eot_token)]
    return enc.decode(ids)


# ---------- sampling: how the next token is picked ----------
# Order (the usual one): penalties -> temperature -> top-k -> top-p -> min-p, then sample (or argmax when greedy).

SAMPLING_DEFAULTS = {"top_p": 1.0, "min_p": 0.0, "repetition_penalty": 1.0, "frequency_penalty": 0.0,
                     "presence_penalty": 0.0, "typical_p": 1.0, "dry_multiplier": 0.0, "dry_base": 1.75,
                     "dry_allowed_length": 2}
PENALTY_KEYS = ("repetition_penalty", "frequency_penalty", "presence_penalty", "dry_multiplier")  # need the history


def dry_penalty(history, multiplier=0.8, base=1.75, allowed_length=2):
    """DRY ("don't repeat yourself", p-e-w 2024): penalize the tokens that would continue a sequence already seen.

    For each earlier place where the text matched its own ending, the token that came next there would repeat
    that passage. If the match is n tokens long and n >= allowed_length, that token's logit is lowered by
    multiplier × base^(n − allowed_length): short echoes (a name, "of the") are left alone, and the longer a
    repeat would get, the harder it is pushed away. Unlike a repetition penalty it doesn't punish single words
    for appearing again, only sequences. history: a list of token ids. Returns {token id: penalty}.
    """
    n = len(history)
    best = {}
    if multiplier <= 0 or n < 2:
        return best
    last = history[-1]
    for j in range(n - 1):              # history[j + 1] followed a place that may match the ending
        if history[j] != last:
            continue
        m = 1
        while m <= j and history[j - m] == history[n - 1 - m]:
            m += 1
        if m >= allowed_length:
            t = history[j + 1]
            best[t] = max(best.get(t, 0), m)
    return {t: multiplier * base ** (m - allowed_length) for t, m in best.items()}


def apply_penalties(logits, history, repetition_penalty=1.0, frequency_penalty=0.0, presence_penalty=0.0,
                    need_counts=True):
    """Discourage tokens that already appear in `history` (1-D tensor of token ids).

    repetition_penalty (CTRL-style): divides a seen token's logit by the penalty if positive, multiplies if
    negative, so it always becomes less likely. frequency_penalty subtracts penalty × (times seen);
    presence_penalty subtracts a flat amount from every token seen at least once (OpenAI-style).
    Returns (new logits, counts); counts is None when there are no penalties and need_counts is False.
    """
    if repetition_penalty == 1.0 and frequency_penalty == 0.0 and presence_penalty == 0.0 and not need_counts:
        return logits, None  # nothing to do (generation skips counting, to save GPU work on every token)
    counts = torch.zeros_like(logits).scatter_add_(0, history, torch.ones_like(history, dtype=logits.dtype))
    if repetition_penalty == 1.0 and frequency_penalty == 0.0 and presence_penalty == 0.0:
        return logits, counts
    seen = counts > 0
    if repetition_penalty != 1.0:
        logits = torch.where(seen, torch.where(logits > 0, logits / repetition_penalty, logits * repetition_penalty), logits)
    return logits - frequency_penalty * counts - presence_penalty * seen.to(logits.dtype), counts


def sampling_distribution(logits, history, temperature=1.0, top_k=None, top_p=1.0, min_p=0.0,
                          repetition_penalty=1.0, frequency_penalty=0.0, presence_penalty=0.0, typical_p=1.0,
                          dry_multiplier=0.0, dry_base=1.75, dry_allowed_length=2, stages=False):
    """Turn one position's logits (1-D) into the distribution the next token is drawn from.

    temperature <= 0 means greedy: all probability on the single most likely token (after penalties).
    top_k keeps the k most likely tokens; top_p keeps the smallest set whose probability adds up to p
    (nucleus sampling); min_p drops tokens less likely than min_p × the top token's probability; typical_p
    (Meier et al. 2022) keeps the tokens whose surprise is closest to the distribution's average surprise (its
    entropy), up to typical_p of the probability: it drops both the very likely, bland choices and the unlikely
    ones. dry_multiplier > 0 adds the DRY penalty against repeating earlier sequences (see dry_penalty).
    Returns (probs, n_kept as a 0-d tensor), or with stages=True a dict with every intermediate step (for the
    explorer).
    """
    logits = logits.float()
    penalized, counts = apply_penalties(logits, history, repetition_penalty, frequency_penalty, presence_penalty,
                                       need_counts=stages)
    if dry_multiplier > 0 and history is not None:
        pen = dry_penalty(history.tolist(), dry_multiplier, dry_base, int(dry_allowed_length))
        if pen:
            idx = torch.tensor(list(pen), dtype=torch.long, device=logits.device)
            val = torch.tensor(list(pen.values()), dtype=penalized.dtype, device=logits.device)
            penalized = penalized.index_put((idx,), -val, accumulate=True)
    greedy = temperature <= 0
    scaled = penalized / (1.0 if greedy else temperature)
    p = F.softmax(scaled, dim=-1)
    cut_k = torch.zeros_like(p, dtype=torch.bool)
    cut_p = torch.zeros_like(cut_k)
    cut_min = torch.zeros_like(cut_k)
    cut_typ = torch.zeros_like(cut_k)
    if greedy:
        keep = torch.zeros_like(cut_k).scatter_(0, torch.argmax(p).view(1), True)
        cut_k = ~keep   # reported as "greedy" by callers
    else:
        if top_k:
            kth = torch.topk(p, min(int(top_k), p.numel())).values[-1]
            cut_k = p < kth
        if top_p < 1.0:
            q = p.masked_fill(cut_k, 0)
            sp, si = torch.sort(q, descending=True)
            before = torch.cumsum(sp, 0) - sp            # probability of the tokens ranked above this one
            drop = before >= top_p * q.sum()
            cut_p = torch.zeros_like(cut_k).scatter_(0, si, drop) & ~cut_k
        if min_p > 0.0:
            cut_min = (p < min_p * p.max()) & ~cut_k & ~cut_p
        if typical_p < 1.0:
            q = p.masked_fill(cut_k | cut_p | cut_min, 0)
            q = q / q.sum()
            surprise = -torch.log(q.clamp_min(1e-30))
            entropy = (q * surprise).sum()
            dist = (surprise - entropy).abs().masked_fill(q == 0, float("inf"))
            sd, si = torch.sort(dist)
            before = torch.cumsum(q[si], 0) - q[si]       # probability of the more typical tokens
            drop = before >= typical_p
            cut_typ = torch.zeros_like(cut_k).scatter_(0, si, drop) & ~(cut_k | cut_p | cut_min)
        keep = ~(cut_k | cut_p | cut_min | cut_typ)
    final = p.masked_fill(~keep, 0)
    final = final / final.sum()
    if not stages:
        return final, keep.sum()  # n_kept stays on the GPU (a 0-d tensor): reading it would wait for the GPU
    n_kept = int(keep.sum())
    return {"raw": F.softmax(logits, dim=-1), "penalized": F.softmax(penalized, dim=-1), "scaled": p, "final": final,
            "counts": counts, "cut_k": cut_k, "cut_p": cut_p, "cut_min": cut_min, "cut_typ": cut_typ, "greedy": greedy, "n_kept": n_kept,
            "kept_mass": float(p[keep].sum())}


def kv_cache_bytes(caches):
    """Memory held by a list of per-layer KV caches."""
    return sum(c[k].numel() * c[k].element_size() for c in caches for k in ("k", "v") if c.get(k) is not None)


def kv_bytes_per_token(config, dtype_bytes=4):
    """Each token stores a key and a value vector in every layer: n_embd numbers each, or n_kv_head · head_dim
    with grouped-query attention."""
    kv_heads = getattr(config, "n_kv_head", 0) or config.n_head
    return 2 * effective_layers(config) * kv_heads * (config.n_embd // config.n_head) * dtype_bytes


def device_memory(device):
    """GPU memory in use right now (bytes), plus the peak where the backend tracks it."""
    device = str(device)
    if device.startswith("cuda"):
        return {"allocated": torch.cuda.memory_allocated(), "peak": torch.cuda.max_memory_allocated(),
                "reserved": torch.cuda.memory_reserved(), "total": torch.cuda.get_device_properties(0).total_memory,
                "name": torch.cuda.get_device_name(0)}
    if device == "mps":
        return {"allocated": torch.mps.current_allocated_memory(), "driver": torch.mps.driver_allocated_memory(),
                "peak": None, "name": "Apple GPU (MPS)"}
    return {"allocated": None, "peak": None, "name": "CPU"}


def _sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    elif str(device) == "mps":
        torch.mps.synchronize()


@torch.no_grad()
def generate_stream(model, prompt_ids, device, max_new_tokens=200, temperature=1.0, top_k=None, n_alternatives=5,
                    use_kv_cache=True, stats=None, stop_token=None, sampling=None):
    """Like GPT.generate, but yields each new token as soon as it's chosen, with timing and probabilities.

    stop_token: stop when the model picks this token (e.g. <|endoftext|>, the end of a story); it isn't yielded.
    sampling: optional top_p, min_p, repetition_penalty, frequency_penalty, presence_penalty (see
    sampling_distribution). temperature <= 0 picks the most likely token every time (greedy).

    Yields dicts: {"id", "prob", "alternatives": [(id, prob), ...], "ms"} where `prob` is the model's own
    probability for the chosen token (before temperature / top-k), and `ms` is the time since the start.

    With use_kv_cache, the keys and values of earlier tokens are kept, so each step only runs the new token
    through the model. Without it, every step re-reads the whole context (up to block_size tokens). This model
    uses learned absolute positions, so once the context window is full the positions shift and the cache has
    to be rebuilt ("recompute"). `stats`, if given, is filled with cache size and recompute counts.
    """
    block = model.config.block_size
    # the tokens so far live in a Python list: growing a GPU tensor every step (torch.cat) waits for the GPU
    ids = list(prompt_ids)
    as_input = lambda toks: torch.tensor([toks], dtype=torch.long, device=device)
    sampling = {**SAMPLING_DEFAULTS, **(sampling or {})}
    penalized = any(sampling[k] != SAMPLING_DEFAULTS[k] for k in PENALTY_KEYS)
    sliding = getattr(model.config, "pos_encoding", "learned") == "rope"
    caches, cache_len, pos, recomputes = None, 0, 0, 0
    t0 = time.perf_counter()
    for _ in range(max_new_tokens):
        if use_kv_cache:
            # Learned positions are tied to slots 0..block_size-1, so a full window means rebuilding the cache
            # with every position shifted. RoPE models just slide: the oldest key/value is overwritten.
            if caches is None or (not sliding and cache_len + 1 > block):
                if caches is not None:
                    recomputes += 1
                caches = [{"max_len": block} for _ in range(effective_layers(model.config))]
                ctx = as_input(ids[-block:])
                logits, _ = model(ctx, kv_caches=caches, start_pos=0)
                cache_len = pos = ctx.size(1)
            else:
                logits, _ = model(as_input(ids[-1:]), kv_caches=caches, start_pos=pos)
                pos += 1
                cache_len = min(cache_len + 1, block)
        else:
            logits, _ = model(as_input(ids[-block:]))
        logits = logits[0, -1, :]
        raw = F.softmax(logits.float(), dim=-1)
        history = torch.tensor(ids[-block:], dtype=torch.long, device=device) if penalized else None
        probs, n_kept = sampling_distribution(logits, history, temperature, top_k, **sampling)
        idx_next = (torch.argmax(probs) if temperature <= 0 else torch.multinomial(probs, num_samples=1)).view(1, 1)
        alt_p, alt_i = torch.topk(raw, n_alternatives)
        # one transfer from the GPU per token instead of several (each read waits for the GPU): token, its
        # probability, how many tokens were kept, then the alternatives (ids < 2^24 are exact in float32)
        packed = torch.cat((idx_next.view(1).float(), raw.gather(0, idx_next.view(1)), n_kept.view(1).float(),
                            alt_i.float(), alt_p)).tolist()  # waits for the GPU, so the timing below is real
        token, prob, n_kept = int(packed[0]), packed[1], int(packed[2])
        alt_ids, alt_probs = packed[3:3 + n_alternatives], packed[3 + n_alternatives:]
        if stop_token is not None and token == stop_token:
            if stats is not None:
                stats["stopped"] = True
            return
        if stats is not None:
            stats.update(kv_cache_bytes=kv_cache_bytes(caches) if caches else 0, cache_tokens=cache_len if caches else 0,
                         recomputes=recomputes, sliding=sliding)
        yield {"id": token, "prob": prob, "kept": n_kept,
               "alternatives": list(zip([int(i) for i in alt_ids], [round(p, 4) for p in alt_probs])),
               "ms": (time.perf_counter() - t0) * 1000}
        ids.append(token)


def _last_hidden(model, idx):
    """The final hidden state (after ln_f) at every position, and the logits: (hidden (T, C), logits (T, V))."""
    out = {}
    h = model.transformer.ln_f.register_forward_hook(lambda m, i, o: out.__setitem__("h", o))
    try:
        logits, _ = model(idx, all_logits=True)
    finally:
        h.remove()
    return out["h"][0], logits[0]


@torch.no_grad()
def contrastive_search(model, prompt_ids, device, max_new_tokens=100, k=4, alpha=0.6, stop_token=None):
    """Contrastive search (Su et al. 2022): deterministic, like greedy, but without its loops.

    At each step take the k most likely tokens and pick the one with the best
        (1 − alpha) · p(token) − alpha · (highest cosine similarity of its hidden state to any earlier token's),
    so a token that's likely but would take the text somewhere it has already been loses to a slightly less
    likely one that says something new. alpha = 0 is greedy. Each step runs the k candidates through the model
    (here without a KV cache: k full passes over the context, so it's slow; fine for a comparison).
    """
    block = model.config.block_size
    ids = list(prompt_ids)
    out = []
    for _ in range(max_new_tokens):
        ctx = ids[-(block - 1):]
        hid, logits = _last_hidden(model, torch.tensor([ctx], device=device))
        probs = F.softmax(logits[-1].float(), dim=-1)
        top_p, top_i = torch.topk(probs, k)
        # each candidate appended to the context, all k in one batch; its hidden state is the last position's
        cand = torch.tensor([ctx + [int(t)] for t in top_i.tolist()], device=device)
        store = {}
        hook = model.transformer.ln_f.register_forward_hook(lambda m, i, o: store.__setitem__("h", o))
        try:
            model(cand, all_logits=True)
        finally:
            hook.remove()
        cand_h = F.normalize(store["h"][:, -1].float(), dim=-1)            # (k, C)
        prev_h = F.normalize(hid.float(), dim=-1)                          # (T, C)
        degeneration = (cand_h @ prev_h.T).max(dim=-1).values              # (k,)
        score = (1 - alpha) * top_p - alpha * degeneration
        token = int(top_i[int(score.argmax())])
        if stop_token is not None and token == stop_token:
            break
        ids.append(token); out.append(token)
    return out


@torch.no_grad()
def beam_search(model, prompt_ids, device, max_new_tokens=100, beams=4, length_penalty=1.0, no_repeat_ngram=0,
                stop_token=None):
    """Beam search: keep the `beams` most probable continuations so far and extend each by every token, keeping
    the best `beams` again. Finds a more probable text than greedy; famous for making repetitive text worse, which
    no_repeat_ngram (never repeat an n-gram, as in many summarization systems) counters. Score: total log
    probability / length^length_penalty. No KV cache here (each step reruns the context for every beam)."""
    block = model.config.block_size
    beams_ = [(0.0, [])]                         # (sum of log probs, new tokens)
    done = []
    for _ in range(max_new_tokens):
        ctxs = [(list(prompt_ids) + toks)[-block:] for _, toks in beams_]
        width = max(len(c) for c in ctxs)
        if any(len(c) != width for c in ctxs):   # beams are always the same length here; keep it simple
            raise RuntimeError("beams of different lengths")
        logits, _ = model(torch.tensor(ctxs, device=device))
        logp = F.log_softmax(logits[:, -1].float(), dim=-1)            # (B, V)
        if no_repeat_ngram > 0:
            n = no_repeat_ngram
            for b, (_, toks) in enumerate(beams_):
                seq = list(prompt_ids) + toks
                if len(seq) >= n - 1:
                    prefix = tuple(seq[len(seq) - (n - 1):]) if n > 1 else ()
                    banned = [seq[i + n - 1] for i in range(len(seq) - n + 1) if tuple(seq[i:i + n - 1]) == prefix]
                    if banned:
                        logp[b, banned] = float("-inf")
        total = torch.tensor([sc for sc, _ in beams_], device=logp.device)[:, None] + logp
        top_v, top_i = torch.topk(total.view(-1), beams)
        V = logp.size(-1)
        nxt = []
        for v, i in zip(top_v.tolist(), top_i.tolist()):
            b, t = divmod(i, V)
            toks = beams_[b][1] + [t]
            if stop_token is not None and t == stop_token:
                done.append((v, beams_[b][1]))
            else:
                nxt.append((v, toks))
        if not nxt:
            break
        beams_ = nxt
        if len(done) >= beams:
            break
    cands = done + beams_
    best = max(cands, key=lambda c: c[0] / max(1, len(c[1])) ** length_penalty)
    return best[1]


@torch.no_grad()
def speculative_generate(target, draft, prompt_ids, device, max_new_tokens=200, k=4, temperature=0.0, top_k=None,
                         seed=None):
    """Speculative decoding (Leviathan et al. 2023; Chen et al. 2023): a small `draft` model guesses k tokens one
    at a time, then the large `target` model checks all of them in a single pass. Every guess the target agrees
    with is kept, and the target's own token replaces the first one it rejects, so each pass of the target adds
    between 1 and k + 1 tokens. Both models must share the tokenizer.

    The output is exactly what the target alone would produce: with temperature <= 0, the same greedy tokens;
    otherwise, tokens drawn from the same distribution (a guess x is kept with probability min(1, p(x) / q(x)),
    p the target's and q the draft's probabilities; on a rejection, the token is drawn from max(0, p − q),
    renormalized). Generation stops at the context window (both caches would have to slide, and a slid cache
    can't be rolled back). Returns (token ids, stats).
    """
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)
    block = min(target.config.block_size, draft.config.block_size)
    ids = list(prompt_ids)[-block:]
    tensor = lambda toks: torch.tensor([toks], dtype=torch.long, device=device)
    greedy = temperature <= 0

    def dist(logits):  # (T, V) logits -> greedy: each row's best token id; else probabilities (CPU, float64)
        if greedy:
            return logits.argmax(dim=-1).tolist()   # only the ids leave the GPU
        logits = logits.double().cpu() / temperature
        if top_k:
            kth = torch.topk(logits, min(int(top_k), logits.size(-1)), dim=-1).values[:, -1:]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        return F.softmax(logits, dim=-1)

    def pick(p):
        return p if greedy else int(torch.multinomial(p, 1, generator=gen))

    tc = [{"max_len": block} for _ in range(effective_layers(target.config))]
    dc = [{"max_len": block} for _ in range(effective_layers(draft.config))]
    # prefill both with all but the last prompt token: the caches always hold every token except the newest one
    if len(ids) > 1:
        target(tensor(ids[:-1]), kv_caches=tc, start_pos=0)
        draft(tensor(ids[:-1]), kv_caches=dc, start_pos=0)
    t_len = d_len = len(ids) - 1           # tokens in each cache
    out, rounds, accepted, drafted = [], 0, 0, 0
    t0 = time.perf_counter()
    while len(out) < max_new_tokens:
        n = len(ids)
        kk = min(k, max_new_tokens - len(out) - 1, block - n)   # room left in the window for the checked tokens
        if kk < 0:
            break
        # 1. the draft guesses kk tokens (first feeding whatever it hasn't seen yet)
        guesses, q = [], []
        feed = ids[d_len:]
        for _ in range(kk):
            lg, _ = draft(tensor(feed), kv_caches=dc, start_pos=d_len)
            d_len += len(feed)
            qi = dist(lg[0, -1:])[0]
            g = pick(qi)
            guesses.append(g); q.append(qi); feed = [g]
        # 2. the target scores the newest token and every guess in one pass
        lg, _ = target(tensor([ids[-1]] + guesses), kv_caches=tc, start_pos=t_len, all_logits=True)
        p = dist(lg[0])                    # p[i]: the target's distribution for the token after guess i - 1
        # 3. keep guesses while the target agrees, then add the target's own token
        n_ok = 0
        for i, g in enumerate(guesses):
            if greedy:
                ok = p[i] == g
            else:
                ok = torch.rand(1, generator=gen).item() < min(1.0, float(p[i][g] / q[i][g]))
            if not ok:
                break
            n_ok += 1
        if n_ok < len(guesses):
            if greedy:
                nxt = p[n_ok]
            else:
                resid = (p[n_ok] - q[n_ok]).clamp(min=0)
                nxt = pick(resid / resid.sum())
        else:
            nxt = pick(p[len(guesses)])    # every guess kept: the target's next token comes for free
        new = guesses[:n_ok] + [nxt]
        ids += new; out += new
        # roll back what was rejected: both caches keep everything up to (not including) the newest token
        t_len = n + n_ok
        truncate_kv_caches(tc, t_len)
        d_len = min(d_len, t_len)
        truncate_kv_caches(dc, d_len)
        rounds += 1; accepted += n_ok; drafted += len(guesses)
    secs = time.perf_counter() - t0
    out = out[:max_new_tokens]
    return out, {"tokens": len(out), "seconds": secs, "tok_s": len(out) / secs if secs else 0.0, "rounds": rounds,
                 "acceptance": accepted / drafted if drafted else 0.0,
                 "tokens_per_target_pass": len(out) / rounds if rounds else 0.0}


def benchmark_generation(model, prompt_ids, device, n_tokens=100, seed=1234, repeats=2):
    """Generate the same tokens with and without the KV cache and compare speed and memory.

    Both runs use the same random seed, so they should produce the same text (tiny float differences can
    occasionally change a sampled token). Each mode runs `repeats` times and the fastest run is kept, which
    smooths out GPU timing noise.
    """
    results = {}
    for use_cache in (False, True):  # warm up both code paths (GPU kernels for each shape) so timing is fair
        for _ in generate_stream(model, prompt_ids, device, min(n_tokens, 30), use_kv_cache=use_cache):
            pass
    for use_cache in (False, True):
        best = None
        for _ in range(repeats):
            if str(device).startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()
            _sync(device)
            torch.manual_seed(seed)
            stats, times, ids = {}, [], []
            for ev in generate_stream(model, prompt_ids, device, n_tokens, temperature=1.0, top_k=50,
                                      use_kv_cache=use_cache, stats=stats):
                times.append(ev["ms"])
                ids.append(ev["id"])
            _sync(device)
            if best is None or times[-1] < best[0][-1]:
                best = (times, ids, stats)
        times, ids, stats = best
        decode_ms = times[-1] - times[0]
        results["kv_cache" if use_cache else "no_cache"] = {
            "ttft_ms": round(times[0], 1), "total_ms": round(times[-1], 1),
            "tokens_per_sec": round((len(times) - 1) / (decode_ms / 1000), 1) if decode_ms > 0 else None,
            "ms_per_token": round(decode_ms / (len(times) - 1), 2) if len(times) > 1 else None,
            "kv_cache_bytes": stats.get("kv_cache_bytes", 0), "recomputes": stats.get("recomputes", 0),
            "memory": device_memory(device), "ids": ids,
            # time for each token (the first includes the prompt's prefill), for per-position charts
            "token_ms": [round(b - a, 2) for a, b in zip([0.0] + times[:-1], times)],
        }
    a, b = results["no_cache"], results["kv_cache"]
    results["speedup"] = round(a["total_ms"] / b["total_ms"], 2) if b["total_ms"] else None
    results["same_output"] = a["ids"] == b["ids"]
    for r in (a, b):
        r.pop("ids")
    return results


@torch.no_grad()
def next_token_probs(model, prompt_ids, device, k=10):
    """The model's top-k candidates for the token after `prompt_ids`: [(id, probability), ...]."""
    idx = torch.tensor(prompt_ids[-model.config.block_size:], dtype=torch.long, device=device).unsqueeze(0)
    logits, _ = model(idx)
    p, i = torch.topk(F.softmax(logits[0, -1], dim=-1), k)
    return list(zip(i.tolist(), p.tolist()))


@torch.no_grad()
def attention_maps(model, prompt_ids, device, layer=0):
    """Attention weights of every head in one layer, for a prompt: tensor of shape (n_head, T, T).

    Re-runs the forward pass up to `layer` and computes the attention weights explicitly (the model itself
    uses PyTorch's fused attention, which doesn't return them).
    """
    idx = torch.tensor(prompt_ids[-model.config.block_size:], dtype=torch.long, device=device).unsqueeze(0)
    T = idx.size(1)
    pos = torch.arange(0, T, dtype=torch.long, device=device)
    rope = None
    if getattr(model.config, "pos_encoding", "learned") == "rope":
        rope = rope_cos_sin(pos, model.config.n_embd // model.config.n_head, model.config.rope_theta)
        x = model.transformer.drop(model.transformer.wte(idx))
    else:
        x = model.transformer.drop(model.transformer.wte(idx) + model.transformer.wpe(pos))
    for i, block in enumerate(model.transformer.h):
        if i == layer:
            a = block.attn
            kv_dim = getattr(a, "kv_dim", a.n_embd)
            q, k, _ = a.c_attn(block.ln1(x)).split([a.n_embd, kv_dim, kv_dim], dim=2)
            hs = a.n_embd // a.n_head
            q = q.view(1, T, a.n_head, hs).transpose(1, 2)
            k = k.view(1, T, kv_dim // hs, hs).transpose(1, 2)
            if getattr(a, "qk_norm", False):
                q, k = a.q_norm(q), a.k_norm(k)
            if rope is not None:
                q, k = apply_rope(q, *rope), apply_rope(k, *rope)
            if hasattr(a, "_expand"):
                k = a._expand(k)
            att = (q @ k.transpose(-2, -1)) / math.sqrt(hs)
            att = att.masked_fill(torch.ones(T, T, device=device).tril() == 0, float("-inf"))
            return F.softmax(att, dim=-1)[0]
        for _ in range(max(1, getattr(model.config, "block_repeat", 1))):
            x = block(x, None, rope)
    raise ValueError(f"layer {layer} out of range (model has {len(model.transformer.h)})")

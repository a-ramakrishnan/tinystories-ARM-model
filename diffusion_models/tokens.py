# The tokenizer: GPT-2's (tiktoken), the same as our autoregressive models, plus one [MASK] token (id 50257).

from tokenization import get_tokenizer

from .config import EOT_ID, GPT2_VOCAB, MASK_ID

enc = get_tokenizer()


def encode(text):
    """Plain text -> token ids (no special tokens: documents are separated by <|endoftext|> in the data)."""
    return enc.encode_ordinary(text)


def decode(ids, mask="█"):
    """Token ids -> text, with each [MASK] shown as `mask`. Ids outside GPT-2's vocabulary (never trained on)
    show as '?'."""
    out, run = [], []
    for i in ids:
        if i < GPT2_VOCAB:
            run.append(i)
            continue
        if run:
            out.append(enc.decode(run))
            run = []
        out.append(mask if i == MASK_ID else "?")
    if run:
        out.append(enc.decode(run))
    return "".join(out)


def token_texts(ids, mask="█"):
    """One string per token (for showing tokens one by one)."""
    return [mask if i == MASK_ID else enc.decode([i]) if i < GPT2_VOCAB else "?" for i in ids]


__all__ = ["enc", "encode", "decode", "token_texts", "EOT_ID", "MASK_ID"]

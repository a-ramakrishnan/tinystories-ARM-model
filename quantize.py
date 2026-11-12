"""Weight quantization for inference (roadmap I3): store the model's matrices in 8 or 4 bits instead of 32.

Each weight matrix is replaced by small integers and a few scales, symmetric around zero:
- 8 bits: one scale per output row, w ≈ q × scale with q in −127…127;
- 4 bits: one scale per group of 64 weights in a row, q in −7…7 (finer scales keep the error down).

The integers are turned back into floats when a layer runs ("weight-only" quantization: the maths is still
done in float). That saves memory and disk, not time: the Mac's GPU has no fast 8-bit matrix multiply in
PyTorch, so it's a little slower. The 4-bit integers are kept one per byte here for simplicity;
`weight_bytes` counts what they would take packed two per byte.

The tied input embedding and output layer share one quantized matrix, as they share one float matrix, kept
at 8 bits even in a 4-bit model (see quantize_model).

    model = quantize_model(model, bits=8)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def quantize_tensor(w, bits=8, group=64):
    """(q int8, scale float32, shape): w ≈ (q.view(-1, g) * scale).view(shape), with g = row length or `group`."""
    qmax = 2 ** (bits - 1) - 1
    rows, cols = w.shape
    g = cols if bits >= 8 or cols % group else group
    wg = w.detach().float().reshape(rows, cols // g, g)
    scale = wg.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / qmax
    q = torch.round(wg / scale).clamp(-qmax, qmax).to(torch.int8)
    return q, scale, (rows, cols)


def dequantize(q, scale, shape, dtype=torch.float32):
    return (q.to(dtype) * scale.to(dtype)).reshape(shape)


class QuantLinear(nn.Module):
    """A linear layer whose weight is stored as small integers and scales (see quantize_tensor)."""

    def __init__(self, linear, bits=8, group=64):
        super().__init__()
        q, scale, shape = quantize_tensor(linear.weight, bits, group)
        self.register_buffer("q", q)
        self.register_buffer("scale", scale)
        self.shape, self.bits, self.group = shape, bits, group
        self.bias = linear.bias

    @property
    def weight(self):
        """The dequantized weight (code that reads .weight directly, e.g. the experts' stacked matrices)."""
        return dequantize(self.q, self.scale, self.shape)

    def weight_rows(self, idx):
        """Rows of the dequantized weight (an embedding lookup for the tied input embedding)."""
        return (self.q[idx].float() * self.scale[idx]).reshape(*idx.shape, self.shape[1])

    def forward(self, x):
        return F.linear(x, dequantize(self.q, self.scale, self.shape, x.dtype), self.bias)


class QuantEmbedding(nn.Module):
    """The input embedding, reading its rows from the (tied) quantized output layer."""

    def __init__(self, qlinear):
        super().__init__()
        self.qlinear = [qlinear]  # a list, so it isn't registered (and counted) twice

    def forward(self, idx):
        return self.qlinear[0].weight_rows(idx)


@torch.no_grad()
def quantize_model(model, bits=8, group=64, embed_bits=8):
    """Replace every nn.Linear (and the tied embedding) with a quantized one, in place. Returns the model.

    embed_bits: bits for the shared embedding / output layer. 8 by default even when bits = 4, as llama.cpp
    does: at 4 bits OpenAI's GPT-2 breaks (held-out loss 3.40 → 7.03 from that one matrix), because its rows
    hold a few large values that leave too few levels for the many small ones."""
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                b = embed_bits if (module is model and child_name == "lm_head") else bits
                setattr(module, child_name, QuantLinear(child, b, group))
    if hasattr(model, "transformer") and isinstance(model.lm_head, QuantLinear):
        model.transformer.wte = QuantEmbedding(model.lm_head)  # weight tying, quantized once
    model.quantized_bits = bits
    return model


def weight_bytes(model):
    """Bytes the model's weights take: 4 per float, bits/8 per quantized weight (packed), plus the scales."""
    total = 0
    for m in model.modules():
        if isinstance(m, QuantLinear):
            total += m.q.numel() * m.bits // 8 + m.scale.numel() * 4
    seen = set()
    for m in model.modules():
        for p in list(m.parameters(recurse=False)) + list(m.buffers(recurse=False)):
            if isinstance(m, QuantLinear) and p is not m.bias:
                continue
            if id(p) in seen:
                continue
            seen.add(id(p))
            total += p.numel() * p.element_size()
    return total

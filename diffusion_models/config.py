# Settings for the masked diffusion language model: the model's shape and how it's trained.
#
# Built from v1_original.py (the first version, kept unchanged). Differences from v1:
#   - GPT-2's tokenizer (50,257 tokens, the same as our autoregressive models) plus one [MASK] token, instead of a
#     BPE tokenizer trained on TinyStories. The vocabulary is padded to 50,304 (a multiple of 64, faster on GPUs).
#   - Plain text, with documents separated by <|endoftext|>, as the autoregressive models read it (v1 wrapped
#     each story in a chat template).
#   - RUN_MODES replaces RUN_MODE's global variables; "colab" matches our TinyStories autoregressive model
#     "Medium 51M · TinyStories · 20k steps" in size, data and number of tokens, for a fair comparison.

from dataclasses import asdict, dataclass, field

GPT2_VOCAB = 50257          # tokens 0..50256; 50256 is <|endoftext|>
EOT_ID = 50256
MASK_ID = 50257             # the one token added to GPT-2's vocabulary
VOCAB_SIZE = 50304          # 50,258 rounded up to a multiple of 64; ids above MASK_ID are never used


@dataclass
class DiffusionLMConfig:
    """The model's shape (as in v1)."""
    vocab_size: int = VOCAB_SIZE
    seq_len: int = 256
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    d_ff: int = 4 * 384
    dropout: float = 0.0
    diffusion_steps: int = 64   # T: noise levels t = 1..T mask t/T of the tokens
    mask_id: int = MASK_ID
    tokenizer: str = "gpt2"


@dataclass
class TrainSettings:
    steps: int = 2000           # optimizer updates
    batch_size: int = 32
    grad_accum: int = 1
    lr: float = 6e-4            # peak; linear warmup, then cosine decay to min_lr
    min_lr: float = 6e-5
    warmup_steps: int = 200
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 250
    eval_batches: int = 20
    sample_interval: int = 0    # every N updates, print a short sample (0: only at the end)
    seed: int = 42
    compile: bool = True        # torch.compile, on an Nvidia GPU only

    @property
    def tokens(self):
        return self.steps * self.batch_size * self.grad_accum


def _model(n_layers, d_model, n_heads, **kw):
    return DiffusionLMConfig(n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_ff=4 * d_model, **kw)


# name -> (model, training, what it's for)
RUN_MODES = {
    # a few minutes on a Mac: checks the code end to end, the text stays rough
    "smoke": (_model(6, 384, 6), TrainSettings(steps=300, batch_size=16, warmup_steps=30, eval_interval=100,
                                               eval_batches=5, compile=False),
              "a few minutes on a Mac or any GPU: checks everything works; the text stays rough"),
    # Colab, first look: the full-size model for 2,000 updates (16M tokens), a few minutes on an L4
    "quick": (_model(8, 512, 8), TrainSettings(steps=2000, batch_size=32, warmup_steps=200, eval_interval=250),
              "the full-size model for 2,000 updates (16M tokens): a few minutes on an L4"),
    # Colab, the comparison run: 51M parameters, 20,000 updates of 32 x 256 tokens = 164M tokens, the same as the
    # autoregressive "Medium 51M · TinyStories · 20k steps" (8 layers x 512, learned positions, GELU, lr 6e-4)
    "colab": (_model(8, 512, 8), TrainSettings(steps=20000, batch_size=32, warmup_steps=1000, eval_interval=1000,
                                               eval_batches=40, sample_interval=5000),
              "51M parameters, 164M tokens: matches our 51M autoregressive TinyStories model; about 30-45 min on an L4"),
}


def run_mode(name):
    """Fresh copies of a preset's model config and training settings."""
    if name not in RUN_MODES:
        raise ValueError(f"RUN_MODE must be one of {', '.join(RUN_MODES)}")
    m, t, _ = RUN_MODES[name]
    return DiffusionLMConfig(**asdict(m)), TrainSettings(**asdict(t))

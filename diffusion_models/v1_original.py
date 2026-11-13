# Ramakrishnan A
# Diffusion Model for predicting Text

# Noising -> Predict Noise -> DeNoising

import os, math, time, json, random
import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from IPython.display import Image as IPyImage, display

from datasets import load_dataset
from dataclasses import dataclass

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.normalizers import NFKC
from tokenizers.processors import TemplateProcessing

from transformers import PreTrainedTokenizerFast
from accelerate import Accelerator
from transformers import get_cosine_schedule_with_warmup

#RUN_MODE = "budget_100"   # "quick" or "budget_100"
RUN_MODE = "quick"

# You can always override individual values later.

if RUN_MODE == "quick":
    # Small + fast: good for verifying everything end-to-end
    TRAIN_EXAMPLES = 50_000
    VAL_EXAMPLES   = 2_000
    TOKENIZER_TRAIN_EXAMPLES = 30_000

    SEQ_LEN = 256
    VOCAB_SIZE = 8_000

    D_MODEL = 384
    N_LAYERS = 6
    N_HEADS = 6
    D_FF = 4 * D_MODEL

    DIFFUSION_STEPS = 64

    TRAIN_STEPS = 2_000
    BATCH_SIZE = 32
    GRAD_ACCUM = 1
    LR = 3e-4
    WEIGHT_DECAY = 0.1
    WARMUP_STEPS = 200

elif RUN_MODE == "budget_100":
    # Heavier: better quality, uses more compute
    TRAIN_EXAMPLES = 1000_000
    VAL_EXAMPLES   = 10_000
    TOKENIZER_TRAIN_EXAMPLES = 150_000

    SEQ_LEN = 256
    VOCAB_SIZE = 26_000

    D_MODEL = 512
    N_LAYERS = 10
    N_HEADS = 8
    D_FF = 4 * D_MODEL

    DIFFUSION_STEPS = 128

    TRAIN_STEPS = 200_000
    BATCH_SIZE = 32
    GRAD_ACCUM = 2
    LR = 2e-4
    WEIGHT_DECAY = 0.1
    WARMUP_STEPS = 1_000

else:
    raise ValueError("RUN_MODE must be 'quick' or 'budget_100'")

print("RUN_MODE:", RUN_MODE)

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("bf16 supported:", torch.cuda.is_bf16_supported())

train_ds = load_dataset("roneneldan/TinyStories", split=f"train[:{TRAIN_EXAMPLES}]")
val_ds   = load_dataset("roneneldan/TinyStories", split=f"validation[:{VAL_EXAMPLES}]")

print(train_ds, val_ds)
print("\nExample:\n", train_ds[0]["text"][:500])

SPECIAL_TOKENS = [
    "[PAD]", "[UNK]", "[BOS]", "[EOS]", "[MASK]",
    "<|user|>", "<|assistant|>", "<|system|>", "<|end|>",
]

def tokenizer_training_iterator(ds, n_examples):
    for i in range(min(n_examples, len(ds))):
        story = ds[i]["text"].strip()
        yield f"<|user|>\nWrite a short story.\n<|assistant|>\n{story}\n<|end|>\n"

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.normalizer = NFKC()
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

trainer = BpeTrainer(
    vocab_size=VOCAB_SIZE,
    min_frequency=2,
    special_tokens=SPECIAL_TOKENS,
)

print("Training tokenizer...")
tokenizer.train_from_iterator(
    tokenizer_training_iterator(train_ds, TOKENIZER_TRAIN_EXAMPLES),
    trainer=trainer
)

bos_id = tokenizer.token_to_id("[BOS]")
eos_id = tokenizer.token_to_id("[EOS]")
tokenizer.post_processor = TemplateProcessing(
    single="[BOS] $A [EOS]",
    special_tokens=[("[BOS]", bos_id), ("[EOS]", eos_id)],
)
tokenizer.decoder = ByteLevelDecoder()

TOKENIZER_DIR = "tokenizer_from_scratch"
os.makedirs(TOKENIZER_DIR, exist_ok=True)
TOKENIZER_FILE = os.path.join(TOKENIZER_DIR, "tokenizer.json")
tokenizer.save(TOKENIZER_FILE)

print("Saved tokenizer to:", TOKENIZER_FILE)
print("Vocab size:", tokenizer.get_vocab_size())



hf_tokenizer = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_FILE)

hf_tokenizer.pad_token  = "[PAD]"
hf_tokenizer.unk_token  = "[UNK]"
hf_tokenizer.bos_token  = "[BOS]"
hf_tokenizer.eos_token  = "[EOS]"
hf_tokenizer.mask_token = "[MASK]"

hf_tokenizer.add_special_tokens({
    "additional_special_tokens": ["<|user|>", "<|assistant|>", "<|system|>", "<|end|>"]
})

PAD_ID  = hf_tokenizer.pad_token_id
MASK_ID = hf_tokenizer.mask_token_id
BOS_ID  = hf_tokenizer.bos_token_id
EOS_ID  = hf_tokenizer.eos_token_id

print("PAD_ID:", PAD_ID, "MASK_ID:", MASK_ID, "BOS_ID:", BOS_ID, "EOS_ID:", EOS_ID)
print("Example encoding:", hf_tokenizer.encode("Hello world!")[:20])

@dataclass
class DiffusionLMConfig:
    vocab_size: int
    seq_len: int
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    dropout: float
    diffusion_steps: int

class DiffusionTransformerLM(nn.Module):
    def __init__(self, cfg: DiffusionLMConfig):
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.time_emb = nn.Embedding(cfg.diffusion_steps + 1, cfg.d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers)
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Tie weights (optional; common in LMs)
        self.lm_head.weight = self.tok_emb.weight

        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, input_ids, timesteps, attention_mask=None):
        # input_ids: [B, L]
        # timesteps: [B] integer diffusion step in [1..T]
        # attention_mask: [B, L] bool, True for non-pad tokens

        B, L = input_ids.shape
        if L > self.cfg.seq_len:
            raise ValueError(f"Sequence length {L} > cfg.seq_len {self.cfg.seq_len}")

        pos = torch.arange(L, device=input_ids.device).unsqueeze(0)  # [1, L]
        x = self.tok_emb(input_ids) + self.pos_emb(pos)

        t_emb = self.time_emb(timesteps).unsqueeze(1)  # [B, 1, D]
        x = x + t_emb
        x = self.drop(x)

        if attention_mask is None:
            src_key_padding_mask = None
        else:
            src_key_padding_mask = ~attention_mask  # invert: True = pad/ignore

        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # [B, L, V]
        return logits

cfg = DiffusionLMConfig(
    vocab_size=len(hf_tokenizer),
    seq_len=SEQ_LEN,
    d_model=D_MODEL,
    n_layers=N_LAYERS,
    n_heads=N_HEADS,
    d_ff=D_FF,
    dropout=0.1,
    diffusion_steps=DIFFUSION_STEPS,
)
model = DiffusionTransformerLM(cfg)

n_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {n_params/1e6:.2f}M")

def format_as_chat(story_text: str) -> str:
    story_text = story_text.strip()
    return f"<|user|>\nWrite a short story.\n<|assistant|>\n{story_text}\n<|end|>\n"

class TokenBlockDataset(IterableDataset):
    def __init__(self, hf_ds, tokenizer, seq_len, shuffle=False, seed=0):
        self.hf_ds = hf_ds
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        indices = list(range(len(self.hf_ds)))
        if self.shuffle:
            rng = random.Random(self.seed)
            rng.shuffle(indices)

        buffer = []
        for idx in indices:
            text = format_as_chat(self.hf_ds[idx]["text"])
            ids = self.tokenizer.encode(text, add_special_tokens=True)
            buffer.extend(ids)

            while len(buffer) >= self.seq_len:
                block = buffer[:self.seq_len]
                buffer = buffer[self.seq_len:]
                yield torch.tensor(block, dtype=torch.long)

train_blocks = TokenBlockDataset(train_ds, hf_tokenizer, SEQ_LEN, shuffle=True, seed=42)
val_blocks   = TokenBlockDataset(val_ds,   hf_tokenizer, SEQ_LEN, shuffle=False)

def collate_blocks(batch):
    input_ids = torch.stack(batch, dim=0)  # [B, L]
    attention_mask = (input_ids != PAD_ID)
    return {"input_ids": input_ids, "attention_mask": attention_mask}

train_loader = DataLoader(train_blocks, batch_size=BATCH_SIZE, collate_fn=collate_blocks)
val_loader   = DataLoader(val_blocks,   batch_size=BATCH_SIZE, collate_fn=collate_blocks)

b = next(iter(train_loader))
print({k: v.shape for k, v in b.items()})
print("Decoded snippet:\n", hf_tokenizer.decode(b["input_ids"][0][:120].tolist()))

def mask_ratio_schedule(t, T: int):
    # Linear schedule: ratio = t/T
    return t.float() / float(T)

@torch.no_grad()
def corrupt_with_mask(input_ids, attention_mask, t, mask_token_id: int, T: int):
    # Returns noisy_ids, labels, mask_positions
    B, L = input_ids.shape
    ratio = mask_ratio_schedule(t, T).unsqueeze(1)  # [B,1]

    can_mask = attention_mask.clone()
    can_mask &= (input_ids != BOS_ID) & (input_ids != EOS_ID) & (input_ids != PAD_ID)

    rand = torch.rand((B, L), device=input_ids.device)
    mask_positions = (rand < ratio) & can_mask

    noisy = input_ids.clone()
    noisy[mask_positions] = mask_token_id

    labels = torch.full_like(input_ids, -100)
    labels[mask_positions] = input_ids[mask_positions]

    return noisy, labels, mask_positions

def diffusion_loss(model, batch, T: int):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    B = input_ids.size(0)
    t = torch.randint(1, T + 1, (B,), device=input_ids.device)

    noisy_ids, labels, _ = corrupt_with_mask(
        input_ids=input_ids,
        attention_mask=attention_mask,
        t=t,
        mask_token_id=MASK_ID,
        T=T,
    )

    logits = model(noisy_ids, timesteps=t, attention_mask=attention_mask)  # [B,L,V]
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
    )
    return loss

if torch.cuda.is_available():
    mixed_precision = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
else:
    mixed_precision = "no"

accelerator = Accelerator(mixed_precision)
device = accelerator.device

model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=WARMUP_STEPS,
    num_training_steps=TRAIN_STEPS,
)

model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
    model, optimizer, train_loader, val_loader, scheduler
)

def eval_loss(n_batches=20):
    model.eval()
    losses = []
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= n_batches:
                break

            loss = diffusion_loss(model, batch, T=cfg.diffusion_steps)

            # gather across processes -> always make it 1D
            gathered = accelerator.gather(loss.detach().float().reshape(1))

            # now gathered is shape [world_size] (or [1] on single GPU)
            losses.append(gathered.cpu())

    model.train()

    if len(losses) == 0:
        return float("nan")

    losses = torch.cat(losses)   # safe: all are 1D tensors
    return losses.mean().item()


model.train()
pbar = tqdm(range(TRAIN_STEPS), disable=not accelerator.is_main_process)
running = []

train_iter = iter(train_loader)

for step in pbar:
    try:
        batch = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        batch = next(train_iter)

    loss = diffusion_loss(model, batch, T=cfg.diffusion_steps) / GRAD_ACCUM
    accelerator.backward(loss)

    if (step + 1) % GRAD_ACCUM == 0:
        accelerator.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    running.append(loss.item() * GRAD_ACCUM)

    if (step + 1) % 50 == 0 and accelerator.is_main_process:
        pbar.set_description(f"loss={np.mean(running[-50:]):.4f} lr={scheduler.get_last_lr()[0]:.2e}")

    if (step + 1) % 500 == 0 and accelerator.is_main_process:
        val_l = eval_loss(n_batches=10)
        print(f"\nStep {step+1} | val_loss ~ {val_l:.4f}")

if accelerator.is_main_process:
    OUT_DIR = "checkpoints/final"
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.save(accelerator.unwrap_model(model).state_dict(), os.path.join(OUT_DIR, "model.pt"))
    with open(os.path.join(OUT_DIR, "config.json"), "w") as f:
        json.dump(cfg.__dict__, f, indent=2)
    hf_tokenizer.save_pretrained(os.path.join(OUT_DIR, "tokenizer"))
    print("Saved final checkpoint to:", OUT_DIR)

@torch.no_grad()
def diffusion_generate(
    model,
    tokenizer,
    prompt_text: str,
    max_new_tokens: int = 128,
    diffusion_steps: int = 64,
    temperature: float = 1.0,
    top_k: int = 0,
    record_steps: bool = True,
):
    model.eval()
    device = next(model.parameters()).device

    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=True)
    prompt_ids = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)  # [1, Lp]

    Lp = prompt_ids.size(1)
    L = min(cfg.seq_len, Lp + max_new_tokens)
    gen_len = L - Lp

    x = torch.full((1, L), MASK_ID, dtype=torch.long, device=device)
    x[:, :Lp] = prompt_ids[:, :Lp]

    fixed = torch.zeros((1, L), dtype=torch.bool, device=device)
    fixed[:, :Lp] = True

    attention_mask = torch.ones((1, L), dtype=torch.bool, device=device)

    frames = []

    def sample_from_logits(logits):
        if temperature != 1.0:
            logits = logits / temperature

        if top_k and top_k > 0:
            topk_vals, topk_idx = torch.topk(logits, k=top_k, dim=-1)
            filtered = torch.full_like(logits, float("-inf"))
            filtered.scatter_(-1, topk_idx, topk_vals)
            logits = filtered

        probs = F.softmax(logits, dim=-1)
        flat = probs.view(-1, probs.size(-1))
        sampled = torch.multinomial(flat, num_samples=1).view(1, L)
        sampled_prob = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)  # [1,L]
        return sampled, sampled_prob

    for s in range(diffusion_steps, 0, -1):
        t = torch.tensor([s], device=device, dtype=torch.long)
        logits = model(x, timesteps=t, attention_mask=attention_mask)
        sampled, conf = sample_from_logits(logits)

        update_pos = ~fixed
        x[update_pos] = sampled[update_pos]

        next_ratio = float(s - 1) / float(diffusion_steps)
        target_masks = int(math.ceil(gen_len * next_ratio))

        gen_positions = torch.arange(L, device=device) >= Lp
        candidates = gen_positions & (~fixed[0])
        cand_idx = torch.where(candidates)[0]

        if target_masks > 0 and cand_idx.numel() > 0:
            cand_conf = conf[0, cand_idx]
            k = min(target_masks, cand_idx.numel())
            _, low_idx = torch.topk(cand_conf, k=k, largest=False)
            remask_positions = cand_idx[low_idx]
            x[0, remask_positions] = MASK_ID

        if record_steps:
            decoded = tokenizer.decode(x[0].tolist())
            decoded = decoded.replace("[MASK]", "█")
            frames.append((s, decoded))

    final = tokenizer.decode(x[0].tolist())
    model.train()
    return final, frames

def chat_prompt(user_msg: str, system_msg: str = None) -> str:
    parts = []
    if system_msg:
        parts.append(f"<|system|>\n{system_msg}\n")
    parts.append(f"<|user|>\n{user_msg}\n")
    parts.append("<|assistant|>\n")
    return "".join(parts)

TEST_USER_PROMPT = "Once upon a time"
prompt_text = chat_prompt(TEST_USER_PROMPT)

final_text, frames = diffusion_generate(
    model=accelerator.unwrap_model(model),
    tokenizer=hf_tokenizer,
    prompt_text=prompt_text,
    max_new_tokens=128,
    diffusion_steps=cfg.diffusion_steps,
    temperature=1.0,
    top_k=50,
    record_steps=True,
)

print("Final decoded (raw):\n")
print(final_text[:1000])
print("\nRecorded frames:", len(frames))

from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio

def get_mono_font(size=20):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()

def render_terminal_frame(lines, width=1200, height=700, font_size=20, margin=20, line_spacing=6):
    bg = (10, 10, 10)
    fg = (230, 230, 230)

    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)
    font = get_mono_font(font_size)

    y = margin
    for line in lines:
        draw.text((margin, y), line, font=font, fill=fg)
        y += font_size + line_spacing
        if y > height - margin:
            break
    return img

def wrap_text_to_width(text, max_chars=90):
    out = []
    for paragraph in text.split("\n"):
        paragraph = paragraph.rstrip()
        if not paragraph:
            out.append("")
            continue
        while len(paragraph) > max_chars:
            out.append(paragraph[:max_chars])
            paragraph = paragraph[max_chars:]
        out.append(paragraph)
    return out

def make_chat_lines(user_msg: str, assistant_text: str):
    header = "============================== multi-turn chat mode ==============================="
    sub = "<Starting a new chat. Type your message.>"
    lines = [header, sub, ""]
    lines += ["[You]:", user_msg, ""]
    lines += ["[Assistant]:"]

    if "<|assistant|>" in assistant_text:
        assistant_text = assistant_text.split("<|assistant|>", 1)[1]
    assistant_text = assistant_text.replace("<|end|>", "").strip()

    lines += wrap_text_to_width(assistant_text, max_chars=90)
    return lines

gif_frames = []
for (s, decoded) in frames:
    lines = make_chat_lines(TEST_USER_PROMPT, decoded)
    lines.insert(2, f"(diffusion step {s:03d}/{cfg.diffusion_steps:03d})")
    img = render_terminal_frame(lines)
    gif_frames.append(np.array(img))

GIF_PATH = "inference.gif"
imageio.mimsave(GIF_PATH, gif_frames, duration=0.08)

print("Saved:", GIF_PATH)

display(IPyImage(filename="inference.gif"))

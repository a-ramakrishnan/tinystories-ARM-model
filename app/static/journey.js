// Journey: how the model got better, as a timeline. Every number was measured in this project (see docs/journey.md).

const J_DOC = "https://github.com/a-ramakrishnan/tinystories-ARM-model/blob/main/docs/journey.md";

// kind: win | tie | loss | fix | tool. best: part of today's best model.
const JOURNEY = [
  { when: "2026-11-14 10:00", kind: "fix", area: "Diffusion", title: "Diffusion text without loops",
    decision: "Score the diffusion models' text on held-out prompts (diffusion_models/quality.py: repetition, and the base model's loss on it) and try samplers; make a repetition penalty of 1.3 the default.",
    why: "On FineWeb-Edu, confident-first sampling filled one word everywhere: \"animals, animals, and animals, …\".",
    result: { label: "Repeated 4-token runs (real text) · judge loss (real text)", text: "FineWeb-Edu 43% → 4% (0%) · 3.62 (3.79) · TinyStories 6.2% → 0.4% (0.8%) · 2.80 (2.68)" },
    calc: ["it reveals the position it's surest about, usually a copy of a neighbouring word; each copy makes the next surer",
      "blocks of 4–16 tokens, left-to-right order, re-masking and random order were all worse",
      "the app's Diffusion page replays the denoising step by step"],
    cost: "no training: about 40 minutes of scoring on the M5", anchor: "33-a-second-kind-of-model-masked-diffusion" },
  { when: "2026-11-13 17:17", kind: "tie", area: "Diffusion", title: "The diffusion model on FineWeb-Edu",
    decision: "Train the same 51M diffusion model on Mini FineWeb-Edu (small) with the same settings.",
    why: "TinyStories is easy; web text is where the base model lives.",
    result: { label: "Likelihood bound, 164M tokens", text: "5.269 nats/token, still falling (the 124M AR base: 3.549 held out after 500M tokens)" },
    calc: ["plateau of word frequencies for ~3,000 updates", "text on topic but loosely connected; a 3× longer run followed"],
    cost: "50 min on a Colab L4 (59,000 tokens/s)", anchor: "33-a-second-kind-of-model-masked-diffusion" },
  { when: "2026-11-13 14:50", kind: "win", area: "Diffusion", title: "A second kind of model: masked diffusion",
    decision: "The author's masked diffusion LM, made modular (GPT-2 tokenizer, our token files, a likelihood bound) and trained like the 51M TinyStories model: same size, tokens and schedule.",
    why: "A model that writes by denoising, all positions in parallel, is the clearest contrast with left-to-right writing.",
    result: { label: "TinyStories, 51M, 164M tokens", steps: [["Autoregressive (exact loss)", 1.583], ["Diffusion (likelihood bound)", 2.525]],
      extra: "coherent stories once revealed tokens are kept fixed; v1's re-sampling of every position jumbled the final text" },
    calc: ["it learned only word frequencies for ~2,500 updates, then used context", "6e-4 beat 2e-3 in a 2,000-update check",
      "64 tokens in 64 passes, 0.3 s on the L4"],
    cost: "49 min on a Colab L4 (59,000 tokens/s)", anchor: "33-a-second-kind-of-model-masked-diffusion" },
  { when: "2026-11-13 13:15", kind: "win", area: "Data", best: true, title: "More kinds of data: the new base model",
    decision: "Same 124M best design and number of updates, but Mini FineWeb-Edu (55%) + Cosmopedia stories, wikihow, openstax, stanford and khanacademy (45%), 512-token context, 500M tokens.",
    why: "The benchmarks showed what was missing: everyday how-to and stories, which educational web pages barely contain.",
    result: { label: "First base → new base (GPT-2 small)", text: "held-out FineWeb-Edu 3.691 → 3.549 (3.401) · ARC-Easy 42.1% → 44.0% (43.8%) · PIQA 58.4% → 62.6% (62.9%) · HellaSwag 27.9% → 29.7% (31.1%) · LAMBADA 9.6% → 15.7% (32.6%)" },
    calc: ["about the same FineWeb-Edu as before plus 255M tokens of Cosmopedia; 30,500 updates of 32 × 512 tokens",
      "greedy repetition 75% → 25%; held-out TinyStories 3.856 → 2.790",
      "the data gained 0.14 on held-out web text, against 0.04 for the whole architecture track at this size"],
    cost: "4 h 20 min on a Colab L4 (32,000 tokens/s with torch.compile)", anchor: "32-more-kinds-of-data" },
  { when: "2026-11-13 08:00", kind: "win", area: "Training", title: "Faster training: torch.compile",
    decision: "Compile the model on Nvidia GPUs, use fused AdamW and TF32, and time each change on its own with speedtest.py on a Colab L4.",
    why: "The next gains are in data, and data costs GPU hours: the 124M run read 246M tokens in 3 h 38 min.",
    result: { label: "Tokens/s, 124M best design, Colab L4", text: "20,869 → 35,344 with torch.compile (1.69×; 7.5 GB instead of 11.2) · fused AdamW + TF32 1.06× · bigger batches: no gain" },
    calc: ["512 tokens of context: 34,381 tokens/s, only 3% slower per token", "same maths: identical losses on the Mac; compiled gradients within 4e-8",
      "no A100 was free on Colab to compare"],
    cost: "about 15 minutes of an L4", anchor: "31-faster-training" },
  { when: "2026-11-12 19:00", kind: "tool", area: "Tooling", title: "Standard benchmarks",
    decision: "Score HellaSwag, LAMBADA, ARC-Easy, PIQA and Winogrande zero-shot, the way EleutherAI's lm-evaluation-harness does; check on GPT-2 first.",
    why: "Our own evaluation suite compares our models with each other; published benchmarks compare them with everyone else's.",
    result: { label: "★ metrics: our 124M best design vs OpenAI GPT-2 small", text: "ARC-Easy 42.1% vs 43.8% · HellaSwag 27.9% vs 31.1% · PIQA 58.4% vs 62.9% · Winogrande 50.6% vs 51.6% · LAMBADA 9.6% vs 32.6%" },
    calc: ["GPT-2 measured here = published: HellaSwag 28.92 / 31.14, LAMBADA 32.56 (ppl 40.06), PIQA 62.89, Winogrande 51.62",
      "close on school science (what educational pages teach); far behind on LAMBADA's novels (no fiction in FineWeb-Edu)"],
    cost: "no training: 9 minutes for GPT-2 and 18 for ours on an M5", anchor: "30-the-best-design-at-gpt-2-size" },
  { when: "2026-11-12 16:45", kind: "win", area: "Architecture", best: true, title: "The best design at GPT-2 size",
    decision: "Train the Large model (124M) again with RoPE, SwiGLU, no biases and QK-norm, learning rate 1.2e-3, 30,000 steps on Mini FineWeb-Edu.",
    why: "Every architecture win was measured on a 30M model over short runs. Does it hold at GPT-2's size over a full run?",
    result: { label: "Held-out Mini FineWeb-Edu, same windows", steps: [["OpenAI GPT-2 small", 3.401], ["our Large, best design", 3.691], ["our Large, GPT-2 style", 3.735]],
      extra: "better by 0.04 (validation 3.603 → 3.563), much less than the 0.3 on the Small model; the gap to GPT-2 shrinks 0.333 → 0.290" },
    calc: ["learning-rate check, 1,000 updates: 6e-4 5.125 · 1.2e-3 5.103 · 1.8e-3 5.105",
      "led by 0.25 at step 1,000, level at 10,000, ahead again as the learning rate decayed",
      "WikiText-2 5.582 → 5.605: slightly worse there"],
    cost: "3 h 38 min on a Colab L4 (18,759 tokens/s), plus a 25-minute learning-rate check", anchor: "30-the-best-design-at-gpt-2-size" },
  { when: "2026-11-12 16:30", kind: "tool", area: "Tooling", title: "The Explorer: watch the real model work",
    decision: "A second view in Inside the transformer: one prompt through the whole model with its real numbers, as a Flow of ribbons or a Diagram, with a logit lens, a per-token trace and ▶ Play the data flow (one token through every block, step by step).",
    why: "The walkthrough uses made-up 4-dimensional numbers to be checkable by hand; this shows what the real model actually does.",
    result: { label: "Logit lens, OpenAI GPT-2, '…the capital of Germany is'", text: "blocks 1–7: 'not' / 'now' → block 8: 'Paris' (15%) → block 9: 'Berlin' (80%) → final 'Berlin' 67.5%" },
    calc: ["explain.py replays the forward pass block by block; its final logits match the model's own (within 0.0004)",
      "attention in three stages: q·k/√d, the causal mask, softmax; click a cell for its calculation",
      "the player, for 'cat': scores 1.1 / 2.3 / 1.1 → weights 20% / 62% / 19%; 3.5% of 1,024 hidden values active; size 15.5 → 40.7"],
    cost: "no training", anchor: "29-seeing-the-model-work-the-explorer" },
  { when: "2026-11-12 15:40", kind: "win", area: "Inference", title: "Batched generation",
    decision: "Generate n samples of one prompt at once: read the prompt once, copy its KV cache to n rows, one batched pass per step.",
    why: "A pass of these models costs mostly fixed overhead, so many tokens per pass should come almost free.",
    result: { label: "Tokens/s in total, 64 samples vs one stream", text: "Small 92 → 2,864 (31×) · Large 124M 60 → 1,682 (28×); greedy rows identical" },
    calc: ["prefix caching skipped: a 250-token prompt takes 17–25 ms to read, at most ~16 ms to save",
      "sampling (Balanced) is 10–25% slower than greedy, measured side by side"],
    cost: "no training: benchmarks on the Mac", anchor: "28-batched-generation-and-prefix-caching" },
  { when: "2026-11-12 15:10", kind: "tie", area: "Inference", title: "8-bit and 4-bit weights",
    decision: "Store each weight matrix as small integers and scales (quantize.py); keep the shared embedding at 8 bits.",
    why: "A 124M model takes 495 MB in 32-bit floats; most of that precision isn't needed to run it.",
    result: { label: "Our Large 124M: held-out loss · weight memory", text: "32-bit 3.7345 · 495 MB → 8-bit 3.7353 · 125 MB → 4-bit 3.7490 · 88 MB; but 1.3–3.5× slower on the Mac" },
    calc: ["OpenAI GPT-2 at 4 bits: 3.40 → 7.03 from the embedding alone (row outliers up to 1.8 vs a typical 0.09); 3.56 with it at 8 bits",
      "the KV cache at 256 tokens is 19 MB next to 495 MB of weights: not worth quantizing here"],
    cost: "no training: evaluations on the Mac", anchor: "27-8-bit-and-4-bit-weights" },
  { when: "2026-11-12 14:30", kind: "win", area: "Inference", title: "Better decoding: DRY, contrastive and beam search",
    decision: "Add DRY (penalize only continuing a repeated sequence) and typical-p to Generate, plus contrastive and beam search; compare all on the suite's prompts.",
    why: "Greedy loops (27.5% of 4-token runs on stories, 76% on web text); the repetition penalty fixes that by punishing every word already used.",
    result: { label: "TinyStories: repeated 4-grams / judge perplexity (Medium 51M)", text: "greedy 27.5% / 2.70 → greedy + DRY 1.2% / 2.67 · Balanced 0.3% / 5.67 → T 0.8, top-p 0.9 + DRY 1.4% / 4.16" },
    calc: ["DRY penalty: 0.8 × 1.75^(n − 2) for a token that would extend an n-token repeat (n ≥ 2)",
      "web text: contrastive search 0.7% repeats at judge perplexity 8.0, vs Balanced 1.2% at 23.4",
      "plain beam search loops (65% on web text); typical sampling: word salad at 30M"],
    cost: "no training: about 15 minutes of generation on the Mac", anchor: "26-better-decoding-algorithms" },
  { when: "2026-11-12 13:00", kind: "tie", area: "Inference", title: "Speculative decoding",
    decision: "A small model drafts a few tokens; the large model checks them all in one pass and keeps those it agrees with.",
    why: "Each pass of the large model could then give several tokens, with exactly the large model's output.",
    result: { label: "Greedy, identical output; best speed vs the target alone", text: "Medium 51M + Tiny drafter 0.99–1.07× · our Large 124M 0.97× · OpenAI GPT-2 1.33× (59–90% of guesses kept)" },
    calc: ["a 124M pass costs 8.2 ms for 1 token and 9.2 ms for 4: checking is cheap",
      "but drafting isn't: 3.5 ms per guess (6 layers), 1.6 ms (4 layers), when the target needs only 4–8 ms per token"],
    cost: "a Tiny drafter: 2,000 updates on the Mac (~27 min)", anchor: "25-winning-back-generation-speed" },
  { when: "2026-11-12 10:15", kind: "win", area: "Inference", best: true, title: "Winning back generation speed",
    decision: "Remove the new design's overhead (cached RoPE tables, fused RMSNorm, batched experts) and every hidden wait for the GPU in the generation loop.",
    why: "The better design generated about 2× slower (66 tokens/s with experts). For small models, the time goes to overhead, not maths.",
    result: { label: "Tokens/s on an M5, KV cache, greedy, identical output", text: "experts 43 → 150 (3.5×) · dense 127 → 232 (1.8×) · GPT-2 style 123–165 → 292–309 (~2×)" },
    calc: ["a list index x[:, [-1]] is copied to the GPU and waits: a slice does the same without waiting",
      "torch.cat on the GPU token history cost ~0.7 ms per token: now a Python list",
      "half precision: no faster at this size; a compiled step: +25–35% for the new designs, not adopted yet"],
    cost: "no training: benchmarks on the Mac", anchor: "25-winning-back-generation-speed" },
  { when: "2026-11-11 22:30", kind: "tool", area: "Tooling", title: "A yardstick: OpenAI's GPT-2",
    decision: "Load OpenAI's GPT-2 small (124M) weights into our GPT class and run it through the same evaluation suite.",
    why: "Our losses only compared our own models. How far is our 124M model from the real GPT-2 of the same size?",
    result: { label: "Held-out Mini FineWeb-Edu, same windows", steps: [["OpenAI GPT-2 small", 3.401], ["our Large, GPT-2 style, 30k steps", 3.735], ["our Small, best + experts", 4.245]],
      extra: "GPT-2 never saw FineWeb-Edu and is still 0.33 ahead: it read billions of tokens, ours 0.25 billion" },
    calc: ["124,439,808 parameters: the official count (1,024 positions × 768 included)",
      "OpenAI's Conv1D weights are (in, out): transposed for nn.Linear; GELU is the tanh approximation"],
    cost: "no training: a 550 MB download and about a minute of evaluation per model on the Mac", anchor: "24-a-yardstick-openais-gpt-2" },
  { when: "2026-11-11 21:40", kind: "win", area: "Data", best: true, title: "The same designs on harder data",
    decision: "Train GPT-2 style, our best dense and our best with experts on Mini FineWeb-Edu (web pages), 5,000 updates each, from the Lab via a Colab run spec.",
    why: "Everything had been measured on children's stories. Would the same choices hold on harder text?",
    result: { label: "Validation loss on Mini FineWeb-Edu, 5,000 updates", steps: [["GPT-2 style", 4.495], ["Our best, dense", 4.206], ["Our best + experts", 4.143]],
      extra: "the gains double on web text: 0.35 vs 0.17 on TinyStories" },
    calc: ["best dense passes GPT-2 style's final loss (4.495) near update 2,900 (4.560 at 2,500, 4.343 at 3,750) → ~40% fewer updates",
      "WikiText-2 held-out: 6.24–6.53 here vs 9.5–9.9 for the TinyStories models"],
    cost: "3 runs, about 45 min on a Colab L4 (including downloading and tokenizing 1.3 GB)", anchor: "23-the-same-designs-on-harder-data-mini-fineweb-edu" },
  { when: "2026-11-11 16:19", kind: "win", area: "Architecture", best: true, title: "All the wins together, over a longer run",
    decision: "Train the GPT-2-style model and the combined best designs for 5,000 updates each, five times the A/B budget.",
    why: "Every earlier result came from a short run, one change at a time. Do the gains add up, and do they last?",
    result: { label: "Validation loss after 5,000 updates", steps: [["GPT-2 style", 1.903], ["+ learning rate 1.8e-3", 1.851], ["+ RoPE, SwiGLU, no biases, QK-norm", 1.782], ["+ 4 experts, 2 per token", 1.732]] },
    calc: ["tokens seen = 5,000 updates × 32 sequences × 256 tokens = 41M", "best dense passes GPT-2's final loss before update 3,750 → 25% fewer updates",
      "with experts: crosses 1.903 near update 3,100 → 38% fewer updates"],
    cost: "4 runs × 9–16 min on a Colab L4, about 50 min in all", anchor: "22-putting-it-together-the-original-model-vs-all-the-wins",
    after: "Generation got slower (232 → 132 tokens/s, 66 with experts) in this unfused implementation: the next thing to fix." },
  { when: "2026-11-11 15:29", kind: "win", area: "Architecture", best: true, title: "A mixture of experts, 2 per token",
    decision: "4 feed-forward experts per block and a router sending each token to its top 2, with a load-balancing loss.",
    why: "More parameters for about the same compute per token.",
    result: { label: "Validation loss, 1,000 updates", from: 2.251, to: 2.200, extra: "1 expert per token was worse: 2.391 (4 experts), 2.464 (8)" },
    calc: ["one expert (SwiGLU) = 3 × 384 × 1,024 = 1.18M parameters", "total = 29.9M + 6 layers × 3 extra experts × 1.18M = 51.2M",
      "used per token = 51.2M − 6 × 2 unused experts × 1.18M = 37.0M"],
    cost: "4 variants, about 20 min on a Colab L4", anchor: "21-architecture-step-ten-a-mixture-of-experts" },
  { when: "2026-11-11 15:05", kind: "loss", area: "Architecture", title: "Multi-token prediction",
    decision: "Extra heads that also learn to predict 2, 3 or 4 tokens ahead (training only).",
    why: "A richer training signal; Meta and DeepSeek use it in big models.",
    result: { label: "Validation loss, 1,000 updates", from: 2.251, to: 2.336, extra: "worse with more heads (2.490, 2.536); the 2-ahead head agrees with the main head 51% of the time: a drafter for speculative decoding" },
    calc: ["each extra head = one transformer block = 1.77M parameters, training only", "averaging the loss over n heads gives the next-token loss 1/n of the weight"],
    cost: "4 variants, about 18 min on a Colab L4", anchor: "20-architecture-step-nine-predicting-several-tokens-at-once" },
  { when: "2026-11-11 14:39", kind: "loss", area: "Architecture", title: "Layer sharing",
    decision: "Apply each block 2 or 3 times in a row with the same weights: more depth, same parameters.",
    why: "MobileLLM found it helps small models.",
    result: { label: "Validation loss, 1,000 updates", from: 2.251, to: 2.346, extra: "3×: 2.518. Shared weights get 2–3× the gradient: in effect a higher learning rate" },
    calc: ["12 layers deep with the parameters of 6", "KV cache per token doubles: 18,432 → 36,864 bytes"],
    cost: "4 variants, about 22 min on a Colab L4", anchor: "19-architecture-step-eight-layer-sharing" },
  { when: "2026-11-11 14:14", kind: "loss", area: "Architecture", title: "Deep and thin vs wide and shallow",
    decision: "Four shapes with the same ~29.9M parameters: 3 × 448, 6 × 384, 11 × 320, 22 × 256.",
    why: "Studies of small models say depth beats width.",
    result: { label: "Validation loss, 1,000 updates", steps: [["3 × 448", 2.271], ["6 × 384", 2.251], ["11 × 320", 2.270], ["22 × 256", 2.401]] },
    calc: ["parameters ≈ 50,257 × width + layers × 12 × width²", "narrower width → smaller embedding → room for more layers",
      "generation: 215 tokens/s at 3 layers vs 42 at 22 (layers run one after another)"],
    cost: "4 variants, about 20 min on a Colab L4", anchor: "18-architecture-step-seven-deep-and-thin-or-wide-and-shallow" },
  { when: "2026-11-11 13:51", kind: "tie", area: "Inference", title: "Grouped-query attention",
    decision: "Let several query heads share one key/value head.",
    why: "The KV cache, not the weights, fills memory in long-context generation.",
    result: { label: "3 of 6 key/value heads", from: 2.251, to: 2.252, extra: "KV cache 2× smaller for free; 2 heads 3× (+0.013), 1 head 6× (+0.033)" },
    calc: ["KV cache per token = 2 × layers × KV heads × 64 × 4 bytes", "6 heads: 2 × 6 × 6 × 64 × 4 = 18,432 bytes; 3 heads: 9,216",
      "Llama 3.1 8B at 128k tokens (16-bit): 16 GB with 8 KV heads, 64 GB without"],
    cost: "4 variants, about 13 min on a Colab L4", anchor: "17-architecture-step-six-grouped-query-attention" },
  { when: "2026-11-11 13:02", kind: "win", area: "Training", best: true, title: "A learning-rate sweep",
    decision: "Train at 5 peak learning rates, with and without QK-norm, and keep the best: 1.8e-3.",
    why: "Tripling the learning rate had helped more than QK-norm, so every earlier A/B ran too cautiously.",
    result: { label: "Best validation loss by learning rate", steps: [["6e-4", 2.336], ["1.8e-3", 2.251], ["3e-3", 2.376], ["6e-3", 2.550], ["1.2e-2", 2.553]], extra: "with QK-norm; without it, 1.2e-2 gives 3.325" },
    calc: ["warm up over 100 updates, cosine decay to a tenth", "QK-norm at 1.2e-2 costs 0.30 over the best; without it, 1.06"],
    cost: "8 variants, about 25 min on a Colab L4", anchor: "16-a-learning-rate-sweep-and-what-qk-norm-is-really-for" },
  { when: "2026-11-11 12:35", kind: "win", area: "Architecture", best: true, title: "QK-norm, and A/B tests on a cloud GPU",
    decision: "Normalize each head's queries and keys before attention. Move A/B tests to Colab with ab_test.py.",
    why: "Keeps attention scores in range; known for stability at high learning rates.",
    result: { label: "Validation loss, 1,000 updates (L4)", from: 2.369, to: 2.336, extra: "the Colab baseline reproduced the Mac's within 0.004" },
    calc: ["+ 2 × head size = 128 parameters per layer", "A/B variant: about 25 min on the Mac → 2½ min on an L4"],
    cost: "4 variants, about 15 min on a Colab L4", anchor: "15-architecture-step-five-qk-norm-and-moving-the-ab-tests-to-a-gpu" },
  { when: "2026-11-11 11:41", kind: "tie", area: "Architecture", best: true, title: "No biases",
    decision: "Drop every bias vector, as Llama does.",
    why: "With normalization before each layer, biases add little.",
    result: { label: "Validation loss, 1,000 updates", from: 2.366, to: 2.365, extra: "29k fewer parameters" },
    calc: ["29,949,312 − 29,920,512 = 28,800 parameters"], cost: "1 run, about 25 min on the Mac", anchor: "14-architecture-step-four-no-biases" },
  { when: "2026-11-11 11:12", kind: "win", area: "Architecture", best: true, title: "A gated feed-forward layer (SwiGLU)",
    decision: "silu(x W) · (x V), with the hidden layer narrowed to keep the parameter count.",
    why: "Multiplicative gating; used by Llama, Mistral and Qwen.",
    result: { label: "Validation loss, 1,000 updates", from: 2.475, to: 2.366, extra: "the baseline's loss reached ~30% sooner" },
    calc: ["hidden width = 8/3 × 384 = 1,024", "3 × 384 × 1,024 = 1,179,648 = 2 × 384 × 1,536: same size as GELU's"],
    cost: "1 run, about 25 min on the Mac", anchor: "13-architecture-step-three-a-gated-feed-forward-layer-swiglu" },
  { when: "2026-11-11 10:44", kind: "tie", area: "Architecture", title: "RMSNorm",
    decision: "Scale-only normalization instead of LayerNorm.",
    why: "Fewer operations; Llama's choice.",
    result: { label: "Validation loss, 1,000 updates", from: 2.475, to: 2.483, extra: "kept as an option. Training first crawled at 17 s/step: the Mac was out of memory" },
    calc: ["output logits = 32 × 256 × 50,257 × 4 bytes = 1.65 GB per batch", "fix: 4 micro-batches of 8 = same gradient, a quarter of the memory → 1.5 s/update"],
    cost: "1 run on the Mac, after two that had to be restarted", anchor: "12-architecture-step-two-rmsnorm" },
  { when: "2026-11-11 06:08", kind: "win", area: "Architecture", best: true, title: "Rotary positions (RoPE) and a sliding KV cache",
    decision: "Rotate queries and keys by their position instead of adding a learned position vector.",
    why: "Attention that depends on distance, not absolute slots; and a KV cache that can slide past the window.",
    result: { label: "Validation loss, 1,000 updates", from: 2.631, to: 2.475, extra: "past the window: cache never rebuilt (1.47× vs 1.14× over no cache)" },
    calc: ["the position table goes: 256 × 384 = 98,304 fewer parameters", "q·k after rotation depends only on m − n"],
    cost: "2 runs, 25–37 min each on the Mac", anchor: "11-architecture-step-one-rotary-positions-rope" },
  { when: "2026-11-10 22:20", kind: "tool", area: "Tooling", title: "Resume training",
    decision: "Save the full training state at every evaluation; finish or extend any run.",
    why: "Long runs die: a closed laptop, a dropped Colab session.",
    result: { label: "Tested", text: "a run stopped at step 25 resumed there; a 60-step run extended to 90 kept its history" },
    calc: ["state ≈ weights + AdamW's two moment buffers ≈ 3 × the weights", "Large (124M): 124M × 4 bytes × 3 ≈ 1.5 GB"],
    cost: "Mac only", anchor: "10-never-losing-a-run-resume-training" },
  { when: "2026-11-10 21:35", kind: "tool", area: "Tooling", title: "A model library and an evaluation suite",
    decision: "Keep every model; test all of them the same way.",
    why: "Each run replaced the last, and results were measured differently.",
    result: { label: "Finding", text: "each model is best on its own kind of text; every undertrained model loops (69–98% greedy repetition)" },
    calc: ["held-out loss: 64 fixed windows × 256 tokens per dataset", "6 prompts × 100 tokens, greedy and Balanced"],
    cost: "15–40 s per model on the Mac", anchor: "9-measuring-progress-a-model-library-and-an-evaluation-suite" },
  { when: "2026-11-10 18:06", kind: "win", area: "Inference", title: "Better decoding",
    decision: "Top-p, min-p, repetition penalties; Balanced (temperature 0.8, top-p 0.9, repetition 1.15) as the default.",
    why: "The FineWeb model looped: copying from context is the easiest prediction.",
    result: { label: "Repeated 4-token runs (FineWeb model)", from: 73, to: 0, unit: "%" },
    calc: ["top-p 0.9: keep the smallest set of tokens whose probabilities add up to 90%", "repetition 1.15: divide the logit of every token already used by 1.15"],
    cost: "no training", anchor: "8-decoding-why-the-model-repeated-itself" },
  { when: "2026-11-10 17:27", kind: "tie", area: "Inference", title: "The KV cache, measured",
    decision: "Keep each layer's keys and values so each new token costs one position, not the whole context.",
    why: "Generation re-read the full context for every token.",
    result: { label: "Medium model, 328 tokens", text: "M5: 13.8 s → 8.8 s (1.58×). L4: 0.92–0.98×, slightly slower: small models are launch-bound on a big GPU" },
    calc: ["KV cache per token = 2 × layers × width × 4 bytes", "Small: 2 × 6 × 384 × 4 = 18,432 bytes; a full 256-token window = 4.7 MB"],
    cost: "no training", anchor: "7-inference-the-kv-cache-measured-honestly" },
  { when: "2026-11-10 20:49", kind: "win", area: "Data", title: "Harder data: Mini FineWeb-Edu, 30,000 steps",
    decision: "Make educational web pages the default dataset; train the Large model for 30,000 steps.",
    why: "Children's stories are easy; real small models train on web text.",
    result: { label: "Validation loss on web text", text: "3.603 (perplexity ~37): fluent, encyclopedic, and confidently wrong" },
    calc: ["tokens seen = 30,000 × 32 × 256 = 246M", "loss only compares within the same data"],
    cost: "162 min on a Colab L4 (bfloat16)", anchor: "6-training-longer-then-on-harder-data" },
  { when: "2026-11-10 13:51", kind: "win", area: "Training", title: "Train longer: 20,000 steps",
    decision: "The Medium model (51M) for 20,000 steps with the fixed settings.",
    why: "With every step now an update, longer runs pay off.",
    result: { label: "TinyStories validation loss", from: 2.214, to: 1.583, extra: "coherent stories that end by themselves" },
    calc: ["tokens seen = 20,000 × 32 × 256 = 164M"], cost: "101 min on a Colab T4", anchor: "6-training-longer-then-on-harder-data" },
  { when: "2026-11-10 11:50", kind: "fix", area: "Training", best: true, title: "The big fix: the model was undertrained",
    decision: "A separator between stories, one update per step, learning rate 6e-4, 256-token context, no dropout, clipping 1.0.",
    why: "Training loss ≈ validation loss: underfitting, not memorizing. And 2,000 steps had been only 250 updates.",
    result: { label: "Validation loss after 2,000 steps (T4)", from: 4.527, to: 2.214, extra: "with a model less than half the size (51M vs 124M)" },
    calc: ["updates = steps ÷ accumulation: 2,000 ÷ 8 = 250; 20,000 ÷ 32 = 625", "a story is ~230 tokens: a 128-token window never saw its beginning"],
    cost: "two short runs on a Colab T4", anchor: "5-the-big-fix-the-model-was-undertrained" },
  { when: "2026-11-10 09:02", kind: "fix", area: "Tooling", title: "Training on a cloud GPU",
    decision: "A notebook for Colab and Kaggle that brings the model home; float16 on GPUs without real bfloat16.",
    why: "Long runs need a bigger GPU than a laptop's.",
    result: { label: "On a Colab T4", text: "1.4 s per step → 3.86 steps/s (5×); writing the token files: 16 min → seconds" },
    calc: ["the T4 reports bfloat16 as supported but emulates it", "compute capability 8.0+ (A100, L4) → bfloat16; otherwise float16 + gradient scaler"],
    cost: "free T4", anchor: "4-training-on-a-cloud-gpu" },
  { when: "2026-11-09 22:46", kind: "tool", area: "Tooling", title: "A web app that shows every step",
    decision: "Dataset, tokenizer, batches, model, attention, training and generation, with real numbers from your data.",
    why: "A training script hides everything worth learning.",
    result: { label: "Built", text: "13-stage walkthrough of the transformer; a parameter count that matches PyTorch exactly" },
    calc: ["Small: 50,257 × 384 (embedding) + 6 × 1.77M (blocks) + 768 = 30.0M"], cost: "Mac only", anchor: "3-making-it-understandable-the-web-app" },
  { when: "2026-11-09 17:36", kind: "fix", area: "Training", best: true, title: "The Apple GPU, and a learning rate that rose",
    decision: "Train on the Mac's GPU (Metal); fix the schedule's minimum.",
    why: "The minimum (5e-4) was above the peak (1e-4), so 'decay' made the learning rate grow.",
    result: { label: "Fixed", text: "cosine decay from the peak to a tenth of it, on the GPU instead of the CPU" },
    calc: ["before: lr goes 1e-4 → 5e-4 (rising)", "after: 6e-4 → 6e-5"], cost: "Mac only", anchor: "2-making-it-run-well-on-a-mac" },
  { when: "2026-11-09 16:54", kind: "tool", area: "Start", title: "Where we started",
    decision: "A 356-line nanoGPT-style script: 6 layers × 384, learned positions, LayerNorm, GELU, on TinyStories.",
    why: "A standard GPT-2 design to learn from.",
    result: { label: "Output", text: "“She was a big house. She wanted to go to a big her mommy”" },
    calc: ["learning rate 1e-4, accumulation 32, context 128, dropout 0.1, clipping 0.5", "no separator between stories; CPU only on a Mac"],
    cost: "–", anchor: "1-where-we-started" },
];

const J_KINDS = { win: ["Win", "good"], tie: ["Tie", ""], loss: ["Loss", "bad"], fix: ["Fix", "good"], tool: ["Tool", ""] };
const jState = { area: "All", bestOnly: false, rendered: false };

function journeyOpen() {
  if (!jState.rendered) { jState.rendered = true; renderJourney(); }
}

function jLadder() {
  // the same budget throughout: the Small model, 1,000 updates on TinyStories, seed 42
  const steps = [["GPT-2 style", 2.631, "learned positions, LayerNorm, GELU, biases, lr 6e-4 (Mac)"], ["+ RoPE", 2.475, "A1"],
    ["+ SwiGLU", 2.366, "A3"], ["+ no biases", 2.365, "A4"], ["+ QK-norm", 2.336, "A5 (Colab L4 from here)"], ["+ learning rate 1.8e-3", 2.251, "sweep"],
    ["+ 4 experts, 2 per token", 2.200, "A10"]];
  const max = steps[0][1], min = 2.1;
  return `<div class="j-ladder">${steps.map(([label, v, note], i) => {
    const d = i ? v - steps[i - 1][1] : null;
    return `<div class="j-lrow"><span class="j-llabel">${esc(label)}<span class="muted small">${esc(note)}</span></span>
      <div class="j-ltrack"><div class="j-lbar ${i === steps.length - 1 ? "best" : ""}" style="width:${((v - min) / (max - min)) * 100}%"></div></div>
      <b>${v.toFixed(3)}</b><span class="j-ldelta ${d != null && d < -0.005 ? "good" : ""}">${d == null ? "" : (d > 0 ? "+" : "−") + Math.abs(d).toFixed(3)}</span></div>`;
  }).join("")}</div>`;
}

function jResult(r) {
  if (r.steps) {
    const vals = r.steps.map((s) => s[1]), lo = Math.min(...vals), hi = Math.max(...vals);
    return `<div class="j-res"><div class="muted small">${esc(r.label)}</div>${r.steps.map(([k, v]) => `<div class="j-step ${v === lo ? "best" : ""}"><span>${esc(k)}</span>
      <div class="j-mini"><div style="width:${hi === lo ? 100 : 25 + ((v - lo) / (hi - lo)) * 75}%"></div></div><b>${v.toFixed(3)}</b></div>`).join("")}
      ${r.extra ? `<div class="muted small">${esc(r.extra)}</div>` : ""}</div>`;
  }
  if (r.from != null) {
    const u = r.unit || "", f = (v) => (u ? `${v}${u}` : v.toFixed(3));
    const better = r.to < r.from;
    return `<div class="j-res"><div class="muted small">${esc(r.label)}</div><div class="j-ft"><span>${f(r.from)}</span><span class="j-arrow">→</span><b class="${better ? "good-text" : r.to > r.from ? "bad-text" : ""}">${f(r.to)}</b>
      <span class="muted small">${better ? "−" : r.to > r.from ? "+" : "±"}${u ? Math.abs(r.to - r.from) + u : Math.abs(r.to - r.from).toFixed(3)}</span></div>
      ${r.extra ? `<div class="muted small">${esc(r.extra)}</div>` : ""}</div>`;
  }
  return `<div class="j-res"><div class="muted small">${esc(r.label)}</div><div>${esc(r.text)}</div></div>`;
}

function renderJourney() {
  const areas = ["All", ...new Set(JOURNEY.map((j) => j.area))];
  const items = [...JOURNEY].sort((x, y) => y.when.localeCompare(x.when)).filter((j) => (jState.area === "All" || j.area === jState.area) && (!jState.bestOnly || j.best));
  let lastDay = null;
  const timeline = items.map((j) => {
    const day = j.when.slice(0, 10), [kind, cls] = J_KINDS[j.kind];
    const dayHead = day !== lastDay ? `<div class="j-day">${new Date(day + "T12:00").toLocaleDateString(undefined, { weekday: "long", month: "long", day: "numeric" })}</div>` : "";
    lastDay = day;
    return `${dayHead}<article class="j-item ${j.kind}">
      <div class="j-dot"></div>
      <div class="j-card">
        <div class="j-head"><span class="j-time">${esc(j.when.slice(11))}</span><span class="tag ${cls}">${kind}</span><span class="tag ghost">${esc(j.area)}</span>${j.best ? '<span class="tag best">in the best model</span>' : ""}</div>
        <h3>${esc(j.title)}</h3>
        <div class="j-body">
          <div class="j-text"><p><b>Decision.</b> ${esc(j.decision)}</p><p><b>Why.</b> ${esc(j.why)}</p>${j.after ? `<p><b>Then.</b> ${esc(j.after)}</p>` : ""}</div>
          ${jResult(j.result)}
        </div>
        <div class="j-calc">${j.calc.map((c) => `<code>${esc(c)}</code>`).join("")}</div>
        <div class="j-foot"><span><b>Cost</b> ${esc(j.cost)}</span><a href="${j.anchor.startsWith("http") ? j.anchor : `${J_DOC}#${j.anchor}`}" target="_blank" rel="noopener">Read the full story →</a></div>
      </div></article>`;
  }).join("");
  $("#journey").innerHTML = `
    <div class="card j-hero">
      <div class="j-hero-main">
        <div class="eyebrow">Best model so far</div>
        <h2>RoPE + SwiGLU + QK-norm + 4 experts</h2>
        <p class="muted">The Small model (30M parameters, 37M used per token, 51M in all), learning rate 1.8e-3, no biases. After 5,000 updates on TinyStories:</p>
        <div class="j-kpis">
          <div><span class="muted small">Validation loss</span><b>1.732</b><span class="small">GPT-2 style: 1.903</span></div>
          <div><span class="muted small">Same quality in</span><b>25–38%</b><span class="small">fewer updates</span></div>
          <div><span class="muted small">Greedy repetition</span><b>7.9%</b><span class="small">GPT-2 style: 27%</span></div>
          <div><span class="muted small">Next-token accuracy</span><b>58.1%</b><span class="small">GPT-2 style: 55.9%</span></div>
        </div>
        <p class="small muted">The cost: generation at 66 tokens/s vs 232 in this unfused implementation, the next thing to work on.</p>
      </div>
      <div class="j-hero-side"><div class="muted small">Five times longer (5,000 updates)</div>
        ${[["GPT-2 style", 1.903], ["+ learning rate", 1.851], ["+ architecture", 1.782], ["+ experts", 1.732]].map(([k, v], i) => `<div class="j-step ${i === 3 ? "best" : ""}"><span>${k}</span><div class="j-mini"><div style="width:${25 + ((v - 1.732) / (1.903 - 1.732)) * 75}%"></div></div><b>${v.toFixed(3)}</b></div>`).join("")}
      </div>
    </div>

    <div class="card">
      <div class="card-head"><h3>The decisions that built it</h3><span class="muted small">same budget: Small model, 1,000 updates on TinyStories, seed 42 · lower is better</span></div>
      ${jLadder()}
      <p class="hint">Each change was tested on its own against the best model before it, and kept only if it helped. Ties (RMSNorm) and losses (deeper shapes, layer sharing, multi-token prediction) are in the timeline below. The Mac and Colab baselines agree within 0.004.</p>
    </div>

    <div class="card">
      <div class="card-head"><h3>Numbers behind the model</h3><span class="muted small">for the Small model, 6 layers × 384</span></div>
      <div class="j-numbers">
        <div><b>Parameters</b><code>50,257 × 384 + 6 × 1.77M + 768 ≈ 30.0M</code><span class="muted small">64% of them are the vocabulary embedding</span></div>
        <div><b>One weight update</b><code>32 sequences × 256 tokens = 8,192 tokens</code><span class="muted small">1,000 updates = 8.2M tokens; 5,000 = 41M</span></div>
        <div><b>Memory to train</b><code>30M × 16 bytes ≈ 480 MB</code><span class="muted small">weights, gradients and AdamW's two moments, plus activations</span></div>
        <div><b>Output layer per batch</b><code>32 × 256 × 50,257 × 4 bytes = 1.65 GB</code><span class="muted small">why the Mac trains in micro-batches of 8</span></div>
        <div><b>KV cache per token</b><code>2 × 6 layers × 384 × 4 bytes = 18,432 bytes</code><span class="muted small">4.7 MB for a full 256-token window</span></div>
        <div><b>Mixture of experts</b><code>29.9M + 6 × 3 × 1.18M = 51.2M; 37.0M per token</code><span class="muted small">4 experts, 2 used per token</span></div>
      </div>
    </div>

    <div class="card">
      <div class="card-head"><h3>Timeline</h3><span class="muted small">${items.length} of ${JOURNEY.length} decisions · newest first</span></div>
      <div class="j-filters">${areas.map((a) => `<button class="chip-btn ${a === jState.area ? "on" : ""}" data-j-area="${esc(a)}">${esc(a)}</button>`).join("")}
        <label class="inline small"><input type="checkbox" id="j-best" ${jState.bestOnly ? "checked" : ""}> Only what's in the best model</label></div>
      <div class="j-timeline">${timeline || '<p class="muted">Nothing matches.</p>'}</div>
    </div>`;
}

document.addEventListener("click", (e) => {
  const a = e.target.closest("[data-j-area]");
  if (a) { jState.area = a.dataset.jArea; renderJourney(); }
});
document.addEventListener("change", (e) => {
  if (e.target.id === "j-best") { jState.bestOnly = e.target.checked; renderJourney(); }
});

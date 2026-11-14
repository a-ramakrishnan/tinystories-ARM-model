# The plan

The project's roadmap: what's done, what's next and who does it. It's kept up to date as work lands; the results
of each step are in [Analysis](analysis.md) and the story in [the journey](journey.md).

**Base model** (what new work builds on): *Large 124M best design · FineWeb-Edu + Cosmopedia · 500M tokens ·
context 512* (`mini-fineweb-edu-202611130718-14ed26`): held-out FineWeb-Edu loss 3.549; ARC-Easy 44.0%, PIQA 62.6%,
HellaSwag 29.7%, LAMBADA 15.7%. The yardstick is OpenAI's GPT-2 small (3.401; 43.8%, 62.9%, 31.1%, 32.6%). The
earlier TinyStories and WikiText models are kept as the history of how we got here.

*Last updated: 14 November 2026, 13:30 IST.*

## Where things stand (14 November, afternoon)

Three things are running or queued; nothing else needs a decision until they finish.

| Lane | What | Status |
|---|---|---|
| Colab L4 | **F2c** the long diffusion run on FineWeb-Edu (492M tokens) | running: 20,500 of 60,000 updates at 12:48, likelihood bound 5.32 and falling (the short run ended at 5.269); done about **14:40** |
| Colab L4, next | **D1** instruction tuning: `sft.py` + `sft_colab.ipynb` | code ✅ and tested on the Mac; the full run (~1 h 15 min) starts when the L4 is free, about 14:45 → done about 16:15 |
| Colab L4, after that | **E2** stage 2: ~2.5B tokens, continued from the base | starts after D1 (this evening); ~20 h on an L4, resumable from Drive; an A100, if one is free, cuts it to ~8 h (E1) |
| Mac | **F5** benchmarks for the two diffusion models | running (slow: shares the Mac's GPU); results this afternoon → Benchmarks page, README, journey |

**What comes next, in order** (suggested):

1. **Evaluate D1** (today, Claude): the chat model against the base on the same questions, its held-out answer loss,
   the five benchmarks (SFT shouldn't lower them much), and greedy repetition, the number D2 will target. Add it to the
   library; Chat mode on Generate is ready for it.
2. **Close F2c and F5** (today, Claude): the long run's bound, generation quality and benchmarks next to the short run
   and the autoregressive models; download the weights only with the author's OK.
3. **While E2 trains (~1 day), work that needs no Colab GPU** (Claude): **F4** the diffusion vs autoregressive
   comparison, then the visualizations **H1** (diffusion half of the side-by-side page) and **H2** (noising slider,
   denoising player). F3's last piece, diffusion in Generate, folds into H1.
4. **After E2**: evaluate stage 2 against the base and GPT-2; if better, it becomes the base, and D1 is re-run on it
   (~1 h).
5. **Then D2** (DPO against repetition) on the new chat model, then **D3** (GRPO).
6. **Any time:** C1, the author's GPTQ (Claude adapts it in C2 when it's ready); G, GitHub Pages, once there's more
   to show.

| # | Phase / step | Who | Status |
|---|---|---|---|
| **A** | **Foundations** | | ✅ done |
| A1 | Fix the training set-up, move to Mini FineWeb-Edu, evaluation suite, model library, Lab, Journey page | Claude | ✅ |
| A2 | Architecture track A1–A10 (RoPE, SwiGLU, no biases, QK-norm, learning rate, GQA, experts, …), each A/B tested | Claude | ✅ |
| A3 | Inference track I0–I8 (1.8–3.5× faster generation, DRY, contrastive and beam search, 8/4-bit weights, speculative decoding, batching) | Claude | ✅ |
| A4 | Stage 1: the best design at 124M (held-out 3.691 vs 3.735 for GPT-2 style) | Claude | ✅ |
| A5 | Explorer (Flow, Diagram, Play the data flow); Benchmarks page (GPT-2 reproduces its published scores) | Claude | ✅ |
| **B** | **Faster training and more kinds of data** (13 Nov) | | ✅ done |
| B1 | L1, faster training: `torch.compile` gives 1.7× on an L4, on by default | Claude | ✅ |
| B2 | Cosmopedia subsets, dataset mixing, streaming the first N rows, 512-token context | Claude | ✅ |
| B3 | Data run: 500M tokens, FineWeb-Edu + Cosmopedia, 4 h 20 min on a Colab L4 | Colab | ✅ |
| B4 | Evaluation suite and the five benchmarks | Claude | ✅ |
| B5 | Compare with the first base and GPT-2; it became the new base model | Claude, then the author | ✅ |
| **C** | **Inference** | | the author's GPTQ (C1) whenever ready |
| C1 | GPTQ quantization with llm-compressor's `oneshot`, written by hand for a standard Hugging Face model (e.g. GPT-2 small) | the author; Claude helps when asked | open |
| C2 | Adapt C1 to our architecture (a local Hugging Face wrapper; RoPE, QK-norm, SwiGLU, tied output layer, 512 context); load the compressed weights back; compare with round-to-nearest `quantize.py` on loss, benchmarks, size and speed; add to the app | Claude | after C1 |
| C3 | The author's other inference ideas | the author decides | after C2 |
| **D** | **Post-training** (on the base model) | | |
| D1 | Supervised fine-tuning on instructions (L7) on the base model: smol-smoltalk conversations in a plain `User:` / `Assistant:` format, loss on the assistant's words only, about 1 h on a Colab L4; a Chat mode in Generate | Claude | **in progress (14 Nov)**: `sft.py` (prepare / train / chat) and `sft_colab.ipynb` written and tested end to end on the Mac; Chat mode on Generate ✅. smol-smoltalk is long for 512 tokens: ~12% of conversations dropped, over half cut to the turns that fit. The full run is next on the L4, after F2c. First, before stage 2 (the author's decision); redone on the stage 2 model later, it's cheap |
| D2 | DPO against repetition (greedy repetition is 25% today) | Claude | after D1 |
| D3 | GRPO with automatically checkable rewards | Claude | after D2 |
| **E** | **Scaling the data** (the data run showed more data helps) | | |
| E1 | Measure training speed on an A100 (none was free on 13 Nov) | Claude | when one is free |
| E2 | Stage 2: a long, resumable run on FineWeb-Edu *full* + Cosmopedia (~2.5B tokens, ~20 tokens per parameter for 124M), continued from the base model; about 20 h on an L4 or ~8 h on an A100, checkpoints on Drive | Colab | **next after D1** (go-ahead given 14 Nov) |
| E3 | L2 (low-memory tokenizing on the Mac), L4 (quality filter) | Claude | as needed |
| **F** | **A masked diffusion language model** | | in progress |
| F1 | The diffusion model (L8), written and first trained on TinyStories | the author | ✅ first version generates text (`diffusion_models/v1_original.py`) |
| F1b | Make it modular in `diffusion_models/` (GPT-2 tokenizer, the project's token files, plain text, a likelihood bound to compare with), plus a Colab notebook; the `colab` run mode matches the 51M autoregressive TinyStories model (164M tokens) | Claude | ✅ |
| F1c | Train it on TinyStories on Colab and check it generates | Colab | ✅ 164M tokens, 49 min on an L4 (59,000 tokens/s): likelihood bound **2.525** nats/token (the 51M autoregressive model's loss: 1.583). It learned only word frequencies for ~2,500 updates, then used context (6e-4 beat 2e-3 in a 2,000-update check). Coherent stories once revealed tokens are kept fixed while sampling (v1's re-sampling of every position broke the final text; kept as `sampler="v1"`) |
| F2 | Train the same diffusion model on Mini FineWeb-Edu (small) on Colab | Claude | ✅ 164M tokens, 50 min on an L4: likelihood bound **5.269** nats/token, still falling (the plateau lasted ~3,000 updates). Confident-first sampling collapses into one repeated word ("fish, fish, fish"); revealing in random order at temperature 0.8 (`sampler="random"`) gives varied, loosely fluent text without much coherence: 164M tokens is little for FineWeb-Edu at 51M parameters (our base AR model saw 500M tokens at 124M) |
| F2b | Fix the diffusion models' generation | Claude | ✅ `diffusion_models/quality.py` scores the text on held-out prompts (repetition, the base model's loss on it). Confident-first sampling copied neighbouring words (43% of 4-token runs repeated on FineWeb-Edu); a repetition penalty of 1.3 (the new default) brings it to 4% on FineWeb-Edu and 0.4% on TinyStories at about real text's judge loss. Blocks, left-to-right order, re-masking, random order, top-p and noise-level choices were measured too (README). FineWeb-Edu text is now on topic but loosely connected: the model's limit; a longer run would be the next lever |
| F2c | A longer FineWeb-Edu run: `colab-long`, 60,000 updates = 492M tokens (3×), about 2 h 30 min on an L4 | Colab | running (14 Nov): bound 6.06 at 6,000 updates, 5.59 at 12,000, 5.32 at 20,000; done about 14:40 |
| F3 | Integrate it into the app (Lab, Generate, Explorer) | Claude | in progress: ✅ a **Diffusion** page (Explore): pick a run from `diffusion_models/runs/`, a held-out prompt, tokens, steps and sampler, and replay the denoising (masked █ positions filling in, a step × position grid of when each was revealed, speed, repetition), plus a table of the trained runs (bound, training, quality scores). Next: diffusion in Generate, folded into the side-by-side page (H1) |
| F4 | Compare on both datasets: the TinyStories diffusion model against the autoregressive *Medium 51M · TinyStories · 20k steps* (the same size, tokens and schedule as the `colab` run mode), and the FineWeb-Edu one against the base model; on the prompt set's TinyStories and FineWeb-Edu prompts: the outputs, loss (the diffusion loss is an upper bound, ours is exact), steps and speed | Claude | after F3 |
| F5 | **Benchmark the diffusion models** like the autoregressive ones: the held-out losses (the likelihood bound on the same held-out sets), the five benchmarks (ARC-Easy, PIQA, HellaSwag, Winogrande scored by each choice's likelihood bound; LAMBADA by masking the last word), and generation quality (repetition, the AR base model's loss on the text, speed), shown on the Benchmarks page next to the AR models and GPT-2 | Claude | in progress: `diffusion_models/benchmarks.py` written (each choice scored by a Monte Carlo likelihood bound, 8 maskings); the Benchmarks page shows diffusion runs; the full runs on both models are going on the Mac (14 Nov). Generation quality ✅ (F2b) |
| **G** | **Publish on GitHub Pages** | | |
| G1 | The documentation as a website | Claude | open |
| G2 | Browser-only versions of the Journey timeline, the Benchmarks results, the model comparison, and the Explorer replaying the base model's real numbers for a few saved prompts | Claude | open |
| G3 | Optional: live generation in the browser with a small model (ONNX / transformers.js or WebGPU) | Claude | optional |
| G4 | A GitHub Actions workflow to build and deploy the site, and links from the README | Claude | open |
| **H** | **Visualizations** (some ideas borrowed from Vizuara's diffusion explainer, built in our own style) | | |
| H1 | **One page, side by side: what happens in an autoregressive model and in a diffusion model.** The same prompt in two lanes (left to right one token per step vs all positions refined in parallel over k steps), a step × position grid for each, causal vs bidirectional attention, the training objective (next token vs masked tokens), and the numbers (steps, speed, loss, repetition) | Claude | autoregressive half ✅ (the *Writing text* page: fixed held-out prompts, step player, step × position grid, loss on the real continuation, saved results); diffusion half next, while E2 trains (the Diffusion page's reveal grid is a start) |
| H2 | **A separate set of illustrations for diffusion models:** a noising slider (a sentence going from clean to fully masked as t grows, with the mask schedule), a denoising player (from all [MASK], positions revealed by confidence, with a steps-vs-quality slider), the loss on masked positions only, and the architecture flow with bidirectional attention and the time embedding | Claude | after H1 |
| H3 | Numbered stage cards in the Flow player (autoregressive): what happens at each stage, its formula and the example's real numbers | Claude | can start any time |

**Open choices**

- ~~Stage 2 (E2) before or after post-training (D)~~ decided on 14 Nov: instruction tuning (D1) first, on today's
  base, then stage 2; D1 is redone on the stage 2 model afterwards.
- Where GitHub Pages (G) goes: it's independent of the rest, and most worth doing once there's more to show.
- What C3 covers depends on the author's ideas.
- The visualizations (H) can replay saved data, so they can also go on GitHub Pages (G2); H1's diffusion half and
  H2 come with the diffusion integration (F2–F4).

**Why the order:** the benchmarks showed that the remaining gap to GPT-2 is mostly data (more of it, and more kinds),
so data and inference come before further architecture work. Post-training changes behaviour more than knowledge;
it's cheap (~1 h), so it's done on today's base first and redone on the stage 2 base. GitHub Pages can only serve static files, so it shows saved results; training, the Lab
and live generation stay in the local app.

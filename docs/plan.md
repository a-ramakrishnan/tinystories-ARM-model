# The plan

The project's roadmap: what's done, what's next and who does it. It's kept up to date as work lands; the results
of each step are in [Analysis](analysis.md) and the story in [the journey](journey.md).

**Base model** (what new work builds on): *Large 124M best design · FineWeb-Edu + Cosmopedia · 500M tokens ·
context 512* (`mini-fineweb-edu-202611130718-14ed26`): held-out FineWeb-Edu loss 3.549; ARC-Easy 44.0%, PIQA 62.6%,
HellaSwag 29.7%, LAMBADA 15.7%. The yardstick is OpenAI's GPT-2 small (3.401; 43.8%, 62.9%, 31.1%, 32.6%). The
earlier TinyStories and WikiText models are kept as the history of how we got here.

*Last updated: 13 November 2026 (evening).*

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
| **C** | **Inference** | | next |
| C1 | GPTQ quantization with llm-compressor's `oneshot`, written by hand for a standard Hugging Face model (e.g. GPT-2 small) | the author; Claude helps when asked | open |
| C2 | Adapt C1 to our architecture (a local Hugging Face wrapper; RoPE, QK-norm, SwiGLU, tied output layer, 512 context); load the compressed weights back; compare with round-to-nearest `quantize.py` on loss, benchmarks, size and speed; add to the app | Claude | after C1 |
| C3 | The author's other inference ideas | the author decides | after C2 |
| **D** | **Post-training** (on the base model) | | |
| D1 | Supervised fine-tuning on instructions (L7), with a Chat mode in Generate | Claude | after C |
| D2 | DPO against repetition (greedy repetition is 25% today) | Claude | after D1 |
| D3 | GRPO with automatically checkable rewards | Claude | after D2 |
| **E** | **Scaling the data** (the data run showed more data helps) | | |
| E1 | Measure training speed on an A100 (none was free on 13 Nov) | Claude | when one is free |
| E2 | Stage 2: a long, resumable run on FineWeb-Edu *full* + Cosmopedia (~2.5B tokens) | Colab, the author's go-ahead | open |
| E3 | L2 (low-memory tokenizing on the Mac), L4 (quality filter) | Claude | as needed |
| **F** | **A masked diffusion language model** | | in progress |
| F1 | The diffusion model (L8), written and first trained on TinyStories | the author | **in progress** (about halfway) |
| F2 | Train the same diffusion model on Mini FineWeb-Edu (small) on Colab | Claude | after F1 |
| F3 | Integrate it into the app (Lab, Generate, Explorer) | Claude | with F2 |
| F4 | Compare on both datasets: the TinyStories diffusion model against an autoregressive model trained on TinyStories at a similar size (to confirm which), and the FineWeb-Edu one against the base model; on the prompt set's TinyStories and FineWeb-Edu prompts: the outputs, loss (the diffusion loss is an upper bound, ours is exact), steps and speed | Claude | after F2 |
| **G** | **Publish on GitHub Pages** | | |
| G1 | The documentation as a website | Claude | open |
| G2 | Browser-only versions of the Journey timeline, the Benchmarks results, the model comparison, and the Explorer replaying the base model's real numbers for a few saved prompts | Claude | open |
| G3 | Optional: live generation in the browser with a small model (ONNX / transformers.js or WebGPU) | Claude | optional |
| G4 | A GitHub Actions workflow to build and deploy the site, and links from the README | Claude | open |
| **H** | **Visualizations** (some ideas borrowed from Vizuara's diffusion explainer, built in our own style) | | |
| H1 | **One page, side by side: what happens in an autoregressive model and in a diffusion model.** The same prompt in two lanes (left to right one token per step vs all positions refined in parallel over k steps), a step × position grid for each, causal vs bidirectional attention, the training objective (next token vs masked tokens), and the numbers (steps, speed, loss, repetition) | Claude | autoregressive half ✅ (the *Writing text* page: fixed held-out prompts, step player, step × position grid, loss on the real continuation, saved results); diffusion half after F1 |
| H2 | **A separate set of illustrations for diffusion models:** a noising slider (a sentence going from clean to fully masked as t grows, with the mask schedule), a denoising player (from all [MASK], positions revealed by confidence, with a steps-vs-quality slider), the loss on masked positions only, and the architecture flow with bidirectional attention and the time embedding | Claude | after F1 |
| H3 | Numbered stage cards in the Flow player (autoregressive): what happens at each stage, its formula and the example's real numbers | Claude | can start any time |

**Open choices**

- Stage 2 (E2) before or after post-training (D): post-training works better on a stronger base, but stage 2 is the
  bigger GPU commitment.
- Where GitHub Pages (G) goes: it's independent of the rest, and most worth doing once there's more to show.
- What C3 covers depends on the author's ideas.
- The visualizations (H) can replay saved data, so they can also go on GitHub Pages (G2); H1's diffusion half and
  H2 come with the diffusion integration (F2–F4).

**Why the order:** the benchmarks showed that the remaining gap to GPT-2 is mostly data (more of it, and more kinds),
so data and inference come before further architecture work. Post-training changes behaviour more than knowledge, so
it waits for a stronger base. GitHub Pages can only serve static files, so it shows saved results; training, the Lab
and live generation stay in the local app.

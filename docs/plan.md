# The plan

The project's roadmap: what's done, what's next and who does it. It's kept up to date as work lands; the results
of each step are in [Analysis](analysis.md) and the story in [the journey](journey.md).

**Base model** (what new work builds on): *Large 124M best design · FineWeb-Edu + Cosmopedia · 500M tokens ·
context 512* (`mini-fineweb-edu-202611130718-14ed26`): held-out FineWeb-Edu loss 3.549; ARC-Easy 44.0%, PIQA 62.6%,
HellaSwag 29.7%, LAMBADA 15.7%. The yardstick is OpenAI's GPT-2 small (3.401; 43.8%, 62.9%, 31.1%, 32.6%). The
earlier TinyStories and WikiText models are kept as the history of how we got here.

*Last updated: 13 November 2026.*

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
| F1 | The diffusion model (L8) | the author | **in progress** |
| F2 | Integrate it into the app (Lab, Generate, Explorer) and compare it with the base model | Claude | after F1 |
| **G** | **Publish on GitHub Pages** | | |
| G1 | The documentation as a website | Claude | open |
| G2 | Browser-only versions of the Journey timeline, the Benchmarks results, the model comparison, and the Explorer replaying the base model's real numbers for a few saved prompts | Claude | open |
| G3 | Optional: live generation in the browser with a small model (ONNX / transformers.js or WebGPU) | Claude | optional |
| G4 | A GitHub Actions workflow to build and deploy the site, and links from the README | Claude | open |

**Open choices**

- Stage 2 (E2) before or after post-training (D): post-training works better on a stronger base, but stage 2 is the
  bigger GPU commitment.
- Where GitHub Pages (G) goes: it's independent of the rest, and most worth doing once there's more to show.
- What C3 covers depends on the author's ideas.

**Why the order:** the benchmarks showed that the remaining gap to GPT-2 is mostly data (more of it, and more kinds),
so data and inference come before further architecture work. Post-training changes behaviour more than knowledge, so
it waits for a stronger base. GitHub Pages can only serve static files, so it shows saved results; training, the Lab
and live generation stay in the local app.

# Theory — why it works

The repo teaches you to *run* policies; this section is *why* they work, so members build intuition, not just muscle memory. Read the one for whatever you're training.

## 📚 Documents, by model

Find your model, read its page. Each is self-contained apart from the prerequisites it names at the top.

| Model | Document | What it is | Read when |
|---|---|---|---|
| **ACT** | ⭐ [act-why](act-why.md) | **story — read start to finish** | before training ACT, or before changing it |
| **ACT** | [act-shapes](act-shapes.md) | reference — every tensor traced | choosing a batch size, or debugging OOM |
| **Diffusion Policy** (shared theory) | ⭐ [diffusion-policy-why](diffusion-policy-why.md) | **story — read start to finish** | before touching any diffusion policy |
| **Diffusion Policy — CNN** | [diffusion-policy-shapes](diffusion-policy-shapes.md) | reference — every tensor, VRAM, config traps | training the default UNet backbone |
| **Diffusion Policy — Transformer** | [diffusion-policy-transformer](diffusion-policy-transformer.md) | story + reference, with our measured results | training `diffusion_transformer`, or reading our A/B |
| **SmolVLA** | [smolvla](smolvla.md) | story + reference, measured from source | training or evaluating a VLA |

**Suggested reading order if you are new:** `diffusion-policy-why` → `diffusion-policy-shapes` → `act-why` → `act-shapes` → `diffusion-policy-transformer` → `smolvla`.

**If you are looking for something to change** (each page ends with, or flags, what is load-bearing versus historical accident): [ACT §Part VI what is load-bearing, and what is accident](act-why.md#part-vi-what-is-load-bearing-and-what-is-accident) is the most direct.

**If you only want the caveats** (things that will mislead you, all measured on this repo): [DP-T §7 the memory mask blocks action 0 from the newest frame](diffusion-policy-transformer.md#7-the-two-masks-and-one-surprising-consequence) · [DP-T §15 the resume trap](diffusion-policy-transformer.md#15-the-resume-trap-steps-is-the-cosine-horizon) · [DP-T §16 what `eval_loss` does and does not tell you](diffusion-policy-transformer.md#16-what-eval_loss-does-and-does-not-tell-you) · [SmolVLA §Part VIII caveats for our rig](smolvla.md#part-viii-caveats-for-our-rig).

## The policy family tree (pick your baseline)

Every imitation policy answers one question: **how do you model `P(action | obs)` without averaging away the valid modes** (multimodality), and do you predict one step or a **chunk**? Cheapest → most expressive:

| Family | Method(s) | Idea | Multimodal? |
|---|---|---|---|
| regress | BC-MLP / BC-ConvMLP | `obs → action`, L2 loss | ✗ averages modes |
| mixture | LSTM-GMM (BC-RNN) | RNN + Gaussian-mixture head | a few modes |
| energy | IBC | learn `E(obs,act)`, search for the min | ✓ but unstable |
| tokenize | BeT → VQ-BeT | discretize actions (k-means → residual VQ-VAE) + transformer | ✓ (VQ-BeT ~5× faster than diffusion) |
| latent chunk | ACT | CVAE emits an action chunk | ✓ |
| generative | Diffusion / flow | denoise noise → action chunk | ✓ strongest, slower |
| retrieval | VINN | frozen visual features + k-NN over demos, no action net | ✓ |
| scale | RT-1 / VLAs | tokenized multi-task transformer | ✓ huge |

**Action chunking (ACT's idea) is the trick almost every modern method adopts.** ACT · Diffusion · VLAs are detailed below; the rest are the classic baselines you meet in results tables.

**Who benchmarks against whom** — so our [leaderboard](../evaluation/README.md) stays comparable to the papers:
- Diffusion Policy → LSTM-GMM, IBC, BeT
- ACT → BC-ConvMLP, BeT, RT-1, VINN
- VQ-BeT → BC, BeT, Diffusion Policy
- Patch Policy → ACT, VQ-BeT, Diffusion Policy, OpenVLA-OFT

Baselines not linked below: [IBC](https://arxiv.org/abs/2109.00137) · [BeT](https://arxiv.org/abs/2206.11251) · [VINN](https://arxiv.org/abs/2112.01511) · [RT-1](https://arxiv.org/abs/2212.06817).

## Tokenized actions (BeT → VQ-BeT)
Turn continuous actions into discrete "words" from a learned codebook, then language-model them with a transformer — **sample a mode instead of averaging**, in a single fast pass (VQ-BeT reports ~5× faster than diffusion). VQ-BeT tokenizes with a **residual VQ-VAE** (coarse + fine codes) plus a small continuous offset for fidelity.
- VQ-BeT paper: https://arxiv.org/abs/2403.03181
- Video explainer: https://youtu.be/V-zL7_jOo7w

## Imitation learning & action chunking (ACT)
Copy expert demos, but predict a **chunk** of future actions at once to avoid compounding error.
- ACT paper: https://arxiv.org/abs/2304.13705
- ⭐ **[ACT — why it is shaped like that](act-why.md)** — **start here.** The parameter ledger closes at 51,571,590, and three of those facts change how you read the architecture: **33.7% of the model never runs on the robot** (the CVAE encoder is training-only and its latent is forced to collapse by `kl_weight=10`), **the decoder is one layer because of an upstream bug that was reproduced deliberately**, and **the decoder's input tokens are literally zeros** — the 50 learned query embeddings are *questions*, and cross-attention over 902 observation tokens supplies every answer. Ends with what is load-bearing versus historical accident.
- **[ACT, shape by shape](act-shapes.md)** — the **reference**: every tensor from dataloader to action, traced from the installed lerobot. Read this before choosing a batch size: it shows why 2 cameras at 640×480 give 602 encoder tokens, and why the convolutional frontend (not attention) is what fills the GPU.

## Generative action heads (Diffusion Policy · flow matching)
Instead of regressing one action, **generate** an action trajectory from noise — captures the "many valid ways" (multimodality) of a task.
- Diffusion Policy: https://arxiv.org/abs/2303.04137
Two docs, and the split is deliberate — **one story you read, one reference you look things up in.**

- ⭐ **[Diffusion Policy — why it is shaped like that](diffusion-policy-why.md)** — **start here, read start to finish.** The whole architecture derived from one problem: regression outputs the *mean* of multimodal demonstrations, and the mean of two valid grasps is a collision. Part I is diffusion itself (why noise, why 100 small steps not one jump, why the multimodality lives in sampling instead of a latent that can collapse — as ours did). Part II is how it sees (the 1×1 conv, SpatialSoftmax, a 2×3 example you can check by hand, and the three ways a keypoint lies). Part III is how it generates (the sinusoidal embedding, FiLM, GroupNorm, the convolution arithmetic). Every section runs: problem → candidate designs → why the winner won → the shapes.
- **[Diffusion Policy, shape by shape](diffusion-policy-shapes.md)** — the **reference**: every tensor measured from the installed lerobot, VRAM and latency, config traps that will mislead you, and the ACT comparison table.
- **[Diffusion Policy — the transformer backbone](diffusion-policy-transformer.md)** — the **same diffusion process with the denoiser swapped**: a 1-D conv UNet over time becomes an 8-layer transformer decoder. Covers our port (vendored from real-stanford, no lerobot fork), parameter arithmetic that reproduces the paper's Table 8, and the results we measured: the regularisation A/B (Table 8's recipe *underfits* our 120 episodes), bucketed evaluation by noise level, and two traps — a resume that silently changes the LR schedule, and what held-out loss does *not* predict about the arm.
- Flow matching (used by pi0 and SmolVLA): the model learns a velocity field that flows noise → action in a few steps. The straight-line path makes Euler integration **exact** — see [smolvla §15](smolvla.md#15-euler-integration-is-exact).

## Vision-Language-Action models (VLAs)
A pretrained vision-language model + an action head, so the robot inherits web-scale "common sense" and follows language.
- **[SmolVLA — a frozen VLM with an action expert bolted on](smolvla.md)** — **start here for VLAs.** Two transformer towers of different widths running in lockstep, meeting in a shared 15×64 head space. Every number measured from source: the parameter ledger (only **1.6 M** of 450 M are new, and 47 M are a dead `lm_head`), the image pipeline (25% of our camera frame is padding), the block-causal mask, the self/cross alternation, and why flow matching's loss measures something DDPM's does not.
- pi0 (VLA + flow matching): https://arxiv.org/abs/2410.24164
- SmolVLA paper: https://huggingface.co/papers/2506.01844
- NVIDIA GR00T N1: https://arxiv.org/abs/2503.14734

## RL fine-tuning (learning from experience)
Imitation plateaus; RL lets a policy improve from its own successes/failures. The hard part on flow/diffusion policies is the intractable action-likelihood.
- RECAP / π*0.6 (advantage-conditioned RL): https://arxiv.org/abs/2511.14759

> Club members: deeper internal notes on these (the RL-VLA landscape, the π-series, flow-matching shapes) live in the club knowledge base — ask a maintainer for the current links.

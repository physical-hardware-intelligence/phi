# Theory — why it works

The repo teaches you to *run* policies; this section is *why* they work, so members build intuition, not just muscle memory. Read the one for whatever you're training.

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
- **[ACT, shape by shape](act-shapes.md)** — every tensor from dataloader to action, traced from the installed lerobot. Read this before choosing a batch size: it shows why 2 cameras at 640×480 give 602 encoder tokens, and why the convolutional frontend (not attention) is what fills the GPU.

## Generative action heads (Diffusion Policy · flow matching)
Instead of regressing one action, **generate** an action trajectory from noise — captures the "many valid ways" (multimodality) of a task.
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- ⭐ **[Diffusion Policy — why it is shaped like that](diffusion-policy-why.md)** — start here. The whole architecture derived from one problem: regression outputs the *mean* of multimodal demonstrations, and the mean of two valid grasps is a collision. Explains why noise, why 100 small steps rather than one jump, and why the multimodality lives in sampling instead of in a latent that can collapse (as ours did).
- **[Diffusion Policy, shape by shape](diffusion-policy-shapes.md)** — every tensor traced from the installed lerobot: why the image is crushed to 64 numbers per camera, why the UNet convolves over *time* rather than space, and why inference costs 100 forward passes (measured 9× over budget on the Mac).
- Flow matching (used by pi0): the model learns a velocity field that flows noise → action in a few steps.

## Vision-Language-Action models (VLAs)
A pretrained vision-language model + an action head, so the robot inherits web-scale "common sense" and follows language.
- pi0 (VLA + flow matching): https://arxiv.org/abs/2410.24164
- SmolVLA: https://huggingface.co/papers/2506.01844
- NVIDIA GR00T N1: https://arxiv.org/abs/2503.14734

## RL fine-tuning (learning from experience)
Imitation plateaus; RL lets a policy improve from its own successes/failures. The hard part on flow/diffusion policies is the intractable action-likelihood.
- RECAP / π*0.6 (advantage-conditioned RL): https://arxiv.org/abs/2511.14759

> Club members: deeper internal notes on these (the RL-VLA landscape, the π-series, flow-matching shapes) live in the club knowledge base — ask a maintainer for the current links.

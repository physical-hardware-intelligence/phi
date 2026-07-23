# Theory — why it works

The repo teaches you to *run* policies; this section is *why* they work, so members build intuition, not just muscle memory. Read the one for whatever you're training.

## Imitation learning & action chunking (ACT)
Copy expert demos, but predict a **chunk** of future actions at once to avoid compounding error.
- ACT paper: https://arxiv.org/abs/2304.13705

## Generative action heads (Diffusion Policy · flow matching)
Instead of regressing one action, **generate** an action trajectory from noise — captures the "many valid ways" (multimodality) of a task.
- Diffusion Policy: https://arxiv.org/abs/2303.04137
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

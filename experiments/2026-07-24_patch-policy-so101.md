# Experiment: patch-policy-so101

**Date:** 2026-07-24   ·   **Author:** Yashvardhan Gupta   ·   **Ladder:** L4
**Status:** PLANNED — gated on the arm being calibrated + a dataset recorded + an ACT baseline in hand.

## Question / hypothesis
Does feeding **dense, frozen-ViT patch features** to a tiny policy (the "Patch Policy" recipe) beat our ACT / Diffusion-Policy baseline on a **precise** SO-101 task, at a fraction of the parameters and compute?

Hypothesis: yes, and the gap should be **largest on spatial / contact-precise tasks** (insertion-style), because global-pooled features throw away *where* things are. Paper: *Patch Policy: Efficient Embodied Control via Dense Visual Representations*, arXiv [2607.18236](https://arxiv.org/abs/2607.18236) (Zhou, Cui, Langford, Tan, **LeCun**, **Pinto** — NYU / Meta-FAIR). Project + code: https://patch-policy.github.io

Why it fits Φ specifically:
- Reported: **+40% vs pooled features**; beats fine-tuned OpenVLA-OFT (7.6B) by **18% using ~0.7% of its params**; **converges in 6.5 GPU-hours on one L40S**; ~11 ms inference.
- That single-GPU / tiny-model regime is exactly our constraint (OMEN RTX 2060 — see [EdgeInfer](../../../wiki), Explorer HPC, Colab). This is our best shot at a "punches-above-its-compute" policy.
- Frozen backbone = drop-in; rides vision-model progress for free. Pairs with the [SO-101 setup playbook](../docs/robots/so-arm101/troubleshooting.md).

## Setup
- **Robot / task:** SO-101 follower + wrist cam + one fixed scene cam. Pick a **precision task** where the payoff should show (peg / cable-insertion analog, or a tight place-in-slot), plus one easy pick-place as a sanity task.
- **Dataset:** a recorded LeRobotDataset, ≥50 episodes, HF id `${HF_USER}/phi-<task>-v1` (see [teleop & data](../docs/robots/so-arm101/03-teleop-and-data.md)).
- **Policy under test:** frozen ViT patch features (**DINOv2** first; also try **V-JEPA 2** since it's a paper backbone and ties to our world-model thread) → block-causal transformer → Diffusion-Policy or VQ-BeT action head, action chunking + receding horizon.
- **Baseline:** `--policy.type=act` on the *same* dataset/task (the zoo's default first policy).
- **Hardware:** train on CUDA (OMEN 2060 / Explorer HPC), fixed seed. Record on the Mac.

## Exact commands (must be reproducible)
```bash
# BASELINE (already supported):
lerobot-train --dataset.repo_id=${HF_USER}/phi-<task>-v1 --policy.type=act \
  --policy.device=cuda --steps=20000 --wandb.enable=true

# PATCH POLICY: not a native LeRobot --policy.type yet.
#   Wire the authors' code (from patch-policy.github.io) as a policy on top of our
#   LeRobotDataset, OR reimplement the block-causal transformer over frozen DINOv2
#   patch tokens with a Diffusion/VQ-BeT head. Commit the config to configs/.
#   TODO: fill exact command once the code is integrated.
```

## Results
- Success rate (k/N) on the precision task + the sanity task, Patch Policy vs ACT.
- Params, GPU-hours, inference latency (ms) for each — the efficiency claim is half the point.
- Put the head-to-head on the [leaderboard](../docs/evaluation/README.md); model cards in `models/`.

## What we learned (honest — negative results count)
_(pending)_

## Next
1. Get to L1/L2 first: calibrate → teleop → record the dataset (blocked on the arm, delivered 2026-07-24, pickup Mon).
2. Train + eval the ACT baseline.
3. Integrate Patch Policy code; run the head-to-head; log it.
4. If it wins on our hardware → strong, concrete result to take back to the authors (Pinto / Bardes-V-JEPA outreach is staged in the wiki, deliberately held until this runs).

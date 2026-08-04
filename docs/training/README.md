# Training — the policy zoo

Training is **shared across robots**. You pick a policy with `--policy.type=...` and point it at a LeRobotDataset. Train on a **GPU box / Explorer HPC / HF Jobs**, not the Mac.

> Running on Northeastern's Explorer cluster? Read [hpc/explorer](../hpc/explorer.md) first — partitions, the cu130-vs-driver-545 trap, `/scratch` ESTALE, and measured ACT throughput. Job scripts are in [`configs/hpc/`](../../configs/hpc/).

> Reference: [lerobot-train](https://huggingface.co/docs/lerobot/en/il_robots#train-a-policy) · policy docs linked below.

## The zoo (what we support, in learning order)

| Policy | `--policy.type` | Install extra | What / when | Doc |
|---|---|---|---|---|
| **ACT** (Action Chunking Transformer) | `act` | base | ~80M imitation transformer; **start here** — trains in a few hours on 1 GPU, ~50 demos | https://huggingface.co/docs/lerobot/en/act |
| **Diffusion Policy** | `diffusion` | `[diffusion]` | generative action trajectories; strong on contact-rich, multimodal tasks | model card: https://huggingface.co/lerobot/diffusion_pusht |
| **SmolVLA** | `smolvla` | `[smolvla]` | HF's 450M vision-language-action model; language-conditioned, runs on consumer HW | https://huggingface.co/docs/lerobot/en/smolvla |
| **pi0 / pi0.5** | `pi0` / `pi05` | `[pi]` | Physical Intelligence flow-matching VLA foundation model; fine-tune from a base | https://huggingface.co/docs/lerobot/en/pi0 |

(LeRobot ships many more — GR00T, X-VLA, etc. — see its policy docs. We add them as members level up.)

**Research candidates (not yet wired):**
- **Patch Policy** — frozen-ViT *patch* features (keep the dense grid, don't pool) + a tiny block-causal transformer. Reportedly beats a 7.6B VLA by 18% at ~0.7% of the params, trains in ~6.5 GPU-hours on one GPU. A strong fit for our single-GPU regime. Planned head-to-head vs ACT: [`experiments/2026-07-24_patch-policy-so101.md`](../../experiments/2026-07-24_patch-policy-so101.md). Paper: [arXiv 2607.18236](https://arxiv.org/abs/2607.18236).

## Train ACT (the baseline every task gets first)
```bash
lerobot-train \
  --dataset.repo_id=${HF_USER}/phi-cube-v1 \
  --policy.type=act \
  --output_dir=outputs/train/act_phi-cube-v1 \
  --job_name=act_phi-cube-v1 \
  --policy.device=cuda \          # mps on Apple Silicon, cpu otherwise
  --steps=20000 \
  --wandb.enable=true \
  --policy.repo_id=${HF_USER}/act-phi-cube-v1
```
Checkpoints → `outputs/train/<name>/checkpoints/`. Resume:
```bash
lerobot-train --config_path=outputs/train/act_phi-cube-v1/checkpoints/last/pretrained_model/train_config.json --resume=true
```
Fine-tune a pretrained VLA instead of training from scratch:
```bash
lerobot-train --policy.path=lerobot/smolvla_base --dataset.repo_id=${HF_USER}/phi-cube-v1 --batch_size=64 --steps=20000
```
Cloud (HF Jobs): add `--job.target=a10g-small` (list flavors: `hf jobs hardware`) and `--save_checkpoint_to_hub=true`.

## The Φ rules
- **Every run is a committed config** in [`configs/`](../../configs/) + a fixed seed + the exact command in the [experiment write-up](../../experiments/). No orphan runs.
- **Every trained model gets a card** in [`models/`](../../models/) (id, dataset, policy, steps, eval score, known failure modes). Weights live on the Hub.
- Baseline first (ACT), then compare. Put the comparison on the [leaderboard](../evaluation/README.md).

## Theory
Why these architectures work (flow matching, VLAs, RL fine-tuning) → [theory](../theory/README.md).

**Before you pick `--batch_size` for ACT**, read [ACT, shape by shape](../theory/act-shapes.md) — measured tensor shapes, the token-count arithmetic (`300 × cameras + 2` at 640×480), where the VRAM actually goes, and why `chunk_size` is nearly free while adding a camera is not.

# Overview

Φ turns "I want to make a robot learn something" into a **repeatable pipeline** anyone in the club can follow.

## The engine: LeRobot
We build on [🤗 LeRobot](https://huggingface.co/docs/lerobot/en/index) — the open framework that handles motors, cameras, datasets, training, and inference. We pin it as a git submodule at [`external/lerobot`](../external/) and add our curation, evaluation, deployment, and docs on top. **When in doubt, the LeRobot docs are the source of truth**; this site is the *curated path* through them for the SO-101.

Key LeRobot references (bookmark these):
- Docs home: https://huggingface.co/docs/lerobot/en/index
- Installation: https://huggingface.co/docs/lerobot/en/installation
- Cheat sheet: https://huggingface.co/docs/lerobot/en/cheat-sheet
- Imitation-learning tutorial (SO-101): https://huggingface.co/docs/lerobot/en/il_robots
- SO-101 robot page: https://huggingface.co/docs/lerobot/en/so101
- LeLab (no-CLI web GUI): https://huggingface.co/docs/lerobot/en/lelab
- Quickstart notebook (auto-generates the exact commands for *your* setup): https://github.com/huggingface/lerobot/blob/main/examples/notebooks/quickstart.ipynb
- LeRobot Discord: https://discord.gg/s3KuuzsPFb

## The pipeline (and where each stage lives)

| Stage | What | Doc |
|---|---|---|
| Hardware | build vs buy, BOM, print, assemble, **safety** | [robots/so-arm101/01-hardware](robots/so-arm101/01-hardware.md) |
| Environment | install LeRobot (mac / cuda / HPC) | [robots/so-arm101/02-setup](robots/so-arm101/02-setup.md) |
| Bring-up | find-port → setup-motors → calibrate → teleop + cameras | [robots/so-arm101/02-setup](robots/so-arm101/02-setup.md) |
| Data | record datasets, LeRobotDataset format, QA, replay | [robots/so-arm101/03-teleop-and-data](robots/so-arm101/03-teleop-and-data.md) |
| Training | policy zoo: ACT · Diffusion · SmolVLA · pi0 | [training](../training/README.md) |
| Evaluation | standardized rollout protocol + leaderboard | [evaluation](../evaluation/README.md) |
| Deployment | on-robot / remote / edge inference | [deployment](../deployment/README.md) |
| Theory | *why* it works (flow matching, VLAs, RL-VLA) | [theory](../theory/README.md) |

## Two ways to work
- **CLI (recommended, fully configurable):** the `lerobot-*` commands, documented here.
- **LeLab (web GUI, no commands):** `uv tool install git+https://github.com/huggingface/leLab.git && lelab`. Does calibrate/teleop/record/train/replay in a browser. ⚠️ **SO-ARM101-only** and its **macOS/Apple-Silicon support is not documented — test before relying on it.** Great for onboarding L0/L1 members who aren't comfortable in a terminal yet.

## Hardware split (from our mastery plan)
- **MacBook = cockpit**: connect the arm over USB, calibrate, teleop, record, replay, run small policies (device = `mps`).
- **Training is offloaded**: a CUDA box, the Northeastern **Explorer** HPC, or **HF Jobs** (`--job.target=...`). A fanless Mac throttles — don't train on it.

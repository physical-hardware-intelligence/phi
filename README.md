# Φ — Physical Hardware Intelligence

**The foundation repository for Northeastern Silicon Valley's robotics SIG (Φ).**
Everything we build — every robot, dataset, policy, training run, evaluation, and deployment — lives here, curated and documented so any student can get started *without starting from scratch*.

> Built on top of [🤗 LeRobot](https://github.com/huggingface/lerobot) (Apache-2.0). We don't reinvent the engine — we add the **curriculum, curation, evaluation, deployment, and reproducibility** layer on top, and keep it open.

---

## Why this repo exists

Robot-learning tutorials are scattered, and the good end-to-end ones are paywalled. Φ is the **open** alternative: one systematic, versioned pipeline from **hardware → data → training → evaluation → deployment**, with real docs and a graded on-ramp for new members.

It is a **multi-robot monorepo**. Our first robot is the **SO-ARM101** arm; future robots (e.g. a quadruped) slot in as new folders under [`docs/robots/`](docs/robots/) — the training/evaluation/deployment infrastructure above them is shared.

## Repository map

```
phi/
├── docs/                     # the curriculum (mkdocs site)
│   ├── 00-overview.md
│   ├── robots/so-arm101/     # robot #1: hardware, setup, teleop+data, troubleshooting
│   ├── training/             # policy zoo (shared): ACT · Diffusion · SmolVLA · pi0
│   ├── evaluation/           # standardized eval protocol + leaderboard
│   ├── deployment/           # on-robot · remote · edge inference
│   └── theory/               # why it works (links to concept notes)
├── external/lerobot/         # the engine (git submodule, pinned)
├── src/phi/                  # thin library: data · train · eval · deploy · utils
├── configs/                  # versioned, seeded training/eval configs
├── tasks/                    # task registry (spec + dataset card + eval rubric)
├── datasets/  models/        # cards + registry (weights/data on HF Hub)
├── experiments/              # dated experiment write-ups
├── env/                      # reproducible environments (mac / cuda)
└── tests/                    # unit + smoke
```

## Quickstart (5 minutes, no robot needed)

```bash
git clone --recurse-submodules <this-repo-url> phi && cd phi
# if you forgot --recurse-submodules:  git submodule update --init --recursive
conda env create -f env/environment.mac.yml   # or env/environment.cuda.yml on a GPU box
make help                                      # see every one-command entrypoint
```
Then read [`docs/00-overview.md`](docs/00-overview.md) and follow the **member ladder** below.

## From arm to trained policy (the L0 → L2 run)

The end-to-end thread once you have the SO-101 in hand. Each step links to the doc with the exact commands. **Record on the Mac, train on a GPU box.**

1. **Install** the env — the Quickstart above, or [`02-setup`](docs/robots/so-arm101/02-setup.md).
2. **Find USB ports** (`lerobot-find-port`, once per arm) → [`02-setup`](docs/robots/so-arm101/02-setup.md).
3. **Set motor IDs**, one servo at a time → [`02-setup`](docs/robots/so-arm101/02-setup.md).
4. **Calibrate** follower + leader — reuse the same `--id` everywhere, **commit the file** → [`02-setup`](docs/robots/so-arm101/02-setup.md).
5. **Teleop check** — move the leader, the follower mirrors → **you're L1** → [`02-setup`](docs/robots/so-arm101/02-setup.md).
6. **Add cameras** — wrist + one fixed scene cam → [`02-setup`](docs/robots/so-arm101/02-setup.md).
7. **Record ≥50 demos**, varying object position + lighting → [`03-teleop-and-data`](docs/robots/so-arm101/03-teleop-and-data.md).
8. **QA + replay** the dataset — no dropped frames, no camera swap → [`03-teleop-and-data`](docs/robots/so-arm101/03-teleop-and-data.md).
9. **Train ACT** on the GPU box (`--policy.type=act`) → **you're L2** → [`training`](docs/training/README.md).
10. **Roll out + score** on the robot (`lerobot-rollout`) → [`evaluation`](docs/evaluation/README.md).
11. **Log it** — commit the config + seed, write the [`experiment`](experiments/), add a [`model card`](models/).

Stuck at any step → [`troubleshooting`](docs/robots/so-arm101/troubleshooting.md).

## The member ladder (how you level up)

| Level | You can… | Start here |
|---|---|---|
| **L0 Onboard** | run a pretrained policy in replay | [overview](docs/00-overview.md) |
| **L1 Operator** | calibrate + teleoperate + record a dataset | [robots/so-arm101/02-setup](docs/robots/so-arm101/02-setup.md) |
| **L2 Trainer** | train ACT on your data + evaluate it | [training](docs/training/README.md), [evaluation](docs/evaluation/README.md) |
| **L3 Contributor** | add a task or a config; close a good-first-issue | [tasks/TEMPLATE](tasks/TEMPLATE.md) |
| **L4 Researcher** | run a new policy / edge / RL experiment + write it up | [experiments/TEMPLATE](experiments/TEMPLATE.md) |
| **L5 Maintainer** | own a module, review PRs | [CONTRIBUTING](CONTRIBUTING.md) |

## The pipeline

```
Hardware → Environment → Calibration/PID → Teleop → Data → Dataset QA
   → Training (policy zoo) → Evaluation (standard protocol) → Deployment (on-robot/remote/edge) → iterate
        └──────────── reproducibility + docs + tests wrap every stage ───────────┘
```

## Status

🌱 **Phase 0 (scaffold).** Docs, environments, and structure are in place; setup/training/eval docs carry the real LeRobot commands. **Phase 1** (first reproducible SO-101 ACT run) begins when the arm arrives.

## License & credits

Apache-2.0 (see [LICENSE](LICENSE)). Built on [LeRobot](https://github.com/huggingface/lerobot) and the [SO-ARM100/101](https://github.com/TheRobotStudio/SO-ARM100) hardware project. Φ is a Student Interest Group at Northeastern University, Silicon Valley — not affiliated with or branded by the university.

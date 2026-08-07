# ACT on cubes+cylinder — three generalization axes, three owners

**Status**: 📋 Planned, splits defined · **Created**: 2026-08-06
**Dataset**: [`phi_so101_cubes_cylinder_v1`](../datasets/phi_so101_cubes_cylinder_v1.md) — 120 episodes, 66,873 frames, [on the Hub](https://huggingface.co/datasets/BrutalCaesar/phi_so101_cubes_cylinder_v1)
**Owners**: Parv (object holdout) · Yash (position holdout + no-CVAE baseline) · Sai (camera pairs + no-CVAE baseline)

## Why this experiment is different from the last two

The 8-bin run asked for spatial generalization from ~15 episodes per position and never got a rollout
number. The redcube run had no held-out anything, so it could only measure fit.

This dataset is the first that is **well-posed and splittable**: only one bin is on the table at a
time, so the target is always the single visible container — no destination multimodality for the L1
objective to average away. And 120 episodes over 3 objects × 2 containers gives real axes to hold out.

**Every configuration below reports a held-out number.** That is the point.

## Shared configuration — do not vary these

| | |
|---|---|
| Policy | ACT, LeRobot 0.6.0 defaults except as noted |
| `chunk_size` / `n_action_steps` | **50** |
| `batch_size` | **8** |
| `steps` | **100,000** |
| `save_freq` | **20,000** (5 checkpoints) |
| `seed` | 1000 |
| `--dataset.video_backend` | `pyav` (torchcodec is broken on Explorer) |

Everything else is LeRobot default, so results stay comparable across the three owners.

> **Note on chunk 50.** The ACT paper's ablation peaks at k=100 **at 50 Hz**, i.e. **2.0 seconds**. We
> record at 30 fps, so the equivalent is **chunk 60**. Chunk 50 = 1.67 s is slightly short of the
> paper's optimum in time terms. Fixed at 50 here for cross-owner comparability; worth one run at 60
> later if results are marginal.

## Axis 1 — Parv: unseen object (does size transfer?)

Train on two objects, hold out the third entirely. The interesting variable is **cube size**:

| Object | Dimensions |
|---|---|
| red cube | **25 × 25 × 25 mm** |
| white cube | **45 × 45 × 45 mm** |
| cylinder | (yellow/orange) |

| Run | Held out | Train episodes | Frames | Epochs @100k | Tests |
|---|---|---|---|---|---|
| **P-A** | red cube (0-39) | `range(40,120)` — 80 eps | 39,601 | 20.2 | **big → small** |
| **P-B** | white cube (80-119) | `range(0,80)` — 80 eps | 48,596 | 16.5 | **small → big** |

Eval each on its held-out 40 episodes' object, on hardware.

**The asymmetry is the finding to watch.** A gripper trained only on a 45 mm cube has never closed to
a 25 mm width; a gripper trained only on 25 mm has never opened wide enough for 45 mm. These are not
symmetric failures, and the direction that breaks is informative about whether ACT is learning
grasp *geometry* or a memorized closure trajectory.

## Axis 2 — Yash: unseen object position

Hold out **5 episodes from each of the 6 object×container blocks** — 30 held out, 90 train. Same
objects, same containers, unseen start positions.

Within-block offsets are 0 / 5 / 10 for red / cylinder / white cube, with the paired white-container
group always **+20** from its cardboard counterpart.

| Block | Held out |
|---|---|
| red cube · cardboard | 0-4 |
| red cube · white bin | 20-24 |
| cylinder · cardboard | 45-49 |
| cylinder · white bin | 65-69 |
| white cube · cardboard | 90-94 |
| white cube · white bin | 110-114 |

**Train: 90 episodes, 49,969 frames, 16.0 epochs at 100k steps.**

```
--dataset.episodes="[5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,115,116,117,118,119]"
```

Held-out set for evaluation:

```
[0,1,2,3,4,20,21,22,23,24,45,46,47,48,49,65,66,67,68,69,90,91,92,93,94,110,111,112,113,114]
```

## Axis 3 — Sai: camera pair, on Yash's split

Same 90-episode split as Axis 2, three two-camera configurations:

| Run | Cameras |
|---|---|
| S-1 | `wrist` + `front` |
| S-2 | `wrist` + `top` |
| S-3 | `front` + `top` (no wrist) |

**Camera keys in this dataset are correct** — wire by name, no transposition. Do not copy the swapped
mapping from `train_8bin_v1.sbatch`.

> 🚨 **Training loss has failed to separate camera pairs twice.** On 8-bin the three pairs landed
> within 0.001; on redcube they were identical to three decimals *including the run with no wrist
> camera at all*. **Do not report a camera-pair conclusion from loss.** Only per-condition rollouts
> can rank these, and if they come out equal again that is itself the result: the cameras are not the
> binding constraint.

## The no-CVAE baseline (Yash + Sai)

One run each with **`--policy.use_vae=false`** — plain behavioural cloning, no latent, L1 only.
Measured earlier: 51.57M → 34.20M parameters.

This is the most valuable single run in the whole experiment, and here is why.

The ACT paper's own ablation: removing the CVAE objective costs **almost nothing on scripted data**
("because dataset is fully deterministic") but drops **human-demonstration performance from 35.3% to
2%**. Our data is human. And we measured our latent as **collapsed** (KL ≈ 0 at `kl_weight=10.0`),
which is functionally the no-CVAE condition already.

So one of two things is true, and this run distinguishes them:

- **`use_vae=false` matches the CVAE run** → the latent was genuinely carrying nothing, our demos are near-deterministic given the observation, and collapse is benign. Earlier verdict stands.
- **`use_vae=false` is much worse** → the latent *was* doing work, and its collapse in our other runs is a real defect that could explain the disappointing 8-bin results better than any of the task-design theories.

Also worth measuring while these run: **the actual KL magnitude, not just "≈ 0"**. A KL of 0.05 nats
carries information; 1e-6 does not. We have been treating those as the same thing.

## Eval protocol

- **20 scored rollouts per held-out condition**, reported **per condition, never averaged**
- Parv: per object, on the held-out object
- Yash/Sai: per held-out position group
- A rollout is a success only if the object ends **inside** the container
- Record the **checkpoint** used — with 5 checkpoints per run, the 20k one may beat the 100k one, and train loss cannot tell you which

⚠️ **Standing debt: 12 models trained so far, zero scored rollouts.** This plan adds at least 7 more.
The training is not the bottleneck and has not been for two weeks.

## What counts as a result

- A held-out success rate per condition, **including 0%**, with a stated reason
- The **direction** of the size-transfer asymmetry (P-A vs P-B)
- Whether `use_vae=false` matches or collapses
- A camera-pair ranking **from rollouts**, or an explicit finding that they do not differ

## Resolved

**Red-cube holdout: episodes 0-4 and 20-24** (confirmed 2026-08-06). The spoken description contained
both "the first five from cardboard … add 20" and "from 25"; the first reading was chosen because it
matches the +20 pairing the other two objects use. Splits above are final.

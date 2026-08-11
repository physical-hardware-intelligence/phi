# ACT on recovery data: image augmentation ON vs OFF

**Date**: 2026-08-10
**Script**: [`configs/hpc/train_recovery.sbatch`](../configs/hpc/train_recovery.sbatch)
**Dataset**: [`phi_so101_cubes_cylinder_recovery_v1`](../datasets/phi_so101_cubes_cylinder_recovery_v1.md) — 143 eps / 81,943 frames
**Control**: the `cvae_3cam` run from [`2026-08-06_act-cubes-cylinder-splits.md`](2026-08-06_act-cubes-cylinder-splits.md)

## What is being tested

Two runs, one variable.

| Task | `image_transforms.enable` | Run name |
|---|---|---|
| 0 | `false` | `act_recovery_3cam_noaug_chunk50` |
| 1 | `true` | `act_recovery_3cam_aug_chunk50` |

Everything else is fixed: ACT + CVAE (`kl_weight=10.0`), 3 cameras, `chunk_size=50`, batch 8, 100k steps, checkpoints every 20k, `seed=1000`.

Both differ from the `cvae_3cam` control only by the **23 added recovery episodes**, and the holdout is byte-identical to it, so all three runs are directly comparable on the same 30 clean episodes.

## Split

113 train (**all 23 recovery episodes included**) + 30 clean holdout.

```
holdout:  0-4 · 20-24 · 45-49 · 65-69 · 90-94 · 110-114
```

`eval_split = 0.209`, and it is **not free to round** — `ceil(143 × s)` must equal 30. `0.2098` and `0.21` both give 31 and would silently hold out an extra episode. The script asserts this before `srun`, along with "all 23 recovery episodes are in the training portion" and "none are in eval."

The dataset deliberately keeps **one task string**: `make_train_eval_datasets` splits per task with `math.ceil`, which is always ≥ 1, so a distinct recovery task string would force a recovery episode into eval.

Episodes 128 / 134 / 140 sit exactly on the 750-frame cap. **Resolved 2026-08-10 — all three completed their recovery, none truncated, all kept.**

## What `enable=true` turns on

Three of six transforms sampled per frame (`max_num_transforms=3`), all weight 1.0:

| | type | range |
|---|---|---|
| brightness | ColorJitter | (0.8, 1.2) |
| contrast | ColorJitter | (0.8, 1.2) |
| saturation | ColorJitter | (0.5, 1.5) |
| hue | ColorJitter | (−0.05, 0.05) |
| sharpness | SharpnessJitter | (0.5, 1.5) |
| **affine** | RandomAffine | degrees (−5, 5), **translate (0.05, 0.05)** |

## Pre-registered prediction

Written before the runs so it can't be rationalised after.

**The five photometric transforms should help.** Our webcams cannot lock exposure, so lighting genuinely varies between sessions; invariance to it is real invariance.

**`affine` is the suspect.** `translate=0.05` is ±5% of the frame — **±32 px of 640 ≈ 2 cm at workspace scale, about one cube width** — and `RandomAffine` moves the *image* while the *action label* stays put. It therefore teaches "these joint targets are still correct even though the object appears a cube to the left," which is precisely the invariance a precision grasp must **not** have.

⇒ If task 1 loses to task 0, that is **not** evidence against augmentation as a category. Re-run the affine-off ablation first:

```bash
sbatch --export=ALL,AFFINE_OFF=1 --array=1 configs/hpc/train_recovery.sbatch
```

**Syntax note (verified against draccus in lerobot 0.6.0, 2026-08-10):** `--dataset.image_transforms.tfs.affine.weight=0.0` is rejected as an unrecognized argument, and passing a partial dict **replaces** `tfs` rather than merging into it — you silently lose the other five transforms. All six entries must be listed. `AUG_TFS` in the sbatch does this correctly.

## What these numbers can and cannot tell you

**Can** — "did adding recovery data hurt clean performance?" Compare either run's `eval_loss` against the `cvae_3cam` control on the same 30 clean episodes.

**Can** — "did augmentation help fit?" Task 0 vs task 1 `eval_loss`. Expect task 1's *train* loss to be higher (harder problem); that is not failure.

**Cannot** — "did the policy get better at recovering?" **No loss can answer this**, because the 30 held-out episodes are all clean and contain no induced failure. Measuring recovery needs a rollout where the object is deliberately displaced or removed mid-episode and you count whether the policy re-approaches.

That rollout protocol is the actual gate on this experiment, and it does not exist yet — see [docs/evaluation](../docs/evaluation/README.md). **19+ models trained, zero scored rollouts** is the standing bottleneck; two more checkpoints do not move it.

## 🚨 Before any rollout

Copy the committed calibration onto the machine first:

```bash
cp configs/calibration/robots/so_follower/phi_follower.json \
   ~/.cache/huggingface/lerobot/calibration/robots/so_follower/
```

A policy emits joint angles in the calibration frame it was trained in. Running it against a different one **fails silently** — no error, no warning, the gripper just goes to the wrong place. Confirmed on the arm 2026-08-10: this exact dataset's policy mis-grasped on Sai's machine and was fixed by copying the file, no retraining. See [so-arm101 setup §3c](../docs/robots/so-arm101/02-setup.md#3c-moving-a-trained-policy-to-someone-elses-machine).

## Results

Both tasks **COMPLETED**, exit 0. Array `9066699`. `eval_loss` = full L1 + KL on the 30 clean held-out episodes.

| Run | Aug | @20k | @40k | @60k | @80k | @100k | wall |
|---|---|---|---|---|---|---|---|
| `recovery_3cam_noaug` (task 0, d4052) | off | 0.2222 | 0.2056 | 0.2058 | 0.2053 | **0.2052** | 1 h 50 m |
| `recovery_3cam_aug` (task 1, d4055) | on | 0.2149 | 0.2112 | 0.2056 | **0.2046** | 0.2088 | 2 h 54 m |

Off-checkpoint evals (every 10k) fill in the shape:

| step | 10k | 30k | 50k | 70k | 90k |
|---|---|---|---|---|---|
| aug off | 0.2278 | 0.2078 | 0.2073 | 0.2069 | 0.2048 |
| aug on | 0.2211 | 0.2133 | 0.2079 | 0.2046 | 0.2037 |

### Reading it honestly

**The augmentation is a wash on this metric.** Aug leads early (10k–20k), trails through the middle (30k–50k), and is level from 60k on. Best-ever values are 0.2048 (off, 90k) vs 0.2037 (on, 90k) — a 0.5% gap on a single seed. **No winner is readable here.**

Two structural limits, both predicted before the runs and both still binding:

1. The holdout is entirely clean, so **nothing in this number tests recovery**.
2. L1 over a 50-step chunk is dominated by transit motion, so it is close to **blind to grasp precision** — which is exactly what the `affine` concern was about. The pre-registered concern is therefore neither confirmed nor refuted; it is unmeasured.

**Cost:** augmentation added 57% wall time (2 h 54 m vs 1 h 50 m) for no measurable held-out gain.

**One catch worth acting on:** the aug run's 100k checkpoint (0.2088) is *worse* than its 80k (0.2046). Pushing `last` blindly would have shipped the weaker weights.

### Published

| Run | HF id | Revision |
|---|---|---|
| aug off | [`act_so101_cubcyl_recovery_chunk50_noaug_3cam`](https://huggingface.co/BrutalCaesar/act_so101_cubcyl_recovery_chunk50_noaug_3cam) | `main` = `cd93421` (100k) |
| aug on | [`act_so101_cubcyl_recovery_chunk50_aug_3cam`](https://huggingface.co/BrutalCaesar/act_so101_cubcyl_recovery_chunk50_aug_3cam) | `main` = `539ef57` (100k) · `step-80000` = `2ffdc6f` |

`main` is the fixed 100k step for both so the head-to-head is not a selection on a noisy metric; the 80k branch exists so a rollout can test whether checkpoint choice matters more than the augmentation. Cards: [noaug](../models/act_so101_cubcyl_recovery_chunk50_noaug_3cam.md) · [aug](../models/act_so101_cubcyl_recovery_chunk50_aug_3cam.md).

### Still open

- **The `cvae_3cam` control row is missing.** Comparing against it on the same 30 clean episodes is what answers "did adding recovery data damage clean performance?" — pull its `eval_loss` from the 2026-08-06 run.
- **Zero scored rollouts.** Three publishable checkpoints, no success rate. Everything above is loss, and loss cannot rank these.

# Diffusion Policy on the recovery dataset: vision-encoder A/B

**Date**: 2026-08-12 · **Job**: array `9088486` · **W&B**: project `phi_dp`
**Script**: [`configs/hpc/train_diffusion_recovery.sbatch`](../configs/hpc/train_diffusion_recovery.sbatch)
**Dataset**: [`phi_so101_cubes_cylinder_recovery_v1`](../datasets/phi_so101_cubes_cylinder_recovery_v1.md) — 143 eps / 81,943 frames
**Paper**: Chi et al., *Diffusion Policy* ([2303.04137](https://arxiv.org/abs/2303.04137)), Table 7 real-world rows

_Written before any result was known. Predictions below are pre-registered so they cannot be retrofitted._

## What is being tested

| Task | `use_group_norm` | `pretrained_backbone_weights` | |
|---|---|---|---|
| **0** `paper` | `true` | `None` (scratch) | the paper's §3.2 recipe |
| **1** `lerobot` | `false` | ImageNet | lerobot's default |

**The pairing is forced.** lerobot raises `ValueError: You can't replace BatchNorm in a pretrained model without ruining the weights!` — correctly, since GroupNorm discards BatchNorm's running statistics while the conv filters were fitted to be consumed *with* them.

## Parameters, and the source of each

| Parameter | Value | Grounding |
|---|---|---|
| `n_obs_steps` (To) | 2 | Table 7; also §B.1 — vision-based CNN DP gets **worse** with a longer observation horizon |
| `horizon` (Tp) | **48** | Table 7 says 16 **at 10 Hz**; see the control-rate section below |
| `n_action_steps` (Ta) | **24** | ditto — 800 ms of open loop, matching the paper's duration |
| `drop_n_last_frames` | 23 | `Tp − Ta − To + 1` |
| `down_dims` | (256, 512, 1024) → **68,665,222** UNet | Table 7 `#D-Params` = **67 M** for real-world rows. lerobot's default 259 M is their *simulation* size |
| `resize_shape` / `crop_ratio` | (240, 320) / 0.9 → **(216, 288)** | Table 7 `ImgRes 320×240`, `CropRes 288×216`. Exactly 0.9 |
| `crop_is_random` | true | §A.3 — random crop at train, static centre crop at inference. lerobot already does exactly this |
| scheduler | DDIM, 100 train / **16** infer | Table 7 `D-Iters 100/16`. (§3.4 says 10; the table governs for real-world) |
| `lr` / `weight_decay` | 1e-4 / 1e-6 | Table 7 |
| lr schedule | cosine, 500 warmup | §A.4, CNN-based DP specifically |
| `batch_size` | **64** | §A.4 — "64 for all image-based experiments" |

Inherited from the ACT A/B so the comparison holds: the same 143-episode order, all 23 recovery episodes in train, the same 30-episode clean holdout, `eval_split=0.209`, `seed=1000`.

## Why Tp/Ta are not the paper's numbers

The paper ran at **10 Hz**; we run at **30 fps**. An action index is dimensionless — it becomes time only when divided by fps.

| | paper @ 10 Hz | Tp=16/Ta=8 @ 30 fps | **Tp=48/Ta=24 @ 30 fps** |
|---|---|---|---|
| predicted | 1.60 s | 0.53 s | **1.60 s** |
| open loop | 800 ms | 267 ms | **800 ms** |

**Copying the step counts makes us 3× more conservative than they were, and it is undeployable.** Measured on the Mac cockpit 2026-08-11, one DP chunk costs **294.4 ms** (74.6 encoders + 219.8 UNet at DDIM-16). The `Ta=8` deadline is 267 ms — it would miss **every chunk**. At `Ta=24` there is 506 ms of headroom. Full numbers: [docs/deployment §4](../docs/deployment/README.md).

`n_action_steps` does **not** appear in `compute_loss` (verified in source); only `horizon` does. So **Ta stays tunable at rollout across 1…48 with no retraining** — train long, sweep Ta on the arm. Minimum viable Ta on the Mac is ≈ 10.

## Sizing, measured not estimated

Calibration jobs `9088440` / `9088457`:

| batch | workers | compute | loader | total | 100k steps | fits 8 h |
|---|---|---:|---:|---:|---:|---|
| 32 | 16 | 156.4 | 17.0 | 173.4 ms | 4.82 h | yes |
| 64 | 16 | 145.7 | 156.4 | 302.1 ms | 8.39 h | **no** |
| **64** | **32** | 139.2 | 106.4 | **245.7 ms** | **6.82 h** | yes, 1.18 h margin |

Two things the naive reading gets backwards:

1. **Compute per step is flat in batch size** — 139 ms at 64 vs 156 ms at 32, with peak memory 12% of an H200. The GPU is far from saturated, so the batch-64 penalty was *entirely* the dataloader. "Batch 64 doesn't fit, use 32" was wrong; "feed it faster" was right.
2. **The loader ceiling is not worker count.** I predicted 2× workers would halve `data_s`; it fell 32% (1,271 → 1,563 decoded frames/s). Sub-linear, so the bottleneck is more likely `/scratch` read bandwidth or memory contention. Do not expect more workers to keep buying time.

⚠️ **15% margin on a hard `MaxTime=08:00:00`.** A busier node can push this to a timeout near ~95k steps. `save_freq=20000` caps the loss at the last 20k steps and `--resume=true` exists.

## Pre-registered predictions

1. **Task 1 (ImageNet + BatchNorm) will beat task 0 (scratch + GroupNorm) on rollouts.** Reasoning: the paper's stated justification for GroupNorm is its interaction with an **EMA of the weights**, and **lerobot implements no EMA** (grepped: no `EMAModel`, no `use_ema`). At batch 64 with `n_obs_steps=2` each encoder sees 128 images, so BatchNorm's statistics are well estimated. Meanwhile 143 episodes is thin for training a ResNet-18 from scratch, and ImageNet features are a real prior. **If task 0 wins anyway, that is the more interesting result** and says the spatial-softmax + GroupNorm combination matters more than the pretrained prior.
2. **Both will train stably.** Calibration already showed loss falling 0.357 → 0.103 over 250 steps in the scratch arm.
3. **DP should beat ACT on the white cube**, which ACT failed 0/4 on. That object is the clearest multimodality case (45 mm, graspable from several approaches), and representing multiple valid grasps is the thing DP exists to do.

## What this cannot tell us

**DP's `eval_loss` is MSE on predicted noise. ACT's is L1 + KL on actions.** Different quantities, different scales, and unlike ACT there is no conversion to degrees. DP vs ACT is a **rollout question only** — `phi.utils.eval_rollouts`, same 15 episodes, same rubric.

Three confounds to record with any DP-vs-ACT number rather than discover afterwards:

| | DP | ACT |
|---|---|---|
| input resolution | 216×288 → **7×9** ResNet grid | 480×640 → **15×20** |
| open-loop commit | 800 ms | 1670 ms |
| encoder init | scratch (task 0) | ImageNet |

This is **method vs method as published**, not a controlled comparison of action decoders. The free control for the second row: re-run ACT with `--n-action-steps 24`, no retraining.

## Results

Both tasks **COMPLETED**, 100k steps, 98.4 epochs, zero errors. `eval_loss` = noise-prediction MSE on the 30 clean held-out episodes.

| step | 10k | 20k | 30k | 40k | 50k | 60k | 70k | 80k | 90k | 100k |
|---|---|---|---|---|---|---|---|---|---|---|
| task 0 `paper` (GroupNorm + scratch) | **0.0155** | 0.0173 | 0.0219 | 0.0248 | 0.0311 | 0.0369 | 0.0454 | 0.0532 | 0.0635 | 0.0636 |
| task 1 `lerobot` (BatchNorm + ImageNet) | **0.0173** | 0.0222 | 0.0271 | 0.0315 | 0.0389 | 0.0453 | 0.0563 | 0.0661 | 0.0805 | 0.0803 |

Final train loss **0.002** for both. Held-out loss rose **~4×** while train loss fell — a ~32× train/eval gap. **Monotonic overfitting that began before the first measurement.**

### Prediction 1: WRONG, and consistently

I predicted task 1 (ImageNet + BatchNorm) would win. **Task 0 beat it at every single eval point**, and the gap widens with training. So on this dataset the paper's spatial-softmax + GroupNorm encoder generalises better than an ImageNet prior — the outcome I labelled "the more interesting one."

Not a marginal call: 0.0155 vs 0.0173 at their respective bests, and 0.0636 vs 0.0803 at 100k.

### 🚨 The optimum is at 10k, and 10k was not checkpointed

`save_freq=20000`, so checkpoints exist at 20k/40k/60k/80k/100k. **The best held-out point is 10k for both runs — the earliest step measured, and one we did not save.** The best *available* checkpoints are therefore 20k: 0.0173 (task 0) and 0.0222 (task 1), both already past the optimum.

This was a judgement error, recorded so it is not repeated. Yash asked directly, at ~47 min in, whether we should have checkpointed at 10k. I said the risk was real but there was "zero evidence the optimum is below 20k" and advised against restarting. Two things I already knew should have outweighed that:

* ~100 epochs over **113 training episodes** is an enormous number of passes.
* The ACT augmented arm in this same project had **already** shown "last checkpoint is not best" (80k beat 100k).

I also said I would set a watch to report each `eval_loss` as it appeared, and did not. The `20k > 10k` signal existed roughly 1.5 h in; a watch would have caught it and allowed a restart with dense checkpoints instead of spending 5+ further hours confirming a monotonic curve.

⇒ **`save_freq=10000` or denser is the default for DP from now on.** Checkpoints are ~1.2 GB (102 M params plus optimizer state); ten of them is ~12 GB on `/scratch`. Disk was never the reason to save sparsely — 20000 was inherited from the ACT script without re-examination.

### Leading explanation for the severity: the missing EMA

Flagged before the run and now load-bearing. **lerobot implements no EMA** (no `EMAModel`, no `use_ema` anywhere in the package), while the paper's §3.2 recipe explicitly assumes one — its stated reason for GroupNorm is that normalisation choice mattering *"when the normalization layer is used in conjunction with"* weight averaging.

An EMA of the weights is a strong implicit regulariser. Its absence reframes this from "our dataset is too small" to **"we removed the paper's regulariser and then trained for 98 epochs."** That is a testable claim, not a conclusion — see next steps.

### What is still unknown

`eval_loss` here is noise-prediction MSE, and this write-up has argued from the start that it is a poor proxy for task success — the loss averages over uniformly sampled diffusion timesteps, and the easy high-noise regime dominates while the hard low-noise regime sets action precision. **So a 4× degradation on this metric is a strong negative signal, not proof the policies are unusable.** Only a rollout settles it.

### Next steps, in priority order

1. **Re-run 20k steps with `save_freq=2000`** (~82 min at the measured 4.07 steps/s) to capture the actual optimum at or below 10k.
2. **Roll out the existing 20k checkpoint regardless.** If it grasps acceptably, that is a genuinely useful result about how little held-out noise-MSE tells us.
3. **Regularisation arms worth testing** in the short re-run, since EMA is not available without writing it: image augmentation on (`dataset.image_transforms.enable=true`), and/or raising `optimizer_weight_decay` from the paper's 1e-6.

Rollout baseline to beat, from the ACT work — `cvae_3cam`, 2026-08-10, 11 held-out rollouts: **27% success / 0.418 mean progress**; per-object red cube 67%, cylinder 25%, **white cube 0%**.

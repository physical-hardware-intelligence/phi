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

⚠️ **CORRECTION, 2026-08-13 — `data_s` is not a stable property of the configuration.** The transformer calibration (job `9145018`) ran the *same* dataset, batch 64, 32 workers and pyav decode path, and measured `data_s` at **46.6 and 49.9 ms** against the 106.4 ms above — a **2.2× swing** with nothing in the configuration changed. So the "43% dataloader-bound" verdict was a property of that node at that moment, not of this setup, and the sizing table below should be read as carrying roughly ±2× uncertainty on its loader term. Point 2's reasoning (the ceiling is `/scratch` bandwidth or contention, not worker count) is *strengthened* by this — contention is exactly what varies run to run — but any future plan that leans on a specific `data_s` figure needs its own calibration on the node it will actually run on.

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

## Rollouts (2026-08-12) — the loss was wrong about everything

Task 0 (`paper`, 20k) rolled out on the arm, 15 episodes, canonical `phi_follower` calibration, same rubric and same held-out episodes as the ACT runs. Task 1 (`lerobot`) not rolled out.

| held-out | `cvae_3cam` (no recovery) | `recovery_noaug` (ACT, 100k) | **DP `paper` (20k)** |
|---|---:|---:|---:|
| n | 11 | 11 | 12 |
| success | 27% | **55%** | 50% |
| mean progress | 0.418 | 0.636 | **0.650** |
| red cube | 2/3 (0.73) | 1/3 (0.47) | 2/4 (0.75) |
| yellow cylinder | 1/4 (0.40) | 3/4 (0.80) | 3/4 (0.80) |
| white cube (45 mm) | 0/4 (0.20) | **2/4 (0.60)** | 1/4 (0.40) |

DP control split (trained-on episodes 5/50/95): 2/3, mean 0.733.

### 🔑 The headline: held-out noise-MSE did not predict task success

> 🚨 **CORRECTED 2026-08-13. The original text of this section is preserved below, struck through, because the conclusion it drew was not supported by the data.**
>
> ~~"The rollouts vindicate the hedge and go further: **the signal was worthless.** A checkpoint we *knew* was past its optimum, from a run whose held-out loss rose monotonically 4×, matched a 100k-step ACT model... ⇒ Do not use DP noise-MSE for model selection or early stopping on this task."~~
>
> **Two errors.**
>
> **1. It conflated the run with the checkpoint.** The run's loss did rise 4× (`0.0155 @10k → 0.0636 @100k`). But the checkpoint actually rolled out was **20k at `eval_loss` 0.0173 — 1.12× the optimum, 12% past it.** The 4×-degraded 100k checkpoint was never put on the arm. "A checkpoint we knew was past its optimum" is true; "past its optimum by 4×" is not.
>
> **2. One point cannot refute a correlation.** Exactly **one** DP checkpoint was ever rolled out, at **n=12 episodes (~29% power)**. To test whether `eval_loss` ranks checkpoints you need *several* DP checkpoints spanning a range of `eval_loss`, rolled out under the same conditions. That experiment has not been run. Comparing DP to ACT does not test it either — their `eval_loss` values are different quantities (noise-MSE vs L1+KL) and are not comparable.
>
> **The supportable statement:** DP `eval_loss` is **unvalidated** as a model-selection metric on this task. Not worthless, not trustworthy — untested. Next-step #1 is therefore *not* demoted: chasing the 10k optimum is still the reasonable default, because we have no evidence against it.

This write-up argued before any rollout that DP's `eval_loss` was a poor proxy, and that "a 4× degradation is a strong negative signal, **not proof the policies are unusable**." That hedge was correct and is what survives.

What the rollout does establish, plainly: a DP checkpoint 12% past its own optimum **matched a 100k-step ACT model and posted the highest mean progress of the three** — at n=12, where only a very large effect would have been detectable.

### DP vs ACT: a dead heat

Paired over the 11 held-out episodes both models ran:

```
  ep       0     1    21    45    46    65    66    90    91   110   111
  ACT    0.2   1.0   0.2   1.0   1.0   1.0   0.2   1.0   0.2   0.2   1.0
  DP     0.2   1.0   0.8   1.0   1.0   0.2   1.0   1.0   0.2   0.2   0.2
  delta +0.0  +0.0  +0.6  +0.0  +0.0  -0.8  +0.8  +0.0  +0.0  +0.0  -0.8

  mean delta -0.018   sd 0.477   se 0.144   t(10) = -0.13
  DP better on 2, worse on 2, tied on 7   sign test p = 1.000
```

Against the baseline, DP is **+0.218** (t = 1.15, p = 0.453) — the same direction and nearly the same magnitude as ACT's **+0.240** (t = 1.41, p = 0.375). Two independently trained architectures agreeing on the sign is worth more than either test; **neither is resolved at n ≈ 11.** Detecting an effect this size at 80% power needs **n ≈ 40**.

⚠️ **Pairing bought almost no power** (se 0.171 paired vs 0.179 unpaired on the ACT comparison). Half the episodes tie at the 0.2 floor, so the scores behave as near-binary and variance stays high. **The cheap fix is repeat trials on already-staged scenes, not more scenes** — staging dominates the cost, `--episodes` accepts duplicates, and repeats attack exactly this variance.

### ✅ The Ta=24 sizing decision was correct

Median wall time per rollout: **DP 67 s** vs ACT 69 s, and a tighter spread (56-95 s vs 60-153 s). No stutter, no slow-FPS warnings. The 294 ms measured chunk cost fit inside the 800 ms budget in practice, exactly as the control-rate section predicted. Had we copied Table 7's `Ta=8`, this would have missed every chunk.

### Prediction 3: WRONG on the fair comparison

Predicted: *"DP should beat ACT on the white cube, which ACT failed 0/4 on."* DP got **1/4**; recovery-trained ACT got **2/4**. DP beat the *baseline* (25% vs 0%) but the baseline is not the right comparison — the recovery-trained ACT is, and DP lost to it.

⇒ **Two of three pre-registered predictions were wrong** (1 and 3). The multimodality argument for DP on the ambiguous-grasp object did not survive contact with the arm.

### DP's failure mode is contact precision, and it is not ACT's

ACT failed mostly by hovering and reach errors. DP's failures are positioned correctly and fail at closure. Operator notes:

> ep 0 — *"at the right location but just was not able to lock in"*
> ep 65 — *"one of the fingers was just tapping on the object"*
> ep 95 — *"stayed a considerable time around the top of the container, which has not been seen before"*

Two candidate causes, distinguishable by experiment:

1. **Input resolution.** 216×288 → a 7×9 ResNet grid, vs ACT's 480×640 → 15×20. Grasping a 25 mm cube is a millimetre-scale problem. This was confound #1 of the pre-registration and now has a matching symptom. **Test: re-train DP at ACT's resolution.**
2. **Overfitting concentrated in the low-noise regime**, which sets fine action detail and which the uniformly-sampled loss under-weights. **Test: the 10k checkpoint** — which does give the dense-checkpoint re-run a purpose, just a different one than "recover the optimum."

**Ruled out: open-loop horizon.** DP re-plans *more* often than ACT (800 ms vs 1670 ms), which predicts the opposite sign.

### One genuine recovery behaviour, not yet general

> ep 90 (success, approach `m`) — *"Initially, it landed in front of the object, but... it was able to lean back and then grab it again and was able to do it successfully."*

**First documented retry-after-failed-grasp in this project** — the 23 recovery episodes doing what they were collected for. It does not generalise: ep 110 *"not able to return back and then grasp"*, ep 65 *"not able to learn from failures."* One of three opportunities.

### Baseline-comparison confound, stated plainly

The baseline's 11 rollouts predate the base-clamp discovery ([setup §3b-bis](../docs/robots/so-arm101/02-setup.md)), so the rig may not have been byte-identical. Working *against* that worry: the 2026-08-12 control rollouts (ACT 33% n=3, DP 67% n=3) were not systematically better than the baseline's controls (50%, n=6), so the rig does not look easier today.

### Next steps, revised after the rollouts

1. **Add power before adding models.** Repeat trials on the 12 staged held-out scenes (3× = 36 rollouts, 12 stagings) for ACT and DP. This is the only cheap route to resolving a ±0.22 effect, and everything below is uninterpretable without it.
2. **Test the resolution hypothesis** — re-train DP with `resize_shape` at ACT's 480×640 (or an intermediate step). This is the leading explanation for the contact-precision failure and it is a clean controlled change.
3. **Re-run 20k with `save_freq=2000`** — now motivated by hypothesis 2 (does the low-noise regime degrade?) rather than by "recover the best loss," since the loss is discredited as a task proxy.
4. **Regularisation arms**, since EMA is unavailable without writing it: `dataset.image_transforms.enable=true`, and/or `optimizer_weight_decay` above the paper's 1e-6. Lower priority than 1 and 2.
5. **Free control worth taking**: re-run ACT with `--n-action-steps 24` to match DP's commit horizon, no retraining. Removes confound #2 from the table above for the cost of 12 rollouts.

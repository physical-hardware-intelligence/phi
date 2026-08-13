# Diffusion Policy, TRANSFORMER backbone: does regularisation explain the overfit?

**Date**: 2026-08-13 · **W&B**: project `phi_dp` (same as the CNN runs, so curves overlay)
**Script**: [`configs/hpc/train_diffusion_transformer.sbatch`](../configs/hpc/train_diffusion_transformer.sbatch)
**Calibration**: [`configs/hpc/calib_diffusion_transformer.sbatch`](../configs/hpc/calib_diffusion_transformer.sbatch)
**Port**: [`src/phi/policies/dp_transformer.py`](../src/phi/policies/dp_transformer.py)
**Dataset**: [`phi_so101_cubes_cylinder_recovery_v1`](../datasets/phi_so101_cubes_cylinder_recovery_v1.md) — 143 eps / 81,943 frames
**Paper**: Chi et al., *Diffusion Policy* ([2303.04137](https://arxiv.org/abs/2303.04137)), §3.1 and **Table 8**

_Written before the job was submitted. Predictions below are pre-registered so they cannot be retrofitted._

## The question

The [CNN A/B](2026-08-12_dp-recovery-encoder-ab.md) ended with train loss **0.002** and held-out loss **rising 0.0155 → 0.0636**, a ~32× gap, degrading monotonically from the first measurement. Two explanations fit that evidence:

| | claim | if true, the fix is | cost |
|---|---|---|---|
| **A** | the paper's regulariser was missing | a config change | hours |
| **B** | 113 training episodes is too few | more teleop | days |

Hypothesis A has a concrete basis: the paper's §3.2 recipe assumes an **EMA of the weights** — its stated reason for GroupNorm is the interaction with weight averaging — and **lerobot implements no EMA** (grepped: no `EMAModel`, no `use_ema` anywhere in the package). That run also used `weight_decay=1e-6`, effectively none.

The two remedies differ by ~two orders of magnitude in cost. That is what makes this worth a GPU pair rather than a guess.

## What is being tested

| task | `weight_decay` | `p_drop_attn` | |
|---|---|---|---|
| **0** `paper` | **1e-3** | **0.3** | Table 8's transformer recipe |
| **1** `unreg` | **1e-6** | **0.0** | the CNN's regularisation, i.e. none |

**This deliberately confounds two variables.** The actionable question is binary — *does regularisation fix this?* — and with two arms the priority is detection power. If the whole package changes nothing, neither knob alone will, and that is real support for B. If it does, a follow-up isolates which.

## Parameters, and the source of each

| Parameter | Value | Grounding |
|---|---|---|
| `n_layer` | 8 | Table 8, **every** row |
| `n_emb` | 256 | Table 8, all simulation rows → measured **9,020,934** backbone params vs its `#D-params 9` |
| `n_head` | 4 | reference wrapper default; Table 8 does not list it |
| `n_cond_layers` | 0 | reference wrapper default → the cond encoder is an MLP, not a transformer |
| `causal_attn` | true | reference wrapper default |
| `p_drop_emb` | 0.0 | reference wrapper default (verified in source) |
| `weight_decay` | **1e-3** / 1e-6 | Table 8 vs Table 7. Its caption reads *"WDecay: weight decay (for transformer only)"* |
| `betas` | (0.9, 0.95) | reference `configure_optimizers()`; Table 8 does not list them |
| `lr` | 1e-4 | Table 8 |
| scheduler | DDIM, 100 train / 16 eval | Table 8 |
| `To`/`Ta`/`Tp` | 2 / 24 / 48 | inherited unchanged from the CNN runs |
| `batch_size` | 64 | §A.4, "64 for all image-based experiments" |
| vision | ResNet-18 scratch + spatial softmax + GroupNorm | §3.2 — and the arm that **won** the CNN A/B |
| `steps` | 30,000 | see below |
| `save_freq` / eval | **2,000** / 2,000 | see below |

Inherited so the comparison holds: the same 143-episode order, all 23 recovery episodes in train, the same 30-episode clean holdout, `eval_split=0.209`, `seed=1000`.

### Two deviations, flagged rather than buried

**`Tp=48`, not Table 8's number.** Table 8's sim rows use `Tp=10` where Table 7's CNN used 16 — the paper runs *shorter* horizons on the transformer. Keeping 48 preserves comparability with the two existing CNN checkpoints, and `Ta` is rollout-tunable anyway.

**30k steps, not 100k.** The CNN's held-out optimum was at 10k and everything after was monotonic decay; 70k of those steps bought only confirmation.

**`save_freq=2000`, not 20000.** The CNN run's true optimum was at 10k and *was never saved* — `save_freq` had been inherited from the ACT script without re-examination. Disk was never the reason: a DP-T checkpoint is 42,612,198 × 4 B = 170 MB plus Adam's two moments ≈ **511 MB**, so 15 per arm is ~7.7 GB and ~15.3 GB for both.

**Eval every 2000, not 10000.** The CNN evaluated every 10k, which is precisely why "the optimum is at or below 10k" could not be resolved.

## The port

lerobot ships **only** `DiffusionConditionalUnet1d` — grep `lerobot/policies/diffusion/` for "transformer" and you get zero hits. The backbone is vendored from `real-stanford/diffusion_policy` (MIT, verified via the GitHub API) and adapted without forking lerobot: `_prepare_global_conditioning` ends in `.flatten(start_dim=1)`, and a row-major flatten is losslessly reversible, so the adapter only reshapes `(B, To*feat)` back to `(B, To, feat)`.

Verified before submission: **9,020,934** backbone params against Table 8's "9", and the 768-wide variant measures **80,540,166** against its "80" — two independent checks on a config not even being used here.

## Pre-registered predictions

1. **Task 0 (paper) will show a materially flatter held-out curve than task 1**, and task 1 will reproduce the CNN's monotonic degradation.
2. **If both arms degrade alike, hypothesis B gains real support** and the next move is data collection, not hyperparameters.
3. **DP-T at 9.0M will not obviously underfit** relative to the 68.7M UNet. If it does, `n_emb=768` (80.5M, Table 8's Kitchen and Real Push-T rows) is the fallback, and the calibration already measures whether it fits the wall clock.

⚠️ **Calibration on my own record**: of the three predictions pre-registered for the CNN A/B, **two were wrong** — the ImageNet arm did not win, and DP did not beat ACT on the white cube. Weight these accordingly.

## What this cannot tell us

**It is evidence about the CNN's overfit, not proof about it.** DP-T differs from DP-CNN in more than regularisation: 9,020,934 backbone params against 68,665,222, and a different architecture. Strictly, this tests whether regularisation controls overfitting *in DP-T*.

**`eval_loss` is a training-health signal here, not a model-selection criterion.** The 2026-08-12 rollouts settled that: a DP checkpoint whose held-out noise-MSE had risen 4× still scored 50% / 0.650 on the arm, statistically tied with a 100k-step ACT model. Held-out noise-MSE did not predict task success.

**Bucket the loss by diffusion timestep `k` before drawing conclusions.** A single averaged `eval_loss` blends the near-free `k>90` regime — where `√ᾱ = 0.0005`, so the input essentially *is* the noise and predicting it is nearly copying — with the `k<5` regime, where recovering `eps` means dividing by `0.0251` and any residual uncertainty about the true action is amplified ~40×. Only the second regime sets whether the gripper closes, and it is exactly the failure mode the DP rollouts showed.

## Results

_Not yet run._

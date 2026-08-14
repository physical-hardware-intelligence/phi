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

## Calibration (job `9145018`, 2026-08-13) — width is free

250 steps, batch 64, 32 workers, H200.

| | backbone params | compute | loader | total | steps/s | peak GPU |
|---|---:|---:|---:|---:|---:|---:|
| `n_emb=256` | 9,020,934 | 172.8 ms | 49.9 ms | **222.7 ms** | 4.49 | 16,332 MiB (11%) |
| `n_emb=768` | 80,540,166 | 177.2 ms | 46.6 ms | **223.8 ms** | 4.47 | 18,596 MiB (13%) |

30,000 steps → **1.86 h**; 100,000 → 6.2 h. Both fit the 8 h wall with room.

### 🔑 An 8.9× parameter difference costs 0.5% of wall time

Adding 71.5M parameters to the denoiser cost **4.4 ms/step**. That is direct evidence the GPU is **not FLOP-limited on the backbone**: if it were, 8.9× the parameters would cost meaningfully more.

Leading explanation, marked as such: 48 tokens × 256 dims is tiny work per kernel, so 8 layers × 3 sublayers of small launches leave an H200 mostly idle and wall time is set by launch overhead rather than arithmetic. **[Hypothesis — kernel launches not profiled.]**

Two consequences:

* **`n_emb=768` is free.** If 256 underfits, the 80.5M fallback costs nothing in wall time. That removes width from the list of things to economise on.
* **A Mac micro-benchmark of the backbone alone is misleading.** Timing `TransformerForDiffusion` at batch 1 on CPU gave "2.8× faster than the UNet per denoising step". On a real training step the 33.6M vision encoder chewing 384 images at 216×288 dominates, and that measurement does not transfer.

### The transformer is *slower* in compute than the UNet, not faster

172.8 ms against the CNN's 139.2 ms, despite a backbone 7.6× smaller. Consistent with the same launch-bound picture: convolutions over a 48-step axis map onto far fewer, larger kernels than 24 attention/FFN sublayers do.

### ⚠️ `data_s` is not reproducible across nodes

46.6 / 49.9 ms here against 106.4 ms in the CNN calibration — same dataset, batch, workers and decode path. A 2.2× swing from node assignment or `/scratch` contention. The CNN write-up's "43% dataloader-bound" has been corrected accordingly. **Do not carry a `data_s` figure between runs; re-calibrate.**

## Results (job `9145402`, 2026-08-13)

Both arms **COMPLETED**, 30,000 steps, 29.5 epochs, zero errors. 16 checkpoints each (2k…30k plus `last`), 7.2 GB per arm. Wall time 2:36 (`paper`) and 2:17 (`unreg`) — the gap is dropout 0.3 costing work the other arm skips.

`eval_loss` = noise-prediction MSE on the 30 clean held-out episodes.

| step | 2k | 4k | 6k | **8k** | 10k | 12k | 14k | 16k | 18k | 20k | 22k | 24k | **26k** | 28k | 30k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **`paper`** wd 1e-3, drop 0.3 | .0741 | .0689 | .0630 | .0610 | .0617 | .0582 | .0566 | .0582 | .0563 | .0546 | .0541 | .0541 | **.0531** | .0532 | .0536 |
| **`unreg`** wd 1e-6, drop 0.0 | .0419 | .0325 | .0307 | **.0287** | .0293 | .0288 | .0288 | .0295 | .0310 | .0310 | .0308 | .0328 | .0340 | .0339 | .0361 |

Final train loss: `paper` **0.061**, `unreg` **0.013** — a 4.7× gap, which is the regularisation doing its job.

### Prediction 1: CORRECT

I predicted `paper` would show a materially flatter held-out curve and `unreg` would reproduce the CNN's monotonic degradation. Both happened. **`unreg` bottoms at 8,000 and climbs monotonically thereafter**; the CNN turned at ≤10,000. `paper` never turns — it descends to 0.0531 at 26k and then plateaus (.0531 / .0532 / .0536).

_Note: `paper` has **converged**, not "still improving". Its last three points are flat. More steps would not help it._

### 🚨 But `paper` is uniformly worse in absolute terms

| | best `eval_loss` | at step | rise from minimum to 30k |
|---|---:|---:|---:|
| `unreg` | **0.0287** | 8,000 | **1.26×** |
| `paper` | 0.0531 | 26,000 | 1.01× (flat) |

**`unreg`'s best beats `paper`'s best by 1.85×.** Regularisation bought curve stability at a large cost in level. Two readings this experiment cannot separate:

1. Table 8's strength (wd 1e-3 **and** dropout 0.3) is too much for 113 episodes, and `paper` is underfitting — it prevents overfitting by preventing fitting.
2. `paper` generalises more stably and noise-MSE is the wrong yardstick for which model is the better *policy*.

Reading 2 is not special pleading: the 2026-08-12 rollouts showed a DP checkpoint whose held-out noise-MSE had risen 4× still scoring 50% / 0.650 on the arm, statistically tied with a 100k-step ACT model.

Because the arms deliberately confounded weight decay and dropout, **we do not know which knob is too strong.**

### What this says about A vs B

| run | backbone | regularisation | rise from minimum |
|---|---:|---|---:|
| DP-CNN | 68,665,222 | none | **4.10×** (0.0155 → 0.0636) |
| DP-T `unreg` | 9,020,934 | none | **1.26×** (0.0287 → 0.0361) |
| DP-T `paper` | 9,020,934 | wd 1e-3 + drop 0.3 | **1.01×** (flat) |

The overfit is strongly controllable by model size and by regularisation. **Hypothesis B — "113 episodes is simply too few" — loses ground as the sole explanation**, because holding the dataset fixed and changing only the model/regularisation moved the degradation from 4.10× to nothing.

But hypothesis A does not win cleanly either. The honest reading is a **middle**: regularisation controls the overfit, and Table 8's strength overshoots on this dataset. The unexplored region is between `wd 1e-6 / drop 0.0` and `wd 1e-3 / drop 0.3`.

### The `save_freq` lesson paid off

`unreg`'s optimum is at **step 8,000** — inside the window the CNN run could not resolve, and this time **it was saved**. Under the old `save_freq=20000` it would have been lost exactly as before.

### What is still unknown

**Which checkpoint is the better policy.** Nothing here measures task success, and the metric in this table has already been shown not to predict it. The candidates to roll out are `unreg @ 8000` (best noise-MSE) and `paper @ 26000` (best stable), against the existing DP-CNN and ACT numbers.

**Which regime moved.** These are uniform-`k` averages, blending the near-free `k>90` regime (√ᾱ = 0.0005, so predicting `eps` is nearly copying the input) with `k<5`, where recovering `eps` means dividing by 0.0251 and any residual uncertainty about the action is amplified ~40×. Only the second sets whether the gripper closes. A `k`-bucketed evaluation over these 32 checkpoints is the cheapest way to find out, and it is inference-only.

### Next steps, in priority order

1. **`k`-bucketed eval** across both arms' checkpoints. No training, no arm time, and it decides whether `paper`'s higher average hides a better low-`k` regime — which is the regime that grasps.
2. **Roll out `unreg @ 8000` and `paper @ 26000`** on the same 15 episodes as the CNN and ACT runs. This is the only thing that settles reading 1 vs reading 2.
3. **An intermediate-regularisation arm** (e.g. wd 1e-4, dropout 0.1), if 1 and 2 suggest `paper` is underfitting rather than simply differently-fit.
4. **`n_emb=768` is free** (calibration: 223.8 ms vs 222.7 ms/step) if capacity turns out to be the limit rather than regularisation.

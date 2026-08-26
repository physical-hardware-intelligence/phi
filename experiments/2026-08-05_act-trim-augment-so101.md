# ACT on 8-bin v1 — opening-pause trim × photometric augmentation (12 runs)

**Status**: All 12 trained, zero robot numbers · **Owner**: Parv · **Arrays**: `8924118` `8926777` `8938455` `8948027` on Explorer
**Dataset**: [`BrutalCaesar/phi_so101_8bin_v1`](../datasets/phi_so101_8bin_v1.md) · **Trim table**: [`Parv-09/phi_so101_8bin_v1_trim`](https://huggingface.co/datasets/Parv-09/phi_so101_8bin_v1_trim)
**Sibling grid**: [act-8bin-6run](2026-08-04_act-8bin-6run.md) (Yash, wrist+top and wrist+front)

## Goal

Two axes the 6-run grid deliberately left out:

1. **Does removing each episode's pre-teleop dead air help?** Predicted in the 6-run eval protocol
   ("expect a hesitant start at block-A positions… the fix is trimming dead time at collection").
   This tests it without re-collecting.
2. **Does photometric augmentation help at 119 episodes?**

## Camera set — the only arm with no physical wrist camera

| Policy input key | Physical camera |
|---|---|
| `observation.images.wrist` | **overhead / top** (EMEET C960) |
| `observation.images.front` | front (Brio 101) |
| `observation.images.top` — **DROPPED** | physical wrist (gripper module) |

🚨 **The dataset's `wrist` and `top` keys hold each other's footage.** Every task in the 6-run grid
keeps `top`, i.e. keeps the physical wrist camera. **This grid is the only one without it**, so it is
the top+front arm of the three-owner camera comparison, not a repeat.

Verified four independent ways: frame decode at t=6/120/240 s, the dataset card, per-episode decode
through `LeRobotDataset` on eps 0 and 80, and the runtime assertion in the trainer. Selection uses
`cfg.input_features` preset before `make_policy` (survives `if not cfg.input_features`), so no
derived dataset and no duplicated video.

## ⚠️ Split diverges from the 6-run protocol — losses are NOT comparable

The 6-run grid trains on all **89** non-held-out episodes and has no validation split. This grid
carves a 12-episode in-distribution val set out of that pool (last 2 of each of the 6 training bins),
leaving **77 train / 12 val / 30 held out**.

Consequence: **do not compare final loss between this file and the 6-run file.** Different training
set size, different measurement. The held-out 30 (left bin 3 = eps 29–43, right bin 2 = eps 74–88)
are identical and untouched in both.

The val split buys checkpoint diagnostics without spending robot time; it costs 12 episodes of
training data. Whether that trade was right is an open question — see [Next](#next).

## Measured: how much dead air, and where

Audit script: [`src/phi/utils/deadtime.py`](../src/phi/utils/deadtime.py) · per-episode table on the Hub.

`DROP` = first frame where any joint departs the start pose by >1.0°. Cumulative, so it is immune to
per-frame jitter, and taking the *first* crossing means it can never reach past the start of real
motion into an interior pause.

| | mean | median | min | max | % of episode |
|---|---|---|---|---|---|
| left (eps 0–58) | **105.8 fr = 3.53 s** | 105 | 72 | 158 | 10.8–22.9% |
| right (eps 59–118) | **41.4 fr = 1.38 s** | 38.5 | 3 | 103 | 0.5–20.8% |

**8,722 of 72,518 frames = 12.0%.**

**Evidence the cut lands on real dead air:** a 20× change in threshold moves the total by 1.5 points.

| threshold | 0.5° | 1.0° | 2.0° | 5.0° | 10.0° |
|---|---|---|---|---|---|
| dropped | 11.6% | **12.0%** | 12.2% | 12.7% | 13.1% |

⚠️ **A per-frame movement threshold was tried and rejected.** Servo jitter plus the operator's hand
resting on the leader exceeds 0.1°/frame from frame 1 in **57 of 119 episodes**, so a per-frame test
reports a 1-frame pause where the arm has not moved for 100+ frames.

**Deliberately not trimmed.** Interior pauses exist in only **4 of 119** episodes (median longest
0.1 s) — and deleting interior frames would be actively harmful, since a chunk is N *contiguous*
frames and removing any teaches trajectories with teleport jumps no arm can execute. Trailing pauses
(117 of 119, left mean 1.9 s) sit at the end pose with the object already in the box, so they do not
recreate the opening absorbing state.

**Why the opening pause is a problem and not just wasted frames.** ACT has `n_obs_steps=1` and no
clock. While the arm sits still in a static scene, every frame is a near-identical observation, but
the recorded action differs by how long the operator happened to wait. At rollout the policy predicts
"hold still", nothing moves, so the next observation is identical, so it predicts "hold still" again
— an absorbing state. With chunk 50 the risk is concentrated on the **left**, where the shortest
pause (72 frames) still exceeds the 50-frame chunk.

**Nothing on disk or on the Hub was modified.** The trim is a `SubsetRandomSampler` index list built
from the per-episode table.

## The grid

| Array | Steps | Trim | Augment | Chunks | Wall/task |
|---|---|---|---|---|---|
| `8924118` | 60,000 | no | no | 50/75/100 | ~105 min |
| `8926777` | 180,000 | no | no | 50/75/100 | ~5.2 h |
| `8938455` | 80,000 | **yes** | no | 50/75/100 | ~2.05 h |
| `8948027` | 80,000 | **yes** | **yes** | 50/75/100 | ~4.2 h |

Constant: batch 8, lr 1e-5, AdamW wd 1e-4, grad clip 10, `kl_weight` 10, `use_vae` true, seed 1000,
`val_every` 2000, H200. Trim changes steps/epoch 5,933 → 5,208.

**Augmentation** = LeRobot photometric defaults, **train loader only**, val kept clean: brightness
and contrast (0.8–1.2), saturation (0.5–1.5), hue (±0.05), sharpness (0.5–1.5), 3 of 5 per frame.
`affine` excluded on purpose — it is the only geometric transform and shifts apparent object position
while the action label stays fixed, blunting exactly the spatial precision an 8-bin task needs.
Magnitudes are library defaults, not tuned: val separates best from final by 1–3% here, so it cannot
rank five hyperparameters without fitting noise.

## Results

### Training loss is not a ranking, again

| chunk | 60k | 180k | trim 80k | trim+aug 80k |
|---|---|---|---|---|
| 50 | 0.1673 @40k | 0.1653 @88k | 0.1782 @66k | **0.1743 @80k** |
| 75 | 0.2072 @58k | 0.2072 @58k | 0.2067 @54k | 0.2057 @36k |
| 100 | 0.2389 @48k | 0.2389 @48k | 0.2385 @30k | 0.2290 @40k |

Val L1, each run's own best checkpoint. **Untrimmed and trimmed columns are not comparable** — the
trim removes the easiest ~12% of val frames, so trimmed val is a harder exam. And as the 6-run file
notes, val cannot rank chunk sizes at all: longer horizons are graded over more, harder timesteps.

**180,000 steps bought nothing.** Chunks 75 and 100 reproduced the *identical* best val at the
*identical* step as the 60,000-step run. That is why later arrays use 80,000.

### The honest generalization gap

`gap_eval.py` scores **both** splits in eval mode (dropout off, `z=0`, no augmentation, same trim),
paired on the identical 6,375 frames. ⚠️ Not yet in this repo — it and the trainer land in a
follow-up PR, because the eval script imports the split and trim helpers from the trainer and that
needs packaging work. `deadtime.py` is here and is standalone.

⚠️ **Training-time train loss cannot be compared to val loss.** It is measured with dropout on and
with `z` supplied by the VAE encoder, which sees the ground-truth action chunk it is helping predict.
Val is measured with dropout off and `z=0`. Subtracting them understates the gap. Sanity check: this
script reproduces the recorded val L1 to four decimals (0.1782 / 0.2064 / 0.2383).

Matched steps, trim vs trim+aug:

| chunk | step | trim ratio | +aug ratio | trim val | +aug val |
|---|---|---|---|---|---|
| 50 | 40k | 2.38 | **2.30** | 0.1837 | **0.1793** |
| 50 | 80k | 3.16 | **3.04** | 0.1800 | **0.1741** |
| 75 | 40k | 2.64 | **2.60** | 0.2070 | 0.2072 |
| 75 | 80k | 3.72 | **3.55** | 0.2081 | 0.2102 |
| 100 | 40k | 3.04 | **2.85** | 0.2441 | **0.2282** |
| 100 | 80k | 3.78 | **3.60** | 0.2463 | **0.2431** |

**Chunks 75 and 100 are over-trained at 80k** — both get *worse* on val from 40k to 80k while train
improves ~30%. Only chunk 50 still improves at the budget limit. Use step 40,000 for 75 and 100.

### Robot

**First rollout, 2026-08-05, `best_c50` (trim+aug, step 80,000), `n_action_steps=50`:**

- ✅ **The arm moves immediately. No freeze at start.** This is the trim's stated purpose and the
  first evidence it works, though the untrimmed control has not been run yet.
- ❌ **Cannot localise the duck.** Moves confidently, reaches similar places, does not grasp.
- chunk 75 performed **worse** than chunk 50, consistent with its higher val.

_Per-bin success rates pending. Protocol and progress score: [6-run §Eval protocol](2026-08-04_act-8bin-6run.md#eval-protocol)._

| Model | chunk | ckpt | Bin L3 | Bin R2 | Froze at start |
|---|---|---|---|---|---|
| trim+aug | 50 | 80,000 | | | no (n=1) |
| trim+aug | 75 | 36,000 | | | |
| trim+aug | 100 | 40,000 | | | |
| **untrimmed control** | 50 | 88,000 | | | ← the freeze comparison |

## What we learned

**The trim is real and it is cheap.** All three trimmed runs beat their "neutral baseline" (untrimmed
best ÷ 0.873, what val would read if the trim changed only the exam and not the model). The first
rollout showed no start freeze.

**Augmentation helps, weakly and consistently, and is too weak as configured.** Gap narrowed 6/6,
ratio improved 6/6, val improved 4/6 — nothing contradicts the direction. But at matched steps
augmented train L1 rose only **0.5–6%**. A regularizer that is actually constraining the model should
make training visibly harder. Cost: **49% throughput** (10.8 → 5.5 it/s), CPU-bound on ColorJitter.
Larger magnitudes, or admitting `affine`, is the motivated next test.

**Posterior collapse, independently reproduced.** Raw KL falls from 0.0969 at epoch 1 (**81% of total
loss**) to 3e-5 by epoch 31 — the whole 32-dim latent carries 0.35 bits at peak. Consistent with the
6-run probe's `kld_loss = 6.7e-5`. Two arms, same conclusion. Past epoch 4 the total loss simply *is*
L1.

**Val loss carries no KL term.** `modeling_act.py:407` gates the VAE encoder on `self.training`, so
in eval mode `latent_sample` is zeros and `loss_dict` has no `kld_loss` key. A `KL nan` on a val log
line is correct reporting. Val was pure action error all along.

### Corrections to earlier claims in this project

Recorded because each was stated confidently and was wrong:

1. **"60k steps is 2.5× more than the data supports."** Wrong basis. ACT's README says to train "at
   least 5000 epochs or 3–4× the length after the loss has plateaued" because "success rate and
   smoothness can improve way after loss plateaus". A flat val loss is the wrong stopping signal here.
2. **"Val loss includes a KL component."** It does not, see above.
3. **Val L1 converted to "degrees of mean joint error."** Removed. The arithmetic is a valid unit
   conversion, but per-joint stds differ 5× (shoulder_lift 52.88° vs wrist_roll 11.29°) so one
   averaged figure hides which joint is wrong, and turning joint error into gripper *position* error
   needs forward kinematics that was never computed.
4. **"Watch the val/train ratio to judge augmentation."** Confounded. Best-val checkpoints land at
   different steps, so train loss moves for reasons unrelated to augmentation. Fixed by the
   matched-step table above.

### 🚨 Operational trap: scp truncates silently on this link

**Three of five 200 MB checkpoint transfers from Explorer failed mid-file.** scp exits **0**, prints
`Connection closed` at most, and leaves a truncated file. One attempt left a 163,722,240-byte stub of
a 206,494,928-byte model. A file listing is not enough — `scp -r` copies alphabetically, so a drop
after `model.safetensors` leaves `BEST.txt`, `config.json`, `model.safetensors` and **none** of the
four `policy_*` files that hold the normalization statistics. `config.json` records only *that*
MEAN_STD is used, not the numbers.

**Always check the byte count.** A complete ACT checkpoint is 7 files.

```bash
ls -1 <ckpt> | wc -l && stat -f%z <ckpt>/model.safetensors    # expect 7 and 206494928
```

## Next

**The task may be spread too thin, and that is testable for free.** ACT reached 80–90% with "only 10
minutes worth of demonstrations"; 119 episodes at ~20 s is ~40 minutes, so raw quantity is not the
constraint. Per *spatial variant* is:

| | episodes | variants | per variant | result |
|---|---|---|---|---|
| ACT paper | 10 min | ~1 fixed position | all of it | 80–90% |
| Sherry Chen | 150 | 6 bins | **25** | 75% OOD |
| **this dataset** | 119 | **8 bins** | **15** | pending |

Two ways to reach ~25/variant: record ~80 more episodes, or **retrain on a 5-bin subset with the data
already collected** (119/5 = 24). The second costs one training run and no recording, and it
distinguishes "pipeline is broken" from "variation is too wide for the data".

**Eval-budget priority**

1. Camera realign check ([`camera_realign.py`](../src/phi/utils/camera_realign.py)) — untested, and a
   drifted mount invalidates every model in this file and the 6-run file alike.
2. Untrimmed control at left-bin positions — the falsifiable freeze test.
3. `n_action_steps` 50 / 25 / 12 on one checkpoint. Inference-only, no retraining, free.
4. 5-bin subset retrain.
5. Seeds. **Everything here is n=1**, so a 2–3% val difference is not established and no chunk-size
   or augmentation claim in this file should go in a paper or the Research Day poster as-is.

## Commands

```bash
# Regenerate the per-episode trim table and the threshold-sensitivity check (in this repo):
python -m phi.utils.deadtime
```

```bash
# Training and paired eval — scripts land in a follow-up PR, recorded here for the record:
sbatch train_act_8bin.sbatch                  # array 0-2, chunks 50/75/100, --trim --augment
python gap_eval.py --ckpt <a> --ckpt <b>      # paired, equal-conditions train-vs-val
```

## Deliverables (Φ rules)

- [ ] Model card per run in [`models/`](../models/)
- [ ] Per-bin success rates, never averaged
- [ ] Untrimmed-vs-trimmed freeze comparison at left-bin positions
- [ ] Leaderboard entry
- [ ] This file updated with actual robot numbers

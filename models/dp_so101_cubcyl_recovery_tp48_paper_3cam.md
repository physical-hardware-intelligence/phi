# dp_so101_cubcyl_recovery_tp48_paper_3cam

**HF id**: [`BrutalCaesar/dp_so101_cubcyl_recovery_tp48_paper_3cam`](https://huggingface.co/BrutalCaesar/dp_so101_cubcyl_recovery_tp48_paper_3cam) · public
**Revision**: `9e0aa39` (uploaded 2026-08-12) · verified: `DiffusionPolicy.from_pretrained` loads **102,256,486 params**, 3 camera features, 9 files
**Policy**: Diffusion Policy (CNN/UNet + FiLM), lerobot 0.6.0 · `n_obs_steps=2`, `horizon=48`, `n_action_steps=24`, `down_dims=(256,512,1024)` → **68,665,222** UNet · DDIM 100 train / **16 infer** · resize 240×320 → crop 216×288
**Encoder**: ResNet-18, spatial softmax (32 keypoints), **use_group_norm=true**, **pretrained=None**
**Dataset**: [`phi_so101_cubes_cylinder_recovery_v1`](https://huggingface.co/datasets/BrutalCaesar/phi_so101_cubes_cylinder_recovery_v1) — 143 eps (120 clean + 23 recovery)
**Steps / batch / seed**: **20,000** of a 100,000-step run · 64 · 1000
**Experiment**: [2026-08-12_dp-recovery-encoder-ab.md](../experiments/2026-08-12_dp-recovery-encoder-ab.md) — array `9088486`, task **0** · W&B project `phi_dp`
**Trained on**: Explorer, 1× H200, 7 h 26 m

## 🚨 This is the 20k checkpoint, and the run overfit past it

| step | 10k | 20k | 40k | 60k | 80k | 100k |
|---|---|---|---|---|---|---|
| `eval_loss` | **0.0155** | **0.0173** ← published | 0.0248 | 0.0369 | 0.0532 | 0.0636 |

Train loss ended at 0.002 while held-out loss rose ~4× over 98 epochs of 113 episodes. **The true optimum is 10k and `save_freq=20000` never saved it** — this is the best checkpoint that exists, not the best possible. Leading explanation: lerobot implements **no EMA**, which the paper's recipe assumes as its regulariser.

**This arm WON the A/B**, beating the ImageNet+BatchNorm sibling at every eval point — the opposite of the pre-registered prediction. On this dataset, spatial softmax + GroupNorm generalises better than an ImageNet prior.

## 🚨 It has a real-time deadline, unlike ACT

ACT runs its network once per chunk; DP runs the UNet `num_inference_steps` times. Measured on the Mac cockpit: **294.4 ms per chunk** (74.6 encoders + 219.8 UNet at DDIM-16). Deadline is `n_action_steps / fps`:

| `n_action_steps` | budget @30fps | |
|---|---|---|
| 8 | 267 ms | ❌ misses |
| **24 (shipped)** | 800 ms | ✅ 506 ms spare |

**Do not go below `n_action_steps` ≈ 10, and do not raise `num_inference_steps` toward lerobot's default 100** (~1370 ms/chunk, will not keep up). Missing the deadline raises no error. Full budget table: [docs/deployment §4](../docs/deployment/README.md).

## Cameras and calibration

Same preconditions as the [ACT models](act_so101_cubcyl_recovery_chunk50_noaug_3cam.md): three cameras `wrist`/`front`/`top` with **correct** keys, and the canonical `phi_follower.json` calibration is mandatory — **including the arm's base clamp position**, which is part of the same frame and which nothing in software records.

## Eval

Scored on the arm **2026-08-12**, 15 rollouts, canonical `phi_follower` calibration.

| split | n | success | mean progress |
|---|---:|---:|---:|
| control (trained on) | 3 | 67% | 0.733 |
| **held out** | **12** | **50%** | **0.650** |

Held-out by object: red cube 2/4 (0.75) · yellow cylinder 3/4 (0.80) · **white cube (45 mm) 1/4 (0.40)**.

**It ties ACT.** Paired over the 11 held-out episodes both models ran, DP − ACT = **−0.018** (t(10) = −0.13; 2 wins, 2 losses, 7 ties; sign p = 1.00). Against the no-recovery baseline: **+0.218** (t = 1.15, p = 0.45). Neither difference is resolved at this n.

| held-out | baseline `cvae` | [ACT recovery](act_so101_cubcyl_recovery_chunk50_noaug_3cam.md) | **this model** |
|---|---:|---:|---:|
| success | 27% | 55% | **50%** |
| mean progress | 0.418 | 0.636 | **0.650** |

🔑 **Held-out noise-MSE did not predict task success.** This checkpoint was knowingly past its own optimum on a loss that rose 4× monotonically, and it still matched a 100k-step ACT model and posted the highest mean progress of the three. The `eval_loss` table above is therefore **not** evidence about deployability — treat DP noise-MSE as near-useless for model selection until something demonstrates otherwise.

✅ **The `Ta=24` deadline held.** Median 67 s per rollout vs ACT's 69 s, with a tighter spread (56-95 s vs 60-153 s). No stutter, no slow-FPS warnings.

### The failure mode is contact, not reaching

Distinct from ACT, which mostly failed by hovering. Operator notes across the failures:

> ep 0 — *"at the right location but just was not able to lock in"* · ep 65 — *"one of the fingers was just tapping on the object"* · ep 95 — *"stayed a considerable time around the top of the container, which has not been seen before"*

Two candidate causes, both testable:

1. **Input resolution.** 216×288 → a 7×9 ResNet grid, against ACT's 480×640 → 15×20. Grasping a 25 mm cube is a millimetre problem. Test: re-train at ACT's resolution.
2. **Overfitting concentrated in the low-noise regime**, which sets fine action detail and which the uniformly-averaged loss under-weights. Test: the 10k checkpoint.

**Not** open-loop horizon — DP re-plans *more* often than ACT (800 ms vs 1670 ms), which predicts the opposite sign.

### One real recovery behaviour, not yet general

> ep 90 — *"Initially, it landed in front of the object, but... it was able to lean back and then grab it again and was able to do it successfully."*

First documented retry-after-failed-grasp in this project. But ep 110 — *"not able to return back and then grasp"* — and ep 65 — *"not able to learn from failures"*. One of three opportunities.

⚠️ DP's `eval_loss` is noise-MSE and is **not comparable** to ACT's L1+KL. DP vs ACT remains a rollout question only, and carries three confounds: 216×288 vs 480×640 input, 800 ms vs 1670 ms commit, and encoder init.

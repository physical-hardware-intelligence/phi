# act_so101_cubcyl_recovery_chunk50_noaug_3cam

**HF id**: [`BrutalCaesar/act_so101_cubcyl_recovery_chunk50_noaug_3cam`](https://huggingface.co/BrutalCaesar/act_so101_cubcyl_recovery_chunk50_noaug_3cam) · public
**Revision**: `cd93421` (uploaded 2026-08-11) · verified: `ACTPolicy.from_pretrained` loads **51,571,590 params** + 45,824 buffers, 3 camera features present, 9 files listed on the Hub
**Policy**: ACT (lerobot 0.6.0) · `chunk_size=50`, `n_action_steps=50`, `use_vae=true`, `kl_weight=10.0`, ResNet-18 ImageNet-pretrained with **BatchNorm frozen**
**Dataset**: [`BrutalCaesar/phi_so101_cubes_cylinder_recovery_v1`](https://huggingface.co/datasets/BrutalCaesar/phi_so101_cubes_cylinder_recovery_v1) — 143 episodes (120 clean + 23 recovery), 81,943 frames
**Steps / batch / seed**: 100,000 / 8 / 1000
**Image augmentation**: **OFF** — this is the control arm of the A/B
**Experiment**: [2026-08-10_act-recovery-augmentation.md](../experiments/2026-08-10_act-recovery-augmentation.md) — array `9066699`, task **0**
**Trained on**: Explorer, node d4052, 1× H200, **1 h 50 m** wall

## Cameras — keys are CORRECT here

No transposition, unlike the two `8bin` models in this zoo. Inherited unchanged from the parent dataset:

| Physical camera | Observation key |
|---|---|
| wrist (gripper module) | `observation.images.wrist` |
| front (Logitech Brio 101, desk level) | `observation.images.front` |
| top (EMEET C960, overhead) | `observation.images.top` |

All three `3 × 480 × 640`. Miss one and it will not load; **swap two and it loads fine and behaves badly.**

> ⚠️ **Verify index → physical camera every session.** macOS has no udev, indices reorder silently. Do not trust any written-down order, including this file's.

## 🚨 Calibration is a precondition, not a detail

Copy the canonical calibration before any rollout:

```bash
cp configs/calibration/robots/so_follower/phi_follower.json \
   ~/.cache/huggingface/lerobot/calibration/robots/so_follower/
python -m phi.utils.compare_calibration phi_follower <whatever-you-were-using>
```

This dataset was recorded on Yash's laptop, so the policy emits angles in **his** calibration frame. Confirmed on the arm 2026-08-10: run against a different calibration and it mis-grasps with **no error and no warning**. See [so-arm101 setup §3c](../docs/robots/so-arm101/02-setup.md#3c-moving-a-trained-policy-to-someone-elses-machine).

**Any number recorded under a non-canonical calibration is void.**

## Eval

Scored on the arm **2026-08-12**, 14 rollouts, canonical `phi_follower` calibration.

| split | n | success | mean progress |
|---|---:|---:|---:|
| control (trained on) | 3 | 33% | 0.533 |
| **held out** | **11** | **55%** | **0.636** |

Held-out by object: red cube 1/3 (0.47) · yellow cylinder 3/4 (0.80) · **white cube (45 mm) 2/4 (0.60)**.

Against the no-recovery baseline `act_so101_cubcyl_poshold_chunk50_cvae_3cam` (2026-08-10, 17 rollouts — control 6, 50%, 0.600; held out 11, **27%, 0.418**; white cube **0/4**):

| held-out | baseline | **this model** | [DP](dp_so101_cubcyl_recovery_tp48_paper_3cam.md) |
|---|---:|---:|---:|
| success | 27% | **55%** | 50% |
| mean progress | 0.418 | **0.636** | 0.650 |
| white cube | 0/4 | **2/4** | 1/4 |

**The white cube went 0/4 → 2/4** — the pre-registered target of the recovery data, and the clearest single movement in the whole evaluation.

⚠️ **But the aggregate improvement is not statistically resolved.** Paired over the 10 common held-out episodes: mean delta **+0.240**, sd 0.540, se 0.171, **t(9) = 1.41**; better on 4, worse on 1, tied on 5; sign test **p = 0.375**. Pairing bought almost no power here (se 0.171 paired vs 0.179 unpaired) because half the episodes tie at the 0.2 floor — the scores are nearly binary, so variance stays high. Detecting an effect this size at 80% power needs **n ≈ 40**.

Two things that raise confidence above what that test alone shows: the direction is consistent, and DP trained on the same recovery data independently posted **+0.218** over the same baseline.

⚠️ **Baseline-comparison confound.** The baseline's rollouts predate the base-clamp discovery ([setup §3b-bis](../docs/robots/so-arm101/02-setup.md)), so the rig may not have been identical. Working *against* that worry: this session's controls (33%, n=3) were **worse** than the baseline's (50%, n=6), so the rig does not look easier today.

🔑 **This model beat the baseline; it did not beat DP.** Paired DP − ACT = −0.018 (t = −0.13, 7 of 11 episodes tied). Treat ACT and DP as tied on this task.

`eval_loss` (full L1 + KL) on the 30 clean held-out episodes:

| step | 20k | 40k | 60k | 80k | 100k |
|---|---|---|---|---|---|
| this run | 0.2222 | 0.2056 | 0.2058 | 0.2053 | **0.2052** |
| [aug sibling](act_so101_cubcyl_recovery_chunk50_aug_3cam.md) | 0.2149 | 0.2112 | 0.2056 | 0.2046 | 0.2088 |

Within ~2% throughout, single seed. **No winner is readable from this**, and two things it structurally cannot measure: whether recovery improved (the holdout is all clean, no induced failures in it) and whether grasp precision changed (L1 over 50 steps is dominated by transit, not by the moment the gripper closes).

Protocol when it runs: see [docs/evaluation](../docs/evaluation/README.md). The recovery-specific test needs the object displaced mid-episode to check whether the policy re-approaches.

## Known failure modes

- Untested on hardware. Everything below is a prediction, not an observation.
- Trained on one task string, so it has no language conditioning to disambiguate object or container.
- Recovery block 1 (yellow cylinder → white bin) has 5 episodes, not 6, so any per-condition rate there rests on 17% fewer samples.

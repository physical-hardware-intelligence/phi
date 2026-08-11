# act_so101_cubcyl_recovery_chunk50_aug_3cam

**HF id**: [`BrutalCaesar/act_so101_cubcyl_recovery_chunk50_aug_3cam`](https://huggingface.co/BrutalCaesar/act_so101_cubcyl_recovery_chunk50_aug_3cam) · public
**Revisions**: `main` = `539ef57` (100k steps) · **`step-80000` branch = `2ffdc6f`** (80k steps, better clean-holdout loss) — both uploaded 2026-08-11
**Verified**: `from_pretrained` loads **51,571,590 params** + 45,824 buffers on both revisions, 3 camera features present, 9 files each
**Policy**: ACT (lerobot 0.6.0) · `chunk_size=50`, `n_action_steps=50`, `use_vae=true`, `kl_weight=10.0`, ResNet-18 ImageNet-pretrained with **BatchNorm frozen**
**Dataset**: [`BrutalCaesar/phi_so101_cubes_cylinder_recovery_v1`](https://huggingface.co/datasets/BrutalCaesar/phi_so101_cubes_cylinder_recovery_v1) — 143 episodes (120 clean + 23 recovery), 81,943 frames
**Steps / batch / seed**: 100,000 / 8 / 1000
**Image augmentation**: **ON** — `dataset.image_transforms.enable=true`
**Experiment**: [2026-08-10_act-recovery-augmentation.md](../experiments/2026-08-10_act-recovery-augmentation.md) — array `9066699`, task **1**
**Trained on**: Explorer, node d4055, 1× H200, **2 h 54 m** wall (57% more than the control, which is the cost of the augmentation)

## What the augmentation actually was

LeRobot's stock stack, `max_num_transforms=3`, so **3 of these 6 are sampled per frame**, all weight 1.0:

| | type | range |
|---|---|---|
| brightness | ColorJitter | (0.8, 1.2) |
| contrast | ColorJitter | (0.8, 1.2) |
| saturation | ColorJitter | (0.5, 1.5) |
| hue | ColorJitter | (−0.05, 0.05) |
| sharpness | SharpnessJitter | (0.5, 1.5) |
| **affine** | RandomAffine | degrees (−5, 5), **translate (0.05, 0.05)** |

**Pre-registered concern about `affine`** (written before the runs, in the experiment note): `translate=0.05` is ±5% of the frame = **±32 px of 640 ≈ 2 cm ≈ one cube width**, and `RandomAffine` moves the *image* while the *action label* stays put. That teaches "these joint targets are still correct even though the object appears a cube to the left" — the opposite of what a precision grasp needs.

If this model loses to the control on real rollouts, run the affine-off ablation before blaming augmentation as a category:

```bash
sbatch --export=ALL,AFFINE_OFF=1 --array=1 configs/hpc/train_recovery.sbatch
```

## Cameras and calibration

Identical to the [control model](act_so101_cubcyl_recovery_chunk50_noaug_3cam.md) — keys are **correct** (wrist/front/top, no transposition), and the canonical `phi_follower.json` calibration is a hard precondition of any rollout. Both warnings are spelled out there and on the Hub card; they are the two ways this policy fails silently.

## Eval

**Not run yet.** No success number exists.

`eval_loss` (full L1 + KL) on the 30 clean held-out episodes:

| step | 20k | 40k | 60k | 80k | 100k |
|---|---|---|---|---|---|
| this run | 0.2149 | 0.2112 | 0.2056 | **0.2046** | 0.2088 |
| [control](act_so101_cubcyl_recovery_chunk50_noaug_3cam.md) | 0.2222 | 0.2056 | 0.2058 | 0.2053 | 0.2052 |

Shape of the curve: augmentation is **ahead early** (0.2211 vs 0.2278 at 10k), **behind in the middle** (30k–50k), and level by 60k. Consistent with a harder training problem that converges more slowly, not with harm.

⚠️ **100k is this run's worst late checkpoint** — it regressed from 0.2046 at 80k. That is why the 80k weights are published on the `step-80000` branch. `main` stays at 100k so the head-to-head against the control uses a **fixed step** rather than selection on a metric this noisy. Rolling out both revisions answers whether checkpoint choice matters more than the augmentation does.

## Known failure modes

- Untested on hardware.
- Late-training instability at 100k (see above) — prefer `step-80000` if you want the best held-out loss, `main` if you want the controlled comparison.
- Single task string, no language conditioning.
- Recovery block 1 has 5 episodes, not 6.

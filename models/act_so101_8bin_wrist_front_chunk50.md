# act_so101_8bin_wrist_front_chunk50

**HF id**: [`BrutalCaesar/act_so101_8bin_wrist_front_chunk50`](https://huggingface.co/BrutalCaesar/act_so101_8bin_wrist_front_chunk50) · public
**Revision**: `main` (uploaded 2026-08-05; single commit)
**Policy**: ACT (lerobot 0.6.0) · `chunk_size=50`, `n_action_steps=50`, `dim_model=512`, ResNet-18 unfrozen, `kl_weight=10.0`
**Dataset**: [`BrutalCaesar/phi_so101_8bin_v1`](https://huggingface.co/datasets/BrutalCaesar/phi_so101_8bin_v1) — 89 episodes, 6 of 8 bins
**Steps / batch / seed**: 100,000 / 8 / 1000 (≈14.6 epochs, 800K samples)
**Final train loss**: 0.055
**Experiment**: [2026-08-04_act-8bin-6run.md](../experiments/2026-08-04_act-8bin-6run.md) — array `8936911`, task **3**
**Trained on**: Explorer, node d4055, **1× H200**, 1 h 49 m wall (2.6× faster than the five sibling runs that landed V100s; identical work)

## Cameras — the swap

Physical **wrist** + **front**. The dataset keys are transposed, so at inference:

| Physical camera | Observation key |
|---|---|
| wrist (gripper module) | `observation.images.top` |
| front (Brio 101, desk level) | `observation.images.front` |

Verified against pixels (episode 0, frame 200): `observation.images.top` holds a gripper close-up,
`observation.images.wrist` holds the overhead view. The naming is wrong in the data; the rig is
labelled correctly.

## Eval

**Not run yet.** No success number exists for this model.

Protocol when it runs: 20 scored rollouts **per bin**, reported per bin, never averaged.
Left bin 3 and right bin 2 are held out of training and are the generalization test.

## Known failure modes

- Unknown until rollouts are scored — the honest answer.
- Structural: `n_obs_steps=1`, so no velocity cue.
- The CVAE latent is collapsed (KL ≈ 0 throughout), so treat this as a deterministic chunk regressor. Expected for ACT at `kl_weight=10.0`, not a bug.
- Sensitive to camera pose drift; the mounts were disturbed on 2026-08-05 and realigned with [`camera_realign`](../src/phi/utils/camera_realign.py) before any rollout.

## Commands

Train (Explorer, array task 3 of [`configs/hpc/train_8bin_v1.sbatch`](../configs/hpc/train_8bin_v1.sbatch)):

```bash
sbatch --array=3 configs/hpc/train_8bin_v1.sbatch
```

Load:

```python
from lerobot.policies.act.modeling_act import ACTPolicy
policy = ACTPolicy.from_pretrained("BrutalCaesar/act_so101_8bin_wrist_front_chunk50")
```

Normalization lives in the shipped `policy_preprocessor*` / `policy_postprocessor*` files, not in
the policy weights — go through `from_pretrained` or actions come out in the wrong units.

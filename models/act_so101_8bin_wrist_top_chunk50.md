# act_so101_8bin_wrist_top_chunk50

**HF id**: [`BrutalCaesar/act_so101_8bin_wrist_top_chunk50`](https://huggingface.co/BrutalCaesar/act_so101_8bin_wrist_top_chunk50) · public
**Revision**: uploaded 2026-08-05
**Policy**: ACT (lerobot 0.6.0) · `chunk_size=50`, `n_action_steps=50`, `dim_model=512`, ResNet-18 unfrozen, `kl_weight=10.0`
**Dataset**: [`BrutalCaesar/phi_so101_8bin_v1`](https://huggingface.co/datasets/BrutalCaesar/phi_so101_8bin_v1) — 89 episodes, 6 of 8 bins
**Steps / batch / seed**: 100,000 / 8 / 1000 (≈14.6 epochs, 800K samples)
**Final train loss**: 0.056
**Experiment**: [2026-08-04_act-8bin-6run.md](../experiments/2026-08-04_act-8bin-6run.md) — array `8936911`, task **0**
**Trained on**: Explorer, node d1012, 1× Tesla V100-32GB, 4 h 41 m
**Sibling**: [act_so101_8bin_wrist_front_chunk50](act_so101_8bin_wrist_front_chunk50.md) — same chunk, front camera instead of overhead

## Cameras — both keys transposed

Physical **wrist** + **top**. This model uses both of the swapped keys, so the names are exactly
backwards:

| Physical camera | Observation key |
|---|---|
| wrist (gripper module) | `observation.images.top` |
| top (EMEET C960, overhead boom) | `observation.images.wrist` |

Wire by name and you get a fully inverted setup — overhead footage where the policy expects a
gripper close-up and vice versa. It loads and runs without complaint.

Verified against pixels (episode 0, frame 200): `observation.images.top` holds the gripper close-up,
`observation.images.wrist` holds the overhead view.

## Eval

**Not run yet.** No success number exists for this model.

Protocol when it runs: 20 scored rollouts **per bin**, reported per bin, never averaged.
Left bin 3 and right bin 2 are held out of training and are the generalization test.

Train loss is within 0.001 of the wrist+front sibling (0.056 vs 0.055), so **loss cannot rank the
two camera pairs**. The rollouts are the only thing that can.

## Known failure modes

- Unknown until rollouts are scored.
- Structural: `n_obs_steps=1`, so no velocity cue.
- CVAE latent collapsed (KL ≈ 0 throughout) — treat as a deterministic chunk regressor. Expected for ACT at `kl_weight=10.0`.
- More pose-sensitive than the wrist+front model: the overhead camera sits ~105 cm up on an articulating boom at 90° dFOV, so sag or a bump moves the observation more than the desk-level Brio does. The mounts were disturbed on 2026-08-05 and realigned with [`camera_realign`](../src/phi/utils/camera_realign.py) before any rollout.

## Commands

Train (Explorer, array task 0 of [`configs/hpc/train_8bin_v1.sbatch`](../configs/hpc/train_8bin_v1.sbatch)):

```bash
sbatch --array=0 configs/hpc/train_8bin_v1.sbatch
```

Load:

```python
from lerobot.policies.act.modeling_act import ACTPolicy
policy = ACTPolicy.from_pretrained("BrutalCaesar/act_so101_8bin_wrist_top_chunk50")
```

Normalization lives in the shipped `policy_preprocessor*` / `policy_postprocessor*` files, not in
the policy weights — go through `from_pretrained` or actions come out in the wrong units.

# π₀.₅ finetune on the 6-task language cubes/cylinder set

**Status**: running
**Started**: 2026-09-15
**Script**: [`configs/hpc/train_pi05_cubcyl.sbatch`](../configs/hpc/train_pi05_cubcyl.sbatch)

## The question

Can a 3.5B-parameter VLA, pretrained on other people's robots, follow **six different
spoken instructions** on our arm from 120 episodes?

ACT and Diffusion Policy cannot answer this at all. Neither takes text. One ACT checkpoint
does one task. The reason to spend 30k H200-steps on π₀.₅ is not a higher success rate on
the red cube, it is **one checkpoint that does all six on command**.

## Setup

| | |
|---|---|
| Base | `lerobot/pi05_base` — 3.5B params, 14.47 GB fp32 on the Hub |
| Dataset | `BrutalCaesar/phi_so101_cubes_cylinder_lang_v1` |
| | 120 episodes · 66,873 frames · 30 fps · 3 cams · state/action 6 · **6 task strings** |
| Precision | bfloat16, full finetune (nothing frozen) |
| Steps / batch | 30,000 / 16 |
| Holdout | `eval_split=0.09`, intended 12 episodes — **verify, see below** |
| Hardware | 1× H200 (141 GB) on the `gpu` partition, 8 h wall |

The six task strings, from `meta/tasks.parquet`:

```
0  pick up the red cube and place it in the cardboard box
1  pick up the red cube and place it in the white bin
2  pick up the white cube and place it in the cardboard box
3  pick up the white cube and place it in the white bin
4  pick up the yellow cylinder and place it in the cardboard box
5  pick up the yellow cylinder and place it in the white bin
```

Two objects × two destinations × a distractor. Clean enough to tell a language failure from
a manipulation failure: if it picks the right object but the wrong bin, that is grounding;
if it fumbles the grasp, that is control.

## Three things had to be fixed before this could run at all

**1. The feature names do not match, and a wrong map does not raise.**

`pi05_base` declares `base_0_rgb` / `left_wrist_0_rgb` / `right_wrist_0_rgb` and a 32-dim
state. Our dataset has `front` / `wrist` / `top` and a 6-dim state. `--rename_map` bridges
them, direction *old key → new key* (`processor/rename_processor.py:46`).

```
front → base_0_rgb           natural, both are scene views
wrist → left_wrist_0_rgb     natural
top   → right_wrist_0_rgb    NOT natural — deliberate mismatch
```

Those `right_wrist` weights were pretrained on a wrist camera and will receive an overhead
view. The bet is that finetuning adapts the tower and a third viewpoint is worth more than
matching pretraining semantics. The alternative was dropping `top` and setting
`--policy.empty_cameras=1`.

⚠️ Passing a `rename_map` **disables** `validate_visual_features_consistency`
(`policies/factory.py:649`). There is no shape check behind this. The state needs no map:
π₀.₅ pads 6 into its 32 slots internally.

**2. `--steps` must equal `scheduler_decay_steps`.**

`PI05Config.scheduler_decay_steps` defaults to 30000. Setting `--steps` to anything else
leaves the cosine schedule not landing at the end of training. Same trap as
[diffusion-policy-transformer §15](../docs/theory/diffusion-policy-transformer.md).

**3. The env was missing the `[pi]` extra.**

`lerobot[pi]` is `transformers` + `scipy`; neither was in `/scratch/gupta.yashv/envs/phi-cuda`.
Installed 2026-09-15 → **transformers 5.17.0, scipy 1.18.1**. That is a major-version
transformers, so the import chain was verified on a compute node before burning a GPU slot:

```
OK: pi05 + paligemma import clean on transformers 5.17.0
   chunk 50 | decay_steps 30000 | lr 2.5e-05
```

`PYTHONNOUSERSITE=1` is still mandatory — without it the user-site cu121 torch shadows the
env and `import torch` dies with `libnvJitLink.so.12`. Reproduced on the login node again
on 2026-09-15.

## Why the job chains itself

The `gpu` partition caps at 8 hours. ACT (51.6 M params) took 3h03m for 100k steps at batch 8
on an A100; π₀.₅ is ~68× the parameters, so 30k steps will not fit one job. The sbatch checks
for `checkpoints/last` and resumes if it finds it, so the same file is submitted repeatedly:

```bash
sbatch configs/hpc/train_pi05_cubcyl.sbatch
sbatch --dependency=afterok:<id> configs/hpc/train_pi05_cubcyl.sbatch
```

`--output_dir` is passed explicitly on the resume branch. Without it lerobot resumes into a
fresh `outputs/train/<date>/<time>_resume` directory (`configs/train.py:203`) and a four-job
chain scatters across four directories.

`save_freq=2500` so a job cut by the wall loses at most ~40 minutes.

## Checks on job 1, before queueing the chain

| # | check | why |
|---|---|---|
| 1 | three cameras found under the **renamed** keys | a silent rename failure looks like training that never learns |
| 2 | held-out episode count is 12 | per-task-string rounding is verified for 1 task (pen), **not** for 6 |
| 3 | it/s × 30,000 | sizes the chain; 3-4 jobs expected, extrapolated not measured |
| 4 | peak VRAM against 141 GB | if near, add `--policy.gradient_checkpointing=true` or halve batch |

## What the smoke job found (2026-09-15)

A 50-minute job on the `sharing` partition, run *while* the real job sat 80-deep in the
`gpu` queue with a 12-hour estimated start. It found two blockers in two submissions, both of
which would have killed the H200 job a minute after it finally started.

**1. `--policy.push_to_hub=false` is mandatory** (job 10368256).

`PI05Config` defaults `push_to_hub=True` with `repo_id=None`, and `cfg.validate()` raises
before training begins:

```
File ".../lerobot/scripts/lerobot_train.py", line 203, in train
    cfg.validate()
ValueError: 'repo_id' argument missing. Please specify it to push the model to the hub.
```

`train_pen_act.sbatch` already carried this flag; the π₀.₅ script did not. Fixed.

**2. 🔴 `google/paligemma-3b-pt-224` is a gated repo** (job 10368295). **UNRESOLVED.**

```
huggingface_hub.errors.GatedRepoError: 401 Client Error.
Cannot access gated repo for url .../google/paligemma-3b-pt-224/resolve/main/config.json
```

π₀.₅ pulls its tokenizer from Google's repo at runtime, *separately* from the
`lerobot/pi05_base` weights. Verified: `gated=manual` on the Google repo, `gated=False` on
`pi05_base`, and `pi05_base` ships **no tokenizer files** — only `config.json` and the two
processor JSONs. Explorer has no HF token.

Clearing it needs two human actions, in this order:

1. Accept the licence at <https://huggingface.co/google/paligemma-3b-pt-224> while signed in
   as the account whose token will be used.
2. `hf auth login` on Explorer, so `$HF_HOME=/scratch/gupta.yashv/.cache/huggingface` holds a
   token with that access.

Do not route around the gate with a community mirror. The gate exists because Google requires
accepting their terms, and a mirror does not change that.

**3. The rename_map is correct.** Confirmed from the resolved config dump before the tokenizer
failure, so check 1 from the table above is **passed**:

```
'rename_map': {'observation.images.front': 'observation.images.base_0_rgb',
               'observation.images.top':   'observation.images.right_wrist_0_rgb',
               'observation.images.wrist': 'observation.images.left_wrist_0_rgb'}
'input_features': {'observation.images.base_0_rgb', 'observation.images.left_wrist_0_rgb',
                   'observation.images.right_wrist_0_rgb', 'observation.state' [32]}
```

Checks 2, 3 and 4 (holdout count, it/s, peak VRAM) are still unmeasured — the job never
reached a training step.

## Results

_Blocked on the PaliGemma licence gate. Main job `10368294` remains queued on the `gpu`
partition, estimated start 2026-09-16 03:29. It is deliberately left queued: if the gate is
cleared before then it runs, and if not it fails in about a minute and releases the node._

| step | eval loss | notes |
|---|---|---|

## What this run cannot tell us

**Whether it beats ACT.** It probably will not, in distribution. From
[robot-data-scaling](../../wiki/concepts/robot-data-scaling.md): TRI's 2,835 blind rollouts
found pretraining buys robustness under shift *"with no statistically significant change in
in-distribution performance."* Judging π₀.₅ on red-cube success rate is judging it on the one
axis it was not bought for.

**Whether language grounding works.** Held-out loss averages over all six tasks and cannot
separate "picked the right object" from "picked the right bin". That needs a rollout protocol
that issues each of the six strings against a fixed scene and scores object and destination
separately. Not built yet.

**Whether the `top → right_wrist_0_rgb` mismatch cost anything.** The honest test is an A/B
against a two-camera run with `--policy.empty_cameras=1`, which is one more chain.

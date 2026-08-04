# ACT on SO-101 — wrist-camera-only baseline

**Status**: Dataset collected, training pending · **Owner**: Yash · **Created**: 2026-07-30

## ▶ Actual run (supersedes the planned numbers below where they differ)

**Dataset**: [`Parv-09/Ava_1.0_20260730_172156`](https://huggingface.co/datasets/Parv-09/Ava_1.0_20260730_172156) (public, Apache-2.0, recorded with **LeLab**)

| Property | Value |
|---|---|
| Episodes | 60 (4 bins) |
| Total frames | **21,522** |
| Mean episode | **~12.0 s** (359 frames) — plan assumed 10 s |
| Camera | `observation.images.wrist_cam`, 480×640, **AV1** crf 30 |
| State / action dim | 6 / 6 · fps 30 · `robot_type: so_follower` |
| Task | pick up the orange and put it in the plate |

**⚠️ Known dataset flaw, deliberately kept:** **one of the four bins is not in the wrist camera's field of view at episode start.** Per §0 this is *the* predicted wrist-only failure mode. Keeping it turns the dataset into an accidental controlled experiment — **report success rate per bin**, and expect that bin to be far worse. Do not average it away.

**Config**: [`configs/act_ava_1.0.sh`](../configs/act_ava_1.0.sh) (`smoke` | `bench` | `train`) · [`configs/act_ava_1.0.slurm`](../configs/act_ava_1.0.slurm) (Explorer)

### Measured throughput — why training moved off the Mac
Benchmarked on the M4 Air cockpit (MPS), 2026-07-30:

| Batch | it/s | `updt_s` | `data_s` | 50k steps |
|---|---|---|---|---|
| 2 | 2.15 | 0.510 | 0.041 | ~6.5 h |
| **8** | **0.55** | 1.701 | 0.040 | **~25.3 h** |

`data_s ≈ 0.04` throughout, so **AV1 decode is not a bottleneck** (M4 hardware decode) — it is pure compute. 25 h on a *fanless* Air that will throttle is not viable. **→ train on CUDA** (Explorer, or HF Jobs with `--job.target=a10g-small`).

Smoke test passed at 200 steps: 51,597,190 params (52 M), loss falling 9.05 → (bench) 6.89 by step 200 @ batch 8.

### Gotchas hit while wiring this up
- `lerobot-train` hard-requires the **`training` extra** (`accelerate`); the Mac env omitted it by design. Added.
- **`policy.push_to_hub` defaults to `True`**, and `validate()` then demands a `repo_id` — every invocation must pass `--policy.push_to_hub=false` unless you really are uploading.
- **`wandb login` must be done before the real run**, or a multi-hour run silently loses its charts.

---

**Hardware**: SO-101 leader+follower (Ava), **1× wrist camera only** (32×32 mm USB module)
**Goal**: Train and evaluate an ACT policy end-to-end, establishing the club's first real baseline.

> References: [ACT/ALOHA paper](https://arxiv.org/abs/2304.13705) · [Sherry Chen, "Train ACT on SO-101"](https://huggingface.co/blog/sherryxychen/train-act-on-so-101) · [LeRobot ACT docs](https://huggingface.co/docs/lerobot/en/act) · repo: [02-setup](../docs/robots/so-arm101/02-setup.md), [03-teleop-and-data](../docs/robots/so-arm101/03-teleop-and-data.md), [training](../docs/training/README.md)

---

## 0. The central constraint: one wrist camera

This is the decision everything else follows from, so it gets analysed first.

**What ACT consumes** (verified in `lerobot/policies/act/`): at each step it takes `n_obs_steps=1` frame per camera **plus joint states (proprioception)**, and emits a **chunk of `chunk_size=100` future actions**.

**What breaks with wrist-only.** The reference ALOHA/ACT setup used **4 cameras** (2 static + 2 wrist); Sherry used **2** (front + top) and explicitly credited the top camera with showing "gripper position relative to block center." With only a wrist cam:

1. **No global context at t=0.** If the object isn't in the wrist camera's field of view when the episode starts, the policy has *no visual information about where to go*. It will regress to the mean trajectory of the training set.
2. **Self-occlusion on approach.** As the gripper closes on the object, the gripper body occludes the object. The final centimetres are effectively blind.
3. **Proprioception becomes a crutch.** With a fixed object position, ACT can solve the task from joint states alone — memorising one trajectory. That *looks* like success but generalises to nothing.

**Why it's still viable — and the two mitigations that make it work:**

- **★ Mitigation A — make the wrist cam a pseudo-overview camera at t=0.** Define a fixed **home pose** where the wrist camera looks *down and out* across the whole workspace. Start **every** episode (record *and* eval) from that exact pose. The single camera then does double duty: wide survey at the start, close-up during the grasp.
- **★ Mitigation B — constrain object placement to the home FOV.** Only place the object where it is visible from home. This shrinks the task, which is the correct trade: a small task learned properly beats a big task learned never.
- **Action chunking covers the blind approach.** ACT predicts ~100 steps (~3.3 s at 30 fps) per inference, so it can *commit* to a full reach-and-grasp after its last clear look. Wrist-only + chunking is a genuinely reasonable pairing — this is the architectural reason the plan is sound.

**Honest expectation setting.** Sherry (2 cameras, 25 eps/bin) reached **90% in-distribution / 75% out-of-distribution**. Nobody in these references ran wrist-only, so **we should not assume those numbers**. Target for this run: **≥60% in-distribution on a constrained region.** Treat it as a pipeline-proving baseline, not a generalisation result.

> 🛒 **Highest-leverage purchase: a second (overview) camera.** It's the single biggest quality jump available to us. Selection criteria, the policy-resolution analysis, and specific recommendations live in [01-hardware § Cameras](../docs/robots/so-arm101/01-hardware.md#cameras-needed-for-vision-policies) — short version: a **720p fixed-focus UVC cam clears every policy in the zoo**, so buy for **field of view**, not megapixels. Run this wrist-only baseline now; re-run with two cameras and put both on the leaderboard. That comparison *is* a legitimate club experiment.

---

## 1. Task definition

**Task**: pick up a single block and place it in a fixed container.
**Language label**: `"Pick up the block and put it in the box"`

Adapted from Sherry's task, narrowed for wrist-only:

| Variable | Sherry (2 cams) | **Ours (wrist-only)** | Why |
|---|---|---|---|
| Object start region | 6 spatial bins, wide | **4 bins in a ~15×15 cm patch inside the home FOV** | Must be visible from home pose |
| Yaw variation | −45° to +45° | **−30° to +30°** | Reduce variance we can't observe well |
| Container | varied (tupperware/bowl/box) | **ONE fixed container, fixed position** | Removes a whole axis of variation |
| Block | varied colours | **ONE block** for v1 | Same reason |

Everything held fixed in v1 is a deliberate *later* axis of variation. Note it in the model card.

---

## 2. Phase 0 — Pre-flight (do once, then freeze)

### 2.1 Protect the hardware ⚠️
Sherry **wore out her gripper motor** with excessive grasp force during teleop. Prevent it:
- Pass **`--robot.max_relative_target=...`** to cap per-step joint deltas (limits velocity/force).
- Teleoperate *gently*. "Be nice to your robot."
- The STS3215 is the wear item; note that we have no spares.

### 2.2 Fix the rig and never move it
- **Tape and mark** the camera mount, the container, and the workspace bounds on the table.
- **Mark the 4 bin positions** on tape so placement is repeatable.
- **Lock camera exposure and white balance** if the driver allows. Auto-exposure drifting between sessions was one of Sherry's real failure modes; the policy overfits to lighting.
- Record and evaluate **under the same lighting**, ideally the same time of day.

### 2.3 Define and record the home pose
1. Teleoperate to a pose where the wrist cam sees the whole 15×15 cm patch **and** the container.
2. **Photograph it** and note the joint angles. Every episode starts here.
3. Sanity-check with the live viewer that the block is clearly visible in all 4 bins from home.

### 2.4 Verify the stack
```bash
source configs/ports.local.sh
lerobot-find-cameras opencv          # note the wrist cam index
```
```bash
lerobot-teleoperate \
  --robot.type=so101_follower --robot.port=$FOLLOWER_PORT --robot.id=$FOLLOWER_ID \
  --teleop.type=so101_leader  --teleop.port=$LEADER_PORT  --teleop.id=$LEADER_ID \
  --robot.cameras="{ wrist: {type: opencv, index_or_path: <IDX>, width: 640, height: 480, fps: 30} }" \
  --display_data=true
```
**Use 640×480, not 1080p.** Sherry hit **USB bandwidth contention** where camera traffic disrupted motor `sync_read` at 30 Hz. ACT downsizes images anyway; 1080p buys nothing and risks dropped frames.

---

## 3. Phase 1 — Record the dataset

### 3.1 Budget
| Item | Value |
|---|---|
| Bins | 4 (marked) |
| Episodes per bin | **15** (Sherry's Try-3 density was 25/bin; 15 is our floor given the narrower region) |
| **Total episodes** | **60** |
| Episode length | ~20 s (`episode_time_s=30` cap) |
| Reset time | 15 s |
| Frames | ≈ 60 × 20 × 30 = **36,000** |
| Wall-clock | **~2.5–3 h** including resets |

### 3.2 Command
```bash
source configs/ports.local.sh
hf auth login --token ${HUGGINGFACE_TOKEN} --add-to-git-credential
HF_USER=$(NO_COLOR=1 hf auth whoami | awk -F': *' 'NR==1 {print $2}')

lerobot-record \
  --robot.type=so101_follower --robot.port=$FOLLOWER_PORT --robot.id=$FOLLOWER_ID \
  --robot.max_relative_target=10 \
  --robot.cameras="{ wrist: {type: opencv, index_or_path: <IDX>, width: 640, height: 480, fps: 30} }" \
  --teleop.type=so101_leader --teleop.port=$LEADER_PORT --teleop.id=$LEADER_ID \
  --display_data=true \
  --dataset.repo_id=${HF_USER}/phi-act-wrist-v1 \
  --dataset.single_task="Pick up the block and put it in the box" \
  --dataset.num_episodes=60 \
  --dataset.episode_time_s=30 \
  --dataset.reset_time_s=15
```
**Keys while recording:** `→`/`n` save & next · `←`/`r` **re-record** · `Esc`/`q` stop+encode+upload.

### 3.3 Demonstration technique (this is where quality is won)
Straight from the blog's hard-won lessons, all of which matter more with one camera:

1. **★ Watch only the camera feed, not the arm.** The policy sees the feed. If you teleoperate using information the camera never captured, you are teaching an unlearnable function.
2. **★ Grasp the block at its CENTRE, not the top edge.** Sherry traced a large share of failures to top-grasps in the training data. *Garbage in, garbage out.*
3. **Start every episode from the exact home pose.** Non-negotiable for wrist-only.
4. **Be consistent in style** — same approach arc, same speed. ACT imitates; inconsistency becomes multimodality it must average over.
5. **Move deliberately and smoothly.** No jerks, no overshoot-and-correct.
6. **Re-record (`r`) any bad episode.** Never keep a failed or sloppy demo.
7. **Rotate bins in order** (bin1 ×15, bin2 ×15, …) and physically re-place the block each episode.
8. **Vary yaw ±30°** within each bin.
9. **Include the final release + small retreat** so the policy learns a clean termination.

### 3.4 Known trap
Sherry found **"uniform xy sampling turned out not to be so random"** — hand-placement clusters and leaves gaps. **Use the marked bins**; don't freehand it.

---

## 4. Phase 2 — QA the data *before* training

Never train on unverified data.

```bash
lerobot-dataset-viz --repo-id ${HF_USER}/phi-act-wrist-v1 --episode-index 0
```
Or the [Hub visualiser](https://huggingface.co/spaces/lerobot/visualize_dataset).

**Checklist** (spot-check ≥8 episodes across all 4 bins):
- [ ] Block visible from the home pose in **every** episode's first frame ← *the wrist-only make-or-break*
- [ ] No dropped/black frames; camera never disconnected
- [ ] Lighting/colour consistent across episodes
- [ ] Gripper actions look clean (no thrash)
- [ ] Grasps are at block **centre**
- [ ] Task label correct; episode count = 60
- [ ] Bin coverage roughly balanced

**Then verify the robot can reproduce an episode:**
```bash
lerobot-replay \
  --robot.type=so101_follower --robot.port=$FOLLOWER_PORT --robot.id=$FOLLOWER_ID \
  --dataset.repo_id=${HF_USER}/phi-act-wrist-v1 --dataset.episode=0
```
If replay fails, the data is bad. Fix before training.

**File the dataset card** in [`datasets/`](../datasets/).

---

## 5. Phase 3 — Train

### 5.1 ACT defaults (verified in the installed lerobot 0.6.0)
Do **not** change these for the baseline; they're the paper's configuration. What each of these does to the tensor shapes, and how they set your VRAM ceiling: [ACT, shape by shape](../docs/theory/act-shapes.md).

| Param | Default | Note |
|---|---|---|
| `chunk_size` | **100** | ~3.3 s of actions at 30 fps |
| `n_action_steps` | **100** | executes the full chunk |
| `n_obs_steps` | 1 | single frame in |
| `vision_backbone` | `resnet18` (ImageNet) | per camera |
| `dim_model` / `n_heads` / `dim_feedforward` | 512 / 8 / 3200 | |
| encoder / decoder layers | 4 / 1 | |
| `use_vae` / `latent_dim` / `kl_weight` | True / 32 / 10.0 | the CVAE objective |
| `optimizer_lr` / backbone lr / wd | 1e-5 / 1e-5 / 1e-4 | |
| `temporal_ensemble_coeff` | None (off) | see §6.3 |
| default `batch_size` / `steps` | 8 / 100k | we override both |

### 5.2 Smoke test on the Mac first (10 min, catches config errors cheaply)
```bash
lerobot-train \
  --dataset.repo_id=${HF_USER}/phi-act-wrist-v1 \
  --policy.type=act --policy.device=mps \
  --steps=200 --batch_size=2 \
  --output_dir=outputs/train/act_smoke --job_name=act_smoke --wandb.enable=false
```
Loss should decrease. This only proves the pipeline runs — **do not train for real on the Mac** (fanless M4 throttles).

### 5.3 Real run — on CUDA
Preferred: **Explorer HPC** (`/scratch/gupta.yashv/`, GPU partition) or **HF Jobs**.
```bash
lerobot-train \
  --dataset.repo_id=${HF_USER}/phi-act-wrist-v1 \
  --policy.type=act \
  --policy.device=cuda \
  --batch_size=32 \
  --steps=30000 \
  --save_freq=5000 \
  --output_dir=outputs/train/act_phi-act-wrist-v1 \
  --job_name=act_phi-act-wrist-v1 \
  --wandb.enable=true \
  --policy.repo_id=${HF_USER}/act-phi-wrist-v1 \
  --seed=1000
```
- **30k steps** suits ~36k frames. Sherry's 52M-param ACT took **~4 h on an RTX 3080**; an H200/A100 will be far quicker.
- If VRAM-limited (e.g. RTX 2060 6 GB), drop `--batch_size=8` and expect a longer run.
- **`--seed=1000`** — reproducibility is a Φ rule.

Resume:
```bash
lerobot-train --config_path=outputs/train/act_phi-act-wrist-v1/checkpoints/last/pretrained_model/train_config.json --resume=true
```

### 5.4 Checkpoint selection
Sherry selected the checkpoint with the **lowest eval loss**, not the last one. Track in W&B and pick accordingly. **Loss does not linearly predict success rate** — verify on the robot.

---

## 6. Phase 4 — Evaluate on the robot

### 6.1 Rollout
```bash
lerobot-rollout \
  --robot.type=so101_follower --robot.port=$FOLLOWER_PORT --robot.id=$FOLLOWER_ID \
  --robot.max_relative_target=10 \
  --robot.cameras="{ wrist: {type: opencv, index_or_path: <IDX>, width: 640, height: 480, fps: 30} }" \
  --policy.path=${HF_USER}/act-phi-wrist-v1 \
  --policy.device=mps
```
> Version note: on-robot inference is `lerobot-rollout` in ≥0.6.0 (**v0.5.1 used `lerobot-record`**). ⚠️ **Keep a hand near the power switch on the first rollout.**

### 6.2 Protocol — adopt Sherry's progress score
Binary success hides all the signal. Score every rollout:

| Stage | Score |
|---|---|
| reached block | 0.2 |
| grasped block | 0.4 |
| reached container | 0.7 |
| released block | 0.8 |
| block in container | **1.0** |

**Run 20 rollouts**: 5 per bin, block re-placed each time, always from the home pose. Log success rate **and** mean progress.
**Target: ≥60% in-distribution.** Also try ~5 rollouts at *unseen* positions inside the patch for a cheap OOD read.

### 6.3 Two knobs to try if it's jittery
- **`--policy.temporal_ensemble_coeff=0.01`** (with `n_action_steps=1`): the paper's temporal ensembling, which smooths actions by averaging overlapping chunks. Off by default; worth an A/B.
- Reduce `n_action_steps` (e.g. 50) for more frequent re-planning — helps when the scene moves, costs compute.

---

## 7. Phase 5 — Diagnose and iterate

Map the symptom to the fix (blog-derived + wrist-only specific):

| Symptom | Likely cause | Fix |
|---|---|---|
| Moves confidently to the **wrong place**, ignores the block | Object not visible at t=0 → policy memorised the mean trajectory | **The wrist-only failure mode.** Fix the home pose/FOV; shrink the region |
| Reaches correctly, **misses the grasp** | Grasps at top instead of centre in the data; blind final approach | Re-record with centre grasps; add episodes |
| **Never recovers** after a failed grasp | No recovery behaviour in the data | Record some demos that fail, then retry |
| Fails only **at region edges** | Coverage gap (Sherry saw exactly this) | Add episodes at the edges |
| Jittery / vibrating | Chunk stitching | Temporal ensembling (§6.3) |
| Good in-distribution, bad OOD | Insufficient diversity | More bins, more yaw variation |

**Iteration order** (cheapest first): more data at failure locations → recovery demos → temporal ensembling → more steps → **add the second camera**.

---

## 8. Deliverables (Φ rules)
- [ ] Dataset card in [`datasets/`](../datasets/) — id, task, 60 eps, wrist-only, collector, known issues
- [ ] Model card in [`models/`](../models/) — dataset, policy, steps, seed, eval score, failure modes
- [ ] Committed config + exact command in this file
- [ ] Result on the [leaderboard](../docs/evaluation/README.md)
- [ ] Update this doc with **actual** numbers (planned → done)

## 9. Sequenced checklist
1. [ ] Buy/borrow a second camera (parallel; doesn't block)
2. [ ] Pre-flight: mount cam, tape rig, lock exposure, define home pose (§2)
3. [ ] Mark 4 bins on tape
4. [ ] Teleop verify: block visible from home in all bins (§2.4)
5. [ ] Record 60 episodes (§3) — ~3 h
6. [ ] QA + replay (§4)
7. [ ] Smoke-test train on Mac, 200 steps (§5.2)
8. [ ] Train 30k steps on CUDA (§5.3)
9. [ ] Pick best checkpoint by eval loss (§5.4)
10. [ ] 20 scored rollouts (§6)
11. [ ] Write up, cards, leaderboard (§8)
12. [ ] **Then** re-run with 2 cameras and compare

# phi_so101_cubes_cylinder_v1

**HF id**: [`BrutalCaesar/phi_so101_cubes_cylinder_v1`](https://huggingface.co/datasets/BrutalCaesar/phi_so101_cubes_cylinder_v1) (public)
**Robot**: SO-ARM101 leader–follower (`so_follower`) · **Task string**: `"pick up the cubes/cylinder and place it in the box"`
**Collected**: 2026-08-06, Room 1012 · **Operator**: Yash
**Format**: LeRobotDataset **v3.0**, `v3.0` tag present · **Local root**: `~/.cache/huggingface/lerobot/BrutalCaesar/phi_so101_cubes_cylinder_v1_20260806_161432`

| | |
|---|---|
| Episodes | **120** |
| Frames | **66,873** (37.2 min) |
| fps | 30 |
| Cameras | 3 × 640×480 |
| Size | 1.07 GB |

## ✅ Camera keys are correct

Verified against pixels at episodes 8, 39 and 67 — i.e. on both sides of both resume boundaries,
because a mid-collection index shift is worse than a consistent swap and nothing in LeRobot detects it.

| Dataset key | Physical camera |
|---|---|
| `observation.images.wrist` | wrist / gripper |
| `observation.images.front` | desk-level front |
| `observation.images.top` | overhead |

No transposition. Do **not** apply `phi_so101_8bin_v1`'s swapped mapping to this dataset.

## Collection design

**3 objects × 2 bins × 20 episodes = 120.** All indices below are zero-based `episode_index`.

| `episode_index` | Object | Bin |
|---|---|---|
| 0-19 | **red** cube (3D printed) | cardboard box |
| 20-39 | **red** cube | white 3D-printed bin |
| 40-59 | orange/yellow cylinder | cardboard box |
| 60-79 | orange/yellow cylinder | white 3D-printed bin |
| 80-99 | white cube (larger) | cardboard box |
| 100-119 | white cube | white 3D-printed bin |

Object start positions were varied within each block. `single_task` writes one identical string to all
120 episodes, so **this table is the only record of the design** — the dataset itself carries no
per-episode object or bin label.

Object colours confirmed from pixels (episodes 5, 45, 85), not from notes: block 1 is **red**, not
yellow.

## ✅ Only one bin was on the table at a time

This is the property that makes the dataset well-posed for ACT. The target is always the single
visible bin, so no two episodes present the same observation with different correct actions. There is
no destination multimodality for the L1 objective to average away.

Contrast [`phi_so101_8bin_v1`](phi_so101_8bin_v1.md), which asked for spatial generalization across 8
bin positions from ~15 episodes each.

## Collection notes and caveats

**Recorded in three sessions**, interrupted twice:

1. Episodes 0-8 — ended by a Feetech bus error (`no status packet` on all six IDs; power/USB, not a servo fault)
2. Episodes 9-39
3. Episodes 40-67 — ended when the iPad powered off
4. Episodes 68-119

Camera mapping was re-verified after each resume and held throughout.

**Episode lengths shorten over the dataset.** The first nine average 23.6 s; the last runs average
15-17 s as the operator stopped riding the 25 s ceiling. Later episodes are the tighter
demonstrations.

**⚠️ [UNVERIFIED] Camera frame rate during roughly the first 50 episodes.** The operator observed an
on-screen warning that camera fps was dropping to ~20 Hz while the MacBook ran on battery; it stopped
after plugging in. This is first-hand and is recorded as reported, but **the data does not currently
corroborate it**:

- Timestamps cannot test it. LeRobot writes them synthetically as `frame_index / fps` (verified), so they are perfectly regular regardless of what the cameras delivered. "0 late frames" is meaningless here.
- A pixel-level stall test — image unchanged between consecutive frames while the joints clearly moved — gives **4-9% across the whole dataset with no early-vs-late pattern**. Some episodes had too few moving frames to measure.

Treat as a hypothesis to check if episodes 0-49 underperform, not as an established defect.

**⚠️ The white cube is low contrast.** A white object on a light-wood table is hard to see in the
overhead view; it nearly disappears at episode 85. The red cube and orange cylinder both stand out
clearly. **If any block underperforms, episodes 80-119 are the better bet** — that disadvantage is
present in every frame, whereas the fps issue is intermittent and unconfirmed.

## Suggested splits

Unlike the earlier datasets, this one has real structure to hold out:

- **Hold out an object** (e.g. 80-119) → tests generalization to an unseen object shape and colour
- **Hold out a bin** (e.g. 20-39, 60-79, 100-119) → tests generalization to an unseen container
- **Both** → the hardest, and the most honest

At 66,873 frames, 100k steps at batch 8 is **≈12 epochs** — a far healthier regime than
`phi_so101_redcube_in_box_v1`, where the same step count meant 71 epochs over 20 episodes.

## Related

- [`phi_so101_redcube_in_box_v1`](phi_so101_redcube_in_box_v1.md) — the 20-episode positive control that preceded this
- [`phi_so101_8bin_v1`](phi_so101_8bin_v1.md) — the 8-bin dataset with transposed camera keys
- [03-teleop-and-data](../docs/robots/so-arm101/03-teleop-and-data.md) — recording, resume and the camera-index traps

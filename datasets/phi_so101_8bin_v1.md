# phi_so101_8bin_v1

**HF id**: [`BrutalCaesar/phi_so101_8bin_v1`](https://huggingface.co/datasets/BrutalCaesar/phi_so101_8bin_v1) (public)
**Robot**: SO-ARM101 leader–follower (`so_follower`) · **Task string**: `"pick up the duck and place it in the box"`
**Collected**: 2026-08-03, Room 1012 · **Collected by**: Yash (operator), Parv + Sai (setup, bin resets)
**Format**: LeRobotDataset **v3.0** · **Visualize**: https://huggingface.co/spaces/lerobot/visualize_dataset

| | |
|---|---|
| Episodes | **119** |
| Frames | **72,518** (40.3 min) |
| fps | 30 |
| Cameras | 3 × 640×480, AV1 crf 30 |
| Size | 1.25 GB |
| Episode length | min 466 / mean 609 / max 750 frames |
| Frame timing | **0 late frames out of 72,518** — every interval exactly 33.33 ms |

Verified with `python -m phi.utils.verify_dataset <root> --split 59`: metadata agrees with data, indices contiguous, every referenced video shard present, `stats.json` complete.

**Loads as** `LeRobotDataset("BrutalCaesar/phi_so101_8bin_v1")` — resolves the `v3.0` tag (added 2026-08-03; see [issue 7](#known-issues)). If you push anything to this repo, **re-point that tag**, or every loader keeps fetching the pre-push snapshot: [03-teleop-and-data §3c](../docs/robots/so-arm101/03-teleop-and-data.md#-after-hf-upload-you-must-create-the-version-tag-yourself).

---

## 🚨 The camera keys are mislabelled. Read this before you train.

**`wrist` and `top` hold each other's footage.** The index-to-name mapping in the record command did not match the physical rig, and we deliberately left it that way rather than fix it mid-collection — a half-corrected dataset would have had one key meaning two different cameras, which is far worse and which `sanity_check_dataset_robot_compatibility` does **not** detect. See [03-teleop-and-data](../docs/robots/so-arm101/03-teleop-and-data.md#-never-change-the-camera-index-mapping-between-sessions).

| Dataset key | Physical camera | Device |
|---|---|---|
| `observation.images.wrist` | **overhead / top** | EMEET C960, 90° dFOV, on the boom arm |
| `observation.images.front` | front _(correct)_ | Logitech Brio 101, 58° dFOV, magnetic stand |
| `observation.images.top` | **wrist / gripper** | 32×32 mm bundled USB module |

The mapping is **consistent across all 119 episodes**, so the dataset is internally valid. But every training config must be written in *dataset keys*, not in physical names.

**Translation for the planned runs:**

| You want (physical) | Load these keys | Drop |
|---|---|---|
| all three | `wrist`, `front`, `top` | — |
| top + front, drop wrist | `wrist`, `front` | **`top`** |
| wrist + top | `top`, `wrist` | **`front`** |
| wrist + front | `top`, `front` | **`wrist`** |

---

## Collection design

8 bin positions, **4 left + 4 right of the follower arm**, 15 episodes each (5 per box × 3 boxes) = 120 planned. Recorded as two blocks with a break, appended into one dataset via `--resume=true`.

| Block | Episodes | Count | Mean episode |
|---|---|---|---|
| **A — left** | 0–58 | 59 | 22.6 s |
| **B — right** | 59–118 | 60 | 18.1 s |

Block B is shorter per episode because the operator started moving on the voice cue instead of waiting it out. Block A carries ~3–4 s of leading dead time per episode; this cannot be trimmed after the fact (`lerobot-edit-dataset` has no frame-trim operation).

**One episode was deleted**: original index 18 (22 frames, 0.7 s — not a demonstration, and shorter than the smallest action chunk). Deletion **renumbered everything after it**, so the left block has 59 rather than 60 episodes and one bin is short by one.

## Bin → episode mapping (CONFIRMED 2026-08-03)

Bins were recorded **in ascending order 1→4 within each block**, left block first. Confirmed by Yash; the `0-15` annotation on the collection whiteboard is not a valid reading and should be ignored.

Two bins are **held out of training entirely** and used for evaluation as out-of-distribution positions (circled in blue on the whiteboard, left bin 3 additionally marked "ood"). Counts and frames below are measured from the dataset, not assumed.

| Block | Bin | Episodes | #ep | Frames | Mean ep | Use |
|---|---|---|---|---|---|---|
| Left | 1 | 0–14 | 15 | 10,173 | 22.6 s | train |
| Left | 2 | 15–28 | **14** | 10,194 | 24.3 s | train _(original ep 18 deleted)_ |
| Left | **3** | **29–43** | 15 | 9,634 | 21.4 s | **🔴 HELD OUT — eval** |
| Left | 4 | 44–58 | 15 | 9,958 | 22.1 s | train |
| Right | 1 | 59–73 | 15 | 7,783 | 17.3 s | train |
| Right | **2** | **74–88** | 15 | 8,114 | 18.0 s | **🔴 HELD OUT — eval** |
| Right | 3 | 89–103 | 15 | 8,724 | 19.4 s | train |
| Right | 4 | 104–118 | 15 | 7,938 | 17.6 s | train |

| | Episodes | Frames | Duration |
|---|---|---|---|
| **TRAIN** | **89** | **54,770** | 30.4 min |
| **EVAL (OOD)** | **30** | **17,748** | 9.9 min |
| Total | 119 | 72,518 | 40.3 min |

Since the dataset carries no per-episode bin label, **this table is the only record of the design** — `single_task` writes one identical string to all 119 episodes.

### The exact train split, for `--dataset.episodes`
```
[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118]
```
And the held-out eval episodes:
```
[29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88]
```
⚠️ These are **post-deletion** indices. Do not re-derive them from the original 120-episode numbering — everything after original episode 18 shifted down by one.

## Training matrix (2026-08-03 plan)

Three axes: **camera set × action chunk × augmentation.**

| Owner | Cameras (physical) | Dataset keys | Chunks | Aug | Runs |
|---|---|---|---|---|---|
| **Sai** | all three | `wrist`, `front`, `top` | 50 / 75 / 100 | on + off | 6 |
| **Parv** | top + front, no wrist | `wrist`, `front` | 50 / 75 / 100 | on + off | 6 |
| **Yash** | wrist + top | `top`, `wrist` | 50 / 75 / 100 | on + off | 6 |
| **Yash** | wrist + front | `top`, `front` | 50 / 75 / 100 | on + off | 6 |

**24 runs total** (4 camera sets × 3 chunks × 2 augmentation states).

### Flags (verified against the installed `lerobot-train`)
```bash
--policy.chunk_size=50 --policy.n_action_steps=50      # set BOTH; n_action_steps > chunk_size raises
--dataset.image_transforms.enable=true                  # or false — the augmentation toggle
--dataset.episodes="[...]"                              # the train split above
```
Related transform knobs if the default set needs tuning: `--dataset.image_transforms.max_num_transforms`, `.random_order`, `.tfs`. Preview what a transform actually does with `lerobot-imgtransform-viz` before spending cluster time on it.

### Keep the augmentation comparison honest
Augmentation on/off is the one axis here that is **cheap to confound**. Vary it alone: same camera set, same chunk, same seed, same train split, same step count. If two runs differ in augmentation *and* anything else, the comparison is worthless — and with 24 runs the temptation to change two things at once is real.

**Report success rate per held-out bin, not averaged.** With 15 eval episodes per bin, a single success is 6.7%, so per-bin numbers are already noisy — and averaging the two OOD bins throws away the left-vs-right comparison the design exists to make.

⚠️ **One confound to control for in every result.** Left-block episodes carry ~3–4 s of leading dead time and average 22–24 s; right-block episodes average 17–19 s (see [Collection design](#collection-design)). So **left bin 3 and right bin 2 are not equally difficult eval targets** — they differ in data quality as well as position. A model doing better on right bin 2 may be telling you about dead time, not about spatial generalization.

## Known issues

1. **`wrist` and `top` keys are swapped** relative to physical cameras (see above). Consistent throughout, so the dataset is valid — but every training config must be written in dataset keys.
2. **Bin identity is not in the dataset.** The mapping table above is the only record; it is confirmed but it lives only here. Do not re-derive it from the original 120-episode numbering.
3. **Block A has ~3–4 s of leading dead time** per episode; recording begins while the "Recording episode N" announcement is still playing. Ambiguous supervision — near-identical observations map to both idle and reaching actions. Block B is much better.
4. **One episode deleted** (original 18), so left bin 2 has 14 episodes rather than 15. Slight imbalance in the factorial design.
5. **No per-episode task labels.** All 119 share one instruction string, so the 3 boxes are not distinguishable from metadata. Fine for ACT; a limitation for any language-conditioned VLA later.
6. A `phi_so101_8bin_v1_20260803_150735_old/` backup (pre-deletion, 120 episodes, 1.2 GB) exists locally. Safe to delete now that the upload is verified.
7. **RESOLVED 2026-08-03 — no `v3.0` tag on the Hub repo.** Because the dataset was pushed with `hf upload` rather than LeRobot's `push_to_hub`, the codebase-version tag was never created, and every `LeRobotDataset(...)` call failed with `RevisionNotFoundError` before downloading a byte. Tag created, load verified. Anyone who hit it can simply retry — nothing to clean up.

## Reproducing / extending

Local root used for collection:
```
~/.cache/huggingface/lerobot/BrutalCaesar/phi_so101_8bin_v1_20260803_150735
```
Record + resume commands, the camera-index rules and the verification step: [03-teleop-and-data](../docs/robots/so-arm101/03-teleop-and-data.md). Rig and camera hardware: [01-hardware](../docs/robots/so-arm101/01-hardware.md).

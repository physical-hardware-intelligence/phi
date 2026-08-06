# SO-ARM101 — Teleop & Data Collection

Good policies come from good data. This is the highest-leverage skill in the club.

> Reference: [il_robots](https://huggingface.co/docs/lerobot/en/il_robots) · [LeRobotDataset](https://huggingface.co/docs/lerobot/en/lerobot-dataset-v3) · ["what makes a good dataset"](https://huggingface.co/blog/lerobot-datasets)

## 1. Log in to the Hugging Face Hub
Datasets live on the Hub, not in git. Use the **new** `hf` CLI (not `huggingface-cli`):
```bash
hf auth login --token ${HUGGINGFACE_TOKEN} --add-to-git-credential
HF_USER=$(NO_COLOR=1 hf auth whoami | awk -F': *' 'NR==1 {print $2}'); echo $HF_USER
```

## 2. Record

**Look before you record.** `lerobot-record` has **no confirmation prompt** — it connects, says "Recording episode 0", and starts writing. To check framing and that the follower tracks, run the identical command as [`lerobot-teleoperate`](02-setup.md#5-add-cameras) with `--display_data=true` and no `--dataset.*` flags. The word "dataset" appears zero times in that script, so it cannot write anything.

```bash
source configs/ports.local.sh
lerobot-record \
  --robot.type=so101_follower --robot.port=$FOLLOWER_PORT --robot.id=$FOLLOWER_ID \
  --teleop.type=so101_leader  --teleop.port=$LEADER_PORT  --teleop.id=$LEADER_ID \
  --robot.cameras="{ \
    wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: MJPG}, \
    front: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30, fourcc: MJPG}, \
    top:   {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: MJPG} }" \
  --display_data=true \
  --dataset.repo_id=${HF_USER}/phi_so101_v1 \
  --dataset.single_task="pick up the duck and place it in the box" \
  --dataset.num_episodes=15 \
  --dataset.episode_time_s=20 \
  --dataset.reset_time_s=15 \
  --dataset.fps=30 \
  --dataset.push_to_hub=false
```
Key `--dataset.*` args: `repo_id` (Hub id) · `single_task="..."` (the language label) · `num_episodes` (default 50; when resuming, this is the number of *additional* episodes) · `episode_time_s` (default 60) · `reset_time_s` (default 60) · `push_to_hub=False` to skip auto-upload · `--resume=true` to append.

`fourcc: MJPG` is not optional at three cameras — see [02-setup §5c](02-setup.md#5c-bandwidth-set-fourcc-mjpg).

### 🚨 `repo_id` gets a timestamp appended — you must reuse the stamped name
On **creation** LeRobot calls `stamp_repo_id()`, so `--dataset.repo_id=you/phi_so101_v1` actually produces:

```
you/phi_so101_v1_20260803_143210
```

The timestamp is *not* re-applied on resume, "where the existing repo_id (already stamped) must be preserved." So **`--resume=true` with the original unstamped name will not find your dataset.** Grab the real name right after the first block starts and reuse it verbatim for every later block:

```bash
ls -t ~/.cache/huggingface/lerobot/${HF_USER}/ | head -1
```

### Recording controls
`→` or `n` = **exit the current phase early** · `←` or `r` = **re-record this episode** · `Esc` or `q` = **stop recording**. Printed at startup as `Right/Left/Esc, or n=next, r=re-record, q=quit`. The letters exist because arrow escape-sequences get mangled over SSH; locally either works.

Two things worth knowing:
- **`→` exits the *reset* phase too.** `reset_time_s` is a **maximum**, not a fixed cost — once the object is repositioned, hit `→` and the next episode begins. A 60-episode block budgeted at 35 s/episode often lands well under that.
- **`r` discards the episode and does not count it** (`clear_episode_buffer()` then `continue`), so you still end up with exactly `num_episodes` good episodes. Re-record freely.

> ⚠️ **macOS: the keys may only work while Terminal has focus.** The global (pynput) listener needs **System Settings → Privacy & Security → Input Monitoring** for your terminal. Without it the code checks `IS_TRUSTED`, stops the global listener, and falls back to reading the TTY — so if you click into the rerun window and press `r` on a bad episode, **nothing happens and the bad episode is saved.** Test it deliberately in the first block: click rerun, press `r`, and confirm the terminal prints "Left arrow key pressed...". Either grant the permission or keep Terminal focused while recording.

### 🚨 `--resume=true` also requires `--dataset.root`
Resume refuses to run without it and the error is explicit: *"resume() requires an explicit 'root' directory because it creates a DatasetWriter. Writing into the revision-safe Hub snapshot cache (used when root=None) would corrupt the shared cache."* `root` is the **full dataset directory**, so it repeats the stamped name:

```bash
--dataset.repo_id=${HF_USER}/phi_so101_8bin_v1_20260803_150735 \
--dataset.root=$HOME/.cache/huggingface/lerobot/${HF_USER}/phi_so101_8bin_v1_20260803_150735 \
--resume=true
```

### Worked example — one dataset across two sessions
The 2026-08-03 collection: 8 bin positions, left half then a break then the right half, appended into a single dataset ([BrutalCaesar/phi_so101_8bin_v1](https://huggingface.co/datasets/BrutalCaesar/phi_so101_8bin_v1)).

**Session 1** — the [§2 record command](#2-record) as written, `--dataset.num_episodes=60`, no `--resume`. Then capture the stamped name immediately, because every later command needs it:
```bash
ls -t ~/.cache/huggingface/lerobot/${HF_USER}/ | head -1
```
**Session 2** — identical command plus the three lines above. `num_episodes=60` again, because it means *additional* episodes on resume.

Result: 119 episodes / 72,518 frames / 40.3 min / 1.25 GB, mean 22.6 s in block A and 18.1 s in block B (block B is shorter because the operator started moving on the voice cue instead of waiting it out — see the dead-time note in [Recording controls](#recording-controls)).

### ⚠️ Never change the camera index mapping between sessions
On 2026-08-03 the `top` and `wrist` folders ended up holding each other's footage, because the index-to-name mapping in the record command did not match the physical rig. **We deliberately did not fix it mid-collection.**

The reason: camera names are arbitrary labels to the policy, which only needs the same key to mean the same physical camera in *every* episode. Correcting the labels halfway would have given `observation.images.top` one camera for episodes 0-58 and a different one for 59-118 — and **`sanity_check_dataset_robot_compatibility` would not have caught it**, because it compares keys and resolutions, not content.

So: **consistency beats correctness once collection has started.** Verify the mapping with `python -m phi.utils.camera_align wrist=0 front=1 top=2` before session 1, reuse whatever indices keep each key on the same physical camera, and record the true mapping in the dataset card. Relabel only after the dataset is complete and backed up — and note that a plain directory rename **will not work**, because `meta/episodes` stores a per-camera `chunk_index`/`file_index` and the shard counts differ per camera.

### Stopping, resuming, and what is safe
- `dataset.save_episode()` runs **after every episode**, and `video_encoding_batch_size` defaults to **1**, so each episode's video is encoded immediately rather than batched.
- `dataset.finalize()` sits in a `finally:` block, so **`Esc` and `Ctrl-C` are both safe** — metadata still gets written. A **hard kill or power loss** can leave the chunk file unfinalised. Exit deliberately.
- `push_to_hub` also runs in that `finally:`, **once, at the very end**. Nothing streams to the Hub while you record.
- On `--resume=true`, `sanity_check_dataset_robot_compatibility` compares **`robot_type`, `fps` and `features`** (every camera key and its resolution) and raises `ValueError` on a mismatch. A missing or misnamed camera fails **loudly**.

> ⚠️ **What that check does NOT catch: two cameras swapping indices.** It verifies the keys exist, not that `top` is still the overhead camera. If `front` and `top` transpose across a break you will record correct-looking keys over the wrong images, silently. **Re-verify the mapping before every resume** — 30 seconds:
> ```bash
> PYTHONPATH=src python -m phi.utils.camera_align wrist=0 front=1 top=2
> ```
> And **never re-calibrate mid-dataset**: calibration sets per-joint normalisation, so later episodes would encode different physical angles behind identical numbers. Keep `--robot.id` / `--teleop.id` fixed for the life of a dataset.

### When do camera indices change?
On macOS `cv2.VideoCapture(index)` uses AVFoundation, where the index is just a **position in the OS device-enumeration list** — so anything that re-registers a device can move it:

| Trigger | Why |
|---|---|
| **Flipping a per-port switch on the powered hub** | electrically an unplug/replug → the device re-registers, possibly later in the list |
| Replugging cameras in a different order | later plug, later position |
| **Reboot, or wake from sleep** | everything re-registers; order can differ |
| Moving a camera to a different physical port | different registration point |
| The Mac's built-in FaceTime camera | always occupies a slot of its own |

- **Mid-run they cannot change.** `cv2.VideoCapture` holds a handle to a specific device for the process lifetime; OS-level re-enumeration does not reassign an open handle.
- **Across runs the order is deterministic, not random.** Touch nothing and the same indices come back — so a stop-and-immediately-restart needs no re-check, while a break where the laptop slept does.
- ⚠️ **macOS offers no stable identifier.** Linux can pin a camera by path (`/dev/video*`, `by-id`); `camera_opencv.py` special-cases Linux only, and macOS is **integer index and nothing else**. Once the mapping is right, leave the hub switches alone and stop the Mac sleeping.

### Metadata the dataset does NOT record
`--dataset.single_task` writes **one identical string into every frame** — there is no per-episode task label from the CLI. So if a session varies the object or the target location, **episode order is the only record of which was which.** Record in blocks (one block per condition), note the episode ranges as you go, and put them in the dataset card. Without that, per-condition success rate is unrecoverable and the experimental design is lost.

Local cache: `~/.cache/huggingface/lerobot/{repo-id}`. Manual push: `hf upload ${HF_USER}/phi_so101_v1_<stamp> ~/.cache/huggingface/lerobot/${HF_USER}/phi_so101_v1_<stamp> --repo-type dataset`.

### Data-collection rules of thumb (from LeRobot)
- **≥50 episodes** for a first ACT task; **~10 episodes per distinct location/arrangement** so the policy generalizes.
- Vary object position, lighting, and start pose — a policy only learns what it *sees* vary.
- Keep the camera **fixed** relative to the workspace between recording and evaluation.
- Bad/aborted episodes: re-record (`r`) rather than keep noise.

## 3. Verify before you train or upload
Run this after collection and **again after any `lerobot-edit-dataset` operation**, which rewrites chunk files:

```bash
PYTHONPATH=src python -m phi.utils.verify_dataset \
  ~/.cache/huggingface/lerobot/${HF_USER}/<stamped-dataset> --split 59
```

It checks what watching videos cannot: metadata agrees with the data, episode and frame indices are contiguous, **every referenced video file exists** (and flags files nothing points at), `from_timestamp` increases within each video shard, `stats.json` is complete, and — the useful one — **whether the camera pipeline ever dropped a frame**. `--split N` prints per-block stats when a dataset was collected in two sessions. Exit code 0 means safe.

On the 2026-08-03 dataset it reported 0 late frames out of 72,518 (every interval exactly 33.33 ms) and caught a **22-frame, 0.7 s episode** that was not a demonstration at all.

> ⚠️ **Read every shard, not the first one.** `data/`, `meta/episodes/` and each camera's `videos/` are all sharded across several files. A check that globs and reads only `[0]` makes a perfectly good dataset look corrupt — that false alarm is why this is a tool and not a snippet.

### Deleting bad episodes
```bash
lerobot-edit-dataset --operation.type=delete_episodes \
  --repo_id=${HF_USER}/<stamped-dataset> \
  --root=$HOME/.cache/huggingface/lerobot/${HF_USER}/<stamped-dataset> \
  --operation.episode_indices="[18]"
```
Two behaviours to expect: it **renumbers** every later episode (so any bin/condition mapping you kept by episode range shifts down), and it leaves a full **`<dataset>_old/`** backup beside the original — keep that until the verifier passes, then delete it to reclaim the space. Available operations: `delete_episodes, split, merge, remove_feature, modify_tasks, convert_image_to_video, recompute_stats, reencode_videos, info`. **There is no frame-trim operation**, so leading dead time cannot be removed after the fact.

## 3b. Visualize (eyeball the demonstrations)
Once the verifier is green, the numbers have already answered "any jitter or dropped frames". Use video for the thing numbers cannot see: **did the grasp actually succeed?**

```bash
lerobot-dataset-viz --repo-id ${HF_USER}/<stamped-dataset> --episode-index 0
```
Opens rerun with every camera feed plus the state and action traces. Sample one episode per condition, plus the longest episodes — an episode that ran the full `episode_time_s` is either a failure or a lot of trailing dead time. Online alternative: paste the dataset id at https://huggingface.co/spaces/lerobot/visualize_dataset

## 3c. Upload to the Hub
`--dataset.push_to_hub=false` during recording keeps sessions fast and offline; upload once at the end:

```bash
hf upload ${HF_USER}/phi_so101_8bin_v1 \
  ~/.cache/huggingface/lerobot/${HF_USER}/phi_so101_8bin_v1_20260803_150735 \
  --repo-type dataset            # add --private to keep it unlisted
```
The Hub name does **not** have to carry the local timestamp — the stamp exists to keep local sessions distinct, so a clean public name is fine. Confirm what actually landed:
```bash
python -c "from huggingface_hub import HfApi; i=HfApi().dataset_info('${HF_USER}/phi_so101_8bin_v1'); \
print(i.id, 'private=',i.private, len(i.siblings),'files')"
```

### 🚨 After `hf upload` you must create the version tag yourself
Uploading the files is **not enough** — a teammate calling `LeRobotDataset("<user>/<name>")` will get:
```
huggingface_hub.errors.RevisionNotFoundError: Your dataset must be tagged with a codebase version.
```
`LeRobotDataset` does not load from `main`. It reads `codebase_version` out of `meta/info.json` (`v3.0`) and resolves **a git tag of that exact name**. `hf upload` only pushes files to `main`; the tag is created by LeRobot's own `push_to_hub(tag_version=True)`, which you skipped by uploading manually. So add it once:
```bash
python -c "from huggingface_hub import HfApi; \
HfApi().create_tag('${HF_USER}/phi_so101_8bin_v1', tag='v3.0', repo_type='dataset')"
```
Then verify the tag resolves — this downloads `meta/` only, so it is cheap:
```bash
python -c "from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata as M; \
m=M('${HF_USER}/phi_so101_8bin_v1', root='/tmp/_tagcheck'); \
print(m.revision, m.total_episodes, m.total_frames, m.camera_keys)"
```
Read the tag name from your own `info.json` rather than hardcoding `v3.0` — it tracks the LeRobot format, not your dataset.

⚠️ **A tag pins one commit.** Push anything afterwards (a README, a re-encoded video) and loaders still fetch the *old* snapshot, because the tag never moved. Re-point it after any change to repo contents:
```bash
python -c "from huggingface_hub import HfApi; a=HfApi(); r='${HF_USER}/phi_so101_8bin_v1'; \
a.delete_tag(r, tag='v3.0', repo_type='dataset'); a.create_tag(r, tag='v3.0', repo_type='dataset')"
```
This is also why a collaborator who hit the error before the tag existed does **not** need to clear any cache: nothing was downloaded, the exception is raised during revision lookup, before the first file transfer.

## 4. Replay (verify the arm reproduces an episode)

Streams a recorded episode's actions to the follower at the dataset's fps, so you watch the arm redo
the demonstration.

```bash
lerobot-replay \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodem5B7B0096441 \
  --robot.id=phi_follower \
  --dataset.repo_id=BrutalCaesar/phi_so101_redcube_in_box_v1 \
  --dataset.root=$HOME/.cache/huggingface/lerobot/BrutalCaesar/phi_so101_v2_20260805_170306 \
  --dataset.episode=19
```

**Use `$HOME`, not `~`** — tilde does not expand after `=` in bash. Omit `--dataset.root` to pull from
the Hub instead of a local copy.

### 🚨 A wrong `--dataset.root` silently re-downloads the whole dataset

`--root` is not validated. Point it somewhere that has no `meta/info.json` and LeRobot treats it as an
empty cache location and **fetches the entire dataset from the Hub into it** — no error, no warning,
just a second copy on disk. An earlier version of this page omitted the `BrutalCaesar/` path segment
and cost us a duplicate 181 MB download (2026-08-06).

The recorded path is `<cache>/<org>/<repo_name>_<timestamp>` — **the org directory is part of it**, and
the timestamp is appended at record time so the directory name is never just the repo name. List what
actually exists rather than typing a path from memory:

```bash
python -m phi.utils.camera_realign --list
```

**Do not pass `--robot.cameras`.** Replay sends joint actions only; it never reads a camera. This is
the one hardware script where the [camera-index problem](#when-do-camera-indices-change) cannot bite
you.

### Why replay is worth running: it is a free calibration check

The actions in a dataset are joint targets recorded against **the calibration you had at record
time**. So:

- Replay reproduces the task → calibration still matches the recording. Any policy trained on this
  data is being evaluated on the same joint frame it learned.
- Replay lands consistently off (a centimetre short, gripper closing beside the object) → **calibration
  has drifted since recording**. That same drift will sabotage a trained policy while looking exactly
  like a policy failure.

Run it after anything that could have moved the arm or changed calibration, and before you spend a
training run. It costs one episode's worth of wall clock.

### 🚨 Two things about how replay behaves

**It jumps to the start pose with no ramp.** The loop is `robot.connect()` and then immediately
`send_action` for frame 0 — there is no interpolation from wherever the arm is parked. If the arm is
far from the episode's first pose, that command is a fast, large motion. **Move the follower roughly
to the start pose by hand first**, and keep a hand near the power.

**Replay is open-loop and does not see the object.** It replays joint targets, nothing else. If the
cube is not where it sat at the start of that episode, the gripper closes on empty table and carries
nothing. That is not a bug — it is what replay is. Place the object at the recorded start position if
you want the motion to look like the task.

### Pick the episode deliberately

Prefer an episode that ended on the **right arrow** over one that hit the `episode_time_s` ceiling. A
capped episode may have been cut mid-motion, so its replay stops mid-task and tells you nothing about
calibration. Episode lengths are in `meta/episodes/`; anything sitting exactly at `fps ×
episode_time_s` is a timer cut.

## The LeRobotDataset format (v3.0) — what you're creating
`lerobot >= 0.4.0`. Many episodes are packed per file (not one-file-per-episode):
- `data/` — frame-by-frame **Parquet** shards (states, actions, timestamps)
- `videos/` — **MP4** shards, one set per camera
- `meta/` — `info.json` (schema, fps, codebase version), `stats.json` (normalization stats), `tasks.jsonl` (task↔id), `episodes/` (per-episode lengths/offsets)

If you build a dataset in Python (`LeRobotDataset.create(...)` + `save_episode()`), you **must call `dataset.finalize()` before `push_to_hub()`** or the files are corrupt. Editing: `lerobot-edit-dataset --operation.type=delete_episodes|split|merge|info ...`.

**Every dataset gets a card** in [`datasets/`](../../datasets/) — id, task, #episodes, cameras, who collected it, known issues. That's how the next student finds and trusts it.

✅ **You are now L1→L2.** Next: [train a policy](../../training/README.md).

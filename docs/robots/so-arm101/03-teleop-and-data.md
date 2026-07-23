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
```bash
lerobot-record \
  --robot.type=so101_follower --robot.port=/dev/tty.usbmodem585A0076841 --robot.id=phi_follower \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30} }" \
  --teleop.type=so101_leader  --teleop.port=/dev/tty.usbmodem575E0031751 --teleop.id=phi_leader \
  --display_data=true \
  --dataset.repo_id=${HF_USER}/phi-cube-v1 \
  --dataset.single_task="Grab the black cube" \
  --dataset.num_episodes=50
```
Key `--dataset.*` args: `repo_id` (Hub id) · `single_task="..."` (the language label) · `num_episodes` (default 50; when resuming, this is the number of *additional* episodes) · `episode_time_s` (default 60) · `reset_time_s` (default 60) · `push_to_hub=False` to skip auto-upload · `--resume=true --dataset.root=<path>` to append.

**Keyboard while recording:** `→` / `n` = save & next · `←` / `r` = re-record · `Esc` / `q` = stop, encode, upload.
Local cache: `~/.cache/huggingface/lerobot/{repo-id}`. Manual push: `hf upload ${HF_USER}/phi-cube-v1 ~/.cache/huggingface/lerobot/phi-cube-v1 --repo-type dataset`.

### Data-collection rules of thumb (from LeRobot)
- **≥50 episodes** for a first ACT task; **~10 episodes per distinct location/arrangement** so the policy generalizes.
- Vary object position, lighting, and start pose — a policy only learns what it *sees* vary.
- Keep the camera **fixed** relative to the workspace between recording and evaluation.
- Bad/aborted episodes: re-record (`r`) rather than keep noise.

## 3. Visualize (QA your data)
Online: paste the dataset id at https://huggingface.co/spaces/lerobot/visualize_dataset
Local (opens rerun.io):
```bash
lerobot-dataset-viz --repo-id ${HF_USER}/phi-cube-v1 --episode-index 0
```
Check: cameras framed right, no dropped frames, gripper actions look clean, task label correct.

## 4. Replay (verify the arm reproduces an episode)
```bash
lerobot-replay \
  --robot.type=so101_follower --robot.port=/dev/tty.usbmodem58760431541 --robot.id=phi_follower \
  --dataset.repo_id=${HF_USER}/phi-cube-v1 --dataset.episode=0
```

## The LeRobotDataset format (v3.0) — what you're creating
`lerobot >= 0.4.0`. Many episodes are packed per file (not one-file-per-episode):
- `data/` — frame-by-frame **Parquet** shards (states, actions, timestamps)
- `videos/` — **MP4** shards, one set per camera
- `meta/` — `info.json` (schema, fps, codebase version), `stats.json` (normalization stats), `tasks.jsonl` (task↔id), `episodes/` (per-episode lengths/offsets)

If you build a dataset in Python (`LeRobotDataset.create(...)` + `save_episode()`), you **must call `dataset.finalize()` before `push_to_hub()`** or the files are corrupt. Editing: `lerobot-edit-dataset --operation.type=delete_episodes|split|merge|info ...`.

**Every dataset gets a card** in [`datasets/`](../../datasets/) — id, task, #episodes, cameras, who collected it, known issues. That's how the next student finds and trusts it.

✅ **You are now L1→L2.** Next: [train a policy](../../training/README.md).

# SO-ARM101 — Troubleshooting

Our growing corpus of "things that bit us." Add yours (PR or `good-first-issue`).

## USB / ports
- **Port not found / changes between plugs:** re-run `lerobot-find-port`; the mac name (`/dev/tty.usbmodem…`) can change per port/reboot.
- **Permission denied (Linux):** `sudo chmod 666 /dev/ttyACM0`.
- **CH340-based board on macOS:** may need a USB-serial driver; most boards enumerate natively as `usbmodem`.

## Motors
- **A motor won't take its id:** connect only that one when prompted; check the Waveshare jumpers are on channel **B (USB)**; verify the power supply.
- **Motor hot / stalled:** cut power — don't leave a stalled STS3215 energized. May need Feetech firmware update (LeRobot docs: `feetech`).

## Calibration / teleop
- **Follower doesn't mirror well:** re-calibrate; make sure you reused the same `--robot.id` / `--teleop.id` you calibrated with.
- **Arm jerks violently near full stretch:** you hit a **kinematic singularity** — the controller commands huge joint moves for a small tip motion. Avoid teleoping into full extension; keep tasks in the comfortable middle of the workspace.
- **Follower's `wrist_roll` snaps ~90–180° during teleop (the full-turn wrap).** Diagnosed on our arm 2026-07-29.
  - **Why:** `wrist_roll` is a **full-rotation joint**. In `so_follower.py` / `so_leader.py` it is deliberately **excluded** from range recording (hence the prompt *"Move all joints except 'wrist_roll'…"* — moving it is simply ignored, **not** an error) and its range is hardcoded to the whole encoder, `[0, 4095]` → normalized `[-100, +100]`. Raw `0` and `4095` are the **same physical place**, so crossing that **seam** flips the normalized value full-scale and the follower slams around.
  - **The seam is always exactly 180° from your calibration pose**, because `set_half_turn_homings()` makes whatever position you're in at the *"move to the middle of its range"* prompt become `2048`. You therefore get **±180° of safe travel and no more** — this is inherent, not a bug to calibrate away.
  - **Fix:** at that prompt, put `wrist_roll` at the **neutral centre of the range you'll actually use** (gripper level/untwisted) so the ±180° is symmetric, then stay inside it while teleoperating. Calibrating with the wrist already twisted spends your headroom before you start.
  - **Not the cause:** differing `homing_offset` values between leader and follower are **not** comparable across arms (each servo's mechanical zero differs) — don't chase that. Verify real alignment by reading both arms' `Present_Position` at the same pose; ours agreed within **4 counts (~0°)**.
  - **Diagnose live** (see §6 of [02-setup.md](02-setup.md) for the bus snippet): print `raw` + `norm` for both arms and watch `min(raw, 4095-raw)` shrink as you approach the seam.
  - ⚠️ **Stale calibration files:** a leader calibrated with `--robot.type=so101_follower` lands in `robots/so_follower/<id>.json` instead of `teleoperators/so_leader/<id>.json`. Harmless if unused, but delete it — a mistyped `--*.id` will silently load the wrong file. Calibrations live in `~/.cache/huggingface/lerobot/calibration/`.

## macOS (the cockpit)
- **`--policy.device=mps`** for any local policy run; the Mac does **not** train well (fanless throttling) — offload training.
- `ffmpeg` must be present (our env installs it) for TorchCodec video decode.

## CUDA (the training box)
- **CUDA OOM:** lower `--batch_size`; for VLAs use LoRA/PEFT (LeRobot `peft_training` doc).
- **Wrong PyTorch/CUDA pairing:** install a CUDA-matched PyTorch *before* `lerobot[training]`.

## Dataset
- **Corrupt dataset after Python creation:** you forgot `dataset.finalize()` before `push_to_hub()`.
- **Old dataset (v2.1):** migrate with `python -m lerobot.scripts.convert_dataset_v21_to_v30 --repo-id=<id>`.

## Python env (conda + pyenv coexistence) — the second-laptop gotcha
Symptom: with `(phi)` in the prompt, `python --version` shows an **old** version (e.g. 3.11.6) and `pip install "lerobot[...]==0.6.0"` fails with *"No matching distribution … requires-python >=3.12."*

Two distinct causes (both bit us setting up a second Mac, 2026-07-29):
- **Env built with the wrong Python.** A bare `conda create -n phi` inherits an old default. **Always** create from the pinned file: `conda env create -f env/environment.mac.yml` (pins `python=3.12`) — never `conda create -n phi` by hand.
- **Stale shell after a delete/recreate (the sneaky one).** If you `conda activate phi`, then delete + recreate the env *in that same terminal*, the session's PATH/hash goes stale and bare `python`/`pip` fall through to the machine's **pyenv shim** (our cockpit's pyenv global is 3.11.6). Fix: **open a fresh terminal** (or `exec zsh`, or `hash -r`) after any env delete/recreate, then `conda activate phi`.

Notes:
- The `lerobot-*` CLIs are **not** pyenv-shimmed (their shebang points straight at the env's python), so they keep working on 3.12 even when a stale bare `python` looks wrong. When in doubt, `which lerobot-find-port` should be `…/miniforge3/envs/phi/bin/…`.
- If bare `python`/`pip` ever misbehave inside the env, use `conda run -n phi pip …` (bypasses PATH/shims).
- Hardening applied 2026-07-29: disabled `eval "$(pyenv virtualenv-init -)"` in `~/.zshrc` (unused; a known pyenv→conda shadow footgun). Verify a clean setup with: `conda activate phi && which python` → must be `…/miniforge3/envs/phi/bin/python` (3.12.13).

## Version drift to watch
- On-robot inference is `lerobot-rollout` now; **v0.5.1 used `lerobot-record`** for inference.
- `pi0` docs page currently served from the `main` branch; extras are `[pi]` (not `[pi0]`).

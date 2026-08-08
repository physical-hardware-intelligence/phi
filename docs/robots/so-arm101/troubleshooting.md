# SO-ARM101 — Troubleshooting

Our growing corpus of "things that bit us." Add yours (PR or `good-first-issue`).

## USB / ports
- **Port not found / changes between plugs:** re-run `lerobot-find-port`; the mac name (`/dev/tty.usbmodem…`) can change per port/reboot.
- **Permission denied (Linux):** `sudo chmod 666 /dev/ttyACM0`.
- **CH340-based board on macOS:** may need a USB-serial driver; most boards enumerate natively as `usbmodem`.

## Motors

### 🚨 Missing motors, or `Input voltage error!` → check you didn't swap the two power adapters
**The leader and follower ship with DIFFERENT adapters — one 5 V, one 12 V — and they look alike.** Unplug both and it is very easy to put them back on the wrong arms. This cost us an afternoon on 2026-08-03 and briefly looked like a dead servo. **Label them.**

Each mistake has its own signature, so the symptom tells you which arm got which:

| What you see | What it means |
|---|---|
| `Missing motor IDs: 2` **and** the scan logs `{2: '[RxPacketError] Input voltage error!'}` | **Over-voltage.** This arm is on the 12 V adapter but wants 5 V. The servo answered — it reported a fault, then latched and stopped responding to normal pings, so LeRobot calls it "missing". |
| `scan_port` returns **`{}`** — nothing at all, at any baud rate | **Under-voltage.** This arm is on the 5 V adapter but wants 12 V. The motors never come up, so nothing enumerates. |

**A motor that reports an error is not a dead motor.** Dead servos do not reply. Read the scan's stderr before concluding anything — the `[RxPacketError]` line is the whole diagnosis and it is easy to miss above the progress bar.

Fix: put the right adapter on each arm, **power-cycle** (a latched voltage error only clears on power cycle), then re-scan.

```bash
source configs/ports.local.sh
python -c "
from lerobot.motors.feetech import FeetechMotorsBus
import os; r = FeetechMotorsBus.scan_port(os.environ['LEADER_PORT'])
for b, ids in r.items(): print('baud', b, 'found', sorted(ids), 'MISSING', sorted(set([1,2,3,4,5,6]) - set(ids)) or 'none')
" 2>&1 | grep -v "it/s\]"
```

Reference healthy readings on the **12 V** arm: **11.8–12.0 V on all six**, spread ≤0.2 V, ~32–33 °C idle, load 0. A large spread would mean a genuine voltage drop down the daisy chain; a uniform reading with one motor erroring does not.

⚠️ Note the headroom problem behind this: the servos ship with `Max_Voltage_Limit = 120` (**12.0 V**) while the 12 V supply reads 11.8–12.0 V. There is **no margin by design**, which is why the wrong adapter trips protection instantly rather than merely running warm. Per-motor voltage/temperature/load script: [02-setup §2b](02-setup.md#2b-verify-the-motors-first-do-this-on-the-assembled-arm-too).

### Other motor issues
- **A motor won't take its id:** connect only that one when prompted; check the Waveshare jumpers are on channel **B (USB)**; verify the power supply.
- **Motor hot / stalled:** cut power — don't leave a stalled STS3215 energized. May need Feetech firmware update (LeRobot docs: `feetech`).
- **A genuinely dead servo** looks different: no reply at all to `scan_port` for that id, often with a **blinking red LED** while the others are steady. That is what got our first leader arm returned (also id 2 — do not confuse the two cases).

## Calibration / teleop
- **Follower doesn't mirror well:** re-calibrate; make sure you reused the same `--robot.id` / `--teleop.id` you calibrated with.
- **Arm jerks violently near full stretch:** you hit a **kinematic singularity** — the controller commands huge joint moves for a small tip motion. Avoid teleoping into full extension; keep tasks in the comfortable middle of the workspace.
- **Follower's `wrist_roll` snaps ~90–180° during teleop (the full-turn wrap).** Diagnosed on our arm 2026-07-29.
  - **Why:** `wrist_roll` is a **full-rotation joint**. In `so_follower.py` / `so_leader.py` it is deliberately **excluded** from range recording (hence the prompt *"Move all joints except 'wrist_roll'…"* — moving it is simply ignored, **not** an error) and its range is hardcoded to the whole encoder, `[0, 4095]` → **`[-180°, +180°]`** (LeRobot 0.6.0 defaults `use_degrees=True`, so this is `MotorNormMode.DEGREES`, *not* the `[-100, +100]` of `RANGE_M100_100`). Raw `0` and `4095` are the **same physical place**, so crossing that **seam** flips the value full-scale and the follower slams around.
  - **The seam is always exactly 180° from your calibration pose**, because `set_half_turn_homings()` makes whatever position you're in at the *"move to the middle of its range"* prompt become `2047` (`pos − int(4095/2)`). You therefore get **±180° of safe travel and no more** — this is inherent, not a bug to calibrate away.
  - **The same hardcoded range also makes this the one joint whose calibration pose never cancels out**, which is why it is the joint that breaks cross-machine policy transfer. See [02-setup §3b](02-setup.md#3b--the-hole-wrist_roll-has-no-stops-so-its-calibration-pose-is-permanent).
  - **Fix:** at that prompt, put `wrist_roll` at the **neutral centre of the range you'll actually use** (gripper level/untwisted) so the ±180° is symmetric, then stay inside it while teleoperating. Calibrating with the wrist already twisted spends your headroom before you start.
  - **Not the cause:** differing `homing_offset` values between leader and follower are **not** comparable across arms (each servo's mechanical zero differs) — don't chase that. Verify real alignment by reading both arms' `Present_Position` at the same pose; ours agreed within **4 counts (~0°)**. (Two calibrations of the *same* arm **can** be compared, but via `range_min + homing_offset` — the absolute stop position — not via `homing_offset` alone. `compare_calibration.py` does exactly that, and uses it as a same-arm test.)
  - **Diagnose live** (see §6 of [02-setup.md](02-setup.md) for the bus snippet): print `raw` + `norm` for both arms and watch `min(raw, 4095-raw)` shrink as you approach the seam.
  - ⚠️ **Stale calibration files:** a leader calibrated with `--robot.type=so101_follower` lands in `robots/so_follower/<id>.json` instead of `teleoperators/so_leader/<id>.json`. Harmless if unused, but delete it — a mistyped `--*.id` will silently load the wrong file. Calibrations live in `~/.cache/huggingface/lerobot/calibration/`.
- **A trained policy works on one laptop and misses on another.** Nothing errors, the motion looks smooth, it just reaches a centimetre or two off — or the jaws come in at the wrong angle. **Suspect calibration before you suspect the model.** A policy emits joint angles in the frame of the calibration it was *trained* with; a different calibration silently reinterprets them.
  ```bash
  python -m phi.utils.compare_calibration <trained-with-id> <their-id-or-path>
  ```
  Pure stdlib, ~50 ms, no robot or env needed. Reports per-joint disagreement in degrees and mm, checks whether both files even describe the same physical arm, and names the fix. Background and the two rules that prevent it: [02-setup §3a–3c](02-setup.md#3a-what-is-actually-in-the-file-and-why-it-mostly-doesnt-matter).
  - **`wrist_roll` is where this bites** — its range is hardcoded, so unlike every other joint the calibration pose never cancels out. Two of us calibrating the same arm landed **60.4° apart** on it while agreeing to <0.7° everywhere else (2026-08-07).
  - **Same physical arm? Copy the file rather than recalibrating.** That drives the error to exactly zero; a fresh calibration just re-rolls the dice.
  - ⚠️ This failure is **invisible to eyeballing a few rollouts** — it costs success *rate*, not smoothness. Score it with `eval_rollouts.py`.

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

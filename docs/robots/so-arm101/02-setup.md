# SO-ARM101 — Setup & Bring-up

End-to-end from a fresh machine to a teleoperating arm with cameras. Commands are the **current LeRobot CLI** (verified against the docs). Ports and ids below are **examples — substitute your own**.

> Reference: [installation](https://huggingface.co/docs/lerobot/en/installation) · [SO-101 page](https://huggingface.co/docs/lerobot/en/so101) · [il_robots tutorial](https://huggingface.co/docs/lerobot/en/il_robots)

## 0. Install
Environment: **Python ≥3.12, PyTorch ≥2.10**, `ffmpeg` (video decode). Use our env file:
```bash
conda env create -f env/environment.mac.yml   # Apple-Silicon cockpit  (or ENV=cuda on a GPU box)
conda activate phi
```
Or install LeRobot directly (what the env file does under the hood):
```bash
# from source (editable) — recommended so you can read the code
git clone https://github.com/huggingface/lerobot.git && cd lerobot
pip install -e ".[feetech,core_scripts]"    # Feetech motors (SO-101) + record/replay/calibrate
# or from PyPI:  pip install "lerobot[feetech,core_scripts]"
```
Extras you'll add later: `training` (train), `smolvla`, `pi` (pi0/pi0.5), `diffusion`.

## 1. Find the USB ports
```bash
lerobot-find-port
```
Unplug the arm's USB when prompted, press Enter, and it prints the port (mac: `/dev/tty.usbmodem…`, Linux: `/dev/ttyACM0`). Run once per arm (leader + follower). On Linux you may need `sudo chmod 666 /dev/ttyACM0`.

> **Tip — save them once (ports are per-laptop).** Drop your two ports into `configs/ports.local.sh` (git-ignored) and `source` it, so every command below can use `$FOLLOWER_PORT` / `$LEADER_PORT` instead of literals:
> ```bash
> source configs/ports.local.sh   # exports LEADER_PORT, FOLLOWER_PORT, ROBOT_PORT, FOLLOWER_ID, LEADER_ID
> ```

## 2. Motor IDs & baud rate — **skip for the assembled Kit Pro**
IDs + baud rate are written **once** into each servo's **EEPROM** (non-volatile). The **assembled SO-ARM101 Kit Pro ships with this already done**, so **skip `lerobot-setup-motors` and go straight to [§3 Calibrate](#3-calibrate).** (The official [LeRobot SO-101 guide](https://huggingface.co/docs/lerobot/en/so101) documents `setup-motors` because it's written for the *DIY assembly* path — "we'll only need to do this once.")

**Only run it** for a DIY kit, repurposed motors, or a **replaced servo** (a fresh servo defaults to id `1` / 1 Mbps). One motor at a time when prompted; Waveshare board jumpers on channel **B (USB)**:
```bash
lerobot-setup-motors --robot.type=so101_follower --robot.port=$FOLLOWER_PORT   # follower
lerobot-setup-motors --teleop.type=so101_leader  --teleop.port=$LEADER_PORT    # leader
```

### 2b. Verify the motors first (do this on the assembled arm too)
No official CLI exists for this, so we use the LeRobot motor API directly (verified on **lerobot 0.6.0**). **Power + connect** the arm first. This is exactly how we caught a dead servo on a returned leader.
Joint → id map: **1** shoulder_pan · **2** shoulder_lift · **3** elbow_flex · **4** wrist_flex · **5** wrist_roll · **6** gripper.

> ⚠️ **Use `$CONDA_PREFIX/bin/python`, not bare `python`.** On a machine with **pyenv** installed (our cockpit Mac), bare `python` can hit a pyenv shim instead of the env's interpreter → `ModuleNotFoundError: No module named 'lerobot'`. `$CONDA_PREFIX` is set by `conda activate phi` and always points at the right interpreter. (A fresh terminal usually resolves `python` correctly too — see [troubleshooting](troubleshooting.md#python-env-conda--pyenv-coexistence--the-second-laptop-gotcha).)

**a) List every servo on the bus** — `scan_port` probes all baud rates → `{baud: [ids]}`; it does **not** error on a missing motor, so run it first:
```bash
"$CONDA_PREFIX/bin/python" -c "from lerobot.motors.feetech import FeetechMotorsBus; print('FOLLOWER', FeetechMotorsBus.scan_port('$FOLLOWER_PORT'))"
"$CONDA_PREFIX/bin/python" -c "from lerobot.motors.feetech import FeetechMotorsBus; print('LEADER  ', FeetechMotorsBus.scan_port('$LEADER_PORT'))"
```
Healthy SO-101 → `{1000000: [1, 2, 3, 4, 5, 6]}`. A gap (e.g. `[1,2,3,4,6]` — no 5) = that joint's servo isn't answering (bad cable, wrong id, or dead motor).

**b) Per-motor voltage / health** — reads `Present_Voltage` (control-table reg 62, units 0.1 V). All six should read within ~1 V of each other; a `NO RESPONSE` or wildly-off value is the faulty servo (that was id 2 / `shoulder_lift` on the arm we returned — its light blinked red):
```bash
"$CONDA_PREFIX/bin/python" - "$FOLLOWER_PORT" <<'PY'
import sys
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus
port = sys.argv[1]
JOINTS = {"shoulder_pan":1,"shoulder_lift":2,"elbow_flex":3,"wrist_flex":4,"wrist_roll":5,"gripper":6}
bus = FeetechMotorsBus(port, motors={n: Motor(i, "sts3215", MotorNormMode.RANGE_M100_100) for n, i in JOINTS.items()})
bus.connect()
for n, i in JOINTS.items():
    try:
        v = bus.read("Present_Voltage", n, normalize=False) / 10
        print(f"id{i} {n:<13} {v:5.1f} V")
    except Exception as e:
        print(f"id{i} {n:<13}  NO RESPONSE ({type(e).__name__})")
bus.disconnect()
PY
```
Swap `"$FOLLOWER_PORT"` → `"$LEADER_PORT"` for the leader. If `connect()` itself errors, a servo is missing — use the `scan_port` check above to find which id.

**Reading the output** (verified on our arm, 2026-07-29): a healthy bus reads **~11.8–11.9 V on all six**, within ~1 V of each other. A single motor reading far off (or `NO RESPONSE`, or a **blinking red** LED while the others are steady) is the faulty servo — that's how we diagnosed the dead `shoulder_lift` (id 2) that got the first leader arm returned.

## 3. Calibrate
Calibration is stored **per-machine**, so each laptop needs its own pass (the arm itself doesn't change). The `id` names the calibration file — **pick a name and reuse it everywhere** (teleop, record, eval). Move all joints to mid-range, press Enter, then sweep each joint through its full range.

> 🚨 **Don't paste the example ports.** Every `/dev/tty.usbmodem…` in these docs is an *example* — yours differ, and macOS names can change per USB port/reboot. Pasting a doc literal gives `SerialException: [Errno 2] could not open port … No such file or directory`. Always `source configs/ports.local.sh` and use the variables below.

```bash
source configs/ports.local.sh
lerobot-calibrate --robot.type=so101_follower --robot.port=$FOLLOWER_PORT --robot.id=$FOLLOWER_ID
lerobot-calibrate --teleop.type=so101_leader  --teleop.port=$LEADER_PORT  --teleop.id=$LEADER_ID
```

## 4. Teleoperate (sanity check)
No camera needed here — this checks the arms only.
```bash
lerobot-teleoperate \
  --robot.type=so101_follower --robot.port=$FOLLOWER_PORT --robot.id=$FOLLOWER_ID \
  --teleop.type=so101_leader  --teleop.port=$LEADER_PORT  --teleop.id=$LEADER_ID
```
Move the leader; the follower should mirror it. (If calibration is missing it auto-runs first.)

## 5. Add cameras
Find them (saves test frames so you can confirm which is which):
```bash
lerobot-find-cameras opencv        # or: lerobot-find-cameras realsense
```
Cameras are passed **inline** as a dict to `--robot.cameras`. Add `--display_data=true` to see live feeds (via `rerun`):
```bash
lerobot-teleoperate \
  --robot.type=so101_follower --robot.port=$FOLLOWER_PORT --robot.id=$FOLLOWER_ID \
  --teleop.type=so101_leader  --teleop.port=$LEADER_PORT  --teleop.id=$LEADER_ID \
  --robot.cameras="{ wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30} }" \
  --display_data=true
```
Replace `index_or_path: 0` with the index `lerobot-find-cameras` reported for **your** camera (indices are per-laptop, like ports). On macOS the terminal needs **Camera** permission (System Settings → Privacy & Security → Camera) or the feed is black.
RealSense uses `serial_number_or_name` instead of `index_or_path` (and `use_depth: true` for depth).

✅ **You are now L1 (Operator).** Next: [record a dataset](03-teleop-and-data.md).

## Gotchas
- Full setup notes and Mac-vs-CUDA issues live in [troubleshooting](troubleshooting.md).
- **No-CLI option:** LeLab does steps 2–4 in a browser (`uv tool install git+https://github.com/huggingface/leLab.git && lelab`), SO-101-only, macOS support unverified.

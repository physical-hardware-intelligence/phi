# SO-ARM101 — Setup & Bring-up

> Installing on the **Explorer HPC** instead of a laptop? Use [hpc/explorer](../../hpc/explorer.md) — the cluster needs a pinned CUDA variant, a job-local conda `$HOME`, and `PYTHONNOUSERSITE=1`.

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

## 0.5 Wire the powered hub *first*
Do this before §1, not after. Ports and camera indices are assigned per physical USB path, so **re-cabling invalidates every port and index you already saved.** Plug the hub into the laptop, then everything else into the hub — two arm boards and all cameras — and only then start finding ports.

Five USB devices (2 arms + 3 cameras) is more than a laptop should power off its own bus; a bus-powered hub browns out mid-recording. Hardware choice and the bandwidth ceiling: [01-hardware](01-hardware.md#if-youre-buying-your-own).

> ⚠️ **Per-port switches re-enumerate the device.** Flipping a port off and on is a physical unplug as far as the OS is concerned, so that camera can come back on a **different OpenCV index**. If a feed goes black or shows the wrong view, re-run `lerobot-find-cameras` (§5a) before debugging anything else.

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

> ⚠️ **`wrist_roll` mid-position differs between the follower and the leader.** The two arms do not share the same neutral rotation angle on that joint, so "the middle of its range of motion" is a **different physical wrist angle on each arm**. Set each arm's `wrist_roll` to *its own* mid-range when prompted; do not copy the leader's wrist angle onto the follower (or vice versa).

```bash
source configs/ports.local.sh
lerobot-calibrate --robot.type=so101_follower --robot.port=$FOLLOWER_PORT --robot.id=$FOLLOWER_ID
lerobot-calibrate --teleop.type=so101_leader  --teleop.port=$LEADER_PORT  --teleop.id=$LEADER_ID
```

### 3a. What is actually in the file, and why it mostly doesn't matter

Calibration files live in `~/.cache/huggingface/lerobot/calibration/robots/so_follower/<id>.json`. Each joint is a Feetech STS3215 with a **12-bit encoder**: one full turn split into **4096 numbered positions ("ticks")**, so **1 tick = 360/4095 = 0.0879°**. Where tick 0 physically points is arbitrary — it depends on how the horn was pressed onto the output shaft at assembly.

Two fields per joint:

| Field | What it is |
|---|---|
| `homing_offset` | Feetech firmware computes `Present_Position = Actual_Position − Homing_Offset`. `set_half_turn_homings()` picks it so the pose you held at the *"move to the middle"* prompt reads **2047**. It records **your pose**, not the arm. |
| `range_min` / `range_max` | the two physical hard stops, recorded **after** homing while you sweep the joint by hand. |

`use_degrees=True` is the default in `SOFollowerConfig`, so joints 1–5 are `MotorNormMode.DEGREES` and the gripper is `RANGE_0_100`:

```
joints 1-5   degrees = (Present_Position − mid) × 360/4095      mid = (range_min + range_max)/2
gripper      percent = (Present_Position − range_min) / (range_max − range_min) × 100
```

**The important property: for a joint with real hard stops, your homing pose cancels out.** Substitute `Present = Actual − homing`. Because `mid` was measured in that same homed frame it carries the same homing term, and the two subtract away:

```
degrees = (Actual − mid_absolute) × 360/4095       mid_absolute = midpoint of the two PHYSICAL stops
```

What's left is *"how far am I from the centre of my mechanical range"* — a fact about plastic and metal. **Two people who both push each joint into its stops get identical degrees**, on any machine. That is why policies transfer between laptops at all, and it is a good design.

### 3b. 🚨 The hole: `wrist_roll` has no stops, so its calibration pose is permanent

`wrist_roll` is a full-rotation joint. `so_follower.py` **excludes it from range recording and hardcodes its range**:

```python
full_turn_motor = "wrist_roll"
range_mins[full_turn_motor], range_maxes[full_turn_motor] = 0, 4095   # typed in, not measured
```

So `mid` is 2047.5 for everybody, there is nothing for the homing term to cancel against, and **the cancellation above does not happen**. Whatever wrist angle you held at the ENTER prompt becomes that joint's permanent zero.

Measured on our arm (2026-08-07): two calibrations of **the same follower** by two people agreed on all four measured joints to within **1–7 ticks (<0.7°)** — and differed on `wrist_roll` by **687 ticks = 60.4°**. Every joint with an anchor agreed; the one joint without an anchor was off by two-thirds of a right angle.

**So, at that prompt: set `wrist_roll` to a repeatable physical landmark** — jaws level and untwisted, parallel to the table edge. (Same pose that keeps your ±180° of teleop travel symmetric — see [troubleshooting](troubleshooting.md#calibration--teleop).) **And push every other joint firmly into both hard stops:** stop short on one side only and `mid` moves half your shortfall. In that same comparison, an 86-tick difference in how far `shoulder_pan` was swept became a **4.2° base-yaw offset ≈ 18 mm** of sideways error at the gripper — most of a 25 mm cube.

### 3c. Moving a trained policy to someone else's machine

> ### ✅ Confirmed on the arm, 2026-08-10 — this is not a theoretical risk
>
> The `phi_so101_cubes_cylinder` dataset was recorded on **Yash's laptop**, so every action in it is expressed in **Yash's calibration frame**, and the policy trained on it inherited that frame.
>
> **Sai then ran that same policy on his own machine, against his own calibration.** Nothing crashed, nothing warned, the rollout looked superficially normal — but **the gripper was visibly wrong**: it went to the wrong place and mis-grasped.
>
> **Copying Yash's calibration file onto Sai's machine fixed it.** Same checkpoint, same arm, same task — the only thing that changed was the reference frame, and the policy started working properly.
>
> Two things this pins down:
> 1. **The failure is silent.** There is no error, no warning, no log line. It presents as "the model isn't very good," which is the single most expensive way for a bug to disguise itself — you go tune hyperparameters against a problem that was a JSON file.
> 2. **The fix is a file copy, not a retrain.** Because it is a frame mismatch and not a modelling failure.
>
> **This is why [`configs/calibration/robots/so_follower/phi_follower.json`](../../../configs/calibration/robots/so_follower/phi_follower.json) exists** and why it is committed to the repo rather than left in each person's cache. It is the one frame every recorded dataset and every trained checkpoint in `models/` shares. **Before running any Φ policy on any machine, make sure that file is the calibration in use.**

A policy outputs degrees in **the frame of the calibration it was trained with**. Run it against a different calibration and nothing errors — the arm just reaches to the wrong place (see the box above, where exactly that happened). Diff the two files first:

```bash
python -m phi.utils.compare_calibration phi_follower <their-id-or-path>
```

Pure stdlib, ~50 ms, no robot and no env needed. It reports per-joint disagreement in degrees and millimetres, and it starts by checking whether the two files even describe **the same physical arm** — comparing `range_min + homing_offset`, the absolute stop position, which is a hardware property. Same arm → stops agree within a few ticks. Different arms → hundreds of ticks, because the horns sit on different spline teeth, and no offset is meaningful.

- **Same arm** → don't recalibrate, **copy the trained-with file** to the other machine. Every offset goes to exactly zero. Recalibrating only re-rolls the `wrist_roll` dice. The canonical one — the calibration **every model in `models/` was trained under** — is committed at [`configs/calibration/robots/so_follower/phi_follower.json`](../../../configs/calibration/robots/so_follower/phi_follower.json), so this is a `git pull` plus:
  ```bash
  cp configs/calibration/robots/so_follower/phi_follower.json \
     ~/.cache/huggingface/lerobot/calibration/robots/so_follower/
  ```
  Re-commit it **only** if the arm is rebuilt or a servo is replaced, and say so in the commit message — it is the reference frame for every dataset we have recorded.
- **Different arm** → recalibrate there, following §3b, then `lerobot-replay` an episode before trusting a rollout ([why replay is a free calibration check](03-teleop-and-data.md#why-replay-is-worth-running-it-is-a-free-calibration-check)).

## 4. Teleoperate (sanity check)
No camera needed here — this checks the arms only.
```bash
lerobot-teleoperate \
  --robot.type=so101_follower --robot.port=$FOLLOWER_PORT --robot.id=$FOLLOWER_ID \
  --teleop.type=so101_leader  --teleop.port=$LEADER_PORT  --teleop.id=$LEADER_ID
```
Move the leader; the follower should mirror it. (If calibration is missing it auto-runs first.)

## 5. Add cameras

### 5a. Find the camera index (authoritative for LeRobot)
`lerobot-find-cameras` **does not stream** — it records ~2 s and writes test frames so you can tell which camera is which:
```bash
lerobot-find-cameras opencv        # or: lerobot-find-cameras realsense
# options: --output-dir <path>  --record-time-s <sec>
```
Use the index it reports in `--robot.cameras` below. Indices are **per-laptop**, like ports.

### 5b. Quick live preview (no arms needed)
Handy for aiming/focusing the camera before you wire anything up. macOS:
```bash
ffmpeg -f avfoundation -list_devices true -i ""      # list cameras by index
ffplay -f avfoundation -framerate 30 -video_size 640x480 -i "1"   # stream index 1
```
> ⚠️ **avfoundation indices are NOT the same as OpenCV indices.** Use `ffplay` only to *look* at the picture; always take the number you put in `--robot.cameras` from `lerobot-find-cameras opencv`.
> ⚠️ **First run will hang or go black until macOS Camera permission is granted.** Run it from a normal interactive terminal and accept the prompt, or pre-approve under **System Settings → Privacy & Security → Camera**. A non-interactive shell will just hang.
Cameras are passed **inline** as a dict to `--robot.cameras`. Add `--display_data=true` to see live feeds (via `rerun`):
```bash
lerobot-teleoperate \
  --robot.type=so101_follower --robot.port=$FOLLOWER_PORT --robot.id=$FOLLOWER_ID \
  --teleop.type=so101_leader  --teleop.port=$LEADER_PORT  --teleop.id=$LEADER_ID \
  --robot.cameras="{ wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: MJPG} }" \
  --display_data=true
```
Replace `index_or_path: 0` with the index `lerobot-find-cameras` reported for **your** camera (indices are per-laptop, like ports). On macOS the terminal needs **Camera** permission (System Settings → Privacy & Security → Camera) or the feed is black.
RealSense uses `serial_number_or_name` instead of `index_or_path` (and `use_depth: true` for depth).

Our three-camera setup ([which cameras and why](01-hardware.md#what-this-lab-actually-runs)) — substitute your own indices:
```bash
--robot.cameras="{ \
  wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: MJPG}, \
  front: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30, fourcc: MJPG}, \
  top:   {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: MJPG} }"
```
`wrist` = the gripper module · `front` = Brio 101 on the desk stand · `top` = C960 overhead on the boom.

⚠️ **These three names are fixed. Do not improvise.** They become the observation keys in the dataset (`observation.images.top`, …), so a policy trained on `top` will not find a camera someone later calls `overview`, and renaming after collection orphans every episode you already have.

### 5c. Bandwidth: set `fourcc: MJPG`
Not optional once you have more than one camera. Every camera and both arm boards are **USB 2.0** devices sharing one **480 Mbps** bus, of which UVC isochronous transfer can use roughly **80%, so ~384 Mbps**.

Uncompressed **YUY2** is 16 bits per pixel, so `width × height × 16 × fps`:

| Stream | Uncompressed cost | Verdict |
|---|---|---|
| 640×480 @ 30 | ~147 Mbps each | 2 cameras fit, **3 do not** (441 Mbps) |
| 1280×720 @ 30 | ~442 Mbps each | **exceeds the whole bus on its own** |

MJPG compresses roughly 10:1 in-camera, which makes all of this a non-issue. **Symptoms of getting it wrong:** a camera that opens fine alone but fails when the others are on, dropped frames, or `lerobot-record` dying partway through a session. Diagnose bandwidth *before* suspecting the cameras.

Related: [why the 3.0 hub doesn't help](01-hardware.md#if-youre-buying-your-own), and [why 640×480 is enough for every policy](01-hardware.md#how-much-resolution-do-we-actually-need).

✅ **You are now L1 (Operator).** Next: [record a dataset](03-teleop-and-data.md).

## Gotchas
- Full setup notes and Mac-vs-CUDA issues live in [troubleshooting](troubleshooting.md).
- **No-CLI option:** LeLab does steps 2–4 in a browser (`uv tool install git+https://github.com/huggingface/leLab.git && lelab`), SO-101-only, macOS support unverified.

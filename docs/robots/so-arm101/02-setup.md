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

## 2. Set motor IDs
Do this **one motor at a time** when prompted (it writes id+baudrate to each servo's EEPROM). Waveshare board: set both jumpers to channel **B (USB)**.
```bash
# Follower (uses --robot.*)
lerobot-setup-motors --robot.type=so101_follower --robot.port=/dev/tty.usbmodem585A0076841
# Leader (uses --teleop.*)
lerobot-setup-motors --teleop.type=so101_leader  --teleop.port=/dev/tty.usbmodem575E0031751
```

## 3. Calibrate
The `id` names the calibration file — **pick a name and reuse it everywhere** (teleop, record, eval). Move all joints to mid-range, press Enter, then sweep each joint through its full range.
```bash
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/tty.usbmodem58760431551 --robot.id=phi_follower
lerobot-calibrate --teleop.type=so101_leader  --teleop.port=/dev/tty.usbmodem575E0031751 --teleop.id=phi_leader
```

## 4. Teleoperate (sanity check)
```bash
lerobot-teleoperate \
  --robot.type=so101_follower --robot.port=/dev/tty.usbmodem58760431541 --robot.id=phi_follower \
  --teleop.type=so101_leader  --teleop.port=/dev/tty.usbmodem575E0031751 --teleop.id=phi_leader
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
  --robot.type=so101_follower --robot.port=/dev/tty.usbmodem58760431541 --robot.id=phi_follower \
  --teleop.type=so101_leader  --teleop.port=/dev/tty.usbmodem575E0031751 --teleop.id=phi_leader \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30} }" \
  --display_data=true
```
RealSense uses `serial_number_or_name` instead of `index_or_path` (and `use_depth: true` for depth).

✅ **You are now L1 (Operator).** Next: [record a dataset](03-teleop-and-data.md).

## Gotchas
- Full setup notes and Mac-vs-CUDA issues live in [troubleshooting](troubleshooting.md).
- **No-CLI option:** LeLab does steps 2–4 in a browser (`uv tool install git+https://github.com/huggingface/leLab.git && lelab`), SO-101-only, macOS support unverified.

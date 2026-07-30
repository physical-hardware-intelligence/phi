# SO-ARM101 — Hardware & Build

The SO-ARM101 is a low-cost 6-DoF **leader–follower** arm pair. You teleoperate by moving the small **leader**; the **follower** mirrors it. Both are the same design; the leader has lighter gearing so it's easy to move by hand.

Canonical hardware source: **https://github.com/TheRobotStudio/SO-ARM100** (the repo covers both SO-100 and the newer SO-101).

## Build vs Buy
- **Buy assembled (what we did):** SO-ARM101 Kit Pro (Assembled) from **Seeed Studio** — arrives pre-built and calibrated-ready. This is the fastest path and what the club's first arm is. Purchased 2026-07-22, ~$372.90.
- **Buy kit + self-assemble:** cheaper, ~2–3 h of assembly.
- **3D-print + source motors:** most control, most time. PLA+, 0.4 mm nozzle @ 0.2 mm layer, 15% infill. Whole-arm print plates exist for Bambu/Prusa/Ender beds (`Ender_Follower_SO101.stl`, `Prusa_*_SO101.stl`, or individual parts under `STL/SO101/Individual`).

## What's inside (two-arm setup, per the SO-ARM100 repo)
- **Motors:** Feetech **STS3215** servos. Follower = 6× at 1/345 gearing; leader mixes ratios (base/shoulder-pan 1/191, shoulder-lift 1/345, elbow 1/191, wrist-flex/roll & gripper 1/147). Voltage variants: 7.4 V (~16.5 kg·cm) and 12 V (~30 kg·cm, needs a 12 V ≥5 A supply).
- **2× motor control board**, USB-C cables, **2× power supply**, table clamps, screwdriver set.
- Repo BOM estimate: two-arm ≈ **$230**, single follower ≈ **$122** (region/time-dependent — verify before quoting).

## Cameras (needed for vision policies)

At least one camera; two is better. A **wrist camera** (on the gripper) + a **front/overview camera** is the standard combo — the wrist view nails gripper/contact outcomes, the overview nails object/scene outcomes. Any UVC webcam works via OpenCV; Intel RealSense (D405/D435) is supported but **not required** (our policies are RGB-only). Wiring and config: [02-setup](02-setup.md).

**Club status (2026-07-30):** 1× wrist camera (32×32 mm USB module, bundled). A second (overview) camera is the highest-leverage outstanding purchase.

### How much resolution do we actually need?

Counter-intuitive but verified against the installed `lerobot` 0.6.0 policy configs — **every policy downsizes hard**, so camera megapixels are almost irrelevant:

| Policy | Vision frontend | Resolution it consumes |
|---|---|---|
| **ACT** | ResNet18 | native (backbone handles it) |
| **Diffusion Policy** | ResNet18 | `resize_shape` (default `None` = native) + crop |
| **VQ-BeT** | ResNet18 | **84×84** crop |
| **pi0 / pi0.5** | PaliGemma VLM | **224×224** (`DEFAULT_IMAGE_SIZE`) |
| **GR00T** | VLM | **256×256** |
| **SmolVLA** | SigLIP VLM | **512×512** (padded) ← largest in the zoo |
| **Patch Policy** _(not yet in LeRobot)_ | frozen DINOv2 ViT | 224, or 518 for the large mode |

> **⇒ Nothing consumes more than 512×512.** A **720p** camera (720 px short side) already clears SmolVLA's 512 *and* DINOv2's 518. **1080p buys no policy-learning benefit.**

**Record at 640×480 @ 30 fps.** Higher costs USB bandwidth (see below) and disk for no modelling gain.

### What actually matters when choosing

Ranked by real impact:

1. **Field of view** — the genuine differentiator for an overview cam. A ~55° lens covers ~53 cm at 60 cm distance; ~78° covers ~84 cm. Pick for your workspace size and how close you must mount (Room 1012 is small).
2. **Fixed focus > autofocus.** AF hunts when the gripper moves close, changing the image mid-episode. Fixed-focus cameras avoid this; on AF cameras, **disable it**.
3. **Manual exposure + white balance lock.** Auto-exposure drift between sessions is a documented cause of policy failure — the model overfits to lighting. See the macOS note below.
4. **A different model from your other camera.** ⚠️ Two *identical* webcams cause USB path/enumeration collisions that crash `lerobot-record` mid-session. Always mix models.
5. **UVC compliance** — plug-and-play, no drivers.
6. Rolling vs global shutter: rolling is fine for slow tabletop manipulation. Global shutter (e.g. Arducam machine-vision modules) only matters for fast/dynamic motion.

### Recommended options

| Option | ~Cost | Verdict |
|---|---|---|
| **Logitech C270** | $20–25 | **Best value for our tasks.** 720p30, ~55° FOV, **fixed focus** (a plus), UVC. Clears every policy above |
| **Logitech C920 / C920x** | $50–70 | Wider **78° FOV** + the most-documented camera in the LeRobot/ROS ecosystem. Choose if the workspace grows or you must mount close. AF must be disabled |
| **Arducam UVC module** (onboard ISP) | [price unverified] | Technically best: **native manual exposure/WB/gain in hardware**, global-shutter options. Board-level, so you solve mounting yourself |

Also budget a **mount** (mini tripod / gooseneck clamp, ~$10–15) and a **powered USB hub** (~$20–30) — 2 arms + 2 cameras = 4 USB devices, and a bus-powered hub can brown out mid-recording.

### ⚠️ macOS: locking exposure needs a third-party tool
macOS does **not** expose UVC camera controls system-wide, and Logitech's own software (Logi Tune / Capture) offers **no exposure control**. On Linux this is native (`v4l2-ctl`). On our Mac cockpit, use either:
- **[`uvcc`](https://github.com/joelpurra/uvcc)** — free, open-source CLI UVC configurator (Node). **Preferred**: scriptable, so the exact exposure/WB values get committed to this repo and re-applied before every session.
- **[Webcam Settings](https://apps.apple.com/us/app/webcam-settings/id533696630)** (~$8, Mac App Store) — GUI; supports C920/C270; can re-write settings at intervals to hold them locked.

### Placement
For an overview camera, **front-high angled down** beats front-and-side: less occlusion during the grasp, and a top-down component reveals **gripper position relative to object centre**. Tape/mark the mount so it is identical between data collection and evaluation — a camera that moves between recording and eval invalidates the policy.

## ⚠️ Safety (read before powering on)
- **Clear the workspace** of hands and fragile objects before teleop or a policy rollout — a trained policy can move unexpectedly.
- **Keep an e-stop reachable**: be ready to cut power / kill the process (`Esc`/`q`). Know where the power switch is.
- **Watch for singularities** — near a fully-stretched or aligned pose the arm can jerk hard as the controller commands large joint moves. Avoid teleoping into full extension; see [troubleshooting](troubleshooting.md).
- **Torque + thermals**: the STS3215 has enough torque to pinch. Don't leave a stalled motor energized (it heats up).
- **Secure the base**: clamp both arms to the table so they can't tip.

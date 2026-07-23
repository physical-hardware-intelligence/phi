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
At least one camera; two is better. A **wrist camera** (on the gripper) + a **front/workspace camera** is the standard combo — the wrist view nails gripper/contact outcomes, the front view nails object/scene outcomes. Any UVC webcam works via OpenCV; Intel RealSense (D405/D435) is supported too. Camera setup is in [02-setup](02-setup.md).

## ⚠️ Safety (read before powering on)
- **Clear the workspace** of hands and fragile objects before teleop or a policy rollout — a trained policy can move unexpectedly.
- **Keep an e-stop reachable**: be ready to cut power / kill the process (`Esc`/`q`). Know where the power switch is.
- **Watch for singularities** — near a fully-stretched or aligned pose the arm can jerk hard as the controller commands large joint moves. Avoid teleoping into full extension; see [troubleshooting](troubleshooting.md).
- **Torque + thermals**: the STS3215 has enough torque to pinch. Don't leave a stalled motor energized (it heats up).
- **Secure the base**: clamp both arms to the table so they can't tip.

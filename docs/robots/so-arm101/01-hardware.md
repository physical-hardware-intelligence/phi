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

> 🚨 **The two power adapters are NOT interchangeable — one is 5 V, one is 12 V, and they look alike.** Swapping them does not merely underperform: the 5 V arm on 12 V trips over-voltage protection (a servo latches `Input voltage error!` and reads as a *missing* motor), and the 12 V arm on 5 V never enumerates at all. **Label each adapter with its arm the first time you unplug them.** Symptom-to-cause table: [troubleshooting](troubleshooting.md#-missing-motors-or-input-voltage-error--check-you-didnt-swap-the-two-power-adapters).
- Repo BOM estimate: two-arm ≈ **$230**, single follower ≈ **$122** (region/time-dependent — verify before quoting).

## Cameras (needed for vision policies)

At least one camera; two is better; we run three. A **wrist** camera on the gripper nails contact and grasp outcomes, while a camera looking at the scene nails where the objects are — we use two of those, one **front** and one **top**. Any UVC webcam works via OpenCV; Intel RealSense (D405/D435) is supported but **not required** (our policies are RGB-only). Wiring and config: [02-setup](02-setup.md).

### What this lab actually runs

Selected 2026-08-02. Three cameras, **three different models on purpose** — identical webcams collide on USB enumeration and crash `lerobot-record` mid-session (see [What actually matters when choosing](#what-actually-matters-when-choosing), item 4) — plus a powered hub.

| Name in configs | Device | The spec that mattered | Mount |
|---|---|---|---|
| **`wrist`** | 32×32 mm USB module (bundled with the Kit Pro) | small enough to sit on the gripper | arm's own bracket |
| **`front`** | **Logitech Brio 101** | **58° dFOV**, fixed focus, 1080p30, USB-A | **AceTaken magnetic stand** (Logitech-foot specific), on the table facing the workspace |
| **`top`** | **EMEET C960** | **90° dFOV**, fixed focus, **1/4″-20 thread**, 1080p30, USB-A | 3-section articulating boom arm, clamped to the table, camera **overhead** |
| **Power** | **Leinsis 7-port powered USB 3.0 hub** | **12 V / 2 A (24 W)**, per-port switch + LED | — |

⚠️ **Use exactly these names — `wrist`, `front`, `top`.** They become the observation keys in every recorded dataset (`observation.images.top`, …), and a policy trained on `top` will not find a camera someone later calls `overview`. Renaming after data collection means the old episodes no longer match.

**Why two different mounts, not two of the same.** This is the detail that wastes an afternoon if you get it wrong: **the C960 has a 1/4″-20 tripod thread and the Brio 101 does not.** The Brio uses Logitech's own mounting foot, which is why the AceTaken magnetic stand is Logitech-specific. So the threaded camera goes on the boom arm and the Logitech goes on the magnetic stand. Check the fit of *your* camera against *your* mount before ordering — "webcam stand" is not a standard.

**Why the C960 goes on top and not the front** — field of view, the one spec that actually differs between them. The overhead camera has to cover the whole bin layout at once, so it gets the wider lens. Horizontal coverage at distance `d` is `W = 2·d·tan(hFOV/2)`, and for a 16:9 sensor `tan(hFOV/2) ≈ 0.872 · tan(dFOV/2)`:

| Camera | dFOV | ≈ hFOV | Visible width at 60 cm |
|---|---|---|---|
| EMEET C960 | 90° | ~82° | **~105 cm** |
| Logitech Brio 101 | 58° | ~52° | **~58 cm** |

The C960 sees **~1.8× wider from the same distance**. With a multi-bin workspace that is the difference between one camera covering the task and having to push the mount so far back the objects get small. Run the formula for your own table before you buy.

> ⚠️ **Two checks still open (do them on arrival, then update this table):** that the boom arm's screw is really 1/4″-20 and takes the C960's weight at full extension, and that the AceTaken's magnet holds the Brio 101 without creep — a mount that sags between recording and evaluation silently invalidates the policy (see [Placement](#placement)).

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

1. **Field of view** — the genuine differentiator for an overview cam, and the reason we run a 90° camera above the table. Use `W = 2·d·tan(hFOV/2)`; numbers for our two cameras are in the table above. Pick for your workspace size and how close you can physically mount.
2. **Fixed focus > autofocus.** AF hunts when the gripper moves close, changing the image mid-episode. Fixed-focus cameras avoid this; on AF cameras, **disable it**.
3. **Manual exposure + white balance lock.** Auto-exposure drift between sessions is a documented cause of policy failure — the model overfits to lighting. See the macOS note below.
4. **A different model from your other camera.** ⚠️ Two *identical* webcams cause USB path/enumeration collisions that crash `lerobot-record` mid-session. Always mix models.
5. **UVC compliance** — plug-and-play, no drivers.
6. Rolling vs global shutter: rolling is fine for slow tabletop manipulation. Global shutter (e.g. Arducam machine-vision modules) only matters for fast/dynamic motion.

### If you're buying your own

Any UVC webcam works. Rank candidates by **FOV → fixed focus → a model you don't already own → mount compatibility**, and ignore megapixels. A ~$25 720p camera clears every policy in the table above.

- **Wide + threaded** (like our C960) for the overview. Wide is the whole point; a 1/4″-20 thread means any tripod or arm fits.
- **Avoid autofocus** if you can. Both of ours are fixed focus, which is one less thing to fight. On an AF camera, disable it.
- **Board-level modules** (e.g. Arducam with an onboard ISP) are technically the best option — **manual exposure/WB/gain in hardware**, global-shutter variants — but you solve mounting yourself.

**Budget for a powered hub.** Ours is 12 V / 2 A. Two arms plus three cameras is **five USB devices**; a bus-powered hub can brown out mid-recording and take a session with it. Per-port switches are worth it: you can power-cycle one camera without crawling behind the table.

> ⚠️ **A USB 3.0 hub does not make USB 2.0 cameras faster.** Every camera here, and both arm control boards, are USB 2.0 devices. They share one **480 Mbps** bus no matter what the hub is rated for. The hub buys you **power and ports, not bandwidth** — for bandwidth you set `fourcc: MJPG`, see [02-setup §5c](02-setup.md#5c-bandwidth-set-fourcc-mjpg).

### ⚠️ Exposure: LeRobot cannot set it, and neither can macOS

Worth knowing before you go looking for the setting. **`OpenCVCameraConfig` in LeRobot 0.6.0 has no exposure, gain, white-balance or focus field.** The complete set is `index_or_path · fps · width · height · color_mode · rotation · warmup_s · fourcc · backend`. There is no knob to turn, so exposure has to be locked *outside* LeRobot, before you record.

And macOS does **not** expose UVC controls system-wide — Logitech's own software (Logi Tune / Capture) offers **no exposure control** either. On Linux this is native (`v4l2-ctl`). On our Mac cockpit, use either:
- **[`uvcc`](https://github.com/joelpurra/uvcc)** — free, open-source CLI UVC configurator (Node). **Preferred**: scriptable, so the exact exposure/WB values get committed to this repo and re-applied before every session.
- **[Webcam Settings](https://apps.apple.com/us/app/webcam-settings/id533696630)** (~$8, Mac App Store) — GUI; can re-write settings at intervals to hold them locked.

**The free mitigation, and do this regardless: fix the lighting.** If the room's light doesn't change, auto-exposure has nothing to drift toward. Record and evaluate under the same lamp, at the same time of day, blinds shut. That costs nothing and removes most of the risk.

### Placement
A **top-down component is what you are really buying** in a scene camera: it reveals gripper position relative to object centre, which a purely head-on view cannot. That is why our wide camera goes overhead (`top`) and the narrower one sits head-on (`front`) — between them you get object layout and approach depth, with less occlusion during the grasp than a side view gives.

**Tape or mark both mounts** so they are identical between data collection and evaluation. A camera that moves between recording and eval invalidates the policy, and the magnetic stand is the one to watch for creep.

## ⚠️ Safety (read before powering on)
- **Clear the workspace** of hands and fragile objects before teleop or a policy rollout — a trained policy can move unexpectedly.
- **Keep an e-stop reachable**: be ready to cut power / kill the process (`Esc`/`q`). Know where the power switch is.
- **Watch for singularities** — near a fully-stretched or aligned pose the arm can jerk hard as the controller commands large joint moves. Avoid teleoping into full extension; see [troubleshooting](troubleshooting.md).
- **Torque + thermals**: the STS3215 has enough torque to pinch. Don't leave a stalled motor energized (it heats up).
- **Secure the base**: clamp both arms to the table so they can't tip.

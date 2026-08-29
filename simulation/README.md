# Simulation — MuJoCo + SO-101

Classical robotics (FK → IK → Jacobians → trajectories) on our own arm, in sim, **on a Mac**.
MuJoCo runs natively on macOS; ManiSkill3 does not (Linux + NVIDIA only).

Curriculum: [Georgia Tech ECE 4560](https://maegantucker.com/ECE4560/) (Prof. Maegan Tucker), SO-101 track.

## Setup

```bash
./simulation/setup.sh                 # creates ~/venvs/so101-sim, installs mujoco
source ~/venvs/so101-sim/bin/activate
python simulation/verify_model.py     # headless: prints joints, ranges, PD gains
cd simulation && mjpython -m mujoco.viewer --mjcf=model/scene.xml
```

**Not the `phi` conda env.** `mujoco` is a plain pip wheel (`absl-py, etils, glfw, numpy, pyopengl` — no torch, no CUDA), and `phi` is the only environment that drives the real arm. Keep them apart. The *code* lives here in the repo; only the *environment* lives outside it.

🚨 **macOS: use `mjpython`, not `python`, for anything that opens a viewer** — it needs the main thread.

🚨 **`mjpython` + a uv venv needs one symlink.** mjpython dlopens the interpreter via `@executable_path/../lib/libpython3.12.dylib`, but a venv's `lib/` holds only site-packages. `setup.sh` links the real dylib into place. Without it you get
`Library not loaded: @executable_path/../lib/libpython3.12.dylib`.

## Model

`model/` is vendored from [TheRobotStudio/SO-ARM100 · Simulation/SO101](https://github.com/TheRobotStudio/SO-ARM100/tree/main/Simulation/SO101) — `scene.xml`, `so101_new_calib.xml`, `joints_properties.xml`, and 13 STL meshes (~16 MB). Committed so a new member is one clone away from a running sim.

Upstream also ships `so101_old_calib`. **Use `new`.**

## Verified facts (2026-08-29)

`nq=6 nu=6 njnt=6 nbody=8`, timestep `0.002` s (500 Hz).

Joint ranges match our own URDF derivation exactly — see `docs/robots/so-arm101/`.

| joint | range | rotation |
|---|---|---|
| `shoulder_pan` | −110° … +110° | yaw |
| `shoulder_lift` | −100° … +100° | pitch |
| `elbow_flex` | −96.8° … +96.8° | pitch |
| `wrist_flex` | −95° … +95° | pitch |
| `wrist_roll` | −157.2° … +162.8° | roll |
| `gripper` | −10° … +100° | the jaw |

**The arm is 5-DOF, not 6** — the gripper is the jaw. The three pitch joints are exactly parallel, so the structure is `[base yaw] → [planar 3R pitch chain] → [tool roll]`. There is **no wrist yaw**: the approach axis is locked to the vertical plane chosen by `shoulder_pan`. Top-down grasps are the exception — the axis is vertical, azimuth is undefined, and `wrist_roll` gives a free 360°.

**All six actuators share one gain pair: `kp = 998.22`, `kv = 2.731`.** That is a default, not tuned — `shoulder_lift` carries far more load than `wrist_roll` and cannot sensibly share gains with it.

⚠️ On the real arm LeRobot hardcodes `P_Coefficient = 16`. **998.22 and 16 are not comparable** — MuJoCo's kp is torque per radian of error, Feetech's is a register value in the servo's own units. Compare *behaviour* (same trajectory, tracking error), never the ratio.

## Layout

```
simulation/
├── setup.sh            # env + the mjpython dylib fix
├── verify_model.py     # headless: load, list joints/actuators, step once
├── model/              # vendored SO-101 MJCF + meshes
└── README.md
```

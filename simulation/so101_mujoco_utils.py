"""Helpers for driving the SO-101 in MuJoCo (ECE 4560, Assignment 4).

Model facts (measured from model/so101_new_calib.xml, 2026-08-29):
  * <compiler angle="radian">  -> qpos and ctrl are RADIANS
  * position actuators, kp=998.22 kv=2.731, forcerange +/-3.35 N.m
  * backlash joints modelled at +/-0.5 deg
  * timestep 0.002 s (500 Hz)
  * no keyframe in the XML, so START_POSE below is our own choice
"""

from __future__ import annotations

import time

import mujoco
import numpy as np

# Order matters: this is the qpos / ctrl order in the model.
JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# Joint limits in degrees, from the model.
LIMITS_DEG = {
    "shoulder_pan": (-110.0, 110.0),
    "shoulder_lift": (-100.0, 100.0),
    "elbow_flex": (-96.8, 96.8),
    "wrist_flex": (-95.0, 95.0),
    "wrist_roll": (-157.2, 162.8),
    "gripper": (-10.0, 100.0),
}

# The arm joints report degrees; the gripper reports 0..100 ("percent open").
ARM_JOINTS = JOINTS[:-1]
GRIPPER = "gripper"

# The model ships no keyframe. Pick a visible, in-range start pose.
# Swap these for the pose in the assignment screenshot if you want to match it.
START_POSE = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -45.0,
    "elbow_flex": 60.0,
    "wrist_flex": 45.0,
    "wrist_roll": 0.0,
    "gripper": 20.0,  # 0..100
}

ZERO_POSE = {j: 0.0 for j in ARM_JOINTS} | {GRIPPER: 0.0}


# --------------------------------------------------------------------------
# Unit conversion: the simulator speaks radians, we speak degrees (and the
# gripper is nicer as 0..100 "percent open" than as an angle).
# --------------------------------------------------------------------------
_GRIP_LO, _GRIP_HI = LIMITS_DEG[GRIPPER]  # -10.0, 100.0 degrees


def _gripper_deg_to_pct(deg: float) -> float:
    """-10..100 degrees  ->  0..100 percent open."""
    return (deg - _GRIP_LO) / (_GRIP_HI - _GRIP_LO) * 100.0


def _gripper_pct_to_deg(pct: float) -> float:
    """0..100 percent open  ->  -10..100 degrees."""
    return _GRIP_LO + (pct / 100.0) * (_GRIP_HI - _GRIP_LO)


def convert_to_dictionary(qpos) -> dict[str, float]:
    """qpos (radians, model order) -> {joint: degrees}, gripper -> 0..100."""
    out = {j: float(np.rad2deg(q)) for j, q in zip(JOINTS, qpos)}
    out[GRIPPER] = _gripper_deg_to_pct(out[GRIPPER])
    return out


def convert_to_list(position: dict[str, float]) -> np.ndarray:
    """Inverse of the above -> np.ndarray of radians in JOINTS order."""
    degrees = dict(position)
    degrees[GRIPPER] = _gripper_pct_to_deg(degrees[GRIPPER])
    return np.deg2rad(np.array([degrees[j] for j in JOINTS], dtype=float))


# --------------------------------------------------------------------------
# Commanding the sim.
# --------------------------------------------------------------------------
def set_initial_pose(d: mujoco.MjData, position: dict[str, float]) -> None:
    """Teleport the robot there, and match ctrl so step 1 does not snap."""
    q = convert_to_list(position)
    d.qpos[: len(JOINTS)] = q
    d.qvel[:] = 0.0
    d.ctrl[:] = q  # <- without this the PD controller yanks toward 0 instantly


def send_position_command(d: mujoco.MjData, position: dict[str, float]) -> None:
    """Set the position-actuator targets. MuJoCo's internal PD does the rest."""
    d.ctrl[:] = convert_to_list(position)


# Redraw every Nth step. The sim runs at 500 Hz; a display runs at 60 Hz, so
# calling viewer.sync() every step is ~8x more redraws than anyone can see, and
# sync() is by far the most expensive thing in the loop (measured 2026-08-30:
# mj_step 0.004 ms, viewer.sync ~2.4 ms).
SYNC_EVERY = 8


def _run(m, d, viewer, n_steps: int, target_fn=None) -> None:
    """Step the sim n_steps, pacing to real time, redrawing occasionally.

    Real-time pacing means sleeping the time we have LEFT in this step, not a
    flat timestep. Sleeping a flat 2 ms on top of work that already took 2.4 ms
    makes the sim run ~2.4x slower than reality.
    """
    dt = m.opt.timestep
    for i in range(n_steps):
        tick = time.perf_counter()
        if target_fn is not None:
            send_position_command(d, target_fn(i))
        mujoco.mj_step(m, d)
        if i % SYNC_EVERY == 0:
            viewer.sync()
        remaining = dt - (time.perf_counter() - tick)
        if remaining > 0:
            time.sleep(remaining)
    viewer.sync()  # make sure the final state is on screen


def move_to_pose(m, d, viewer, desired: dict[str, float], duration: float) -> None:
    """Linearly interpolate from the current pose to `desired` over `duration`.

    We blend in HUMAN units (degrees / percent) and convert once at the end,
    so the gripper's 0..100 scale is interpolated on the same scale you read.
    """
    start = convert_to_dictionary(d.qpos[: len(JOINTS)].copy())
    n = max(1, int(duration / m.opt.timestep))

    def target(i: int) -> dict[str, float]:
        alpha = (i + 1) / n  # 0 -> 1 across the move
        return {j: (1.0 - alpha) * start[j] + alpha * desired[j] for j in JOINTS}

    _run(m, d, viewer, n, target)


def hold_position(m, d, viewer, duration: float) -> None:
    """Keep the current ctrl target and let the sim settle."""
    _run(m, d, viewer, max(1, int(duration / m.opt.timestep)))

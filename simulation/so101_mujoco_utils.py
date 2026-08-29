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
# Unit conversion.  TODO(you): this is the part worth doing by hand.
# --------------------------------------------------------------------------
def convert_to_dictionary(qpos) -> dict[str, float]:
    """qpos (radians, model order) -> {joint: degrees}, gripper -> 0..100."""
    out = {j: float(np.rad2deg(q)) for j, q in zip(JOINTS, qpos)}
    # TODO: remap out[GRIPPER] from its degree range LIMITS_DEG[GRIPPER] to 0..100
    return out


def convert_to_list(position: dict[str, float]) -> np.ndarray:
    """Inverse of the above -> np.ndarray of radians in JOINTS order."""
    # TODO: undo the gripper 0..100 remap, then deg2rad everything
    raise NotImplementedError


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


def move_to_pose(m, d, viewer, desired: dict[str, float], duration: float) -> None:
    """Linearly interpolate from the current pose to `desired` over `duration`."""
    start = convert_to_dictionary(d.qpos[: len(JOINTS)].copy())
    n = max(1, int(duration / m.opt.timestep))
    for i in range(n):
        # TODO: alpha from 0 -> 1 across the loop, blend start and desired per joint
        raise NotImplementedError
        mujoco.mj_step(m, d)
        viewer.sync()
        time.sleep(m.opt.timestep)


def hold_position(m, d, viewer, duration: float) -> None:
    """Keep the current ctrl target and let the sim settle."""
    for _ in range(max(1, int(duration / m.opt.timestep))):
        mujoco.mj_step(m, d)
        viewer.sync()
        time.sleep(m.opt.timestep)

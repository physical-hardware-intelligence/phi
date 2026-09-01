"""ECE 4560 Assignment 6, part 2a -- pick and place in simulation.

    mjpython run_pick_and_place.py

Six joint configurations, no inverse kinematics anywhere. We know the angles
we want; forward kinematics only tells us WHERE those angles put the object,
so we can go put a real cube there on the hardware.

    1  start,  jaw open      pick position, approach
    2  start,  jaw closed    grasp
    3  lifted over pick      clear the table
    4  lifted over place     traverse
    5  final,  jaw closed    descend
    6  final,  jaw open      release
"""

from __future__ import annotations

import pathlib

import mujoco
import mujoco.viewer

from so101_forward_kinematics import get_forward_kinematics
from so101_mujoco_utils import hold_position, move_to_pose, set_initial_pose

MODEL = pathlib.Path(__file__).parent / "model" / "scene.xml"

OPEN, CLOSED = 50, 5  # gripper, percent open

# Given by the assignment.
starting_configuration = {
    "shoulder_pan": -45.0, "shoulder_lift": 45.0, "elbow_flex": -45.0,
    "wrist_flex": 90.0, "wrist_roll": 0.0, "gripper": OPEN,
}
final_configuration = {**starting_configuration, "shoulder_pan": 45.0}

starting_configuration_closed = {**starting_configuration, "gripper": CLOSED}
final_configuration_closed = {**final_configuration, "gripper": CLOSED}

# Ours to choose. shoulder_lift 45 -> 0 raises the object from 17 mm to
# 212 mm while x and y move less than 2 mm, so the cube goes almost straight
# up and comes almost straight back down.
lifted_over_start = {**starting_configuration_closed, "shoulder_lift": 0.0}
lifted_over_final = {**final_configuration_closed, "shoulder_lift": 0.0}

SEQUENCE = [
    ("grasp",            starting_configuration_closed, 1.5),
    ("lift",             lifted_over_start,             2.0),
    ("traverse",         lifted_over_final,             3.0),
    ("descend",          final_configuration_closed,    2.0),
    ("release",          final_configuration,           1.5),
]


def show_cubes(viewer, starting_config, final_config, halfwidth=0.013):
    """Draw where forward kinematics says the cube starts and ends up."""
    for i, (cfg, rgba) in enumerate([(starting_config, (1, 0, 0, 0.2)),
                                     (final_config, (0, 1, 0, 0.2))]):
        position, rotation = get_forward_kinematics(cfg)
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[i],
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[halfwidth, halfwidth, halfwidth],
            pos=position,
            mat=rotation.flatten(),
            rgba=rgba,
        )
    viewer.user_scn.ngeom = 2
    viewer.sync()


def main() -> None:
    m = mujoco.MjModel.from_xml_path(str(MODEL))
    d = mujoco.MjData(m)

    for name, cfg in [("pick ", starting_configuration),
                      ("place", final_configuration)]:
        p, _ = get_forward_kinematics(cfg)
        print(f"{name} the cube at  x={p[0]*1000:6.1f}  y={p[1]*1000:6.1f}  "
              f"z={p[2]*1000:5.1f} mm")

    set_initial_pose(d, starting_configuration)
    mujoco.mj_forward(m, d)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        show_cubes(viewer, starting_configuration, final_configuration)
        hold_position(m, d, viewer, duration=1.0)
        for label, cfg, duration in SEQUENCE:
            print(f"  {label}")
            move_to_pose(m, d, viewer, cfg, duration=duration)
            hold_position(m, d, viewer, duration=0.4)


if __name__ == "__main__":
    main()

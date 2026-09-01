"""ECE 4560 Assignment 6, part 1b -- is the forward kinematics right?

    mjpython run_fk_test.py

Draws a red cylinder at the object frame our FK predicts. If the FK is
correct the cylinder sits in the jaws. If it is wrong the cylinder floats
somewhere else, which is a much better error message than a number.
"""

from __future__ import annotations

import pathlib

import mujoco
import mujoco.viewer

from so101_forward_kinematics import get_forward_kinematics
from so101_mujoco_utils import hold_position, send_position_command, set_initial_pose

MODEL = pathlib.Path(__file__).parent / "model" / "scene.xml"

# Degrees, despite what the assignment's comment says. so101_mujoco_utils
# does the deg->rad conversion; the gripper is 0..100 percent open.
test_configuration = {
    "shoulder_pan": -45.0,
    "shoulder_lift": 45.0,
    "elbow_flex": -45.0,
    "wrist_flex": 90.0,
    "wrist_roll": 0.0,
    "gripper": 10,
}


def show_cylinder(viewer, position, rotation, radius=0.0245, halfheight=0.05,
                  rgba=(1, 0, 0, 1)):
    """Draw a cylinder whose long axis is the object frame's z-axis."""
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[0],
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[radius, halfheight, 0],
        pos=position,
        mat=rotation.flatten(),
        rgba=rgba,
    )
    viewer.user_scn.ngeom = 1
    viewer.sync()


def main() -> None:
    m = mujoco.MjModel.from_xml_path(str(MODEL))
    d = mujoco.MjData(m)

    set_initial_pose(d, test_configuration)
    send_position_command(d, test_configuration)
    mujoco.mj_forward(m, d)

    position, rotation = get_forward_kinematics(test_configuration)
    print(f"object frame at x={position[0]*1000:.1f} y={position[1]*1000:.1f} "
          f"z={position[2]*1000:.1f} mm")

    with mujoco.viewer.launch_passive(m, d) as viewer:
        show_cylinder(viewer, position, rotation)
        hold_position(m, d, viewer, duration=20.0)


if __name__ == "__main__":
    main()

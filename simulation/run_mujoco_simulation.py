"""ECE 4560 Assignment 4 -- the three-move sequence.

    mjpython run_mujoco_simulation.py

Must be mjpython, not python: launch_passive puts the UI on the main thread
while this script keeps its own loop. See simulation/README.md.
"""

from __future__ import annotations

import pathlib

import mujoco
import mujoco.viewer

from so101_mujoco_utils import (
    START_POSE,
    ZERO_POSE,
    hold_position,
    move_to_pose,
    set_initial_pose,
)

MODEL = pathlib.Path(__file__).parent / "model" / "scene.xml"


def main() -> None:
    m = mujoco.MjModel.from_xml_path(str(MODEL))
    d = mujoco.MjData(m)

    set_initial_pose(d, START_POSE)
    mujoco.mj_forward(m, d)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        move_to_pose(m, d, viewer, ZERO_POSE, duration=2.0)
        hold_position(m, d, viewer, duration=2.0)
        move_to_pose(m, d, viewer, START_POSE, duration=2.0)


if __name__ == "__main__":
    main()

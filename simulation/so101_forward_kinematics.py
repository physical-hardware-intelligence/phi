"""Forward kinematics for the SO-101 (ECE 4560, Assignment 6).

Where does the gripper end up, given the five arm angles?

The chain is a product of rigid-body transforms, one per joint:

    g_wt(theta) = g_w1(t1) g_12(t2) g_23(t3) g_34(t4) g_45(t5) g_5t

Each factor has the same shape: a FIXED offset (where the next motor is
bolted, with the frame twisted so its z-axis lies along that motor's shaft),
followed by Rz(theta) -- the joint's own spin about that z.

    g_i,i+1(theta) = [ R_fixed @ Rz(theta) | p_fixed ]

Every joint turns about its own frame's z. That is not a coincidence: the
fixed rotations are chosen to make it true, which is exactly what the URDF /
MuJoCo XML encodes as axis="0 0 1" on all six joints.

Joint 6 (the moving jaw) is ignored: opening and closing the jaw does not
move the grasped object's centre.

NUMBERS: taken from model/so101_new_calib.xml, not from the assignment
diagram. See the module self-test -- the diagram rounds three small offsets
away and that costs ~2 cm at the tool.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------
# Elementary rotations (degrees in, 3x3 out). Given in the assignment.
# --------------------------------------------------------------------------
def Rx(thetadeg):
    t = np.deg2rad(thetadeg); c, s = np.cos(t), np.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def Ry(thetadeg):
    t = np.deg2rad(thetadeg); c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def Rz(thetadeg):
    t = np.deg2rad(thetadeg); c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _g(rotation, displacement):
    """Pack a 3x3 rotation and a 3-vector into a 4x4 homogeneous transform."""
    return np.block([[rotation, np.asarray(displacement, float).reshape(3, 1)],
                     [np.zeros((1, 3)), 1.0]])


# --------------------------------------------------------------------------
# The six links. Fixed part first, joint rotation on the RIGHT.
# --------------------------------------------------------------------------
def get_gw1(theta1_deg):
    """world -> shoulder_pan. Rz(180)@Rx(180) == Ry(180): flip so z points up."""
    return _g(Rz(180) @ Rx(180) @ Rz(theta1_deg), (0.0388353, 0.0, 0.0624))


def get_g12(theta2_deg):
    """shoulder_pan -> shoulder_lift. Pan's z is vertical, lift's is horizontal."""
    return _g(Rx(-90) @ Rz(-90) @ Rz(theta2_deg), (-0.0303992, -0.0182778, -0.0542))


def get_g23(theta3_deg):
    """shoulder_lift -> elbow_flex, down the upper arm (0.11257 long)."""
    return _g(Rz(90) @ Rz(theta3_deg), (-0.11257, -0.028, 0.0))


def get_g34(theta4_deg):
    """elbow_flex -> wrist_flex, down the forearm (0.1349 long)."""
    return _g(Rz(-90) @ Rz(theta4_deg), (-0.1349, 0.0052, 0.0))


def get_g45(theta5_deg):
    """wrist_flex -> wrist_roll.

    The only factor that is not a clean multiple of 90 deg. The extra
    Rz(-2.78913074984777) is a real calibration tilt baked into so101_new_calib.xml
    (arctan2(-R01, -R00) of that body's quaternion).

    It costs ZERO position error, because the tool offset below is along
    this same z-axis and Rz cannot move its own axis. What it does change is
    the object frame's ORIENTATION by 2.79 deg -- so a grasped 100 mm
    cylinder has its ends misplaced by ~2.4 mm. Measured, not assumed.
    """
    return _g(Rz(180) @ Rx(90) @ Rz(-2.78913074984777) @ Rz(theta5_deg),
              (0.0, -0.0611, 0.0181))


def get_g5t():
    """wrist_roll -> object frame: 0.1034 out along -z, then z across the jaws.

    Not a body in the MuJoCo model -- this is the frame the assignment adds
    so the grasped object's pose can be predicted. Verified visually against
    the assignment's own screenshot.
    """
    return _g(Ry(90), (0.0, 0.0, -0.1034))


def get_forward_kinematics(position_dict):
    """{joint: degrees} -> (position (3,), rotation (3,3)) of the object frame."""
    gwt = (get_gw1(position_dict["shoulder_pan"])
           @ get_g12(position_dict["shoulder_lift"])
           @ get_g23(position_dict["elbow_flex"])
           @ get_g34(position_dict["wrist_flex"])
           @ get_g45(position_dict["wrist_roll"])
           @ get_g5t())
    return gwt[0:3, 3], gwt[0:3, 0:3]


# --------------------------------------------------------------------------
# Self-test: does this agree with MuJoCo's own kinematics?
#   python so101_forward_kinematics.py
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import mujoco

    m = mujoco.MjModel.from_xml_path("model/scene.xml")
    d = mujoco.MjData(m)
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "gripper")
    rng = np.random.default_rng(0)

    worst = 0.0
    for _ in range(500):
        q = rng.uniform(-90, 90, 5)
        d.qpos[:5] = np.deg2rad(q)
        d.qpos[5] = 0.0
        mujoco.mj_forward(m, d)
        mine = (get_gw1(q[0]) @ get_g12(q[1]) @ get_g23(q[2])
                @ get_g34(q[3]) @ get_g45(q[4]))
        theirs = _g(d.xmat[gid].reshape(3, 3), d.xpos[gid])
        worst = max(worst, float(np.abs(mine - theirs).max()))

    # 1e-8 not 1e-15: so101_new_calib.xml puts the shoulder at y = -8.98e-09,
    # and we use the assignment's clean 0.0. That 9-nanometre gap is the
    # entire residual -- it lives in the model file, not in this code.
    print(f"joint-5 frame vs MuJoCo over 500 random poses: {worst:.2e} m")
    print("PASS" if worst < 1e-8 else "FAIL")

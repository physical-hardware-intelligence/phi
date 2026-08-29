"""Load the SO-101 MuJoCo model and print its kinematic + actuator facts.

Run:  ~/venvs/so101-sim/bin/python simulation/verify_model.py
"""
import pathlib
import mujoco
import numpy as np

MODEL = pathlib.Path(__file__).parent / "model" / "scene.xml"


def main() -> None:
    m = mujoco.MjModel.from_xml_path(str(MODEL))
    d = mujoco.MjData(m)
    print(f"nq={m.nq} nu={m.nu} njnt={m.njnt} nbody={m.nbody} timestep={m.opt.timestep}")

    print("\njoint            range (deg)")
    for i in range(m.njnt):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
        lo, hi = np.rad2deg(m.jnt_range[i])
        print(f"  {name:15s} {lo:8.1f} … {hi:7.1f}")

    # MuJoCo position actuator: bias = [0, -kp, -kv]
    print("\nactuator            kp        kv")
    for i in range(m.nu):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        kp = m.actuator_gainprm[i][0]
        kv = -m.actuator_biasprm[i][2]
        print(f"  {name:15s} {kp:9.2f} {kv:9.3f}")

    mujoco.mj_step(m, d)
    print("\nstepped once ok, qpos =", np.round(d.qpos[: m.nq], 4))


if __name__ == "__main__":
    main()

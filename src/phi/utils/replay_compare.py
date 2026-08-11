"""Replay a recorded episode on the arm and WATCH IT, in Rerun, against the recording.

`lerobot-replay` sends the human's recorded joint angles to the arm. It has no
`--display_data`, it shows you nothing, and — the important part — **it never
reads the arm back**, so it cannot tell you whether the arm actually went where
it was told. This does both.

    python -m phi.utils.replay_compare --episode 5 \
        --port /dev/tty.usbmodem5B7B0096441 \
        --cameras wrist=0,top=1,front=2

Rerun opens with four things on one timeline:

    recorded/<cam>     the frames from the dataset      "what it looked like then"
    live/<cam>         the frames from the cameras now  "what it looks like now"
    commanded/<joint>  the human's recorded angle
    achieved/<joint>   what the arm reports back
    error/<joint>      achieved - commanded, signed

WHY THIS IS THE RIGHT DIAGNOSTIC
--------------------------------
No policy and no perception are in the loop. The commanded angles are a fixed
recording. So the test cleanly separates two families of fault:

  A. THE ARM DOES NOT GO WHERE IT IS TOLD. A weak or thermally-throttled servo,
     sag, backlash, or a calibration frame mismatch. Shows up as a persistent
     one-sided `error/<joint>`, and `shoulder_pan` error is the one that reads as
     "the arm leans to one side".
  B. THE POLICY IS TOLD THE WRONG THING. A camera nudged since recording, or an
     index reordered this session. Invisible in `error/*` — you see it by
     flicking between `recorded/<cam>` and `live/<cam>` and spotting the shift.

Run it on a CONTROL episode (5, 50, 95) — one you know succeeded when recorded.
If a replay of a known-good episode does not reproduce it, no policy result from
that rig means anything, and per docs/evaluation any score taken on it is void.

🚨 The arm moves on its own. Keep a hand near the power. `--max-step-deg`
(default 15) rate-limits each joint per tick so a frame-0 jump to the episode's
start pose becomes a creep rather than a slam; the first ~1 s of `error/*` is
that catch-up, not a fault. Pass 0 to disable.

🚨 ONE THING TO EXPECT ON `shoulder_lift`, FOUND 2026-08-11
-----------------------------------------------------------
The recorded ACTIONS go past what the follower can physically reach. Actions are
the LEADER's joint angles, and the leader has its own calibration and its own
range, so teleop can command a pose the follower cannot hit:

    joint          follower reachable      action min/max      overshoot
    shoulder_lift  -104.40 .. +104.40   -107.03 .. +70.64   2.64 deg past the LOW stop
    elbow_flex      -97.01 ..  +97.01    -96.97 .. +97.05   0.04 deg (marginal)

`shoulder_lift` is commanded below its low stop in **290 of 1709 frames (17.0%)**
across control episodes 5, 50 and 95. Two readings, and the replay tells them
apart:

  (a) The follower really is being driven into its hard stop 17% of the time.
      A stalled servo draws stall current, which is the mechanism behind the
      "motor blinking - torque overload" note from 2026-08-10. Signature in
      Rerun: `error/shoulder_lift` SATURATES — a flat line while `commanded`
      keeps going down. It is also baked into the training data, so every policy
      trained on this dataset inherits the behaviour, and it will get worse
      within a session as the servo heats and protection engages.
  (b) The range-of-motion recording under-swept `shoulder_lift`'s low side by
      ~2.6 deg. Then there is no stop collision, but `mid` sits ~1.3 deg off and
      EVERY angle on that joint carries that bias. Signature: a small constant
      offset with no saturation.

Either way it is worth knowing before blaming a policy.

READING THE NUMBERS
-------------------
The summary at the end is per-joint signed mean error and absolute p95, skipping
the catch-up window. A healthy replay is a few tenths of a degree of lag with the
sign flipping about zero. What indicts hardware is a mean that is LARGE and
SIGNED — the arm sitting consistently to one side of where it was told.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
DEFAULT_DS = "BrutalCaesar/phi_so101_cubes_cylinder_recovery_v1"


def parse_cameras(spec: str) -> dict[str, int | str]:
    out: dict[str, int | str] = {}
    for part in spec.split(","):
        if "=" not in part:
            raise argparse.ArgumentTypeError(f"expected name=index, got {part!r}")
        name, _, idx = part.partition("=")
        name, idx = name.strip(), idx.strip()
        out[name] = int(idx) if idx.lstrip("-").isdigit() else idx
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episode", type=int, required=True, help="control episodes are 5, 50, 95")
    p.add_argument("--dataset", default=DEFAULT_DS)
    p.add_argument("--root", default=None, help="local dataset dir, if not the HF cache")
    p.add_argument("--port", required=True)
    p.add_argument("--cameras", type=parse_cameras, default=None,
                   help="name=index pairs. Omit to replay with NO live cameras (joints only)")
    p.add_argument("--robot-id", default="phi_follower")
    p.add_argument("--fps", type=float, default=None, help="default: the dataset's fps")
    p.add_argument("--max-step-deg", type=float, default=15.0, help="per-tick rate limit; 0 disables")
    p.add_argument("--settle-s", type=float, default=1.0,
                   help="seconds excluded from the error summary while the arm reaches the start pose")
    p.add_argument("--no-viewer", action="store_true", help="log to Rerun without spawning the GUI")
    args = p.parse_args()

    import rerun as rr
    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.robots.so_follower import SO101FollowerConfig
    from lerobot.robots import make_robot_from_config
    from lerobot.utils.robot_utils import precise_sleep

    ds = LeRobotDataset(args.dataset, root=args.root, episodes=[args.episode])
    fps = args.fps or ds.fps
    n = ds.num_frames
    recorded_cams = [k.split(".")[-1] for k in ds.meta.features if k.startswith("observation.images.")]
    live_cams = args.cameras or {}

    print(f"  dataset  {args.dataset}  episode {args.episode}  ({n} frames, {n / fps:.1f}s at {fps:.0f} fps)")
    print(f"  recorded cameras : {recorded_cams}")
    print(f"  live cameras     : {list(live_cams) or 'NONE (joints only)'}")
    print(f"  rate limit       : {args.max_step_deg or 'DISABLED'} deg/tick")

    robot_cfg = SO101FollowerConfig(
        port=args.port, id=args.robot_id,
        cameras={name: OpenCVCameraConfig(index_or_path=i, width=640, height=480, fps=30)
                 for name, i in live_cams.items()},
        max_relative_target=args.max_step_deg if args.max_step_deg > 0 else None,
    )
    robot = make_robot_from_config(robot_cfg)

    rr.init("phi_replay_compare", spawn=not args.no_viewer)
    rr.log("readme", rr.TextDocument(
        f"Replay of episode {args.episode} from {args.dataset}.\n"
        "recorded/* = the dataset. live/* = the cameras right now.\n"
        "A persistent SIGNED error/<joint> indicts the arm; a shift between "
        "recorded/ and live/ indicts the cameras.", media_type="text/markdown"))

    robot.connect()
    print("\n  🚨 the arm is about to move. Ctrl-C aborts.\n")
    errs: list[np.ndarray] = []
    prev_cmd: np.ndarray | None = None
    try:
        for i in range(n):
            t0 = time.perf_counter()
            frame = ds[i]
            cmd = frame["action"].numpy()

            obs = robot.get_observation()
            achieved = np.array([obs[f"{j}.pos"] for j in JOINTS], dtype=float)

            rr.set_time("frame", sequence=i)
            for cam in recorded_cams:
                img = (frame[f"observation.images.{cam}"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                rr.log(f"recorded/{cam}", rr.Image(img))
            for cam in live_cams:
                if cam in obs:
                    rr.log(f"live/{cam}", rr.Image(obs[cam]))

            for k, j in enumerate(JOINTS):
                rr.log(f"commanded/{j}", rr.Scalars(float(cmd[k])))
                rr.log(f"achieved/{j}", rr.Scalars(float(achieved[k])))

            # achieved at tick i is the result of the command sent at i-1
            if prev_cmd is not None:
                e = achieved - prev_cmd
                errs.append(e)
                for k, j in enumerate(JOINTS):
                    rr.log(f"error/{j}", rr.Scalars(float(e[k])))

            robot.send_action({f"{j}.pos": float(cmd[k]) for k, j in enumerate(JOINTS)})
            prev_cmd = cmd
            precise_sleep(max(1 / fps - (time.perf_counter() - t0), 0.0))
    except KeyboardInterrupt:
        print("  aborted")
    finally:
        robot.disconnect()

    if not errs:
        sys.exit("  no frames completed — nothing to summarise")
    E = np.array(errs)[int(args.settle_s * fps):]
    if len(E) == 0:
        sys.exit(f"  every frame fell inside the {args.settle_s}s settle window; lower --settle-s")

    print(f"\n  tracking error, achieved - commanded, over {len(E)} frames "
          f"(first {args.settle_s}s excluded as start-pose catch-up)")
    print(f"  {'joint':16s} {'signed mean':>13s} {'abs p95':>9s} {'abs max':>9s}   verdict")
    print("  " + "-" * 68)
    for k, j in enumerate(JOINTS):
        e = E[:, k]
        mean, p95, mx = e.mean(), np.percentile(np.abs(e), 95), np.abs(e).max()
        # A large SIGNED mean is the hardware signature: consistently to one side.
        if abs(mean) > 2.0 and abs(mean) > 0.5 * p95:
            side = "HIGH" if mean > 0 else "LOW"
            v = f"🚨 sits {side} of command"
        elif p95 > 5.0:
            v = "noisy but unbiased"
        else:
            v = "ok"
        print(f"  {j:16s} {mean:+13.2f} {p95:9.2f} {mx:9.2f}   {v}")
    print("\n  A signed mean near zero with the sign flipping = normal servo lag.")
    print("  A large signed mean = the arm sits to one side of where it was told:")
    print("  shoulder_pan is the joint that reads as 'the arm leans right'.\n")
    print("  If every joint is clean, the fault is NOT the arm. Compare recorded/<cam>")
    print("  against live/<cam> in Rerun, then: python -m phi.utils.camera_realign\n")


if __name__ == "__main__":
    main()

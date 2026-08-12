"""Live camera view against a RECORDED episode, with replay of the recording on demand.

`lerobot-replay` sends the human's recorded joint angles to the arm. It has no
`--display_data`, it shows you nothing, and — the important part — **it never
reads the arm back**, so it cannot tell you whether the arm actually went where
it was told. This does both, and it does them in the order the job needs.

    python -m phi.utils.replay_compare --episodes 5,45,90,120,135 \
        --port /dev/tty.usbmodem5B7B0096441 \
        --cameras wrist=1,top=2,front=0

THE LIVE VIEW IS UP BEFORE THE ARM MOVES, AND STAYS UP
------------------------------------------------------
The point of comparing `recorded/<cam>` against `live/<cam>` is to decide whether
the scene is staged right **before** committing to a replay. So this opens in
STAGING mode: cameras stream continuously next to a recorded reference frame you
can scrub, and nothing moves until you ask. Replay as many times as you like,
against the same live feed, without restarting.

Keys (no Enter needed):

    r        replay the recorded actions of this episode
    n / p    next / previous episode in --episodes
    [ / ]    scrub the recorded reference frame back / forward (10 frames)
    { / }    scrub 100 frames
    0        reference frame back to 0
    g        go to the episode's start pose without replaying
    s        print the tracking-error summary of the last replay again
    q        quit

Rerun shows, on a `tick` timeline that spans the whole session:

    recorded/<cam>     the dataset's frames             "what it looked like then"
    live/<cam>         the cameras right now            "what it looks like now"
    commanded/<joint>  the human's recorded angle       (replay only)
    achieved/<joint>   what the arm reports back
    error/<joint>      achieved - commanded, signed     (replay only)

WHAT THIS DIAGNOSTIC SEPARATES — AND WHAT IT DOES NOT
-----------------------------------------------------
No policy and no perception are in the loop; the commanded angles are a fixed
recording. That cleanly separates three faults, and they have DIFFERENT signals:

  A. THE ARM DOES NOT GO WHERE IT IS TOLD. Weak or thermally-throttled servo,
     sag, backlash, calibration-frame mismatch. → persistent one-sided
     `error/<joint>`. `shoulder_pan` is the one that reads as "the arm leans".
  B. THE CAMERAS MOVED since recording, or an index reordered this session.
     → invisible in `error/*`; you see it by flicking `recorded/<cam>` against
     `live/<cam>`, which is why staging mode exists.
  C. THE WORLD MOVED relative to the base — the clamp was re-seated, so recorded
     angles no longer land on the object. → **also invisible in `error/*`**: the
     arm tracks its command perfectly and still closes on empty air. The only
     signal is the VISUAL OUTCOME. Stage to match `recorded/`, replay, and watch
     whether the recorded human trajectory actually picks the object up.

⇒ Do not read the error summary as the answer to C. A clean error table means
"the arm is fine", not "the rig is fine".

Run it on a CONTROL episode (5, 50, 95) — one you know succeeded when recorded.
If a replay of a known-good episode does not reproduce it, no policy result from
that rig means anything, and per docs/evaluation any score taken on it is void.

🚨 The arm moves when you press `r` or `g`. Keep a hand near the power.
`--max-step-deg` (default 15) rate-limits each joint per tick, so the move to the
start pose is a creep rather than a slam. Pass 0 to disable.

Because the start pose is reached BEFORE the timed replay begins (see `g`), the
error summary no longer has catch-up garbage at the front and `--settle-s` can
stay small.

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
Per-joint signed mean error and absolute p95. A healthy replay is a few tenths of
a degree of lag with the sign flipping about zero. What indicts hardware is a mean
that is LARGE and SIGNED — the arm sitting consistently to one side of where it
was told. Repeated replays of the same episode should agree; if the signed mean
grows run over run, that is the servo heating up, which is itself the finding.
"""

from __future__ import annotations

import argparse
import contextlib
import select
import sys
import termios
import time
import tty

import numpy as np

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
DEFAULT_DS = "BrutalCaesar/phi_so101_cubes_cylinder_recovery_v1"
STAGE_HZ = 12.0          # camera poll rate while staging; plenty for eyeballing a scene
REACH_TOL_DEG = 2.0      # "close enough" to the start pose
REACH_TIMEOUT_S = 12.0


def parse_cameras(spec: str) -> dict[str, int | str]:
    out: dict[str, int | str] = {}
    for part in spec.split(","):
        if "=" not in part:
            raise argparse.ArgumentTypeError(f"expected name=index, got {part!r}")
        name, _, idx = part.partition("=")
        name, idx = name.strip(), idx.strip()
        out[name] = int(idx) if idx.lstrip("-").isdigit() else idx
    return out


def parse_episodes(spec: str) -> list[int]:
    return [int(x) for x in spec.replace(" ", "").split(",") if x != ""]


@contextlib.contextmanager
def key_reader():
    """Single-keypress reads without Enter, and without eating Ctrl-C.

    cbreak (not raw) leaves ISIG on, so Ctrl-C still raises KeyboardInterrupt --
    which matters a great deal when the arm is moving. Falls back to a no-op
    poller when stdin is not a tty, so the module stays importable under pytest.
    """
    if not sys.stdin.isatty():
        yield lambda: None
        return
    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)

        def poll() -> str | None:
            if select.select([sys.stdin], [], [], 0)[0]:
                return sys.stdin.read(1)
            return None

        yield poll
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)


class Session:
    """Owns the arm, the Rerun stream and the monotonic tick counter."""

    def __init__(self, robot, rr, fps: float, live_cams: dict) -> None:
        self.robot, self.rr, self.fps, self.live_cams = robot, rr, fps, live_cams
        self.tick = 0

    def advance(self) -> None:
        self.tick += 1
        self.rr.set_time("tick", sequence=self.tick)

    def observe(self) -> tuple[dict, np.ndarray]:
        obs = self.robot.get_observation()
        pos = np.array([obs[f"{j}.pos"] for j in JOINTS], dtype=float)
        return obs, pos

    def log_live(self, obs: dict) -> None:
        for cam in self.live_cams:
            if cam in obs:
                self.rr.log(f"live/{cam}", self.rr.Image(obs[cam]))

    def log_achieved(self, pos: np.ndarray) -> None:
        for k, j in enumerate(JOINTS):
            self.rr.log(f"achieved/{j}", self.rr.Scalars(float(pos[k])))

    def send(self, cmd: np.ndarray) -> None:
        self.robot.send_action({f"{j}.pos": float(cmd[k]) for k, j in enumerate(JOINTS)})


def log_recorded(session: Session, ep: "Episode", idx: int) -> None:
    frame = ep.frame(idx)
    for cam in ep.recorded_cams:
        img = (frame[f"observation.images.{cam}"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        session.rr.log(f"recorded/{cam}", session.rr.Image(img))


class Episode:
    """One episode's frames, loaded lazily and cached across navigation."""

    _cache: dict[tuple[str, int], "Episode"] = {}

    def __init__(self, dataset: str, root: str | None, index: int) -> None:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.index = index
        self.ds = LeRobotDataset(dataset, root=root, episodes=[index])
        self.n = self.ds.num_frames
        self.fps = self.ds.fps
        self.recorded_cams = [k.split(".")[-1] for k in self.ds.meta.features
                              if k.startswith("observation.images.")]

    @classmethod
    def get(cls, dataset: str, root: str | None, index: int) -> "Episode":
        key = (dataset, index)
        if key not in cls._cache:
            print(f"  loading episode {index} ...", flush=True)
            cls._cache[key] = cls(dataset, root, index)
        return cls._cache[key]

    def frame(self, idx: int):
        return self.ds[max(0, min(idx, self.n - 1))]

    def action(self, idx: int) -> np.ndarray:
        return self.frame(idx)["action"].numpy()


def goto_start(session: Session, ep: Episode, ref_idx: int = 0) -> bool:
    """Ramp gently to the episode's pose at ref_idx, before any timed replay.

    Relies on the robot's own max_relative_target to clamp each step, so this is
    a creep rather than a slam even from across the workspace. Doing it here --
    rather than inside the timed loop as the old one-shot version did -- is what
    keeps catch-up transients out of the error summary.
    """
    target = ep.action(ref_idx)
    gap = float("inf")
    deadline = time.perf_counter() + REACH_TIMEOUT_S
    from lerobot.utils.robot_utils import precise_sleep

    while time.perf_counter() < deadline:
        session.advance()
        obs, pos = session.observe()
        session.log_live(obs)
        session.log_achieved(pos)
        gap = float(np.abs(pos - target).max())
        if gap < REACH_TOL_DEG:
            print(f"  at start pose (worst joint {gap:.2f} deg off)")
            return True
        session.send(target)
        precise_sleep(1.0 / session.fps)
    print(f"  ⚠️  did not reach the start pose within {REACH_TIMEOUT_S:.0f}s "
          f"(worst joint {gap:.2f} deg off) — a joint may be against a stop")
    return False


def replay(session: Session, ep: Episode, settle_s: float) -> np.ndarray | None:
    """Send the recorded actions, reading the arm back every tick."""
    from lerobot.utils.robot_utils import precise_sleep

    print(f"\n  🚨 replaying episode {ep.index}: {ep.n} frames, "
          f"{ep.n / session.fps:.1f}s. Ctrl-C aborts.")
    goto_start(session, ep, 0)

    errs: list[np.ndarray] = []
    prev_cmd: np.ndarray | None = None
    try:
        for i in range(ep.n):
            t0 = time.perf_counter()
            cmd = ep.action(i)

            session.advance()
            session.rr.set_time("episode_frame", sequence=i)
            obs, pos = session.observe()

            log_recorded(session, ep, i)
            session.log_live(obs)
            session.log_achieved(pos)
            for k, j in enumerate(JOINTS):
                session.rr.log(f"commanded/{j}", session.rr.Scalars(float(cmd[k])))

            # achieved at tick i is the result of the command sent at i-1
            if prev_cmd is not None:
                e = pos - prev_cmd
                errs.append(e)
                for k, j in enumerate(JOINTS):
                    session.rr.log(f"error/{j}", session.rr.Scalars(float(e[k])))

            session.send(cmd)
            prev_cmd = cmd
            precise_sleep(max(1.0 / session.fps - (time.perf_counter() - t0), 0.0))
    except KeyboardInterrupt:
        print("\n  replay aborted — arm holding position")

    if not errs:
        return None
    E = np.array(errs)[int(settle_s * session.fps):]
    return E if len(E) else None


def summarise(E: np.ndarray, settle_s: float) -> None:
    print(f"\n  tracking error, achieved - commanded, over {len(E)} frames "
          f"(first {settle_s}s excluded)")
    print(f"  {'joint':16s} {'signed mean':>13s} {'abs p95':>9s} {'abs max':>9s}   verdict")
    print("  " + "-" * 68)
    for k, j in enumerate(JOINTS):
        e = E[:, k]
        mean, p95, mx = e.mean(), np.percentile(np.abs(e), 95), np.abs(e).max()
        # A large SIGNED mean is the hardware signature: consistently to one side.
        if abs(mean) > 2.0 and abs(mean) > 0.5 * p95:
            v = f"🚨 sits {'HIGH' if mean > 0 else 'LOW'} of command"
        elif p95 > 5.0:
            v = "noisy but unbiased"
        else:
            v = "ok"
        print(f"  {j:16s} {mean:+13.2f} {p95:9.2f} {mx:9.2f}   {v}")
    print("\n  Signed mean near zero, sign flipping = normal servo lag.")
    print("  Large signed mean = the arm sits to one side of where it was told.")
    print("  🔑 A CLEAN table does NOT clear the rig — it only clears the ARM.")
    print("     Whether the recorded trajectory still lands on the OBJECT is the")
    print("     visual outcome you just watched, not a number here.\n")


BANNER = """
  ─────────────────────────────────────────────────────────────────────
   r  replay recorded actions      n / p  next / prev episode
   g  go to start pose (no replay) [ / ]  scrub reference -/+ 10 frames
   s  re-print last error summary  { / }  scrub reference -/+ 100 frames
   q  quit                         0      reference back to frame 0
  ─────────────────────────────────────────────────────────────────────"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=parse_episodes, default=None,
                   help="comma list to navigate with n/p, e.g. 5,45,90,120,135")
    p.add_argument("--episode", type=int, default=None,
                   help="single episode (kept for older docs; --episodes supersedes it)")
    p.add_argument("--dataset", default=DEFAULT_DS)
    p.add_argument("--root", default=None, help="local dataset dir, if not the HF cache")
    p.add_argument("--port", required=True)
    p.add_argument("--cameras", type=parse_cameras, default=None,
                   help="name=index pairs. Omit for joints only (no live view -- "
                        "which defeats the point of staging mode)")
    p.add_argument("--robot-id", default="phi_follower")
    p.add_argument("--fps", type=float, default=None, help="default: the dataset's fps")
    p.add_argument("--max-step-deg", type=float, default=15.0, help="per-tick rate limit; 0 disables")
    p.add_argument("--settle-s", type=float, default=0.5,
                   help="seconds excluded from the error summary; smaller than the old "
                        "default because the start pose is now reached before timing starts")
    p.add_argument("--no-viewer", action="store_true", help="log to Rerun without spawning the GUI")
    args = p.parse_args()

    episodes = args.episodes or ([args.episode] if args.episode is not None else None)
    if not episodes:
        raise SystemExit("need --episodes 5,45,90 (or --episode 5). "
                         "Controls are 5, 50, 95; recovery episodes are 120-142.")

    import rerun as rr
    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower import SO101FollowerConfig
    from lerobot.robots import make_robot_from_config
    from lerobot.utils.robot_utils import precise_sleep

    live_cams = args.cameras or {}
    first = Episode.get(args.dataset, args.root, episodes[0])
    fps = args.fps or first.fps

    print(f"\n  dataset  {args.dataset}")
    print(f"  episodes {episodes}")
    print(f"  recorded cameras : {first.recorded_cams}")
    print(f"  live cameras     : {list(live_cams) or '⚠️  NONE (joints only)'}")
    print(f"  rate limit       : {args.max_step_deg or 'DISABLED'} deg/tick")

    robot = make_robot_from_config(SO101FollowerConfig(
        port=args.port, id=args.robot_id,
        cameras={name: OpenCVCameraConfig(index_or_path=i, width=640, height=480, fps=30)
                 for name, i in live_cams.items()},
        max_relative_target=args.max_step_deg if args.max_step_deg > 0 else None,
    ))

    rr.init("phi_replay_compare", spawn=not args.no_viewer)
    rr.log("readme", rr.TextDocument(
        "STAGING: live/* streams continuously against a scrubbable recorded/* frame.\n"
        "Press r to replay the recording. Nothing moves until you do.\n\n"
        "A persistent SIGNED error/<joint> indicts the ARM.\n"
        "A shift between recorded/ and live/ indicts the CAMERAS.\n"
        "A clean error table plus a failed grasp indicts the RIG GEOMETRY -- "
        "that one is only visible as the visual outcome.", media_type="text/markdown"))

    robot.connect()
    session = Session(robot, rr, fps, live_cams)
    cursor, ref_idx, last_E = 0, 0, None
    ep = first
    print(BANNER)
    print(f"\n  STAGING episode {ep.index}  (reference frame {ref_idx}/{ep.n - 1}) "
          f"— arm is idle. Match the scene to recorded/*, then press r.")

    try:
        with key_reader() as getkey:
            while True:
                t0 = time.perf_counter()
                session.advance()
                obs, pos = session.observe()
                session.log_live(obs)
                session.log_achieved(pos)
                log_recorded(session, ep, ref_idx)

                k = getkey()
                if k in ("q", "\x04"):
                    break
                elif k == "r":
                    E = replay(session, ep, args.settle_s)
                    if E is None:
                        print("  no frames completed — nothing to summarise")
                    else:
                        last_E = E
                        summarise(E, args.settle_s)
                    print(f"  STAGING episode {ep.index} — r replays again, n next, q quit")
                elif k == "g":
                    goto_start(session, ep, ref_idx)
                elif k == "s":
                    if last_E is None:
                        print("  no replay yet")
                    else:
                        summarise(last_E, args.settle_s)
                elif k in ("n", "p"):
                    cursor = (cursor + (1 if k == "n" else -1)) % len(episodes)
                    ep = Episode.get(args.dataset, args.root, episodes[cursor])
                    ref_idx = 0
                    tag = "RECOVERY-recorded" if ep.index >= 120 else "clean-recorded"
                    print(f"\n  STAGING episode {ep.index}  [{tag}]  "
                          f"({ep.n} frames) — reference frame 0")
                elif k in ("[", "]", "{", "}", "0"):
                    step = {"[": -10, "]": 10, "{": -100, "}": 100}.get(k, 0)
                    ref_idx = 0 if k == "0" else max(0, min(ref_idx + step, ep.n - 1))
                    print(f"  reference frame {ref_idx}/{ep.n - 1}")
                precise_sleep(max(1.0 / STAGE_HZ - (time.perf_counter() - t0), 0.0))
    except KeyboardInterrupt:
        print("\n  interrupted")
    finally:
        robot.disconnect()
        print("  disconnected")


if __name__ == "__main__":
    main()

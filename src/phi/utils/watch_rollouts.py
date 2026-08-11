"""Watch a policy drive the arm, one episode at a time. No scoring, no recording.

This is the "just let me see it" tool. You press Enter, the policy runs for a
fixed number of seconds, the arm glides back to where it started, and you press
Enter again. Nothing is written to disk and nothing is scored.

    python -m phi.utils.watch_rollouts \
        --model BrutalCaesar/act_so101_cubcyl_recovery_chunk50_noaug_3cam \
        --port /dev/tty.usbmodem5A460816421 \
        --cameras wrist=0,top=1,front=2

    # the augmented sibling, and its 80k checkpoint
    --model BrutalCaesar/act_so101_cubcyl_recovery_chunk50_aug_3cam
    --model BrutalCaesar/act_so101_cubcyl_recovery_chunk50_aug_3cam --revision step-80000

When you want NUMBERS instead of a look, use `phi.utils.eval_rollouts`, which
owns the partial-credit rubric and the per-episode staging plan. This module is
deliberately not that: no rubric, no aggregation, no dataset.

WHY IT DELEGATES THE CONTROL LOOP
---------------------------------
The observation -> preprocessor -> select_action -> postprocessor -> send_action
pipeline is NOT reimplemented here. It is easy to get subtly wrong, and a
mis-normalised action makes every policy look broken, which is the most
expensive way to waste an afternoon. We build LeRobot's own `RolloutContext` and
run its `BaseStrategy`, exactly what `lerobot-rollout --strategy.type=base`
does. What this module adds is the two preflight gates below and the
one-episode-at-a-time sequencing that the base CLI has no notion of.

THE TWO GATES, AND WHY THEY ARE HARD STOPS
------------------------------------------
Both of these failure modes are SILENT. No exception, no warning, no log line.
The arm just moves smoothly to the wrong place and you conclude the policy is
bad.

1. CALIBRATION. A policy emits joint angles in the calibration frame of the
   machine that recorded its training data. Confirmed on our arm 2026-08-10: a
   cubes/cylinder policy mis-grasped on a second laptop and was fixed by copying
   the original calibration file across, no retraining. So this script refuses
   to run unless the active calibration matches the repo's canonical
   `phi_follower.json`. See docs/robots/so-arm101/02-setup.md section 3c.

2. CAMERA INDEX. macOS has no udev, so device indices reorder silently between
   sessions. Miss a camera and the policy will not load; SWAP two and it loads
   fine and behaves badly. So this script shows you one live frame per camera,
   labelled with the observation key it will feed, and waits for you to confirm.
   `phi.utils.camera_realign` is the fuller tool if the mapping is wrong.

Neither gate is bypassable by accident. `--force` exists and prints a loud
banner; if you use it, do not report a number from that session.

SAFETY
------
The arm moves autonomously and this policy has never been scored. Keep a hand
near the power. `--max-step-deg` (default 15) rate-limits how far any joint is
commanded per control tick, so a wildly wrong first action becomes a slow creep
instead of a slam. Pass 0 to disable it.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CANONICAL_CALIB = REPO_ROOT / "configs/calibration/robots/so_follower/phi_follower.json"
ACTIVE_CALIB_DIR = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so_follower"

DEFAULT_TASK = "pick up the cubes/cylinder and place it in the box"
BANNER = "=" * 78


def parse_cameras(spec: str) -> dict[str, int | str]:
    """`wrist=0,top=1,front=2` -> {"wrist": 0, "top": 1, "front": 2}.

    Values that are not integers are passed through as device paths.
    """
    out: dict[str, int | str] = {}
    for part in spec.split(","):
        if "=" not in part:
            raise argparse.ArgumentTypeError(f"expected name=index, got {part!r}")
        name, _, idx = part.partition("=")
        name, idx = name.strip(), idx.strip()
        out[name] = int(idx) if idx.lstrip("-").isdigit() else idx
    return out


def check_port(port: str, force: bool) -> None:
    """Fail fast on a bad port instead of blocking on a serial timeout.

    🚨 Do NOT get the port with `--port $(lerobot-find-port)`. That tool is
    INTERACTIVE: it prints "Remove the USB cable ... and press Enter" and then
    blocks on input(). Command substitution swallows the prompt, so the terminal
    goes silent and looks hung forever, and even on success `$(...)` captures all
    four printed lines rather than the port alone. Run `lerobot-find-port` on its
    own, read the port, paste it in.
    """
    if Path(port).exists():
        print(f"  port OK          {port}")
        return
    candidates = sorted(str(p) for p in Path("/dev").glob("tty.usbmodem*"))
    _fail(
        f"no such device: {port}\n"
        + (f"    USB serial devices visible now:\n"
           + "\n".join(f"      {c}" for c in candidates) if candidates
           else "    NO USB serial devices are visible at all — is the arm plugged in and powered?")
        + "\n    To find it:  run `lerobot-find-port` ON ITS OWN (it is interactive and"
          "\n                 will block invisibly inside $(...)), then paste the port here.",
        force,
    )


def check_calibration(robot_id: str, force: bool) -> None:
    """Refuse to run unless the active calibration IS the canonical one."""
    active = ACTIVE_CALIB_DIR / f"{robot_id}.json"
    if not CANONICAL_CALIB.exists():
        _fail(f"canonical calibration missing from the repo: {CANONICAL_CALIB}", force)
        return
    if not active.exists():
        _fail(
            f"no active calibration at {active}\n"
            f"    cp {CANONICAL_CALIB} {ACTIVE_CALIB_DIR}/",
            force,
        )
        return

    want = json.loads(CANONICAL_CALIB.read_text())
    have = json.loads(active.read_text())
    if want == have:
        print(f"  calibration OK   {active.name} == repo canonical")
        return

    diffs = []
    for joint in sorted(set(want) | set(have)):
        w, h = want.get(joint), have.get(joint)
        if w != h:
            diffs.append(f"      {joint}: repo={w} active={h}")
    _fail(
        "active calibration DIFFERS from the repo canonical file.\n"
        + "\n".join(diffs)
        + f"\n    Fix:  cp {CANONICAL_CALIB} {ACTIVE_CALIB_DIR}/\n"
        f"    Diff: python -m phi.utils.compare_calibration phi_follower {robot_id}",
        force,
    )


def _fail(msg: str, force: bool) -> None:
    if force:
        print(f"\n{BANNER}\n  FORCED PAST A GATE — do not report a number from this session\n  {msg}\n{BANNER}\n")
        return
    sys.exit(f"\n  REFUSING TO RUN: {msg}\n")


def confirm_cameras(cameras: dict[str, int | str], force: bool) -> None:
    """Show one frame per camera, labelled with the key it feeds, and ask."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        _fail("opencv/numpy unavailable, cannot verify camera mapping", force)
        return

    tiles = []
    for name, idx in cameras.items():
        cap = cv2.VideoCapture(idx)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            _fail(
                f"camera {name}={idx} gave no frame.\n"
                "    Three causes, in order of likelihood:\n"
                "      1. wrong index          -> python -m phi.utils.camera_realign\n"
                "      2. device busy          -> close Photo Booth / LeLab / another rollout\n"
                "      3. macOS camera permission NOT granted to this terminal\n"
                "         (OpenCV prints 'not authorized to capture video'). Grant it in\n"
                "         System Settings > Privacy & Security > Camera, for the app running\n"
                "         python (Terminal / iTerm / VS Code), then restart the terminal.",
                force,
            )
            return
        frame = cv2.resize(frame, (426, 320))
        cv2.rectangle(frame, (0, 0), (426, 28), (0, 0, 0), -1)
        cv2.putText(
            frame, f"{name}  (index {idx})  -> observation.images.{name}",
            (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
        )
        tiles.append(frame)

    # cv2.waitKey blocks on a keypress IN THE OPENCV WINDOW, not in the terminal,
    # and the window can open behind whatever you are looking at. Say so, or this
    # reads as a second silent hang.
    print("\n  A window is opening with one tile per camera. It may appear BEHIND this"
          "\n  terminal. Click the window, press any key to close it, then come back here.",
          flush=True)
    cv2.imshow("confirm camera mapping - click me, then press any key", np.hstack(tiles))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("\n  Does each tile show the camera named on it?")
    for name in cameras:
        print(f"      {name:6s} -> observation.images.{name}")
    if input("  type 'yes' to continue: ").strip().lower() not in {"yes", "y"}:
        sys.exit("  Stopped. Fix the mapping with: python -m phi.utils.camera_realign\n")


def build_config(args, cameras: dict[str, int | str]):
    import torch
    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.robots.so_follower import SO101FollowerConfig
    from lerobot.rollout import RolloutConfig
    from lerobot.rollout.configs import BaseStrategyConfig

    policy_cfg = PreTrainedConfig.from_pretrained(args.model, revision=args.revision)
    policy_cfg.pretrained_path = args.model

    # Re-plan rate. Trained with n_action_steps=50, i.e. the arm commits to 1.67 s
    # of plan from one observation. Measured open-loop drift against a held-out
    # episode on 2026-08-11 grows monotonically across the chunk:
    #   steps 0-9 after a re-plan  16.1 deg  ->  steps 40-49  37.2 deg
    # Lowering this costs nothing and needs no retraining: the policy still
    # predicts 50 actions, it just discards more of the stale tail. Worth trying
    # 10-15 if the arm looks confident early then wanders.
    if args.n_action_steps:
        if args.n_action_steps > policy_cfg.chunk_size:
            sys.exit(f"  --n-action-steps must be <= chunk_size ({policy_cfg.chunk_size})")
        policy_cfg.n_action_steps = args.n_action_steps

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    robot_cfg = SO101FollowerConfig(
        port=args.port,
        id=args.robot_id,
        cameras={
            name: OpenCVCameraConfig(index_or_path=idx, width=640, height=480, fps=30)
            for name, idx in cameras.items()
        },
        max_relative_target=args.max_step_deg if args.max_step_deg > 0 else None,
    )

    return RolloutConfig(
        robot=robot_cfg,
        policy=policy_cfg,
        strategy=BaseStrategyConfig(),
        dataset=None,          # base strategy REQUIRES this to be None
        fps=args.fps,
        duration=args.duration,
        device=device,
        task=args.task,
        display_data=args.display_data,
        return_to_initial_position=True,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="HF repo id or local checkpoint path")
    p.add_argument("--revision", default=None, help="HF branch/tag, e.g. step-80000")
    p.add_argument("--port", required=True, help="follower serial port (lerobot-find-port)")
    p.add_argument("--cameras", required=True, type=parse_cameras,
                   help="name=index pairs, e.g. wrist=0,top=1,front=2")
    p.add_argument("--robot-id", default="phi_follower")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--duration", type=float, default=30.0, help="seconds per episode")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--task", default=DEFAULT_TASK)
    p.add_argument("--max-step-deg", type=float, default=15.0,
                   help="per-tick joint rate limit; 0 disables")
    p.add_argument("--n-action-steps", type=int, default=None,
                   help="actions executed per re-plan (trained: 50 = 1.67s open loop). "
                        "Lower = re-plans more often, no retraining needed. Try 10-15.")
    p.add_argument("--device", default=None, help="cuda / mps / cpu (auto-detected)")
    p.add_argument("--display-data", action="store_true", help="stream to Rerun")
    p.add_argument("--force", action="store_true", help="bypass the preflight gates (loudly)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    print(f"\n{BANNER}\n  PREFLIGHT\n{BANNER}", flush=True)
    check_port(args.port, args.force)
    check_calibration(args.robot_id, args.force)
    confirm_cameras(args.cameras, args.force)

    from lerobot.rollout import build_rollout_context, create_strategy

    cfg = build_config(args, args.cameras)
    print(f"\n  model      {args.model}{'@' + args.revision if args.revision else ''}")
    print(f"  device     {cfg.device}")
    print(f"  cameras    {', '.join(f'{k}={v}' for k, v in args.cameras.items())}")
    print(f"  rate limit {args.max_step_deg or 'DISABLED'} deg/tick")
    print(f"  episodes   {args.episodes} x {args.duration:.0f}s at {cfg.fps:.0f} fps")

    # Policy is built before hardware is touched, so a bad --model fails without
    # ever energising a motor.
    shutdown_event = threading.Event()
    ctx = build_rollout_context(cfg, shutdown_event)
    strategy = create_strategy(cfg.strategy)
    strategy.setup(ctx)

    print(f"\n{BANNER}\n  THE ARM WILL MOVE ON ITS OWN. Keep a hand near the power.\n{BANNER}")
    try:
        for ep in range(1, args.episodes + 1):
            if input(f"\n  [{ep}/{args.episodes}] stage the scene, then Enter (q to quit): ").strip().lower() == "q":
                break

            # Clear the action queue so episode N does not begin by executing
            # leftover actions predicted from episode N-1's final observation.
            # ACT holds n_action_steps=50, i.e. 1.67 s of stale plan.
            strategy._engine.reset()

            print(f"  running {args.duration:.0f}s ...")
            strategy.run(ctx)

            print("  returning to the start pose ...")
            type(strategy)._return_to_initial_position(ctx.hardware)
    except KeyboardInterrupt:
        print("\n  interrupted")
    finally:
        strategy.teardown(ctx)

    print("\n  Done. Nothing was recorded or scored.")
    print("  For numbers:  python -m phi.utils.eval_rollouts --help\n")


if __name__ == "__main__":
    main()

"""Find out WHICH part of the control loop is too slow.

`lerobot-rollout` and `lerobot-record` both print the same warning:

    Record loop is running slower (3.6 Hz) than the target FPS (30.0 Hz).
    Common causes are: 1) Camera FPS not keeping up 2) Policy inference
    taking too long 3) CPU starvation

Three named causes and no way to tell them apart. This times the pieces
separately so you stop guessing.

    # everything: motor bus + all three cameras
    python -m phi.utils.diagnose_loop --cameras wrist=0,top=1,front=2

    # motor bus alone — if this stalls, cameras were never the problem
    python -m phi.utils.diagnose_loop --no-cameras

    # cameras alone, no robot connection at all
    python -m phi.utils.diagnose_loop --cameras wrist=0,top=1,front=2 --no-robot

What we already know on this rig, measured rather than assumed:

  * ACT inference is NOT the cause. 49 of 50 ticks are a queue pop costing
    ~0.2 ms; the one chunk-refill tick costs ~46 ms (21.8 Hz) once every 1.67 s.
    Reported rates of 3.6 Hz (280 ms) and 8.6 Hz (116 ms) are 6x and 2.5x worse
    than the policy's worst possible tick, so it cannot be responsible.
  * The same warning appears during `lerobot-record`, where no policy runs at
    all. Whatever it is, it is not the policy.
  * `read_latest()` is non-blocking off a background thread, so a slow camera
    does not stall the loop directly — it only competes for USB and CPU.

That leaves the motor bus (a `sync_read` retry blocks the loop for as long as
the serial timeout), USB bandwidth contention (cameras and the motor controller
share the bus), and CPU/thermal throttling. Those are what this separates.
"""

from __future__ import annotations

import argparse
import statistics
import time


def parse_cameras(s: str) -> dict[str, int]:
    out = {}
    for tok in (t.strip() for t in s.split(",")):
        if not tok:
            continue
        n, _, i = tok.partition("=")
        out[n.strip()] = int(i)
    return out


def pct(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def report(name: str, xs: list[float], budget_ms: float) -> None:
    if not xs:
        print(f"  {name:22s} (no samples)")
        return
    over = sum(1 for x in xs if x > budget_ms)
    print(f"  {name:22s} median {statistics.median(xs):6.1f}  p90 {pct(xs,0.90):6.1f}  "
          f"p99 {pct(xs,0.99):7.1f}  max {max(xs):7.1f} ms   over budget: {over}/{len(xs)}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cameras", default="wrist=0,top=1,front=2")
    ap.add_argument("--no-cameras", action="store_true", help="motor bus only")
    ap.add_argument("--no-robot", action="store_true", help="cameras only, no serial connection")
    ap.add_argument("--port", default="/dev/tty.usbmodem5B7B0096441")
    ap.add_argument("--robot-id", default="phi_follower")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seconds", type=int, default=20)
    ap.add_argument("--test-send", action="store_true",
                    help="also time send_action, the serial WRITE the rollout does every tick "
                         "and a plain read loop does not. Re-sends the arm's CURRENT position, "
                         "so it should hold still — but torque is engaged, so watch it.")
    args = ap.parse_args(argv)

    budget = 1000.0 / args.fps
    cams = {} if args.no_cameras else parse_cameras(args.cameras)

    robot = None
    raw_caps = {}
    if not args.no_robot:
        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
        from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
        from lerobot.robots.utils import make_robot_from_config
        cfg = SO101FollowerConfig(
            port=args.port, id=args.robot_id, use_degrees=True,
            cameras={n: OpenCVCameraConfig(index_or_path=i, fps=args.fps,
                                           width=640, height=480, fourcc="MJPG")
                     for n, i in cams.items()},
        )
        robot = make_robot_from_config(cfg)
        print(f"connecting to {args.port} with cameras {sorted(cams) or 'none'} ...")
        robot.connect()
    else:
        import cv2
        for n, i in cams.items():
            c = cv2.VideoCapture(i)
            c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            c.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            c.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            raw_caps[n] = c
        print(f"cameras only: {sorted(cams)}")

    # 🚨 macOS CANNOT TELL YOU THE PIXEL FORMAT. OpenCV's AVFoundation backend
    #    returns fourcc=0 for these cameras, and its `set(CAP_PROP_FOURCC)` is
    #    largely a no-op — so "failed to set fourcc=MJPG" is NOT evidence that the
    #    stream is uncompressed, and an empty fourcc is "unreported", not "raw".
    #    Do not infer bandwidth from it. The only trustworthy signal on this
    #    platform is the DELIVERED frame rate measured below.
    if robot is not None:
        import cv2 as _cv
        print("camera formats actually in effect:")
        for n, cam in robot.cameras.items():
            vc = getattr(cam, "videocapture", None)
            if vc is None:
                continue
            v = int(vc.get(_cv.CAP_PROP_FOURCC))
            cc = "".join(chr((v >> (8 * i)) & 0xFF) for i in range(4)).replace("\x00", "").strip()
            cc = cc or "unreported"
            note = " (macOS/AVFoundation does not expose this — infer nothing from it)" \
                if cc == "unreported" else ""
            print(f"   {n:6s} fourcc={cc}{note}")
        print("   For reference, IF a stream were uncompressed YUYV it would be ~18.4 MB/s at")
        print("   640x480x30, against ~35-40 MB/s of practical USB 2.0. Whether that applies")
        print("   here is decided by the delivered frame rate below, not by the fourcc.")
    print(f"\ntarget {args.fps} Hz = {budget:.1f} ms per tick · running {args.seconds} s\n")

    bus_ms: list[float] = []
    send_ms: list[float] = []
    cam_ms: dict[str, list[float]] = {n: [] for n in cams}
    # Read latency is NOT a camera health metric: read_latest() peeks a buffer and
    # returns instantly whether or not a new frame ever arrived. What matters is
    # how often the frame actually CHANGES.
    fresh: dict[str, int] = {n: 0 for n in cams}
    last_ts: dict[str, float] = {}
    tick_ms: list[float] = []
    stalls: list[tuple[float, str, float]] = []

    t_end = time.perf_counter() + args.seconds
    t0 = time.perf_counter()
    try:
        while time.perf_counter() < t_end:
            tick = time.perf_counter()

            if robot is not None:
                t = time.perf_counter()
                robot.bus.sync_read("Present_Position")
                dt = (time.perf_counter() - t) * 1e3
                bus_ms.append(dt)
                if dt > budget:
                    stalls.append((time.perf_counter() - t0, "motor bus", dt))

                if args.test_send:
                    # Send back what we just read: a no-op target, but it exercises the
                    # same serial WRITE path the rollout uses every tick.
                    t = time.perf_counter()
                    try:
                        robot.send_action({k: v for k, v in
                                           robot.bus.sync_read("Present_Position").items()})
                    except Exception as e:
                        print(f"  send_action raised {type(e).__name__}: {e}")
                    dt = (time.perf_counter() - t) * 1e3
                    send_ms.append(dt)
                    if dt > budget:
                        stalls.append((time.perf_counter() - t0, "send_action", dt))

                for n, cam in robot.cameras.items():
                    t = time.perf_counter()
                    try:
                        cam.read_latest()
                    except Exception as e:
                        print(f"  [{time.perf_counter()-t0:5.1f}s] {n} read_latest raised "
                              f"{type(e).__name__}: {e}")
                    dt = (time.perf_counter() - t) * 1e3
                    cam_ms[n].append(dt)
                    ts = getattr(cam, "latest_timestamp", None)
                    if ts is not None and last_ts.get(n) != ts:
                        fresh[n] += 1
                        last_ts[n] = ts
                    if dt > budget:
                        stalls.append((time.perf_counter() - t0, f"camera {n}", dt))
            else:
                for n, c in raw_caps.items():
                    t = time.perf_counter()
                    c.read()
                    dt = (time.perf_counter() - t) * 1e3
                    cam_ms[n].append(dt)
                    if dt > budget:
                        stalls.append((time.perf_counter() - t0, f"camera {n}", dt))

            dt = (time.perf_counter() - tick) * 1e3
            tick_ms.append(dt)
            # absolute deadline, not sleep(remaining): sleep overshoots and the
            # error compounds, which is why an idle loop reported 27.1 of 30 Hz.
            next_tick = t0 + len(tick_ms) * (budget / 1000)
            time.sleep(max(0.0, next_tick - time.perf_counter()))
    except KeyboardInterrupt:
        print("\ninterrupted.")
    finally:
        if robot is not None:
            robot.disconnect()
        for c in raw_caps.values():
            c.release()

    print(f"\n--- {len(tick_ms)} ticks, budget {budget:.1f} ms ---")
    if bus_ms:
        report("motor bus sync_read", bus_ms, budget)
    if send_ms:
        report("motor bus send_action", send_ms, budget)
    for n, xs in cam_ms.items():
        report(f"camera {n} read", xs, budget)
    if any(fresh.values()):
        print("\n  DELIVERED frame rate (how often the frame actually changed):")
        for n, c in fresh.items():
            rate = c / args.seconds
            flag = "  <-- below target" if rate < args.fps * 0.9 else ""
            print(f"    {n:6s} {rate:5.1f} fps of {args.fps}{flag}")
    report("whole tick", tick_ms, budget)

    achieved = len(tick_ms) / args.seconds
    print(f"\nachieved ~{achieved:.1f} Hz against a {args.fps} Hz target")

    if stalls:
        print(f"\n{len(stalls)} stalls over budget. Worst 10:")
        for at, what, dt in sorted(stalls, key=lambda x: -x[2])[:10]:
            print(f"   t={at:6.1f}s   {what:16s} {dt:8.1f} ms")
        worst = {}
        for _, what, dt in stalls:
            worst[what] = worst.get(what, 0) + 1
        print(f"\nstalls by component: {dict(sorted(worst.items(), key=lambda x: -x[1]))}")
        print("\nReading it:")
        print("  motor bus dominates    -> serial retries. Suspect the USB cable/hub shared with")
        print("                            the cameras, or bus contention. Try --no-cameras: if the")
        print("                            bus is clean without them, it is contention, not the arm.")
        print("  one camera dominates   -> that camera is not delivering MJPG (check it fell back")
        print("                            to uncompressed YUYV) or is on a starved hub port.")
        print("  spread evenly          -> CPU or thermal throttling. Check the Mac is plugged in.")
    else:
        print("\nno stalls over budget — this configuration is healthy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

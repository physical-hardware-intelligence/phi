"""The robot worker: the only code that touches the rig's buses.

`RigWorker` is pure control logic driven one `tick()` at a time, so tests run it
deterministically on the mock rig. `run_worker` wraps it in the threads of the real worker
process (ARCHITECTURE ADR-S2).

Rules it enforces, whatever the UI sends:
  * torque only after every arm's identity is confirmed (session state READY)
  * goals only while MOVING, clipped per cycle to `max_step`
  * Stop and heartbeat loss freeze followers at their present position; torque stays on (ADR-S5)
  * any bus error or servo fault bit freezes every reachable follower and faults the session
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from typing import Any

from phi.studio.identity import Calibration, match_fingerprint
from phi.studio.rig import JOINTS, ArmBus, JointHealth
from phi.studio.session import IllegalTransition, Session, State

# WHY these step limits: LeRobot's max_relative_target defaults to None (config_so_follower.py:36),
# and a gripper motor burned from over-tightening (RECORDING_DAY.md). 8 deg per cycle is 240 deg/s
# at 30 Hz. Placeholder values until a hardware session with Parv sets them (GAP LEDGER).
DEFAULT_MAX_STEP = {j: 8.0 for j in JOINTS} | {"gripper": 5.0}
HEARTBEAT_TIMEOUT_S = 1.0  # Franka stops a hold-to-run link silent for over 1 s (research/02)


def pair_arms(arms: list[ArmBus]) -> list[tuple[ArmBus, ArmBus]]:
    """(leader, follower) pairs by name: `x_leader` drives `x_follower`."""
    by_name = {a.name: a for a in arms}
    pairs = []
    for f in arms:
        if f.role == "follower":
            lead = by_name.get(f.name.replace("follower", "leader"))
            if lead is not None and lead.role == "leader":
                pairs.append((lead, f))
    return pairs


class RigWorker:
    def __init__(
        self,
        rig: Any,
        send: Callable[[dict[str, Any]], None],
        clock: Callable[[], float] = time.monotonic,
        loop_hz: float = 30.0,
        calibrations: dict[str, Calibration] | None = None,
        health_every: int = 10,
    ) -> None:
        self.rig, self.send, self.clock, self.loop_hz = rig, send, clock, loop_hz
        self.arms: list[ArmBus] = list(rig.arms)
        self.followers = [a for a in self.arms if a.role == "follower"]
        self.pairs = pair_arms(self.arms)
        self.calibrations = calibrations if calibrations is not None else rig.calibration_files()
        self.health_every = health_every
        self.max_step = dict(DEFAULT_MAX_STEP)
        self.session = Session(on_change=lambda s: self.send({"type": "state", **s.snapshot()}))
        self.last_heartbeat = clock()
        self.dead: set[str] = set()  # arms whose bus stopped answering
        self.torque: dict[str, bool] = {a.name: False for a in self.arms}
        self.health: dict[str, dict[str, JointHealth]] = {}
        self.identity: list[dict[str, Any]] = []
        self._ticks: deque[float] = deque(maxlen=int(loop_hz * 2))
        self._cost_ms: deque[float] = deque(maxlen=int(loop_hz * 10))
        self._n = 0

    # -- commands from the UI ---------------------------------------------------------------------
    def handle(self, msg: dict[str, Any]) -> None:
        cmd = msg.get("cmd", "")
        fn = getattr(self, f"_cmd_{cmd}", None)
        if fn is None:
            self.error(f"unknown command {cmd!r}")
            return
        try:
            fn(msg)
        except IllegalTransition as e:
            self.error(str(e))
        except ConnectionError as e:
            self._fault(str(e))

    def error(self, message: str, fix: str = "") -> None:
        self.send({"type": "error", "message": message, "fix": fix})

    def _cmd_heartbeat(self, msg: dict[str, Any]) -> None:
        self.last_heartbeat = self.clock()

    def _cmd_connect(self, msg: dict[str, Any]) -> None:
        self.session.connected()
        self.dead.clear()
        self._identify()

    def _cmd_identify(self, msg: dict[str, Any]) -> None:
        self._identify()

    def _cmd_confirm(self, msg: dict[str, Any]) -> None:
        bad = [a for a in self.identity if not a["exact"]]
        if bad and self.session.state is State.IDENTIFIED:
            a = bad[0]
            self.error(
                f"{a['name']} does not match any calibration file exactly",
                f"nearest is {a['match']}, {a['max_deg']:.1f} deg off on {a['worst_joint']}. "
                "Calibrate this arm, or connect the arm that file belongs to.",
            )
            return
        self.session.confirmed()

    def _cmd_arm(self, msg: dict[str, Any]) -> None:
        self.session.armed()  # raises unless READY, before any torque write
        for f in self.followers:
            self._set_torque(f, True)
        self.last_heartbeat = self.clock()

    def _cmd_start(self, msg: dict[str, Any]) -> None:
        activity = msg.get("activity", "teleop")
        if activity != "teleop":
            self.error(f"{activity} is not available yet")
            return
        if not self.pairs:
            self.error("teleop needs a leader and a follower", "connect a leader arm")
            return
        self.session.started(activity)

    def _cmd_stop(self, msg: dict[str, Any]) -> None:
        if self.session.state in (State.MOVING, State.ARMED):
            self._freeze()
            self.session.stopped("user")
        elif self.session.state is State.FAULT:
            self._freeze()  # already frozen; Stop still acts while torque is on

    def _cmd_resume(self, msg: dict[str, Any]) -> None:
        self.session.resumed()
        self.last_heartbeat = self.clock()

    def _cmd_release(self, msg: dict[str, Any]) -> None:
        # WHY allowed in FAULT: an overloaded gripper keeps squeezing while torque is on.
        faulted = self.session.state is State.FAULT
        if not faulted and self.session.state not in (State.ARMED, State.STOPPED):
            self.session.released()  # raises with the reason
        for f in self.followers:
            if f.name not in self.dead:
                self._set_torque(f, False)
        if not faulted:
            self.session.released()

    def _cmd_clear(self, msg: dict[str, Any]) -> None:
        self.session.cleared()
        self.dead.clear()
        self._identify()

    def _cmd_disconnect(self, msg: dict[str, Any]) -> None:
        for f in self.followers:
            if f.name not in self.dead:
                try:
                    self._set_torque(f, False)
                except ConnectionError:
                    pass
        self.session.disconnected()

    def _cmd_inject(self, msg: dict[str, Any]) -> None:
        """Mock rig only: inject a fault so the UI's error paths can be exercised."""
        target = next((a for a in self.arms if a.name == msg.get("arm")), None)
        if target is None or not hasattr(target, "inject"):
            self.error("fault injection works only on the mock rig")
            return
        target.inject(msg["kind"], joint=msg.get("joint"))

    # -- internals --------------------------------------------------------------------------------
    def _identify(self) -> None:
        out = []
        for a in self.arms:
            regs = a.read_calibration()
            best = match_fingerprint(regs, self.calibrations)[0] if self.calibrations else None
            out.append({
                "name": a.name, "role": a.role,
                "port": getattr(a, "port", f"mock://{a.name}"),
                "serial": getattr(a, "serial", f"MOCK-{a.name}"),
                "match": best.name if best else None,
                "max_deg": round(best.distance.max_deg, 2) if best else None,
                "worst_joint": best.distance.worst_joint if best else None,
                "exact": bool(best and best.distance.exact),
            })  # fmt: skip
        self.identity = out
        self.send({"type": "identity", "arms": out})  # before the state, so the UI has the matches
        self.session.identified()

    def _set_torque(self, a: ArmBus, on: bool) -> None:
        a.set_torque(on)
        self.torque[a.name] = on

    def _freeze(self) -> None:
        """Goal = present position on every reachable follower with torque on. Torque stays on."""
        for f in self.followers:
            if f.name in self.dead:
                continue
            try:
                f.write_goals(f.read_positions())
            except ConnectionError:
                self.dead.add(f.name)

    def _fault(self, why: str) -> None:
        self._freeze()
        self.session.faulted(why)

    def _clip(self, goal: dict[str, float], present: dict[str, float]) -> dict[str, float]:
        return {
            j: present[j] + max(-self.max_step[j], min(self.max_step[j], goal[j] - present[j]))
            for j in goal
        }

    def tick(self) -> None:
        """One control cycle: heartbeat, read, act, health, publish."""
        t_start, now = time.perf_counter(), self.clock()
        self._ticks.append(now)
        self._n += 1
        st = self.session.state
        if st in (State.ARMED, State.MOVING) and now - self.last_heartbeat > HEARTBEAT_TIMEOUT_S:
            self._freeze()
            self.session.heartbeat_lost()
        if st is State.DISCONNECTED:
            return
        pos: dict[str, dict[str, float]] = {}
        current = None
        try:
            for a in self.arms:
                if a.name not in self.dead:
                    current = a
                    pos[a.name] = a.read_positions()
            if self.session.may_move and self.session.activity == "teleop":
                for lead, fol in self.pairs:
                    current = fol
                    fol.write_goals(self._clip(pos[lead.name], pos[fol.name]))
            if self._n % self.health_every == 0:
                for a in self.arms:
                    if a.name not in self.dead:
                        current = a
                        self.health[a.name] = a.read_health()
                self._check_health()
        except ConnectionError as e:
            if current is not None:
                self.dead.add(current.name)
            self._fault(f"{current.name if current else 'A bus'} is not answering: {e}")
        self._cost_ms.append((time.perf_counter() - t_start) * 1e3)
        self._publish(pos)

    def _check_health(self) -> None:
        if self.session.state is State.FAULT:
            return
        for name, joints in self.health.items():
            for j, h in joints.items():
                if h.faults:
                    self._fault(f"{name} {j}: {', '.join(h.faults)}")
                    return

    def _publish(self, pos: dict[str, dict[str, float]]) -> None:
        span = self._ticks[-1] - self._ticks[0] if len(self._ticks) > 1 else 0.0
        costs = sorted(self._cost_ms)
        arms: dict[str, Any] = {}
        for a in self.arms:
            h = self.health.get(a.name, {})
            arms[a.name] = {
                "role": a.role,
                "online": a.name not in self.dead,
                "torque": self.torque[a.name],
                "pos": {j: round(v, 2) for j, v in pos.get(a.name, {}).items()},
                "health": {j: {"load": round(x.load_pct, 1), "temp": round(x.temperature_c, 1),
                               "volt": round(x.voltage_v, 2), "faults": x.faults}
                           for j, x in h.items()},
            }  # fmt: skip
        self.send({
            "type": "telemetry", "t": self.clock(), "arms": arms,
            "loop": {"hz": round((len(self._ticks) - 1) / span, 1) if span else 0.0,
                     "p50_ms": round(costs[len(costs) // 2], 2) if costs else 0.0,
                     "p99_ms": round(costs[int(len(costs) * 0.99)], 2) if costs else 0.0},
        })  # fmt: skip


# -- the worker process --------------------------------------------------------------------------
def build_rig(spec: dict[str, Any]) -> Any:
    if spec.get("kind", "mock") == "mock":
        from phi.studio.mock import mock_rig

        cams = tuple(spec.get("cameras", ("front", "wrist", "top")))
        return mock_rig(pairs=int(spec.get("pairs", 1)), cameras=cams)
    raise ValueError(f"rig kind {spec.get('kind')!r} is not available yet; use the mock rig")


def encode_jpeg(frame: Any, quality: int = 80) -> bytes:
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def run_worker(conn: Any, spec: dict[str, Any]) -> None:
    """Entry point of the spawned worker process. `conn` is one end of a multiprocessing Pipe.

    Threads: a reader that queues commands, one publisher per camera, and the bus loop, which is
    the only thread that calls the rig's arms (one owner per bus, ADR-S4).
    """
    import queue
    import threading

    lock = threading.Lock()

    def send(msg: dict[str, Any]) -> None:
        with lock:
            try:
                conn.send(msg)
            except (BrokenPipeError, OSError):
                pass

    try:
        rig = build_rig(spec)
    except Exception as e:  # report, do not die silently
        send({"type": "error", "message": f"could not build the rig: {e}", "fix": ""})
        return
    hz = float(spec.get("hz", 30))
    w = RigWorker(rig, send, loop_hz=hz)
    inbox: queue.SimpleQueue[dict[str, Any]] = queue.SimpleQueue()
    done = threading.Event()

    def reader() -> None:
        while not done.is_set():
            try:
                inbox.put(conn.recv())
            except (EOFError, OSError):
                inbox.put({"cmd": "__exit__"})
                return

    def camera(cam: Any, fps: float) -> None:
        period, nxt = 1.0 / fps, time.monotonic()
        while not done.is_set():
            try:
                frame, t, seq = cam.read_latest()
                send({"type": "frame", "key": cam.key, "t": t, "seq": seq,
                      "w": int(frame.shape[1]), "h": int(frame.shape[0]),
                      "jpeg": encode_jpeg(frame)})  # fmt: skip
            except ConnectionError as e:
                send({"type": "camera", "key": cam.key, "online": False, "message": str(e)})
            nxt += period
            time.sleep(max(0.0, nxt - time.monotonic()))

    threading.Thread(target=reader, daemon=True).start()
    for cam in rig.cameras:
        threading.Thread(target=camera, args=(cam, float(spec.get("preview_fps", 15))),
                         daemon=True).start()  # fmt: skip
    send({"type": "state", **w.session.snapshot()})
    period, nxt = 1.0 / hz, time.monotonic()
    try:
        while True:
            while True:
                try:
                    msg = inbox.get_nowait()
                except queue.Empty:
                    break
                if msg.get("cmd") == "__exit__":
                    return
                w.handle(msg)
            w.tick()
            nxt += period
            delay = nxt - time.monotonic()
            if delay < -period:  # fell behind by a whole cycle: resynchronise, do not burst
                nxt = time.monotonic()
            time.sleep(max(0.0, delay))
    finally:
        # WHY freeze then release: the server is gone, so nobody can stop the arm from now on.
        # LeRobot itself releases torque on disconnect (config_so_follower.py:31).
        done.set()
        w._freeze()
        w.handle({"cmd": "disconnect"})
        for cam in rig.cameras:
            cam.close()

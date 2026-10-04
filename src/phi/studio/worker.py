"""The robot worker: the only code that touches the rig's buses.

`RigWorker` is pure control logic driven one `tick()` at a time, so tests run it
deterministically on the mock rig. `run_worker` wraps it in the threads of the real worker
process (ARCHITECTURE ADR-S2).

Rules it enforces, whatever the UI sends:
  * torque only after every arm's identity is confirmed (session state READY)
  * torque only on an arm whose registers match its own calibration file (a swap is refused)
  * enabling torque first writes goal = present position, so the arm holds where it is
  * goals only while MOVING, at most `max_step` ahead of the present position
  * Stop and heartbeat loss freeze followers at their present position; torque stays on (ADR-S5)
  * any bus error or servo fault bit freezes every reachable follower and faults the session, and
    so does any error the worker did not expect: a bug must stop the rig, not kill the worker
  * a fault clears only with torque off, since clearing re-checks every arm's identity
  * calibration runs only with every arm's torque off, never on an arm whose cable is swapped, and
    restores the old registers if it ends any way other than Save
  * a policy drives followers only while MOVING, through the same clip as teleop, and stops itself
    at its time limit
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from phi.studio.identity import Calibration, JointCal, match_fingerprint, save_calibration
from phi.studio.policy import Policy, PolicyInfo, catalog, is_finite_number
from phi.studio.rig import BUS_ERRORS, JOINTS, ArmBus, JointHealth
from phi.studio.session import IllegalTransition, Session, State

# WHY these step limits: LeRobot's max_relative_target defaults to None (config_so_follower.py:36),
# and a gripper motor burned from over-tightening (RECORDING_DAY.md). The clip is measured from the
# present position, so it bounds how far the goal leads the arm, not the arm's speed: the speed
# comes from the servo's position loop on that lead (unmeasured). Placeholder values until a
# hardware session with Parv sets them (GAP LEDGER).
DEFAULT_MAX_STEP = {j: 8.0 for j in JOINTS} | {"gripper": 5.0}
HEARTBEAT_TIMEOUT_S = 1.0  # Franka stops a hold-to-run link silent for over 1 s (research/02)
STOP_REASONS = {"user", "window closed", "control moved"}  # what a stop message may claim
# LeRobot records no range for wrist_roll and fixes it at 0..4095 (so_follower.py:135-143).
FULL_TURN = "wrist_roll"
FULL_RANGE = (0, 4095)
POLICY_LIMIT_S = (1.0, 600.0)


@dataclass
class CalRun:
    """One arm's calibration in progress: middle -> ranges -> review, then save or cancel."""

    arm: ArmBus
    old: Calibration  # registers before calibration, written back on cancel
    step: str = "middle"
    written: bool = False  # the servo's registers have changed
    homings: dict[str, int] = field(default_factory=dict)
    pos: dict[str, int] = field(default_factory=dict)
    mins: dict[str, int] = field(default_factory=dict)
    maxes: dict[str, int] = field(default_factory=dict)
    new: Calibration | None = None

    def view(self) -> dict[str, Any]:
        joints = {}
        for j in self.pos:
            lo, hi = FULL_RANGE if j == FULL_TURN else (self.mins[j], self.maxes[j])
            joints[j] = {"min": lo, "pos": self.pos[j], "max": hi, "fixed": j == FULL_TURN}
        return {"arm": self.arm.name, "role": self.arm.role, "step": self.step, "joints": joints,
                "old": {j: c._asdict() for j, c in self.old.items()},
                "new": ({j: c._asdict() for j, c in self.new.items()}
                        if self.new else None)}  # fmt: skip


@dataclass
class PolicyRun:
    info: PolicyInfo
    policy: Policy
    task: str
    limit_s: float
    t0: float
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])  # one per episode
    queue: deque[dict[str, dict[str, float]]] = field(default_factory=deque)
    step: int = 0
    episode_s: float = 0.0
    chunk_ms: deque[float] = field(default_factory=lambda: deque(maxlen=100))
    action: dict[str, dict[str, float]] = field(default_factory=dict)
    running: bool = True
    ended: str | None = None

    def view(self) -> dict[str, Any]:
        ms = sorted(self.chunk_ms)
        return {"id": self.info.id, "name": self.info.name, "task": self.task,
                "run_id": self.run_id,
                "limit_s": self.limit_s, "episode_s": round(self.episode_s, 2), "step": self.step,
                "chunk": self.policy.chunk, "running": self.running, "ended": self.ended,
                "chunk_ms": round(self.chunk_ms[-1], 3) if ms else 0.0,
                "chunk_ms_p50": round(ms[len(ms) // 2], 3) if ms else 0.0,
                "chunk_ms_max": round(ms[-1], 3) if ms else 0.0,
                "action": {f: {j: round(v, 2) for j, v in g.items()}
                           for f, g in self.action.items()}}  # fmt: skip


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


def _identity_problem(a: dict[str, Any]) -> tuple[str, str]:
    """(message, fix) for an arm whose registers do not match its own calibration file."""
    name, match = a["name"], a["match"]
    if match is None:
        return (f"{name}: no calibration files to compare with",
                "Calibrate the arm, or point Studio at the calibration directory.")  # fmt: skip
    if a["exact"]:
        return (f"{name} has {match}'s calibration: are their cables swapped?",
                f"Swap the USB cables of {name} and {match}, then read again.")  # fmt: skip
    return (f"{name} does not match its calibration file exactly",
            f"Nearest is {match}, {a['max_deg']:.1f} deg off on {a['worst_joint']}. Calibrate "
            "this arm, or connect the arm that file belongs to.")  # fmt: skip


class RigWorker:
    def __init__(
        self,
        rig: Any,
        send: Callable[[dict[str, Any]], None],
        clock: Callable[[], float] = time.monotonic,
        loop_hz: float = 30.0,
        calibrations: dict[str, Calibration] | None = None,
        health_every: int = 10,
        cal_dir: Path | None = None,
        policies: list[PolicyInfo] | None = None,
    ) -> None:
        self.rig, self.send, self.clock, self.loop_hz = rig, send, clock, loop_hz
        self.arms: list[ArmBus] = list(rig.arms)
        self.followers = [a for a in self.arms if a.role == "follower"]
        self.pairs = pair_arms(self.arms)
        self.calibrations = calibrations if calibrations is not None else rig.calibration_files()
        self.health_every = health_every
        self.max_step = dict(DEFAULT_MAX_STEP)
        self.cal_dir = cal_dir
        self.mock = all(hasattr(a, "inject") for a in self.arms)
        found = policies if policies is not None else catalog(self.mock)
        self.policies = {p.id: p for p in found}
        self.cal: CalRun | None = None
        self.run: PolicyRun | None = None
        self.session = Session(on_change=self._on_state)
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
        except BUS_ERRORS as e:
            self._fault(f"A bus stopped answering during {cmd}: {e}")
        except Exception as e:  # a bug: stop the rig and say so, do not kill the worker
            if self.session.state is State.DISCONNECTED:
                self.error(f"Studio hit an internal error running {cmd}: {e!r}")
            else:
                self._fault(f"Studio hit an internal error running {cmd}: {e!r}")

    def error(self, message: str, fix: str = "") -> None:
        self.send({"type": "error", "message": message, "fix": fix})

    def _on_state(self, s: Session) -> None:
        if self.run is not None and self.run.running and s.state is not State.MOVING:
            self.run.running = False
            self.run.ended = s.stop_reason or ("fault" if s.state is State.FAULT else "stopped")
        self.send({"type": "state", **s.snapshot()})

    def describe(self) -> dict[str, Any]:
        """What this rig is, for the UI: sent once at start and replayed to every new window."""
        return {"type": "rig", "mock": self.mock,
                "arms": [{"name": a.name, "role": a.role} for a in self.arms],
                "policies": [p.public() for p in self.policies.values()],
                "cal_dir": str(self.cal_dir) if self.cal_dir else None}  # fmt: skip

    def _cmd_heartbeat(self, msg: dict[str, Any]) -> None:
        self.last_heartbeat = self.clock()

    def _cmd_connect(self, msg: dict[str, Any]) -> None:
        self.session.connected()
        self.dead.clear()
        self._identify()

    def _cmd_identify(self, msg: dict[str, Any]) -> None:
        if self.cal is not None:
            self.error("Finish or cancel the calibration first.")
            return
        self._identify()

    def _cmd_confirm(self, msg: dict[str, Any]) -> None:
        bad = [a for a in self.identity if not a["ok"]]
        if bad and self.session.state is State.IDENTIFIED:
            self.error(*_identity_problem(bad[0]))
            return
        self.session.confirmed()
        if any(self.torque[f.name] for f in self.followers):
            # Torque left on by an earlier session: the arm is holding, so say so.
            self._freeze()
            self.session.armed()

    def _cmd_arm(self, msg: dict[str, Any]) -> None:
        self.session.armed()  # raises unless READY, before any torque write
        for f in self.followers:
            # WHY: a servo may drive to the goal it last stored once torque is on, and the arm may
            # have been moved by hand since (bench test 21). LeRobot does not do this:
            # enable_torque writes only Torque_Enable and Lock (feetech.py:302-305).
            f.write_goals(f.read_positions())
            self._set_torque(f, True)
        self.last_heartbeat = self.clock()

    def _cmd_start(self, msg: dict[str, Any]) -> None:
        activity = msg.get("activity", "teleop")
        if activity == "policy":
            self._start_policy(msg)
            return
        if activity != "teleop":
            self.error(f"{activity} is not available yet")
            return
        if not self.pairs:
            self.error("teleop needs a leader and a follower", "connect a leader arm")
            return
        self.session.started(activity)

    def _start_policy(self, msg: dict[str, Any]) -> None:
        pid = msg.get("policy")
        info = self.policies.get(pid) if isinstance(pid, str) else None
        if info is None or info.factory is None or not info.available:
            why = info.note if info else f"no policy named {msg.get('policy')!r}"
            self.error(f"{info.name if info else 'That policy'} cannot run here", why)
            return
        limit, task = msg.get("limit_s", 30.0), msg.get("task", "")
        lo, hi = POLICY_LIMIT_S
        if not is_finite_number(limit) or not lo <= limit <= hi:
            self.error(f"The time limit must be between {lo:.0f} and {hi:.0f} seconds.")
            return
        if not isinstance(task, str) or len(task) > 300:
            self.error("The task must be text of at most 300 characters.")
            return
        if not self.followers:
            self.error("A policy needs at least one follower arm.")
            return
        policy = info.factory(self.loop_hz)
        self.session.started("policy")  # raises unless ARMED: after a stop, resume first
        start = {f.name: f.read_positions() for f in self.followers}
        policy.reset(start, task)
        self.run = PolicyRun(info, policy, task, float(limit), t0=self.clock())

    def _policy_goals(
        self, now: float, pos: dict[str, dict[str, float]]
    ) -> list[tuple[ArmBus, dict[str, float]]]:
        """This tick's clipped goal per follower, or none once the time limit is reached."""
        r = self.run
        assert r is not None
        r.episode_s = now - r.t0
        if r.episode_s >= r.limit_s:
            self.session.stopped("time limit")
            self._freeze()
            return []
        if not r.queue:
            # WHY inline: the mock policy takes microseconds. A real policy must not run here.
            t = time.perf_counter()
            state = {f.name: pos[f.name] for f in self.followers}
            r.queue.extend(r.policy.infer(r.episode_s, state))
            r.chunk_ms.append((time.perf_counter() - t) * 1e3)
        r.action = r.queue.popleft()
        r.step += 1
        return [(f, self._clip(r.action[f.name], pos[f.name])) for f in self.followers]

    # -- calibration: LeRobot so_follower.py:126-157, one step per command ------------------------
    def _cmd_cal_start(self, msg: dict[str, Any]) -> None:
        a = next((x for x in self.arms if x.name == msg.get("arm")), None)
        if a is None:
            self.error(f"There is no arm named {msg.get('arm')!r}.")
            return
        holding = [n for n, on in self.torque.items() if on]
        if holding:
            self.error(f"Turn torque off before calibrating: {', '.join(holding)} holds torque.",
                       "Support the arm, then use Torque off.")  # fmt: skip
            return
        mine = next((x for x in self.identity if x["name"] == a.name), None)
        if mine and mine["exact"] and mine["match"] != a.calibration_id:
            self.error(f"{a.name} has {mine['match']}'s calibration: are their cables swapped?",
                       f"Fix the cables first. Calibrating now would write {a.name}'s calibration "
                       f"into {mine['match']}'s servos and hide the swap.")  # fmt: skip
            return
        self.session.calibration_started()  # raises unless IDENTIFIED or READY
        a.set_torque(False)  # LeRobot disables torque first (so_follower.py:127)
        self.cal = CalRun(a, old=a.read_calibration())

    def _cal_at(self, step: str) -> CalRun | None:
        if self.cal is None:
            self.error("No calibration is running.", "Start one from the Calibrate page.")
            return None
        if self.cal.step != step:
            self.error(f"Calibration is at the {self.cal.step} step, not {step}.")
            return None
        return self.cal

    def _cmd_cal_middle(self, msg: dict[str, Any]) -> None:
        c = self._cal_at("middle")
        if c is None:
            return
        c.written = True  # before the write: a write that fails halfway still changed registers
        c.homings = c.arm.set_half_turn_homings()
        c.pos = c.arm.read_raw_positions()
        c.mins = {j: v for j, v in c.pos.items() if j != FULL_TURN}
        c.maxes = dict(c.mins)
        c.step = "ranges"

    def _cmd_cal_finish(self, msg: dict[str, Any]) -> None:
        c = self._cal_at("ranges")
        if c is None:
            return
        still = [j for j in c.mins if c.mins[j] == c.maxes[j]]
        if still:  # LeRobot raises here (motors_bus.py:844-846); Studio lets the user carry on
            self.error(f"These joints did not move: {', '.join(still)}.",
                       "Move each one through its full range, then finish.")  # fmt: skip
            return
        c.new = {}
        for j, old in c.old.items():
            lo, hi = FULL_RANGE if j == FULL_TURN else (c.mins[j], c.maxes[j])
            c.new[j] = JointCal(old.id, 0, c.homings[j], lo, hi)  # drive_mode 0, as LeRobot
        c.step = "review"

    def _cmd_cal_save(self, msg: dict[str, Any]) -> None:
        c = self._cal_at("review")
        if c is None or c.new is None:
            return
        c.arm.write_calibration(c.new)
        self.calibrations[c.arm.calibration_id] = c.new
        path = None
        if self.cal_dir is not None:
            path = self.cal_dir / f"{c.arm.calibration_id}.json"
            try:
                save_calibration(c.new, path)
            except OSError as e:  # not a bus error: the registers are written
                self.error(f"Wrote {c.arm.name}'s registers but could not save {path}: {e}",
                           "Studio uses the new calibration until it restarts.")  # fmt: skip
                path = None
        self.cal = None
        self.session.calibration_ended()
        self.send({"type": "calibrated", "arm": c.arm.name, "path": str(path) if path else None})
        self._identify()

    def _cmd_cal_cancel(self, msg: dict[str, Any]) -> None:
        if self.cal is None:
            self.error("No calibration is running.")
            return
        self._cal_abort()
        self.session.calibration_ended()
        self._identify()

    def _cal_abort(self) -> None:
        """Drop the calibration in progress and write back the registers it changed."""
        c, self.cal = self.cal, None
        if c is None or not c.written:
            return
        try:
            c.arm.write_calibration(c.old)
        except BUS_ERRORS as e:
            self.error(f"Could not restore {c.arm.name}'s calibration registers: {e}",
                       "They no longer match its file. Calibrate it again once it "
                       "answers.")  # fmt: skip

    def _cmd_stop(self, msg: dict[str, Any]) -> None:
        reason = msg.get("reason", "user")
        if self.session.state in (State.MOVING, State.ARMED):
            self.session.stopped(reason if reason in STOP_REASONS else "user")
            self._freeze()
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
        silent = self._release_all()
        if silent:
            self.error(f"{', '.join(silent)} did not answer, so its torque may still be on",
                       "Support the arm and cut its power, then reconnect it.")  # fmt: skip
        if not faulted:
            self.session.released()

    def _cmd_clear(self, msg: dict[str, Any]) -> None:
        holding = [f.name for f in self.followers if self.torque[f.name]]
        if holding and self.session.state is State.FAULT:
            self.error(f"Turn torque off before clearing: {', '.join(holding)} still holds",
                       "Clearing re-checks every arm, so no arm may hold torque. Support each "
                       "follower, then use Torque off.")  # fmt: skip
            return
        self.session.cleared()
        self.dead.clear()
        self._identify()

    def _cmd_disconnect(self, msg: dict[str, Any]) -> None:
        self._cal_abort()
        self._release_all()
        self.run = None
        self.session.disconnected()

    def _cmd_inject(self, msg: dict[str, Any]) -> None:
        """Mock rig only: inject a fault so the UI's error paths can be exercised."""
        target = next((a for a in self.arms if a.name == msg.get("arm")), None)
        kinds = {"overload", "overheat", "voltage", "unplug", "replug", "clear", "swap", "hand",
                 "still"}  # fmt: skip
        if target is None or not hasattr(target, "inject"):
            self.error("fault injection works only on the mock rig")
            return
        if msg.get("kind") not in kinds:
            self.error(f"unknown fault {msg.get('kind')!r}", f"one of {', '.join(sorted(kinds))}")
            return
        if msg["kind"] == "swap":
            self.rig.swap_cables(target.name)
            return
        target.inject(msg["kind"], joint=msg.get("joint"))

    # -- internals --------------------------------------------------------------------------------
    def _identify(self) -> None:
        out = []
        for a in self.arms:
            regs = a.read_calibration()
            self.torque[a.name] = a.read_torque()  # the servo's word, not what Studio last sent
            best = match_fingerprint(regs, self.calibrations)[0] if self.calibrations else None
            exact = bool(best and best.distance.exact)
            out.append({
                "name": a.name, "role": a.role,
                "port": getattr(a, "port", f"mock://{a.name}"),
                "serial": getattr(a, "serial", f"MOCK-{a.name}"),
                "expected": a.calibration_id,
                "match": best.name if best else None,
                "max_deg": round(best.distance.max_deg, 2) if best else None,
                "worst_joint": best.distance.worst_joint if best else None,
                "exact": exact,
                "ok": exact and best is not None and best.name == a.calibration_id,
            })  # fmt: skip
        self.identity = out
        self.send({"type": "identity", "arms": out})  # before the state, so the UI has the matches
        self.session.identified()

    def _set_torque(self, a: ArmBus, on: bool) -> None:
        a.set_torque(on)
        self.torque[a.name] = on

    def _release_all(self) -> list[str]:
        """Torque off on every follower that may hold it, dead ones included (a replugged arm
        answers again). Returns the followers that did not answer."""
        silent = []
        for f in self.followers:
            if not self.torque[f.name]:
                continue
            try:
                self._set_torque(f, False)
            except BUS_ERRORS:
                silent.append(f.name)
        return silent

    def _freeze(self) -> None:
        """Goal = present position on every reachable follower with torque on. Torque stays on.
        A follower that does not answer is marked dead and faults the session."""
        lost = None
        for f in self.followers:
            # WHY skip limp followers: a goal write may enable torque on some Feetech firmware
            # (bench test 22), and a limp arm has nothing to hold.
            if f.name in self.dead or not self.torque[f.name]:
                continue
            try:
                f.write_goals(f.read_positions())
            except BUS_ERRORS as e:
                self.dead.add(f.name)
                lost = lost or f"{f.name} is not answering: {e}"
        if lost and self.session.state is not State.FAULT:
            self.session.faulted(lost)

    def _fault(self, why: str) -> None:
        self.session.faulted(why)
        self._freeze()
        self._cal_abort()

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
            self.session.heartbeat_lost()
            self._freeze()
        if st is State.DISCONNECTED:
            self._publish({})  # so torque badges and the Stop button reflect the release
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
            elif self.session.may_move and self.session.activity == "policy" and self.run:
                for fol, goal in self._policy_goals(now, pos):
                    current = fol
                    fol.write_goals(goal)
            if self.cal is not None and self.cal.step == "ranges":
                c = self.cal
                current = c.arm
                c.pos = c.arm.read_raw_positions()
                for j in c.mins:
                    c.mins[j], c.maxes[j] = min(c.mins[j], c.pos[j]), max(c.maxes[j], c.pos[j])
            if self._n % self.health_every == 0:
                for a in self.arms:
                    if a.name not in self.dead:
                        current = a
                        self.health[a.name] = a.read_health()
                self._check_health()
        except BUS_ERRORS as e:
            if current is not None:
                self.dead.add(current.name)
            self._fault(f"{current.name if current else 'A bus'} is not answering: {e}")
        except Exception as e:  # a bug: stop the rig and say so, do not kill the worker
            self._fault(f"Studio hit an internal error in the control loop: {e!r}")
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
            "calibration": self.cal.view() if self.cal else None,
            "policy": self.run.view() if self.run else None,
            "loop": {"hz": round((len(self._ticks) - 1) / span, 1) if span else 0.0,
                     "p50_ms": round(costs[len(costs) // 2], 2) if costs else 0.0,
                     "p99_ms": round(costs[int(len(costs) * 0.99)], 2) if costs else 0.0},
        })  # fmt: skip


# -- the worker process --------------------------------------------------------------------------
class Outbox:
    """Worker -> server messages. `put` never blocks, so a slow or stuck server cannot stall the bus
    loop or its heartbeat watchdog; one sender thread does the blocking Pipe writes. As in the
    server's Client, the newest telemetry and the newest frame per camera win, and every event is
    kept, in order."""

    def __init__(self, send: Callable[[dict[str, Any]], None]) -> None:
        self._send = send
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._events: deque[dict[str, Any]] = deque()
        self._telemetry: dict[str, Any] | None = None
        self._frames: dict[str, dict[str, Any]] = {}

    def put(self, msg: dict[str, Any]) -> None:
        with self._lock:
            kind = msg.get("type")
            if kind == "telemetry":
                self._telemetry = msg
            elif kind == "frame":
                self._frames[msg["key"]] = msg
            else:
                self._events.append(msg)
        self._wake.set()

    def _take(self) -> list[dict[str, Any]]:
        with self._lock:
            out = list(self._events)
            self._events.clear()
            if self._telemetry is not None:
                out.append(self._telemetry)
                self._telemetry = None
            out.extend(self._frames.values())
            self._frames = {}
        return out

    def run(self, done: threading.Event) -> None:
        while not done.is_set():
            self._wake.wait(0.1)
            self._wake.clear()
            for m in self._take():
                self._send(m)


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

    Threads: a reader that queues commands, one publisher per camera, the outbox sender, and the
    bus loop, which is the only thread that calls the rig's arms (one owner per bus, ADR-S4).
    """
    import queue

    def send(msg: dict[str, Any]) -> None:  # only the outbox thread calls this after startup
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
    outbox = Outbox(send)
    # WHY a separate directory for the mock: its calibrations must never land in LeRobot's own
    # calibration directory, where a real arm's file would be overwritten.
    data = spec.get("data_dir")
    mock = spec.get("kind", "mock") == "mock"
    cal_dir = Path(data) / "mock-calibration" if data and mock else None
    w = RigWorker(rig, outbox.put, loop_hz=hz, cal_dir=cal_dir)
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
                outbox.put({"type": "frame", "key": cam.key, "t": t, "seq": seq,
                      "w": int(frame.shape[1]), "h": int(frame.shape[0]),
                      "jpeg": encode_jpeg(frame)})  # fmt: skip
            except ConnectionError as e:
                outbox.put({"type": "camera", "key": cam.key, "online": False, "message": str(e)})
            nxt += period
            time.sleep(max(0.0, nxt - time.monotonic()))

    threading.Thread(target=reader, daemon=True).start()
    threading.Thread(target=outbox.run, args=(done,), daemon=True).start()
    for cam in rig.cameras:
        threading.Thread(target=camera, args=(cam, float(spec.get("preview_fps", 15))),
                         daemon=True).start()  # fmt: skip
    outbox.put(w.describe())
    outbox.put({"type": "state", **w.session.snapshot()})
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
        w._release_all()
        w._cal_abort()
        for cam in rig.cameras:
            cam.close()

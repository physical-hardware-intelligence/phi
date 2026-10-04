"""Robot worker control logic on the mock rig (SPEC TEL-1..6, ID-2), tick by tick."""

from __future__ import annotations

import pytest

from phi.studio.mock import FakeClock, mock_rig
from phi.studio.worker import Outbox, RigWorker

HZ = 30
DT = 1 / HZ


def make(pairs: int = 1) -> tuple[RigWorker, FakeClock, list[dict]]:
    clock = FakeClock()
    rig = mock_rig(pairs=pairs, cameras=("front",), clock=clock)
    out: list[dict] = []
    w = RigWorker(rig, send=out.append, clock=clock, loop_hz=HZ)
    return w, clock, out


def run(w: RigWorker, clock: FakeClock, seconds: float, heartbeat: bool = True) -> None:
    for _ in range(int(seconds * HZ)):
        if heartbeat:
            w.handle({"cmd": "heartbeat"})
        clock.advance(DT)
        w.tick()


def of(out: list[dict], kind: str) -> list[dict]:
    return [m for m in out if m.get("type") == kind]


def armed(pairs: int = 1) -> tuple[RigWorker, FakeClock, list[dict]]:
    w, clock, out = make(pairs)
    for c in ("connect", "confirm", "arm"):
        w.handle({"cmd": c})
    return w, clock, out


def arm(w: RigWorker, name: str):
    return next(a for a in w.rig.arms if a.name == name)


def test_connect_identifies_every_arm_by_its_registers() -> None:
    w, _, out = make(pairs=2)
    w.handle({"cmd": "connect"})
    ident = of(out, "identity")[-1]["arms"]
    assert {a["name"] for a in ident} == {
        "left_leader",
        "left_follower",
        "right_leader",
        "right_follower",
    }
    assert all(a["match"] == a["expected"] and a["exact"] and a["ok"] for a in ident)
    # LeRobot's bimanual ids: the section id plus the side (bi_so_follower.py:56, 66)
    assert {a["name"]: a["expected"] for a in ident}["right_leader"] == "mock_leader_right"
    assert w.session.state.name == "IDENTIFIED"


def test_torque_refused_until_confirmed_and_only_followers_get_it() -> None:
    w, _, out = make()
    w.handle({"cmd": "connect"})
    w.handle({"cmd": "arm"})
    assert "confirm" in of(out, "error")[-1]["message"]
    assert not any(a.torque for a in w.rig.arms)
    w.handle({"cmd": "confirm"})
    w.handle({"cmd": "arm"})
    assert arm(w, "follower").torque and not arm(w, "leader").torque


def test_teleop_makes_the_follower_track_the_leader() -> None:
    w, clock, _ = armed()
    w.handle({"cmd": "start", "activity": "teleop"})
    run(w, clock, 2.0)
    lead, fol = arm(w, "leader").read_positions(), arm(w, "follower").read_positions()
    assert max(abs(lead[j] - fol[j]) for j in lead) < 15


def test_bimanual_pairs_are_not_crossed() -> None:
    w, clock, _ = armed(pairs=2)
    arm(w, "right_leader")._t0 += 2.0  # desynchronise the two scripted hands
    w.handle({"cmd": "start", "activity": "teleop"})
    run(w, clock, 2.0)
    for side in ("left", "right"):
        lead = arm(w, f"{side}_leader").read_positions()
        fol = arm(w, f"{side}_follower").read_positions()
        assert max(abs(lead[j] - fol[j]) for j in lead) < 15, side


def test_per_cycle_goal_jump_is_clipped() -> None:
    w, clock, _ = armed()
    w.max_step = {j: 2.0 for j in w.max_step}
    fol = arm(w, "follower")
    before = fol.read_positions()["shoulder_pan"]
    arm(w, "leader").scripted = False  # hold the leader still
    arm(w, "leader").pos["shoulder_pan"] = before + 90  # then jump it
    w.handle({"cmd": "start", "activity": "teleop"})
    w.handle({"cmd": "heartbeat"})
    clock.advance(DT)
    w.tick()
    assert fol.goal["shoulder_pan"] - before == pytest.approx(2.0, abs=1e-6)


def test_stop_freezes_the_follower_while_the_leader_keeps_moving() -> None:
    w, clock, out = armed()
    w.handle({"cmd": "start", "activity": "teleop"})
    run(w, clock, 1.0)
    w.handle({"cmd": "stop"})
    frozen = arm(w, "follower").read_positions()
    run(w, clock, 1.0)
    now = arm(w, "follower").read_positions()
    assert max(abs(frozen[j] - now[j]) for j in now) < 1e-6
    assert arm(w, "follower").torque  # stop holds; it does not drop the arm
    assert of(out, "state")[-1]["state"] == "STOPPED"


def test_heartbeat_silence_stops_motion_within_1_2_s() -> None:
    w, clock, out = armed()
    w.handle({"cmd": "start", "activity": "teleop"})
    run(w, clock, 0.5)
    run(w, clock, 1.2, heartbeat=False)
    s = of(out, "state")[-1]
    assert s["state"] == "STOPPED" and s["stop_reason"] == "heartbeat"


def test_overload_faults_the_session_and_names_arm_joint_and_fault() -> None:
    w, clock, out = armed()
    w.handle({"cmd": "start", "activity": "teleop"})
    run(w, clock, 0.5)
    arm(w, "follower").inject("overload", joint="gripper")
    run(w, clock, 1.0)
    s = of(out, "state")[-1]
    assert s["state"] == "FAULT"
    assert "Follower Gripper: overload" in s["fault"]


def test_unplug_mid_teleop_faults_and_freezes_the_other_arms() -> None:
    w, clock, out = armed(pairs=2)
    w.handle({"cmd": "start", "activity": "teleop"})
    run(w, clock, 0.5)
    arm(w, "left_follower").inject("unplug")
    run(w, clock, 0.2)
    s = of(out, "state")[-1]
    assert s["state"] == "FAULT" and "left_follower" in s["fault"]
    right = arm(w, "right_follower")
    frozen = right.read_positions()
    run(w, clock, 0.5)
    assert max(abs(frozen[j] - right.read_positions()[j]) for j in frozen) < 1e-6


def test_telemetry_carries_positions_and_loop_timing() -> None:
    w, clock, out = armed()
    run(w, clock, 1.0)
    t = of(out, "telemetry")[-1]
    assert set(t["arms"]) == {"leader", "follower"}
    assert "shoulder_pan" in t["arms"]["follower"]["pos"]
    assert t["loop"]["hz"] > 0 and "p99_ms" in t["loop"]


def test_unknown_command_is_an_error_not_a_crash() -> None:
    w, _, out = make()
    w.handle({"cmd": "self_destruct"})
    assert "self_destruct" in of(out, "error")[-1]["message"]


def test_torque_is_reported_per_arm_and_survives_a_fault() -> None:
    w, clock, out = armed()
    w.handle({"cmd": "start", "activity": "teleop"})
    run(w, clock, 0.5)
    arm(w, "follower").inject("overload", joint="gripper")
    run(w, clock, 0.5)
    t = of(out, "telemetry")[-1]
    assert of(out, "state")[-1]["state"] == "FAULT"
    assert t["arms"]["follower"]["torque"] is True and t["arms"]["leader"]["torque"] is False


def test_torque_off_is_allowed_during_a_fault_and_the_fault_stays() -> None:
    # WHY: an overloaded gripper keeps squeezing while torque is on; the user must be able to let
    # go.
    w, clock, out = armed()
    arm(w, "follower").inject("overload", joint="gripper")
    run(w, clock, 0.5)
    w.handle({"cmd": "release"})
    run(w, clock, 0.1)
    assert not arm(w, "follower").torque
    assert of(out, "state")[-1]["state"] == "FAULT"
    assert of(out, "telemetry")[-1]["arms"]["follower"]["torque"] is False


# -- review fixes (2026-10-03 adversarial review) -------------------------------------------------
def test_swapped_cables_are_refused_at_confirm() -> None:
    # The 2026-08-06 incident: each arm matches the OTHER arm's file exactly.
    w, _, out = make()
    lead, fol = arm(w, "leader"), arm(w, "follower")
    w.handle({"cmd": "inject", "arm": "leader", "kind": "swap"})
    w.handle({"cmd": "connect"})
    ident = {a["name"]: a for a in of(out, "identity")[-1]["arms"]}
    assert ident["leader"]["exact"] and not ident["leader"]["ok"]
    assert (
        ident["leader"]["match"] == "mock_follower" and ident["leader"]["expected"] == "mock_leader"
    )
    w.handle({"cmd": "confirm"})
    assert "swapped" in of(out, "error")[-1]["message"]
    assert w.session.state.name == "IDENTIFIED"
    w.handle({"cmd": "arm"})
    assert not lead.torque and not fol.torque


def test_stop_right_after_an_unplug_faults_instead_of_stopping() -> None:
    w, clock, out = armed(pairs=2)
    w.handle({"cmd": "start", "activity": "teleop"})
    run(w, clock, 0.5)
    arm(w, "left_follower").inject("unplug")
    w.handle({"cmd": "stop"})  # before any tick sees the unplug
    assert w.session.state.name == "FAULT" and "Left Follower" in w.session.fault
    w.handle({"cmd": "resume"})
    run(w, clock, 0.5)  # must not raise
    assert w.session.state.name == "FAULT"
    assert arm(w, "right_follower").torque  # the other follower still holds


@pytest.mark.parametrize(
    "exc", [OSError("device disconnected"), RuntimeError("[RxPacketError] overload")]
)
def test_serial_and_servo_errors_fault_instead_of_killing_the_worker(exc: Exception) -> None:
    w, clock, out = armed()
    w.handle({"cmd": "start", "activity": "teleop"})
    run(w, clock, 0.3)
    fol = arm(w, "follower")

    def boom() -> dict:
        raise exc

    fol.read_positions = boom  # type: ignore[method-assign]
    run(w, clock, 0.2)
    s = of(out, "state")[-1]
    assert s["state"] == "FAULT" and "follower" in s["fault"]


def test_bad_commands_are_errors_not_crashes() -> None:
    w, _, out = make()
    w.handle({"cmd": "inject", "arm": "follower", "kind": "gremlins"})
    assert of(out, "error")
    w2 = RigWorker(mock_rig(cameras=()), send=(out2 := []).append, calibrations={})
    w2.handle({"cmd": "connect"})
    w2.handle({"cmd": "confirm"})  # no calibration files at all
    assert "calibration" in of(out2, "error")[-1]["message"]
    assert w2.session.state.name == "IDENTIFIED"


def test_enabling_torque_holds_a_follower_moved_by_hand() -> None:
    w, clock, _ = make()
    fol = arm(w, "follower")
    fol.pos["shoulder_pan"] += 40.0  # moved by hand with torque off; goal register is stale
    moved = dict(fol.pos)
    for c in ("connect", "confirm", "arm"):
        w.handle({"cmd": c})
    run(w, clock, 1.0)
    assert max(abs(fol.read_positions()[j] - moved[j]) for j in moved) < 1e-6


def test_stop_never_writes_goals_to_a_limp_follower() -> None:
    # A goal write may enable torque on some Feetech firmware (unverified, bench test 22).
    w, clock, out = armed()
    arm(w, "follower").inject("overload", joint="gripper")
    run(w, clock, 0.5)
    w.handle({"cmd": "release"})
    fol = arm(w, "follower")
    writes: list[dict] = []
    fol.write_goals = writes.append  # type: ignore[method-assign]
    w.handle({"cmd": "stop"})
    run(w, clock, 0.2)
    assert writes == []


def test_clear_is_refused_while_a_follower_holds_torque() -> None:
    w, clock, out = armed()
    fol = arm(w, "follower")
    fol.inject("overload", joint="gripper")
    run(w, clock, 0.5)
    fol.inject("clear")
    w.handle({"cmd": "clear"})
    assert "torque" in of(out, "error")[-1]["message"].lower()
    assert w.session.state.name == "FAULT"
    w.handle({"cmd": "release"})
    w.handle({"cmd": "clear"})
    assert w.session.state.name == "IDENTIFIED"


def test_catch_up_ticks_never_push_the_goal_further_ahead() -> None:
    # The clip is measured from the present position, so extra ticks with no time between them
    # leave the goal where it was.
    w, clock, _ = armed()
    w.max_step = {j: 2.0 for j in w.max_step}
    lead, fol = arm(w, "leader"), arm(w, "follower")
    lead.scripted = False
    before = fol.read_positions()["shoulder_pan"]
    lead.pos["shoulder_pan"] = before + 90
    w.handle({"cmd": "start", "activity": "teleop"})
    w.handle({"cmd": "heartbeat"})
    clock.advance(DT)
    w.tick()
    w.tick()  # catch-up tick, no time has passed
    w.tick()
    assert fol.goal["shoulder_pan"] - before == pytest.approx(2.0, abs=1e-6)


def test_freeze_is_what_holds_the_other_follower_after_an_unplug() -> None:
    w, clock, out = armed(pairs=2)
    w.handle({"cmd": "start", "activity": "teleop"})
    run(w, clock, 0.5)
    arm(w, "left_follower").inject("unplug")
    run(w, clock, DT)  # the tick that sees it
    right = arm(w, "right_follower")
    assert max(abs(right.goal[j] - right.pos[j]) for j in right.pos) < 1e-6


def test_heartbeat_loss_freezes_the_follower() -> None:
    # The leader jumps far away, so the follower is still chasing a goal 8 deg ahead when the
    # watchdog trips. Only the freeze puts the goal back on the present position at that tick.
    w, clock, out = armed()
    lead, fol = arm(w, "leader"), arm(w, "follower")
    lead.scripted = False
    lead.pos["shoulder_pan"] += 400.0
    w.handle({"cmd": "start", "activity": "teleop"})
    run(w, clock, 0.2)
    for _ in range(2 * HZ):
        clock.advance(DT)
        w.tick()
        if w.session.state.name == "STOPPED":
            break
    assert w.session.state.name == "STOPPED"
    assert lead.pos["shoulder_pan"] - fol.pos["shoulder_pan"] > 8  # still chasing
    assert abs(fol.goal["shoulder_pan"] - fol.pos["shoulder_pan"]) < 1e-6


def test_a_bug_in_the_control_loop_faults_instead_of_killing_the_worker() -> None:
    w, clock, out = armed()
    w.handle({"cmd": "start", "activity": "teleop"})

    def bug() -> dict:
        raise KeyError("shoulder_pan")  # not a bus error: a programming mistake

    arm(w, "follower").read_health = bug  # type: ignore[method-assign]
    run(w, clock, 0.5)  # must not raise
    s = of(out, "state")[-1]
    assert s["state"] == "FAULT" and "internal error" in s["fault"]


def test_telemetry_shows_torque_off_after_disconnect() -> None:
    w, clock, out = armed()
    run(w, clock, 0.2)
    w.handle({"cmd": "disconnect"})
    run(w, clock, 0.2)
    t = of(out, "telemetry")[-1]["arms"]["follower"]
    assert t["torque"] is False and t["health"] == {}  # no frozen readings from a rig that is gone


def test_torque_left_on_by_an_earlier_session_is_read_back() -> None:
    w, clock, out = make()
    fol = arm(w, "follower")
    fol.torque = True  # e.g. an earlier worker was killed with torque on
    for c in ("connect", "confirm"):
        w.handle({"cmd": c})
    assert w.session.state.name == "ARMED"  # holding, not "Torque off"
    run(w, clock, 0.1)
    assert of(out, "telemetry")[-1]["arms"]["follower"]["torque"] is True


def test_stop_reason_is_whitelisted() -> None:
    w, _, out = armed()
    w.handle({"cmd": "stop", "reason": "window closed"})
    assert of(out, "state")[-1]["stop_reason"] == "window closed"
    w2, _, out2 = armed()
    w2.handle({"cmd": "stop", "reason": "<script>"})
    assert of(out2, "state")[-1]["stop_reason"] == "user"


def test_a_stuck_server_never_blocks_the_bus_loop() -> None:
    import threading
    import time

    gate, sent, done = threading.Event(), [], threading.Event()

    def send(m: dict) -> None:
        gate.wait()  # the pipe is full: every write blocks
        sent.append(m)

    ob = Outbox(send)
    threading.Thread(target=ob.run, args=(done,), daemon=True).start()
    threading.Timer(1.0, gate.set).start()  # so a put that blocks fails the test, not hangs it
    t0 = time.perf_counter()
    for i in range(500):
        ob.put({"type": "telemetry", "i": i})
        ob.put({"type": "frame", "key": "front", "i": i})
    ob.put({"type": "state", "state": "STOPPED"})
    assert time.perf_counter() - t0 < 0.05  # put never waited on the pipe
    gate.set()
    end = time.monotonic() + 2
    while time.monotonic() < end and not any(m["type"] == "state" for m in sent):
        time.sleep(0.01)
    done.set()
    tele = [m["i"] for m in sent if m["type"] == "telemetry"]
    assert tele[-1] == 499 and len(tele) <= 2  # newest wins, nothing queued behind it
    assert any(m["type"] == "state" for m in sent)  # events are never dropped


def test_replugging_after_a_swap_lets_the_arms_confirm() -> None:
    w, _, out = make(pairs=2)
    w.handle({"cmd": "inject", "arm": "right_follower", "kind": "swap"})
    w.handle({"cmd": "connect"})
    bad = {a["name"] for a in of(out, "identity")[-1]["arms"] if not a["ok"]}
    assert bad == {"right_leader", "right_follower"}  # the left pair is untouched
    for a in ("right_leader", "right_follower"):
        w.handle({"cmd": "inject", "arm": a, "kind": "replug"})
    w.handle({"cmd": "identify"})
    w.handle({"cmd": "confirm"})
    assert w.session.state.name == "READY"

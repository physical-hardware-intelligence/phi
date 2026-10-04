"""Robot worker control logic on the mock rig (SPEC TEL-1..6, ID-2), tick by tick."""

from __future__ import annotations

import pytest

from phi.studio.mock import FakeClock, mock_rig
from phi.studio.worker import RigWorker

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
    assert all(a["match"] == a["name"] and a["exact"] for a in ident)
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
    assert "follower" in s["fault"] and "gripper" in s["fault"] and "overload" in s["fault"]


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

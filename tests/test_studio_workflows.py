"""Calibration and policy runs in the robot worker, on the mock rig, tick by tick.

Calibration follows LeRobot's own steps (so_follower.py:115-157): torque off, hold at mid-range and
write half-turn homings, record each joint's range by hand (wrist_roll fixed at 0..4095), then write
the registers and save the file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from phi.studio.identity import fingerprint_distance, load_calibration
from phi.studio.mock import FakeClock, mock_rig
from phi.studio.rig import JOINTS
from phi.studio.worker import RigWorker

HZ = 30
DT = 1 / HZ


def make(pairs: int = 1, cal_dir: Path | None = None) -> tuple[RigWorker, FakeClock, list[dict]]:
    clock = FakeClock()
    rig = mock_rig(pairs=pairs, cameras=("front",), clock=clock)
    out: list[dict] = []
    w = RigWorker(rig, send=out.append, clock=clock, loop_hz=HZ, cal_dir=cal_dir)
    return w, clock, out


def run(w: RigWorker, clock: FakeClock, seconds: float, heartbeat: bool = True) -> None:
    for _ in range(int(round(seconds * HZ))):
        if heartbeat:
            w.handle({"cmd": "heartbeat"})
        clock.advance(DT)
        w.tick()


def of(out: list[dict], kind: str) -> list[dict]:
    return [m for m in out if m.get("type") == kind]


def arm(w: RigWorker, name: str):
    return next(a for a in w.rig.arms if a.name == name)


def armed(pairs: int = 1) -> tuple[RigWorker, FakeClock, list[dict]]:
    w, clock, out = make(pairs)
    for c in ("connect", "confirm", "arm"):
        w.handle({"cmd": c})
    return w, clock, out


def cal(out: list[dict]) -> dict | None:
    return of(out, "telemetry")[-1]["calibration"]


def recorded(w: RigWorker, clock: FakeClock, out: list[dict], name: str = "follower") -> None:
    """Start calibrating `name`, set homing, sweep it by hand for 3 s."""
    w.handle({"cmd": "cal_start", "arm": name})
    w.handle({"cmd": "cal_middle"})
    arm(w, name).inject("hand")
    run(w, clock, 3.0, heartbeat=False)


# -- calibration -------------------------------------------------------------------------------
def test_calibration_saves_a_lerobot_file_that_matches_the_registers(tmp_path: Path) -> None:
    w, clock, out = make(cal_dir=tmp_path)
    w.handle({"cmd": "connect"})
    w.handle({"cmd": "cal_start", "arm": "follower"})
    assert w.session.state.name == "CALIBRATING"
    w.tick()
    assert cal(out)["arm"] == "follower" and cal(out)["step"] == "middle"

    w.handle({"cmd": "cal_middle"})
    w.tick()
    assert cal(out)["step"] == "ranges"
    assert all(v["pos"] == 2047 for v in cal(out)["joints"].values())  # half-turn homing

    arm(w, "follower").inject("hand")
    run(w, clock, 3.0, heartbeat=False)
    live = cal(out)["joints"]
    assert all(live[j]["min"] < 2047 < live[j]["max"] for j in JOINTS if j != "wrist_roll")

    w.handle({"cmd": "cal_finish"})
    w.tick()
    review = cal(out)
    assert review["step"] == "review"
    assert review["new"]["wrist_roll"]["range_min"] == 0
    assert review["new"]["wrist_roll"]["range_max"] == 4095
    assert set(review["old"]) == set(JOINTS)

    w.handle({"cmd": "cal_save"})
    saved = tmp_path / "follower.json"
    raw = json.loads(saved.read_text())
    assert set(raw["gripper"]) == {"id", "drive_mode", "homing_offset", "range_min", "range_max"}
    assert all(v["drive_mode"] == 0 for v in raw.values())  # LeRobot writes 0 (so_follower.py:149)
    regs = arm(w, "follower").read_calibration()
    assert fingerprint_distance(load_calibration(saved), regs).exact
    assert w.session.state.name == "IDENTIFIED"
    ident = {a["name"]: a for a in of(out, "identity")[-1]["arms"]}
    assert ident["follower"]["ok"] and ident["leader"]["ok"]
    done = {"type": "calibrated", "arm": "follower", "path": str(saved)}
    assert of(out, "calibrated")[-1] == done
    w.tick()
    assert cal(out) is None


def test_cancel_after_homing_restores_the_old_registers() -> None:
    w, clock, out = make()
    w.handle({"cmd": "connect"})
    before = arm(w, "follower").read_calibration()
    recorded(w, clock, out)
    assert not fingerprint_distance(arm(w, "follower").read_calibration(), before).exact
    w.handle({"cmd": "cal_cancel"})
    assert fingerprint_distance(arm(w, "follower").read_calibration(), before).exact
    assert w.session.state.name == "IDENTIFIED"
    assert all(a["ok"] for a in of(out, "identity")[-1]["arms"])


def test_finish_refuses_joints_that_never_moved_and_keeps_recording() -> None:
    # LeRobot raises on min == max (motors_bus.py:844-846); Studio says which joints and lets the
    # user keep moving them.
    w, clock, out = make()
    w.handle({"cmd": "connect"})
    w.handle({"cmd": "cal_start", "arm": "follower"})
    w.handle({"cmd": "cal_middle"})
    run(w, clock, 0.5, heartbeat=False)  # nobody moves the follower
    w.handle({"cmd": "cal_finish"})
    err = of(out, "error")[-1]
    assert "did not move" in err["message"] and "shoulder_pan" in err["message"]
    assert "wrist_roll" not in err["message"]  # its range is fixed, never recorded
    w.tick()
    assert cal(out)["step"] == "ranges" and w.session.state.name == "CALIBRATING"


def test_calibration_is_refused_while_any_arm_holds_torque() -> None:
    w, _, out = armed()
    w.handle({"cmd": "cal_start", "arm": "follower"})
    assert w.session.state.name == "ARMED" and of(out, "error")

    w, _, out = make()
    arm(w, "follower").torque = True  # left on by an earlier session, read back on connect
    w.handle({"cmd": "connect"})
    w.handle({"cmd": "cal_start", "arm": "leader"})
    assert "torque" in of(out, "error")[-1]["message"].lower()
    assert w.session.state.name == "IDENTIFIED"


def test_calibrating_an_arm_on_a_swapped_cable_is_refused() -> None:
    # The port named follower reaches the leader: calibrating it would write follower registers into
    # the leader and hide the swap from every later identity check.
    w, _, out = make()
    w.rig.swap_cables("follower")
    w.handle({"cmd": "connect"})
    w.handle({"cmd": "cal_start", "arm": "follower"})
    assert "swapped" in of(out, "error")[-1]["message"]
    assert w.session.state.name == "IDENTIFIED"


def test_unknown_arm_and_out_of_order_steps_are_errors() -> None:
    w, _, out = make()
    w.handle({"cmd": "connect"})
    w.handle({"cmd": "cal_start", "arm": "nope"})
    assert "nope" in of(out, "error")[-1]["message"]
    w.handle({"cmd": "cal_save"})
    assert of(out, "error")[-1]["message"]
    w.handle({"cmd": "cal_start", "arm": "follower"})
    n = len(of(out, "error"))
    w.handle({"cmd": "cal_finish"})  # before homing
    w.handle({"cmd": "cal_save"})  # before review
    assert len(of(out, "error")) == n + 2 and w.session.state.name == "CALIBRATING"


def test_stop_during_calibration_keeps_the_progress() -> None:
    # Esc is the global Stop. Nothing moves while calibrating, so it must not throw away a sweep.
    w, clock, out = make()
    w.handle({"cmd": "connect"})
    recorded(w, clock, out)
    w.handle({"cmd": "stop", "reason": "user"})
    w.tick()
    assert w.session.state.name == "CALIBRATING" and cal(out)["step"] == "ranges"


def test_disconnect_mid_calibration_restores_the_old_registers() -> None:
    w, clock, out = make()
    w.handle({"cmd": "connect"})
    before = arm(w, "follower").read_calibration()
    recorded(w, clock, out)
    w.handle({"cmd": "disconnect"})
    assert w.session.state.name == "DISCONNECTED"
    assert fingerprint_distance(arm(w, "follower").read_calibration(), before).exact


def test_unplug_mid_calibration_faults_and_drops_the_draft() -> None:
    w, clock, out = make()
    w.handle({"cmd": "connect"})
    recorded(w, clock, out)
    arm(w, "follower").inject("unplug")
    run(w, clock, 0.2, heartbeat=False)
    assert w.session.state.name == "FAULT"
    assert cal(out) is None


def test_saving_without_a_calibration_directory_still_writes_the_registers() -> None:
    w, clock, out = make(cal_dir=None)
    w.handle({"cmd": "connect"})
    recorded(w, clock, out)
    w.handle({"cmd": "cal_finish"})
    w.handle({"cmd": "cal_save"})
    assert of(out, "calibrated")[-1]["path"] is None
    assert all(a["ok"] for a in of(out, "identity")[-1]["arms"])  # Studio compares with the new one


# -- policy runs -------------------------------------------------------------------------------
def start_policy(w: RigWorker, limit_s: float = 5.0, policy: str = "mock-reach") -> None:
    w.handle({"cmd": "start", "activity": "policy", "policy": policy,
              "task": "put the cube in the box", "limit_s": limit_s})  # fmt: skip


def test_the_rig_lists_its_policies_and_says_why_one_is_unavailable() -> None:
    w, _, _ = make()
    rig = w.describe()
    pols = {p["id"]: p for p in rig["policies"]}
    assert pols["mock-reach"]["available"] is True
    off = [p for p in pols.values() if not p["available"]]
    assert off and all(p["note"] for p in off)


def test_a_policy_moves_the_followers_and_stops_at_its_time_limit() -> None:
    w, clock, out = armed()
    start = arm(w, "follower").read_positions()
    start_policy(w, limit_s=5.0)
    assert w.session.state.name == "MOVING"
    run(w, clock, 2.5)
    mid = arm(w, "follower").read_positions()
    assert max(abs(mid[j] - start[j]) for j in JOINTS) > 10
    p = of(out, "telemetry")[-1]["policy"]
    assert p["running"] and p["task"] == "put the cube in the box" and p["limit_s"] == 5.0
    assert set(p["action"]) == {"follower"} and p["chunk_ms"] >= 0
    run(w, clock, 3.0)
    s = of(out, "state")[-1]
    assert s["state"] == "STOPPED" and s["stop_reason"] == "time limit"
    p = of(out, "telemetry")[-1]["policy"]
    assert not p["running"] and p["ended"] == "time limit"
    assert p["episode_s"] == pytest.approx(5.0, abs=2 * DT)
    assert arm(w, "follower").torque  # holds at the end; it does not drop the arm


def test_policy_goals_are_clipped_like_teleop() -> None:
    w, clock, _ = armed()
    w.max_step = {j: 1.0 for j in w.max_step}
    fol = arm(w, "follower")
    leads: list[float] = []
    real = fol.write_goals

    def spy(goals: dict[str, float]) -> None:
        present = fol.read_positions()
        leads.extend(abs(goals[j] - present[j]) for j in goals)
        real(goals)

    fol.write_goals = spy  # type: ignore[method-assign]
    start_policy(w)
    run(w, clock, 2.0)
    assert leads and max(leads) <= 1.0 + 1e-6


def test_a_second_episode_starts_after_resume_from_step_zero() -> None:
    w, clock, out = armed()
    start_policy(w, limit_s=1.0)
    run(w, clock, 1.5)
    assert w.session.state.name == "STOPPED"
    start_policy(w, limit_s=1.0)  # must resume first
    assert w.session.state.name == "STOPPED"
    w.handle({"cmd": "resume"})
    start_policy(w, limit_s=1.0)
    w.tick()
    p = of(out, "telemetry")[-1]["policy"]
    assert p["running"] and p["step"] <= 1


def test_user_stop_ends_the_episode_and_names_the_reason() -> None:
    w, clock, out = armed()
    start_policy(w)
    run(w, clock, 1.0)
    w.handle({"cmd": "stop", "reason": "user"})
    w.tick()
    p = of(out, "telemetry")[-1]["policy"]
    assert not p["running"] and p["ended"] == "user"


@pytest.mark.parametrize("msg", [
    {"policy": "act"},  # listed but not available
    {"policy": "nope"},
    {"limit_s": "ten"},
    {"limit_s": 0},
    {"task": 7},
])  # fmt: skip
def test_bad_policy_starts_are_refused(msg: dict) -> None:
    w, _, out = armed()
    base = {"cmd": "start", "activity": "policy", "policy": "mock-reach", "task": "t", "limit_s": 5}
    w.handle(base | msg)
    assert w.session.state.name == "ARMED" and of(out, "error")


def test_a_bimanual_policy_drives_both_followers() -> None:
    w, clock, out = armed(pairs=2)
    start = {n: arm(w, n).read_positions() for n in ("left_follower", "right_follower")}
    start_policy(w)
    run(w, clock, 2.5)
    for n, s in start.items():
        now = arm(w, n).read_positions()
        assert max(abs(now[j] - s[j]) for j in JOINTS) > 10, n
    assert set(of(out, "telemetry")[-1]["policy"]["action"]) == {"left_follower", "right_follower"}

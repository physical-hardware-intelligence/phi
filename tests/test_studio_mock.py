"""The mock rig behaves enough like SO-101 hardware to drive every Studio flow without an arm
(ADR-S13)."""

from __future__ import annotations

import pytest

from phi.studio.identity import fingerprint_distance
from phi.studio.mock import FakeClock, MockArm, MockCamera, mock_rig
from phi.studio.rig import JOINTS, decode_status


def test_status_bits_decode_every_fault_not_just_the_first() -> None:
    assert decode_status(0) == []
    assert decode_status(1 | 32) == ["voltage", "overload"]


def test_follower_moves_toward_goal_only_with_torque_on() -> None:
    clock = FakeClock()
    arm = MockArm("f", "follower", clock=clock)
    start = arm.read_positions()["elbow_flex"]
    arm.write_goals({"elbow_flex": start + 30})
    clock.advance(1.0)
    assert arm.read_positions()["elbow_flex"] == pytest.approx(start)  # torque off: goal ignored
    arm.set_torque(True)
    arm.write_goals({"elbow_flex": start + 30})
    clock.advance(0.05)
    mid = arm.read_positions()["elbow_flex"]
    assert start < mid < start + 30  # rate limited, not a jump
    clock.advance(2.0)
    assert arm.read_positions()["elbow_flex"] == pytest.approx(start + 30)


def test_leader_with_torque_off_follows_a_hand() -> None:
    clock = FakeClock()
    arm = MockArm("l", "leader", clock=clock)
    a = arm.read_positions()
    clock.advance(0.5)
    b = arm.read_positions()
    assert any(abs(a[j] - b[j]) > 1 for j in JOINTS)


def test_overload_injection_shows_on_the_right_joint() -> None:
    arm = MockArm("f", "follower", clock=FakeClock())
    arm.inject("overload", joint="gripper")
    h = arm.read_health()
    assert h["gripper"].faults == ["overload"] and h["elbow_flex"].faults == []


def test_unplug_makes_every_read_raise() -> None:
    arm = MockArm("f", "follower", clock=FakeClock())
    arm.inject("unplug")
    with pytest.raises(ConnectionError):
        arm.read_positions()
    with pytest.raises(ConnectionError):
        arm.write_goals({"gripper": 10})


def test_registers_hold_the_given_calibration() -> None:
    rig = mock_rig(pairs=2)
    names = sorted(a.name for a in rig.arms)
    assert names == ["left_follower", "left_leader", "right_follower", "right_leader"]
    a, b = rig.arms[0], rig.arms[1]
    assert fingerprint_distance(a.read_calibration(), a.read_calibration()).exact
    assert not fingerprint_distance(a.read_calibration(), b.read_calibration()).exact  # arms differ


def test_camera_frames_are_rgb_with_rising_sequence() -> None:
    clock = FakeClock()
    cam = MockCamera("front", clock=clock, fps=30)
    f1, t1, n1 = cam.read_latest()
    clock.advance(0.1)
    f2, t2, n2 = cam.read_latest()
    assert f1.shape == (480, 640, 3) and f1.dtype.name == "uint8"
    assert n2 == n1 + 3 and t2 > t1
    assert (f1 != f2).any()


def test_camera_unplug_raises() -> None:
    cam = MockCamera("wrist", clock=FakeClock())
    cam.inject("unplug")
    with pytest.raises(ConnectionError):
        cam.read_latest()

"""Session state machine (SPEC TEL-2, TEL-4, TEL-5): what may happen in which state."""

from __future__ import annotations

import pytest

from phi.studio.session import IllegalTransition, Session, State


def ready() -> Session:
    s = Session()
    s.connected()
    s.identified()
    s.confirmed()
    return s


def test_happy_path_to_moving_and_back() -> None:
    s = ready()
    s.armed()
    s.started("teleop")
    assert s.state is State.MOVING and s.activity == "teleop"
    s.stopped("user")
    assert s.state is State.STOPPED and s.activity is None
    s.resumed()
    assert s.state is State.ARMED
    s.released()
    assert s.state is State.READY


def test_torque_cannot_be_enabled_before_roles_are_confirmed() -> None:
    s = Session()
    s.connected()
    s.identified()
    with pytest.raises(IllegalTransition, match="confirm"):
        s.armed()


def test_cannot_start_motion_unless_armed() -> None:
    s = ready()
    with pytest.raises(IllegalTransition):
        s.started("teleop")


def test_resume_after_stop_is_explicit_not_a_start() -> None:
    s = ready()
    s.armed()
    s.started("policy")
    s.stopped("user")
    with pytest.raises(IllegalTransition):
        s.started("policy")  # must resume (back to ARMED) first


def test_heartbeat_loss_stops_only_when_something_could_move() -> None:
    s = ready()
    assert s.heartbeat_lost() is False and s.state is State.READY
    s.armed()
    s.started("teleop")
    assert s.heartbeat_lost() is True
    assert s.state is State.STOPPED and s.stop_reason == "heartbeat"


def test_fault_from_any_state_and_recovery_requires_reidentify() -> None:
    s = ready()
    s.armed()
    s.started("teleop")
    s.faulted("motor 1 overload")
    assert s.state is State.FAULT and s.fault == "motor 1 overload"
    s.cleared()
    assert s.state is State.CONNECTED and s.fault is None
    with pytest.raises(IllegalTransition):
        s.armed()


def test_disconnect_from_anywhere() -> None:
    for prep in (
        lambda s: None,
        lambda s: (s.connected(),),
        lambda s: (s.connected(), s.identified()),
    ):
        s = Session()
        prep(s)
        s.disconnected()
        assert s.state is State.DISCONNECTED


def test_every_state_has_a_label_tone_and_next_action() -> None:
    for st in State:
        assert (
            st.label
            and st.tone in {"neutral", "info", "ok", "active", "warn", "danger"}
            and st.next_action
        )


def test_listener_sees_every_change_once() -> None:
    seen: list[State] = []
    s = Session(on_change=lambda sess: seen.append(sess.state))
    s.connected()
    s.identified()
    s.identified()  # same state again is not a change
    assert seen == [State.CONNECTED, State.IDENTIFIED]


def test_one_notification_per_transition_and_no_change_on_illegal() -> None:
    seen: list[tuple[State, str | None]] = []
    s = Session(on_change=lambda sess: seen.append((sess.state, sess.activity)))
    s.connected()
    s.identified()
    s.confirmed()
    s.armed()
    seen.clear()
    s.started("teleop")
    assert seen == [(State.MOVING, "teleop")]
    with pytest.raises(IllegalTransition):
        s.resumed()
    assert s.state is State.MOVING and s.activity == "teleop" and len(seen) == 1


def test_calibration_starts_before_torque_and_ends_in_a_fresh_identity_check() -> None:
    for prep in (lambda s: None, lambda s: s.confirmed()):  # IDENTIFIED, READY
        s = Session()
        s.connected()
        s.identified()
        prep(s)
        s.calibration_started()
        assert s.state is State.CALIBRATING and s.activity == "calibration"
        s.calibration_ended()
        assert s.state is State.CONNECTED and s.activity is None
    s = ready()
    s.armed()
    with pytest.raises(IllegalTransition):
        s.calibration_started()  # torque may be on


def test_nothing_can_enable_torque_or_start_while_calibrating() -> None:
    s = ready()
    s.calibration_started()
    for event in (s.armed, s.confirmed, lambda: s.started("teleop"), s.resumed):
        with pytest.raises(IllegalTransition):
            event()
    assert s.heartbeat_lost() is False and s.state is State.CALIBRATING
    assert not s.torque_allowed

"""One rig session's state: the single source for what may move and when.

    DISCONNECTED -> CONNECTED -> IDENTIFIED -> READY -> ARMED -> MOVING
                                                ^        ^  |      |
                                                |        |  v      v stop / heartbeat loss
                                             release  resume STOPPED
    any state -> FAULT -> (clear) -> CONNECTED, so a fault always forces a fresh identity check.

WHY a hand-written table rather than flags: every torque or goal write in the worker asks this
object first, so
"can this arm move now" has exactly one answer (SPEC TEL-2, design rule 5).
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum


class State(Enum):
    # value: (label, tone, next action shown to the user)
    DISCONNECTED = ("Disconnected", "neutral", "Connect the rig")
    CONNECTED = ("Connected", "info", "Checking which arm is on which port")
    IDENTIFIED = ("Identified", "info", "Confirm each arm's role and calibration")
    READY = ("Ready", "ok", "Enable torque to start")
    ARMED = ("Torque on", "warn", "Start teleop, a policy, or a replay")
    MOVING = ("Moving", "active", "Stop with Esc")
    STOPPED = ("Stopped", "warn", "Resume, or release torque")
    FAULT = ("Fault", "danger", "Read the fault, fix it, then clear")

    @property
    def label(self) -> str:
        return self.value[0]

    @property
    def tone(self) -> str:
        return self.value[1]

    @property
    def next_action(self) -> str:
        return self.value[2]


S = State
# event -> states it is legal from. FAULT and DISCONNECTED are reachable from anywhere.
_FROM: dict[str, tuple[State, ...]] = {
    "connected": (S.DISCONNECTED,),
    "identified": (S.CONNECTED, S.IDENTIFIED),
    "confirmed": (S.IDENTIFIED,),
    "armed": (S.READY,),
    "started": (S.ARMED,),
    "stopped": (S.MOVING, S.ARMED),
    "resumed": (S.STOPPED,),
    "released": (S.ARMED, S.STOPPED),
    "cleared": (S.FAULT,),
}
_WHY = {
    "armed": "confirm each arm's role and calibration before torque is enabled",
    "started": "enable torque first, and after a stop resume explicitly",
}


class IllegalTransition(RuntimeError):
    pass


class Session:
    def __init__(self, on_change: Callable[[Session], None] | None = None) -> None:
        self.state = S.DISCONNECTED
        self.activity: str | None = None  # teleop | policy | replay | calibration
        self.stop_reason: str | None = None
        self.fault: str | None = None
        self._on_change = on_change

    def _go(self, event: str | None, to: State, **fields: str | None) -> None:
        """Check legality (event None = reachable from anywhere), apply fields, then notify once."""
        if event is not None and self.state not in _FROM[event]:
            why = _WHY.get(event, f"not allowed while {self.state.label.lower()}")
            raise IllegalTransition(f"{event}: {why} (state is {self.state.label})")
        changed = to is not self.state or any(getattr(self, k) != v for k, v in fields.items())
        for k, v in fields.items():
            setattr(self, k, v)
        self.state = to
        if changed and self._on_change:
            self._on_change(self)

    def connected(self) -> None:
        self._go("connected", S.CONNECTED)

    def identified(self) -> None:
        self._go("identified", S.IDENTIFIED)

    def confirmed(self) -> None:
        self._go("confirmed", S.READY)

    def armed(self) -> None:
        self._go("armed", S.ARMED)

    def started(self, activity: str) -> None:
        self._go("started", S.MOVING, activity=activity, stop_reason=None)

    def stopped(self, reason: str) -> None:
        self._go("stopped", S.STOPPED, activity=None, stop_reason=reason)

    def resumed(self) -> None:
        self._go("resumed", S.ARMED)

    def released(self) -> None:
        self._go("released", S.READY)

    def heartbeat_lost(self) -> bool:
        """Stop if anything could be moving. Returns whether it stopped."""
        if self.state in (S.MOVING, S.ARMED):
            self.stopped("heartbeat")
            return True
        return False

    def faulted(self, why: str) -> None:
        self._go(None, S.FAULT, activity=None, fault=why)

    def cleared(self) -> None:
        self._go("cleared", S.CONNECTED, fault=None)

    def disconnected(self) -> None:
        self._go(None, S.DISCONNECTED, activity=None, stop_reason=None, fault=None)

    @property
    def may_move(self) -> bool:
        return self.state is S.MOVING

    @property
    def torque_allowed(self) -> bool:
        return self.state in (S.ARMED, S.MOVING, S.STOPPED)

    def snapshot(self) -> dict[str, str | None]:
        return {
            "state": self.state.name,
            "label": self.state.label,
            "tone": self.state.tone,
            "next_action": self.state.next_action,
            "activity": self.activity,
            "stop_reason": self.stop_reason,
            "fault": self.fault,
        }

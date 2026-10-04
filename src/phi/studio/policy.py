"""Policies the worker can run on the followers in place of the leader arms.

A policy takes the followers' present joint positions and returns a chunk of goal positions, one
per control tick, which the worker plays out before asking for the next chunk. The worker clips
every goal exactly as it clips teleop goals.

Only the scripted mock policy runs today. Real checkpoints (ACT, pi0.5) need the hardware backend
and inference on its own thread: a forward pass that takes longer than one control period would
stall the bus loop and its heartbeat watchdog if it ran inline. The catalog lists them as
unavailable and says why, rather than hiding them.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from phi.studio.rig import JOINTS

State = dict[str, dict[str, float]]  # follower name -> joint -> degrees (gripper 0-100)


class Policy(Protocol):
    chunk: int

    def reset(self, start: State, task: str) -> None: ...
    def infer(self, t: float, state: State) -> list[State]:
        """Goals for ticks t, t + dt, ..., one entry per tick, for every follower."""
        ...


@dataclass(frozen=True)
class PolicyInfo:
    id: str
    name: str
    available: bool
    note: str = ""
    factory: Callable[[float], Policy] | None = None  # control rate in Hz -> policy

    def public(self) -> dict[str, object]:
        return {"id": self.id, "name": self.name, "available": self.available, "note": self.note}


# Keyframes of a reach, grasp, lift, carry and release, in absolute joint degrees (gripper 0-100,
# open is high). Illustrative poses for the mock rig, not measured on an SO-101.
_KEYFRAMES: tuple[tuple[float, dict[str, float]], ...] = (
    (1.5, {"shoulder_pan": 20, "shoulder_lift": -40, "elbow_flex": 40, "wrist_flex": 70,
           "wrist_roll": 0, "gripper": 60}),
    (2.5, {"shoulder_pan": 20, "shoulder_lift": -25, "elbow_flex": 30, "wrist_flex": 75,
           "wrist_roll": 0, "gripper": 60}),
    (3.2, {"shoulder_pan": 20, "shoulder_lift": -25, "elbow_flex": 30, "wrist_flex": 75,
           "wrist_roll": 0, "gripper": 10}),
    (4.2, {"shoulder_pan": 20, "shoulder_lift": -60, "elbow_flex": 70, "wrist_flex": 60,
           "wrist_roll": 0, "gripper": 10}),
    (5.5, {"shoulder_pan": -25, "shoulder_lift": -60, "elbow_flex": 70, "wrist_flex": 60,
           "wrist_roll": 0, "gripper": 10}),
    (6.3, {"shoulder_pan": -25, "shoulder_lift": -60, "elbow_flex": 70, "wrist_flex": 60,
           "wrist_roll": 0, "gripper": 60}),
    (7.5, {"shoulder_pan": 0, "shoulder_lift": -90, "elbow_flex": 90, "wrist_flex": 60,
           "wrist_roll": 0, "gripper": 5}),
)  # fmt: skip


def _smooth(x: float) -> float:
    """Smoothstep: zero velocity at both ends of each segment, so the arm does not jerk."""
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


class MockReachPolicy:
    """Scripted reach and place from wherever the follower starts. A right arm mirrors the pan."""

    def __init__(self, hz: float, chunk: int = 10) -> None:
        self.dt, self.chunk = 1.0 / hz, chunk
        self._paths: dict[str, list[tuple[float, dict[str, float]]]] = {}

    def reset(self, start: State, task: str) -> None:
        self._paths = {}
        for name, pose in start.items():
            sign = -1.0 if name.startswith("right") else 1.0
            frames = [(t, {j: sign * v if j == "shoulder_pan" else v for j, v in p.items()})
                      for t, p in _KEYFRAMES]  # fmt: skip
            self._paths[name] = [(0.0, {j: pose[j] for j in JOINTS}), *frames]

    def _at(self, name: str, t: float) -> dict[str, float]:
        path = self._paths[name]
        for (t0, a), (t1, b) in zip(path, path[1:], strict=False):
            if t < t1:
                s = _smooth((t - t0) / (t1 - t0))
                return {j: a[j] + s * (b[j] - a[j]) for j in JOINTS}
        return dict(path[-1][1])

    def infer(self, t: float, state: State) -> list[State]:
        return [{name: self._at(name, t + k * self.dt) for name in self._paths}
                for k in range(self.chunk)]  # fmt: skip


_NOT_WIRED = ("Not wired yet: real checkpoints need the hardware backend and inference on its own "
              "thread.")  # fmt: skip


def catalog(mock: bool) -> list[PolicyInfo]:
    return [
        PolicyInfo("mock-reach", "Scripted reach and place", available=mock,
                   note="" if mock else "Mock rig only.",
                   factory=lambda hz: MockReachPolicy(hz)),
        PolicyInfo("act", "ACT checkpoint", available=False, note=_NOT_WIRED),
        PolicyInfo("pi05", "π0.5 checkpoint", available=False, note=_NOT_WIRED),
    ]  # fmt: skip


def is_finite_number(x: object) -> bool:
    return isinstance(x, int | float) and not isinstance(x, bool) and math.isfinite(x)

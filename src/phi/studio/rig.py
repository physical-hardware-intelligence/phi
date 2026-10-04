"""What the robot worker needs from hardware, real or mock.

The worker talks only to these two protocols, so the real LeRobot backend and the mock rig are
interchangeable
(ARCHITECTURE ADR-S13) and every flow can run in CI with no arm attached.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from phi.studio.identity import Calibration

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")

# STS3215 Status register (65) bits. Same layout as Unloading_Condition (19) and
# LED_Alarm_Condition (20) in
# Waveshare's ST3215 map; scservo_sdk names five of them (protocol_packet_handler.py:18-22, 51-67).
STATUS_BITS = {
    1: "voltage",
    2: "angle sensor",
    4: "overheat",
    8: "overcurrent",
    16: "angle",
    32: "overload",
}


def decode_status(status: int) -> list[str]:
    """Every set fault bit by name. The SDK reports only the first, so a GUI must decode all of
    them."""
    return [name for bit, name in STATUS_BITS.items() if status & bit]


@dataclass(frozen=True)
class JointHealth:
    position: float  # degrees; gripper 0-100
    load_pct: float  # signed percent of rated duty
    temperature_c: float
    voltage_v: float
    status: int  # raw Status register

    @property
    def faults(self) -> list[str]:
        return decode_status(self.status)


class ArmBus(Protocol):
    """One arm on one serial bus. Only the worker's bus thread calls these."""

    name: str
    role: str  # leader | follower

    def read_calibration(self) -> Calibration: ...
    def read_positions(self) -> dict[str, float]: ...
    def write_goals(self, goals: dict[str, float]) -> None: ...
    def set_torque(self, on: bool) -> None: ...
    def read_health(self) -> dict[str, JointHealth]: ...
    def close(self) -> None: ...


class Camera(Protocol):
    key: str

    def read_latest(self) -> tuple[np.ndarray, float, int]:
        """(RGB uint8 HxWx3, capture time on time.monotonic, sequence number)."""
        ...

    def close(self) -> None: ...

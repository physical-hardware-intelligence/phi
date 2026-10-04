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


def label(key: str) -> str:
    """How a key reads in a message: left_follower becomes Left Follower (the web app's
    lib/labels.ts uses the same rule). File names, ports and LeRobot ids stay literal."""
    return " ".join(w[:1].upper() + w[1:] for w in key.replace("_", " ").split())


def labels(keys: list[str]) -> str:
    """Left Follower, or Left Follower and Right Follower, or A, B and C."""
    out = [label(k) for k in keys]
    return " and ".join(out) if len(out) <= 2 else f"{', '.join(out[:-1])} and {out[-1]}"


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


# WHY both: LeRobot raises ConnectionError when a packet gets no answer and RuntimeError when the
# servo answers with an error status (motors_bus.py:965-970, 1056-1058), and pyserial raises
# SerialException, an IOError, when the port disappears (serialutil.py:92). Each one is a fault.
BUS_ERRORS: tuple[type[Exception], ...] = (OSError, RuntimeError)


class ArmBus(Protocol):
    """One arm on one serial bus. Only the worker's bus thread calls these. Every method may raise
    one of BUS_ERRORS."""

    name: str
    role: str  # leader | follower
    calibration_id: str  # the calibration file this arm is registered with

    def read_calibration(self) -> Calibration: ...
    def read_positions(self) -> dict[str, float]: ...
    # Calibration (LeRobot motors_bus.py:788-796): raw encoder ticks with homing applied, set
    # every joint's homing so its present position reads half a turn, and write a whole
    # calibration to the servos.
    def read_raw_positions(self) -> dict[str, int]: ...
    def set_half_turn_homings(self) -> dict[str, int]: ...
    def write_calibration(self, cal: Calibration) -> None: ...
    def write_goals(self, goals: dict[str, float]) -> None: ...
    def set_torque(self, on: bool) -> None: ...
    def read_torque(self) -> bool: ...
    def read_health(self) -> dict[str, JointHealth]: ...
    def close(self) -> None: ...


class Camera(Protocol):
    key: str

    def read_latest(self) -> tuple[np.ndarray, float, int]:
        """(RGB uint8 HxWx3, capture time on time.monotonic, sequence number)."""
        ...

    def close(self) -> None: ...

"""A rig with no hardware: arms and cameras that behave enough like an SO-101 setup to drive every
flow.

Behaviour, so tests and members without an arm see what a real rig would do:
  * a follower with torque on moves toward its goal at a capped joint speed. The goal register
    keeps its value with torque off, so enabling torque drives to whatever goal it holds. That is
    the worst case; whether an STS3215 does it is unverified (bench test 21), so the worker must
    write goal = present before enabling
  * a leader with torque off follows a slow scripted "hand"; a follower does too after
    inject("hand"), so calibration has a range to record
  * raw encoder ticks follow LeRobot's Feetech rule, Present_Position = Actual_Position -
    Homing_Offset (feetech.py:278-289), so the calibration steps run as they would on a servo
  * calibration registers are fixed per arm, and differ between arms, like real calibrations
  * faults can be injected: overload / overheat / voltage on one joint, unplug on a whole arm or
    camera
Numbers for load, temperature and voltage are illustrative, not measured on an SO-101.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from phi.studio.identity import TICKS_PER_REV, Calibration, JointCal
from phi.studio.rig import JOINTS, JointHealth

MAX_SPEED_DEG_S = 180.0
HALF_TURN = (TICKS_PER_REV - 1) // 2  # 2047, LeRobot's int(max_res / 2) (feetech.py:286-287)
TICKS_PER_DEG = TICKS_PER_REV / 360.0
_HOME = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -90.0,
    "elbow_flex": 90.0,
    "wrist_flex": 60.0,
    "wrist_roll": 0.0,
    "gripper": 5.0,
}
_FAULT_BIT = {"voltage": 1, "overheat": 4, "overload": 32}


class FakeClock:
    """Deterministic time for tests. Real runs use time.monotonic."""

    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _calibration(seed: int) -> Calibration:
    rng = np.random.default_rng(seed)
    return {
        j: JointCal(
            i + 1,
            0,
            int(rng.integers(-2000, 2000)),
            int(rng.integers(700, 1000)),
            int(rng.integers(3000, 3400)),
        )
        for i, j in enumerate(JOINTS)
    }


@dataclass
class MockArm:
    name: str
    role: str
    clock: object = field(default=time.monotonic)
    seed: int = 0
    pos: dict[str, float] = field(default_factory=lambda: dict(_HOME))
    goal: dict[str, float] = field(default_factory=lambda: dict(_HOME))
    torque: bool = False
    faults: dict[str, int] = field(default_factory=dict)
    unplugged: bool = False
    scripted: bool = True  # leader follows the scripted hand
    hand: bool = False  # a limp follower follows a wider scripted hand (calibration sweeps)
    calibration_id: str = ""  # the calibration file this arm is registered with; default its name

    def __post_init__(self) -> None:
        self.calibration_id = self.calibration_id or self.name
        self._own_cal = _calibration(self.seed)
        self._cal = self._own_cal  # what the port reaches; swap_cables changes it
        rng = np.random.default_rng(1000 + self.seed)
        # Encoder ticks at the home pose: where each servo's magnet happens to sit.
        self._enc0 = {j: int(rng.integers(1300, 2800)) for j in JOINTS}
        self._t = self.clock()  # type: ignore[operator]
        self._t0 = self._t

    # -- fault injection
    # -------------------------------------------------------------------------------------
    def inject(self, kind: str, joint: str | None = None) -> None:
        if kind == "unplug":
            self.unplugged = True
        elif kind == "replug":
            self.unplugged = False
            self._cal = self._own_cal  # replugged into the right port
        elif kind == "clear":
            self.faults.clear()
        elif kind == "hand":
            self.hand = not self.hand
        else:
            assert joint in JOINTS, f"joint required for {kind}"
            self.faults[joint] = self.faults.get(joint, 0) | _FAULT_BIT[kind]

    # -- ArmBus ------------------------------------------------------------------------------
    # ----------------
    def _check(self) -> None:
        if self.unplugged:
            raise ConnectionError("no status packet (port unplugged)")

    def _step(self) -> None:
        now = self.clock()  # type: ignore[operator]
        dt, self._t = now - self._t, now
        if self.role == "leader" and not self.torque:
            if not self.scripted:
                return
            s = now - self._t0
            for i, j in enumerate(JOINTS):
                amp = 30.0 if j != "gripper" else 20.0
                self.pos[j] = _HOME[j] + amp * math.sin(2 * math.pi * 0.25 * s + i)
            return
        if not self.torque:
            if self.hand:
                s = now - self._t0
                for i, j in enumerate(JOINTS):
                    amp = 70.0 if j != "gripper" else 45.0
                    self.pos[j] = _HOME[j] + amp * math.sin(2 * math.pi * 0.3 * s + i)
            return
        step = MAX_SPEED_DEG_S * dt
        for j in JOINTS:
            if self.faults.get(j, 0) & _FAULT_BIT["overload"]:
                continue  # protection: the joint stops tracking
            err = self.goal[j] - self.pos[j]
            self.pos[j] += max(-step, min(step, err))

    def read_calibration(self) -> Calibration:
        self._check()
        return dict(self._cal)

    def read_positions(self) -> dict[str, float]:
        self._check()
        self._step()
        return dict(self.pos)

    def _actual(self) -> dict[str, int]:
        """Encoder ticks before homing. The gripper's 0-100 is treated as degrees: a mock."""
        return {j: self._enc0[j] + round(self.pos[j] * TICKS_PER_DEG) for j in JOINTS}

    def read_raw_positions(self) -> dict[str, int]:
        self._check()
        self._step()
        return {j: v - self._cal[j].homing_offset for j, v in self._actual().items()}

    def set_half_turn_homings(self) -> dict[str, int]:
        """LeRobot motors_bus.py:788-796: reset (homing 0, limits 0..4095), read, write homing."""
        self._check()
        self._step()
        homings = {j: v - HALF_TURN for j, v in self._actual().items()}
        self._cal = self._own_cal = {
            j: JointCal(c.id, c.drive_mode, homings[j], 0, TICKS_PER_REV - 1)
            for j, c in self._cal.items()
        }  # the servo keeps these: a replug does not undo them
        return homings

    def write_calibration(self, cal: Calibration) -> None:
        """Writes the registers of the arm this port reaches. The mock assumes that is this arm:
        Studio refuses to calibrate an arm on a swapped cable."""
        self._check()
        self._cal = self._own_cal = dict(cal)

    def write_goals(self, goals: dict[str, float]) -> None:
        self._check()
        self._step()
        self.goal.update(goals)

    def set_torque(self, on: bool) -> None:
        self._check()
        self._step()
        self.torque = on

    def read_torque(self) -> bool:
        self._check()
        return self.torque

    def read_health(self) -> dict[str, JointHealth]:
        self._check()
        self._step()
        volts = 12.0 if self.role == "follower" else 5.0
        out = {}
        for j in JOINTS:
            bits = self.faults.get(j, 0)
            load = 0.0 if not self.torque else min(100.0, abs(self.goal[j] - self.pos[j]) * 2 + 8)
            if bits & _FAULT_BIT["overload"]:
                load = 95.0
            temp = 38.0 + (40.0 if bits & _FAULT_BIT["overheat"] else 0.0)
            v = volts * (0.7 if bits & _FAULT_BIT["voltage"] else 1.0)
            out[j] = JointHealth(self.pos[j], load, temp, v, bits)
        return out

    def close(self) -> None:
        self.torque = False


@dataclass
class MockCamera:
    key: str
    clock: object = field(default=time.monotonic)
    fps: int = 30
    width: int = 640
    height: int = 480
    hue: int = 0
    unplugged: bool = False

    def __post_init__(self) -> None:
        self._t0 = self.clock()  # type: ignore[operator]
        yy, xx = np.mgrid[0 : self.height, 0 : self.width]
        self._base = np.stack(
            [(xx * 255 // self.width), (yy * 255 // self.height), np.full_like(xx, self.hue % 256)],
            -1,
        ).astype(np.uint8)

    def inject(self, kind: str) -> None:
        self.unplugged = kind == "unplug"

    def read_latest(self) -> tuple[np.ndarray, float, int]:
        if self.unplugged:
            raise ConnectionError(f"camera {self.key}: no frame (device gone)")
        now = self.clock()  # type: ignore[operator]
        seq = int(round((now - self._t0) * self.fps))
        frame = self._base.copy()
        # A bar that sweeps across the frame, so a frozen feed is visible at a glance.
        x = (seq * 8) % self.width
        frame[:, x : x + 12] = 255
        return frame, now, seq

    def close(self) -> None:
        pass


@dataclass
class MockRig:
    arms: list[MockArm]
    cameras: list[MockCamera]

    def calibration_files(self) -> dict[str, Calibration]:
        """What a calibration directory would hold for these arms: each arm's own file."""
        return {a.name: a._own_cal for a in self.arms}

    def swap_cables(self, name: str) -> None:
        """Swap the USB cables of `name` and its partner (leader <-> follower): each port now
        reaches the other arm's servos. Only the calibration registers are swapped, which is all
        the identity check reads. Replugging either arm restores it."""
        other = name.replace("leader", "@").replace("follower", "leader").replace("@", "follower")
        a, b = (next(x for x in self.arms if x.name == n) for n in (name, other))
        a._cal, b._cal = b._own_cal, a._own_cal


def mock_rig(
    pairs: int = 1,
    cameras: tuple[str, ...] = ("front", "wrist", "top"),
    clock: object = time.monotonic,
) -> MockRig:
    """1 pair (leader, follower) or 2 pairs (left_*, right_*), the shapes LeRobot's so_* and bi_so_*
    classes use."""
    sides = [""] if pairs == 1 else ["left_", "right_"]
    arms = [
        MockArm(f"{s}{role}", role, clock=clock, seed=10 * k + r)
        for k, s in enumerate(sides)
        for r, role in enumerate(("leader", "follower"))
    ]
    cams = [MockCamera(c, clock=clock, hue=60 * i) for i, c in enumerate(cameras)]
    return MockRig(arms, cams)

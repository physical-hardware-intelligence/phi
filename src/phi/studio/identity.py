"""Which physical arm is on which port.

Two independent signals, because neither is enough alone:

  * USB adapter serial. Survives a change of socket, reboot, or Mac, where the
    /dev path does not (the 2026-08-06 port swap drove the leader as a follower).
    It names the adapter, not the arm, so a board moved to another arm lies.
  * Calibration registers. LeRobot writes Homing_Offset and Min/Max_Position_Limit
    into every servo at calibration and compares them with the file on connect
    (lerobot/motors/feetech/feetech.py:226-245). Read back with torque untouched,
    they name the calibration file the arm was last calibrated with.

Pure data in, data out: no serial I/O and no LeRobot import, so CI can test it.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import yaml

# WHY 4096: STS3215 resolution (lerobot/motors/feetech/tables.py:190). Every SO-101 servo is an
# STS3215.
TICKS_PER_REV = 4096
ROLES = ("leader", "follower")


class JointCal(NamedTuple):
    id: int
    drive_mode: int
    homing_offset: int
    range_min: int
    range_max: int


Calibration = dict[str, JointCal]


def load_calibration(src: Path | str | dict[str, Any]) -> Calibration:
    """A LeRobot calibration JSON (path or parsed dict) as {joint: JointCal}."""
    raw = src if isinstance(src, dict) else json.loads(Path(src).read_text())
    return {joint: JointCal(**{f: int(v[f]) for f in JointCal._fields}) for joint, v in raw.items()}


def save_calibration(cal: Calibration, path: Path) -> None:
    """Write `cal` as a LeRobot calibration JSON, the shape `load_calibration` reads. Atomic: a
    crash mid-write leaves the previous file whole."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({j: c._asdict() for j, c in cal.items()}, indent=4))
    os.replace(tmp, path)


@dataclass(frozen=True)
class Distance:
    per_joint_deg: dict[str, float]

    @property
    def max_deg(self) -> float:
        return max(self.per_joint_deg.values())

    @property
    def worst_joint(self) -> str:
        return max(self.per_joint_deg, key=self.per_joint_deg.__getitem__)

    @property
    def exact(self) -> bool:
        # WHY exact equality: LeRobot's own is_calibrated check is exact, so anything else
        # means LeRobot will
        # prompt to rewrite the servos on connect.
        return self.max_deg == 0


def fingerprint_distance(a: Calibration, b: Calibration) -> Distance:
    """Worst register difference per joint, in degrees of servo rotation."""
    if set(a) != set(b):
        missing = sorted(set(a) ^ set(b))
        raise ValueError(f"calibrations cover different joints: {', '.join(missing)}")
    deg = 360.0 / TICKS_PER_REV
    return Distance(
        {
            j: deg
            * max(
                abs(a[j].homing_offset - b[j].homing_offset),
                abs(a[j].range_min - b[j].range_min),
                abs(a[j].range_max - b[j].range_max),
            )
            for j in a
        }
    )


@dataclass(frozen=True)
class Match:
    name: str
    distance: Distance


def match_fingerprint(registers: Calibration, files: dict[str, Calibration]) -> list[Match]:
    """Every candidate file, nearest first. The caller shows the first and how far it is."""
    return sorted(
        (Match(n, fingerprint_distance(registers, c)) for n, c in files.items()),
        key=lambda m: m.distance.max_deg,
    )


@dataclass(frozen=True)
class ArmRecord:
    name: str
    role: str
    serial: str
    calibration: str  # path relative to the LeRobot calibration root


def load_registry(path: Path | str) -> list[ArmRecord]:
    """configs/studio/arms.yaml: the club's arms, keyed by USB adapter serial."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    records = [
        ArmRecord(str(a["name"]), str(a["role"]), str(a["serial"]), str(a["calibration"]))
        for a in raw.get("arms", [])
    ]
    seen: dict[str, str] = {}
    for r in records:
        if r.role not in ROLES:
            raise ValueError(f"arm {r.name}: role {r.role!r} is not one of {ROLES}")
        if r.serial in seen:
            raise ValueError(f"serial {r.serial} is listed for both {seen[r.serial]} and {r.name}")
        seen[r.serial] = r.name
    return records


@dataclass(frozen=True)
class PortInfo:
    device: str
    serial: str | None = None
    vid: int | None = None
    pid: int | None = None
    description: str = ""


@dataclass(frozen=True)
class ArmPresence:
    port: PortInfo
    record: ArmRecord | None  # None: a USB serial adapter not in the registry


def resolve_arms(records: list[ArmRecord], ports: list[PortInfo]) -> list[ArmPresence]:
    """USB serial adapters on this machine, each tied to its registry record if it has one."""
    by_serial = {r.serial: r for r in records}
    return [
        ArmPresence(p, by_serial.get(p.serial)) for p in ports if p.serial and p.vid is not None
    ]


def list_ports() -> list[PortInfo]:
    """The machine's serial ports via pyserial (no port is opened)."""
    from serial.tools import list_ports as lp

    return [
        PortInfo(p.device, p.serial_number, p.vid, p.pid, p.description or "")
        for p in lp.comports()
    ]

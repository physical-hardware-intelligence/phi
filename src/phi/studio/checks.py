"""Pre-flight checks: one click runs every read-only check Studio can make about this Mac, the rig
and itself, and says for each what it measured, whether that is fine, and what to do if not.

WHY pure functions over a snapshot: the server gathers the inputs (telemetry, identity, ports, the
Claude status), and every check is then a function of plain data, so each one is tested without
hardware. No check moves an arm or writes a register. The one write is a temporary file in the data
folder, deleted at once, to prove the folder is writable.
"""

from __future__ import annotations

import importlib.metadata as md
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from phi.studio.files import root_order
from phi.studio.identity import TICKS_PER_REV, load_calibration
from phi.studio.rig import JOINTS

# Thresholds. The UI colours the same values (JointTable.tsx, TopBar.tsx); keep them in step.
TEMP_WARN_C = 60.0
LOAD_WARN_PCT = 80.0
LOOP_TARGET_HZ = 30.0
LOOP_WARN_HZ = 27.0  # 10 % under target
LOOP_P99_WARN_MS = 1000.0 / LOOP_TARGET_HZ  # a cycle longer than one period drops a tick
CAMERA_STALE_S = 2.0
# [JUDGEMENT] Room for a recording session; not derived from a measured dataset size.
DISK_WARN_GB = 5.0
DISK_FAIL_GB = 1.0
# LeRobot 0.6.0 stores Homing_Offset sign-magnitude with the sign in bit 11
# (motors/feetech/tables.py:209), so a written offset fits in +-2047.
MAX_HOMING = 2047
MAX_TICK = TICKS_PER_REV - 1

GROUPS = ("This Mac", "Rig", "Studio")
STATUSES = ("pass", "info", "warn", "fail", "skip")


@dataclass
class Inputs:
    rig_kind: str = "mock"
    state: dict[str, Any] | None = None  # newest state message
    identity: list[dict[str, Any]] = field(default_factory=list)
    telemetry: dict[str, Any] | None = None
    cameras: dict[str, dict[str, Any]] = field(
        default_factory=dict
    )  # key -> online, message, fps, age_s
    ports: dict[str, Any] = field(default_factory=dict)  # files.list_ports()
    assistant: dict[str, Any] = field(default_factory=dict)  # ClaudeCLI.status()
    roots: dict[str, Path] = field(default_factory=dict)  # file roots by key
    rig_cal_dir: Path | None = None  # where this rig's calibration files live
    data_dir: Path | None = None
    code_root: Path | None = None
    static_dir: Path | None = None
    worker_alive: bool = True
    telemetry_age_s: float | None = None  # seconds since the newest telemetry; None: none yet


def result(
    id: str,
    group: str,
    title: str,
    status: str,
    detail: str,
    fix: str | None = None,
    file: dict[str, Any] | None = None,
) -> dict[str, Any]:
    assert group in GROUPS and status in STATUSES
    return {
        "id": id,
        "group": group,
        "title": title,
        "status": status,
        "detail": detail,
        "fix": fix,
        "file": file,
    }


# -- this Mac ----------------------------------------------------------------------
def check_packages(_: Inputs) -> dict[str, Any]:
    def ver(name: str) -> str | None:
        try:
            return md.version(name)
        except md.PackageNotFoundError:
            return None

    have = {n: ver(n) for n in ("lerobot", "pyserial", "aiohttp")}
    env = Path(os.environ.get("CONDA_PREFIX", "")).name or Path(sys.prefix).name
    found = ", ".join(f"{n} {v}" for n, v in have.items() if v)
    detail = f"Python {platform.python_version()} in {env}: {found}"
    if have["pyserial"] is None:
        return result(
            "packages",
            "This Mac",
            "Python packages",
            "fail",
            detail + ". pyserial is missing, so Studio cannot list serial ports.",
            "In the phi env: pip install pyserial",
        )
    if have["lerobot"] is None:
        return result(
            "packages",
            "This Mac",
            "Python packages",
            "warn",
            detail + ". LeRobot is missing; real arms need it.",
            "In the phi env: pip install lerobot",
        )
    return result("packages", "This Mac", "Python packages", "pass", detail)


@dataclass(frozen=True)
class ConfigArm:
    section: str  # robot, teleop, ...
    id: str
    port: str
    line: int  # 1-based line of the port value
    type: str | None = None  # LeRobot type alias, e.g. so101_follower

    @property
    def kind(self) -> str | None:
        """LeRobot's calibration subfolder for this arm: robots or teleoperators."""
        t = (self.type or "").lower()
        if t.endswith("_leader") or "teleop" in self.section:
            return "teleoperators"
        if t.endswith("_follower") or "robot" in self.section:
            return "robots"
        return None


def find_rig_config(inp: Inputs) -> tuple[str, Path] | None:
    for key in root_order("robot-config.yaml"):  # the same copy the Files page shows
        root = inp.roots.get(key)
        if root is not None and (root / "robot-config.yaml").is_file():
            return key, root / "robot-config.yaml"
    return None


def parse_rig_config(text: str) -> list[ConfigArm]:
    """Every top-level section with both an `id` and a `port`, with the port's line. Raises
    yaml.YAMLError on a malformed file."""
    node = yaml.compose(text)
    arms: list[ConfigArm] = []
    if not isinstance(node, yaml.MappingNode):
        return arms
    for key, value in node.value:
        if not isinstance(value, yaml.MappingNode):
            continue
        fields = {k.value: v for k, v in value.value if isinstance(k, yaml.ScalarNode)}
        rid, port, kind = fields.get("id"), fields.get("port"), fields.get("type")
        if isinstance(rid, yaml.ScalarNode) and isinstance(port, yaml.ScalarNode) and port.value:
            t = kind.value if isinstance(kind, yaml.ScalarNode) else None
            arms.append(ConfigArm(key.value, rid.value, port.value, port.start_mark.line + 1, t))
    return arms


def _config(inp: Inputs) -> tuple[dict[str, Any] | None, list[ConfigArm], str | None]:
    """(file ref, arms, parse error)."""
    found = find_rig_config(inp)
    if found is None:
        return None, [], None
    root, path = found
    ref = {"root": root, "path": "robot-config.yaml", "line": None}
    try:
        return ref, parse_rig_config(path.read_text()), None
    except yaml.YAMLError as e:
        mark = getattr(e, "problem_mark", None)
        line = mark.line + 1 if mark is not None else None
        return {**ref, "line": line}, [], f"line {line}: {getattr(e, 'problem', e)}"
    except OSError as e:
        return ref, [], str(e)


def check_rig_config(inp: Inputs) -> dict[str, Any]:
    ref, arms, err = _config(inp)
    title = "Rig config"
    if ref is None:
        return result(
            "rig_config",
            "This Mac",
            title,
            "warn",
            "No robot-config.yaml in the main checkout, so Studio cannot cross-check ports.",
            "Create robot-config.yaml in the main checkout with each arm's id and port.",
        )
    if err:
        return result(
            "rig_config",
            "This Mac",
            title,
            "fail",
            f"robot-config.yaml does not parse: {err}",
            "Fix the YAML at that line.",
            ref,
        )
    if not arms:
        return result(
            "rig_config",
            "This Mac",
            title,
            "warn",
            "robot-config.yaml lists no section with both an id and a port.",
            None,
            ref,
        )
    names = ", ".join(f"{a.id} ({a.section})" for a in arms)
    return result("rig_config", "This Mac", title, "pass", f"{len(arms)} arms: {names}", None, ref)


def _live(ports: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Every live port by both of its macOS names."""
    out = {}
    for p in ports.get("ports", []):
        out[p["device"]] = p
        if p.get("tty"):
            out[p["tty"]] = p
    return out


def _device(port: str) -> str:
    """macOS lists each serial device twice, /dev/tty.X and /dev/cu.X. One name for both."""
    return port.replace("/dev/cu.", "/dev/tty.", 1)


def check_config_ports(inp: Inputs) -> dict[str, Any]:
    title = "Config ports are plugged in"
    ref, arms, err = _config(inp)
    if ref is None or err or not arms:
        return result(
            "config_ports", "This Mac", title, "skip", "Needs a readable robot-config.yaml."
        )
    if inp.ports.get("error"):
        return result(
            "config_ports", "This Mac", title, "skip", inp.ports["error"], inp.ports.get("fix")
        )
    live = _live(inp.ports)
    seen: dict[str, tuple[str, str]] = {}  # device -> (arm id, port as written)
    for a in arms:
        dev = _device(a.port)
        if dev in seen:
            other, written = seen[dev]
            same = a.port if written == a.port else f"{written} and {a.port}, the same device"
            return result(
                "config_ports",
                "This Mac",
                title,
                "fail",
                f"{other} and {a.id} both name {same}.",
                "Each arm has its own USB adapter. Run lerobot-find-port for each arm and fix "
                "robot-config.yaml.",
                {**ref, "line": a.line},
            )
        seen[dev] = (a.id, a.port)
    missing = [a for a in arms if a.port not in live]
    listed = {a.port for a in arms}
    usb = sorted({p.get("tty") or p["device"] for p in inp.ports.get("ports", []) if p.get("usb")})
    others = [
        p for p in usb if p not in listed and p.replace("/dev/tty.", "/dev/cu.", 1) not in listed
    ]
    if missing:
        first = missing[0]
        what = "; ".join(f"{a.id}: {a.port}" for a in missing)
        extra = (
            f" Plugged in now: {', '.join(others)}."
            if others
            else " No other USB serial port is plugged in."
        )
        return result(
            "config_ports",
            "This Mac",
            title,
            "fail",
            f"Not plugged in: {what}.{extra}",
            "Plug in the arm's USB and power. If it is plugged in, macOS renamed the port: run "
            f"lerobot-find-port and update robot-config.yaml line {first.line}.",
            {**ref, "line": first.line},
        )
    detail = ", ".join(f"{a.id} on {a.port}" for a in arms)
    if others:
        return result(
            "config_ports",
            "This Mac",
            title,
            "info",
            f"{detail}. Also plugged in, not in the config: {', '.join(others)}.",
            None,
            ref,
        )
    return result("config_ports", "This Mac", title, "pass", detail, None, ref)


def _cal_dirs(inp: Inputs) -> list[Path]:
    out = [inp.rig_cal_dir] if inp.rig_cal_dir is not None else []
    if "calibration" in inp.roots and inp.roots["calibration"] not in out:
        out.append(inp.roots["calibration"])
    return out


def _ref(inp: Inputs, path: Path, line: int | None = None) -> dict[str, Any] | None:
    """A file reference the Files page can open, or None when no root holds `path`."""
    for key, root in inp.roots.items():
        if root == path or root in path.parents:
            return {"root": key, "path": path.relative_to(root).as_posix(), "line": line}
    return None


def calibration_problems(raw: Any) -> list[str]:
    """What is wrong with one parsed calibration file, in the terms LeRobot would refuse it."""
    try:
        cal = load_calibration(raw)
    except (TypeError, KeyError, ValueError, AttributeError) as e:
        return [f"not a LeRobot calibration ({e})"]
    out = []
    missing = [j for j in JOINTS if j not in cal]
    if missing:
        out.append(f"missing {', '.join(missing)}")
    for j, c in cal.items():
        if c.range_min == c.range_max:  # LeRobot raises on this (motors_bus.py:846)
            out.append(f"{j} did not move during calibration")
        elif c.range_min > c.range_max:
            out.append(f"{j} range is reversed")
        if not (0 <= c.range_min <= MAX_TICK and 0 <= c.range_max <= MAX_TICK):
            out.append(f"{j} range is outside 0 to {MAX_TICK}")
        if abs(c.homing_offset) > MAX_HOMING:
            out.append(f"{j} homing offset {c.homing_offset} does not fit the register")
    return out


ROLE_KIND = {"leader": "teleoperators", "follower": "robots"}


def _cal_targets(inp: Inputs, arms: list[ConfigArm]) -> dict[tuple[str, str | None], Path | None]:
    """(arm id, calibration subfolder) -> the exact file when known. One id can name a leader
    and a follower, which LeRobot keeps apart (calibration/robots/, calibration/teleoperators/).
    WHY not narrow by the config's `type`: LeRobot names the folder after the class
    (robots/so_follower) while the config uses an alias (so101_follower); robots/robot.py:49."""
    out: dict[tuple[str, str | None], Path | None] = {}
    if inp.rig_kind == "mock":
        # WHY not the mock's arms: the mock keeps its calibrations in memory and writes a file
        # only when one is calibrated in Studio, so a missing mock file is normal. Files it did
        # write are still checked.
        if inp.rig_cal_dir is not None and inp.rig_cal_dir.is_dir():
            for p in sorted(inp.rig_cal_dir.glob("*.json")):
                out[(p.stem, None)] = p
    else:
        for a in inp.identity:
            if a.get("expected"):
                out.setdefault((a["expected"], ROLE_KIND.get(a.get("role", ""))), None)
    for c in arms:
        out.setdefault((c.id, c.kind), None)
    return out


def check_calibrations(inp: Inputs) -> dict[str, Any]:
    title = "Calibration files"
    _, arms, _ = _config(inp)
    targets = _cal_targets(inp, arms)
    dirs = _cal_dirs(inp)
    if not targets:
        return result(
            "calibrations",
            "This Mac",
            title,
            "skip",
            "No arm ids yet. Connect the rig or add robot-config.yaml.",
        )
    if not dirs:
        return result(
            "calibrations",
            "This Mac",
            title,
            "warn",
            "No calibration folder exists yet.",
            "Calibrate each arm on the Calibrate page.",
        )
    every = [p for d in dirs for p in sorted(d.rglob("*.json"))]
    missing: list[str] = []
    bad: list[tuple[str, list[str], Path]] = []
    stale: list[tuple[str, Path, Path]] = []  # (id, a valid file, an invalid one with the same id)
    for (i, kind), exact in targets.items():
        files = (
            [exact]
            if exact
            else [p for p in every if p.stem == i and (kind is None or kind in p.parts)]
        )
        files = list(dict.fromkeys(files))
        if not files:
            missing.append(f"{i} ({kind[:-1]})" if kind else i)
            continue
        verdicts = []
        for f in files:
            try:
                verdicts.append((f, calibration_problems(json.loads(f.read_text()))))
            except (OSError, ValueError) as e:
                verdicts.append((f, [f"unreadable ({e})"]))
        good = [f for f, probs in verdicts if not probs]
        for f, probs in verdicts:
            if probs and good:
                stale.append((i, good[0], f))
            elif probs:
                bad.append((i, probs, f))
    if missing:
        return result(
            "calibrations",
            "This Mac",
            title,
            "fail",
            f"No calibration file for {', '.join(missing)}. Looked in "
            f"{', '.join(str(d) for d in dirs)}.",
            "Calibrate that arm on the Calibrate page, or check its id in robot-config.yaml.",
        )
    if bad:
        i, problems, path = bad[0]
        more = f" Also: {', '.join(b[0] for b in bad[1:])}." if len(bad) > 1 else ""
        return result(
            "calibrations",
            "This Mac",
            title,
            "fail",
            f"{i}: {'; '.join(problems)} ({path.parent.name}/{path.name}).{more}",
            "Calibrate that arm again on the Calibrate page and move every joint through its "
            "full range.",
            _ref(inp, path),
        )
    if stale:
        i, ok, old = stale[0]
        return result(
            "calibrations",
            "This Mac",
            title,
            "warn",
            f"{i} has a valid file in {ok.parent.name}/ and an invalid one in {old.parent.name}/. "
            "LeRobot reads the one in its class folder (so_follower or so_leader for an SO-101).",
            "Delete or rename the invalid file so nobody loads it by mistake.",
            _ref(inp, old),
        )
    ids = sorted({i for i, _ in targets})
    return result(
        "calibrations",
        "This Mac",
        title,
        "pass",
        f"{', '.join(ids)}: {len(JOINTS)} joints each, ranges and offsets valid",
    )


def check_disk(inp: Inputs) -> dict[str, Any]:
    title = "Data folder"
    d = inp.data_dir
    if d is None:
        return result("disk", "This Mac", title, "skip", "This Studio runs without a data folder.")
    try:
        d.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=d, prefix=".phi-check-"):
            pass
    except OSError as e:
        return result(
            "disk",
            "This Mac",
            title,
            "fail",
            f"{d} is not writable: {e}",
            "Fix the folder's permissions, or start Studio with another --data-dir.",
        )
    free = shutil.disk_usage(d).free / 1e9
    detail = f"{d}, {free:.1f} GB free"
    if free < DISK_FAIL_GB:
        return result(
            "disk",
            "This Mac",
            title,
            "fail",
            detail,
            "Free disk space before recording or evaluating.",
        )
    if free < DISK_WARN_GB:
        return result(
            "disk", "This Mac", title, "warn", detail, "Free some space before a recording session."
        )
    return result("disk", "This Mac", title, "pass", detail)


# -- rig ----------------------------------------------------------------------
SKIP_RIG = "Connect the rig first."
WORKER_DOWN = "The robot worker stopped, so every value Studio holds is stale. Restart Studio."
TELEMETRY_STALE_S = 2.0  # the worker sends telemetry at the loop rate, so 2 s of silence is a stall


def _rig_skip(inp: Inputs) -> str | None:
    """Why the rig checks cannot trust what Studio holds, or None when it is live.
    WHY: after the worker dies or stalls, the last state and telemetry stay in memory; checking
    them would pass a rig nobody is reading."""
    if not inp.worker_alive:
        return WORKER_DOWN
    s = (inp.state or {}).get("state")
    if not s or s == "DISCONNECTED" or not (inp.telemetry or {}).get("arms"):
        return SKIP_RIG
    if inp.telemetry_age_s is not None and inp.telemetry_age_s > TELEMETRY_STALE_S:
        return f"No telemetry for {inp.telemetry_age_s:.0f} s, so the last values are stale."
    return None


def _identity_skip(inp: Inputs) -> str | None:
    if not inp.worker_alive:
        return WORKER_DOWN
    return None if inp.identity else SKIP_RIG


def check_arms_answer(inp: Inputs) -> dict[str, Any]:
    title = "Every arm answers"
    if why := _rig_skip(inp):
        return result("arms_answer", "Rig", title, "skip", why)
    arms = inp.telemetry["arms"]
    dead = [n for n, a in arms.items() if not a.get("online")]
    if dead:
        return result(
            "arms_answer",
            "Rig",
            title,
            "fail",
            f"Not answering: {', '.join(dead)}.",
            "Check that arm's USB cable and its power supply, then clear the fault.",
        )
    return result(
        "arms_answer",
        "Rig",
        title,
        "pass",
        f"{len(arms)} of {len(arms)} arms answer on their buses",
    )


def check_identity(inp: Inputs) -> dict[str, Any]:
    title = "Each arm matches its calibration"
    if why := _identity_skip(inp):
        return result("identity", "Rig", title, "skip", why)
    bad = [a for a in inp.identity if not a.get("ok")]
    if bad:
        a = bad[0]
        got = (
            f"its registers match {a['match']}"
            if a.get("match")
            else "its registers match no calibration file"
        )
        gap = (
            f", {a['max_deg']:.1f} degrees off at {a['worst_joint']}"
            if a.get("max_deg") is not None
            else ""
        )
        return result(
            "identity",
            "Rig",
            title,
            "fail",
            f"{a['name']} on {a['port']} should be {a['expected']}, but {got}{gap}.",
            "If two arms are swapped, swap their USB cables or fix the ports in robot-config.yaml. "
            "Otherwise calibrate that arm again.",
        )
    worst = max((a.get("max_deg") or 0.0 for a in inp.identity), default=0.0)
    return result(
        "identity",
        "Rig",
        title,
        "pass",
        f"{len(inp.identity)} arms, largest register gap {worst:.1f} degrees",
    )


def check_ports_vs_identity(inp: Inputs) -> dict[str, Any]:
    title = "Config ports match the arms"
    if inp.rig_kind == "mock":
        return result("ports_identity", "Rig", title, "skip", "Mock rig: its ports are not real.")
    if why := _identity_skip(inp):
        return result("ports_identity", "Rig", title, "skip", why)
    ref, arms, err = _config(inp)
    if ref is None or err:
        return result("ports_identity", "Rig", title, "skip", "Needs a readable robot-config.yaml.")
    by_id = {a.get("expected"): a for a in inp.identity}
    for c in arms:
        a = by_id.get(c.id)
        if a is None:
            continue
        same = {a["port"], a["port"].replace("/dev/cu.", "/dev/tty.", 1)}
        if c.port not in same:
            return result(
                "ports_identity",
                "Rig",
                title,
                "fail",
                f"{c.id} answered on {a['port']}, but robot-config.yaml says {c.port}.",
                f"Update robot-config.yaml line {c.line} to {a['port']}, or swap the cables back.",
                {**ref, "line": c.line},
            )
    return result(
        "ports_identity",
        "Rig",
        title,
        "pass",
        "Every configured arm answered on its configured port",
        None,
        ref,
    )


def _health(inp: Inputs) -> list[tuple[str, str, dict[str, Any]]]:
    return [
        (n, j, h)
        for n, a in (inp.telemetry or {}).get("arms", {}).items()
        for j, h in (a.get("health") or {}).items()
    ]


def check_faults(inp: Inputs) -> dict[str, Any]:
    title = "No servo fault bits"
    if why := _rig_skip(inp):
        return result("faults", "Rig", title, "skip", why)
    hits = [f"{n} {j}: {', '.join(h['faults'])}" for n, j, h in _health(inp) if h.get("faults")]
    if hits:
        return result(
            "faults",
            "Rig",
            title,
            "fail",
            "; ".join(hits),
            "Free the joint, check the supply, let a hot servo cool, then clear the fault.",
        )
    return result(
        "faults", "Rig", title, "pass", f"{len(_health(inp))} servos report a clean status register"
    )


def check_temperature(inp: Inputs) -> dict[str, Any]:
    title = "Servo temperature"
    rows = _health(inp)
    if (why := _rig_skip(inp)) or not rows:
        return result("temperature", "Rig", title, "skip", why or SKIP_RIG)
    n, j, h = max(rows, key=lambda r: r[2]["temp"])
    detail = f"Hottest: {n} {j} at {h['temp']:.0f} °C"
    if h["temp"] >= TEMP_WARN_C:
        return result(
            "temperature",
            "Rig",
            title,
            "warn",
            f"{detail}, at or over {TEMP_WARN_C:.0f} °C.",
            "Let it cool with torque off, and check nothing is stalling the joint.",
        )
    return result("temperature", "Rig", title, "pass", detail)


def check_load(inp: Inputs) -> dict[str, Any]:
    title = "Servo load"
    rows = _health(inp)
    if (why := _rig_skip(inp)) or not rows:
        return result("load", "Rig", title, "skip", why or SKIP_RIG)
    n, j, h = max(rows, key=lambda r: abs(r[2]["load"]))
    detail = f"Highest: {n} {j} at {abs(h['load']):.0f} % of rated"
    if abs(h["load"]) > LOAD_WARN_PCT:
        return result(
            "load",
            "Rig",
            title,
            "warn",
            f"{detail}, over {LOAD_WARN_PCT:.0f} %.",
            "Something is pushing on that joint or the gripper is squeezing. Free it.",
        )
    return result("load", "Rig", title, "pass", detail)


def check_voltage(inp: Inputs) -> dict[str, Any]:
    title = "Supply voltage"
    if why := _rig_skip(inp):
        return result("voltage", "Rig", title, "skip", why)
    per: dict[str, list[float]] = {}
    for n, _, h in _health(inp):
        per.setdefault(n, []).append(h["volt"])
    if not per:
        return result("voltage", "Rig", title, "skip", SKIP_RIG)
    # WHY info, not pass/fail: the right voltage depends on which servo variant each arm uses, which
    # Studio does not know. Undervoltage itself shows up as a fault bit in the check above.
    detail = ", ".join(
        f"{n} {min(v):.1f} V" if max(v) - min(v) < 0.05 else f"{n} {min(v):.1f} to {max(v):.1f} V"
        for n, v in per.items()
    )
    return result("voltage", "Rig", title, "info", detail)


def check_leader_torque(inp: Inputs) -> dict[str, Any]:
    title = "Leaders move freely"
    if why := _rig_skip(inp):
        return result("leader_torque", "Rig", title, "skip", why)
    if (inp.state or {}).get("state") == "CALIBRATING":
        return result("leader_torque", "Rig", title, "skip", "Calibration is in progress.")
    held = [
        n for n, a in inp.telemetry["arms"].items() if a.get("role") == "leader" and a.get("torque")
    ]
    if held:
        return result(
            "leader_torque",
            "Rig",
            title,
            "warn",
            f"Torque is on: {', '.join(held)}.",
            "A leader with torque fights your hand. Press Torque off.",
        )
    return result("leader_torque", "Rig", title, "pass", "Every leader has torque off")


def check_loop(inp: Inputs) -> dict[str, Any]:
    title = "Control loop rate"
    loop = (inp.telemetry or {}).get("loop") or {}
    if (why := _rig_skip(inp)) or not loop.get("hz"):
        return result("loop", "Rig", title, "skip", why or SKIP_RIG)
    p99 = loop.get("p99_ms", 0.0)
    detail = f"{loop['hz']:.1f} Hz, p99 cycle {p99:.1f} ms (target {LOOP_TARGET_HZ:.0f} Hz)"
    if loop["hz"] < LOOP_WARN_HZ or loop.get("p99_ms", 0) > LOOP_P99_WARN_MS:
        return result(
            "loop",
            "Rig",
            title,
            "warn",
            detail,
            "Close other heavy apps. Too many cameras on one USB hub also slow the servo bus.",
        )
    return result("loop", "Rig", title, "pass", detail)


def check_cameras(inp: Inputs) -> dict[str, Any]:
    title = "Cameras stream"
    if not inp.worker_alive:
        return result("cameras", "Rig", title, "skip", WORKER_DOWN)
    if not inp.cameras:
        return result("cameras", "Rig", title, "skip", "No camera has reported yet.")
    down = [
        k
        for k, c in inp.cameras.items()
        if not c.get("online") or c.get("age_s") is None or c["age_s"] > CAMERA_STALE_S
    ]
    rates = ", ".join(
        f"{k} {c['fps']:.0f} fps" if c.get("fps") else f"{k} no frames"
        for k, c in sorted(inp.cameras.items())
    )
    if down:
        why = "; ".join(
            f"{k}: {inp.cameras[k].get('message') or 'no frame in the last 2 s'}" for k in down
        )
        return result(
            "cameras",
            "Rig",
            title,
            "fail",
            f"{why}. {rates}.",
            "Check the camera's USB cable. On macOS camera indices change between sessions: "
            "re-run lerobot-find-cameras opencv.",
        )
    return result("cameras", "Rig", title, "pass", rates)


# -- Studio ----------------------------------------------------------------------
def check_assistant(inp: Inputs) -> dict[str, Any]:
    title = "Claude assistant"
    a = inp.assistant
    ver = f"Claude Code {a['version']}. " if a.get("version") else ""
    if a.get("available"):
        how = f" ({a['method']})" if a.get("method") else ""
        return result("assistant", "Studio", title, "pass", f"{ver}Signed in{how}")
    return result(
        "assistant", "Studio", title, "warn", a.get("error") or "Not available.", a.get("fix")
    )


def stamp(t: float) -> str:
    return time.strftime("%d %b %H:%M", time.localtime(t))


def check_ui_build(inp: Inputs) -> dict[str, Any]:
    title = "Interface build"
    index = inp.static_dir / "index.html" if inp.static_dir else None
    if index is None or not index.is_file():
        return result(
            "ui_build",
            "Studio",
            title,
            "fail",
            "The interface is not built.",
            "Run make studio-web.",
        )
    built = index.stat().st_mtime
    src = inp.code_root / "studio" / "web" / "src" if inp.code_root else None
    if src is None or not src.is_dir():
        return result(
            "ui_build",
            "Studio",
            title,
            "pass",
            f"Built {stamp(built)}",
        )
    newest = max(src.rglob("*"), key=lambda p: p.stat().st_mtime if p.is_file() else 0.0)
    if newest.stat().st_mtime > built + 1:
        rel = newest.relative_to(inp.code_root).as_posix()
        return result(
            "ui_build",
            "Studio",
            title,
            "warn",
            f"The build is older than its source: {rel} changed after it.",
            "Run make studio-web, then reload this page.",
            {"root": "code", "path": rel, "line": None},
        )
    return result(
        "ui_build",
        "Studio",
        title,
        "pass",
        f"Built {stamp(built)}, newer than every source file",
    )


def check_code(inp: Inputs) -> dict[str, Any]:
    title = "Running code"
    root = inp.code_root
    if root is None:
        return result("code", "Studio", title, "skip", "Unknown.")

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True, timeout=3, check=True
        ).stdout.strip()

    try:
        branch, sha = git("rev-parse", "--abbrev-ref", "HEAD"), git("rev-parse", "--short", "HEAD")
        dirty = len([ln for ln in git("status", "--porcelain").splitlines() if ln.strip()])
    except (OSError, subprocess.SubprocessError):
        return result("code", "Studio", title, "info", f"{root}, not a git checkout")
    changes = (
        f", {dirty} uncommitted {'change' if dirty == 1 else 'changes'}" if dirty else ", clean"
    )
    return result("code", "Studio", title, "info", f"{branch} at {sha}{changes} ({root})")


CHECKS: tuple[Callable[[Inputs], dict[str, Any]], ...] = (
    check_packages,
    check_rig_config,
    check_config_ports,
    check_calibrations,
    check_disk,
    check_arms_answer,
    check_identity,
    check_ports_vs_identity,
    check_faults,
    check_temperature,
    check_load,
    check_voltage,
    check_leader_torque,
    check_loop,
    check_cameras,
    check_assistant,
    check_ui_build,
    check_code,
)


def run_checks(inp: Inputs) -> list[dict[str, Any]]:
    """Every check, in order. A check that raises is reported as failed, never hidden."""
    out = []
    for fn in CHECKS:
        try:
            out.append(fn(inp))
        except Exception as e:  # noqa: BLE001  WHY: one broken check must not hide the others
            name = fn.__name__.removeprefix("check_")
            out.append(
                result(
                    name,
                    "Studio",
                    f"Check {name}",
                    "fail",
                    f"The check itself failed: {e!r}",
                    "This is a Studio bug. Ask Claude, or report it with this text.",
                )
            )
    return out

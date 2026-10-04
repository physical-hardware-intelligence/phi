"""Arm identity: USB serial lookup and calibration-register fingerprints (SPEC ID-1, ID-2)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from phi.studio.identity import (
    ArmRecord,
    PortInfo,
    fingerprint_distance,
    load_calibration,
    load_registry,
    match_fingerprint,
    resolve_arms,
)

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


def cal(offset: int) -> dict:
    """A plausible SO-101 calibration, shifted by `offset` ticks on every register."""
    return {
        j: {
            "id": i + 1,
            "drive_mode": 0,
            "homing_offset": 100 * i + offset,
            "range_min": 800 + offset,
            "range_max": 3200 + offset,
        }
        for i, j in enumerate(JOINTS)
    }


def test_distance_is_zero_for_identical_registers() -> None:
    d = fingerprint_distance(load_calibration(cal(0)), load_calibration(cal(0)))
    assert d.max_deg == 0 and d.exact


def test_distance_reports_worst_joint_in_degrees() -> None:
    a, b = load_calibration(cal(0)), load_calibration(cal(0))
    b["wrist_roll"] = b["wrist_roll"]._replace(homing_offset=b["wrist_roll"].homing_offset + 1024)
    d = fingerprint_distance(a, b)
    assert d.worst_joint == "wrist_roll"
    assert d.max_deg == pytest.approx(90.0)  # 1024 of 4096 ticks
    assert not d.exact


def test_match_names_the_exact_file_first() -> None:
    files = {"phi_follower": load_calibration(cal(0)), "yash_follower": load_calibration(cal(300))}
    matches = match_fingerprint(load_calibration(cal(0)), files)
    assert matches[0].name == "phi_follower" and matches[0].distance.exact
    assert matches[1].name == "yash_follower" and not matches[1].distance.exact


def test_match_reports_nearest_when_nothing_is_exact() -> None:
    files = {"a": load_calibration(cal(0)), "b": load_calibration(cal(500))}
    matches = match_fingerprint(load_calibration(cal(20)), files)
    assert matches[0].name == "a" and not matches[0].distance.exact


def test_joint_sets_must_agree() -> None:
    partial = {k: v for k, v in cal(0).items() if k != "gripper"}
    with pytest.raises(ValueError, match="gripper"):
        fingerprint_distance(load_calibration(cal(0)), load_calibration(partial))


def test_load_calibration_reads_the_lerobot_file_format(tmp_path: Path) -> None:
    p = tmp_path / "x.json"
    p.write_text(json.dumps(cal(0)))
    c = load_calibration(p)
    assert set(c) == set(JOINTS) and c["gripper"].id == 6


REGISTRY = """
arms:
  - name: phi_follower
    role: follower
    serial: 5B7B009644
    calibration: robots/so_follower/phi_follower.json
  - name: phi_leader
    role: leader
    serial: 5B7B013931
    calibration: teleoperators/so_leader/phi_leader.json
"""


def test_registry_and_ports_resolve_by_serial_not_by_path(tmp_path: Path) -> None:
    reg = tmp_path / "arms.yaml"
    reg.write_text(REGISTRY)
    records = load_registry(reg)
    assert records[0] == ArmRecord(
        "phi_follower", "follower", "5B7B009644", "robots/so_follower/phi_follower.json"
    )
    # The leader is in the socket the follower used yesterday: the serial decides, not the path.
    ports = [
        PortInfo("/dev/cu.usbmodem5B7B0096441", serial="5B7B013931", vid=0x1A86, pid=0x55D3),
        PortInfo("/dev/cu.usbmodem5B7B0139311", serial="5B7B009644", vid=0x1A86, pid=0x55D3),
        PortInfo("/dev/cu.usbmodemFFFF1", serial="NOT_IN_REGISTRY", vid=0x1A86, pid=0x55D3),
        PortInfo("/dev/cu.Bluetooth-Incoming-Port", serial=None),
    ]
    found = {p.port.device: p for p in resolve_arms(records, ports)}
    assert found["/dev/cu.usbmodem5B7B0096441"].record.name == "phi_leader"
    assert found["/dev/cu.usbmodem5B7B0139311"].record.name == "phi_follower"
    assert found["/dev/cu.usbmodemFFFF1"].record is None  # shown as "unregistered", never guessed
    assert "/dev/cu.Bluetooth-Incoming-Port" not in found  # not a USB serial adapter


def test_registry_rejects_duplicate_serials(tmp_path: Path) -> None:
    reg = tmp_path / "arms.yaml"
    reg.write_text(REGISTRY.replace("5B7B013931", "5B7B009644"))
    with pytest.raises(ValueError, match="5B7B009644"):
        load_registry(reg)


def test_registry_rejects_unknown_role(tmp_path: Path) -> None:
    reg = tmp_path / "arms.yaml"
    reg.write_text(REGISTRY.replace("role: leader", "role: puppet"))
    with pytest.raises(ValueError, match="puppet"):
        load_registry(reg)

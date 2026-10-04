"""Pre-flight checks: each is a function of plain data, so each is tested without hardware."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from phi.studio import checks as C
from phi.studio.checks import Inputs, calibration_problems, parse_rig_config, run_checks

CONFIG = """\
machine: test-mac

robot:                      # the follower
  type: so101_follower
  id: phi_follower
  port: /dev/tty.usbmodemF

teleop:
  type: so101_leader
  id: phi_leader
  port: /dev/tty.usbmodemL
"""


def good_cal() -> dict:
    return {
        j: {
            "id": i + 1,
            "drive_mode": 0,
            "homing_offset": -100,
            "range_min": 800,
            "range_max": 3200,
        }
        for i, j in enumerate(C.JOINTS)
    }


def port(dev: str, usb: bool = True) -> dict:
    return {
        "device": dev.replace("/dev/tty.", "/dev/cu.", 1),
        "tty": dev if "tty." in dev else None,
        "usb": usb,
        "vid": 0x1A86 if usb else None,
        "pid": 0x55D3 if usb else None,
    }


def mac(tmp: Path, config: str | None = CONFIG, cals: dict | None = None) -> Inputs:
    repo, cal = tmp / "repo", tmp / "cal"
    repo.mkdir(parents=True)
    if config is not None:
        (repo / "robot-config.yaml").write_text(config)
    for name, body in (
        cals if cals is not None else {"phi_follower": good_cal(), "phi_leader": good_cal()}
    ).items():
        # LeRobot's layout: calibration/<robots|teleoperators>/<class name>/<id>.json
        sub = (
            ("teleoperators", "so_leader")
            if name.endswith("_leader")
            else ("robots", "so_follower")
        )
        p = cal.joinpath(*sub, f"{name}.json")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(body))
    return Inputs(
        roots={"repo": repo, "calibration": cal},
        ports={"ports": [port("/dev/tty.usbmodemF"), port("/dev/tty.usbmodemL")], "error": None},
    )


def connected(**arm_overrides: dict) -> Inputs:
    def arm(role: str, **kw) -> dict:
        h = {j: {"load": 10.0, "temp": 35.0, "volt": 12.0, "faults": []} for j in C.JOINTS}
        return {"role": role, "online": True, "torque": role == "follower", "health": h, **kw}

    arms = {"follower_1": arm("follower"), "leader_1": arm("leader")}
    for name, kw in arm_overrides.items():
        arms[name] = {**arms[name], **kw}
    return Inputs(
        state={"state": "READY"}, telemetry={"arms": arms, "loop": {"hz": 30.0, "p99_ms": 4.0}}
    )


def by_id(inp: Inputs) -> dict[str, dict]:
    return {r["id"]: r for r in run_checks(inp)}


# -- this Mac ----------------------------------------------------------------------
def test_rig_config_arms_and_port_lines() -> None:
    arms = parse_rig_config(CONFIG)
    assert [(a.section, a.id, a.port, a.line, a.kind) for a in arms] == [
        ("robot", "phi_follower", "/dev/tty.usbmodemF", 6, "robots"),
        ("teleop", "phi_leader", "/dev/tty.usbmodemL", 11, "teleoperators"),
    ]


def test_malformed_rig_config_fails_at_its_line(tmp_path: Path) -> None:
    r = C.check_rig_config(mac(tmp_path, config="robot:\n  id: a\n  port: [unclosed\n"))
    assert r["status"] == "fail" and r["file"]["path"] == "robot-config.yaml" and r["file"]["line"]


def test_missing_rig_config_warns(tmp_path: Path) -> None:
    assert C.check_rig_config(mac(tmp_path, config=None))["status"] == "warn"


def test_config_ports_pass_when_both_are_plugged_in(tmp_path: Path) -> None:
    r = C.check_config_ports(mac(tmp_path))
    assert r["status"] == "pass" and "phi_follower on /dev/tty.usbmodemF" in r["detail"]


def test_config_port_missing_names_the_line_and_what_is_plugged_in(tmp_path: Path) -> None:
    inp = mac(tmp_path)
    inp.ports = {"ports": [port("/dev/tty.usbmodemL"), port("/dev/tty.usbmodemNEW")], "error": None}
    r = C.check_config_ports(inp)
    assert r["status"] == "fail"
    assert (
        "phi_follower: /dev/tty.usbmodemF" in r["detail"] and "/dev/tty.usbmodemNEW" in r["detail"]
    )
    assert r["file"] == {"root": "repo", "path": "robot-config.yaml", "line": 6}
    assert "line 6" in r["fix"]


def test_config_naming_one_port_twice_fails(tmp_path: Path) -> None:
    r = C.check_config_ports(mac(tmp_path, config=CONFIG.replace("usbmodemL", "usbmodemF")))
    assert r["status"] == "fail" and "both name" in r["detail"]


def test_unlisted_usb_port_is_reported_not_failed(tmp_path: Path) -> None:
    inp = mac(tmp_path)
    inp.ports["ports"].append(port("/dev/tty.usbmodemX"))
    inp.ports["ports"].append(port("/dev/cu.Bluetooth-Incoming-Port", usb=False))
    r = C.check_config_ports(inp)
    assert r["status"] == "info" and "usbmodemX" in r["detail"] and "Bluetooth" not in r["detail"]


@pytest.mark.parametrize(
    "edit,problem",
    [
        ({"range_min": 2000, "range_max": 2000}, "did not move"),
        ({"range_min": 3000, "range_max": 1000}, "reversed"),
        ({"range_max": 5000}, "outside 0 to 4095"),
        ({"homing_offset": 2048}, "does not fit"),
    ],
)
def test_calibration_problems_in_lerobots_terms(edit: dict, problem: str) -> None:
    cal = good_cal()
    cal["elbow_flex"] = {**cal["elbow_flex"], **edit}
    found = calibration_problems(cal)
    assert len(found) == 1 and found[0].startswith("elbow_flex") and problem in found[0]


def test_calibration_problems_missing_joint_and_not_a_calibration() -> None:
    cal = good_cal()
    del cal["gripper"]
    assert calibration_problems(cal) == ["missing gripper"]
    assert calibration_problems({"x": 1})[0].startswith("not a LeRobot calibration")
    assert calibration_problems(good_cal()) == []


def test_calibration_files_pass_missing_and_bad(tmp_path: Path) -> None:
    assert C.check_calibrations(mac(tmp_path / "a"))["status"] == "pass"
    r = C.check_calibrations(mac(tmp_path / "b", cals={"phi_follower": good_cal()}))
    assert r["status"] == "fail" and "phi_leader" in r["detail"]
    bad = good_cal()
    bad["gripper"]["range_max"] = bad["gripper"]["range_min"]
    r = C.check_calibrations(
        mac(tmp_path / "c", cals={"phi_follower": good_cal(), "phi_leader": bad})
    )
    assert r["status"] == "fail" and "gripper did not move" in r["detail"]
    assert r["file"] == {
        "root": "calibration",
        "path": "teleoperators/so_leader/phi_leader.json",
        "line": None,
    }


def test_calibration_is_checked_per_kind_not_by_name(tmp_path: Path) -> None:
    broken = good_cal()
    broken["gripper"]["range_max"] = broken["gripper"]["range_min"]
    # A leader and a follower may share an id: each is checked, in its own folder.
    shared = CONFIG.replace("phi_follower", "arm").replace("phi_leader", "arm")
    inp = mac(tmp_path / "a", config=shared, cals={})
    for sub, body in (
        (("robots", "so_follower"), good_cal()),
        (("teleoperators", "so_leader"), broken),
    ):
        f = tmp_path.joinpath("a", "cal", *sub, "arm.json")
        f.parent.mkdir(parents=True)
        f.write_text(json.dumps(body))
    r = C.check_calibrations(inp)
    assert r["status"] == "fail" and r["file"]["path"] == "teleoperators/so_leader/arm.json"
    # A stale file from an older folder name next to a valid one: a warning naming the stale one.
    inp = mac(tmp_path / "b")
    old = tmp_path / "b" / "cal" / "robots" / "so100_follower" / "phi_follower.json"
    old.parent.mkdir(parents=True)
    old.write_text(json.dumps(broken))
    r = C.check_calibrations(inp)
    assert r["status"] == "warn" and "so100_follower" in r["detail"]
    assert r["file"]["path"] == "robots/so100_follower/phi_follower.json"


def test_tty_and_cu_names_of_one_port_are_a_duplicate(tmp_path: Path) -> None:
    cfg = CONFIG.replace("/dev/tty.usbmodemL", "/dev/cu.usbmodemF")
    r = C.check_config_ports(mac(tmp_path, config=cfg))
    assert r["status"] == "fail" and "the same device" in r["detail"] and r["file"]["line"] == 11


def test_checks_and_files_read_the_same_rig_config(tmp_path: Path) -> None:
    from phi.studio.files import Files, Root

    main, wt = tmp_path / "main", tmp_path / "wt"
    for d in (main, wt):
        d.mkdir()
        (d / "robot-config.yaml").write_text(CONFIG)
    f = Files([Root("code", "Studio code", wt), Root("repo", "Main checkout", main)])
    note = next(n for n in f.notes() if n["path"] == "robot-config.yaml")
    assert note["root"] == "repo" and f.locate("robot-config.yaml") == ("repo", "robot-config.yaml")
    assert C.find_rig_config(Inputs(roots={"code": wt, "repo": main})) == (
        "repo",
        main / "robot-config.yaml",
    )


def test_data_folder_free_space_and_writability(tmp_path: Path) -> None:
    r = C.check_disk(Inputs(data_dir=tmp_path))
    assert r["status"] in ("pass", "warn", "fail") and "GB free" in r["detail"]
    assert not list(tmp_path.iterdir())  # the probe file is gone
    if os.geteuid() != 0:
        locked = tmp_path / "locked"
        locked.mkdir(mode=0o500)
        try:
            assert C.check_disk(Inputs(data_dir=locked))["status"] == "fail"
        finally:
            locked.chmod(0o700)


# -- rig ---------------------------------------------------------------------------
def test_rig_checks_skip_until_connected() -> None:
    res = by_id(Inputs())
    for key in (
        "arms_answer",
        "identity",
        "faults",
        "temperature",
        "load",
        "voltage",
        "leader_torque",
        "loop",
    ):
        assert res[key]["status"] == "skip", key


def test_rig_checks_skip_when_the_worker_died_or_telemetry_stalled() -> None:
    dead = connected()
    dead.worker_alive = False
    dead.identity = [{"name": "f", "ok": True, "expected": "phi_follower", "max_deg": 0.1}]
    dead.cameras = {"top": {"online": True, "fps": 15.0, "age_s": 0.1}}
    res = by_id(dead)
    for key in ("arms_answer", "identity", "faults", "temperature", "load", "loop", "cameras"):
        assert res[key]["status"] == "skip" and "worker stopped" in res[key]["detail"], key
    stalled = connected()
    stalled.telemetry_age_s = 9.0
    res = by_id(stalled)
    assert res["faults"]["status"] == "skip" and "No telemetry for 9 s" in res["faults"]["detail"]
    live = connected()
    live.telemetry_age_s = 0.05
    assert by_id(live)["faults"]["status"] == "pass"


def test_a_healthy_connected_rig_passes() -> None:
    res = by_id(connected())
    for key in ("arms_answer", "faults", "temperature", "load", "leader_torque", "loop"):
        assert res[key]["status"] == "pass", (key, res[key])
    assert res["voltage"]["status"] == "info" and "12.0 V" in res["voltage"]["detail"]


def test_rig_problems_are_found() -> None:
    hot = {
        j: {
            "load": 95.0,
            "temp": 61.0,
            "volt": 11.0,
            "faults": ["overload"] if j == "gripper" else [],
        }
        for j in C.JOINTS
    }
    inp = connected(follower_1={"online": False, "health": hot}, leader_1={"torque": True})
    inp.telemetry["loop"] = {"hz": 22.0, "p99_ms": 40.0}
    res = by_id(inp)
    assert res["arms_answer"]["status"] == "fail" and "follower_1" in res["arms_answer"]["detail"]
    assert (
        res["faults"]["status"] == "fail"
        and "follower_1 gripper: overload" in res["faults"]["detail"]
    )
    assert res["temperature"]["status"] == "warn" and "61" in res["temperature"]["detail"]
    assert res["load"]["status"] == "warn" and "95" in res["load"]["detail"]
    assert res["leader_torque"]["status"] == "warn" and "leader_1" in res["leader_torque"]["detail"]
    assert res["loop"]["status"] == "warn"


def test_identity_mismatch_says_what_the_registers_match() -> None:
    inp = Inputs(
        identity=[
            {
                "name": "follower_1",
                "port": "/dev/tty.A",
                "expected": "phi_follower",
                "ok": False,
                "match": "phi_leader",
                "max_deg": 41.2,
                "worst_joint": "wrist_flex",
            }
        ]
    )
    r = C.check_identity(inp)
    assert (
        r["status"] == "fail"
        and "should be phi_follower" in r["detail"]
        and "phi_leader" in r["detail"]
    )


def test_swapped_ports_point_at_the_config_line(tmp_path: Path) -> None:
    inp = mac(tmp_path)
    inp.rig_kind = "hardware"
    inp.identity = [
        {"name": "f", "expected": "phi_follower", "port": "/dev/cu.usbmodemL", "ok": True},
        {"name": "l", "expected": "phi_leader", "port": "/dev/cu.usbmodemF", "ok": True},
    ]
    r = C.check_ports_vs_identity(inp)
    assert r["status"] == "fail" and r["file"]["line"] == 6 and "/dev/cu.usbmodemL" in r["fix"]
    inp.identity[0]["port"], inp.identity[1]["port"] = "/dev/cu.usbmodemF", "/dev/cu.usbmodemL"
    assert C.check_ports_vs_identity(inp)["status"] == "pass"
    inp.rig_kind = "mock"
    assert C.check_ports_vs_identity(inp)["status"] == "skip"


def test_cameras_stale_or_offline_fail() -> None:
    ok = {"online": True, "fps": 30.0, "age_s": 0.1}
    assert C.check_cameras(Inputs(cameras={"front": ok}))["status"] == "pass"
    r = C.check_cameras(Inputs(cameras={"front": ok, "top": {**ok, "age_s": 5.0}}))
    assert r["status"] == "fail" and r["detail"].startswith("top:")
    r = C.check_cameras(
        Inputs(
            cameras={
                "wrist": {"online": False, "message": "index 2 failed", "fps": None, "age_s": None}
            }
        )
    )
    assert r["status"] == "fail" and "index 2 failed" in r["detail"]


# -- Studio ------------------------------------------------------------------------
def test_stale_interface_build_names_the_newer_source(tmp_path: Path) -> None:
    static, src = tmp_path / "static", tmp_path / "studio" / "web" / "src"
    static.mkdir()
    src.mkdir(parents=True)
    (static / "index.html").write_text("<html>")
    old = time.time() - 60
    os.utime(static / "index.html", (old, old))
    (src / "App.tsx").write_text("x")
    r = C.check_ui_build(Inputs(static_dir=static, code_root=tmp_path))
    assert r["status"] == "warn" and "studio/web/src/App.tsx" in r["detail"]
    os.utime(static / "index.html")
    assert C.check_ui_build(Inputs(static_dir=static, code_root=tmp_path))["status"] == "pass"
    assert C.check_ui_build(Inputs(static_dir=tmp_path / "none"))["status"] == "fail"


def test_signed_out_assistant_warns_with_the_fix() -> None:
    status = {
        "available": False,
        "error": "Claude is not signed in on this Mac.",
        "fix": "claude auth login",
    }
    r = C.check_assistant(Inputs(assistant=status))
    assert r["status"] == "warn" and r["fix"] == "claude auth login"


def test_a_ready_assistant_names_its_version() -> None:
    status = {"available": True, "method": "claude.ai", "version": "2.1.290"}
    r = C.check_assistant(Inputs(assistant=status))
    assert r["status"] == "pass" and r["detail"].startswith("Claude Code 2.1.290.")


def test_a_check_that_raises_is_reported_not_hidden(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(_: Inputs) -> dict:
        raise KeyError("boom")

    monkeypatch.setattr(C, "CHECKS", (broken, C.check_assistant))
    res = run_checks(Inputs())
    assert res[0]["status"] == "fail" and "boom" in res[0]["detail"] and len(res) == 2


def test_every_check_runs_on_empty_inputs() -> None:
    res = run_checks(Inputs())
    assert len(res) == len(C.CHECKS) == 18
    assert len({r["id"] for r in res}) == 18
    assert not [r for r in res if "The check itself failed" in r["detail"]]


def test_mock_arms_need_no_file_but_a_written_mock_file_is_checked(tmp_path: Path) -> None:
    inp = mac(tmp_path)
    inp.identity = [{"expected": "left_follower"}, {"expected": "left_leader"}]
    inp.rig_cal_dir = tmp_path / "mock-calibration"
    inp.rig_cal_dir.mkdir()
    inp.roots["studio"] = tmp_path
    assert C.check_calibrations(inp)["status"] == "pass"  # mock: in memory, no file needed
    bad = good_cal()
    bad["wrist_roll"]["homing_offset"] = 3000
    (inp.rig_cal_dir / "left_follower.json").write_text(json.dumps(bad))
    r = C.check_calibrations(inp)
    assert r["status"] == "fail" and "left_follower: wrist_roll homing offset" in r["detail"]
    assert r["file"] == {
        "root": "studio",
        "path": "mock-calibration/left_follower.json",
        "line": None,
    }
    inp.rig_kind = "hardware"
    r = C.check_calibrations(inp)
    assert r["status"] == "fail" and "No calibration file for left_leader" in r["detail"]

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
    assert len(found) == 1 and found[0].startswith("Elbow Flex") and problem in found[0]


def test_calibration_problems_missing_joint_and_not_a_calibration() -> None:
    cal = good_cal()
    del cal["gripper"]
    assert calibration_problems(cal) == ["missing Gripper"]
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
    assert r["status"] == "fail" and "Gripper did not move" in r["detail"]
    assert r["fix"].startswith("Run that arm's Calibrate command on the LeRobot setup page again")
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


def test_a_file_lerobot_does_not_read_does_not_pass(tmp_path: Path) -> None:
    # LeRobot 0.6.0 reads only calibration/<kind>/<so_follower|so_leader>/<id>.json
    # (robot.py:49-51).
    # Valid files under an older class folder must fail, and the detail must say where they are.
    inp = mac(tmp_path, cals={})
    for sub, name in (
        (("robots", "so101_follower"), "phi_follower"),
        (("teleoperators", "so101_leader"), "phi_leader"),
    ):
        f = tmp_path.joinpath("cal", *sub, f"{name}.json")
        f.parent.mkdir(parents=True)
        f.write_text(json.dumps(good_cal()))
    r = C.check_calibrations(inp)
    assert r["status"] == "fail"
    assert "phi_follower" in r["detail"] and "so101_follower" in r["detail"]
    assert "does not read" in r["detail"]
    assert f"phi_follower.json into {tmp_path / 'cal' / 'robots' / 'so_follower'}" in r["fix"]
    assert "LeRobot setup page" in r["fix"] and "the Calibrate page" not in r["fix"]


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
    assert res["arms_answer"]["status"] == "fail" and "Follower 1" in res["arms_answer"]["detail"]
    assert (
        res["faults"]["status"] == "fail"
        and "Follower 1 Gripper: overload" in res["faults"]["detail"]
    )
    assert res["temperature"]["status"] == "warn" and "61" in res["temperature"]["detail"]
    assert res["load"]["status"] == "warn" and "95" in res["load"]["detail"]
    assert res["leader_torque"]["status"] == "warn" and "Leader 1" in res["leader_torque"]["detail"]
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
    assert r["status"] == "fail" and r["detail"].startswith("Top:")
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
    assert len(res) == len(C.CHECKS) == 21
    assert len({r["id"] for r in res}) == 21
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
    f = inp.rig_cal_dir / "robots" / "so_follower" / "left_follower.json"
    f.parent.mkdir(parents=True)
    f.write_text(json.dumps(bad))
    r = C.check_calibrations(inp)
    assert r["status"] == "fail" and "left_follower: Wrist Roll homing offset" in r["detail"]
    assert r["fix"].startswith("Calibrate that arm again on the Calibrate page")
    assert r["file"] == {
        "root": "studio",
        "path": "mock-calibration/robots/so_follower/left_follower.json",
        "line": None,
    }
    inp.rig_kind = "hardware"
    r = C.check_calibrations(inp)
    assert r["status"] == "fail" and "No calibration file for left_leader" in r["detail"]


# -- the config as LeRobot reads it --------------------------------------------------
BI = """\
robot:
  type: bi_so_follower
  id: bi_f
  left_arm_config: {port: /dev/a, use_degrees: false, max_relative_target: 10}
  right_arm_config: {port: /dev/b, use_degrees: false}
teleop:
  type: bi_so_leader
  id: bi_l
  left_arm_config: {port: /dev/c, use_degrees: false}
  right_arm_config: {port: /dev/d}
"""


def test_units_must_match_and_a_bimanual_leader_is_always_degrees(tmp_path: Path) -> None:
    assert C.check_units(mac(tmp_path / "a"))["status"] == "pass"
    single = CONFIG.replace("  id: phi_leader\n", "  id: phi_leader\n  use_degrees: false\n")
    r = C.check_units(mac(tmp_path / "b", config=single))
    assert (
        r["status"] == "fail" and "Follower reads degrees, Leader reads -100 to 100" in r["detail"]
    )
    # use_degrees: false on the bimanual leader is ignored by LeRobot 0.6.0 (bi_so_leader.py:42-52)
    r = C.check_units(mac(tmp_path / "c", config=BI))
    assert r["status"] == "fail" and "Left Follower reads -100 to 100" in r["detail"]
    assert "always reads degrees" in r["detail"]


def test_step_limit_warns_per_follower(tmp_path: Path) -> None:
    r = C.check_step_limit(mac(tmp_path / "a", config=BI))
    assert r["status"] == "warn" and "Right Follower" in r["detail"] and "Left" not in r["detail"]
    capped = CONFIG.replace(
        "  id: phi_follower\n", "  id: phi_follower\n  max_relative_target: 8\n"
    )
    assert C.check_step_limit(mac(tmp_path / "b", config=capped))["status"] == "pass"


def test_step_limit_lerobot_would_misuse_fails_with_the_arms_own_path(tmp_path: Path) -> None:
    for i, cap in enumerate(("0", "{shoulder_pan: 5}", "-2")):
        bi = BI.replace("max_relative_target: 10", f"max_relative_target: {cap}")
        r = C.check_step_limit(mac(tmp_path / str(i), config=bi))
        assert r["status"] == "fail" and r["detail"].startswith("Left Follower"), cap
        assert "robot.left_arm_config.max_relative_target" in r["fix"]
    r = C.check_step_limit(mac(tmp_path / "w", config=BI))
    assert "Set robot.right_arm_config.max_relative_target" in r["fix"]


def test_a_realsense_camera_is_picked_by_its_serial(tmp_path: Path) -> None:
    cams = "cameras:\n  d: {type: intelrealsense, serial_number_or_name: TBD}\n"
    r = C.check_camera_config(mac(tmp_path / "a", config=CONFIG + cams))
    assert r["status"] == "warn" and "lerobot-find-cameras intelrealsense" in r["fix"]
    assert "serial_number_or_name" in r["fix"]
    ok = cams.replace("TBD", "'123'")
    assert C.check_camera_config(mac(tmp_path / "b", config=CONFIG + ok))["status"] == "pass"


def test_a_calibration_dir_file_is_found_without_a_calibration_folder(tmp_path: Path) -> None:
    own = tmp_path / "own"
    own.mkdir()
    (own / "phi_follower.json").write_text(json.dumps(good_cal()))
    config = CONFIG
    for i in ("phi_follower", "phi_leader"):
        config = config.replace(f"  id: {i}\n", f"  id: {i}\n  calibration_dir: {own}\n")
    inp = mac(tmp_path / "m", config=config, cals={})
    inp = Inputs(roots={"repo": inp.roots["repo"]}, ports=inp.ports)  # no calibration folder at all
    r = C.check_calibrations(inp)
    # WHY this shape: the follower's file is read and passes; the leader's is missing, not
    # unreadable
    assert r["status"] == "fail" and r["detail"].startswith("No calibration file for phi_leader")
    assert str(own) in r["detail"]


def test_camera_indices_must_be_filled_in(tmp_path: Path) -> None:
    cams = (
        "cameras:\n  front: {type: opencv, index_or_path: TBD}\n"
        "  top: {type: opencv, index_or_path: 0}\n"
    )
    r = C.check_camera_config(mac(tmp_path / "a", config=CONFIG + cams))
    assert r["status"] == "warn" and "Front" in r["detail"] and "Top" not in r["detail"]
    ok = cams.replace("TBD", "1")
    r = C.check_camera_config(mac(tmp_path / "b", config=CONFIG + ok))
    assert r["status"] == "pass" and "observation.images.front" in r["detail"]
    assert C.check_camera_config(mac(tmp_path / "c"))["status"] == "skip"


def test_bimanual_config_names_lerobots_calibration_files(tmp_path: Path) -> None:
    r = C.check_calibrations(mac(tmp_path, config=BI))
    assert r["status"] == "fail"
    for f in ("bi_f_left (robot)", "bi_f_right (robot)", "bi_l_left (teleoperator)"):
        assert f in r["detail"]

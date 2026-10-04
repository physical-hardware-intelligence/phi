"""robot-config.yaml read the way LeRobot 0.6.0 reads its configs, checked against its classes."""

from __future__ import annotations

import importlib.util
import shlex
from pathlib import Path

import pytest

from phi.studio import rigspec

SINGLE = """\
robot:
  type: so101_follower
  id: phi_follower
  port: /dev/tty.usbmodemF
  cameras:
    front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}
teleop:
  type: so101_leader
  id: phi_leader
  port: /dev/tty.usbmodemL
"""

BIMANUAL = """\
robot:
  type: bi_so_follower
  id: bimanual_follower
  left_arm_config:
    port: /dev/tty.usbmodemFL
    cameras:
      wrist: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}
  right_arm_config:
    port: /dev/tty.usbmodemFR
  cameras:
    top: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}
teleop:
  type: bi_so_leader
  id: bimanual_leader
  left_arm_config:
    port: /dev/tty.usbmodemLL
  right_arm_config:
    port: /dev/tty.usbmodemLR
"""

# Every per-arm option LeRobot takes, and this Mac's real shape: a top-level cameras: block whose
# entries carry a `hardware:` note LeRobot does not know. CALDIR is swapped for a temp folder.
SINGLE_OPTS = """\
robot:
  type: so101_follower
  id: phi_follower
  port: /dev/tty.usbmodemF
  calibration_dir: CALDIR
  max_relative_target: 10
  use_degrees: false
  disable_torque_on_disconnect: false
teleop:
  type: so101_leader
  id: phi_leader
  port: /dev/tty.usbmodemL
  use_degrees: false
cameras:
  wrist: {hardware: Arducam, type: opencv, index_or_path: TBD, width: 640, height: 480, fps: 30}
  front: {hardware: C920, type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}
"""

CAPS = {"shoulder_pan": 8, "shoulder_lift": 8, "elbow_flex": 8, "wrist_flex": 8, "wrist_roll": 8,
        "gripper": 5}  # fmt: skip
BIMANUAL_OPTS = f"""\
robot:
  type: bi_so_follower
  id: bimanual_follower
  calibration_dir: CALDIR
  left_arm_config:
    port: /dev/tty.usbmodemFL
    max_relative_target: {CAPS}
    disable_torque_on_disconnect: false
  right_arm_config:
    port: /dev/tty.usbmodemFR
    max_relative_target: 8
teleop:
  type: bi_so_leader
  id: bimanual_leader
  left_arm_config:
    port: /dev/tty.usbmodemLL
  right_arm_config:
    port: /dev/tty.usbmodemLR
"""


def test_single_arm_ids_ports_and_pairs():
    s = rigspec.parse(SINGLE)
    assert not s.bimanual and s.problems == ()
    assert [(a.key, a.lerobot_id, a.port, a.port_line) for a in s.arms] == [
        ("follower", "phi_follower", "/dev/tty.usbmodemF", 4),
        ("leader", "phi_leader", "/dev/tty.usbmodemL", 10),
    ]
    assert [(lead.key, f.key) for lead, f in s.pairs()] == [("leader", "follower")]
    assert s.action_features()[0] == "shoulder_pan.pos" and len(s.action_features()) == 6


def test_bimanual_ids_take_the_side_as_a_suffix_and_features_as_a_prefix():
    s = rigspec.parse(BIMANUAL)
    assert s.bimanual and s.problems == ()
    ids = {a.key: (a.lerobot_id, a.port) for a in s.arms}
    assert ids == {
        "left_follower": ("bimanual_follower_left", "/dev/tty.usbmodemFL"),
        "right_follower": ("bimanual_follower_right", "/dev/tty.usbmodemFR"),
        "left_leader": ("bimanual_leader_left", "/dev/tty.usbmodemLL"),
        "right_leader": ("bimanual_leader_right", "/dev/tty.usbmodemLR"),
    }
    assert [(lead.key, f.key) for lead, f in s.pairs()] == [
        ("left_leader", "left_follower"),
        ("right_leader", "right_follower"),
    ]
    assert {c.feature for c in s.cameras} == {
        "observation.images.left_wrist",
        "observation.images.top",
    }
    assert s.action_features()[:1] + s.action_features()[6:7] == [
        "left_shoulder_pan.pos",
        "right_shoulder_pan.pos",
    ]


def test_calibration_paths_follow_lerobot_folders_and_the_dir_override(tmp_path):
    s = rigspec.parse(BIMANUAL)
    paths = {a.key: a.calibration_path(tmp_path) for a in s.arms}
    assert paths["left_follower"] == tmp_path / "robots/so_follower/bimanual_follower_left.json"
    assert paths["right_leader"] == tmp_path / "teleoperators/so_leader/bimanual_leader_right.json"
    own = rigspec.parse(
        SINGLE.replace(
            "  id: phi_follower\n", f"  id: phi_follower\n  calibration_dir: {tmp_path}/x\n"
        )
    )
    assert own.arms[0].calibration_path(tmp_path) == tmp_path / "x/phi_follower.json"


def test_problems_lerobot_would_hit():
    legacy = rigspec.parse(
        "robot:\n  type: bi_so100_follower\n  id: b\n"
        "  left_arm_port: /dev/a\n  right_arm_port: /dev/b\n"
    )
    assert [a.port for a in legacy.arms] == ["/dev/a", "/dev/b"]  # still shown
    assert "bi_so_follower" in legacy.problems[0]
    noid = rigspec.parse("robot:\n  type: so101_follower\n  port: /dev/a\n")
    assert "None.json" in noid.problems[0] and noid.arms[0].calibration_path(Path("/c")) is None
    mixed = rigspec.parse(
        BIMANUAL.split("teleop:")[0] + "teleop:\n  type: so101_leader\n  id: l\n  port: /dev/l\n"
    )
    assert any("bimanual" in p for p in mixed.problems)
    odd = rigspec.parse("robot:\n  type: koch_follower\n  id: k\n  port: /dev/k\n")
    assert "not a LeRobot 0.6.0 SO-100/SO-101 type" in odd.problems[0]


def test_placeholder_cameras_stay_out_of_commands():
    text = SINGLE.replace("index_or_path: 0", "index_or_path: TBD")
    spec = rigspec.parse(text)
    cmds = {c["id"]: c for c in spec.commands()}
    assert "--robot.cameras" not in cmds["teleoperate"]["cmd"]
    assert "--robot.cameras" not in cmds["record"]["cmd"]
    left_out = [c.feature for c in spec.cameras]
    assert left_out and all(f in cmds["record"]["why"] for f in left_out)
    assert "leaves out" not in rigspec.parse(SINGLE).commands()[-1]["why"]


def _args(text: str, cmd_id: str) -> list[str]:
    return shlex.split({c["id"]: c["cmd"] for c in rigspec.parse(text).commands()}[cmd_id])


def test_commands_carry_every_option_the_config_sets():
    args = _args(SINGLE_OPTS.replace("CALDIR", "/c/f"), "teleoperate")
    for a in ("--robot.max_relative_target=10", "--robot.use_degrees=false",
              "--robot.disable_torque_on_disconnect=false", "--robot.calibration_dir=/c/f",
              "--teleop.use_degrees=false"):  # fmt: skip
        assert a in args
    bi = _args(BIMANUAL_OPTS.replace("CALDIR", "/c/b"), "calibrate_robot")
    assert '--robot.left_arm_config.max_relative_target={"shoulder_pan":8,' in " ".join(bi)
    assert "--robot.right_arm_config.max_relative_target=8" in bi
    assert "--robot.left_arm_config.disable_torque_on_disconnect=false" in bi
    assert "--robot.calibration_dir=/c/b" in bi
    assert rigspec.parse(BIMANUAL_OPTS).problems == ()


def test_camera_commands_keep_only_lerobots_fields():
    s = rigspec.parse(SINGLE_OPTS)
    assert [c.key for c in s.cameras] == ["wrist", "front"] and s.problems == ()
    cams = [a for a in _args(SINGLE_OPTS, "teleoperate") if a.startswith("--robot.cameras=")]
    assert cams == ['--robot.cameras={"front":{"type":"opencv","index_or_path":1,"width":640,'
                    '"height":480,"fps":30}}']  # fmt: skip  # wrist is TBD; hardware is dropped
    rs = rigspec.parse(
        SINGLE.replace(
            "fps: 30}\n", "fps: 30}\n    d: {type: intelrealsense, serial_number_or_name: '123'}\n"
        )
    )
    assert [c.source for c in rs.cameras] == [0, "123"]
    odd = rigspec.parse(SINGLE.replace("type: opencv", "type: zed"))
    assert odd.cameras[0].source is None and "opencv and intelrealsense" in odd.problems[0]


def test_values_lerobot_would_refuse_become_problems_not_flags():
    bad_cap = {"0": "max_relative_target: 0", "keys": "max_relative_target: {shoulder_pan: 5}",
               "text": "max_relative_target: fast", "deg": "use_degrees: nope"}  # fmt: skip
    for why, line in bad_cap.items():
        s = rigspec.parse(SINGLE.replace("  id: phi_follower\n", f"  id: phi_follower\n  {line}\n"))
        assert len(s.problems) == 1, why
        assert "use_degrees=" not in " ".join(c["cmd"] for c in s.commands())
    zero = SINGLE.replace("  id: phi_follower\n", "  id: phi_follower\n  max_relative_target: 0\n")
    assert "holds the follower still" in rigspec.parse(zero).problems[0]
    listy = rigspec.parse(SINGLE.replace("id: phi_follower", "id: [a, b]"))
    assert listy.arms[0].lerobot_id is None and "None.json" in listy.problems[0]
    portless = rigspec.parse(SINGLE.replace("port: /dev/tty.usbmodemF", "port: {a: 1}"))
    assert portless.arms[0].port is None


def test_bimanual_leader_options_and_leader_cameras_are_flagged_not_passed():
    text = BIMANUAL.replace(
        "    port: /dev/tty.usbmodemLL\n",
        "    port: /dev/tty.usbmodemLL\n    use_degrees: false\n"
        "    cameras: {w: {type: opencv, index_or_path: 3}}\n",
    )
    s = rigspec.parse(text)
    assert any("use_degrees is ignored" in p for p in s.problems)
    assert any("a leader has no cameras" in p for p in s.problems)
    assert "use_degrees" not in " ".join(c["cmd"] for c in s.commands())
    assert len(s.cameras) == 2  # the leader's camera is not one of the robot's
    single = rigspec.parse(SINGLE + "  cameras: {w: {type: opencv, index_or_path: 3}}\n")
    assert any(p.startswith("teleop.cameras") for p in single.problems)


def test_tilde_calibration_dir_is_flagged_and_not_expanded():
    s = rigspec.parse(SINGLE.replace("  id: phi_follower\n",
                                     "  id: phi_follower\n  calibration_dir: ~/cal\n"))  # fmt: skip
    assert "starts with ~" in s.problems[0]
    assert s.arms[0].calibration_path(Path("/c")) == Path("~/cal/phi_follower.json")


def test_camera_name_on_both_levels_is_a_problem():
    s = rigspec.parse(BIMANUAL.replace("    top:", "    wrist:"))
    assert any("wrist" in p and "bi_so_follower refuses" in p for p in s.problems)


# -- against LeRobot itself ----------------------------------------------------------------------

# WHY a mark, not a module-level importorskip: that would skip the parse tests above too
# wherever LeRobot is not installed (CI installs only phi's own extras).
needs_lerobot = pytest.mark.skipif(
    importlib.util.find_spec("lerobot") is None, reason="LeRobot is not installed"
)


def _parse_cli(cmd: str):
    """Parse a generated command with the config class its LeRobot entry point uses."""
    import draccus
    from lerobot.scripts import (
        lerobot_calibrate,
        lerobot_record,
        lerobot_setup_motors,
        lerobot_teleoperate,
    )

    tool, *args = shlex.split(cmd)
    cls = {"lerobot-calibrate": lerobot_calibrate.CalibrateConfig,
           "lerobot-setup-motors": lerobot_setup_motors.SetupConfig,
           "lerobot-teleoperate": lerobot_teleoperate.TeleoperateConfig,
           "lerobot-record": lerobot_record.RecordConfig}[tool]  # fmt: skip
    return draccus.parse(config_class=cls, args=args)


ALL = pytest.mark.parametrize(
    "text",
    [SINGLE, BIMANUAL, SINGLE_OPTS, BIMANUAL_OPTS],
    ids=["single", "bimanual", "single-options", "bimanual-options"],
)


@needs_lerobot
@ALL
def test_every_generated_command_parses_with_lerobots_own_cli_config(text, tmp_path):
    from lerobot.scripts.lerobot_setup_motors import COMPATIBLE_DEVICES

    text = text.replace("CALDIR", str(tmp_path / "own"))
    for c in rigspec.parse(text).commands():
        if c["cmd"] == "lerobot-find-port":
            continue
        cfg = _parse_cli(c["cmd"])  # raises on any flag LeRobot does not know
        if c["id"].startswith("setup_motors"):
            assert cfg.device.type in COMPATIBLE_DEVICES


@needs_lerobot
@ALL
def test_ids_paths_and_features_match_lerobots_robot_objects(text, tmp_path, monkeypatch):
    import lerobot.robots.robot as robot_mod
    import lerobot.teleoperators.teleoperator as teleop_mod
    from lerobot.robots.utils import make_robot_from_config
    from lerobot.teleoperators.utils import make_teleoperator_from_config

    # WHY: LeRobot's constructors mkdir their calibration folder; keep that out of the real one.
    monkeypatch.setattr(robot_mod, "HF_LEROBOT_CALIBRATION", tmp_path)
    monkeypatch.setattr(teleop_mod, "HF_LEROBOT_CALIBRATION", tmp_path)
    spec = rigspec.parse(text.replace("CALDIR", str(tmp_path / "own")))
    cmds = {c["id"]: c["cmd"] for c in spec.commands()}
    # WHY drop cameras: not what this test checks, and a camera config can touch devices.
    cmd = shlex.join(
        a for a in shlex.split(cmds["teleoperate"]) if not a.startswith("--robot.cameras")
    )
    cfg = _parse_cli(cmd)
    robot, teleop = make_robot_from_config(cfg.robot), make_teleoperator_from_config(cfg.teleop)
    by_key = {a.key: a for a in spec.arms}
    if spec.bimanual:
        found = {"left_follower": robot.left_arm, "right_follower": robot.right_arm,
                 "left_leader": teleop.left_arm, "right_leader": teleop.right_arm}  # fmt: skip
    else:
        found = {"follower": robot, "leader": teleop}
    for key, dev in found.items():
        assert dev.id == by_key[key].lerobot_id
        assert dev.calibration_fpath == by_key[key].calibration_path(tmp_path)
        # The options reach the arm object LeRobot builds, as robot-config.yaml wrote them.
        want = {"use_degrees": True, "disable_torque_on_disconnect": True,
                "max_relative_target": None, **dict(by_key[key].options)}  # fmt: skip
        for k, v in want.items():
            if hasattr(dev.config, k):
                assert getattr(dev.config, k) == v, (key, k)
    assert list(robot.action_features) == spec.action_features()

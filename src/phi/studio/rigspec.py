"""The rig as LeRobot sees it: types, ids, calibration paths, feature names and CLI commands.

The one module in Studio that knows LeRobot's naming rules. Every rule cites the installed
LeRobot 0.6.0 source, and tests/test_studio_rigspec.py checks each one against LeRobot's own
classes, so a LeRobot upgrade that changes a rule fails a test instead of misnaming a file.

robot-config.yaml uses LeRobot's own CLI keys: `robot:` and `teleop:` sections whose fields are
what `--robot.<field>` and `--teleop.<field>` take. A bimanual section nests each arm's port the
way LeRobot does (`left_arm_config: {port: ...}`, config_bi_so_follower.py:30-31).
"""

from __future__ import annotations

import json
import shlex
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from phi.studio.rig import JOINTS

# Registered choice names in LeRobot 0.6.0 (config_so_follower.py:45-46, config_so_leader.py:33-34,
# config_bi_so_follower.py:25, config_bi_so_leader.py:23).
SINGLE_FOLLOWER = ("so101_follower", "so100_follower")
SINGLE_LEADER = ("so101_leader", "so100_leader")
BI_FOLLOWER = ("bi_so_follower",)
BI_LEADER = ("bi_so_leader",)
# Bimanual names from older LeRobot releases, with a flat left_arm_port / right_arm_port. Studio
# reads them so an old config still shows its arms, and a check says what 0.6.0 calls them.
LEGACY_BI = {"bi_so100_follower": "bi_so_follower", "bi_so100_leader": "bi_so_leader"}

# The calibration subfolder is the class's `name`, not the CLI alias (so_follower.py:44,
# so_leader.py:37; robot.py:49-53, teleoperator.py:47-53).
FOLDER = {"follower": ("robots", "so_follower"), "leader": ("teleoperators", "so_leader")}
SIDES = ("left", "right")
# lerobot-setup-motors accepts only these device types (lerobot_setup_motors.py:46-66), so a
# bimanual rig sets up its motors one arm at a time under the single-arm type.
SETUP_MOTORS_TYPES = SINGLE_FOLLOWER + SINGLE_LEADER
# Per-arm options each config class takes besides port (config_so_follower.py:28-41,
# config_so_leader.py:24-27). A bi_so_leader passes only id, calibration_dir and port to its arms
# (bi_so_leader.py:42-52), so its arms take none.
ARM_OPTIONS = {
    "follower": ("max_relative_target", "use_degrees", "disable_torque_on_disconnect"),
    "leader": ("use_degrees",),
}
# The fields each camera config takes (cameras/configs.py:61-63, configuration_opencv.py:61-66,
# configuration_realsense.py:58-63), and the one that picks the device. WHY a list: draccus refuses
# any other key, so a note such as `hardware:` in robot-config.yaml must stay out of the commands.
CAMERA_FIELDS = {
    "opencv": ("type", "fps", "width", "height", "index_or_path", "color_mode", "rotation",
               "warmup_s", "fourcc", "backend"),
    "intelrealsense": ("type", "fps", "width", "height", "serial_number_or_name", "color_mode",
                       "use_rgb", "use_depth", "rotation", "warmup_s"),
}  # fmt: skip
CAMERA_SOURCE = {"opencv": "index_or_path", "intelrealsense": "serial_number_or_name"}


@dataclass(frozen=True)
class ArmSpec:
    """One physical arm."""

    key: str  # Studio's name for it: follower, leader, left_follower, right_leader
    role: str  # leader | follower
    side: str | None  # left | right on a bimanual rig
    lerobot_id: str | None  # the id LeRobot gives this arm; None if the config has no id
    port: str | None
    port_line: int | None  # 1-based line of the port in robot-config.yaml
    section: str  # robot | teleop
    type: str  # the section's type, e.g. so101_follower or bi_so_follower
    single_type: str  # this arm alone, for lerobot-setup-motors: so101_follower or so101_leader
    calibration_dir: str | None = None  # the config's calibration_dir override, if any
    use_degrees: bool = True  # False: body joints read -100..100 instead of degrees
    max_relative_target: Any = None  # follower step limit; None means LeRobot does not clip
    options: tuple[tuple[str, Any], ...] = ()  # ARM_OPTIONS as written, passed on to LeRobot

    @property
    def kind(self) -> str:
        return FOLDER[self.role][0]

    @property
    def folder(self) -> str:
        return FOLDER[self.role][1]

    def calibration_path(self, root: Path) -> Path | None:
        """Where LeRobot reads and writes this arm's calibration. `root` is HF_LEROBOT_CALIBRATION.
        A calibration_dir in the config replaces the whole folder, kind and class included
        (robot.py:49-51); a bimanual config passes its own to both arms (bi_so_follower.py:57).
        WHY no expanduser: LeRobot uses the path as written, so a ~ is a folder named ~."""
        if self.lerobot_id is None:
            return None
        if self.calibration_dir:
            return Path(self.calibration_dir) / f"{self.lerobot_id}.json"
        return root / self.kind / self.folder / f"{self.lerobot_id}.json"


@dataclass(frozen=True)
class CameraSpec:
    key: str  # the key under cameras: in the config
    side: str | None  # left | right if declared on one arm of a bimanual rig
    fields: dict[str, Any] = field(default_factory=dict)

    @property
    def source(self) -> Any:
        """What picks the device (an opencv index or path, a RealSense serial), or None when it
        is missing, a TBD placeholder, or the type is one Studio does not build commands for."""
        v = self.fields.get(CAMERA_SOURCE.get(str(self.fields.get("type")), ""))
        return None if v in (None, "", "TBD") else v

    def lerobot_fields(self) -> dict[str, Any]:
        """The fields LeRobot's camera config takes; notes and unknown keys stay behind."""
        keep = CAMERA_FIELDS.get(str(self.fields.get("type")), ())
        return {k: v for k, v in self.fields.items() if k in keep}

    @property
    def feature(self) -> str:
        """The dataset key: per-arm cameras of a bimanual robot get the side as a prefix, top-level
        ones keep their key (bi_so_follower.py:91-98)."""
        return (
            f"observation.images.{self.side}_{self.key}"
            if self.side
            else f"observation.images.{self.key}"
        )


@dataclass(frozen=True)
class RigSpec:
    arms: tuple[ArmSpec, ...]
    cameras: tuple[CameraSpec, ...]
    robot: dict[str, Any]  # the robot section as written
    teleop: dict[str, Any]
    problems: tuple[str, ...] = ()  # what LeRobot 0.6.0 would refuse or misread

    @property
    def bimanual(self) -> bool:
        return any(a.side for a in self.arms)

    def pairs(self) -> list[tuple[ArmSpec, ArmSpec]]:
        """(leader, follower) by side; LeRobot pairs them the same way, left with left."""
        out = []
        for f in (a for a in self.arms if a.role == "follower"):
            lead = next((a for a in self.arms if a.role == "leader" and a.side == f.side), None)
            if lead is not None:
                out.append((lead, f))
        return out

    def unit_mismatches(self) -> list[tuple[ArmSpec, ArmSpec]]:
        """Pairs whose leader and follower read joints in different units. Teleop passes the
        leader's numbers to the follower unchanged (identity processors, processor/factory.py:28-47;
        lerobot_teleoperate.py:201-210), so -100..100 read as degrees drives the wrong angle."""
        return [(lead, f) for lead, f in self.pairs() if lead.use_degrees != f.use_degrees]

    def action_features(self) -> list[str]:
        """`<motor>.pos` per follower joint; bimanual ones carry the side
        (bi_so_follower.py:81-89)."""
        followers = [a for a in self.arms if a.role == "follower"]
        return [f"{a.side}_{j}.pos" if a.side else f"{j}.pos" for a in followers for j in JOINTS]

    def commands(self) -> list[dict[str, str]]:
        """The LeRobot CLI commands for this rig, in the order a new rig needs them."""
        return lerobot_commands(self)


def _scalar(node: Any) -> Any:
    return yaml.safe_load(yaml.serialize(node)) if node is not None else None


def _fields(node: Any) -> dict[str, Any]:
    if not isinstance(node, yaml.MappingNode):
        return {}
    return {k.value: v for k, v in node.value if isinstance(k, yaml.ScalarNode)}


def _line(node: Any) -> int | None:
    return node.start_mark.line + 1 if node is not None else None


def _text(v: Any) -> str | None:
    """A port, id or path as text; None for a missing value or one that is not a scalar."""
    if v in (None, "") or isinstance(v, (dict, list, bool)):
        return None
    return str(v)


def step_limit_issue(v: Any) -> str | None:
    """Why LeRobot would misuse this max_relative_target, or None when it is fine or unset.
    so_follower.py:221-232 clips each goal to present +/- the limit (robots/utils.py:90-110)."""
    if v is None:
        return None
    if isinstance(v, dict):
        if set(v) != set(JOINTS):
            return (f"lists {', '.join(map(str, v)) or 'no joints'}; LeRobot needs exactly the six "
                    f"motors or it raises on every action (robots/utils.py:99-100)")  # fmt: skip
        vals = list(v.values())
    elif isinstance(v, bool) or not isinstance(v, (int, float)):
        return "is not a number or a per-joint mapping"
    else:
        vals = [v]
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in vals):
        return "has a value that is not a number"
    if any(x <= 0 for x in vals):
        return "is 0 or below, which holds the follower still: LeRobot clips every step to it"
    return None


def parse(text: str) -> RigSpec:
    """Read robot-config.yaml. Raises yaml.YAMLError on a malformed file; anything well-formed but
    wrong for LeRobot (types, missing ids, bad values) is reported in `problems`, never raised."""
    root = _fields(yaml.compose(text))
    arms: list[ArmSpec] = []
    cams: list[CameraSpec] = []
    problems: list[str] = []
    sections: dict[str, dict[str, Any]] = {}

    def add_cameras(where: str, node: Any, side: str | None) -> None:
        for key, cam in _fields(node).items():
            val = _scalar(cam)
            if not isinstance(val, dict):
                problems.append(f"{where}.{key} is not a mapping of camera settings.")
            elif str(val.get("type")) not in CAMERA_FIELDS:
                kinds = " and ".join(CAMERA_FIELDS)
                problems.append(f"{where}.{key} has type {val.get('type')!r}; Studio builds "
                                f"commands for {kinds} cameras only.")  # fmt: skip
                cams.append(CameraSpec(key, side, val))
            else:
                cams.append(CameraSpec(key, side, val))

    def options(src: dict[str, Any], role: str, where: str) -> tuple[tuple[str, Any], ...]:
        out = []
        for k in (k for k in ARM_OPTIONS[role] if k in src):
            v = _scalar(src[k])
            if k == "max_relative_target" and (why := step_limit_issue(v)):
                problems.append(f"{where}.max_relative_target {why}.")
            elif k != "max_relative_target" and not isinstance(v, bool):
                problems.append(f"{where}.{k} is {v!r}; LeRobot takes true or false.")
                continue  # WHY drop it: draccus would refuse the whole command over it
            out.append((k, v))
        return tuple(out)

    for section, role in (("robot", "follower"), ("teleop", "leader")):
        f = _fields(root.get(section))
        if not f:
            continue
        sections[section] = {k: _scalar(v) for k, v in f.items()}
        typ = str(_scalar(f.get("type")) or "")
        rid = _text(_scalar(f.get("id")))
        raw_dir = _scalar(f.get("calibration_dir"))
        cal_dir = _text(raw_dir)
        if raw_dir not in (None, "") and cal_dir is None:
            problems.append(f"{section}.calibration_dir is not a path.")
        if cal_dir and cal_dir.startswith("~"):
            problems.append(f"{section}.calibration_dir starts with ~, which LeRobot does "
                            "not expand: it makes a folder named ~. Write the full "
                            "path.")  # fmt: skip
        single = SINGLE_FOLLOWER if role == "follower" else SINGLE_LEADER
        bi = BI_FOLLOWER if role == "follower" else BI_LEADER
        if typ in LEGACY_BI:
            problems.append(
                f"{section}.type {typ} is from an older LeRobot. 0.6.0 calls it "
                f"{LEGACY_BI[typ]} and takes {section}.left_arm_config.port."
            )
        if typ in bi or typ in LEGACY_BI:
            single_type = single[0]
            for side in SIDES:
                sub = _fields(f.get(f"{side}_arm_config"))
                port_node = sub.get("port") if sub else f.get(f"{side}_arm_port")
                where = f"{section}.{side}_arm_config"
                if role == "leader":
                    # WHY leaders are always degrees and take no options: 0.6.0's bi_so_leader
                    # builds each arm from id, calibration_dir and port (bi_so_leader.py:42-52).
                    if "use_degrees" in sub:
                        problems.append(f"{where}.use_degrees is ignored: bi_so_leader always "
                                        "reads degrees (bi_so_leader.py:42-52).")  # fmt: skip
                    if "cameras" in sub:
                        problems.append(f"{where}.cameras: a leader has no cameras in "
                                        "LeRobot (config_so_leader.py:24-27); put them under "
                                        "robot.")  # fmt: skip
                    opts: tuple[tuple[str, Any], ...] = ()
                    deg = True
                else:
                    opts = options(sub, role, where)
                    deg = _scalar(sub.get("use_degrees")) is not False
                    add_cameras(f"{where}.cameras", sub.get("cameras"), side)
                arms.append(ArmSpec(
                    key=f"{side}_{role}", role=role, side=side,
                    lerobot_id=f"{rid}_{side}" if rid else None,  # bi_so_follower.py:56, 66
                    port=_text(_scalar(port_node)), port_line=_line(port_node), section=section,
                    type=typ, single_type=single_type, calibration_dir=cal_dir, use_degrees=deg,
                    max_relative_target=dict(opts).get("max_relative_target"), options=opts,
                ))  # fmt: skip
        else:
            if typ and typ not in single:
                problems.append(
                    f"{section}.type {typ!r} is not a LeRobot 0.6.0 SO-100/SO-101 type "
                    f"({', '.join(single + bi)})."
                )
            port_node = f.get("port")
            opts = options(f, role, section)
            arms.append(ArmSpec(
                key=role, role=role, side=None, lerobot_id=rid, port=_text(_scalar(port_node)),
                port_line=_line(port_node), section=section, type=typ or single[0],
                single_type=typ if typ in single else single[0], calibration_dir=cal_dir,
                use_degrees=_scalar(f.get("use_degrees")) is not False,
                max_relative_target=dict(opts).get("max_relative_target"), options=opts,
            ))  # fmt: skip
            if role == "leader" and "cameras" in f:
                problems.append("teleop.cameras: a leader has no cameras in LeRobot "
                                "(config_so_leader.py:24-27); put them under robot.")  # fmt: skip
        if role == "follower":
            add_cameras(f"{section}.cameras", f.get("cameras"), None)
        if rid is None:
            problems.append(
                f"{section} has no id. LeRobot would save its calibration as None.json."
            )
    # Studio's own top-level cameras: block, for configs that keep cameras outside robot:.
    if not any(c.side is None for c in cams):
        add_cameras("cameras", root.get("cameras"), None)
    top = {c.key for c in cams if c.side is None}
    clash = sorted(top & {c.key for c in cams if c.side})
    if clash:
        problems.append(f"Camera names {', '.join(clash)} are used both at the top and on an arm; "
                        "bi_so_follower refuses that (bi_so_follower.py:46-52).")  # fmt: skip
    sides = {a.side for a in arms}
    if None in sides and len(sides) > 1:
        problems.append(
            "One section is bimanual and the other is single-arm; LeRobot's teleoperate "
            "pairs a bi_so_leader with a bi_so_follower."
        )
    robot, teleop = sections.get("robot", {}), sections.get("teleop", {})
    return RigSpec(tuple(arms), tuple(cams), robot, teleop, tuple(problems))


# -- commands ------------------------------------------------------------------------------------


def _cli_value(v: Any) -> str:
    """A value as draccus reads it from the command line: JSON for mappings, lowercase bools."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (dict, list)):
        return json.dumps(v, separators=(",", ":"))
    return str(v)


def _device_args(prefix: str, arms: list[ArmSpec], sec: dict[str, Any]) -> list[str]:
    """--robot.* or --teleop.* flags for one section, as LeRobot's draccus parser takes them:
    type, ports, id, calibration_dir, and each arm's options (step limit, units, torque on exit).
    WHY every option: a step limit in robot-config.yaml that the command leaves out is a step
    limit LeRobot never applies."""
    a0 = arms[0]
    args = [f"--{prefix}.type={LEGACY_BI.get(a0.type, a0.type)}"]
    for a in arms:
        at = f"{prefix}.{a.side}_arm_config" if a.side else prefix
        args.append(f"--{at}.port={a.port or '<port>'}")
        args += [f"--{at}.{k}={_cli_value(v)}" for k, v in a.options if v is not None]
    if a0.lerobot_id:
        args.append(f"--{prefix}.id={_text(sec.get('id'))}")
    if a0.calibration_dir:
        args.append(f"--{prefix}.calibration_dir={a0.calibration_dir}")
    return args


def _cameras_args(spec: RigSpec) -> list[str]:
    """--robot.cameras, plus --robot.<side>_arm_config.cameras for a bimanual arm's own cameras."""
    out = []
    for side in (None, *SIDES):
        cams = {c.key: c.lerobot_fields() for c in spec.cameras if c.side == side and _usable(c)}
        if cams:
            flag = f"--robot.{side}_arm_config.cameras" if side else "--robot.cameras"
            out.append(f"{flag}={json.dumps(cams, separators=(',', ':'))}")
    return out


def _usable(cam: CameraSpec) -> bool:
    """A camera LeRobot can open: a known type with its device filled in, not a placeholder."""
    return cam.source is not None


def _cmd(id_: str, title: str, why: str, tool: str, args: Sequence[str] = ()) -> dict[str, str]:
    return {"id": id_, "title": title, "why": why,
            "cmd": " ".join([tool, *(shlex.quote(a) for a in args)])}  # fmt: skip


def lerobot_commands(spec: RigSpec) -> list[dict[str, str]]:
    robots = [a for a in spec.arms if a.section == "robot"]
    teleops = [a for a in spec.arms if a.section == "teleop"]
    out = [_cmd("find_port", "Find each arm's port",
                "Unplug one arm when it asks; the port that disappears is that arm's.",
                "lerobot-find-port")]  # fmt: skip
    for a in robots + teleops:
        dev = "robot" if a.section == "robot" else "teleop"
        why = "Once per new arm: connect one motor at a time when asked (ids 1 to 6)."
        if a.side:
            why += " lerobot-setup-motors refuses bimanual types, so each arm runs as a single arm."
        args = [f"--{dev}.type={a.single_type}", f"--{dev}.port={a.port or '<port>'}"]
        title = f"Set motor ids, {a.key.replace('_', ' ')}"
        out.append(_cmd(f"setup_motors_{a.key}", title, why, "lerobot-setup-motors", args))
    for dev, arms, sec in (("robot", robots, spec.robot), ("teleop", teleops, spec.teleop)):
        if arms:
            noun = ("follower" if dev == "robot" else "leader") + ("s" if len(arms) > 1 else "")
            files = ", ".join(f"{a.lerobot_id}.json" for a in arms if a.lerobot_id)
            out.append(_cmd(f"calibrate_{dev}", f"Calibrate the {noun}", f"Writes {files}.",
                            "lerobot-calibrate", _device_args(dev, arms, sec)))  # fmt: skip
    if robots and teleops:
        args = (
            _device_args("robot", robots, spec.robot)
            + _cameras_args(spec)
            + _device_args("teleop", teleops, spec.teleop)
        )
        out.append(_cmd("teleoperate", "Teleoperate",
                        "LeRobot's default rate is 60 fps (lerobot_teleoperate.py:140).",
                        "lerobot-teleoperate", [*args, "--display_data=true"]))  # fmt: skip
        rec = [*args, "--dataset.repo_id=<hf_user>/<dataset>", "--dataset.single_task=<task>",
               "--dataset.num_episodes=10", "--display_data=true"]  # fmt: skip
        # WHY: a camera left out of --robot.cameras is a dataset with no frames from it, so the
        # command that writes the dataset says so, not only the Dataset keys panel.
        left_out = [c.feature for c in spec.cameras if not _usable(c)]
        gap = (f" It leaves out {', '.join(left_out)}: no usable device in robot-config.yaml, "
               "so the dataset gets no frames from " + ("it." if len(left_out) == 1 else "them.")
               if left_out else "")  # fmt: skip
        # keyboard_input.py:160-170; push_to_hub defaults to true (configs/dataset.py:43)
        out.append(_cmd("record", "Record a dataset",
                        "Right arrow ends an episode early, left arrow records it again, Esc "
                        "stops. It uploads to the Hub at the end unless "
                        "--dataset.push_to_hub=false." + gap,
                        "lerobot-record", rec))  # fmt: skip
    return out

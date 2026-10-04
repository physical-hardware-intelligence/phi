"""Read-only file view: the files a person needs when something goes wrong (this Mac's rig config
with its ports, LeRobot calibration files, Studio's data, the code), and the live serial-port list.

WHY fixed roots plus a deny list: the page and the assistant can open any file under a root, so the
roots bound what can leave this Mac, and the deny list keeps secrets out even inside a root.
"""

from __future__ import annotations

import fnmatch
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MAX_READ = 512 * 1024  # bytes shown in the viewer; the rest is cut with a note
MAX_SEARCH_FILE = 1024 * 1024
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".mypy_cache", ".pytest_cache",
             ".ruff_cache", ".ipynb_checkpoints", "dist", "build", "static"}  # fmt: skip
# Heavy folders at the top of a repo (datasets, checkpoints, run outputs): skipped by search only
# there, so a code package that happens to be called `data` is still searched.
SKIP_TOP = {"datasets", "data", "models", "outputs", "checkpoints", "wandb", "env", "trim_upload"}
SECRET = (".env", ".env.*", "*.env", "*token*", "*secret*", "*credential*", "*password*", "*.pem",
          "*.key", "*.p12", "id_rsa*", "id_ed25519*", ".netrc", ".ssh", ".git")  # fmt: skip
# Files that answer "which port, which arm, which camera" on this machine, when a root has them.
# Files git excludes, so the main checkout holds the real copy; any worktree copy is a leftover.
MACHINE_FILES = ("robot-config.yaml",)


def root_order(rel: str) -> tuple[str, ...]:
    """Which roots to try first for a relative path. Docs: the running code's copy."""
    return ("repo", "code") if rel in MACHINE_FILES else ("code", "repo")


NOTES = (
    ("robot-config.yaml", "Rig config for this Mac"),
    ("configs/ports.local.sh", "Ports captured by capture-ports"),
    ("configs/camera-map.log", "Camera index log"),
    ("docs/robots/so-arm101/troubleshooting.md", "SO-101 troubleshooting notes"),
    ("docs/robots/so-arm101/02-setup.md", "SO-101 setup notes"),
)


class FileError(ValueError):
    pass


@dataclass(frozen=True)
class Root:
    key: str
    label: str
    path: Path

    def public(self) -> dict[str, str]:
        return {"key": self.key, "label": self.label, "path": str(self.path)}


def is_secret(name: str) -> bool:
    n = name.lower()
    return any(fnmatch.fnmatch(n, pat) for pat in SECRET)


def main_checkout(code: Path) -> Path | None:
    """The main working tree of the git repo `code` belongs to. Machine files that git excludes,
    such as robot-config.yaml, exist only there, not in a worktree."""
    try:
        out = subprocess.run(
            ["git", "-C", str(code), "rev-parse", "--path-format=absolute", "--git-common-dir"],
            capture_output=True, text=True, timeout=3, check=True,
        ).stdout.strip()  # fmt: skip
    except (OSError, subprocess.SubprocessError):
        return None
    common = Path(out)
    return common.parent if common.name == ".git" else None


def lerobot_calibration_dir() -> Path:
    """Where LeRobot keeps calibration files, resolved the way lerobot/utils/constants.py does."""
    if os.getenv("HF_LEROBOT_CALIBRATION"):
        return Path(os.environ["HF_LEROBOT_CALIBRATION"]).expanduser()
    hf_home = Path(os.getenv("HF_HOME", "~/.cache/huggingface")).expanduser()
    return Path(os.getenv("HF_LEROBOT_HOME", hf_home / "lerobot")).expanduser() / "calibration"


def default_roots(code: Path, data_dir: Path | None) -> list[Root]:
    roots = [Root("code", "Studio code", code.resolve())]
    main = main_checkout(code)
    if main is not None and main.resolve() != code.resolve():
        roots.append(Root("repo", "Main checkout", main.resolve()))
    cal = lerobot_calibration_dir()
    if cal.is_dir():
        roots.append(Root("calibration", "LeRobot calibration", cal.resolve()))
    if data_dir is not None:
        roots.append(Root("studio", "Studio data", Path(data_dir).expanduser().resolve()))
    return roots


class Files:
    def __init__(self, roots: list[Root]) -> None:
        self.roots = {r.key: r for r in roots}

    # -- paths ------------------------------------------------------------------------------------
    def resolve(self, root: Any, rel: Any) -> tuple[Root, Path]:
        r = self.roots.get(root) if isinstance(root, str) else None
        if r is None:
            raise FileError(f"There is no file root named {root!r}.")
        if not isinstance(rel, str) or not rel or len(rel) > 1000 or "\0" in rel:
            raise FileError("Give a file path inside the root.")
        p = (r.path / rel).resolve()  # WHY resolve first: a symlink must not lead out of the root
        if p != r.path and r.path not in p.parents:
            raise FileError("That path is outside the folders Studio can show.")
        if any(is_secret(part) for part in p.relative_to(r.path).parts):
            raise FileError("Studio does not show files that may hold secrets.")
        return r, p

    def locate(self, path: Any) -> tuple[str, str] | None:
        """(root, relative path) for an absolute path, or a path relative to some root."""
        if not isinstance(path, str) or not path:
            return None
        p = Path(path).expanduser()
        first = () if p.is_absolute() else root_order(p.as_posix())
        for key in dict.fromkeys([*first, *self.roots]):
            r = self.roots.get(key)
            if r is None:
                continue
            cand = p if p.is_absolute() else r.path / p
            try:
                rel = cand.resolve().relative_to(r.path)
            except (ValueError, OSError):
                continue
            if (r.path / rel).is_file():
                return r.key, rel.as_posix()
        return None

    # -- reading ----------------------------------------------------------------------------------
    def read(self, root: Any, rel: Any) -> dict[str, Any]:
        r, p = self.resolve(root, rel)
        if not p.is_file():
            raise FileError(f"{rel} is not a file in {r.label}.")
        st = p.stat()
        with p.open("rb") as fh:
            raw = fh.read(MAX_READ)
        if b"\0" in raw[:8192]:
            raise FileError(f"{rel} is a binary file.")
        return {"root": r.key, "path": p.relative_to(r.path).as_posix(), "abs": str(p),
                "text": raw.decode("utf-8", errors="replace"), "size": st.st_size,
                "mtime": st.st_mtime, "truncated": st.st_size > MAX_READ}  # fmt: skip

    def search(self, query: Any, limit: int = 200, budget_s: float = 2.0) -> dict[str, Any]:
        """Case-insensitive literal search over the text files of every root."""
        if not isinstance(query, str) or not 2 <= len(query.strip()) <= 200:
            raise FileError("Search for 2 to 200 characters.")
        q = query.strip().lower()
        hits: list[dict[str, Any]] = []
        scanned, end, stopped = 0, time.monotonic() + budget_s, False
        for r in self.roots.values():
            for dirpath, dirnames, filenames in os.walk(r.path):
                top = Path(dirpath) == r.path
                dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not is_secret(d)
                               and not (top and d in SKIP_TOP)]  # fmt: skip
                for name in filenames:
                    if len(hits) >= limit or time.monotonic() > end:
                        stopped = True
                        break
                    if is_secret(name):
                        continue
                    p = Path(dirpath) / name
                    try:
                        if p.stat().st_size > MAX_SEARCH_FILE:
                            continue
                        raw = p.read_bytes()
                    except OSError:
                        continue
                    if b"\0" in raw[:8192]:
                        continue
                    scanned += 1
                    text = raw.decode("utf-8", errors="replace")
                    if q not in text.lower():
                        continue
                    rel = p.relative_to(r.path).as_posix()
                    for i, line in enumerate(text.splitlines(), 1):
                        if q in line.lower():
                            hits.append({"root": r.key, "path": rel, "line": i,
                                         "text": line.strip()[:200]})  # fmt: skip
                            if len(hits) >= limit:
                                break
                if stopped:
                    break
            if stopped:
                break
        return {"query": query, "hits": hits, "scanned": scanned, "stopped": stopped}

    # -- what to show first -----------------------------------------------------------------------
    def notes(self) -> list[dict[str, Any]]:
        out = []
        for rel, label in NOTES:  # in NOTES order: the rig config first
            for key in root_order(rel):
                r = self.roots.get(key)
                if r is not None and (r.path / rel).is_file():
                    out.append({"root": key, "path": rel, "label": label,
                                "mtime": (r.path / rel).stat().st_mtime})  # fmt: skip
                    break
        return out

    def calibrations(self) -> list[dict[str, Any]]:
        r = self.roots.get("calibration")
        if r is None:
            return []
        out = []
        for p in sorted(r.path.rglob("*.json")):
            if is_secret(p.name):
                continue
            out.append({"root": r.key, "path": p.relative_to(r.path).as_posix(), "id": p.stem,
                        "kind": p.parent.name, "mtime": p.stat().st_mtime})  # fmt: skip
        return out

    def lerobot(self) -> dict[str, Any] | None:
        """robot-config.yaml as LeRobot reads it: each arm's id, port and calibration file, the
        dataset feature names, what LeRobot would refuse, and the CLI commands for this rig."""
        import yaml

        from phi.studio import rigspec

        for key in root_order("robot-config.yaml"):
            r = self.roots.get(key)
            if r is None or not (r.path / "robot-config.yaml").is_file():
                continue
            ref = {"root": key, "path": "robot-config.yaml"}
            try:
                spec = rigspec.parse((r.path / "robot-config.yaml").read_text())
            except (yaml.YAMLError, OSError) as e:
                return {"file": ref, "error": str(e).splitlines()[0]}
            cal = self.roots.get("calibration")
            arms = []
            for a in spec.arms:
                path = a.calibration_path(cal.path) if cal else None
                rel = (
                    path.relative_to(cal.path).as_posix()
                    if cal and path and cal.path in path.parents
                    else None
                )
                arms.append({"key": a.key, "role": a.role, "side": a.side, "type": a.type,
                             "id": a.lerobot_id, "port": a.port, "line": a.port_line,
                             "calibration": rel, "calibrated": bool(path and path.is_file()),
                             "use_degrees": a.use_degrees,
                             "max_relative_target": a.max_relative_target})  # fmt: skip
            return {"file": ref, "bimanual": spec.bimanual, "arms": arms,
                    "cameras": [{"key": c.key, "side": c.side, "feature": c.feature,
                                 "index": c.fields.get("index_or_path")} for c in spec.cameras],
                    "features": spec.action_features(), "problems": list(spec.problems),
                    "commands": spec.commands()}  # fmt: skip
        return None

    def index(self) -> dict[str, Any]:
        return {"roots": [r.public() for r in self.roots.values()], "notes": self.notes(),
                "calibrations": self.calibrations(), "lerobot": self.lerobot()}  # fmt: skip


def list_ports(identity: list[dict[str, Any]]) -> dict[str, Any]:
    """Serial ports on this machine, with the arm Studio identified on each. A read-only probe."""
    try:
        from serial.tools import list_ports as lp
    except ImportError:
        return {"ports": [], "error": "pyserial is not installed in Studio's environment.",
                "fix": "Install it in the phi env: pip install pyserial"}  # fmt: skip
    by_port = {a.get("port"): a.get("name") for a in identity}
    rows = []
    for p in lp.comports():
        # macOS lists each USB serial device as /dev/cu.X; LeRobot configs name /dev/tty.X.
        tty = p.device.replace("/dev/cu.", "/dev/tty.", 1)
        rows.append({"device": p.device, "tty": tty if tty != p.device else None,
                     "description": p.description if p.description != "n/a" else None,
                     "vid": p.vid, "pid": p.pid, "serial": p.serial_number,
                     "manufacturer": p.manufacturer, "usb": p.vid is not None,
                     "arm": by_port.get(p.device) or by_port.get(tty)})  # fmt: skip
    rows.sort(key=lambda r: (not r["usb"], r["device"]))
    return {"ports": rows, "error": None, "fix": None}

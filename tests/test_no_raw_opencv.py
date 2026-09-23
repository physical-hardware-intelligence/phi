"""Guard: nothing in phi may open a camera with raw cv2.

A Windows student could not run camera_align because four files each hand-rolled
`cv2.VideoCapture`, and each got the same three platform details wrong: the MSMF
hardware-transform variable, the backend, and the FOURCC ordering. LeRobot's
OpenCVCamera already handles all three.

The rule: if LeRobot does it, call LeRobot. phi.utils.camera_backend is the only
module allowed to touch cv2 capture, and it delegates.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "phi"
ALLOWED = {"camera_backend.py"}


def _py_files() -> list[Path]:
    """Real sources only.

    Excludes `._*` AppleDouble files: this repo lives on an exFAT volume, where
    macOS writes a `._<name>` resource-fork sidecar next to every file. They are
    binary, so reading them as text raises UnicodeDecodeError.
    """
    return [
        p
        for p in SRC.rglob("*.py")
        if "__pycache__" not in p.parts and not p.name.startswith("._")
    ]


@pytest.mark.parametrize("path", _py_files(), ids=lambda p: p.name)
def test_no_raw_videocapture(path: Path) -> None:
    if path.name in ALLOWED:
        pytest.skip("the one module allowed to delegate to cv2")
    hits = [
        f"{path.name}:{n}"
        for n, line in enumerate(path.read_text().splitlines(), 1)
        if "cv2.VideoCapture(" in line and not line.lstrip().startswith("#")
    ]
    assert not hits, (
        f"raw cv2.VideoCapture in {hits}. Use phi.utils.camera_backend.open_camera "
        "or probe_camera — they go through LeRobot's OpenCVCamera, which gets the "
        "Windows MSMF variable, backend and FOURCC ordering right."
    )


@pytest.mark.parametrize("path", _py_files(), ids=lambda p: p.name)
def test_no_bare_cv2_import(path: Path) -> None:
    """`import cv2` before lerobot defeats the MSMF workaround on Windows."""
    if path.name in ALLOWED:
        pytest.skip("the one module allowed to import cv2 directly")
    bare = [
        f"{path.name}:{n}"
        for n, line in enumerate(path.read_text().splitlines(), 1)
        if re.match(r"\s*import cv2\s*$", line)
    ]
    assert not bare, (
        f"bare `import cv2` in {bare}. Import it from phi.utils.camera_backend "
        "instead, which imports it only after lerobot has set "
        "OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS=0."
    )


def test_camera_backend_actually_delegates() -> None:
    """The helper must wrap LeRobot, not reimplement it."""
    src = (SRC / "utils" / "camera_backend.py").read_text()
    assert "OpenCVCamera" in src, "camera_backend must delegate to lerobot's OpenCVCamera"
    assert "MSMF" not in src or "lerobot" in src.lower(), (
        "camera_backend is hand-rolling the MSMF workaround again"
    )

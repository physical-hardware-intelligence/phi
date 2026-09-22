"""Open a UVC camera the way each OS actually wants. Import this BEFORE cv2.

Every tool in this package opened cameras with the same six lines, and those six
lines are wrong on Windows in three separate ways. This module is the one place
that knows about it.

    from phi.utils.camera_backend import cv2, open_camera   # cv2 re-exported, see below

    cap = open_camera(0, width=640, height=480, fps=30)

WHY THIS EXISTS

1. MSMF HARDWARE TRANSFORMS. On Windows, OpenCV's default Media Foundation
   backend hangs or returns black frames unless
   `OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS=0` is set *before cv2 is imported*.
   LeRobot does this in `cameras/opencv/camera_opencv.py:31`, but only helps you
   if lerobot is imported first — our tools import cv2 directly and never touch
   lerobot. Hence the re-export: `from phi.utils.camera_backend import cv2` gets
   you a cv2 that was imported after the variable was set.

2. THE BACKEND. Windows defaults to MSMF, which is slow to open and frequently
   ignores `set()`. DirectShow is the reliable choice for UVC webcams. macOS and
   Linux are fine on their defaults (AVFoundation / V4L2).

3. PROPERTY ORDER. On Windows the FOURCC must be applied AFTER width, height and
   fps or it is silently dropped; everywhere else it must go FIRST or the driver
   picks a resolution for the old format. LeRobot flags the same asymmetry at
   `camera_opencv.py:205`. Getting this backwards costs you MJPG, and without
   MJPG three cameras do not fit in the USB 2.0 budget (02-setup.md section 5c).

None of the three fails loudly. You get a black window, or a camera that opens
at 640x480 YUY2 and starves the others.
"""

from __future__ import annotations

import os
import platform

IS_WINDOWS = platform.system() == "Windows"

# MUST precede `import cv2`. Setting it afterwards has no effect: the MSMF
# backend reads it once, at module import.
if IS_WINDOWS:
    os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

import cv2  # noqa: E402  — the import order above is the entire point

__all__ = ["cv2", "open_camera", "IS_WINDOWS", "backend_name"]


def backend_name() -> str:
    """Human-readable backend, for preflight output."""
    return "DirectShow" if IS_WINDOWS else ("AVFoundation" if platform.system() == "Darwin" else "V4L2")


def open_camera(
    index: int | str,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    fourcc: str | None = "MJPG",
) -> cv2.VideoCapture:
    """Open one camera with the right backend and the right property order.

    Returns the capture whether or not it opened — callers already check
    `isOpened()` and produce their own message naming which camera failed.
    """
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if IS_WINDOWS else cv2.VideoCapture(index)

    def _fourcc() -> None:
        if fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))

    def _size() -> None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)

    # See point 3 above.
    if IS_WINDOWS:
        _size()
        _fourcc()
    else:
        _fourcc()
        _size()

    return cap

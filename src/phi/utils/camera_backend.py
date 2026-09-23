"""Open cameras through LeRobot, with a probe-friendly wrapper.

    from phi.utils.camera_backend import cv2, open_camera, probe_camera

    cap = open_camera(0, width=640, height=480)     # raises if it is not there
    cap = probe_camera(7)                           # returns None if it is not there
    ok, frame = cap.read()                          # BGR, ready for cv2.imshow
    cap.release()

WHY THIS DELEGATES INSTEAD OF REIMPLEMENTING

`lerobot.cameras.opencv.OpenCVCamera` already solves the three things that make
raw `cv2.VideoCapture` wrong on Windows, and it solves them the same way for
everyone:

  * sets OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS=0 before importing cv2
  * selects a backend (DirectShow on Windows) rather than taking the default
  * applies FOURCC AFTER width/height/fps on Windows and BEFORE elsewhere

We used to hand-roll all three in four separate files and got all three wrong,
which is how a Windows student ended up unable to run camera_align. Anything
LeRobot does, call LeRobot.

WHY A WRAPPER IS STILL NEEDED

`OpenCVCamera` is built for a configured rig, and our tools probe. Three
mismatches, all handled here:

  * `connect()` RAISES ConnectionError on a missing index. Probe loops want a
    None, not an exception — hence `probe_camera`.
  * `read()` returns a frame and raises on failure, and needs the background
    thread that `connect()` starts. Our display loops expect the OpenCV
    `(ok, frame)` idiom, so `_Cam.read()` restores it.
  * the default colour mode is RGB, which `cv2.imshow` renders with red and
    blue swapped. We force BGR, because everything here is for human eyes.

`disconnect()` MUST run or the read thread keeps the process alive. `_Cam` is a
context manager, and `release()` is aliased to it so existing call sites that
say `cap.release()` keep working.
"""

from __future__ import annotations

from typing import Any

from lerobot.cameras.configs import ColorMode
from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

# Re-exported so callers never `import cv2` themselves. By the time lerobot has
# been imported, the MSMF variable is already set; importing cv2 first is what
# defeats it.
import cv2  # noqa: E402

__all__ = ["cv2", "open_camera", "probe_camera", "find_cameras", "_Cam"]


class _Cam:
    """OpenCVCamera with the `(ok, frame)` / `release()` idiom our tools use."""

    def __init__(self, cam: OpenCVCamera, index: int | str) -> None:
        self._cam = cam
        self.index = index

    def read(self) -> tuple[bool, Any]:
        try:
            return True, self._cam.read()
        except Exception:
            return False, None

    def isOpened(self) -> bool:  # noqa: N802 — matches cv2.VideoCapture
        return self._cam.is_connected

    def release(self) -> None:
        try:
            self._cam.disconnect()
        except Exception:
            pass

    def __enter__(self) -> _Cam:
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


def open_camera(
    index: int | str,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    fourcc: str | None = "MJPG",
    warmup: bool = False,
) -> _Cam:
    """Open one camera. Raises ConnectionError if it is not there.

    `warmup=False` by default: LeRobot waits `warmup_s` (1 s) per camera for a
    first frame, which is right for a recording session and four seconds of
    dead air in a four-camera preview.

    MJPG is not cosmetic. Uncompressed YUY2 blows the USB 2.0 budget once more
    than two cameras share a bus — see 02-setup.md section 5c.
    """
    cam = OpenCVCamera(
        OpenCVCameraConfig(
            index_or_path=index,
            width=width,
            height=height,
            fps=fps,
            fourcc=fourcc,
            color_mode=ColorMode.BGR,
        )
    )
    cam.connect(warmup=warmup)
    return _Cam(cam, index)


def probe_camera(index: int | str, **kw: Any) -> _Cam | None:
    """`open_camera` that returns None instead of raising. For scanning indices."""
    try:
        return open_camera(index, **kw)
    except Exception:
        return None


def find_cameras() -> list[dict[str, Any]]:
    """Every camera LeRobot can see. Linux enumerates /dev/video*, others scan indices."""
    return OpenCVCamera.find_cameras()

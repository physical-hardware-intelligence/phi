"""Live side-by-side camera preview for aiming and aligning the rig.

Shows every camera at the settings we actually record with (640x480, MJPG, 30 fps)
so the framing you align is the framing that lands in the dataset. Overlays a
thirds grid and a centre crosshair.

    python -m phi.utils.camera_align                       # indices 0 and 1
    python -m phi.utils.camera_align 0 1 2                 # three cameras
    python -m phi.utils.camera_align wrist=0 front=1 top=2  # our rig, labelled
    python -m phi.utils.camera_align --width 1280 --height 720

Our three feeds are `wrist` (gripper module), `front` (Brio 101 on the desk
stand) and `top` (C960 overhead on the boom). Use those exact names: they become
the observation keys in the recorded dataset, so a policy trained on `top` will
not find a camera someone later calls `overview`. Confirm the index-to-name
mapping here before writing --robot.cameras.

Tiles are scaled down automatically so the window fits on screen; capture always
happens at the real --width/--height, so what you see is what gets recorded.

Get the indices from `lerobot-find-cameras opencv` — that is authoritative for
what goes in `--robot.cameras`. Do NOT use ffplay/avfoundation indices here;
they are a different numbering and a different crop. See
docs/robots/so-arm101/02-setup.md#5a-find-the-camera-index-authoritative-for-lerobot

macOS: needs Camera permission. Run from an interactive terminal and accept the
prompt, or pre-approve in System Settings > Privacy & Security > Camera —
a non-interactive shell will just hang on a black window.
"""

from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np

GRID = (0, 220, 0)
CROSS = (0, 0, 255)
LABEL = (0, 255, 255)


def open_camera(index: int, width: int, height: int, fps: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    # MJPG matters: uncompressed YUY2 blows the USB 2.0 budget once you have
    # more than two cameras on the bus. See 02-setup.md section 5c.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def annotate(frame: np.ndarray, label: str, width: int, height: int) -> np.ndarray:
    for x in (width // 3, 2 * width // 3):
        cv2.line(frame, (x, 0), (x, height), GRID, 1)
    for y in (height // 3, 2 * height // 3):
        cv2.line(frame, (0, y), (width, y), GRID, 1)
    cv2.drawMarker(frame, (width // 2, height // 2), CROSS, cv2.MARKER_CROSS, 44, 2)
    cv2.putText(frame, label, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, LABEL, 2)
    return frame


def parse_feed(token: str) -> tuple[str, int]:
    """`2` -> ("index 2", 2);  `wrist=0` -> ("wrist", 0)."""
    if "=" in token:
        name, _, idx = token.partition("=")
        return name.strip(), int(idx)
    return f"index {int(token)}", int(token)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("feeds", nargs="*", default=None,
                    help="camera indices, optionally named: 0 1 2  or  wrist=0 front=1 top=2")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--max-window-width", type=int, default=1560,
                    help="tiles scale down to fit this (default 1560, sized for a 13in laptop)")
    args = ap.parse_args(argv)

    feeds = [parse_feed(t) for t in (args.feeds or ["0", "1"])]
    cams = [(name, i, open_camera(i, args.width, args.height, args.fps)) for name, i in feeds]

    dead = [i for _, i, cap in cams if not cap.isOpened()]
    if dead:
        print(f"could not open index {dead} — see what is attached with "
              f'`ffmpeg -f avfoundation -list_devices true -i ""`, then probe indices here '
              f"(AVFoundation numbering != OpenCV numbering)", file=sys.stderr)
    if all(not cap.isOpened() for _, _, cap in cams):
        return 1

    # keep the window on screen: capture stays at full res, only the display shrinks
    scale = min(1.0, args.max_window_width / (len(cams) * args.width))
    disp = (int(args.width * scale), int(args.height * scale))

    win = "phi camera align  —  q to quit"
    print(f"showing {[n for n, _, _ in cams]} at {args.width}x{args.height} MJPG"
          f"{'' if scale == 1.0 else f', displayed at {int(scale * 100)}%'} — press q or Esc to quit")
    try:
        while True:
            tiles = []
            for name, _, cap in cams:
                ok, frame = cap.read()
                if not ok or frame is None:
                    frame = np.zeros((args.height, args.width, 3), np.uint8)
                    cv2.putText(frame, "no frame", (12, args.height - 16),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, CROSS, 2)
                elif frame.shape[1] != args.width or frame.shape[0] != args.height:
                    # the camera refused the requested mode; show what it gave us, scaled
                    frame = cv2.resize(frame, (args.width, args.height))
                frame = annotate(frame, name, args.width, args.height)
                tiles.append(frame if scale == 1.0 else cv2.resize(frame, disp))
            cv2.imshow(win, np.hstack(tiles))
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        pass
    finally:
        for _, _, cap in cams:      # cams holds (name, index, cap)
            cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

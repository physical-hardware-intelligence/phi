"""Put a live camera feed next to the frame we actually recorded, in rerun.

For when a mount has drifted and you need the rig back where the dataset thinks
it is. Logs three views per camera: the live feed, the recorded reference frame,
and a 50/50 blend — the blend is the one to watch, because misalignment shows up
as ghosting and disappears when you have it right.

    python -m phi.utils.camera_realign front=1 top=2
    python -m phi.utils.camera_realign front=1 --episode 0 --frame 0

🚨 WHY THIS EXISTS: a camera that moved between recording and evaluation silently
invalidates the policy. Six ACT models were trained on phi_so101_8bin_v1's exact
framing; if a mount drifts, "align it nicely" is the wrong move — you have to put
it back. See docs/robots/so-arm101/01-hardware.md#placement.

🚨 THE DATASET KEYS ARE SWAPPED. In phi_so101_8bin_v1 the physical wrist camera
is stored under `observation.images.top` and the physical top camera under
`observation.images.wrist`. Pass PHYSICAL names here (wrist/front/top) and this
script does the translation. Align the top camera against the wrist camera's
footage and you will make the rig worse, confidently.

Camera indices: `ffmpeg -f avfoundation -list_devices true -i ""` is the fast way
to see what is attached and what it is called. But AVFoundation indices are a
DIFFERENT numbering from OpenCV's, which is what this script and
`--robot.cameras` use — so confirm the mapping by looking at the labelled feeds,
not by trusting the number ffmpeg printed.

macOS needs Camera permission; run from an interactive terminal and accept it.
"""

from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np

# physical camera name -> the dataset key that actually holds its footage
PHYSICAL_TO_KEY = {
    "wrist": "observation.images.top",
    "front": "observation.images.front",
    "top": "observation.images.wrist",
}
DEFAULT_ROOT = "~/.cache/huggingface/lerobot/BrutalCaesar/phi_so101_8bin_v1_20260803_150735"
REPO_ID = "BrutalCaesar/phi_so101_8bin_v1"


def parse_feed(token: str) -> tuple[str, int]:
    """`front=1` -> ("front", 1). A bare index is rejected: we need the name."""
    if "=" not in token:
        raise SystemExit(
            f"'{token}': give PHYSICAL name=index, e.g. front=1 top=2. "
            f"The name selects which recorded camera to compare against."
        )
    name, _, idx = token.partition("=")
    name = name.strip()
    if name not in PHYSICAL_TO_KEY:
        raise SystemExit(f"unknown camera '{name}'. Use one of: {', '.join(PHYSICAL_TO_KEY)}")
    return name, int(idx)


def reference_frames(names: list[str], root: str, episode: int, frame: int) -> dict[str, np.ndarray]:
    """Pull one recorded frame per camera, as uint8 RGB (H, W, 3)."""
    import os

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = os.path.expanduser(root)          # LeRobotDataset does not expand `~`
    if not os.path.isdir(root):
        raise SystemExit(f"no dataset at {root}\n(pass --root; note the local dir carries the "
                         f"recording timestamp, so it is NOT the repo_id path)")
    ds = LeRobotDataset(REPO_ID, root=root, episodes=[episode])
    if frame >= len(ds):
        raise SystemExit(f"episode {episode} has {len(ds)} frames; --frame {frame} is out of range")
    item = ds[frame]
    out = {}
    for n in names:
        key = PHYSICAL_TO_KEY[n]
        if key not in item:
            raise SystemExit(f"{key} not in the dataset (have: {[k for k in item if 'image' in k]})")
        img = item[key]                                  # (3, H, W) float in [0, 1]
        out[n] = (img.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("feeds", nargs="+", help="PHYSICAL name=index, e.g. front=1 top=2")
    ap.add_argument("--episode", type=int, default=0, help="episode to take the reference frame from")
    ap.add_argument("--frame", type=int, default=0, help="frame index within that episode")
    ap.add_argument("--root", default=DEFAULT_ROOT, help="local dataset dir (NOT the repo_id path)")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    args = ap.parse_args(argv)

    feeds = [parse_feed(f) for f in args.feeds]
    names = [n for n, _ in feeds]

    print("physical camera -> dataset key (the keys are SWAPPED, this is deliberate):")
    for n in names:
        print(f"   {n:6s} -> {PHYSICAL_TO_KEY[n]}")
    print(f"reference: episode {args.episode}, frame {args.frame}\n")

    refs = reference_frames(names, args.root, args.episode, args.frame)

    import rerun as rr
    rr.init("phi-camera-realign", spawn=True)

    caps = {}
    for name, idx in feeds:
        cap = cv2.VideoCapture(idx)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        if not cap.isOpened():
            raise SystemExit(f"could not open camera index {idx} ({name})")
        caps[name] = cap
        rr.log(f"{name}/recorded", rr.Image(refs[name]), static=True)

    print("rerun is open. Watch the `blend` view: ghosting means misaligned.")
    print("Ctrl-C here when the live feed sits on top of the recorded frame.")
    t = 0
    try:
        while True:
            for name, cap in caps.items():
                ok, bgr = cap.read()
                if not ok:
                    continue
                live = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                ref = refs[name]
                if live.shape != ref.shape:                # capture fell back to another size
                    ref = cv2.resize(ref, (live.shape[1], live.shape[0]))
                rr.set_time_sequence("frame", t)
                rr.log(f"{name}/live", rr.Image(live))
                rr.log(f"{name}/blend", rr.Image(cv2.addWeighted(live, 0.5, ref, 0.5, 0)))
            t += 1
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        for cap in caps.values():
            cap.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

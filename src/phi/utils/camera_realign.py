"""Put the rig back where the dataset thinks it is, every morning, in about a minute.

The problem this solves: we tear the setup down daily, and a camera that moved
between recording and evaluation silently invalidates every policy trained on
that dataset. Nothing errors. The policy just gets worse and you blame the
policy.

So: pick a dataset, pull a frame where the follower arm was at REST, and overlay
the live feed on it until they coincide.

    python -m phi.utils.camera_realign --list
    python -m phi.utils.camera_realign --dataset cubes_cylinder_v1 wrist=0 top=1 front=2
    python -m phi.utils.camera_realign --dataset 8bin wrist=0 top=1 --episode 5

    ⚠️ The indices in those examples are whatever they happened to be on one day.
       They are NOT a mapping to reuse. Get today's from `camera_align` first.

Why a resting frame: it is the one pose you can reproduce by hand in seconds.
Mid-reach frames are useless as a target because you cannot put the arm back
there accurately. This script FINDS the quietest frame for you by reading joint
states from the parquet (no video decode) and taking the argmin of motion over
the opening seconds — so you do not have to guess that frame 0 was actually
still.

🚨 PUT THE FOLLOWER ARM IN ITS RESTING POSE before judging alignment. The arm is
   in the reference frame too. If it is somewhere else, everything ghosts and the
   score is meaningless.

🚨 CAMERA KEYS ARE PER-DATASET. phi_so101_8bin_v1 has wrist and top transposed;
   phi_so101_redcube_in_box_v1 does not. Always pass PHYSICAL names (wrist /
   front / top) and let KEY_OVERRIDES below do the translation. The mapping in
   use is printed at startup — read it.

🚨 DO NOT TRUST A WRITTEN-DOWN INDEX ORDER, INCLUDING ONE IN THIS FILE. Camera
   indices are OpenCV's, not AVFoundation's, and macOS re-enumerates them based
   on USB port, plug order, hub power-up timing and reboots. An earlier version
   of this docstring asserted a fixed order (0=front, 1=wrist, 2=top); it was
   observed to be 0=wrist, 1=top, 2=front on 2026-08-10. Both are "correct" on
   their day, which is the point: there is no stable answer to write down.

   LOOK, EVERY SESSION, before passing indices:
       python -m phi.utils.camera_align 0 1 2 3
   `ffmpeg -f avfoundation -list_devices true -i ""` tells you what is attached,
   NOT what index OpenCV will hand you, so it does not settle this.

   You also get a second, free check: this script overlays live against recorded.
   If an index is wrong you will be staring at a gripper close-up ghosted onto an
   overhead view. A nonsense overlay means a wrong index, not a moved camera.

Keys while running:  1 blend · 2 difference · 3 edges · 4 side-by-side
                     s save a snapshot · q / Esc quit
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from dataclasses import dataclass

import cv2
import numpy as np

# Physical camera name -> dataset key, per dataset. Substring match on the
# dataset directory name; first hit wins. Anything not listed uses IDENTITY,
# i.e. the keys mean what they say.
IDENTITY = {
    "wrist": "observation.images.wrist",
    "front": "observation.images.front",
    "top": "observation.images.top",
}
KEY_OVERRIDES: dict[str, dict[str, str]] = {
    # 🚨 Recorded with the index-to-name mapping wrong. Verified against pixels:
    # `...top` holds the gripper close-up, `...wrist` holds the overhead view.
    "phi_so101_8bin_v1": {
        "wrist": "observation.images.top",
        "front": "observation.images.front",
        "top": "observation.images.wrist",
    },
}

ALIGNED_PX = 2.0        # per-axis pixel offset we call "aligned"
FONT = cv2.FONT_HERSHEY_SIMPLEX


@dataclass
class Dataset:
    name: str           # directory name, what the user sees and matches on
    root: str           # pass to LeRobotDataset(root=...)
    repo_id: str
    episodes: int
    frames: int
    cameras: list[str]
    task: str = ""      # the single_task string — often the only human-readable ID

    @property
    def key_map(self) -> tuple[dict[str, str], str | None]:
        """(physical -> dataset key, which override matched or None)."""
        for pattern, mapping in KEY_OVERRIDES.items():
            if pattern in self.name:
                return mapping, pattern
        return IDENTITY, None


def lerobot_home() -> str:
    return os.path.expanduser(
        os.environ.get("HF_LEROBOT_HOME", "~/.cache/huggingface/lerobot")
    )


def discover() -> list[Dataset]:
    """Find every local LeRobotDataset, in both layouts the cache uses.

    Recorded locally  ->  <home>/<org>/<name>_<timestamp>/meta/info.json
    Pulled from Hub   ->  <home>/hub/datasets--<org>--<name>/snapshots/<sha>/meta/info.json
    Stray copy        ->  <home>/<name>/meta/info.json

    That third layout is not hypothetical: a `--dataset.root` missing the org
    segment makes LeRobot silently download a whole second copy there, so this
    lists it too — otherwise duplicates hide from exactly the tool you would use
    to notice them.
    """
    home = lerobot_home()
    found: list[Dataset] = []

    for info_path in glob.glob(os.path.join(home, "*", "*", "meta", "info.json")):
        root = os.path.dirname(os.path.dirname(info_path))
        if os.sep + "hub" + os.sep in info_path:
            continue
        org, name = root.split(os.sep)[-2:]
        found.append(_read(root, name, f"{org}/{name}"))

    for info_path in glob.glob(os.path.join(home, "*", "meta", "info.json")):
        root = os.path.dirname(os.path.dirname(info_path))
        name = os.path.basename(root)
        found.append(_read(root, f"{name}  ⚠️ stray copy (no org dir)", f"unknown/{name}"))

    for info_path in glob.glob(
        os.path.join(home, "hub", "datasets--*", "snapshots", "*", "meta", "info.json")
    ):
        root = os.path.dirname(os.path.dirname(info_path))
        repo_dir = info_path.split(os.sep + "snapshots" + os.sep)[0]
        repo_id = os.path.basename(repo_dir).removeprefix("datasets--").replace("--", "/")
        found.append(_read(root, repo_id.split("/")[-1] + " (hub)", repo_id))

    return sorted((f for f in found if f is not None), key=lambda d: d.name)


def _read(root: str, name: str, repo_id: str) -> Dataset | None:
    try:
        m = json.load(open(os.path.join(root, "meta", "info.json")))
    except Exception:
        return None
    if not m.get("total_episodes"):
        return None                                   # empty/aborted recording

    # The local directory name is whatever repo_id you recorded under, which is
    # often not what the dataset ended up being called on the Hub. The task
    # string is the one field that says what the data actually shows.
    task = ""
    try:
        import pandas as pd
        tp = os.path.join(root, "meta", "tasks.parquet")
        if os.path.exists(tp):
            df = pd.read_parquet(tp)
            # In v3.0 the task STRING is the index and `task_index` is the only
            # column — reading df["task"] silently gives you the integer id.
            if df.index.name == "task":
                task = str(df.index[0])
            elif "task" in df.columns:
                task = str(df["task"].iloc[0])
    except Exception:
        pass

    return Dataset(
        name=name, root=root, repo_id=repo_id,
        episodes=m["total_episodes"], frames=m["total_frames"],
        cameras=[k for k in m.get("features", {}) if "image" in k],
        task=task,
    )


def print_datasets(dss: list[Dataset]) -> None:
    print(f"\nLocal datasets under {lerobot_home()}:\n")
    for i, d in enumerate(dss):
        swap = d.key_map[1]
        flag = f"  🚨 keys transposed ({swap})" if swap else ""
        print(f"  [{i}] {d.name}")
        print(f"      {d.episodes} episodes · {d.frames} frames · "
              f"{len(d.cameras)} cameras{flag}")
        if d.task:
            print(f'      task: "{d.task}"')
    print()


def choose(dss: list[Dataset], want: str | None) -> Dataset:
    if not dss:
        raise SystemExit(f"no datasets found under {lerobot_home()}")

    if want is not None:
        if want.isdigit() and int(want) < len(dss):
            return dss[int(want)]
        # match name, repo_id or task text — the local dir name is often stale
        w = want.lower()
        hits = [d for d in dss
                if w in d.name.lower() or w in d.repo_id.lower() or w in d.task.lower()]
        if len(hits) == 1:
            return hits[0]
        if not hits:
            print_datasets(dss)
            raise SystemExit(f"no dataset matches '{want}'")
        print_datasets(dss)
        raise SystemExit(f"'{want}' matches {len(hits)}: {[d.name for d in hits]} — be more specific")

    if len(dss) == 1:
        return dss[0]

    print_datasets(dss)
    if not sys.stdin.isatty():
        raise SystemExit("several datasets found — pass --dataset <name or index>")
    reply = input(f"select dataset [0-{len(dss)-1}]: ").strip()
    if not reply.isdigit() or int(reply) >= len(dss):
        raise SystemExit("not a valid selection")
    return dss[int(reply)]


def resting_frame(ds: Dataset, episode: int, search: int) -> tuple[int, float]:
    """Index of the quietest frame in the opening `search` frames of `episode`.

    Reads joint states straight from the parquet — no video decode — so this is
    fast even over a few hundred frames. Motion is the L1 norm of the
    frame-to-frame state delta, smoothed over 3 frames so a single noisy encoder
    reading cannot win.
    """
    import pandas as pd

    files = sorted(glob.glob(os.path.join(ds.root, "data", "**", "*.parquet"), recursive=True))
    if not files:
        raise SystemExit(f"no parquet under {ds.root}/data")

    frames = []
    for f in files:
        df = pd.read_parquet(f, columns=["episode_index", "frame_index", "observation.state"])
        df = df[df["episode_index"] == episode]
        if len(df):
            frames.append(df)
    if not frames:
        raise SystemExit(f"episode {episode} not found (dataset has {ds.episodes} episodes)")

    df = pd.concat(frames).sort_values("frame_index")
    states = np.stack(df["observation.state"].to_numpy())[:search]
    if len(states) < 4:
        return 0, 0.0

    motion = np.abs(np.diff(states, axis=0)).sum(axis=1)
    kernel = np.ones(3) / 3
    smooth = np.convolve(motion, kernel, mode="valid")
    best = int(np.argmin(smooth)) + 1                 # +1: centre of the window
    return best, float(smooth[best - 1])


def reference_frames(ds: Dataset, names: list[str], episode: int, frame: int) -> dict[str, np.ndarray]:
    """Decode one recorded frame per requested camera -> uint8 RGB."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # Validate BEFORE constructing: LeRobotDataset(episodes=[out_of_range]) dies
    # deep inside `datasets` with `Instruction "train" corresponds to no data!`,
    # which tells the user nothing.
    if not 0 <= episode < ds.episodes:
        raise SystemExit(f"episode {episode} out of range — this dataset has "
                         f"{ds.episodes} episodes (0-{ds.episodes - 1})")

    lds = LeRobotDataset(ds.repo_id, root=ds.root, episodes=[episode])
    if frame >= len(lds):
        raise SystemExit(f"episode {episode} has {len(lds)} frames; --frame {frame} out of range")
    item = lds[frame]
    mapping, _ = ds.key_map

    out = {}
    for n in names:
        key = mapping[n]
        if key not in item:
            raise SystemExit(
                f"'{n}' maps to {key}, which is not in this dataset.\n"
                f"available: {[k for k in item if 'image' in k]}"
            )
        img = item[key]                               # (3, H, W) float in [0, 1]
        out[n] = (img.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    return out


def offset(live_gray: np.ndarray, ref_gray: np.ndarray, window: np.ndarray) -> tuple[float, float, float]:
    """How far the live image sits from the reference, in pixels, sub-pixel.

    Phase correlation recovers TRANSLATION only — not rotation or scale — but
    translation is what a bumped mount mostly produces, and a number beats
    squinting at a blend. Returns (dx, dy, response); response near 0 means the
    two images have little in common and the offset is not trustworthy.
    """
    (dx, dy), resp = cv2.phaseCorrelate(np.float32(ref_gray), np.float32(live_gray), window)
    return dx, dy, resp


def annotate(tile: np.ndarray, name: str, dx: float, dy: float, resp: float, mad: float) -> np.ndarray:
    ok = abs(dx) < ALIGNED_PX and abs(dy) < ALIGNED_PX
    colour = (0, 220, 0) if ok else (0, 165, 255)
    h = tile.shape[0]
    cv2.rectangle(tile, (0, 0), (tile.shape[1] - 1, h - 1), colour, 2)
    cv2.putText(tile, name, (10, 26), FONT, 0.7, (255, 255, 0), 2)

    # The correction is the negation of the measured offset. Panning a camera one
    # way moves the scene the other way, hence "pan the opposite way" below.
    label = "ALIGNED" if ok else f"move image  x {-dx:+.1f}  y {-dy:+.1f} px"
    cv2.putText(tile, label, (10, h - 34), FONT, 0.6, colour, 2)
    cv2.putText(tile, f"mad {mad:5.1f}   conf {resp:.2f}", (10, h - 12), FONT, 0.5, (200, 200, 200), 1)
    return tile


def compose(mode: int, live: np.ndarray, ref: np.ndarray) -> np.ndarray:
    if mode == 1:
        return cv2.addWeighted(live, 0.5, ref, 0.5, 0)
    if mode == 2:
        return cv2.absdiff(live, ref)
    if mode == 3:
        # live edges in green, reference edges in magenta: fringes = misaligned
        le = cv2.Canny(cv2.cvtColor(live, cv2.COLOR_RGB2GRAY), 60, 180)
        re = cv2.Canny(cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY), 60, 180)
        out = np.zeros_like(live)
        out[..., 1] = le
        out[..., 0] = re
        out[..., 2] = re
        return out
    half = (live.shape[1] // 2, live.shape[0] // 2)
    return np.hstack([cv2.resize(ref, half), cv2.resize(live, half)])


MODE_NAMES = {1: "blend", 2: "difference", 3: "edges", 4: "side-by-side"}


def parse_feed(token: str) -> tuple[str, int]:
    if "=" not in token:
        raise SystemExit(f"'{token}': use PHYSICAL name=index, e.g. wrist=1 front=0")
    name, _, idx = token.partition("=")
    name = name.strip()
    if name not in IDENTITY:
        raise SystemExit(f"unknown camera '{name}'. use one of: {', '.join(IDENTITY)}")
    return name, int(idx)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("feeds", nargs="*", help="PHYSICAL name=index, e.g. wrist=1 front=0")
    ap.add_argument("--list", action="store_true", help="list local datasets and exit")
    ap.add_argument("--dataset", help="dataset name substring or index from --list")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--frame", type=int, default=None,
                    help="reference frame; default = quietest frame found automatically")
    ap.add_argument("--search", type=int, default=90,
                    help="how many opening frames to search for the resting pose (default 90 = 3 s)")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--max-window-width", type=int, default=1600)
    ap.add_argument("--snapshot-dir", default="outputs/camera-realign")
    args = ap.parse_args(argv)

    datasets = discover()
    if args.list:
        print_datasets(datasets)
        return 0

    ds = choose(datasets, args.dataset)
    if not 0 <= args.episode < ds.episodes:
        raise SystemExit(f"--episode {args.episode} out of range — '{ds.name}' has "
                         f"{ds.episodes} episodes (0-{ds.episodes - 1})")
    mapping, override = ds.key_map

    print(f"\ndataset : {ds.name}")
    print(f"          {ds.episodes} episodes · {ds.frames} frames · {ds.root}")
    if override:
        print(f"🚨 keys transposed in this dataset (matched '{override}'):")
    else:
        print("keys are identity-mapped (they mean what they say):")
    for n in ("wrist", "front", "top"):
        if mapping[n] in ds.cameras:
            print(f"          {n:6s} -> {mapping[n]}")

    if not args.feeds:
        raise SystemExit("\nnothing to align. add PHYSICAL name=index, e.g. wrist=1 front=0")
    feeds = [parse_feed(f) for f in args.feeds]
    names = [n for n, _ in feeds]

    if args.frame is None:
        frame, motion = resting_frame(ds, args.episode, args.search)
        print(f"\nresting frame : episode {args.episode}, frame {frame} "
              f"(quietest of the first {args.search}; motion {motion:.4f})")
    else:
        frame = args.frame
        print(f"\nreference     : episode {args.episode}, frame {frame} (you chose it)")

    refs = reference_frames(ds, names, args.episode, frame)

    caps = {}
    for name, idx in feeds:
        cap = cv2.VideoCapture(idx)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        if not cap.isOpened():
            for c in caps.values():
                c.release()
            raise SystemExit(f"could not open camera index {idx} ({name}) — "
                             f"check with `python -m phi.utils.camera_align 0 1 2 3`")
        caps[name] = cap

    window = cv2.createHanningWindow((args.width, args.height), cv2.CV_32F)
    ref_gray = {n: cv2.cvtColor(r, cv2.COLOR_RGB2GRAY) for n, r in refs.items()}

    print("\n🚨 put the follower arm in its RESTING pose — it is in the reference frame too.\n")
    print("   1 blend · 2 difference · 3 edges · 4 side-by-side · s snapshot · q quit")
    print(f"   green border = within {ALIGNED_PX:.0f} px on both axes\n")

    mode = 1
    win = "phi camera realign"
    try:
        while True:
            tiles = []
            for name, cap in caps.items():
                ok, bgr = cap.read()
                if not ok:
                    continue
                live = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                ref = refs[name]
                if live.shape != ref.shape:
                    ref = cv2.resize(ref, (live.shape[1], live.shape[0]))

                lg = cv2.cvtColor(live, cv2.COLOR_RGB2GRAY)
                rg = ref_gray[name]
                if lg.shape != rg.shape:
                    rg = cv2.resize(rg, (lg.shape[1], lg.shape[0]))
                w = window if lg.shape[::-1] == (args.width, args.height) else \
                    cv2.createHanningWindow(lg.shape[::-1], cv2.CV_32F)
                dx, dy, resp = offset(lg, rg, w)
                mad = float(np.abs(lg.astype(np.int16) - rg.astype(np.int16)).mean())

                tile = compose(mode, live, ref)
                tiles.append(annotate(tile, name, dx, dy, resp, mad))

            if tiles:
                target_h = max(t.shape[0] for t in tiles)
                tiles = [cv2.resize(t, (int(t.shape[1] * target_h / t.shape[0]), target_h))
                         for t in tiles]
                grid = np.hstack(tiles)
                scale = min(1.0, args.max_window_width / grid.shape[1])
                if scale < 1.0:
                    grid = cv2.resize(grid, (int(grid.shape[1] * scale), int(grid.shape[0] * scale)))
                cv2.putText(grid, MODE_NAMES[mode], (10, grid.shape[0] - 10), FONT, 0.6,
                            (255, 255, 255), 1)
                cv2.imshow(win, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))

            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            if k in (ord("1"), ord("2"), ord("3"), ord("4")):
                mode = k - ord("0")
            if k == ord("s") and tiles:
                os.makedirs(args.snapshot_dir, exist_ok=True)
                path = os.path.join(args.snapshot_dir,
                                    f"{ds.name.replace(' ', '')}_ep{args.episode}_f{frame}.png")
                cv2.imwrite(path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
                print(f"saved {path}")
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        pass
    finally:
        for cap in caps.values():
            cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

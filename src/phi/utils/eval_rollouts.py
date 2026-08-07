"""Score policy rollouts on the real arm, with partial credit.

Binary success throws away almost everything. A policy that reaches, grasps and
carries the object to the rim of the bin before dropping it is not the same as
one that never moves, and pooling them into "0/20" hides the difference that
actually tells you what to fix next.

Rubric is taken verbatim from Sherry Chen's SO-101 ACT write-up so our numbers
are comparable to a published result on the same hardware:
https://huggingface.co/blog/sherryxychen/train-act-on-so-101

    reach object 0.2 · grasp 0.4 · reach container 0.7 · release 0.8 · in container 1.0

The staging window shows each camera's RECORDED reference frame above its LIVE
feed, so a dead or misindexed camera is caught before the arm moves.

Note the spacing is deliberately uneven — the 0.4 -> 0.7 jump weights transport
most heavily. Do not "tidy" it to 0.6; that would break comparability.

    # list the held-out episodes and what each one needs staged
    python -m phi.utils.eval_rollouts --dataset 161432 --plan

    # score a model
    python -m phi.utils.eval_rollouts --dataset 161432 \
        --model BrutalCaesar/act_so101_cubcyl_poshold_chunk50_cvae_3cam \
        --cameras wrist=0,top=1,front=2

    # aggregate whatever has been scored so far
    python -m phi.utils.eval_rollouts --report

WHAT THIS DOES AND DOES NOT DO. The rollout itself is delegated to
`lerobot-rollout --strategy.type=base`, which owns the observation ->
preprocessor -> select_action -> postprocessor -> send_action pipeline. That
plumbing is easy to get subtly wrong (a mis-normalised action makes every policy
look broken), so it is not reimplemented here. This module owns the parts
LeRobot has no opinion about: showing the operator the scene to reproduce,
sequencing episodes identically across models, capturing the score, and
reporting it without pooling.

🚨 The arm moves autonomously during a rollout. Keep a hand near the power.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from datetime import datetime, timezone

import cv2
import numpy as np

from phi.utils.camera_realign import choose, discover

# (score, key, label). Verbatim from the blog — see the module docstring.
RUBRIC = [
    (0.0, "0", "no meaningful attempt"),
    (0.2, "1", "reached the object"),
    (0.4, "2", "grasped the object"),
    (0.7, "3", "carried it to the container"),
    (0.8, "4", "released it"),
    (1.0, "5", "object in the container"),
]

# episode_index -> (object, container), for phi_so101_cubes_cylinder_v1 only.
BLOCKS = [
    (0, 19, "red cube (25mm)", "cardboard box"),
    (20, 39, "red cube (25mm)", "white bin"),
    (40, 59, "yellow cylinder", "cardboard box"),
    (60, 79, "yellow cylinder", "white bin"),
    (80, 99, "white cube (45mm)", "cardboard box"),
    (100, 119, "white cube (45mm)", "white bin"),
]
BLOCKS_APPLY_TO = "cubes_cylinder"          # guard: refuse to label a different dataset

HELD_OUT = [*range(0, 5), *range(20, 25), *range(45, 50),
            *range(65, 70), *range(90, 95), *range(110, 115)]
# Two per block: enough for a per-object and per-container number without
# spending an hour of arm time per model.
DEFAULT_EPISODES = [0, 1, 20, 21, 45, 46, 65, 66, 90, 91, 110, 111]

# 🚨 WHAT KIND OF HOLDOUT THIS IS. The 30 held-out episodes hold out *positions*:
#    every object and both containers appear in training, and what the policy has
#    not seen is this particular object placement. That is interpolation — a test
#    split — NOT out-of-distribution. True OOD for this dataset means an unseen
#    OBJECT, which is Parv's axis (P-A holds out the red cube, P-B the white one).
#    Do not report position-holdout numbers as generalisation to new objects.
#
# CONTROL EPISODES: three episodes the policy DID train on, one per object, all
# with the cardboard box so no container swap is needed. Run these FIRST. If a
# policy cannot do an episode it was trained on, the fault is the rig — camera
# alignment, indices, calibration drift — not the policy, and the remaining 24
# rollouts would be measuring a broken setup. They also give the train-vs-heldout
# gap, which is the actual generalisation measurement; a held-out score alone
# tells you the level, not the gap.
CONTROL_EPISODES = [5, 50, 95]

CSV_PATH = "outputs/rollout_scores.csv"
FIELDS = ["timestamp", "model", "episode", "split", "object", "container",
          "stage", "score", "grasp_approach", "hesitated", "notes"]
FONT = cv2.FONT_HERSHEY_SIMPLEX


def split_of(ep: int) -> str:
    return "heldout" if ep in HELD_OUT else "train"


def block_of(ep: int) -> tuple[str, str]:
    for lo, hi, obj, cont in BLOCKS:
        if lo <= ep <= hi:
            return obj, cont
    raise SystemExit(f"episode {ep} is outside the known blocks (0-119)")


def reference_frames(ds, episode: int, names: list[str]) -> dict[str, np.ndarray | None]:
    """The episode's first recorded frame per PHYSICAL camera name, as BGR.

    Goes through the dataset's key map, so this stays correct on datasets whose
    keys are transposed (phi_so101_8bin_v1) as well as ones where they are not.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if not 0 <= episode < ds.episodes:
        raise SystemExit(f"episode {episode} out of range (dataset has {ds.episodes})")
    lds = LeRobotDataset(ds.repo_id, root=ds.root, episodes=[episode])
    item = lds[0]
    mapping, _ = ds.key_map
    out: dict[str, np.ndarray | None] = {}
    for n in names:
        key = mapping.get(n)
        if key is None or key not in item:
            out[n] = None
            continue
        img = (item[key].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        out[n] = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return out


def open_cameras(cameras: dict[str, int], width: int = 640, height: int = 480) -> dict:
    caps = {}
    for n, idx in cameras.items():
        cap = cv2.VideoCapture(idx)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        caps[n] = cap
    return caps


def _tile(text: str, size: tuple[int, int], colour=(0, 0, 255)) -> np.ndarray:
    t = np.zeros((size[1], size[0], 3), np.uint8)
    cv2.putText(t, text, (12, size[1] // 2), FONT, 0.6, colour, 2)
    cv2.rectangle(t, (0, 0), (size[0] - 1, size[1] - 1), colour, 2)
    return t


def _label(img: np.ndarray, text: str, colour) -> np.ndarray:
    cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), colour, 2)
    cv2.putText(img, text, (10, 24), FONT, 0.55, colour, 2)
    return img


def show_setup(refs: dict, caps: dict, ep: int, obj: str, cont: str,
               n: int, total: int, split: str) -> str:
    """Recorded reference (top row) against LIVE feeds (bottom row).

    Two jobs at once. Staging: match the object and container to the recorded
    frame. Rig check: a dead or misindexed camera is visible here, before the arm
    moves, and SPACE is refused until every camera delivers frames. A camera that
    has drifted since recording also shows up as a mismatch between the rows —
    a lightweight version of `camera_realign` folded into the eval loop.

    Returns 'go' | 'skip' | 'quit'.
    """
    names = list(caps.keys())
    size = (426, 320)
    win = "phi eval - stage the scene (top: recorded / bottom: live)"
    try:
        while True:
            rec, live, dead = [], [], []
            for nm in names:
                r = refs.get(nm)
                rec.append(_label(cv2.resize(r, size).copy(), f"RECORDED {nm}", (0, 200, 255))
                           if r is not None else _tile(f"{nm}: not in dataset", size, (0, 140, 255)))
                ok, frame = caps[nm].read()
                if not ok or frame is None:
                    dead.append(nm)
                    live.append(_tile(f"{nm}: NO SIGNAL", size))
                else:
                    live.append(_label(cv2.resize(frame, size), f"LIVE {nm}", (0, 230, 0)))

            grid = np.vstack([np.hstack(rec), np.hstack(live)])
            banner = np.zeros((116, grid.shape[1], 3), np.uint8)
            tag = "CONTROL - trained on, tests the rig" if split == "train" else "HELD OUT"
            lines = [
                f"[{n}/{total}]  episode {ep}  [{tag}]   {obj}  ->  {cont}",
                "Match the object and container to the RECORDED row above.",
                (f"!! NO SIGNAL from {dead} - fix before running !!" if dead
                 else "SPACE = run rollout    s = skip    q = quit"),
            ]
            for i, t in enumerate(lines):
                col = (0, 255, 255) if i == 0 else ((0, 0, 255) if dead and i == 2 else (220, 220, 220))
                cv2.putText(banner, t, (12, 34 + i * 30), FONT, 0.62, col, 2 if i != 1 else 1)

            frame = np.vstack([banner, grid])
            scale = min(1.0, 1700 / frame.shape[1])
            if scale < 1.0:
                frame = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))
            cv2.imshow(win, frame)

            k = cv2.waitKey(30) & 0xFF
            if k == ord(" "):
                if dead:
                    print(f"  refusing to run: no signal from {dead}")
                    continue
                return "go"
            if k == ord("s"):
                return "skip"
            if k in (ord("q"), 27):
                return "quit"
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                return "quit"
    finally:
        cv2.destroyWindow(win)


def run_rollout(model: str, cameras: dict[str, int], task: str, duration: int,
                port: str, robot_id: str, dry_run: bool) -> int:
    cams = ", ".join(
        f"{n}: {{type: opencv, index_or_path: {i}, width: 640, height: 480, fps: 30, fourcc: MJPG}}"
        for n, i in cameras.items()
    )
    cmd = [
        "lerobot-rollout",
        "--strategy.type=base",
        f"--policy.path={model}",
        "--robot.type=so101_follower",
        f"--robot.port={port}",
        f"--robot.id={robot_id}",
        f"--robot.cameras={{ {cams} }}",
        f"--task={task}",
        f"--duration={duration}",
    ]
    print("\n  " + " ".join(cmd) + "\n", flush=True)
    if dry_run:
        print("  [--dry-run: not executing]")
        return 0
    return subprocess.call(cmd)


def ask_score() -> tuple[str, float, str, str, str] | None:
    """Prompt for the rubric stage plus the two fields loss cannot see."""
    print("\n  stage reached:")
    for score, key, label in RUBRIC:
        print(f"    {key} = {score:.1f}  {label}")
    print("    r = redo this episode (operator error, not a policy failure)")
    while True:
        k = input("  > ").strip().lower()
        if k == "r":
            return None
        hit = [(s, lb) for s, key, lb in RUBRIC if key == k]
        if hit:
            score, label = hit[0]
            break
        print("    not one of 0-5 or r")

    # Which of the ~3 demonstrated grasp approaches the policy chose. Held-out L1
    # cannot see this, and an averaging policy tends to sit between approaches.
    approach = input("  grasp approach [a/b/c, m=mixed/hedging, - = never grasped]: ").strip().lower() or "-"
    # The training data has leading dead time; the prediction is that policies
    # hesitate at episode start. Never checked until now.
    hes = ""
    while hes not in ("y", "n"):
        hes = (input("  hesitated at start? [y/n]: ").strip().lower() or "n")[:1]
    notes = input("  notes (optional): ").strip()
    return label, score, approach, hes, notes


def append_row(row: dict) -> None:
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    new = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def report() -> int:
    if not os.path.exists(CSV_PATH):
        raise SystemExit(f"no scores yet at {CSV_PATH}")
    rows = list(csv.DictReader(open(CSV_PATH)))
    if not rows:
        raise SystemExit("score file is empty")

    def agg(rs):
        sc = [float(r["score"]) for r in rs]
        succ = sum(1 for s in sc if s >= 1.0)
        return len(sc), succ / len(sc), sum(sc) / len(sc)

    for model in sorted({r["model"] for r in rows}):
        mr = [r for r in rows if r["model"] == model]
        print(f"\n=== {model}")
        by_split = {}
        for sp, label in (("train", "CONTROL (trained on)"), ("heldout", "HELD OUT (unseen positions)")):
            rs = [r for r in mr if r.get("split", "heldout") == sp]
            if rs:
                n, sr, mp = agg(rs)
                by_split[sp] = mp
                print(f"    {label:28s} n={n:3d}   success {sr:5.0%}   mean progress {mp:.3f}")
        if len(by_split) == 2:
            print(f"    generalisation gap (control - heldout mean progress): "
                  f"{by_split['train'] - by_split['heldout']:+.3f}")
        mr = [r for r in mr if r.get("split", "heldout") == "heldout"] or mr
        print(f"    -- breakdown below is HELD-OUT only --")
        # Never pool: a model can be fine on two objects and hopeless on the third.
        for label, field in (("by object", "object"), ("by container", "container")):
            print(f"    {label}:")
            for v in sorted({r[field] for r in mr}):
                n, sr, mp = agg([r for r in mr if r[field] == v])
                print(f"      {v:20s} n={n:3d}   success {sr:5.0%}   mean progress {mp:.3f}")
        hes = sum(1 for r in mr if r["hesitated"] == "y")
        appr = {}
        for r in mr:
            appr[r["grasp_approach"]] = appr.get(r["grasp_approach"], 0) + 1
        print(f"    hesitated at start: {hes}/{len(mr)}")
        print(f"    grasp approaches  : {dict(sorted(appr.items()))}")
    print(f"\n({len(rows)} rollouts in {CSV_PATH})")
    return 0


def parse_cameras(s: str) -> dict[str, int]:
    out = {}
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "=" not in tok:
            raise SystemExit(f"'{tok}': use name=index, e.g. wrist=0,top=1,front=2")
        n, _, i = tok.partition("=")
        out[n.strip()] = int(i)
    return out


def required_cameras(model: str) -> list[str]:
    """Which camera keys the checkpoint expects — read config.json, no torch."""
    import json
    try:
        if os.path.isdir(model):
            cfg = json.load(open(os.path.join(model, "config.json")))
        else:
            from huggingface_hub import hf_hub_download
            cfg = json.load(open(hf_hub_download(model, "config.json")))
    except Exception as e:
        print(f"  (could not read {model} config: {type(e).__name__}; skipping camera check)")
        return []
    return sorted(k.split(".")[-1] for k in cfg.get("input_features", {}) if "image" in k)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="HF id or local pretrained_model dir")
    ap.add_argument("--dataset", help="dataset name substring or --list index")
    ap.add_argument("--episodes", help=f"comma list (default: {CONTROL_EPISODES} then {DEFAULT_EPISODES})")
    ap.add_argument("--controls", type=int, default=len(CONTROL_EPISODES),
                    help="how many trained-on control episodes to run first (0 to skip)")
    ap.add_argument("--cameras", default="wrist=0,top=1,front=2")
    ap.add_argument("--duration", type=int, default=30, help="seconds per rollout")
    ap.add_argument("--port", default="/dev/tty.usbmodem5B7B0096441")
    ap.add_argument("--robot-id", default="phi_follower")
    ap.add_argument("--plan", action="store_true", help="print the episode plan and exit")
    ap.add_argument("--report", action="store_true", help="aggregate scores and exit")
    ap.add_argument("--dry-run", action="store_true", help="everything except moving the arm")
    args = ap.parse_args(argv)

    if args.report:
        return report()

    if args.episodes:
        episodes = [int(x) for x in args.episodes.split(",")]
    else:
        # controls first, deliberately: a failure here means stop and fix the rig
        episodes = list(CONTROL_EPISODES[:args.controls]) + list(DEFAULT_EPISODES)
    controls = [e for e in episodes if split_of(e) == "train"]
    if controls:
        print(f"note: {controls} are CONTROL episodes the policy trained on. They test the "
              f"rig, not generalisation, and are reported separately.\n")

    if args.plan:
        print(f"\n{len(episodes)} episodes, 2 per object x container block:\n")
        for e in episodes:
            obj, cont = block_of(e)
            tag = "CONTROL (trained on)" if split_of(e) == "train" else "held out"
            print(f"  episode {e:3d}   {obj:18s} -> {cont:15s} [{tag}]")
        print(f"\nrubric: " + " · ".join(f"{s:.1f} {lb}" for s, _, lb in RUBRIC[1:]))
        return 0

    if not args.model or not args.dataset:
        raise SystemExit("need --model and --dataset (or use --plan / --report)")

    ds = choose(discover(), args.dataset)
    if BLOCKS_APPLY_TO not in ds.name:
        raise SystemExit(f"'{ds.name}' is not the cubes+cylinder dataset — the object/container "
                         f"block map in this script does not apply to it")

    cameras = parse_cameras(args.cameras)
    need = required_cameras(args.model)
    missing = [c for c in need if c not in cameras]
    if missing:
        raise SystemExit(f"{args.model} requires camera(s) {missing}, which you did not pass.\n"
                         f"It will not run without them. You gave: {sorted(cameras)}")
    if need:
        print(f"model expects {len(need)} camera(s): {need}")
    extra = [c for c in cameras if c not in need] if need else []
    if extra:
        print(f"  note: {extra} passed but unused by this policy")

    print(f"dataset : {ds.name}")
    print(f"model   : {args.model}")
    print(f"episodes: {episodes}")
    print("\n🚨 The arm moves on its own during a rollout. Hand near the power.\n")

    done = 0
    for i, ep in enumerate(episodes, 1):
        obj, cont = block_of(ep)
        task = f"pick up the {obj.split(' (')[0]} and place it in the {cont}"
        refs = reference_frames(ds, ep, list(cameras))
        while True:
            # Cameras must be OURS for staging and FREE for the rollout, so they are
            # opened and released around each staging step rather than held open.
            caps = open_cameras(cameras)
            try:
                action = show_setup(refs, caps, ep, obj, cont, i, len(episodes), split_of(ep))
            finally:
                for c in caps.values():
                    c.release()
            if action == "quit":
                print("\nstopped.")
                return report() if done else 0
            if action == "skip":
                break
            rc = run_rollout(args.model, cameras, task, args.duration,
                             args.port, args.robot_id, args.dry_run)
            if rc != 0:
                print(f"  rollout exited rc={rc} — not scoring this one")
                break
            scored = ask_score()
            if scored is None:
                print("  redoing this episode\n")
                continue
            label, score, approach, hes, notes = scored
            append_row({
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "model": args.model, "episode": ep, "split": split_of(ep),
                "object": obj, "container": cont,
                "stage": label, "score": score, "grasp_approach": approach,
                "hesitated": hes, "notes": notes,
            })
            done += 1
            print(f"  recorded {score:.1f} ({label})\n")
            break

    print(f"\n{done} rollouts scored.")
    return report() if done else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Measure the pre-teleop dead air at the start of every episode, and emit a trim table.

Every episode begins with the arm sitting still while the operator has not yet moved the leader.
Those frames are not merely wasted: ACT has `n_obs_steps=1` and no clock, so while the arm is
stationary in a static scene every frame is a near-identical observation, yet the recorded action
differs by how long the operator happened to wait. At rollout the policy predicts "hold still",
nothing moves, the next observation is identical, and it predicts "hold still" again — an absorbing
state the arm can never leave.

    python -m phi.utils.deadtime
    python -m phi.utils.deadtime --repo-id BrutalCaesar/phi_so101_8bin_v1 --out trim.csv
    python -m phi.utils.deadtime --lead-tol 2.0        # sensitivity check

Output is a per-episode CSV whose `DROP` column is the number of leading frames to exclude. Apply it
at the SAMPLER (a `SubsetRandomSampler` index list), never by deleting frames from disk: an action
chunk is N *contiguous* frames, so removing interior frames would teach trajectories with teleport
jumps no arm can execute.

🚨 ONLY the opening pause is reported for trimming. Interior pauses (4 of 119 episodes in
phi_so101_8bin_v1) and trailing pauses (117 of 119) are measured and reported, but must be LEFT
ALONE. Trailing pauses sit at the end pose with the object already placed, so they do not recreate
the opening absorbing state, and stopping when the task is done is behaviour worth keeping.

WHY the cut is defined on cumulative departure and not per-frame movement: servo jitter plus the
operator's hand resting on the leader exceeds 0.1 deg/frame from frame 1 in 57 of 119 episodes, so a
per-frame test reports a 1-frame pause where the arm has in fact not moved for 100+ frames. Taking
the FIRST crossing of a cumulative threshold is immune to jitter and can never reach past the start
of real motion into an interior pause.
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FPS = 30
LEAD_TOL = 1.0  # degrees of departure from the start pose
WIN = 15  # 0.5 s window for the jitter-robust interior/trailing test
INTERIOR_MIN = 15  # flag interior static runs of at least this many frames


def find_snapshot(repo_id: str) -> Path:
    """Locate the local LeRobot cache snapshot for a dataset."""
    slug = "datasets--" + repo_id.replace("/", "--")
    pattern = str(Path.home() / ".cache/huggingface/lerobot/hub" / slug / "snapshots" / "*")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise SystemExit(
            f"no local snapshot for {repo_id!r}.\n"
            f"Looked in: {pattern}\n"
            f"Fetch it first, e.g.  python -c \"from lerobot.datasets.lerobot_dataset import \"\n"
            f"  \"LeRobotDataset as D; D('{repo_id}', episodes=[0])\"\n"
            f"or pass --snapshot explicitly."
        )
    return Path(matches[-1])


def first_departure(a: np.ndarray, tol: float) -> int:
    """First frame whose max per-joint deviation from the start pose exceeds `tol`."""
    dev = np.abs(a - a[0]).max(axis=1)
    return int(np.argmax(dev > tol)) if (dev > tol).any() else len(a)


def windowed_static(a: np.ndarray, tol: float, win: int) -> np.ndarray:
    """static[t] = total travel over frames [t, t+win) is under `tol`.

    Vectorised with sliding_window_view; a per-frame Python loop is slow enough to be killed on a
    login node.
    """
    n = len(a)
    if n < win:
        return np.zeros(n, dtype=bool)
    w = np.lib.stride_tricks.sliding_window_view(a, win, axis=0)  # (n-win+1, joints, win)
    travel = (w.max(axis=-1) - w.min(axis=-1)).max(axis=-1)
    out = np.zeros(n, dtype=bool)
    out[: len(travel)] = travel < tol
    out[len(travel) :] = out[len(travel) - 1] if len(travel) else False
    return out


def runs_of(flags: np.ndarray) -> list[tuple[int, int]]:
    """[(start, length)] of maximal True runs."""
    runs, i = [], 0
    while i < len(flags):
        if flags[i]:
            j = i
            while j < len(flags) and flags[j]:
                j += 1
            runs.append((i, j - i))
            i = j
        else:
            i += 1
    return runs


def load_actions(snapshot: Path) -> dict[int, np.ndarray]:
    """{episode_index: (frames, joints)} read straight from parquet — no video decode."""
    files = sorted(glob.glob(str(snapshot / "data" / "**" / "*.parquet"), recursive=True))
    if not files:
        raise SystemExit(f"no parquet under {snapshot / 'data'}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return {
        int(ep): np.stack(g.sort_values("frame_index")["action"].to_numpy())
        for ep, g in df.groupby("episode_index")
    }


def audit(eps: dict[int, np.ndarray], lead_tol: float, left_max_ep: int) -> pd.DataFrame:
    rows = []
    for ep, a in sorted(eps.items()):
        n = len(a)
        drop = first_departure(a, lead_tol)
        tail = windowed_static(a, lead_tol, WIN)[drop:]
        rr = runs_of(tail)
        trail = rr[-1][1] if rr and rr[-1][0] + rr[-1][1] == len(tail) else 0
        interior = [(st, ln) for st, ln in rr if st != 0 and st + ln != len(tail)]
        rows.append(
            {
                "ep": ep,
                "side": "left" if ep <= left_max_ep else "right",
                "frames": n,
                "DROP": drop,
                "drop_s": round(drop / FPS, 2),
                "drop_pct": round(100 * drop / n, 1),
                "kept": n - drop,
                "trail": trail,
                "trail_s": round(trail / FPS, 2),
                "interior_max": max((ln for _, ln in interior), default=0),
                "interior_runs": sum(1 for _, ln in interior if ln >= INTERIOR_MIN),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-id", default="BrutalCaesar/phi_so101_8bin_v1")
    p.add_argument("--snapshot", type=Path, default=None, help="override the cache lookup")
    p.add_argument("--out", type=Path, default=Path("deadtime_per_episode.csv"))
    p.add_argument("--lead-tol", type=float, default=LEAD_TOL, help="degrees of departure (default 1.0)")
    p.add_argument("--left-max-ep", type=int, default=58, help="last episode of the left-side block")
    p.add_argument("--expect-episodes", type=int, default=119, help="0 to disable the guard")
    args = p.parse_args()

    snapshot = args.snapshot or find_snapshot(args.repo_id)
    eps = load_actions(snapshot)

    # A partially-fetched cache is the silent failure here: LeRobotDataset(..., episodes=[0]) pulls
    # only the parquet shard containing episode 0, so an audit run against it reports plausible
    # numbers computed over a fraction of the data. Left-side pauses are ~2.5x longer than
    # right-side ones, so a left-only cache biases the trim badly.
    if args.expect_episodes and len(eps) != args.expect_episodes:
        raise SystemExit(
            f"found {len(eps)} episodes, expected {args.expect_episodes}. The local snapshot is "
            f"probably partial ({snapshot}).\nFetch the whole dataset, or pass "
            f"--expect-episodes {len(eps)} if a subset is genuinely what you want."
        )
    r = audit(eps, args.lead_tol, args.left_max_ep)
    r.to_csv(args.out, index=False)

    total, dropped = int(r.frames.sum()), int(r.DROP.sum())
    print(f"{len(r)} episodes, {total} frames  ->  {args.out}\n")
    print("=== DROP (opening pause only) ===")
    print(r.groupby("side")[["DROP", "drop_s", "drop_pct"]].agg(["mean", "median", "min", "max"]).round(1))
    print(f"\ntotal dropped {dropped} of {total} = {100 * dropped / total:.1f}%")

    print("\n=== KEPT: trailing + interior ===")
    print(r.groupby("side")[["trail_s", "interior_max", "interior_runs"]].agg(["mean", "median", "max"]).round(1))
    print(f"episodes with an interior pause >= 0.5s: {(r.interior_runs > 0).sum()} of {len(r)}")
    print(f"episodes with a trailing pause >= 0.5s : {(r.trail >= WIN).sum()} of {len(r)}")

    print("\n=== SAFETY ===")
    print(f"DROP == 0            : {(r.DROP == 0).sum()}  {list(r[r.DROP == 0].ep)}")
    print(f"DROP > 35% of episode: {(r.drop_pct > 35).sum()}")
    print(f"kept < 300 frames    : {(r.kept < 300).sum()}")

    # If the cut were landing on slow task motion rather than genuine dead air, the total would fan
    # out with the threshold. On phi_so101_8bin_v1 a 20x change moves it by 1.5 percentage points.
    print("\n=== threshold sensitivity (degrees -> frames dropped) ===")
    for tol in (0.5, 1.0, 2.0, 5.0, 10.0):
        tot = sum(first_departure(a, tol) for a in eps.values())
        print(f"  {tol:>5} deg -> {tot:>6} ({100 * tot / total:.1f}%)")


if __name__ == "__main__":
    sys.exit(main())

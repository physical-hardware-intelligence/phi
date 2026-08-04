"""Structural integrity check for a LeRobotDataset (v3.0) before you train or upload.

Answers the questions you cannot answer by watching videos: does the metadata agree
with the data, is every referenced video file actually present, and did the camera
pipeline ever drop a frame.

    python -m phi.utils.verify_dataset ~/.cache/huggingface/lerobot/<user>/<dataset>
    python -m phi.utils.verify_dataset <root> --split 59     # per-block stats either side of an index

Exit code 0 = safe, 1 = failures found. Run it before `hf upload` and after any
`lerobot-edit-dataset` operation, which rewrites chunk files.

Why per-file globbing matters: data, episode metadata and videos are all sharded into
several files. Reading only the first shard makes a complete dataset look broken — that
false alarm is the reason this exists as a script instead of a one-off snippet.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq


def _cat(pattern: str, column: str) -> np.ndarray:
    """Concatenate one column across every parquet shard matching `pattern`."""
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        raise FileNotFoundError(f"no parquet files matched {pattern}")
    return np.concatenate([np.array(pq.read_table(f).column(column)) for f in files])


def _dir_size_mb(root: str) -> float:
    return sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(root) for f in fs) / 1e6


def verify(root: str, split: int | None = None) -> int:
    root = os.path.expanduser(root.rstrip("/"))
    info_path = f"{root}/meta/info.json"
    if not os.path.exists(info_path):
        print(f"❌ no meta/info.json under {root}", file=sys.stderr)
        return 1

    info = json.load(open(info_path))
    fps, n_ep, n_frames = info["fps"], info["total_episodes"], info["total_frames"]
    cams = [k for k in info["features"] if "image" in k]
    fail: list[str] = []
    warn: list[str] = []

    data_glob = f"{root}/data/**/*.parquet"
    meta_glob = f"{root}/meta/episodes/**/*.parquet"

    ep = _cat(data_glob, "episode_index")
    ts = _cat(data_glob, "timestamp").astype(float)
    frame_idx = _cat(data_glob, "frame_index")
    global_idx = _cat(data_glob, "index")

    # metadata agrees with data
    if len(ep) != n_frames:
        fail.append(f"data rows {len(ep)} != info.total_frames {n_frames}")
    if sorted(set(ep.tolist())) != list(range(n_ep)):
        fail.append("episode_index is not contiguous 0..total_episodes-1")
    if not np.array_equal(global_idx, np.arange(len(global_idx))):
        fail.append("global 'index' column is not contiguous")
    n_meta = len(_cat(meta_glob, "episode_index"))
    if n_meta != n_ep:
        fail.append(f"meta/episodes rows {n_meta} != info.total_episodes {n_ep}")

    # per-episode: length and frame timing
    lengths = np.array([int((ep == e).sum()) for e in range(n_ep)])
    empty = [int(e) for e in np.where(lengths == 0)[0]]
    if empty:
        fail.append(f"empty episodes (0 frames): {empty}")
    short = [(int(e), int(lengths[e])) for e in range(n_ep) if 0 < lengths[e] < 2 * fps]
    if short:
        warn.append(f"episodes under 2 s (not real demonstrations, delete them): {short}")

    late_eps = []
    for e in range(n_ep):
        mask = ep == e
        if mask.sum() < 2:
            continue
        if (np.diff(ts[mask]) > 1.5 / fps).sum():
            late_eps.append(int(e))
        if not np.array_equal(frame_idx[mask], np.arange(mask.sum())):
            fail.append(f"ep{e}: frame_index is not 0..n-1")

    # every referenced video file exists; flag files nothing points at
    meta: dict[str, list] = {}
    for f in sorted(glob.glob(meta_glob, recursive=True)):
        for k, v in pq.read_table(f).to_pydict().items():
            meta.setdefault(k, []).extend(v)

    print("=== video integrity ===")
    for cam in cams:
        refs = set(zip(meta[f"videos/{cam}/chunk_index"], meta[f"videos/{cam}/file_index"]))
        disk = {
            (int(p.split("chunk-")[1][:3]), int(p.split("file-")[1][:3]))
            for p in glob.glob(f"{root}/videos/{cam}/**/*.mp4", recursive=True)
        }
        line = f"  {cam.split('.')[-1]:6s} referenced {sorted(refs)}  on disk {sorted(disk)}"
        if refs - disk:
            line += f"  ❌ MISSING {sorted(refs - disk)}"
            fail.append(f"{cam}: referenced video files missing {sorted(refs - disk)}")
        if disk - refs:
            line += f"  ⚠️ ORPHAN {sorted(disk - refs)}"
            warn.append(f"{cam}: video files on disk that no episode references: {sorted(disk - refs)}")
        print(line)
        # timestamps inside one video file must increase with episode order
        for ch, fl in sorted(refs):
            sel = [
                j for j in range(n_ep)
                if (meta[f"videos/{cam}/chunk_index"][j], meta[f"videos/{cam}/file_index"][j]) == (ch, fl)
            ]
            froms = [meta[f"videos/{cam}/from_timestamp"][j] for j in sel]
            if froms != sorted(froms):
                fail.append(f"{cam} chunk{ch:03d}/file{fl:03d}: from_timestamp not monotonic")

    stats = json.load(open(f"{root}/meta/stats.json"))
    missing_stats = [k for k in set(cams) | {"action", "observation.state"} if k not in stats]
    if missing_stats:
        fail.append(f"meta/stats.json missing keys: {missing_stats}")

    print(f"\n=== {os.path.basename(root)} ===")
    print(f"episodes {n_ep} · frames {n_frames} · {n_frames / fps / 60:.1f} min · "
          f"{_dir_size_mb(root):.0f} MB · {info.get('total_tasks')} task(s)")
    print(f"cameras  {[c.split('.')[-1] for c in cams]}   robot {info.get('robot_type')}   "
          f"codebase {info.get('codebase_version')}")
    print(f"lengths  min {lengths.min()} max {lengths.max()} mean {lengths.mean():.0f} frames "
          f"({lengths.mean() / fps:.1f} s)")
    print(f"timing   episodes with any frame later than 1.5x interval: {len(late_eps)} {late_eps[:12]}")
    if split is not None and 0 < split < n_ep:
        a, b = lengths[:split], lengths[split:]
        print(f"block A  ep 0-{split - 1:<4} frames {a.sum():>6}  mean {a.mean() / fps:.1f} s")
        print(f"block B  ep {split}-{n_ep - 1:<4} frames {b.sum():>6}  mean {b.mean() / fps:.1f} s")

    print()
    for w in warn:
        print("⚠️ ", w)
    if fail:
        print("\n❌ FAILURES:\n  " + "\n  ".join(fail))
        return 1
    print("\n✅ ALL CHECKS PASS")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="dataset directory (the one containing meta/, data/, videos/)")
    ap.add_argument("--split", type=int, default=None,
                    help="episode index where a second recording block starts, for per-block stats")
    args = ap.parse_args(argv)
    try:
        return verify(args.root, args.split)
    except FileNotFoundError as e:
        print(f"❌ {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

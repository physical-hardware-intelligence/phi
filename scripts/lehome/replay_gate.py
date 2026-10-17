#!/usr/bin/env python3
"""B6 replay gate and B5 latency for Ilia Larchenko's `lehome_real` policy server.

The server is only set up correctly if it predicts what the organizer's operator actually
did. So: send it frames from the organizer's own recordings, exactly as the real robot
client would, and compare the actions it returns with the recorded ones.

    replay_gate.py dry-run     --data data/lehome_real          # decode + build requests, no server
    replay_gate.py mock-server --port 8765                      # a protocol stand-in that holds still
    replay_gate.py run         --data data/lehome_real --server ws://localhost:8000
    replay_gate.py latency     --data data/lehome_real --server ws://localhost:8000   # B5 · M1
    replay_gate.py report      outputs/replay_gate/replay_gate_<stamp>   # post-hoc stats on a saved run

What goes on the wire is what `scripts/record_real_dagger.py` sends (lehome_solution @ ca73828,
`PolicyClient.get_action_chunk`, :352-366): the 12-joint state in `MOTOR_NAMES` order, the three
camera frames as raw uint8 RGB at native resolution under `observation.images.{top_rgb,left_rgb,
right_rgb}` (from `right_front`, `left_wrist`, `right_wrist`), its `INFERENCE_CONFIG`, and a fixed
`garment_type_id`, so the client-side garment bootstrap (:386-403) never engages. No
`initial_actions`: every frame is a fresh session. With the real-round flags
(`--actions_to_execute 5`) the server returns the 5 actions the robot would execute.

**The test must be able to fail**, so each control breaks one thing the gate exists to catch:
C1 swaps the wrist cameras, C2 rotates the top camera 180 deg (the sim convention), C3 pairs each
state with another episode's images, and C4 scales the arm joints by 0.9 (the units-bug size; it
is reported, not graded). Two pass rules, both fixed in the plan before the run they grade:
  v1 (run 1): policy error < 0.5 x the "hold" baseline, and each of C1-C3 raises the error > 1.5 x.
  v2 (default; fixed 2026-09-28 19:40 PDT, after run 1 failed v1): policy < 0.5 x hold, and each of
      C1-C3 is reliably worse than the policy: bootstrap 95% CI of the ratio above 1.0, exact sign
      test p < 0.01, more samples worse than better. v2 is graded on episodes run 1 never used.

Reproducibility: the server's noise starts at `jax.random.key(0)` on launch and splits once per
call (openpi `policies/policy.py:65`), and requests go out in a fixed order, one at a time. A
freshly started server reproduces every number. Needs: numpy, pyarrow, av (PyAV), websockets.

Plan: experiments/2026-09-27_learning-to-fold-so101.md, Track B, B5 and B6.
"""
from __future__ import annotations

import argparse
import base64
import datetime
import glob
import json
import os
import statistics
import sys
import time

import numpy as np

# --- Copied from lehome_solution @ ca73828; each line cites its source. -----------------------
GARMENT_TYPES = ("top_long", "top_short", "pant_long", "pant_short")  # constants.py:9, id = index
MOTOR_NAMES = (                                                        # real_robot_config.py:30
    "left_shoulder_pan.pos", "left_shoulder_lift.pos", "left_elbow_flex.pos",
    "left_wrist_flex.pos", "left_wrist_roll.pos", "left_gripper.pos",
    "right_shoulder_pan.pos", "right_shoulder_lift.pos", "right_elbow_flex.pos",
    "right_wrist_flex.pos", "right_wrist_roll.pos", "right_gripper.pos",
)
INFERENCE_CONFIG = {"noise_temperature": 0.9, "cfg_scale": 5.0,        # record_real_dagger.py:305
                    "time_threshold_inpaint": 0.5}
WIRE = {                                                               # record_real_dagger.py:358
    "right_front": "observation.images.top_rgb",
    "left_wrist": "observation.images.left_rgb",
    "right_wrist": "observation.images.right_rgb",
}
# -------------------------------------------------------------------------------------------------

N_ACT = 5                     # actions per reply with --actions_to_execute 5
ARM = [i for i, n in enumerate(MOTOR_NAMES) if "gripper" not in n]   # C4 scales these
CONDITIONS = ("policy", "C1_swap_wrists", "C2_top_rot180", "C3_other_episode", "C4_state_x0.9")
GRADED = ("C1_swap_wrists", "C2_top_rot180", "C3_other_episode")
HOLD_RATIO_MAX = 0.5          # policy error must be below 0.5 x hold
CONTROL_RATIO_MIN = 1.5       # each graded control must raise the error above 1.5 x policy


# --- Data: organizer LeRobot v3.0 subsets, read with pyarrow + PyAV (no lerobot needed) ---------

def _subset_dirs(data_root: str) -> list[tuple[str, str]]:
    out = []
    for g in GARMENT_TYPES:
        d = os.path.join(data_root, f"{g}_merged")
        if not os.path.isfile(os.path.join(d, "meta", "info.json")):
            raise SystemExit(f"missing subset {d} (expected all of {GARMENT_TYPES})")
        out.append((g, d))
    return out


def _episodes(subset: str) -> list[dict]:
    import pyarrow.parquet as pq
    rows = []
    for f in sorted(glob.glob(os.path.join(subset, "meta", "episodes", "chunk-*", "file-*.parquet"))):
        t = pq.read_table(f, columns=None)
        keep = [c for c in t.column_names if not c.startswith("stats/")]
        rows += t.select(keep).to_pylist()
    return sorted(rows, key=lambda r: r["episode_index"])


def _decode(path: str, timestamps: list[float], fps: float) -> dict[float, np.ndarray]:
    """Decode the frames at `timestamps` (seconds) from one video file, as uint8 RGB."""
    import av
    tol = 0.25 / fps   # a quarter frame: catches any off-by-one
    out = {}
    with av.open(path) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        for ts in sorted(timestamps):
            c.seek(int(ts / s.time_base), stream=s, backward=True, any_frame=False)
            got = None
            for fr in c.decode(s):
                if fr.time is not None and fr.time >= ts - tol:
                    got = fr
                    break
            if got is None or abs(got.time - ts) > tol:
                raise RuntimeError(f"{path}: no frame at t={ts:.4f}s (got {None if got is None else got.time})")
            out[ts] = got.to_ndarray(format="rgb24")
    return out


def load_samples(data_root: str, eps_per_garment: int, frames_per_episode: int,
                 episode_list: list[int] | None = None) -> list[dict]:
    """Episodes per garment (evenly spaced, or exactly `episode_list`), evenly spaced frames, fully checked."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    samples = []
    for gid, (garment, subset) in enumerate(_subset_dirs(data_root)):
        info = json.load(open(os.path.join(subset, "meta", "info.json")))
        fps = float(info["fps"])
        for k in ("observation.state", "action"):
            if tuple(info["features"][k]["names"]) != MOTOR_NAMES:
                raise SystemExit(f"{subset}: {k} joint names differ from the robot client's MOTOR_NAMES")
        eps = _episodes(subset)
        if episode_list:
            pos = {ep["episode_index"]: i for i, ep in enumerate(eps)}
            missing = [e for e in episode_list if e not in pos]
            if missing:
                raise SystemExit(f"{garment}: no episodes {missing} (has {len(eps)})")
            pick = [pos[e] for e in episode_list]
        else:
            pick = sorted(set(np.linspace(0, len(eps) - 1, eps_per_garment).round().astype(int).tolist()))
        for slot_e, ei in enumerate(pick):
            ep = eps[ei]
            e, length = ep["episode_index"], ep["length"]
            f = info["data_path"].format(chunk_index=ep["data/chunk_index"], file_index=ep["data/file_index"])
            tab = pq.read_table(os.path.join(subset, f),
                                columns=["episode_index", "frame_index", "timestamp", "observation.state", "action"])
            tab = tab.filter(pc.equal(tab["episode_index"], e)).sort_by("frame_index")
            fidx = np.asarray(tab["frame_index"].to_pylist())
            if len(fidx) != length or not np.array_equal(fidx, np.arange(length)):
                raise RuntimeError(f"{garment} ep{e}: frame_index is not 0..{length - 1}")
            ts_col = np.asarray(tab["timestamp"].to_pylist(), dtype=np.float64)
            if np.abs(ts_col - fidx / fps).max() > 1e-3:
                raise RuntimeError(f"{garment} ep{e}: timestamps are not frame_index / fps")
            state = np.asarray(tab["observation.state"].to_pylist(), dtype=np.float32)
            action = np.asarray(tab["action"].to_pylist(), dtype=np.float32)
            ts = np.linspace(0, length - N_ACT, frames_per_episode).round().astype(int).tolist()
            images = {}
            for cam in WIRE:
                key = f"observation.images.{cam}"
                vf = info["video_path"].format(video_key=key, chunk_index=ep[f"videos/{key}/chunk_index"],
                                               file_index=ep[f"videos/{key}/file_index"])
                t0 = ep[f"videos/{key}/from_timestamp"]
                got = _decode(os.path.join(subset, vf), [t0 + t / fps for t in ts], fps)
                want = tuple(info["features"][key]["shape"])
                images[cam] = [got[t0 + t / fps] for t in ts]
                if images[cam][0].shape != want:
                    raise RuntimeError(f"{garment} ep{e} {cam}: decoded {images[cam][0].shape}, info.json says {want}")
            for slot_t, t in enumerate(ts):
                samples.append({
                    "garment": garment, "garment_type_id": gid, "episode": e, "t": t,
                    "slot": (slot_e, slot_t),
                    "state": state[t], "actions": action[t:t + N_ACT],
                    "images": {cam: images[cam][slot_t] for cam in WIRE},
                })
        print(f"  {garment:10s} id {gid}: episodes {[eps[i]['episode_index'] for i in pick]}, "
              f"{frames_per_episode} frames each", flush=True)
    return samples


# --- Requests ------------------------------------------------------------------------------------

def _b64(img: np.ndarray) -> dict:                                     # record_real_dagger.py:312
    img = np.ascontiguousarray(img)
    return {"base64": base64.b64encode(img.tobytes()).decode("ascii"),
            "shape": list(img.shape), "dtype": str(img.dtype)}


def _images_for(cond: str, s: dict, by_slot: dict) -> dict[str, np.ndarray]:
    im = dict(s["images"])
    if cond == "C1_swap_wrists":
        im["left_wrist"], im["right_wrist"] = s["images"]["right_wrist"], s["images"]["left_wrist"]
    elif cond == "C2_top_rot180":
        im["right_front"] = np.rot90(s["images"]["right_front"], 2)
    elif cond == "C3_other_episode":
        slot_e, slot_t = s["slot"]
        n_e = 1 + max(k[1] for k in by_slot if k[0] == s["garment"])
        im = by_slot[(s["garment"], (slot_e + 1) % n_e, slot_t)]["images"]   # same garment, next episode
    return im


def build_request(cond: str, s: dict, by_slot: dict) -> dict:
    state = s["state"].copy()
    if cond == "C4_state_x0.9":
        state[ARM] *= 0.9
    im = _images_for(cond, s, by_slot)
    msg = {"type": "infer_chunk", "session_id": "replay_gate", "observation.state": state.tolist()}
    for cam, wire_key in WIRE.items():
        msg[wire_key] = _b64(im[cam])
    msg["inference_config"] = dict(INFERENCE_CONFIG)
    msg["garment_type_id"] = int(s["garment_type_id"])
    return msg


def _connect(url: str, compression: str | None = None):
    """Ilia's client settings (record_real_dagger.py:337), except compression.

    websockets compresses every message by default (permessage-deflate). Our 6 MB of images barely
    compress, and deflating them cost ~0.3-0.5 s per request. With compression=None the server falls
    back to plain frames: identical bytes reach the model, so predictions cannot change.
    """
    from websockets.sync.client import connect
    return connect(url, max_size=200 * 1024 * 1024, open_timeout=30, compression=compression)


def _infer(ws, msg: dict) -> tuple[np.ndarray, float]:
    body = json.dumps(msg)
    t0 = time.perf_counter()
    ws.send(body)
    resp = json.loads(ws.recv())
    dt = time.perf_counter() - t0
    if "error" in resp:
        raise RuntimeError(f"server error: {resp['error']}")
    a = np.asarray(resp["actions"], dtype=np.float32)
    if a.shape != (N_ACT, len(MOTOR_NAMES)):
        raise RuntimeError(f"expected actions {(N_ACT, len(MOTOR_NAMES))}, got {a.shape}: check the server flags")
    return a, dt


# --- Commands ------------------------------------------------------------------------------------

def _err(pred: np.ndarray, s: dict) -> np.ndarray:
    """Mean |pred - recorded| over the 5 steps, per joint: degrees, gripper in 0-100 units."""
    return np.abs(pred - s["actions"]).mean(axis=0)


def _hold(s: dict) -> np.ndarray:
    return np.broadcast_to(s["state"], (N_ACT, len(MOTOR_NAMES)))


def paired_stats(ctl: np.ndarray, pol: np.ndarray, boot: int = 10000, seed: int = 0) -> dict:
    """Control vs policy on per-sample errors (each averaged over passes and joints).

    ratio of means with a bootstrap 95% CI over samples, and an exact two-sided sign test on the
    per-sample differences (ties dropped, as the sign test requires).
    """
    import math
    n = len(pol)
    idx = np.random.default_rng(seed).integers(0, n, size=(boot, n))
    lo, hi = np.percentile(ctl[idx].mean(axis=1) / pol[idx].mean(axis=1), [2.5, 97.5])
    d = ctl - pol
    worse, better = int((d > 0).sum()), int((d < 0).sum())
    m, k = worse + better, max(worse, better)
    p = min(1.0, 2 * sum(math.comb(m, i) for i in range(k, m + 1)) * 0.5 ** m)
    return {"ratio": float(ctl.mean() / pol.mean()), "ci95": [float(lo), float(hi)],
            "worse": worse, "better": better, "sign_p": float(p)}


def _index(samples: list[dict]) -> dict:
    return {(s["garment"],) + tuple(s["slot"]): s for s in samples}


def _comp(a) -> str | None:
    return None if a.compression == "none" else a.compression


def cmd_dry_run(a) -> int:
    samples = load_samples(a.data, a.episodes, a.frames, a.episode_list)
    by_slot = _index(samples)
    sizes = {c: len(json.dumps(build_request(c, samples[0], by_slot))) for c in CONDITIONS}
    hold = np.stack([_err(_hold(s), s) for s in samples])
    print(f"\n{len(samples)} samples · request size {sizes['policy'] / 1e6:.2f} MB "
          f"(all conditions: {sorted(set(round(v / 1e6, 2) for v in sizes.values()))} MB)")
    print(f"hold baseline, mean |state[t] - action[t..t+4]|: {hold.mean():.3f}")
    for g in GARMENT_TYPES:
        m = np.stack([_err(_hold(s), s) for s in samples if s["garment"] == g])
        print(f"  {g:10s} {m.mean():.3f}   per joint: " + " ".join(f"{v:.2f}" for v in m.mean(axis=0)))
    return 0


def cmd_run(a) -> int:
    if (len(a.episode_list) if a.episode_list else a.episodes) < 2:
        raise SystemExit("need >= 2 episodes per garment: C3 borrows images from another episode of the same garment")
    samples = load_samples(a.data, a.episodes, a.frames, a.episode_list)
    by_slot = _index(samples)
    hold = np.stack([_err(_hold(s), s) for s in samples])            # (n, 12), deterministic
    errs = {c: np.zeros((a.passes, len(samples), len(MOTOR_NAMES))) for c in CONDITIONS}
    preds = {c: np.zeros((a.passes, len(samples), N_ACT, len(MOTOR_NAMES)), np.float32) for c in CONDITIONS}
    t_req = []
    total, done, t_start = a.passes * len(CONDITIONS) * len(samples), 0, time.time()
    with _connect(a.server, _comp(a)) as ws:
        for p in range(a.passes):
            for c in CONDITIONS:
                for i, s in enumerate(samples):
                    pred, dt = _infer(ws, build_request(c, s, by_slot))
                    preds[c][p, i], errs[c][p, i] = pred, _err(pred, s)
                    t_req.append(dt)
                    done += 1
                    if done % 50 == 0 or done == total:
                        print(f"  {done}/{total}  pass {p + 1} {c:17s}  median t_req {statistics.median(t_req) * 1e3:.0f} ms"
                              f"  elapsed {time.time() - t_start:.0f} s", flush=True)

    # Summary: overall = mean over samples and joints; criteria on the mean over passes.
    hold_all = float(hold.mean())
    per_pass = {c: [float(errs[c][p].mean()) for p in range(a.passes)] for c in CONDITIONS}
    mean = {c: float(np.mean(per_pass[c])) for c in CONDITIONS}
    r_hold = mean["policy"] / hold_all
    r_ctrl = {c: mean[c] / mean["policy"] for c in CONDITIONS if c != "policy"}
    v1 = {"policy < 0.5 x hold": r_hold < HOLD_RATIO_MAX}
    v1.update({f"{c} > 1.5 x policy": r_ctrl[c] > CONTROL_RATIO_MIN for c in GRADED})
    pol_s = errs["policy"].mean(axis=(0, 2))                          # per sample, over passes + joints
    stats = {c: paired_stats(errs[c].mean(axis=(0, 2)), pol_s) for c in CONDITIONS if c != "policy"}
    v2 = {"policy < 0.5 x hold": r_hold < HOLD_RATIO_MAX}
    for c in GRADED:
        st = stats[c]
        v2[f"{c}: CI95 above 1.0, sign p < 0.01, more worse than better"] = (
            st["ci95"][0] > 1.0 and st["sign_p"] < 0.01 and st["worse"] > st["better"])
    checks = v2 if a.rule == "v2" else v1
    verdict = "PASS" if all(checks.values()) else "FAIL"

    print(f"\nhold baseline {hold_all:.3f}")
    for c in CONDITIONS:
        ratio = f"{r_hold:.3f} x hold" if c == "policy" else f"{r_ctrl[c]:.2f} x policy"
        print(f"  {c:17s} {mean[c]:.3f}  ({ratio})   passes: " + " ".join(f"{v:.3f}" for v in per_pass[c]))
    for c, st in stats.items():
        print(f"  {c:17s} CI95 [{st['ci95'][0]:.2f}, {st['ci95'][1]:.2f}]  worse {st['worse']} / better {st['better']}"
              f"  sign p={st['sign_p']:.1e}")
    for name, rule in (("v1", v1), ("v2", v2)):
        tag = "GRADED" if name == a.rule else "reference only"
        print(f"  rule {name} ({tag}):")
        for k, ok in rule.items():
            print(f"    {'ok  ' if ok else 'FAIL'} {k}")
    print(f"  per joint, policy: " + " ".join(f"{v:.2f}" for v in errs["policy"].mean(axis=(0, 1))))
    print(f"  per joint, hold:   " + " ".join(f"{v:.2f}" for v in hold.mean(axis=0)))
    print(f"  t_req median {statistics.median(t_req) * 1e3:.0f} ms, p95 {np.percentile(t_req, 95) * 1e3:.0f} ms"
          f" over {len(t_req)} requests\n  -> {verdict}")

    os.makedirs(a.out, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    record = {
        "verdict": verdict, "rule": a.rule, "checks": checks, "checks_v1": v1, "checks_v2": v2,
        "paired_stats": stats, "server": a.server, "data": os.path.abspath(a.data),
        "passes": a.passes, "compression": a.compression, "episode_list": a.episode_list, "episodes_per_garment": a.episodes,
        "frames_per_episode": a.frames,
        "inference_config": INFERENCE_CONFIG, "hold": hold_all, "mean": mean, "per_pass": per_pass,
        "ratio_policy_to_hold": r_hold, "ratio_control_to_policy": r_ctrl,
        "per_joint": {"names": MOTOR_NAMES, "hold": hold.mean(axis=0).tolist(),
                      **{c: errs[c].mean(axis=(0, 1)).tolist() for c in CONDITIONS}},
        "t_req_s": {"median": statistics.median(t_req), "p95": float(np.percentile(t_req, 95)), "n": len(t_req)},
        "samples": [{"garment": s["garment"], "episode": s["episode"], "t": s["t"]} for s in samples],
        "note": "reproducible from a freshly started server: noise starts at jax.random.key(0)",
    }
    base = os.path.join(a.out, f"replay_gate_{stamp}")
    with open(base + ".json", "w") as fh:
        json.dump(record, fh, indent=1)
    np.savez_compressed(base + ".npz", hold=hold, t_req=np.asarray(t_req),
                        **{f"pred_{c}": preds[c] for c in CONDITIONS}, **{f"err_{c}": errs[c] for c in CONDITIONS})
    print(f"  saved {base}.json and .npz")
    return 0 if verdict == "PASS" else 1


def cmd_report(a) -> int:
    """Post-hoc view of a saved run. Not the pre-registered gate: that verdict is `run`'s.

    Per garment: policy / hold. Per control: the ratio with a bootstrap 95% CI over samples, the
    share of samples the control made worse (paired, with an exact two-sided sign test), and the
    ratio per arm. Seed noise: how much the policy's own error moves between passes.
    """
    rec = json.load(open(a.base + ".json"))
    z = np.load(a.base + ".npz")
    garments = np.array([s["garment"] for s in rec["samples"]])
    hold = z["hold"]                                               # (n, 12)
    e = {c: z[f"err_{c}"] for c in CONDITIONS}                     # (passes, n, 12)
    pol = e["policy"].mean(axis=(0, 2))                            # (n,) per-sample policy error
    n, passes = len(pol), e["policy"].shape[0]
    print(f"{a.base}  ({n} samples x {passes} passes)  -- post-hoc, not the pre-registered verdict")
    print("per garment: policy / hold")
    for g in GARMENT_TYPES:
        m = garments == g
        print(f"  {g:10s} {pol[m].mean():.3f} / {hold[m].mean():.3f} = {pol[m].mean() / hold[m].mean():.3f}")
    if passes > 1:
        pp = e["policy"].mean(axis=2)                              # (passes, n)
        noise = np.mean([np.abs(pp[i] - pp[j]).mean() for i in range(passes) for j in range(i + 1, passes)])
        print(f"seed noise: mean |pass_i - pass_j| of a sample's policy error = {noise:.3f} "
              f"(vs policy error {pol.mean():.3f})")
    left, right = slice(0, 6), slice(6, 12)
    print(f"controls vs policy (bootstrap {a.boot} over samples; sign test on per-sample means):")
    for c in CONDITIONS[1:]:
        st = paired_stats(e[c].mean(axis=(0, 2)), pol, a.boot)
        arm = [e[c][:, :, s].mean() / e["policy"][:, :, s].mean() for s in (left, right)]
        print(f"  {c:17s} ratio {st['ratio']:.2f} [{st['ci95'][0]:.2f}, {st['ci95'][1]:.2f}]  worse {st['worse']}"
              f" / better {st['better']} (sign test p={st['sign_p']:.1e})  left arm {arm[0]:.2f} · right arm {arm[1]:.2f}")
    return 0


def cmd_latency(a) -> int:
    samples = load_samples(a.data, 1, 1)
    msg = build_request("policy", samples[0], _index(samples))
    with _connect(a.server, _comp(a)) as ws:
        for _ in range(a.warmup):
            _infer(ws, msg)
        t = [_infer(ws, msg)[1] for _ in range(a.n)]
    t_ms = np.asarray(t) * 1e3
    print(f"[compression {a.compression}] t_req over {a.n} calls after {a.warmup} warm-up: median {np.median(t_ms):.1f} ms, "
          f"p95 {np.percentile(t_ms, 95):.1f} ms, min {t_ms.min():.1f}, max {t_ms.max():.1f}; "
          f"request {len(json.dumps(msg)) / 1e6:.2f} MB")
    return 0


def cmd_mock_server(a) -> int:
    """Speaks the server's protocol and returns 'hold' (the state, 5 times). The gate must FAIL on it."""
    import asyncio
    from websockets.asyncio.server import serve

    async def handle(ws):
        async for message in ws:
            d = json.loads(message)
            if d.get("type") == "ping":
                await ws.send(json.dumps({"status": "ok"}))
                continue
            try:
                for key in WIRE.values():                              # same checks the real server's
                    v = d[key]                                          # decode + _process_obs would hit
                    np.frombuffer(base64.b64decode(v["base64"]), dtype=v["dtype"]).reshape(v["shape"])
                st = np.asarray(d["observation.state"], dtype=np.float32)
                if st.shape != (len(MOTOR_NAMES),):
                    raise ValueError(f"state shape {st.shape}")
                acts = np.broadcast_to(st, (N_ACT + 1, len(MOTOR_NAMES)))
                await ws.send(json.dumps({"actions": acts[:N_ACT].tolist(), "next_initial_actions": acts[N_ACT:].tolist(),
                                          "execute_in_n_steps": N_ACT, "actions_to_keep": 1}))
            except Exception as e:  # noqa: BLE001 - report any malformed request to the client
                await ws.send(json.dumps({"error": f"{type(e).__name__}: {e}"}))

    async def main():
        async with serve(handle, "localhost", a.port, max_size=100 * 1024 * 1024):
            print(f"mock server (hold) on ws://localhost:{a.port}", flush=True)
            await asyncio.Future()

    asyncio.run(main())
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def data_args(p, eps=5, frames=10):
        p.add_argument("--data", required=True, help="folder holding the four *_merged subsets")
        p.add_argument("--episodes", type=int, default=eps, help="episodes per garment")
        p.add_argument("--frames", type=int, default=frames, help="frames per episode")
        p.add_argument("--episode-list", type=int, nargs="+", default=None,
                       help="exact episode indices per garment (overrides --episodes)")

    p = sub.add_parser("dry-run", help="decode, build requests, report sizes and the hold baseline")
    data_args(p)
    p = sub.add_parser("run", help="the B6 gate against a live server")
    data_args(p)
    p.add_argument("--server", default="ws://localhost:8000")
    p.add_argument("--passes", type=int, default=3)
    p.add_argument("--compression", choices=["none", "deflate"], default="none",
                   help="websocket compression; 'deflate' is the library default Ilia's client used")
    p.add_argument("--rule", choices=["v1", "v2"], default="v2",
                   help="v1: controls > 1.5x (run 1). v2: pre-registered 2026-09-28 19:40 PDT (see the plan's B6)")
    p.add_argument("--out", default="outputs/replay_gate")
    p = sub.add_parser("latency", help="B5 · M1: t_req on one fixed request")
    p.add_argument("--data", required=True)
    p.add_argument("--server", default="ws://localhost:8000")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--compression", choices=["none", "deflate"], default="none")
    p.add_argument("--n", type=int, default=50)
    p = sub.add_parser("mock-server", help="protocol stand-in that holds still")
    p.add_argument("--port", type=int, default=8765)
    p = sub.add_parser("report", help="post-hoc: per garment, paired controls, bootstrap CIs")
    p.add_argument("base", help="saved run path without extension, e.g. outputs/replay_gate/replay_gate_<stamp>")
    p.add_argument("--boot", type=int, default=10000)
    a = ap.parse_args()
    return {"dry-run": cmd_dry_run, "run": cmd_run, "latency": cmd_latency, "mock-server": cmd_mock_server,
            "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())

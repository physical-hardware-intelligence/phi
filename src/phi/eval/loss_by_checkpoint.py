"""HELD-OUT L1 ACROSS EVERY CHECKPOINT OF ONE RUN, so you can pick one.

    python -m phi.eval.loss_by_checkpoint \
        --run /scratch/$USER/phi-results/pi05_cubcyl_lang \
        --max-batches 40 --out outputs/pi05_cubcyl_ckpts.csv

WHY THIS EXISTS SEPARATELY FROM act_loss_by_horizon
---------------------------------------------------
`act_loss_by_horizon` answers "which of these two RUNS is better", scoring one
checkpoint each on a common horizon. This answers "which STEP of one run should I
put on the arm", scoring every checkpoint on the identical holdout.

That question is not answerable from the training log when the logger dies, which
is what happened to pi05_cubcyl_lang (job 10369421): 30,000 steps completed and
exactly one eval_loss was recorded.

Every checkpoint carries its own train_config.json, and they all came from the
same run, so the holdout episodes are identical across them by construction. No
re-splitting, nothing to keep in sync.

POLICY-AGNOSTIC BY CONSTRUCTION
-------------------------------
Nothing here is π-specific. The policy class comes from `cfg.policy.type` and the
processors come from the checkpoint, so a saved `rename_map` (front -> base_0_rgb
for our π₀.₅ run) is applied automatically as preprocessor step 0. Feeding the
raw dataset keys by hand would silently miss it.

⚠️ WHAT THIS CANNOT TELL YOU
Held-out L1 is not success rate, and on this repo's own evidence the ordering can
invert: experiments/2026-08-12 has a checkpoint whose held-out loss rose 4x and
still matched a 100k-step model on the arm. Use this to decide which two or three
checkpoints are worth arm time. Rollouts decide.
"""

from __future__ import annotations

import argparse
import csv
import gc
from pathlib import Path

import torch


def _checkpoints(run: Path) -> list[Path]:
    """Every numbered checkpoint in a run, ascending. Skips the `last` symlink."""
    d = run / "checkpoints" if (run / "checkpoints").is_dir() else run
    out = [p for p in d.iterdir() if p.is_dir() and not p.is_symlink() and p.name.isdigit()]
    if not out:
        raise SystemExit(f"no numbered checkpoints under {d}")
    return sorted(out, key=lambda p: int(p.name))


def _load(ckpt: Path, device: str):
    import draccus
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.datasets.factory import make_train_eval_datasets
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    pm = ckpt / "pretrained_model" if (ckpt / "pretrained_model").is_dir() else ckpt
    with open(pm / "train_config.json") as f:
        cfg = draccus.load(TrainPipelineConfig, f)

    policy = get_policy_class(cfg.policy.type).from_pretrained(pm).to(device).eval()
    _, eval_ds = make_train_eval_datasets(cfg)
    if eval_ds is None:
        raise SystemExit(f"{ckpt}: run has no eval split")

    # 🚨 Load the processors FROM THE CHECKPOINT, not from the policy config. The
    #    saved preprocessor carries the rename_map; rebuilding it from scratch
    #    drops that step and the policy then looks for cameras the batch does not
    #    have. Same class of silent failure as passing a wrong rename_map, which
    #    disables validate_visual_features_consistency (factory.py:649).
    pre, _ = make_pre_post_processors(
        cfg.policy, pretrained_path=pm,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return cfg, policy, eval_ds, pre


@torch.no_grad()
def score(ckpt: Path, horizon: int, batch_size: int, max_batches: int, device: str):
    from lerobot.utils.constants import ACTION

    cfg, policy, ds, pre = _load(ckpt, device)
    h = min(horizon, cfg.policy.chunk_size)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)

    tot = torch.zeros(h, dtype=torch.float64)
    cnt = torch.zeros(h, dtype=torch.float64)

    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        # The dataset yields uint8 images. VISUAL normalization is IDENTITY for
        # π₀.₅, so nothing downstream converts them and the vision tower would be
        # fed integers. Converting here is a no-op when they are already float.
        batch = {k: (v.float() / 255.0 if (torch.is_tensor(v) and v.dtype == torch.uint8) else v)
                 for k, v in batch.items()}
        batch = pre(batch)
        pred = policy.predict_action_chunk(batch)[:, :h]
        gt = batch[ACTION][:, :h]
        pred = pred[..., : gt.shape[-1]]          # π pads actions to 32; score the real dims

        l1 = (pred.float() - gt.float()).abs().mean(dim=-1)
        pad = batch.get("action_is_pad")
        keep = (~pad[:, :h]).float() if pad is not None else torch.ones_like(l1)

        tot += (l1 * keep).sum(dim=0).double().cpu()
        cnt += keep.sum(dim=0).double().cpu()

    curve = (tot / cnt.clamp(min=1)).tolist()
    del policy, pre, ds
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return curve


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="a run dir, or its checkpoints/ dir")
    ap.add_argument("--horizon", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-batches", type=int, default=40,
                    help="cap batches per checkpoint (0 = whole holdout). The cap is "
                         "identical across checkpoints, so comparisons stay fair.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    ckpts = _checkpoints(args.run)
    print(f"\nscoring {len(ckpts)} checkpoints on the same holdout "
          f"({args.max_batches or 'all'} batches x {args.batch_size})\n")
    print(f"  {'step':>8} {'mean L1':>10} {'first 10':>10} {'last 10':>10}")

    rows, summary = [], []
    for c in ckpts:
        curve = score(c, args.horizon, args.batch_size, args.max_batches, args.device)
        mean = sum(curve) / len(curve)
        head = sum(curve[:10]) / min(10, len(curve))
        tail = sum(curve[-10:]) / min(10, len(curve))
        summary.append((int(c.name), mean, head, tail))
        print(f"  {c.name:>8} {mean:>10.5f} {head:>10.5f} {tail:>10.5f}", flush=True)
        rows += [{"step": int(c.name), "horizon_index": i, "l1": v} for i, v in enumerate(curve)]

    best = min(summary, key=lambda r: r[1])
    print(f"\n  lowest held-out L1: step {best[0]} ({best[1]:.5f})")
    if best[0] != summary[-1][0]:
        last = summary[-1]
        print(f"  the LAST checkpoint ({last[0]}) is {(last[1]/best[1]-1)*100:.1f}% worse — "
              f"the run peaked early. Do not reflexively deploy the final step.")
    print("\n  Held-out L1 is not success rate. Use this to choose what gets arm time.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["step", "horizon_index", "l1"])
            w.writeheader()
            w.writerows(rows)
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()

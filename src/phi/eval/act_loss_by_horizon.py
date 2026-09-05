"""Held-out ACT L1 BROKEN OUT BY POSITION IN THE ACTION CHUNK.

    # compare two runs over their common horizon
    python -m phi.eval.act_loss_by_horizon \
        --checkpoints /scratch/$USER/phi-results/act_pen_chunk50/checkpoints/100000 \
                      /scratch/$USER/phi-results/act_pen_chunk100/checkpoints/100000 \
        --horizon 50 --out outputs/act_pen_horizon.csv

WHY THE LOGGED `eval_loss` CANNOT RANK TWO CHUNK SIZES
------------------------------------------------------
ACT's loss is L1 averaged over the WHOLE predicted chunk. At 30 fps:

    chunk 50   averages predictions out to 1.67 s
    chunk 100  averages predictions out to 3.33 s

The far end of a longer horizon is intrinsically harder to predict, so a
chunk-100 run carries a structural penalty that has nothing to do with whether
its policy is better on the robot. Comparing the two logged numbers ranks
horizon length as much as it ranks the policies.

This script removes that confound: it evaluates every checkpoint on the SAME
first `--horizon` predicted steps, so the comparison is like for like. It also
prints the per-index curve, which is the more interesting object — it shows
where each policy's prediction actually degrades.

⚠️ NOT `eval_by_k.py`. That script's `k` is the DIFFUSION NOISE LEVEL and only
   means anything for Diffusion Policy. ACT has no noise schedule. The two
   scripts answer unrelated questions.

WHAT THIS STILL CANNOT TELL YOU
-------------------------------
Held-out L1 on teleop data is not success rate. See experiments/2026-08-12 for
this project's own cautionary case. Use this to choose which checkpoints are
worth putting on the arm, not to declare a winner. Rollouts decide.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch


def _load(checkpoint: Path, device: str):
    """Load policy + its own held-out split + the processors SAVED IN THE CHECKPOINT.

    Using the checkpoint's own train_config.json matters: it carries the
    eval_split that produced the holdout, so two runs trained on the same
    dataset are scored on exactly the same episodes.
    """
    import draccus
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.datasets.factory import make_train_eval_datasets
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    pm = checkpoint / "pretrained_model" if (checkpoint / "pretrained_model").is_dir() else checkpoint
    with open(pm / "train_config.json") as f:
        cfg = draccus.load(TrainPipelineConfig, f)

    policy = get_policy_class(cfg.policy.type).from_pretrained(pm).to(device).eval()

    _, eval_dataset = make_train_eval_datasets(cfg)
    if eval_dataset is None:
        raise SystemExit(f"{checkpoint}: run has no eval split (eval_split must be > 0)")

    # 🚨 In lerobot 0.6.0 normalization lives in a SEPARATE processor pipeline, not
    #    inside the policy. Feeding a raw dataset batch to the policy reports a
    #    wildly wrong loss. See experiments/2026-08-04.
    preprocessor, _ = make_pre_post_processors(
        cfg.policy, pretrained_path=pm,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return cfg, policy, eval_dataset, preprocessor


@torch.no_grad()
def loss_by_horizon(checkpoint: Path, horizon: int, batch_size: int, device: str):
    from lerobot.utils.constants import ACTION

    cfg, policy, ds, pre = _load(checkpoint, device)
    chunk = cfg.policy.chunk_size
    if chunk < horizon:
        raise SystemExit(f"{checkpoint}: chunk_size {chunk} < --horizon {horizon}")

    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)

    # accumulate |pred - gt| and a count per horizon index, so padded steps never
    # contribute to the mean
    tot = torch.zeros(horizon, dtype=torch.float64)
    cnt = torch.zeros(horizon, dtype=torch.float64)

    for batch in loader:
        batch = pre(batch)
        pred = policy.predict_action_chunk(batch)[:, :horizon]     # [B, H, D]
        gt = batch[ACTION][:, :horizon]                            # [B, H, D]

        l1 = (pred - gt).abs().mean(dim=-1)                        # [B, H]
        # ACT masks padded steps in its own loss; do the same or the tail is junk
        pad = batch.get("action_is_pad")
        keep = (~pad[:, :horizon]).float() if pad is not None else torch.ones_like(l1)

        tot += (l1 * keep).sum(dim=0).double().cpu()
        cnt += keep.sum(dim=0).double().cpu()

    per_index = (tot / cnt.clamp(min=1)).tolist()
    return chunk, per_index, cfg.policy.n_action_steps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    ap.add_argument("--horizon", type=int, default=50,
                    help="compare over the first N predicted steps (default: 50)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, help="write the per-index curve to this CSV")
    args = ap.parse_args()

    rows, summary = [], []
    for ckpt in args.checkpoints:
        chunk, curve, n_act = loss_by_horizon(ckpt, args.horizon, args.batch_size, args.device)
        mean = sum(curve) / len(curve)
        summary.append((ckpt, chunk, mean, curve))
        rows += [{"checkpoint": str(ckpt), "chunk_size": chunk, "horizon_index": i, "l1": v}
                 for i, v in enumerate(curve)]

    # a checkpoint dir is named for its STEP (".../act_pen_chunk50/checkpoints/100000"),
    # so ckpt.name is "100000" for every run. Label with the run name + step instead.
    def label(c: Path) -> str:
        parts = [p for p in c.parts if p not in ("checkpoints", "pretrained_model")]
        return "/".join(parts[-2:]) if len(parts) >= 2 else c.name

    width = max(len(label(c)) for c, *_ in summary) + 2
    print(f"\nL1 on the SAME first {args.horizon} predicted steps (lower is better)\n")
    print(f"  {'checkpoint':<{width}} {'chunk':>6} {'mean L1':>10} {'step 0':>9} {'step ' + str(args.horizon - 1):>9}")
    for ckpt, chunk, mean, curve in summary:
        print(f"  {label(ckpt):<{width}} {chunk:>6} {mean:>10.5f} {curve[0]:>9.5f} {curve[-1]:>9.5f}")

    if len(summary) == 2:
        (a, ca, ma, _), (b, cb, mb, _) = summary
        better, worse = (a, ma, ca), (b, mb, cb)
        if mb < ma:
            better, worse = (b, mb, cb), (a, ma, ca)
        gap = (worse[1] - better[1]) / worse[1] * 100
        print(f"\n  chunk {better[2]} is {gap:.1f}% lower on the common horizon.")
        print("  This is a like-for-like number. It is still not a success rate — roll out.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["checkpoint", "chunk_size", "horizon_index", "l1"])
            w.writeheader()
            w.writerows(rows)
        print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()

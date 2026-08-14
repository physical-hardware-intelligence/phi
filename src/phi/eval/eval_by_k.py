"""Held-out diffusion loss BUCKETED BY NOISE LEVEL k, instead of averaged over it.

    python -m phi.eval.eval_by_k \
        --checkpoint /scratch/$USER/phi/results/dpt_unreg_emb256/checkpoints/008000 \
        --out outputs/k_eval_unreg_8000.csv

    # sweep a whole run
    python -m phi.eval.eval_by_k --sweep /scratch/$USER/phi/results/dpt_unreg_emb256 \
        --steps 2000,8000,16000,30000 --out outputs/k_eval_unreg.csv

WHY THE AVERAGED NUMBER CANNOT ANSWER THE QUESTION
--------------------------------------------------
`compute_loss` draws ONE k per sample, uniformly from [0, num_train_timesteps),
and the reported `eval_loss` averages over that draw. But the job being scored is
wildly different at each k. Measured on this project's schedule (100 steps,
squaredcos_cap_v2):

    k=0    signal 0.9997   noise 0.0251   recovering eps means dividing by
                                          0.0251, so any residual uncertainty
                                          about the action is amplified ~40x
    k=50   signal 0.6916   noise 0.7223   balanced
    k=99   signal 0.0005   noise 1.0000   the input essentially IS the noise;
                                          predicting it is nearly copying it

So one scalar blends a nearly-free sub-problem with a brutally hard one, and it
CANNOT say which regime moved. That matters because the hard regime -- low k,
fine action detail -- is the one that decides whether the gripper closes on a
25 mm cube, and "reaches correctly, fails to close" is exactly the DP failure
mode scored on the arm on 2026-08-12.

It also matters because on that date a DP checkpoint whose averaged eval_loss had
risen 4x still scored 50% / 0.650 on the arm, tied with a 100k-step ACT model.
Held-out noise-MSE did not predict task success. This script exists to find out
whether the averaging is why.

WHAT MAKES THIS A MEASUREMENT AND NOT A PLOT
--------------------------------------------
1. 🔑 THE SAME eps IS REUSED ACROSS EVERY BUCKET. eps is drawn from a generator
   seeded per batch index, so bucket k=5 and bucket k=95 see the IDENTICAL noise
   tensor, and so does every other checkpoint you run. Without this the noise
   draw confounds both the across-k and the across-checkpoint comparison.
2. 🔑 VARIANCE EXPLAINED IS REPORTED ALONGSIDE RAW MSE. Predicting zero gives
   MSE = E[eps^2] = 1, so `1 - MSE` is the fraction of the noise the model
   actually explains. Without that normalisation the curve just reproduces the
   schedule shape (high MSE at low k, low at high k) and says nothing about the
   model. Read the normalised column.
3. The dataset split, the preprocessor and the loss are lerobot's own -- via
   `make_train_eval_datasets` and the preprocessor SAVED IN THE CHECKPOINT -- so
   a k-averaged run of this script reproduces training's `eval_loss`, and the
   numbers are directly comparable to the curves in experiments/.

Works on both backbones: the CNN (`diffusion`) and the transformer
(`diffusion_transformer`), since both expose `.unet` behind the same signature.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

import phi.policies.dp_transformer  # noqa: F401  -- registers diffusion_transformer

DEFAULT_BUCKETS = "0,2,5,10,15,20,30,40,50,60,70,80,90,95,99"


def _stack_images(policy, batch: dict) -> dict:
    """Reproduce DiffusionPolicy.forward's image stacking, which compute_loss expects."""
    from lerobot.constants import OBS_IMAGES

    if not policy.config.image_features:
        return batch
    batch = dict(batch)
    for key in policy.config.image_features:
        if policy.config.n_obs_steps == 1 and batch[key].ndim == 4:
            batch[key] = batch[key].unsqueeze(1)
    batch[OBS_IMAGES] = torch.stack([batch[key] for key in policy.config.image_features], dim=-4)
    return batch


@torch.no_grad()
def loss_at_k(model, batch: dict, k: int, eps: torch.Tensor,
              global_cond: torch.Tensor) -> torch.Tensor:
    """`compute_loss` with the timestep PINNED to k and the noise supplied.

    Everything else is copied from lerobot's compute_loss so the value is the same
    quantity training reports -- only the sampling of k and eps is taken over.
    Returns one loss per sample, so buckets can be averaged with correct weights.

    🔑 `global_cond` is passed IN, not computed here. It is the output of the three
    ResNet-18 encoders and does not depend on k, so hoisting it out of the bucket
    loop turns 15 buckets from 15x(vision + denoiser) into 1x vision + 15x
    denoiser. Vision is the dominant cost -- 33,591,264 params on 384 images per
    batch against the transformer backbone's 9,020,934 -- so this is most of the
    runtime, not a micro-optimisation.
    """
    from lerobot.constants import ACTION

    trajectory = batch[ACTION]
    ts = torch.full((trajectory.shape[0],), k, dtype=torch.long, device=trajectory.device)
    noisy = model.noise_scheduler.add_noise(trajectory, eps, ts)
    pred = model.unet(noisy, ts, global_cond=global_cond)

    if model.config.prediction_type == "epsilon":
        target = eps
    elif model.config.prediction_type == "sample":
        target = trajectory
    else:
        raise ValueError(f"unsupported prediction_type {model.config.prediction_type}")

    return F.mse_loss(pred, target, reduction="none").mean(dim=tuple(range(1, pred.ndim)))


def evaluate(checkpoint: Path, buckets: list[int], device: str, max_batches: int | None,
             seed: int) -> list[dict]:
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.datasets.factory import make_train_eval_datasets
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    import draccus

    pm = checkpoint / "pretrained_model" if (checkpoint / "pretrained_model").is_dir() else checkpoint
    cfg_path = pm / "train_config.json"
    if not cfg_path.is_file():
        raise SystemExit(f"no train_config.json under {pm}")
    with open(cfg_path) as f:
        cfg = draccus.load(TrainPipelineConfig, f)

    policy = get_policy_class(cfg.policy.type).from_pretrained(pm)
    policy.to(device).eval()

    _, eval_dataset = make_train_eval_datasets(cfg)
    if eval_dataset is None:
        raise SystemExit("this run has no eval split; eval_split must be > 0")

    preprocessor, _ = make_pre_post_processors(
        cfg.policy,
        pretrained_path=pm,
        preprocessor_overrides={"device_processor": {"device": device}},
    )

    loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, drop_last=False,
        pin_memory=(device == "cuda"),
    )

    model = policy.diffusion
    sums = {k: 0.0 for k in buckets}
    counts = {k: 0 for k in buckets}

    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        from lerobot.constants import ACTION

        for key in list(batch.keys()):
            if torch.is_tensor(batch[key]) and batch[key].dtype == torch.uint8:
                batch[key] = batch[key].to(dtype=torch.float32) / 255.0
        batch = preprocessor(batch)
        batch = _stack_images(policy, batch)

        # 🔑 one eps per BATCH, reused for every bucket and every checkpoint.
        g = torch.Generator(device="cpu").manual_seed(seed + bi)
        eps = torch.randn(batch[ACTION].shape, generator=g).to(batch[ACTION].device)

        # vision runs ONCE per batch: global_cond has no k dependence.
        with torch.no_grad():
            global_cond = model._prepare_global_conditioning(batch)

        for k in buckets:
            per_sample = loss_at_k(model, batch, k, eps, global_cond)
            sums[k] += float(per_sample.sum())
            counts[k] += per_sample.numel()
        print(f"  batch {bi + 1}: {counts[buckets[0]]} samples scored", end="\r", flush=True)

    print()
    rows = []
    for k in buckets:
        mse = sums[k] / max(counts[k], 1)
        rows.append({
            "checkpoint": str(checkpoint),
            "k": k,
            "mse": round(mse, 6),
            # predicting zero gives MSE = E[eps^2] = 1, so this is the fraction explained
            "var_explained": round(1.0 - mse, 6),
            "n": counts[k],
        })
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, help="a checkpoints/<step> dir (or its pretrained_model)")
    p.add_argument("--sweep", type=Path, help="a run dir; scores several of its checkpoints")
    p.add_argument("--steps", default=None, help="with --sweep: comma list of steps, else all")
    p.add_argument("--buckets", default=DEFAULT_BUCKETS,
                   help=f"comma list of k values (default: {DEFAULT_BUCKETS})")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-batches", type=int, default=None, help="cap batches, for a smoke test")
    p.add_argument("--seed", type=int, default=1000, help="seeds eps; keep FIXED across checkpoints")
    p.add_argument("--out", type=Path, default=None, help="write CSV here")
    args = p.parse_args()

    buckets = [int(x) for x in args.buckets.split(",")]
    targets: list[Path] = []
    if args.sweep:
        d = args.sweep / "checkpoints"
        want = set(args.steps.split(",")) if args.steps else None
        for c in sorted(d.iterdir()):
            if c.name == "last" or not c.is_dir():
                continue
            if want is None or c.name.lstrip("0") in {w.lstrip("0") for w in want}:
                targets.append(c)
    elif args.checkpoint:
        targets = [args.checkpoint]
    else:
        raise SystemExit("need --checkpoint or --sweep")

    all_rows = []
    for t in targets:
        print(f"\n=== {t} ===")
        rows = evaluate(t, buckets, args.device, args.max_batches, args.seed)
        all_rows += rows
        print(f"  {'k':>4} {'mse':>10} {'var explained':>14}")
        for r in rows:
            print(f"  {r['k']:>4} {r['mse']:10.5f} {r['var_explained']:14.5f}")
        avg = sum(r["mse"] for r in rows) / len(rows)
        print(f"  mean over buckets: {avg:.5f}   (comparable to training's eval_loss "
              f"only if buckets are uniform over [0,100))")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["checkpoint", "k", "mse", "var_explained", "n"])
            w.writeheader()
            w.writerows(all_rows)
        print(f"\nwrote {len(all_rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()

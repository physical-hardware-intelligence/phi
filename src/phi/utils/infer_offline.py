"""Run a policy on RECORDED episodes. No arm, no cameras, no risk.

Inference without hardware. Feed a held-out episode's real observations to the
policy frame by frame and compare what it commands against what the human
actually did, in DEGREES rather than in normalised loss units.

    python -m phi.utils.infer_offline \
        --model BrutalCaesar/act_so101_cubcyl_recovery_chunk50_noaug_3cam \
        --episode 0

    # compare the A/B pair on the same episode
    --model BrutalCaesar/act_so101_cubcyl_recovery_chunk50_aug_3cam
    --model BrutalCaesar/act_so101_cubcyl_recovery_chunk50_aug_3cam --revision step-80000

Roughly 10 s for 120 frames on an M-series Mac. Use episodes from the HOLDOUT
(0-4, 20-24, 45-49, 65-69, 90-94, 110-114) — anything else was trained on and
will flatter the model.

WHAT THIS IS FOR, AND WHAT IT IS NOT
------------------------------------
`eval_loss=0.2052` is not a quantity anyone can picture. This turns the same
prediction into "the shoulder is N degrees away from where the human put it",
which you can reason about against a 25 mm cube.

It is NOT a substitute for a rollout. The policy never moves the arm here, so
every observation it sees is a real one from the human demo. On the real robot
its own errors change what it sees next, and that feedback is exactly where
imitation policies fail. A good number here is necessary, not sufficient.

TWO MODES, AND WHY THE GAP BETWEEN THEM IS THE INTERESTING PART
---------------------------------------------------------------
    queue   what the robot actually does. ACT predicts `chunk_size` actions and
            executes `n_action_steps` of them before looking again, so most
            commands come from an observation up to 1.67 s old.
    fresh   re-plan on every single frame. The policy's one-step accuracy, with
            no staleness.

`queue` minus `fresh` is the cost of open-loop execution. Measured on episode 0
of the recovery dataset, 2026-08-11, the queue-mode error grows monotonically
with distance from the last re-plan:

    steps  0-9 after a re-plan   16.1 deg
    steps 10-19                  22.8 deg
    steps 20-29                  31.8 deg
    steps 30-39                  36.2 deg
    steps 40-49                  37.2 deg

That is a 2.3x spread inside one chunk, and it is the argument for lowering
`n_action_steps` at inference time — which needs no retraining, since the policy
still predicts all 50 and simply discards more of the stale tail. See
`phi.utils.watch_rollouts --n-action-steps`.

🚨 UNRESOLVED, DO NOT QUOTE THE ABSOLUTE NUMBERS AS SETTLED. Converting
`eval_loss=0.2052` through the action normalisation (MEAN_STD; per-joint std
from dataset stats) implies about 6.4 deg averaged over joints, but the
`fresh`-mode measurement on episode 0 came out near 11.8 deg. Those should be
closer. Candidate causes not yet separated: eval_loss averages over all 30
holdout episodes while this reads one, eval_loss includes the KL term
(kl_weight=10) so its L1 share is smaller than the printed total, and eval_loss
averages over all chunk positions rather than position 0. Resolve before any of
this goes in a report.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
HOLDOUT = [*range(0, 5), *range(20, 25), *range(45, 50), *range(65, 70), *range(90, 95), *range(110, 115)]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--revision", default=None, help="HF branch/tag, e.g. step-80000")
    p.add_argument("--dataset", default="BrutalCaesar/phi_so101_cubes_cylinder_recovery_v1")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--frames", type=int, default=120, help="0 = the whole episode")
    args = p.parse_args()

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.act.modeling_act import ACTPolicy

    if args.episode not in HOLDOUT:
        print(f"  ⚠️  episode {args.episode} is NOT in the holdout — it was trained on, "
              f"so these numbers flatter the model.\n")

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    ds = LeRobotDataset(args.dataset, episodes=[args.episode])
    n = ds.num_frames if args.frames == 0 else min(args.frames, ds.num_frames)

    policy = ACTPolicy.from_pretrained(args.model, revision=args.revision).to(device).eval()
    cfg = PreTrainedConfig.from_pretrained(args.model, revision=args.revision)
    cfg.pretrained_path = args.model
    # The checkpoint bakes in device="cuda" from the H200 that trained it, and the
    # device_processor step hard-asserts torch.cuda.is_available(). Override or it
    # raises on any Mac. `lerobot-rollout` does the same thing internally.
    pre, post = make_pre_post_processors(
        cfg, pretrained_path=args.model, pretrained_revision=args.revision,
        preprocessor_overrides={"device_processor": {"device": device}},
    )

    print(f"  model    {args.model}{'@' + args.revision if args.revision else ''}")
    print(f"  episode  {args.episode} ({ds.num_frames} frames, reading {n})  device {device}")
    print(f"  chunk    {cfg.chunk_size}, n_action_steps {cfg.n_action_steps} "
          f"({cfg.n_action_steps / 30:.2f}s open loop)\n")

    def run(fresh: bool) -> np.ndarray:
        policy.reset()
        out = []
        for i in range(n):
            f = ds[i]
            if fresh:
                policy.reset()
            batch = {k: f[k].unsqueeze(0).to(device) for k in f if k.startswith("observation.")}
            batch["task"] = [f["task"]]
            with torch.no_grad():
                a = post(policy.select_action(pre(batch)))
            out.append(np.abs(a.squeeze(0).cpu().numpy() - f["action"].numpy()))
        return np.array(out)

    q, r = run(False), run(True)

    print(f"  |commanded - human|, mean over the episode")
    print(f"  {'joint':16s} {'queue (the robot)':>20s} {'fresh (1-step)':>18s} {'staleness cost':>16s}")
    print("  " + "-" * 72)
    for j, name in enumerate(JOINTS):
        unit = "" if name == "gripper" else "°"
        print(f"  {name:16s} {q[:, j].mean():18.2f}{unit:2s} {r[:, j].mean():16.2f}{unit:2s} "
              f"{q[:, j].mean() - r[:, j].mean():14.2f}")
    print(f"  {'ALL':16s} {q.mean():18.2f}   {r.mean():16.2f}   {q.mean() - r.mean():14.2f}")

    step = cfg.n_action_steps
    print(f"\n  queue error by position within the {step}-action chunk (arm joints only):")
    pos = np.array([q[i, :5].mean() for i in range(n)])
    bucket = max(step // 5, 1)
    for lo in range(0, step, bucket):
        idx = [i for i in range(n) if lo <= (i % step) < lo + bucket]
        if idx:
            print(f"     steps {lo:3d}-{min(lo + bucket, step) - 1:3d} after a re-plan : {pos[idx].mean():6.2f}°")
    print("\n  Rising with distance from the re-plan = open-loop drift. Try "
          "`watch_rollouts --n-action-steps 15`.\n")


if __name__ == "__main__":
    main()

# Evaluation — the standard protocol + leaderboard

**This is Φ's differentiator.** A policy isn't "done" because training loss went down — it's done when it succeeds on the real arm under a **fixed, fair protocol**. Everyone evaluates the same way so numbers are comparable.

> Reference: [lerobot-rollout / inference](https://huggingface.co/docs/lerobot/en/inference)

## Run a policy on the arm
```bash
lerobot-rollout \
  --strategy.type=base \
  --policy.path=${HF_USER}/act-phi-cube-v1 \
  --robot.type=so101_follower --robot.port=/dev/tty.usbmodem58760431541 --robot.id=phi_follower \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30} }" \
  --task="Grab the black cube" \
  --duration=60
```
`--strategy.type`: `base` (autonomous) · `sentry` (record + auto-upload eval episodes) · `dagger` (human-in-the-loop) · `episodic`. For slow VLAs (pi0/SmolVLA) add `--inference.type=rtc` (Real-Time Chunking).
⚠️ Camera config **must match** what was used at recording time.

## The Φ evaluation protocol (do this for every reported number)
1. **Fixed trial count:** N = 20 rollouts (state it if different).
2. **Declared initial conditions:** list the object positions / start poses you test (e.g. "cube at 5 grid positions × 4 trials").
3. **Binary success rubric written *before* eval** — one sentence, unambiguous (e.g. "cube lifted ≥5 cm and held 2 s"). Put it in the [task card](../../tasks/TEMPLATE.md).
4. **Report:** success rate (k/N), + median time-to-success, + a one-line failure-mode note.
5. **Reproducibility:** the exact `lerobot-rollout` command, the policy Hub id + revision, and the seed, in the [experiment write-up](../../experiments/).

## Leaderboard
Each task keeps a results table (in its task card and mirrored here). Columns:

| Policy | Dataset | Steps | Success (k/N) | Median time | Notes | Model card |
|---|---|---|---|---|---|---|
| _e.g._ ACT | phi-cube-v1 | 20k | — | — | baseline | — |

The point: a new member can see "ACT got 14/20 on cube-v1" and try to beat it — that's the ladder in action.

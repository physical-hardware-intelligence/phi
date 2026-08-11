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

### 🚨 Precondition: use the repo's calibration, not your own

**Before any rollout that produces a number**, make sure the machine is running the committed calibration:

```bash
cp configs/calibration/robots/so_follower/phi_follower.json \
   ~/.cache/huggingface/lerobot/calibration/robots/so_follower/
python -m phi.utils.compare_calibration phi_follower <whatever-you-were-using>
```

A policy emits joint angles in **the calibration frame it was trained in**. Run it against a different one and there is **no error and no warning** — the arm just reaches to the wrong place, and it looks exactly like a bad policy. **We hit this for real on 2026-08-10**: a cubes/cylinder policy trained on data recorded on Yash's laptop mis-grasped on Sai's machine, and copying Yash's calibration file over fixed it with no retraining. Full explanation: [so-arm101 setup §3c](../robots/so-arm101/02-setup.md#3c-moving-a-trained-policy-to-someone-elses-machine).

⇒ **Any leaderboard number recorded under a non-canonical calibration is void.** State the calibration you used alongside the number.

### Just want to watch it move?

`lerobot-rollout` runs until you stop it. To step through episodes one at a time with a scene reset between each, and no recording or scoring:

```bash
lerobot-find-port          # run this FIRST, on its own, and read the port off it

python -m phi.utils.watch_rollouts \
  --model BrutalCaesar/act_so101_cubcyl_recovery_chunk50_noaug_3cam \
  --port /dev/tty.usbmodemXXXX \
  --cameras wrist=0,top=1,front=2 \
  --n-action-steps 15
```

> 🚨 **Never write `--port $(lerobot-find-port)`.** That tool is interactive — it prints "Remove the USB cable … and press Enter" and blocks on `input()`. Command substitution swallows the prompt, so the terminal goes completely silent and looks hung forever. On success it also prints four lines, so `$(...)` would capture prose, not a port. Run it separately and paste the value.

`--n-action-steps 15` makes the arm re-plan every 0.5 s instead of every 1.67 s. Measured open-loop drift within one 50-action chunk (`phi.utils.infer_offline`, episode 0, 2026-08-11) rose from 16.1° at steps 0-9 to 37.2° at steps 40-49, so most of the apparent error is staleness rather than the model. It is an inference-time setting and needs no retraining.

It enforces the calibration precondition above and makes you eyeball the camera → key mapping before the arm energises, then rate-limits each joint to 15°/tick so a wrong first action creeps instead of slams. **It produces no numbers by design** — for those use `phi.utils.eval_rollouts`, which owns the partial-credit rubric.

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

# Deployment — on-robot, remote, and edge inference

Training a policy is only half the job. Deployment is about **running the policy where the robot is**, efficiently and reliably.

> Reference: [inference](https://huggingface.co/docs/lerobot/en/inference) · [async inference](https://huggingface.co/docs/lerobot/en/async) · [Real-Time Chunking](https://huggingface.co/docs/lerobot/en/rtc)

## 1. On-robot — run a policy from your terminal

On the machine the arm is plugged into (the Mac cockpit, device `mps`). Fine for ACT and Diffusion Policy.

```bash
source configs/ports.local.sh        # exports $FOLLOWER_PORT / $FOLLOWER_ID

lerobot-rollout \
  --policy.path=BrutalCaesar/act_so101_pen_chunk50_3cam \
  --policy.device=mps \
  --robot.type=so101_follower --robot.port=$FOLLOWER_PORT --robot.id=$FOLLOWER_ID \
  --robot.cameras="{ \
    wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: MJPG}, \
    front: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30, fourcc: MJPG}, \
    top:   {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: MJPG} }" \
  --task="pick up the pen and place it in the container" \
  --strategy.type=base \
  --duration=60
```

> **Never** paste a `/dev/tty.usbmodem…` literal from these docs. Ports differ per laptop and change per USB port. Always `source configs/ports.local.sh`. ([why](../robots/so-arm101/02-setup.md#3-calibrate))

### Choosing the model — `--policy.path`

One flag, two forms:

| | |
|---|---|
| **Hub** | `BrutalCaesar/act_so101_pen_chunk50_3cam` — downloaded and cached |
| **Local** | `checkpoints/<run>/checkpoints/<step>/pretrained_model` |

Training saves every `save_freq` steps, so a local run holds several. **The last checkpoint is not automatically the best.** We have a case where held-out loss rose 4× and the policy still matched a 100k-step model on the arm. Choose by rollout, not by step number.

### There is no dataset to choose

This is the part that confuses people. **At inference you do not load a dataset.** The model already carries everything it took from one:

| what it carries | where it lives |
|---|---|
| normalization statistics | `policy_preprocessor_*.safetensors` |
| which camera keys it expects | `config.json` |
| the calibration frame its joint angles mean something in | nowhere — it is implicit, which is why it bites |

`--dataset.repo_id` at rollout is an **output**: where `--strategy.type=sentry` writes the eval episodes it records. Omit it for a plain run.

So the real question is not *which dataset* but **does what I am feeding it match what it was trained on?** Three things must match, and all three fail silently, with no error and no warning:

| must match | how to check before the arm energises |
|---|---|
| camera key → physical camera | `--display_data=true` and look at each window |
| calibration frame | `python -m phi.utils.compare_calibration` ([why](../evaluation/README.md)) |
| control rate | `--fps` (default 30) against the dataset's fps |

The camera one is not hypothetical: `saimaligi/pen_pick_and_place_20260817_162529` has its three keys rotated by one, so a policy trained on it needs remapping at deployment.

### The one knob worth tuning

`--policy.n_action_steps` — how much of each predicted chunk to execute before re-planning. **Inference-time only, no retraining** (it does not appear in `compute_loss`). Lower is more reactive and costs more compute; the chunk tail is always the least accurate part. Start at `Tp/2`. It has a hard deadline to meet — see [§4](#-4-the-control-rate-budget--measured-and-it-is-tight).

### Strategies

`--strategy.type` = `base` (autonomous, no recording) · `sentry` (records + uploads eval episodes) · `dagger` (human takes over) · `episodic` · `highlight`. For slow VLAs (π₀, SmolVLA) add `--inference.type=rtc`.

### Just want to watch it move, safely?

```bash
python -m phi.utils.watch_rollouts --model <hub-id-or-path> --port $FOLLOWER_PORT \
  --cameras wrist=0,top=1,front=2 --n-action-steps 15
```

Resets the scene between episodes, makes you eyeball the camera mapping first, and rate-limits each joint to 15°/tick so a wrong first action creeps instead of slams. **Produces no numbers by design** — for scored rollouts use `phi.utils.eval_rollouts` and the [evaluation protocol](../evaluation/README.md).

## 2. Remote inference (big models, small robot computer)
Split it: a lightweight **client** on the robot machine streams observations to an **inference server** on a GPU box that runs the policy and streams actions back. Use **async inference** so the control loop isn't blocked, and **RTC** (`--inference.type=rtc`) so slow VLAs (pi0/SmolVLA) stay smooth. See the async + RTC docs above.

## 3. Edge inference (the Φ research edge)
Getting a VLA to run in real time on a **small on-robot GPU** (e.g. Jetson, or a 6 GB card) is an open, valuable problem — and it's where the club has a genuine research angle (ties to member edge-inference work: quantization, distillation, speculative/low-NFE sampling, latency budgets).

Roadmap for this folder (Phase 4):
- quantize a trained policy (INT8/INT4) and measure success-rate vs. latency trade-off
- measure the control-frequency budget on the target device
- document a "which policy runs at what Hz on what hardware" table

---

## 🚨 4. The control-rate budget — measured, and it is tight

This is the section to read **before** picking `n_action_steps` for any policy, and it closes the "control-frequency budget" roadmap item above.

### The rule

A chunked policy predicts `Tp` (`horizon`) actions and executes `Ta` (`n_action_steps`) of them before looking again. **You must produce the next chunk before the current one runs out.** That is a hard real-time deadline:

```
deadline = Ta / fps
```

An action index is dimensionless. It becomes time only when divided by fps — which is why the same config means different things at different control rates:

| | at 10 Hz | at 30 fps (ours) |
|---|---|---|
| 1 action | 100 ms | **33.3 ms** |
| `Ta` = 8 | 800 ms | **267 ms** |
| `Ta` = 24 | 2400 ms | **800 ms** |

### Measured on the Mac cockpit (MPS, fp32, batch 1), 2026-08-11

**Diffusion Policy**, `down_dims=(256,512,1024)` (68.7 M UNet), 3 cameras, `n_obs_steps=2`, crop 216×288, DDIM:

| component | cost | how often |
|---|---:|---|
| 3 × ResNet-18 encoders, 2 frames each | **74.6 ms** | once per chunk |
| UNet × DDIM-16 at `Tp=48` | **219.8 ms** | once per chunk |
| **total per chunk** | **294.4 ms** | |

⇒ **`Ta=8` (267 ms) MISSES the deadline. `Ta=24` (800 ms) fits with 506 ms of headroom.** Minimum viable `Ta` on this hardware is ≈ **10**.

**This is why our DP config deviates from the Diffusion Policy paper's Table 7.** The paper's real-world rows use `Ta=8`, but they ran at **10 Hz**, where `Ta=8` *is* 800 ms. Copying their step counts at 30 fps would have produced a policy that misses its deadline every single chunk. Matching their **duration** is the faithful choice; matching their **step count** is not.

### Why the denoising loop is the whole problem

| | UNet passes per chunk |
|---|---|
| ACT | **1** |
| Diffusion Policy | **`num_inference_steps`** (16 here, 100 at lerobot's default) |

At lerobot's default `num_inference_steps=None` → 100, the same UNet costs **~1370 ms** per chunk on this Mac. DDIM is not a nicety, it is the difference between deployable and not. The paper reports 0.1 s with DDIM-10 on a 3080 (§3.4) — our gap to that is the *device*, not the model.

### Latency vs UNet width, same Mac

| `down_dims` | UNet params | ms / denoise step | DDIM-16 | DDIM-100 |
|---|---:|---:|---:|---:|
| (512, 1024, 2048) — lerobot default | 259,095,686 | 105.2 | 1683 ms | 10,520 ms |
| **(256, 512, 1024) — paper's real-world 67 M** | 68,665,222 | 11.6 | **186 ms** | 1160 ms |
| (128, 256, 512) | 19,161,350 | 8.3 | 133 ms | 830 ms |

Note the **9× speedup for a 3.8× parameter cut** — the widest layers fall off a memory-bandwidth cliff on MPS. Do not assume latency scales with parameter count.

### What this means in practice

- **`Ta` is an inference-time knob.** `n_action_steps` does **not** appear in `compute_loss` (only `horizon` does), so you can sweep it at rollout without retraining. Train with a generous `Tp`, then tune `Ta` on the arm.
- **Missing the deadline does not raise an error.** `lerobot-rollout` logs `Record loop is running slower than the target FPS`. The arm stalls at its last commanded pose or stutters. Watch for that line.
- **Longer `Ta` trades reactivity for budget.** The chunk tail is always the least accurate part — measured on ACT, error grew from **16° at chunk steps 0-9 to 37° at steps 40-49**. Executing more of the chunk means executing more of the bad part. Standard practice is `Ta ≈ Tp/2`.
- **These numbers are the Mac.** On CUDA they will be several times better; the ranking between configurations should hold, the absolute values will not.

## The Φ rule
Any deployment result is reported with **hardware + control frequency (Hz) + latency + success rate** — not just "it ran."

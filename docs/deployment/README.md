# Deployment — on-robot, remote, and edge inference

Training a policy is only half the job. Deployment is about **running the policy where the robot is**, efficiently and reliably.

> Reference: [inference](https://huggingface.co/docs/lerobot/en/inference) · [async inference](https://huggingface.co/docs/lerobot/en/async) · [Real-Time Chunking](https://huggingface.co/docs/lerobot/en/rtc)

## 1. On-robot (simplest)
Run `lerobot-rollout` directly on the machine the arm is plugged into (the Mac cockpit for small policies like ACT; device `mps`). Good enough for ACT/Diffusion-scale models.

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

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

## The Φ rule
Any deployment result is reported with **hardware + control frequency (Hz) + latency + success rate** — not just "it ran."

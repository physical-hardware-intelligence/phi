# SO-ARM101 — Troubleshooting

Our growing corpus of "things that bit us." Add yours (PR or `good-first-issue`).

## USB / ports
- **Port not found / changes between plugs:** re-run `lerobot-find-port`; the mac name (`/dev/tty.usbmodem…`) can change per port/reboot.
- **Permission denied (Linux):** `sudo chmod 666 /dev/ttyACM0`.
- **CH340-based board on macOS:** may need a USB-serial driver; most boards enumerate natively as `usbmodem`.

## Motors
- **A motor won't take its id:** connect only that one when prompted; check the Waveshare jumpers are on channel **B (USB)**; verify the power supply.
- **Motor hot / stalled:** cut power — don't leave a stalled STS3215 energized. May need Feetech firmware update (LeRobot docs: `feetech`).

## Calibration / teleop
- **Follower doesn't mirror well:** re-calibrate; make sure you reused the same `--robot.id` / `--teleop.id` you calibrated with.
- **Arm jerks violently near full stretch:** you hit a **kinematic singularity** — the controller commands huge joint moves for a small tip motion. Avoid teleoping into full extension; keep tasks in the comfortable middle of the workspace.

## macOS (the cockpit)
- **`--policy.device=mps`** for any local policy run; the Mac does **not** train well (fanless throttling) — offload training.
- `ffmpeg` must be present (our env installs it) for TorchCodec video decode.

## CUDA (the training box)
- **CUDA OOM:** lower `--batch_size`; for VLAs use LoRA/PEFT (LeRobot `peft_training` doc).
- **Wrong PyTorch/CUDA pairing:** install a CUDA-matched PyTorch *before* `lerobot[training]`.

## Dataset
- **Corrupt dataset after Python creation:** you forgot `dataset.finalize()` before `push_to_hub()`.
- **Old dataset (v2.1):** migrate with `python -m lerobot.scripts.convert_dataset_v21_to_v30 --repo-id=<id>`.

## Version drift to watch
- On-robot inference is `lerobot-rollout` now; **v0.5.1 used `lerobot-record`** for inference.
- `pi0` docs page currently served from the `main` branch; extras are `[pi]` (not `[pi0]`).

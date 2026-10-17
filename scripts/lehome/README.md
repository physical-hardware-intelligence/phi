# `scripts/lehome/` — running Ilia Larchenko's LeHome policy on our SO-101s

Plan: [`experiments/2026-09-27_learning-to-fold-so101.md`](../../experiments/2026-09-27_learning-to-fold-so101.md).

| file | what it does |
|---|---|
| `record_real_dagger_mac.py` | **The robot program, on the Mac.** Ilia's runner, adapted (each change marked `[phi-mac]`). Reads both arms + 3 cameras, asks the model server for the next 5 moves, moves the arms, records episodes. You can take over with the leader arms (DAgger) |
| `real_robot_config.py` | Ilia's helper, vendored unchanged |
| `replay_gate.py` | Tests the model server without a robot: replay test (B6), latency (B5), a stand-in server |
| `verify_hub_download.py` | Proves a Hugging Face download is complete: every file's size and hash |
| `sim/setup_sim_box.sh` | Installs the simulator on the GPU box (Isaac Sim 5.1, IsaacLab fork, lehome). No sudo, stops on any error, accepts NVIDIA's license only with `ACCEPT_ISAACSIM_EULA=yes` |
| `sim/gpu_guard.sh` | Go/no-go before any GPU work on the shared box |
| `sim/run_sim_eval.sh` | One simulation evaluation, fully recorded (versions, GPU every 5 s, log) |
| `LICENSE.lehome_solution` | Ilia's Apache-2.0 license, covering the two files taken from his repo |

## Run the robot program

**Once:** copy `configs/lehome/real_robot_mac.example.yaml` to `real_robot_mac.local.yaml` and fill in your ports and camera indices. The `phi` env needs `websockets` (`pip install websockets`). macOS will ask to grant **Input Monitoring** to your terminal (for the keys).

**Start the model server first** (GPU box, `scripts/serve.py` with the real-round flags; see the plan, B4). Then from the repo root:

```bash
python scripts/lehome/record_real_dagger_mac.py --config configs/lehome/real_robot_mac.local.yaml \
  --robot_id phi_bi_follower --leader_id phi_bi_leader \
  --server ws://<gpu-box>:8000 --garment pant_short --dry_run
```

- **`--dry_run` first, every new setup:** everything runs, nothing moves. Recordings go to `<datasets_root>/_dryrun`.
- Drop `--dry_run` for real runs. Add `--max_relative_target 10` to cap each follower step (degrees) on early runs.
- `--manual_only`: teleop recording, no server needed.

**Keys** (any window): `Space` = robot ↔ you · `→` = save episode · `←` = discard · `Esc` = stop.

⚠️ **In autonomous mode the leader arms are powered and copy the followers** (so a takeover starts from the same pose). Keep hands clear of the leaders until you take over.

## Status (2026-10-03)

- ✅ Runs on the Mac (lerobot 0.6.0); every import resolves.
- ✅ Link to the real model, tested with demo frames: correct 5×12 moves back; garment auto-detect right for all 4 garments; a missing joint stops it with an error.
- ⏳ Not yet run with the arms and cameras: that needs the rig.
- ⚠️ Speed from the Mac: **2.46 s per answer through the Tailscale relay** (6 MB of images per request). Sending 224×224 images is the next fix (plan §6).

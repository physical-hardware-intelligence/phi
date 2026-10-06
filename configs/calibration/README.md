# Arm calibrations

There is one calibration per physical arm, and everyone uses these files. LeRobot reads them from
`~/.cache/huggingface/lerobot/calibration/`, or from `$HF_LEROBOT_CALIBRATION`, in the same
subfolders.

| File | Arm | macOS port (the USB board's serial) |
|---|---|---|
| `robots/so_follower/phi_bi_follower_left.json` | left follower (the original arm) | `/dev/tty.usbmodem5B7B0096441` |
| `robots/so_follower/phi_bi_follower_right.json` | right follower | `/dev/tty.usbmodem5B790809471` |
| `teleoperators/so_leader/phi_bi_leader_left.json` | left leader (the original arm) | `/dev/tty.usbmodem5B790807501` |
| `teleoperators/so_leader/phi_bi_leader_right.json` | right leader | `/dev/tty.usbmodem5B7B0139311` |

`phi_follower.json` and `phi_leader.json` hold the same numbers as the two left files. They use the
single-arm names that every trained checkpoint, script and doc expects, so keep them.

These are four different arms, checked on 2026-10-05: each joint's zero differs from the other arm's
by 12 to 168 degrees. The right follower's rest pose, read through its file, is a physically
consistent folded pose in simulation.

## Install on a machine

```bash
cp -R configs/calibration/robots configs/calibration/teleoperators ~/.cache/huggingface/lerobot/calibration/
```

## Rules

- **Copy these files. Don't calibrate on your own machine.** A new calibration on one laptop makes the
  arms disagree with every other laptop.
- **When LeRobot reports a "Mismatch between calibration values in the motor and the calibration file"
  and asks for ENTER, stop.** ENTER writes that file into the arm's servos. If the port points at the
  wrong arm, that arm now holds another arm's calibration: its angles are off by tens of degrees, and
  teleop drives joints into their stops. On 2026-10-05 the right follower held the left follower's
  calibration, and the left leader held the right leader's. Check the ports first.
- **Ports follow the USB board, not the arm.** If a board moves to another arm, update the table.

Note: the right follower's gripper still holds the auto-calibration from 2026-10-05, so its gripper
values differ from its file. On that arm's own port, ENTER at the mismatch prompt puts the file's
gripper values back, and that is safe.

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

Checked on 2026-10-08 against the arms themselves. Each arm, held in the middle pose, reads within
23 degrees of zero through its own file and 84 to 161 degrees off through the other side's. Every
arm's servos hold exactly its own file. The left follower's file also replays 120 recorded episodes
with the gripper at the table.

## Install on a machine

```bash
cp -R configs/calibration/robots configs/calibration/teleoperators ~/.cache/huggingface/lerobot/calibration/
cp configs/ports.local.sh.example configs/ports.local.sh   # the rig's ports and these ids
```

## Which file LeRobot uses

LeRobot picks the **file by the id** you pass and the **arm by the port**, and the two must name the
same arm. If they don't, it reports a mismatch (see Rules).

| You run | id flags | Files it reads |
|---|---|---|
| Bimanual (`bi_so_follower`, `bi_so_leader`) | `--robot.id=phi_bi_follower` `--teleop.id=phi_bi_leader` | LeRobot adds `_left`/`_right` itself: `phi_bi_follower_left/right.json`, `phi_bi_leader_left/right.json` |
| Single arm, **left** pair (`so101_follower`, `so101_leader`) | `--robot.id=phi_follower` `--teleop.id=phi_leader` | `phi_follower.json`, `phi_leader.json` (the left arms' numbers) |
| Single arm, **right** pair (`so101_follower`, `so101_leader`) | `--robot.id=phi_bi_follower_right` `--teleop.id=phi_bi_leader_right` | `phi_bi_follower_right.json`, `phi_bi_leader_right.json` |

Never run the right arm with `phi_follower`. That is the left arm's file, and ENTER at the mismatch
prompt would write it into the right arm.

## Commands

The variables come from `configs/ports.local.sh`. Load them first:

```bash
source configs/ports.local.sh
```

Bimanual:

```bash
lerobot-teleoperate \
  --robot.type=bi_so_follower --robot.id=$BI_FOLLOWER_ID \
  --robot.left_arm_config.port=$LEFT_FOLLOWER_PORT \
  --robot.right_arm_config.port=$RIGHT_FOLLOWER_PORT \
  --teleop.type=bi_so_leader --teleop.id=$BI_LEADER_ID \
  --teleop.left_arm_config.port=$LEFT_LEADER_PORT \
  --teleop.right_arm_config.port=$RIGHT_LEADER_PORT
```

Single arm, left pair:

```bash
lerobot-teleoperate \
  --robot.type=so101_follower --robot.port=$FOLLOWER_PORT --robot.id=$FOLLOWER_ID \
  --teleop.type=so101_leader  --teleop.port=$LEADER_PORT  --teleop.id=$LEADER_ID
```

Single arm, right pair:

```bash
lerobot-teleoperate \
  --robot.type=so101_follower --robot.port=$RIGHT_FOLLOWER_PORT --robot.id=$RIGHT_FOLLOWER_ID \
  --teleop.type=so101_leader  --teleop.port=$RIGHT_LEADER_PORT  --teleop.id=$RIGHT_LEADER_ID
```

`lerobot-record` takes the same `--robot.*` and `--teleop.*` flags, plus its dataset flags.

Recalibrate an arm only when it really needs it (for example, after a servo is replaced). Use that
arm's id, then commit the new file here, so every machine gets it:

```bash
lerobot-calibrate --robot.type=so101_follower --robot.port=$RIGHT_FOLLOWER_PORT --robot.id=$RIGHT_FOLLOWER_ID
lerobot-calibrate --teleop.type=so101_leader  --teleop.port=$RIGHT_LEADER_PORT  --teleop.id=$RIGHT_LEADER_ID
```

`lerobot-calibrate` overwrites that file on your machine and writes the result into the servos.

## Rules

- **Copy these files. Don't calibrate on your own machine.** A new calibration on one laptop makes the
  arms disagree with every other laptop.
- **When LeRobot reports a "Mismatch between calibration values in the motor and the calibration file"
  and asks for ENTER, stop.** ENTER writes that file into the arm's servos. If the port or the id
  names the wrong arm, that arm now holds another arm's calibration: its angles are off by tens of
  degrees, and teleop drives joints into their stops. On 2026-10-05 the right follower held the left
  follower's calibration. Until 2026-10-08 the two leader files were named after the wrong sides, so a
  single-arm run on the left pair offered the right leader's file to the left leader. Check the ports
  and ids against the tables above first.
- **Ports follow the USB board, not the arm.** If a board moves to another arm, update both tables
  and `configs/ports.local.sh.example`.

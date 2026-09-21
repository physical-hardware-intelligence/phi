# Set up Φ on your laptop

From nothing to a working environment. Budget 30 minutes, most of it downloads.

This page is the software. [Hardware bring-up](robots/so-arm101/02-setup.md) — ports,
motors, calibration, cameras — comes after, and assumes you finished this.

---

## 0. What you need first

| | why | check |
|---|---|---|
| **git** | clone the repo | `git --version` |
| **conda** (miniforge) | the env is a conda env, not a venv, because `ffmpeg` is a conda package | `conda --version` |
| **macOS 13+, Ubuntu 22.04+, or Windows 10/11** | all three are supported; see the platform notes below | — |
| ~15 GB free | env is ~8 GB, datasets add more | `df -h ~` |

No conda? Install **miniforge**, not Anaconda — smaller, conda-forge by default, which is
the channel this env uses.

```bash
# macOS (Apple Silicon)
brew install miniforge && conda init "$(basename "$SHELL")"
# Linux
wget -qO- https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh | bash
```

Windows: download the Miniforge3 Windows installer from the
[releases page](https://github.com/conda-forge/miniforge/releases/latest), then use the
**Miniforge Prompt** from the Start menu. To use PowerShell instead, run `conda init powershell`
once.

Open a **new terminal** afterwards. `conda init` edits your shell profile and the current shell
will not have it.

---

## 1. Get the code

```bash
git clone https://github.com/physical-hardware-intelligence/phi.git && cd phi
```

No submodules, nothing to recurse into. LeRobot is installed from pip at a pinned version,
not vendored.

---

## 2. Create the environment

```bash
conda env create -f env/environment.mac.yml     # Apple Silicon
conda env create -f env/environment.cuda.yml    # NVIDIA box
conda activate phi
pip install -e .
```

### What each piece is for

| package | why it is there |
|---|---|
| `python=3.12` | `pyproject.toml` sets `requires-python = ">=3.12"`; LeRobot 0.6.0 needs it |
| `ffmpeg` | **conda-level**, not pip. TorchCodec decodes dataset videos through it. Without it `LeRobotDataset` fails at load, not at install |
| `uv` | fast resolver for the pip block |
| `lerobot[...]==0.6.0` | the engine. Pinned deliberately |
| ↳ `feetech` | driver for the STS3215 servos. No arm without it |
| ↳ `core_scripts` | the `lerobot-calibrate` / `-teleoperate` / `-record` / `-rollout` CLIs |
| ↳ `training` | `lerobot-train` hard-requires `accelerate`, which only this extra pulls |
| `wandb` | training charts |
| `pip install -e .` | the thin Φ layer: `phi.utils.*`, `phi.eval.*`. Only pulls pyyaml/typer/rich — the heavy deps are the env's job |

**The pin is not cosmetic.** Mac and CUDA envs must be on the same LeRobot version or a
checkpoint trained on one will not load on the other. Change it in both files or neither.

Per-policy extras, installed only when you need them:

```bash
pip install "lerobot[smolvla]==0.6.0"     # SmolVLA
pip install "lerobot[pi]==0.6.0"          # π₀ / π₀.₅ — this is transformers + scipy
pip install "lerobot[diffusion]==0.6.0"   # Diffusion Policy
pip install peft grpcio                   # LoRA finetuning / remote inference
```

---

## 3. Verify it actually worked

Do not skip this. Every failure below is silent if you do.

```bash
conda activate phi
python -c "
import torch, lerobot, sys
print('python     ', sys.version.split()[0], '   <- must be 3.12.x')
print('which python', sys.executable, '   <- must contain envs/phi')
print('lerobot    ', lerobot.__version__, '   <- must be 0.6.0')
print('torch      ', torch.__version__)
print('accelerator', 'mps' if torch.backends.mps.is_available() else
                     ('cuda' if torch.cuda.is_available() else 'CPU ONLY'))
import torchcodec; print('torchcodec ', torchcodec.__version__, '  <- video decode OK')
"
lerobot-find-port --help >/dev/null && echo 'CLIs on PATH'
```

All six lines must be right. `CPU ONLY` on a Mac means the wrong Python — see below.

---

## 4. Hugging Face

Needed to pull datasets and pretrained policies. Public Φ datasets work unauthenticated but
rate-limit; π₀ needs auth because its tokenizer lives in a **gated** Google repo.

```bash
hf auth login
```

For π₀ or π₀.₅ you must also accept the licence at
<https://huggingface.co/google/paligemma-3b-pt-224> while signed in as that account, or
training dies with `GatedRepoError` at processor construction.

---

## 4.5 Windows

Run Φ **natively on Windows. Do not use WSL.** LeRobot 0.6.0 supports Windows directly, and
WSL2 has no native USB passthrough — you would need `usbipd-win` for the arm and a custom
kernel for the cameras. Native is both simpler and better supported.

What LeRobot does differently on Windows, from its source:

| | |
|---|---|
| Serial ports | `lerobot-find-port` lists **`COM3`, `COM4`…** via pyserial, not `/dev/tty*` (`lerobot_find_port.py:36`) |
| Cameras | sets `OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS=0` before importing cv2 (`camera_opencv.py:31`) — a workaround for an MSMF bug that otherwise gives black frames or a hang |
| Camera config | FOURCC is applied **after** width/height/fps, and pre-validation is skipped (`camera_opencv.py:205`) |
| Timing | busy-waits instead of sleeping, because Windows timer granularity is ~15 ms (`robot_utils.py:39`) |
| `--play_sounds` | speaks through PowerShell's `SpeechSynthesizer` |

### Three things to set up

**USB-serial driver.** macOS enumerates the SO-101 control board natively; Windows often does
not. If the board does not appear in Device Manager under *Ports (COM & LPT)*, install the
**CH340** or **CP210x** driver depending on your board, then replug.

**A bash shell for the `.sh` files.** `configs/ports.local.sh` and everything under
`configs/hpc/` are shell scripts. Use **Git Bash**, which ships with Git for Windows —
`source configs/ports.local.sh` works there and exports real environment variables that the
Python CLIs inherit. In PowerShell, set them directly instead:

```powershell
$env:FOLLOWER_PORT = "COM4"
$env:LEADER_PORT   = "COM3"
$env:FOLLOWER_ID   = "phi_follower"
$env:LEADER_ID     = "phi_leader"
```

**Developer Mode**, for the Hugging Face cache. `huggingface_hub` uses symlinks, which need
either Developer Mode or an elevated shell. Without it the cache silently falls back to full
file copies — it still works, but a 14 GB model is stored twice. Settings → System → For
developers → Developer Mode.

### Known Windows limits

- **Long paths.** The HF cache nests deeply and can exceed the 260-character `MAX_PATH`.
  Enable long paths: `git config --system core.longpaths true`, and set
  `HF_HOME` to something short such as `C:\hf`.
- **`import cv2` before lerobot.** LeRobot sets the MSMF workaround at import time. If your own
  script imports `cv2` first, the variable is never applied and cameras may hang. Import
  `lerobot` first, or set `OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS=0` yourself.
- **No MPS.** Use `--policy.device=cuda` on an NVIDIA laptop, `cpu` otherwise. CPU inference is
  fine for ACT at low `n_action_steps` and too slow for a VLA.
- **`configs/hpc/*.sbatch` are for the Linux cluster** and are not meant to run locally.

---

## 5. Your machine's ports file

**This file does not exist after a clone.** It is git-ignored because serial ports differ per
laptop, and nearly every command in these docs begins by sourcing it.

```bash
lerobot-find-port        # run TWICE — once per arm, unplugging when prompted
```

On macOS and Linux this prints `/dev/tty.usbmodem…` or `/dev/ttyACM0`. On Windows it prints
`COM3`-style names, and those are what go in the file.

Then copy the template and put your two ports in it:

```bash
cp configs/ports.local.sh.example configs/ports.local.sh
$EDITOR configs/ports.local.sh          # replace both REPLACE_ME values
source configs/ports.local.sh
```

Keep the ids exactly as written. They name your calibration files, and every doc, checkpoint
and rollout command assumes those two strings.

> **Never** `--port $(lerobot-find-port)`. That tool is interactive — it prints "Remove the
> USB cable… press Enter" and blocks on `input()`. Command substitution swallows the prompt
> and the terminal looks hung forever.

---

## 6. macOS camera permission

Your **terminal** needs it, not Python. Without it every camera opens black with no error.

System Settings → Privacy & Security → Camera → enable Terminal (or iTerm/VS Code). Then
**fully quit and reopen** the app; a new tab is not enough.

```bash
python -m phi.utils.camera_align 0 1 2 3     # should show live feeds
```

---

## 7. What goes wrong, in the order it bites

### `ModuleNotFoundError: No module named 'lerobot'` with the env active

A **pyenv shim** is ahead of conda on PATH. `which python` will not say `envs/phi`.

```bash
$CONDA_PREFIX/bin/python -m phi.utils.camera_align 0 1 2 3   # works around it
```

`$CONDA_PREFIX` is set by `conda activate` and always points at the right interpreter. A
fresh terminal usually resolves it properly too.

### `ImportError: libnvJitLink.so.12` (Linux / HPC)

A stale torch in `~/.local/lib/python3.*/site-packages` shadows the env's. Adding the env to
PATH does not help; only disabling user-site does.

```bash
export PYTHONNOUSERSITE=1
```

Put it in every sbatch script. Without it a job can *silently train with the wrong torch*
rather than erroring.

### Everything works, then stops after you recreate the env

A shell that had the old env activated keeps a dead path. Open a new terminal. Do not debug
this one for twenty minutes like we did.

### `Could not connect on port ''`

`configs/ports.local.sh` was never sourced in this shell. zsh expands an unset variable to
nothing without complaining, so `--robot.port=$FOLLOWER_PORT` becomes `--robot.port=`.

### Serial port busy / `device reports readiness to read but returned no data`

Something else holds the port — a second script, or LeLab.

```bash
lsof /dev/tty.usbmodem*
```

Stop the other process from **its own window with Ctrl-C**, not `kill -9`: a hard kill can
leave torque disabled and drop the arm.

### The policy runs but reaches to the wrong place, with no error

Calibration frame mismatch. `wrist_roll` is a full-turn joint with no hard stops, so its zero
is whatever pose was held at the calibration prompt — two calibrations of the *same arm*
differed by **60.4°**.

```bash
cp configs/calibration/robots/so_follower/phi_follower.json \
   ~/.cache/huggingface/lerobot/calibration/robots/so_follower/
```

Do this before any rollout that produces a number. Full explanation:
[02-setup §3c](robots/so-arm101/02-setup.md).

### A camera panel shows the wrong view

Camera **indices** are OpenCV's and change with USB port, plug order and reboots. Never reuse
a written-down index, including one in these docs. Re-run `camera_align` every session.

Separately, some **datasets** have their keys rotated relative to the physical cameras
(`saimaligi/pen_pick_and_place_*` does). A policy trained on one needs the same remap at
rollout, and the panel labelled `wrist` will correctly show a different view.

### `rerun` window never appears

`--display_data` defaults to `false`. Add `--display_data=true`. If it still does not appear,
`rerun` must be on PATH, which it is inside the `phi` env and not outside it.

---

## Next

| | |
|---|---|
| Bring up the arm | [02-setup](robots/so-arm101/02-setup.md) |
| Record a dataset | [03-teleop-and-data](robots/so-arm101/03-teleop-and-data.md) |
| Train | [training](training/README.md) · [Explorer HPC](hpc/explorer.md) |
| Run a policy | [deployment §1](deployment/README.md) |
| Something else broke | [troubleshooting](robots/so-arm101/troubleshooting.md) |

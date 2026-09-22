"""One command that says what is wrong with your setup.

    make doctor              # or: python -m phi.utils.doctor
    python -m phi.utils.doctor --cameras 0 1 2 3

Checks run in dependency order and each prints PASS, WARN or FAIL with the fix.
Nothing here touches the arm, so it is safe to run at any time.

Exit code is 0 unless something FAILed, so CI can use it.
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
from pathlib import Path

OK, WARN, BAD = "PASS", "WARN", "FAIL"
_results: list[str] = []


def say(state: str, label: str, detail: str = "", fix: str = "") -> None:
    mark = {OK: "  ok  ", WARN: " warn ", BAD: " FAIL "}[state]
    print(f"[{mark}] {label:28s} {detail}")
    if fix and state != OK:
        for line in fix.splitlines():
            print(f"           -> {line}")
    _results.append(state)


def check_python() -> None:
    v = sys.version_info
    in_env = "envs" in sys.executable and sys.executable.endswith(("python", "python3", "python.exe"))
    if v < (3, 12):
        say(BAD, "python", f"{v.major}.{v.minor}", "pyproject requires >=3.12. Recreate the env: make install")
    elif not in_env:
        say(WARN, "python", f"{v.major}.{v.minor}.{v.micro}  {sys.executable}",
            "Not obviously inside a conda env. If imports fail, a pyenv shim is ahead of\n"
            "conda on PATH -- use $CONDA_PREFIX/bin/python, or open a fresh terminal.")
    else:
        say(OK, "python", f"{v.major}.{v.minor}.{v.micro}")


def check_core() -> None:
    for mod, want in (("lerobot", "0.6.0"), ("torch", None), ("torchcodec", None)):
        try:
            m = __import__(mod)
            got = getattr(m, "__version__", "?")
            if want and got != want:
                say(WARN, mod, got, f"expected {want}. Mac and CUDA envs must match or checkpoints will not load.")
            else:
                say(OK, mod, got)
        except ImportError as e:
            fix = "make install"
            if "libnvJitLink" in str(e):
                fix = "export PYTHONNOUSERSITE=1   (a user-site torch is shadowing the env)"
            say(BAD, mod, "not importable", fix)


def check_accelerator() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.backends.mps.is_available():
        say(OK, "accelerator", "mps")
    elif torch.cuda.is_available():
        say(OK, "accelerator", f"cuda ({torch.cuda.get_device_name(0)})")
    else:
        say(WARN, "accelerator", "CPU only",
            "Fine for ACT at low n_action_steps. Too slow for a VLA.")


def check_cameras(indices: list[int]) -> None:
    try:
        from phi.utils.camera_backend import backend_name, open_camera
    except ImportError:
        say(BAD, "camera backend", "phi not installed", "pip install -e .")
        return
    say(OK, "camera backend", backend_name())

    opened = []
    for i in indices:
        cap = open_camera(i)
        ok, frame = (cap.read() if cap.isOpened() else (False, None))
        cap.release()
        if ok and frame is not None:
            opened.append(f"{i}({frame.shape[1]}x{frame.shape[0]})")
    if opened:
        say(OK, "cameras", " ".join(opened))
    else:
        if platform.system() == "Darwin":
            fix = ("macOS blocks camera access per-app. System Settings > Privacy & Security >\n"
                   "Camera, enable your terminal, then FULLY QUIT and reopen it.")
        elif platform.system() == "Windows":
            fix = ("Settings > Privacy & security > Camera > 'Let desktop apps access your camera'.\n"
                   "If a window opens but is black, something imported cv2 before\n"
                   "phi.utils.camera_backend, so the MSMF workaround never applied.")
        else:
            fix = "Check permissions on /dev/video*, and that the user is in the 'video' group."
        say(WARN, "cameras", f"none of {indices} gave a frame", fix)


def check_ports() -> None:
    try:
        from serial.tools import list_ports
    except ImportError:
        say(BAD, "pyserial", "missing", "make install")
        return
    devs = sorted(
        {p.device for p in list_ports.comports()}
        | {str(p) for p in Path("/dev").glob("tty.usbmodem*")}
    )
    real = [d for d in devs if "Bluetooth" not in d and "debug" not in d]
    if real:
        say(OK, "serial devices", " ".join(real))
    else:
        say(WARN, "serial devices", "none",
            "Is the arm plugged in and powered? On Windows check Device Manager >\n"
            "Ports (COM & LPT); you may need the CH340 or CP210x driver.")


def check_ports_file(repo: Path) -> None:
    f = repo / "configs" / "ports.local.sh"
    if not f.exists():
        say(BAD, "ports.local.sh", "missing",
            "cp configs/ports.local.sh.example configs/ports.local.sh   (then edit it)")
        return
    if "REPLACE_ME" in f.read_text():
        say(BAD, "ports.local.sh", "still has REPLACE_ME",
            "lerobot-find-port   (run it twice, once per arm)")
        return
    sourced = bool(os.environ.get("FOLLOWER_PORT"))
    if sourced:
        say(OK, "ports.local.sh", f"sourced, FOLLOWER_PORT={os.environ['FOLLOWER_PORT']}")
    else:
        say(WARN, "ports.local.sh", "exists but NOT sourced in this shell",
            "source configs/ports.local.sh\n"
            "Unsourced, $FOLLOWER_PORT expands to nothing and you get \"port ''\".")


def check_calibration(repo: Path) -> None:
    canon = repo / "configs/calibration/robots/so_follower/phi_follower.json"
    active = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so_follower/phi_follower.json"
    if not canon.exists():
        say(WARN, "calibration", "committed file missing from the repo")
    elif not active.exists():
        say(WARN, "calibration", "canonical frame not installed",
            f"cp {canon.relative_to(repo)} \\\n   ~/.cache/huggingface/lerobot/calibration/robots/so_follower/")
    elif canon.read_bytes() == active.read_bytes():
        say(OK, "calibration", "matches the committed phi_follower")
    else:
        say(WARN, "calibration", "active frame DIFFERS from the committed one",
            "Any number produced under a non-canonical calibration is void.\n"
            "python -m phi.utils.compare_calibration phi_follower <the-other-one>")


def check_hf() -> None:
    try:
        from huggingface_hub import whoami
        say(OK, "huggingface", whoami()["name"])
    except Exception:
        say(WARN, "huggingface", "not logged in",
            "hf auth login   (needed for gated repos such as PaliGemma, used by pi0/pi0.5)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cameras", type=int, nargs="*", default=[0, 1, 2, 3])
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[3]
    print(f"\nphi doctor  ·  {platform.system()} {platform.release()}  ·  {repo}\n")
    check_python()
    check_core()
    check_accelerator()
    check_cameras(args.cameras)
    check_ports()
    check_ports_file(repo)
    check_calibration(repo)
    check_hf()

    bad, warn = _results.count(BAD), _results.count(WARN)
    print(f"\n{len(_results)} checks · {bad} failed · {warn} warnings\n")
    if bad:
        print("Fix the FAILs first; the warnings below them are often downstream.\n")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

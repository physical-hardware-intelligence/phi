#!/usr/bin/env bash
# Track S · S1 — install the LeHome simulator on the GPU box: Isaac Sim 5.1 + IsaacLab fork + lehome.
# Userspace only (no sudo). Replaces lehome_solution's setup.sh step 5 (+ the sudo parts of step 1) with:
#   * fail loudly: setup.sh hides every error with `2>/dev/null || true`; here a failing step stops the run
#   * pinned versions: lehome-challenge @ 5ea947e (submodule), IsaacLab fork @ 69f6fa5 (setup.sh clones main)
#   * NVIDIA's Isaac Sim license is accepted ONLY when you set ACCEPT_ISAACSIM_EULA=yes
# Run on the box:   ACCEPT_ISAACSIM_EULA=yes bash ~/phi/sim/setup_sim_box.sh 2>&1 | tee ~/phi/logs/sim/setup.log
set -euo pipefail
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL CONDA_EXE CONDA_PYTHON_EXE _CE_CONDA _CE_M VIRTUAL_ENV PYTHONPATH
export PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin"
export UV_HTTP_TIMEOUT=600
CH=~/phi/lehome_solution/lehome-challenge
ISAACLAB_REPO=https://github.com/lehome-official/IsaacLab.git
ISAACLAB_PIN=69f6fa548c3a3520e3cb26ed24bb8abe60baeef3
step() { echo; echo "=== [$(date '+%F %H:%M:%S')] $*"; }

step "0/7 preflight"
if [ "${ACCEPT_ISAACSIM_EULA:-}" != "yes" ]; then
  echo "Isaac Sim needs NVIDIA's license accepted. Read it, then re-run with ACCEPT_ISAACSIM_EULA=yes."
  echo "https://docs.isaacsim.omniverse.nvidia.com/5.1.0/common/NVIDIA_Omniverse_License_Agreement.html"
  echo "Nothing was installed."; exit 2
fi
test "$(git -C "$CH" rev-parse --short=7 HEAD)" = 5ea947e
df -h ~ /tmp | awk 'NR == 1 || /nvme/'
nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv,noheader
for u in https://pypi.nvidia.com/ https://download.pytorch.org/whl/cu128 https://pypi.org/simple/ https://github.com; do
  printf '%-42s ' "$u"; curl -sS -o /dev/null -m 20 -w '%{http_code}\n' "$u"
done

step "1/7 zenity stand-in in ~/.local/bin (setup.sh puts one in /usr/local/bin with sudo; Isaac Sim calls zenity, which blocks headless runs)"
mkdir -p ~/.local/bin
[ -x ~/.local/bin/zenity ] || { printf '#!/bin/sh\nexit 0\n' > ~/.local/bin/zenity; chmod +x ~/.local/bin/zenity; }
command -v zenity

step "2/7 challenge venv: uv sync (isaacsim 5.1.0 from pypi.nvidia.com, torch 2.7.0 cu128)"
cd "$CH"
time uv sync
cp uv.lock ~/phi/lehome_challenge_uv.lock

step "3/7 Isaac Sim license (accepted via ACCEPT_ISAACSIM_EULA=yes)"
EULA="$CH/.venv/lib/python3.11/site-packages/isaacsim/kit/EULA_ACCEPTED"
mkdir -p "$(dirname "$EULA")"
echo yes > "$EULA"

step "4/7 IsaacLab fork @ ${ISAACLAB_PIN:0:7}"
[ -d third_party/IsaacLab/.git ] || git clone "$ISAACLAB_REPO" third_party/IsaacLab
git -C third_party/IsaacLab checkout -q "$ISAACLAB_PIN"
git -C third_party/IsaacLab rev-parse HEAD

step "5/7 IsaacLab + lehome packages (setup.sh's sequence, errors visible)"
set +e
( source .venv/bin/activate && ./third_party/IsaacLab/isaaclab.sh -i none ); rc=$?
set -e
[ $rc -eq 0 ] || echo "WARN: isaaclab.sh -i none exited $rc. setup.sh ignores this too; the next lines install isaaclab directly."
uv pip install --python .venv/bin/python "setuptools<70" wheel
uv pip install --python .venv/bin/python flatdict==4.0.1 --no-build-isolation
uv pip install --python .venv/bin/python -e third_party/IsaacLab/source/isaaclab
uv pip install --python .venv/bin/python -e source/lehome
uv pip install --python .venv/bin/python "warp-lang==1.12.1"   # newer warp breaks Isaac Sim 5.1 (setup.sh note)

step "6/7 versions"
.venv/bin/python - <<'PY'
import importlib.metadata as m, sys
print("python", sys.version.split()[0])
for p in ["isaacsim", "isaaclab", "lehome", "torch", "warp-lang", "flatdict", "numpy"]:
    try:
        print(f"{p:10s} {m.version(p)}")
    except Exception:
        print(f"{p:10s} MISSING")
PY

step "7/7 DONE"

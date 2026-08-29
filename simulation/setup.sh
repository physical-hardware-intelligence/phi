#!/usr/bin/env bash
# Set up the SO-101 MuJoCo sim environment (macOS / Linux).
#   ./simulation/setup.sh [venv_path]
# Deliberately NOT the `phi` conda env: mujoco is a plain pip wheel and `phi`
# is the only environment that talks to the real arm. Keep them apart.
set -euo pipefail

VENV="${1:-$HOME/venvs/so101-sim}"
PYVER=3.12

command -v uv >/dev/null || { echo "uv not found: https://docs.astral.sh/uv/"; exit 1; }

uv venv --python "$PYVER" "$VENV"
uv pip install --python "$VENV/bin/python" mujoco

# --- macOS: make mjpython work -------------------------------------------------
# mjpython is a launcher that dlopens the interpreter as a shared library via
# @executable_path/../lib/libpython3.X.dylib. A uv venv's lib/ holds only
# site-packages, so that path misses. Link the real dylib into place.
if [[ "$OSTYPE" == darwin* ]]; then
  real_py="$(python3 -c "import os;print(os.path.realpath('$VENV/bin/python'))")"
  dylib="$(cd "$(dirname "$real_py")/.." && pwd)/lib/libpython${PYVER}.dylib"
  target="$VENV/lib/libpython${PYVER}.dylib"
  if [[ -f "$dylib" && ! -e "$target" ]]; then
    ln -s "$dylib" "$target"
    echo "linked libpython${PYVER}.dylib for mjpython"
  fi
fi

echo
echo "done. now:"
echo "  source $VENV/bin/activate"
echo "  python simulation/verify_model.py          # headless check"
echo "  cd simulation && mjpython -m mujoco.viewer --mjcf=model/scene.xml"

# ─────────────────────────────────────────────────────────────────────────────
# Φ — sourced by every Explorer job AFTER `conda activate`. Fails fast and loudly
# if the env is not usable, instead of letting the job die on a cryptic error.
#
# WHY THIS EXISTS: on 2026-08-04 the env on /scratch lost its Python stdlib .py
# files between 00:57 (a full ACT training step passed) and 12:29 (next job).
# `lib/python3.12/encodings/` was left holding nothing but an empty __pycache__.
# The failure surfaced as:
#
#   Could not find platform independent libraries <prefix>
#   Fatal Python error: init_fs_encoding: failed to get the Python codec ...
#   LookupError: no codec search functions registered: can't find encoding
#
# Cause never established. Files with June mtimes survived while stdlib .py from
# the same day did not, so it is NOT a simple age-based purge. Note `python -V`
# still SUCCEEDS in this state (it short-circuits before full init), so version
# checks are worthless as a health test — you must actually import something.
#
# Usage:  source "$REPO/configs/hpc/_preflight.sh"
# ─────────────────────────────────────────────────────────────────────────────

phi_preflight() {
  local prefix="${CONDA_PREFIX:-}"
  echo "--- preflight: $prefix"

  # 1. the env python must complete interpreter startup and import the stdlib.
  #    `import encodings` is the exact thing that was missing; ssl+sqlite3 catch
  #    the other conda extensions that break quietly.
  if ! python -c "import os, sys, encodings, codecs, ssl, sqlite3" 2>/dev/null; then
    echo "❌ PREFLIGHT FAILED: the conda env's Python cannot import its own stdlib."
    echo "   This is the /scratch corruption seen on 2026-08-04, not a config error."
    echo "   Diagnose:  ls $prefix/lib/python3.12/encodings/"
    echo "              (if it holds only __pycache__, the stdlib is gone)"
    echo "   Fix:       rm -rf $prefix && sbatch configs/hpc/build_env.sbatch"
    return 1
  fi

  # 2. the pieces this project actually needs
  if ! python -c "
import torch, lerobot, accelerate
from lerobot.policies.act.modeling_act import ACTPolicy
print(f'    torch {torch.__version__} | cuda build {torch.version.cuda} | lerobot ok | accelerate {accelerate.__version__}')
" 2>/dev/null; then
    echo "❌ PREFLIGHT FAILED: stdlib is fine but torch/lerobot will not import."
    python -c "import torch" 2>&1 | tail -3
    return 1
  fi

  # 3. CUDA, only when the job asked for a GPU. cu130 wheels on Explorer's
  #    driver 545 give available=False and an EMPTY arch list — catch that here
  #    rather than 20 minutes into a training run.
  if [ -n "${SLURM_JOB_GPUS:-${SLURM_STEP_GPUS:-}}" ] || [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    python - <<'PY' || return 1
import sys, torch
if not torch.cuda.is_available():
    print("❌ PREFLIGHT FAILED: torch.cuda.is_available() is False on a GPU job.")
    print("   Most likely a CUDA-major mismatch: Explorer's driver is 545.x (CUDA 12.3),")
    print("   so a cu130 wheel cannot run. env/environment.cuda.yml pins cu126 for this.")
    print("   torch:", torch.__version__, "built for cuda", torch.version.cuda)
    sys.exit(1)
cap = torch.cuda.get_device_capability(0)
tag = f"sm_{cap[0]}{cap[1]}"
archs = torch.cuda.get_arch_list()
if tag not in archs:
    print(f"❌ PREFLIGHT FAILED: no {tag} kernels for {torch.cuda.get_device_name(0)}.")
    print("   arch list:", archs)
    sys.exit(1)
print(f"    cuda ok: {torch.cuda.get_device_name(0)} ({tag}) in {len(archs)} arch build")
PY
  fi

  echo "--- preflight OK"
  return 0
}

phi_preflight || exit 1

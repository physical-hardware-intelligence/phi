#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Φ — build the LeRobot CUDA env on Northeastern "Explorer" (SLURM).
#
# Dependency list is NOT duplicated here: it comes from env/environment.cuda.yml
# (lerobot[training,feetech,core_scripts]==0.6.0, pinned to match the Mac cockpit).
#
# PRAGMATIC CHOICES
#  - CPU partition, no GPU. pip installs prebuilt wheels; a GPU would idle for
#    an hour. CUDA is verified separately by configs/hpc/verify_env.sbatch.
#    (`short` also gives 2 days, vs gpu-short's 2 h cap — a slow proxy can't kill us.)
#  - Job-local $HOME with its own .condarc. Explorer's shared pkgs cache causes
#    repeated CondaVerificationError, and CONDA_PKGS_DIRS/CONDARC only MERGE
#    pkgs_dirs rather than replace them. The user's ~/.condarc currently points
#    pkgs_dirs at /scratch/$USER/matrix-game/conda-pkgs, WHICH NO LONGER EXISTS —
#    inheriting it is a guaranteed failure. We rewrite it, keeping the proxy lines.
#  - Compute nodes have NO direct internet. conda reads the proxy from .condarc;
#    pip does not, so http_proxy/https_proxy are exported explicitly.
#
# Submit:  sbatch configs/hpc/build_env.sbatch
# Watch:   tail -f /scratch/$USER/phi/logs/build_env-<jobid>.out
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=phi-build-env
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/gupta.yashv/phi/logs/build_env-%j.out
#SBATCH --error=/scratch/gupta.yashv/phi/logs/build_env-%j.err

set -uo pipefail

PROJECT=/scratch/$USER/phi
PREFIX=$PROJECT/envs/lerobot
YML=$PROJECT/repo/env/environment.cuda.yml
mkdir -p "$PROJECT"/{logs,conda-pkgs,condahome,hf,results}

echo "node=$(hostname)  date=$(date)  prefix=$PREFIX"

if [ ! -f "$YML" ]; then
  echo "FATAL: $YML missing. Sync the repo to $PROJECT/repo first."; exit 1
fi

# ---- proxy: compute nodes have no direct route out. pip needs this explicitly.
# 🚨 A stale torch in ~/.local/lib/python3.12/site-packages shadows the conda env
# (Python puts user-site ahead of it) and dies with `libnvJitLink.so.12: cannot
# open shared object file`. Every Φ job on Explorer must set this.
export PYTHONNOUSERSITE=1

export http_proxy=http://10.99.0.130:3128
export https_proxy=http://10.99.0.130:3128
export HF_HOME=$PROJECT/hf

module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"

# ---- job-local HOME + a .condarc we control (see PRAGMATIC CHOICES above)
export CONDA_PKGS_DIRS="$PROJECT/conda-pkgs"
REAL_HOME="$HOME"; export HOME="$PROJECT/condahome"
python - "$REAL_HOME/.condarc" "$CONDA_PKGS_DIRS" > "$HOME/.condarc" <<'PY'
import sys
src, pkgs = sys.argv[1], sys.argv[2]
try:
    txt = open(src).read()
except FileNotFoundError:
    txt = ""
keep, skip = [], False
for line in txt.splitlines():                 # keep proxy/ssl/channels, drop pkgs_dirs
    if line.strip().startswith("pkgs_dirs"):
        skip = True; continue
    if skip and (line.startswith(" ") or line.startswith("-")):
        continue
    skip = False
    keep.append(line)
keep += ["pkgs_dirs:", f"  - {pkgs}"]
print("\n".join(keep))
PY
echo "--- .condarc in effect ---"; cat "$HOME/.condarc"; echo "-------------------------"

# ---- create the env from the repo's yml (-p overrides the yml's `name:`)
if [ -d "$PREFIX" ]; then
  echo "env already exists at $PREFIX — reusing. Delete it to rebuild from scratch."
else
  conda env create -p "$PREFIX" -f "$YML" || { echo "FATAL: conda env create failed"; exit 1; }
fi

conda activate "$PREFIX"

echo "=== what we got ==="
python -c "import sys; print('python', sys.version.split()[0])"
python -c "import torch; print('torch', torch.__version__, '| cuda build', torch.version.cuda)"
python -c "import lerobot; print('lerobot', lerobot.__version__ if hasattr(lerobot,'__version__') else 'installed')"
python -c "from lerobot.policies.act.modeling_act import ACTPolicy; print('ACT import OK')"
python -c "import accelerate; print('accelerate', accelerate.__version__, '(the `training` extra — lerobot-train hard-requires it)')"
which lerobot-train lerobot-dataset-viz 2>/dev/null

echo "=== pinning what we actually resolved, for reproducibility ==="
pip freeze > "$PROJECT/envs/lerobot.requirements.lock.txt"
echo "wrote $PROJECT/envs/lerobot.requirements.lock.txt ($(wc -l < "$PROJECT/envs/lerobot.requirements.lock.txt") pkgs)"

echo "DONE $(date).  Next: sbatch configs/hpc/verify_env.sbatch"

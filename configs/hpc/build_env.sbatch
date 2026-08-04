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
#
# 🚨 "Does the directory exist" is NOT "is the env usable". A failed
# `conda env create` leaves a skeleton behind (just conda-meta/ and etc/), and a
# /scratch ESTALE event can gut a working env while leaving the tree in place.
# Both cases previously made this job SKIP the build and exit 0 — it reported
# python 3.9.18 (the base anaconda interpreter leaking through a broken activate)
# and wrote a 0-package lock file. Health-check by importing, never by ls.
if [ -x "$PREFIX/bin/python" ] && "$PREFIX/bin/python" -c "import encodings, ssl" 2>/dev/null; then
  echo "env at $PREFIX is present and its stdlib imports — reusing."
else
  if [ -d "$PREFIX" ]; then
    BROKEN="$PREFIX.broken.$(date +%H%M%S)"
    echo "env at $PREFIX exists but is NOT usable -> moving aside to $BROKEN"
    # rename, never rm: a plain rm -rf of a multi-GB tree on this filesystem can
    # outlast the job and ESTALE makes it fail partway, leaving a worse mess.
    mv "$PREFIX" "$BROKEN" || { echo "FATAL: could not move the broken env aside"; exit 1; }
  fi
  conda env create -p "$PREFIX" -f "$YML" || { echo "FATAL: conda env create failed"; exit 1; }
fi

conda activate "$PREFIX"

# Reset mtime/atime across the tree. Conda extraction preserves upstream
# timestamps, so a freshly built env can contain files that look years old to an
# age-based cleanup sweep. Cheap insurance; see configs/hpc/_preflight.sh.
find "$PREFIX" -exec touch -a -m {} + 2>/dev/null
echo "touched $(find "$PREFIX" -type f | wc -l) files to reset mtime/atime"

echo "=== what we got ==="
# 🚨 The interpreter MUST be the one inside $PREFIX. When activation half-fails,
# `python` silently resolves to the base anaconda 3.9 and every check below
# "passes" against the wrong environment. Assert the path first.
ACTUAL=$(python -c "import sys; print(sys.executable)" 2>/dev/null)
if [ "$ACTUAL" != "$PREFIX/bin/python" ]; then
  echo "FATAL: python resolved to '$ACTUAL', expected '$PREFIX/bin/python'."
  echo "       conda activate did not take. Refusing to report success."
  exit 1
fi

# Same guard the consumer jobs use, so a build can never be declared good while
# prefetch/bench/train would reject the env three seconds in.
source "$PROJECT/repo/configs/hpc/_preflight.sh"

which lerobot-train lerobot-dataset-viz 2>/dev/null

echo "=== pinning what we actually resolved, for reproducibility ==="
pip freeze > "$PROJECT/envs/lerobot.requirements.lock.txt"
NPKG=$(wc -l < "$PROJECT/envs/lerobot.requirements.lock.txt")
echo "wrote $PROJECT/envs/lerobot.requirements.lock.txt ($NPKG pkgs)"
# a real build resolves ~118 packages; single digits means we locked an empty env
if [ "$NPKG" -lt 50 ]; then
  echo "FATAL: only $NPKG packages in the lock file — this env is not complete."; exit 1
fi

echo "DONE $(date).  Next: sbatch configs/hpc/prefetch_dataset.sbatch"

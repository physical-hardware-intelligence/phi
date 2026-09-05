#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Φ — build the CUDA training env for Explorer (lands on /scratch, see below).
#
# WHY /scratch AND NOT $HOME:
#   Home was the intent (scratch gets purged — /scratch/gupta.yashv/phi was found
#   EMPTY on 2026-09-05). But home has a per-user quota that neither `quota` nor
#   `df` will report, and it is already ~77 GB full: microDLM 27G, .conda 26G,
#   .cache 19G. Build 9989313 died at 7.8 GB with "Disk quota exceeded" — not
#   OOM, MaxRSS was 12.7 GB of the 32 GB requested. A complete env needs ~11 GB,
#   so home cannot hold it without deleting the user's work.
#
#   Mitigation for the purge risk is to make rebuilding CHEAP rather than to
#   fight the quota: this script is in git and takes ~10 min. After a purge,
#   recovery is one `sbatch`. That is a better trade than a 27 GB eviction.
#
#   To use home instead, free ~15 GB first (~/.cache at 19 GB is the softest
#   target) and flip ENV_PREFIX back.
#
# WHY A COMPUTE NODE:
#   The login node kills `conda create`. Always build via sbatch.
#
# WHY A JOB-LOCAL HOME FOR THE BUILD:
#   A shared/pinned conda pkgs cache corrupts and throws repeated
#   CondaVerificationError. CONDA_PKGS_DIRS and CONDARC only *merge* into
#   pkgs_dirs, they do not replace it. The fix that works is a job-local $HOME
#   carrying its own .condarc, so the build gets a clean cache. The env itself
#   is still written to the real home via an explicit --prefix.
#
#   Submit:  sbatch configs/hpc/setup_env.sbatch
#   Watch:   tail -f ~/phi/logs/setup-<jobid>.out
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=phi-setup-env
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/gupta.yashv/phi/logs/setup-%j.out
#SBATCH --error=/home/gupta.yashv/phi/logs/setup-%j.err

set -euo pipefail

REAL_HOME=/home/gupta.yashv
PHI=$REAL_HOME/phi
ENV_PREFIX=/scratch/gupta.yashv/envs/phi-cuda   # see the note above

mkdir -p "$PHI/logs" "$(dirname "$ENV_PREFIX")"

# --- job-local HOME so conda gets a clean pkgs cache ---------------------------
BUILD_HOME=$(mktemp -d /tmp/phi-condabuild-XXXXXX)
trap 'rm -rf "$BUILD_HOME"' EXIT
cat > "$BUILD_HOME/.condarc" <<'RC'
channels: [conda-forge]
channel_priority: strict
always_yes: true
RC
export HOME="$BUILD_HOME"

module load anaconda3/2024.06
eval "$(conda shell.bash hook)"

echo "=== building $ENV_PREFIX (build HOME=$BUILD_HOME) ==="
conda create -y -p "$ENV_PREFIX" python=3.12 ffmpeg pip -c conda-forge

# --- pip layer -----------------------------------------------------------------
# 🚨 cu126, NOT cu130. Explorer GPU nodes run driver 545.23.08 (CUDA 12.3);
#    a cu130 wheel fails with "driver is too old (found version 12030)" and
#    torch.cuda.is_available() returns False. cu126 works on 545 AND on newer
#    drivers, so it is correct on every node type. Measured 2026-08-03.
"$ENV_PREFIX/bin/pip" install --no-cache-dir \
  --extra-index-url https://download.pytorch.org/whl/cu126 \
  "torch==2.11.0" "torchvision==0.26.0"

"$ENV_PREFIX/bin/pip" install --no-cache-dir \
  "lerobot[training,feetech,core_scripts]==0.6.0"

# --- verify --------------------------------------------------------------------
echo "=== verify ==="
"$ENV_PREFIX/bin/python" - <<'PY'
import torch, lerobot
print("lerobot     ", lerobot.__version__)
print("torch       ", torch.__version__)
print("torch cuda  ", torch.version.cuda)
print("cuda avail  ", torch.cuda.is_available(), "(False here is EXPECTED — this is a CPU partition)")
PY
"$ENV_PREFIX/bin/lerobot-train" --help >/dev/null && echo "lerobot-train  OK"

echo "=== done. env at $ENV_PREFIX ==="

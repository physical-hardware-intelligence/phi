#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Φ — pull phi_so101_8bin_v1 onto /scratch ONCE, before the training array.
#
# WHY A SEPARATE JOB: six array tasks starting together would each try to
# download the same 1.25 GB and race on the same cache directory. Prefetch once,
# then every task reads a warm local copy.
#
# CPU partition — this is pure network + disk. No GPU.
#
# Submit:  sbatch configs/hpc/prefetch_dataset.sbatch
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=phi-prefetch
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/gupta.yashv/phi/logs/prefetch-%j.out
#SBATCH --error=/scratch/gupta.yashv/phi/logs/prefetch-%j.err

set -uo pipefail
PROJECT=/scratch/$USER/phi
REPO_ID=BrutalCaesar/phi_so101_8bin_v1

# compute nodes have no direct route out
# 🚨 A stale torch in ~/.local/lib/python3.12/site-packages shadows the conda env
# (Python puts user-site ahead of it) and dies with `libnvJitLink.so.12: cannot
# open shared object file`. Every Φ job on Explorer must set this.
export PYTHONNOUSERSITE=1

export http_proxy=http://10.99.0.130:3128
export https_proxy=http://10.99.0.130:3128
export HF_HOME=$PROJECT/hf
export HF_LEROBOT_HOME=$PROJECT/lerobot-data      # every job must export this identically
mkdir -p "$HF_HOME" "$HF_LEROBOT_HOME"

module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PROJECT/envs/lerobot"

echo "node=$(hostname) date=$(date)"
echo "HF_LEROBOT_HOME=$HF_LEROBOT_HOME"

python - "$REPO_ID" <<'PY'
import sys
from lerobot.datasets.lerobot_dataset import LeRobotDataset
repo_id = sys.argv[1]
# Resolves the `v3.0` codebase-version TAG, not `main`. If this raises
# RevisionNotFoundError the tag is missing on the Hub — see
# docs/robots/so-arm101/03-teleop-and-data.md (§3c).
ds = LeRobotDataset(repo_id)
m = ds.meta
print(f"OK  episodes={m.total_episodes}  frames={m.total_frames}  fps={m.fps}")
print(f"    revision={m.revision}  robot={m.robot_type}")
print(f"    cameras={m.camera_keys}")
assert m.total_episodes == 119, f"expected 119 episodes, got {m.total_episodes}"
assert m.total_frames == 72518, f"expected 72518 frames, got {m.total_frames}"
print("    counts match the dataset card")
PY
rc=$?

echo "--- on disk ---"
du -sh "$HF_LEROBOT_HOME"/* 2>/dev/null
echo "prefetch rc=$rc  $(date)"
exit $rc

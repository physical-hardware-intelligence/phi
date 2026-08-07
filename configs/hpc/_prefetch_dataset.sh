#!/bin/bash
# Prefetch a Hub dataset into $PROJECT/lerobot-data so training tasks do not race
# on the download.
#
#   sbatch --partition=short --cpus-per-task=4 --mem=16G --time=00:45:00 \
#     --job-name=phi-prefetch --output=/scratch/$USER/phi/logs/prefetch-%j.out \
#     configs/hpc/_prefetch_dataset.sh BrutalCaesar/phi_so101_cubes_cylinder_v1
#
# 🚨 Must be sbatch, not srun and not the login node:
#    - the login node KILLS the download partway through
#    - srun is tethered to your SSH session and dies with it
set -uo pipefail

REPO_ID="${1:?usage: _prefetch_dataset.sh <hf_repo_id>}"
PROJECT=/scratch/$USER/phi

export PYTHONNOUSERSITE=1
export http_proxy=http://10.99.0.130:3128
export https_proxy=http://10.99.0.130:3128
export HF_HOME=$PROJECT/hf
export HF_LEROBOT_HOME=$PROJECT/lerobot-data

module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PROJECT/envs/lerobot"

python - "$REPO_ID" <<'PY'
import sys
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset(sys.argv[1])
print("PREFETCH OK:", ds.meta.total_episodes, "episodes,", ds.meta.total_frames, "frames")
print("cameras:", ds.meta.camera_keys)
print("root:", ds.root)
PY

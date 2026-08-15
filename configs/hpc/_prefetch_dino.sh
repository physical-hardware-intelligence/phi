#!/bin/bash
# Prefetch DINOv2 into $PROJECT/torchhub so training tasks do not race on the
# download (and so an array of 2 does not fetch it twice).
#
#   sbatch --partition=short --cpus-per-task=4 --mem=16G --time=00:30:00 \
#     --job-name=phi-prefetch-dino --output=/scratch/$USER/phi/logs/prefdino-%j.out \
#     configs/hpc/_prefetch_dino.sh
#
# 🚨 Must be sbatch, not srun and not the login node -- same reason as
#    _prefetch_dataset.sh: the login node kills the download partway through.
#
# Pins the same commit the reference implementation pins
# (gaoyuezhou/patch_policy models/encoder/dino.py: torch.hub.load("...dinov2:b48308a")).
set -uo pipefail

PROJECT=/scratch/$USER/phi

export PYTHONNOUSERSITE=1
export http_proxy=http://10.99.0.130:3128
export https_proxy=http://10.99.0.130:3128
export TORCH_HOME=$PROJECT/torchhub
mkdir -p "$TORCH_HOME" "$PROJECT/logs"

module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PROJECT/envs/lerobot"

python - <<'PY'
import os, torch
torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
m = torch.hub.load("facebookresearch/dinov2:b48308a", "dinov2_vits14", verbose=False)
n = sum(p.numel() for p in m.parameters())
assert n == 22_056_576, f"expected 22,056,576 params, got {n:,}"

# 210x280 is the ONLY size the pipeline uses; prove it here rather than in a GPU job.
x = torch.rand(1, 3, 210, 280)
with torch.no_grad():
    p = m.forward_features(x)["x_norm_patchtokens"]
assert tuple(p.shape) == (1, 300, 384), f"expected (1,300,384), got {tuple(p.shape)}"

print(f"PREFETCH OK: DINOv2 ViT-S/14, {n:,} params")
print(f"  210x280 -> {tuple(p.shape)}  (15x20 = 300 patches, zero remainder)")
print(f"  TORCH_HOME={os.environ['TORCH_HOME']}")
PY

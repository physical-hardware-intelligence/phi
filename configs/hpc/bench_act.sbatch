#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Φ — verify CUDA and MEASURE throughput/VRAM before committing 6 training runs.
#
# We do not guess --batch_size or --steps. The Mac's MPS extrapolation
# (0.49 GB + 0.575 GB/sample, see docs/theory/act-shapes.md) is indicative only;
# this job measures the real numbers on the real GPU, then prints the batch size
# and step budget that actually fit an 8 h window.
#
# Runs 300 real lerobot-train steps per batch size and reads it/s off the logs.
#
# Submit:  sbatch configs/hpc/bench_act.sbatch
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=phi-bench-act
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:1
#SBATCH --exclude=d1025
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/gupta.yashv/phi/logs/bench-%j.out
#SBATCH --error=/scratch/gupta.yashv/phi/logs/bench-%j.err

set -uo pipefail
PROJECT=/scratch/$USER/phi
REPO_ID=BrutalCaesar/phi_so101_8bin_v1

# 🚨 A stale torch in ~/.local/lib/python3.12/site-packages shadows the conda env
# (Python puts user-site ahead of it) and dies with `libnvJitLink.so.12: cannot
# open shared object file`. Every Φ job on Explorer must set this.
export PYTHONNOUSERSITE=1

export http_proxy=http://10.99.0.130:3128
export https_proxy=http://10.99.0.130:3128
export HF_HOME=$PROJECT/hf
export HF_LEROBOT_HOME=$PROJECT/lerobot-data

module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PROJECT/envs/lerobot"

source "$PROJECT/repo/configs/hpc/_preflight.sh"

echo "=== node/GPU ==="
hostname
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| available', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"

# Physical wrist + top.  ⚠️ THE DATASET KEYS ARE SWAPPED: physical wrist is
# stored as `top` and physical top as `wrist`. See datasets/phi_so101_8bin_v1.md.
FEATS='{"observation.state": {"type": "STATE", "shape": [6]},
        "observation.images.top": {"type": "VISUAL", "shape": [3, 480, 640]},
        "observation.images.wrist": {"type": "VISUAL", "shape": [3, 480, 640]}}'

for BS in 16 32 48; do
  echo ""
  echo "################ batch_size=$BS ################"
  OUT=$PROJECT/results/bench_bs${BS}
  rm -rf "$OUT"
  /usr/bin/time -v lerobot-train \
    --dataset.repo_id=$REPO_ID \
    --policy.type=act \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --policy.input_features="$FEATS" \
    --policy.chunk_size=100 --policy.n_action_steps=100 \
    --batch_size=$BS \
    --steps=300 --log_freq=50 \
    --num_workers=8 \
    --save_checkpoint=false \
    --seed=1000 \
    --output_dir="$OUT" --job_name=bench_bs${BS} \
    --wandb.enable=false 2>&1 | grep -Ei "smpl|updt_s|data_s|it/s|steps|CUDA out of memory|Error|Maximum resident" | tail -12
  echo "peak GPU memory after bs=$BS:"
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
done

echo ""
echo "=== how to read this ==="
echo "chunk_size=100 is the WORST case for VRAM of our three (50/75/100), so a"
echo "batch that fits here fits all six runs. Take the largest BS that did not OOM,"
echo "read its steps/s, then: steps_that_fit_8h = it/s * 3600 * 7.5"
echo "(7.5 h, leaving 30 min for dataset load, checkpointing and slack)."

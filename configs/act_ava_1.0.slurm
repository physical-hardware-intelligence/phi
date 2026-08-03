#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Φ — ACT baseline on Ava (SO-101), Northeastern "Explorer" HPC (SLURM).
#
#   Dataset : Parv-09/Ava_1.0_20260730_172156  (public — no HF auth needed)
#   Plan    : experiments/2026-07-30_act-wrist-only-so101.md
#   Why here: measured 0.55 it/s on the M4 Air (MPS) => ~25 h for 50k steps.
#             The Mac cockpit is for record/teleop/smoke only.
#
#   Submit:  sbatch configs/act_ava_1.0.slurm
#   Watch:   squeue -u $USER   ·   tail -f logs/act_ava_1.0-<jobid>.out
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=act_ava_1.0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
mkdir -p logs

SCRATCH=/scratch/$USER
REPO="$SCRATCH/phi"                 # clone the repo here
ENV_PREFIX="$SCRATCH/envs/phi"      # build envs in /scratch, home is quota-limited

# Caches on /scratch too — HF datasets + wandb will fill a home quota fast.
export HF_HOME="$SCRATCH/.cache/huggingface"
export WANDB_DIR="$SCRATCH/wandb"
export WANDB_CACHE_DIR="$SCRATCH/.cache/wandb"
# export WANDB_API_KEY=...        # set in your shell profile, do NOT commit it

module load anaconda3 2>/dev/null || true
source activate "$ENV_PREFIX"

cd "$REPO"
nvidia-smi

lerobot-train \
  --dataset.repo_id=Parv-09/Ava_1.0_20260730_172156 \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --seed=1000 \
  --num_workers=8 \
  --batch_size=32 \
  --steps=50000 \
  --save_freq=5000 \
  --output_dir="$SCRATCH/outputs/train/act_ava_1.0" \
  --job_name=act_ava_1.0 \
  --wandb.enable=true \
  --wandb.project=phi-act \
  --wandb.notes="ACT baseline, wrist-cam only. 60 eps / 4 bins; one bin is NOT in the wrist FOV at episode start (expected failure mode, measured on purpose)."

# ── First-time setup on Explorer (run ONCE, on a compute node — the login node
# ── kills conda create):
#   srun --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=48G --time=01:00:00 --pty bash
#   git clone https://github.com/physical-hardware-intelligence/phi.git /scratch/$USER/phi
#   cd /scratch/$USER/phi
#   conda env create -p /scratch/$USER/envs/phi -f env/environment.cuda.yml
#   source activate /scratch/$USER/envs/phi && pip install -e . && wandb login

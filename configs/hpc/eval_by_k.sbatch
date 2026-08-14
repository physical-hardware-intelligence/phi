#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Φ — held-out diffusion loss BUCKETED BY NOISE LEVEL k, for both DP-T arms.
#
#   Submit:  sbatch configs/hpc/eval_by_k.sbatch
#   Read:    /scratch/$USER/phi/logs/kevalp-<jobid>_<n>.out
#
# Inference only: no training, no arm time. See src/phi/eval/eval_by_k.py for why
# the k-AVERAGED eval_loss cannot answer the question this does.
#
# Short version: at k=99 the signal is 0.0005 of the input, so predicting eps is
# nearly copying it; at k=0 recovering eps means dividing by 0.0251, amplifying
# any residual uncertainty about the action ~40x. Only the second regime sets
# whether the gripper closes -- and "reaches correctly, fails to close" is the DP
# failure mode scored on the arm on 2026-08-12. A single average blends them.
#
# The decision it informs: unreg @ 8000 has the best AVERAGE (0.0287) but
# overfits; paper @ 26000 is stable but 1.85x worse on that average. If paper's
# advantage is concentrated at LOW k, the average is hiding the thing that
# matters and paper is the better rollout candidate despite the worse number.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=phi-kevalp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-1
#SBATCH --output=/scratch/gupta.yashv/phi/logs/kevalp-%A_%a.out
#SBATCH --error=/scratch/gupta.yashv/phi/logs/kevalp-%A_%a.err

set -uo pipefail
PROJECT=/scratch/$USER/phi

case $SLURM_ARRAY_TASK_ID in
  0) ARM=unreg; STEPS=2000,8000,16000,30000 ;;   # 8000 is its held-out minimum
  1) ARM=paper; STEPS=2000,8000,16000,26000,30000 ;;  # 26000 is its minimum
  *) echo "bad array index"; exit 1 ;;
esac

export PYTHONNOUSERSITE=1
export http_proxy=http://10.99.0.130:3128
export https_proxy=http://10.99.0.130:3128
export HF_HOME=$PROJECT/hf
export HF_LEROBOT_HOME=$PROJECT/lerobot-data
export PYTHONPATH=$PROJECT/repo/src${PYTHONPATH:+:$PYTHONPATH}
mkdir -p "$PROJECT/logs" "$PROJECT/repo/outputs"

module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PROJECT/envs/lerobot"
source "$PROJECT/repo/configs/hpc/_preflight.sh"

echo "=============================================================="
echo " k-BUCKETED EVAL   arm=$ARM   steps=$STEPS"
echo " node $(hostname)"
echo "=============================================================="

srun python -m phi.eval.eval_by_k \
  --sweep $PROJECT/results/dpt_${ARM}_emb256 \
  --steps $STEPS \
  --device cuda \
  --seed 1000 \
  --out $PROJECT/repo/outputs/k_eval_${ARM}.csv

echo "rc=$?"
echo " NOTE: read the var_explained column, not raw mse. Raw mse mostly"
echo " reproduces the schedule shape (high at low k by construction)."

#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Train ACT on Explorer. Nothing in here is specific to one person's account.
#
#   sbatch --export=ALL,REPO_ID=<hf-dataset>,RUN=<your-team-name> \
#          configs/hpc/train_act_student.sbatch
#
#   squeue -u $USER                       # is it queued / running
#   tail -f ~/phi-logs/<RUN>-<jobid>.out  # watch it
#
# Everything below is overridable the same way, e.g. ...,STEPS=100000,CHUNK=100
#   (except SEED, which is fixed -- see below)
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=phi-act
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

# --- everything you might change -------------------------------------------
REPO_ID=${REPO_ID:?set REPO_ID=<huggingface-dataset-id>}
RUN=${RUN:-act_${USER}}
STEPS=${STEPS:-60000}
CHUNK=${CHUNK:-50}
BATCH_SIZE=${BATCH_SIZE:-8}
SAVE_FREQ=${SAVE_FREQ:-10000}
EVAL_STEPS=${EVAL_STEPS:-10000}
EVAL_SPLIT=${EVAL_SPLIT:-0.15}
# FIXED for the competition. Varying the seed and submitting the best of ten
# runs rewards queue time, not understanding. Everything else is yours.
SEED=1000

# --- paths, all derived from YOUR account ----------------------------------
SCRATCH=/scratch/$USER
LOGS=$HOME/phi-logs
RESULTS=$SCRATCH/phi-results
ENV_PREFIX=${ENV_PREFIX:-$SCRATCH/envs/phi-cuda}   # or the shared one, see docs
mkdir -p "$LOGS" "$RESULTS" "$SCRATCH/.cache"

# 🚨 Without this, a stale torch in ~/.local shadows the env. On most nodes that
#    dies with "ImportError: libnvJitLink.so.12"; on the rest it SILENTLY TRAINS
#    WITH THE WRONG TORCH. Adding the env to PATH does not prevent it.
export PYTHONNOUSERSITE=1
export HF_HOME=$SCRATCH/.cache/huggingface
export PYTHONPATH=$HOME/phi/src:${PYTHONPATH:-}

module load anaconda3/2024.06 2>/dev/null || true
export PATH="$ENV_PREFIX/bin:$PATH"

OUT=$RESULTS/$RUN

# A cancelled attempt leaves an empty $OUT and lerobot-train then refuses to
# start. Clear it ONLY when it holds no checkpoint, so a real run is never lost.
if [ -d "$OUT" ] && [ -z "$(find "$OUT" -name '*.safetensors' -print -quit 2>/dev/null)" ]; then
  echo "note: removing empty leftover $OUT"
  rm -rf "$OUT"
fi

echo "──────────────────────────────────────────────"
echo " user     : $USER"
echo " run      : $RUN"
echo " dataset  : $REPO_ID"
echo " steps    : $STEPS   chunk: $CHUNK   batch: $BATCH_SIZE"
echo " env      : $ENV_PREFIX"
echo " out      : $OUT"
echo " node     : $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch;print('cuda avail:',torch.cuda.is_available(),'| built for',torch.version.cuda)"
echo "──────────────────────────────────────────────"

srun lerobot-train \
  --dataset.repo_id="$REPO_ID" \
  --dataset.eval_split=$EVAL_SPLIT \
  --dataset.video_backend=pyav \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.chunk_size=$CHUNK \
  --policy.n_action_steps=$CHUNK \
  --batch_size=$BATCH_SIZE \
  --steps=$STEPS \
  --save_freq=$SAVE_FREQ \
  --eval_steps=$EVAL_STEPS \
  --seed=$SEED \
  --num_workers=16 \
  --output_dir="$OUT" \
  --job_name="$RUN" \
  --wandb.enable=false

echo "finished rc=$? at $(date)"
echo "checkpoints: $OUT/checkpoints"

# ─────────────────────────────────────────────────────────────────────────────
# THE LAST CHECKPOINT IS NOT AUTOMATICALLY THE BEST ONE.
#
# Runs overfit, and held-out loss can rise while you keep training. Score every
# checkpoint on the same holdout before you pick one to submit:
#
#   python -m phi.eval.loss_by_checkpoint --run $OUT --max-batches 40
#
# That is a real part of the job, not an optional extra. On our pi0.5 run the
# best checkpoint was 22500 of 30000, and the final one was 3.5% worse.
#
# And held-out loss is STILL not success rate -- experiments/2026-08-12 has a
# checkpoint whose loss rose 4x and matched a 100k-step model on the arm. Use
# the score to pick which two or three get arm time. Rollouts decide.
# ─────────────────────────────────────────────────────────────────────────────

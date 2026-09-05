#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Φ — ACT on Sai's pen pick-and-place dataset. Chunk 50 vs 100.
#
#   Dataset : saimaligi/pen_pick_and_place_20260817_162529  (public)
#             60 episodes · 32,032 frames · 30 fps · 3 cams 480x640 · state/action 6
#             Task string: "pick up the pen and place it in the container"
#   Plan    : experiments/2026-09-05_act-pen-chunk-50-100.md
#
#   Submit:  sbatch configs/hpc/train_pen_act.sbatch
#   Watch:   squeue -u $USER  ·  tail -f ~/phi/logs/pen-<jobid>_0.out
#
# ─────────────────────────────────────────────────────────────────────────────
# 🚨 THREE THINGS TO READ BEFORE TRUSTING ANY OUTPUT OF THIS RUN
#
# 1. THE CAMERA KEYS IN THIS DATASET ARE ROTATED BY ONE.
#    Verified 2026-09-05 by decoding frames from all three streams:
#
#        key "observation.images.front"  actually holds  WRIST footage
#        key "observation.images.top"    actually holds  FRONT footage
#        key "observation.images.wrist"  actually holds  TOP   footage
#
#    Training is unaffected — ACT does not care what the channels are named, it
#    just fits whatever is in them. DEPLOYMENT IS NOT UNAFFECTED. At inference
#    SOFollower puts the physical wrist camera into `observation.images.wrist`,
#    which during training held the TOP view. The policy then receives the wrong
#    image on every channel and fails silently, with no error.
#    ⇒ Any rollout MUST remap, or the dataset must be re-uploaded with the keys
#      corrected. Same bug class as phi_so101_8bin_v1 (see experiments/2026-08-04).
#
# 2. THIS DATA IS IN SAI'S CALIBRATION FRAME, NOT phi_follower.json.
#    A policy trained here emits joint angles in the frame of the arm as Sai had
#    it calibrated on 2026-08-17. Running it against a different calibration
#    mis-grasps silently — confirmed on the arm 2026-08-10, docs §3c. Get Sai's
#    calibration file before any rollout, or re-derive it.
#
# 3. THE RECOVERY EPISODES ARE NOT LABELLED AND NOT SEPARABLE.
#    total_tasks == 1, so there is no field distinguishing them. Episode length
#    is a weak proxy at best here: the distribution is continuous from 13.0s to
#    24.5s (median 18.0s), not bimodal like the cubes/cylinder set was
#    (21.8s recovery vs 18.6s clean). The 7 longest are [0,7,20,25,26,32,37] —
#    scattered, NOT a trailing block. Consequence: the holdout below is NOT
#    stratified by recovery-vs-clean, and no loss from this run can tell you
#    whether recovery behaviour improved. That needs a rollout protocol where you
#    deliberately displace the pen mid-episode. See docs/evaluation.
#
#    All episodes end with the pen correctly placed (confirmed by Yash), so this
#    is DAgger/RaC-style positive supervision at off-distribution states, not a
#    set of negative examples. Safe for ACT's L1 objective.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=phi-act-pen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-1
#SBATCH --output=/home/gupta.yashv/phi/logs/pen-%A_%a.out
#SBATCH --error=/home/gupta.yashv/phi/logs/pen-%A_%a.err

set -euo pipefail

# --- everything in HOME: /scratch is purged (found empty 2026-09-05) -----------
REAL_HOME=/home/gupta.yashv
PHI=$REAL_HOME/phi
ENV_PREFIX=$REAL_HOME/.conda/envs/phi-cuda

export HF_HOME=$REAL_HOME/.cache/huggingface
export WANDB_DIR=$PHI/wandb
export WANDB_CACHE_DIR=$REAL_HOME/.cache/wandb
mkdir -p "$PHI/logs" "$WANDB_DIR"

# --- the one axis this array varies -------------------------------------------
CHUNKS=(50 100)
CHUNK=${CHUNKS[$SLURM_ARRAY_TASK_ID]}

REPO_ID=saimaligi/pen_pick_and_place_20260817_162529
TAG=pen
SEED=1000
BATCH_SIZE=${BATCH_SIZE:-8}
STEPS=${STEPS:-100000}
SAVE_FREQ=${SAVE_FREQ:-20000}
EVAL_STEPS=${EVAL_STEPS:-10000}

# 🚨 eval_split does NOT round freely. make_train_eval_datasets holds out
#    ceil(n_episodes * eval_split) PER TASK STRING. For n=60 we want 12:
#      11 < 60*s <= 12   ->   0.18334 < s <= 0.2
#    0.2 lands exactly on the boundary (60*0.2 == 12.0); 0.195 gives 11.7 and is
#    safely inside. Holdout = the LAST 12 episodes by index, i.e. 48-59.
EVAL_SPLIT=0.195

CAMS="\"observation.images.wrist\": {\"type\": \"VISUAL\", \"shape\": [3, 480, 640]},
      \"observation.images.front\": {\"type\": \"VISUAL\", \"shape\": [3, 480, 640]},
      \"observation.images.top\":   {\"type\": \"VISUAL\", \"shape\": [3, 480, 640]}"
FEATS="{\"observation.state\": {\"type\": \"STATE\", \"shape\": [6]}, $CAMS}"

RUN=act_${TAG}_chunk${CHUNK}
OUT=$PHI/results/$RUN

module load anaconda3/2024.06 2>/dev/null || true
export PATH="$ENV_PREFIX/bin:$PATH"

echo "──────────────────────────────────────────────────────────────"
echo " run           : $RUN     (array task $SLURM_ARRAY_TASK_ID)"
echo " dataset       : $REPO_ID"
echo " chunk_size    : $CHUNK   (open-loop $(python -c "print(f'{$CHUNK/30:.2f}')") s at 30 fps)"
echo " eval_split    : $EVAL_SPLIT -> $(python -c "import math;print(math.ceil(60*$EVAL_SPLIT))") episodes held out (48-59)"
echo " batch/steps   : $BATCH_SIZE / $STEPS"
echo " node          : $(hostname)"
echo "──────────────────────────────────────────────────────────────"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -c "import torch;print('cuda avail:',torch.cuda.is_available(),'| built for',torch.version.cuda)"

# NOTE on chunk choice: at 30 fps, chunk 50 = 1.67 s and chunk 100 = 3.33 s.
# The ACT paper's ablation peaks at k=100 *at 50 Hz*, i.e. 2.0 s — which at our
# 30 fps is chunk 60. So 50 is slightly short of the paper's optimum in time
# terms and 100 is well past it. That is the point of running both.

srun lerobot-train \
  --dataset.repo_id=$REPO_ID \
  --dataset.eval_split=$EVAL_SPLIT \
  --dataset.video_backend=pyav \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.use_vae=true \
  --policy.kl_weight=10.0 \
  --policy.input_features="$FEATS" \
  --policy.chunk_size=$CHUNK \
  --policy.n_action_steps=$CHUNK \
  --batch_size=$BATCH_SIZE \
  --steps=$STEPS \
  --save_freq=$SAVE_FREQ \
  --eval_steps=$EVAL_STEPS \
  --seed=$SEED \
  --output_dir="$OUT" \
  --job_name="$RUN" \
  --wandb.enable=true \
  --wandb.project=phi-act \
  --wandb.notes="Sai's pen pick-and-place, 60 eps, 3 cams. Only axis varied vs the sibling task is chunk_size ($CHUNK). Holdout = last 12 episodes, NOT stratified by recovery. Camera keys in this dataset are rotated by one — see the header of configs/hpc/train_pen_act.sbatch before any rollout."

echo "task $SLURM_ARRAY_TASK_ID finished rc=$? at $(date)"

# ─────────────────────────────────────────────────────────────────────────────
# WHAT THE TWO EVAL LOSSES CAN AND CANNOT TELL YOU
#
# CAN:     "does a 1.67 s or a 3.33 s action chunk fit this task better?"
#          The holdout is identical across both tasks, so eval_loss is directly
#          comparable. This is the question the run was launched to answer.
#
# CANNOT:  "which chunk is better on the robot?" Held-out L1 on teleop data is
#          not success rate. Their ordering can and does invert — see
#          experiments/2026-08-12, where a checkpoint whose held-out loss rose
#          4x matched a 100k-step ACT model on the arm. Score rollouts.
#
# CANNOT:  "did it learn to recover?" The holdout contains no induced failure.
#          Recovery needs a rollout where you displace the pen mid-episode and
#          count re-approaches.
# ─────────────────────────────────────────────────────────────────────────────

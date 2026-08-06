#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Φ — ACT on phi_so101_redcube_in_box_v1: 3 camera pairs × 1 action chunk (75).
#
#   Dataset : BrutalCaesar/phi_so101_redcube_in_box_v1  (20 eps, 11,264 frames)
#   Card    : datasets/phi_so101_redcube_in_box_v1.md
#   Shapes  : docs/theory/act-shapes.md   ← why 602 tokens, where the VRAM goes
#
#   Task: pick up a red 3D-printed cube, place it in a fixed cardboard box.
#   Deliberately simple. This is the POSITIVE CONTROL for the 8-bin run, which
#   confounded 8 bin positions with a hard-to-grasp duck. See the experiment doc.
#
# ✅ THE CAMERA KEYS ARE CORRECT IN THIS DATASET — unlike phi_so101_8bin_v1.
#       observation.images.wrist = wrist (gripper)
#       observation.images.front = desk-level front
#       observation.images.top   = overhead
#    Verified against pixels before upload (episode 2, frame 150). Do NOT copy
#    the swapped-key mapping from train_8bin_v1.sbatch into this file.
#
# ⚠️ NO HELD-OUT SPLIT. All 20 episodes train. There is no second bin position
#    and no varied object placement, so there is nothing to hold out. Success
#    here therefore says NOTHING about generalization — it says the pipeline
#    works end to end. Do not report it as a generalization result.
#
#   Submit:  sbatch configs/hpc/train_redcube_v1.sbatch
#   Watch:   squeue -u $USER  ·  tail -f /scratch/$USER/phi/logs/redcube-<jobid>_<n>.err
#            (lerobot-train logs step lines to STDERR, not stdout)
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=phi-act-redcube
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-2
#SBATCH --output=/scratch/gupta.yashv/phi/logs/redcube-%A_%a.out
#SBATCH --error=/scratch/gupta.yashv/phi/logs/redcube-%A_%a.err

set -uo pipefail
PROJECT=/scratch/$USER/phi
REPO_ID=BrutalCaesar/phi_so101_redcube_in_box_v1

# --- the grid -----------------------------------------------------------------
# One chunk size, three camera pairs. Chunk 75 = 2.5 s of open-loop action at
# 30 fps. Task 2 is the interesting one: no wrist camera at all, which tests
# whether the close-up view is load-bearing or whether two third-person views
# are enough.
CHUNK=${CHUNK:-75}
BATCH_SIZE=${BATCH_SIZE:-8}
STEPS=${STEPS:-100000}
SAVE_FREQ=${SAVE_FREQ:-25000}

case $SLURM_ARRAY_TASK_ID in
  0) SET_NAME="wrist+front"; CAM_A=observation.images.wrist; CAM_B=observation.images.front ;;
  1) SET_NAME="wrist+top";   CAM_A=observation.images.wrist; CAM_B=observation.images.top   ;;
  2) SET_NAME="front+top";   CAM_A=observation.images.front; CAM_B=observation.images.top   ;;
  *) echo "bad array index $SLURM_ARRAY_TASK_ID (expected 0-2)"; exit 1 ;;
esac

FEATS="{\"observation.state\": {\"type\": \"STATE\", \"shape\": [6]},
        \"$CAM_A\": {\"type\": \"VISUAL\", \"shape\": [3, 480, 640]},
        \"$CAM_B\": {\"type\": \"VISUAL\", \"shape\": [3, 480, 640]}}"

RUN=act_redcube_$(printf "%s" "$SET_NAME" | tr '+' '_')_chunk${CHUNK}
OUT=$PROJECT/results/$RUN

# --- epochs: this dataset is 6.5x smaller than 8bin, at the same step count ----
# 11,264 frames. 100,000 steps x batch 8 = 800,000 samples = ~71 EPOCHS.
# The 8bin run did the same 100k steps for only 14.6 epochs (54,800 frames).
# So "same steps" is NOT the same amount of learning per sample — this run sees
# every frame ~5x more often and will overfit far harder. That is why SAVE_FREQ
# is 25,000: four checkpoints, so an earlier one can be evaluated if the last
# one has memorized. Compare checkpoints on the ARM, not on train loss.

# --- environment -------------------------------------------------------------
# 🚨 A stale torch in ~/.local/lib/python3.12/site-packages shadows the conda env
# (Python puts user-site ahead of it) and dies with `libnvJitLink.so.12: cannot
# open shared object file`. Every Φ job on Explorer must set this.
export PYTHONNOUSERSITE=1

export http_proxy=http://10.99.0.130:3128
export https_proxy=http://10.99.0.130:3128
export HF_HOME=$PROJECT/hf
export HF_LEROBOT_HOME=$PROJECT/lerobot-data     # prefetched; do not let 3 tasks race
export WANDB_DIR=$PROJECT/wandb                  # NOT $HOME — home is quota-limited
export WANDB_CACHE_DIR=$PROJECT/wandb/cache
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$PROJECT/results"

module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PROJECT/envs/lerobot"

source "$PROJECT/repo/configs/hpc/_preflight.sh"

# Fail loudly if the dataset was not prefetched — 3 tasks racing on a download
# through the proxy is how you corrupt a cache.
#
# 🚨 A Hub dataset does NOT land at $HF_LEROBOT_HOME/<repo_id>. It goes into the
#    HF hub cache layout underneath it:
#      $HF_LEROBOT_HOME/hub/datasets--<org>--<name>/snapshots/<sha>/
#    So checking for <repo_id>/meta/info.json always fails. Ask the library where
#    it is instead of guessing, with HF_HUB_OFFLINE=1 so a cache miss errors
#    immediately rather than silently starting 3 parallel downloads.
#
#    Prefetch (detached, survives an SSH drop — srun does not):
#      sbatch --partition=short --cpus-per-task=4 --mem=16G --time=00:30:00 \
#        configs/hpc/_prefetch_redcube.sh
HF_HUB_OFFLINE=1 python - "$REPO_ID" <<'PY' || exit 1
import sys
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(sys.argv[1])
except Exception as e:
    sys.exit(f"dataset not prefetched ({type(e).__name__}: {e})\n"
             f"Submit configs/hpc/_prefetch_redcube.sh with sbatch first.")
m = ds.meta
cams = [k for k in m.features if "image" in k]
print(f"dataset OK: {m.total_episodes} episodes, {m.total_frames} frames, fps {m.fps}")
print("cameras present:", cams)
assert len(cams) == 3, f"expected 3 cameras, found {cams}"
assert m.total_episodes == 20, f"expected 20 episodes, found {m.total_episodes}"
PY

# W&B: online only if a credential already exists, else log to disk and sync later.
# `wandb login` is the user's to run; nothing here reads the key.
if grep -q "api.wandb.ai" "$HOME/.netrc" 2>/dev/null || [ -n "${WANDB_API_KEY:-}" ]; then
  WANDB_MODE_ARG=online
else
  WANDB_MODE_ARG=offline
  echo "NOTE: no W&B credential found -> logging OFFLINE to $WANDB_DIR."
  echo "      Sync later from the LOGIN node:  wandb sync $WANDB_DIR/offline-run-*"
fi

echo "=============================================================="
echo " task $SLURM_ARRAY_TASK_ID  run=$RUN"
echo " cameras            : $SET_NAME"
echo " dataset keys       : $CAM_A + $CAM_B      <- keys are CORRECT in this dataset"
echo " chunk_size         : $CHUNK   (open-loop $(python -c "print(f'{$CHUNK/30:.2f}')") s at 30 fps)"
echo " batch/steps/save   : $BATCH_SIZE / $STEPS / $SAVE_FREQ"
echo " epochs             : $(python -c "print(f'{$STEPS*$BATCH_SIZE/11264:.1f}')")  (8bin run was 14.6)"
echo " wandb              : $WANDB_MODE_ARG"
echo " node/GPU           : $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=============================================================="

srun lerobot-train \
  --dataset.repo_id=$REPO_ID \
  --dataset.video_backend=pyav \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.input_features="$FEATS" \
  --policy.chunk_size=$CHUNK \
  --policy.n_action_steps=$CHUNK \
  --batch_size=$BATCH_SIZE \
  --steps=$STEPS \
  --save_freq=$SAVE_FREQ \
  --num_workers=16 \
  --seed=1000 \
  --output_dir="$OUT" \
  --job_name="$RUN" \
  --wandb.enable=true \
  --wandb.mode=$WANDB_MODE_ARG \
  --wandb.project=phi-act \
  --wandb.notes="redcube v1 (red cube -> cardboard box, 20 eps, keys CORRECT). Cameras $SET_NAME = $CAM_A + $CAM_B. chunk=$CHUNK. All 20 episodes train; NO held-out split, so this measures fit, not generalization."

echo "task $SLURM_ARRAY_TASK_ID finished rc=$? at $(date)"

#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Φ — ACT on phi_so101_8bin_v1: 2 camera sets × 3 action chunks = 6 runs.
#
#   Dataset : BrutalCaesar/phi_so101_8bin_v1  (119 eps, 72,518 frames, public)
#   Card    : datasets/phi_so101_8bin_v1.md   ← bin→episode map lives ONLY there
#   Shapes  : docs/theory/act-shapes.md       ← why 602 tokens, where VRAM goes
#
#   Train on 89 episodes (6 bins). Left bin 3 and right bin 2 are HELD OUT for
#   out-of-distribution eval and never appear below.
#
# 🚨 THE DATASET CAMERA KEYS ARE SWAPPED vs the physical rig:
#       physical wrist  → observation.images.top
#       physical top    → observation.images.wrist
#       physical front  → observation.images.front   (correct)
#    So BOTH camera sets below include `top`. This is not a typo. The mapping is
#    consistent across all 119 episodes, and sanity_check_dataset_robot_
#    compatibility does NOT detect it. See the dataset card.
#
#   Submit:  sbatch configs/hpc/train_8bin_v1.sbatch
#   Watch:   squeue -u $USER  ·  tail -f /scratch/$USER/phi/logs/train-<jobid>_<n>.out
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=phi-act-8bin
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=d1025,c2204,c2205,c2206,c2207
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-5
#SBATCH --output=/scratch/gupta.yashv/phi/logs/train-%A_%a.out
#SBATCH --error=/scratch/gupta.yashv/phi/logs/train-%A_%a.err

set -uo pipefail
PROJECT=/scratch/$USER/phi
REPO_ID=BrutalCaesar/phi_so101_8bin_v1

# --- MEASURED on V100-SXM2-32GB, 2 cams, chunk 100, pyav (bench 8935610) -----
#   batch 16  updt_s 0.299  data_s 0.005  3.29 steps/s  53 smpl/s   6.91 GB
#   batch 32  updt_s 0.577  data_s 0.009  1.71 steps/s  55 smpl/s  13.20 GB
#   batch 48  updt_s 0.851  data_s 0.012  1.16 steps/s  56 smpl/s  19.49 GB
#
# smpl/s is FLAT (53->56 for 3x the batch): the GPU is saturated by batch 16, so
# wall time tracks samples processed, not batch size. VRAM is linear at
# 0.62 GB + 0.393 GB/sample, so batch 71 would fit 32 GB — memory is not the
# constraint. NOTE these are reference numbers only; we do NOT pick a batch size
# from them. Because smpl/s is flat, a matched-EPOCH run costs the same wall time
# at any batch size, so there is no throughput argument for deviating from the
# default. (A larger batch also looks better at equal STEPS in a bench — 2.74 /
# 2.44 / 2.34 at 16/32/48 — but only because it consumed 2x and 3x the data by
# then. That is not evidence of better optimisation.)
#
# Under-shooting is cheap (--resume from `last`); over-running the 8 h wall
# loses the whole run. See docs/hpc/explorer.md.
#
# EXCLUSIONS (see #SBATCH above): d1025 is the only sub-32GB node (t4), and
# c2204-c2207 are the Zen-1 nodes (AMD EPYC 7351, 2.4 GHz max) whose HOST CPU is
# the bottleneck. Measured 2026-08-04 on identical work: updt_s 1.712 vs 0.570 and
# 17 vs 55 smpl/s against an Intel Skylake node — a 3x penalty — with the GPU
# drawing only 76 W of 250 W at 1365/1380 MHz and PCIe x16, i.e. NOT the GPU.
# ACT is kernel-launch bound at these tensor sizes, so host single-thread speed
# sets updt_s. At 1.86 s/step a 34k-step run needs 17.5 h and the 8 h wall would
# kill it at ~45%. The A100 nodes are Zen 2 and are fine; only Zen 1 is excluded.
# ⇒ LEROBOT DEFAULTS, deliberately. batch 8 / 100,000 steps / save every 20,000
# is what `lerobot-train` ships with, and optimizer_lr=1e-5 is the LR paired with
# THAT batch size (nothing in LeRobot rescales the LR — verified). Sai, Parv and
# Yash all run these numbers so the camera-set comparison is apples-to-apples:
# step count is NOT comparable across batch sizes (34,000 steps is 20 epochs at
# batch 32 but 5 epochs at batch 8), so the shared unit has to be the config.
#   100,000 x 8 = 800,000 samples / 54,770 frames = 14.6 epochs
#   updt_s(8) ~ 0.161 s from the linear fit over the benched 16/32/48
#   -> ~4.5 h on a V100-SXM2, ~1.7 h on an H200. Both inside the 8 h wall.
BATCH_SIZE=${BATCH_SIZE:-8}
STEPS=${STEPS:-100000}
SAVE_FREQ=${SAVE_FREQ:-20000}

# --- the 6-run grid ----------------------------------------------------------
# task 0,1,2 = camera set A (physical wrist + top)   at chunk 50, 75, 100
# task 3,4,5 = camera set B (physical wrist + front) at chunk 50, 75, 100
CHUNKS=(50 75 100)
CHUNK=${CHUNKS[$(( SLURM_ARRAY_TASK_ID % 3 ))]}
SET_ID=$(( SLURM_ARRAY_TASK_ID / 3 ))

if [ "$SET_ID" -eq 0 ]; then
  SET_NAME="wrist+top"        # physical
  CAM_A=observation.images.top    # = physical wrist
  CAM_B=observation.images.wrist  # = physical top
else
  SET_NAME="wrist+front"      # physical
  CAM_A=observation.images.top    # = physical wrist
  CAM_B=observation.images.front  # = physical front
fi

FEATS="{\"observation.state\": {\"type\": \"STATE\", \"shape\": [6]},
        \"$CAM_A\": {\"type\": \"VISUAL\", \"shape\": [3, 480, 640]},
        \"$CAM_B\": {\"type\": \"VISUAL\", \"shape\": [3, 480, 640]}}"

RUN=act_8bin_$(printf "%s" "$SET_NAME" | tr '+' '_')_chunk${CHUNK}
OUT=$PROJECT/results/$RUN

# --- environment -------------------------------------------------------------
# 🚨 A stale torch in ~/.local/lib/python3.12/site-packages shadows the conda env
# (Python puts user-site ahead of it) and dies with `libnvJitLink.so.12: cannot
# open shared object file`. Every Φ job on Explorer must set this.
export PYTHONNOUSERSITE=1

export http_proxy=http://10.99.0.130:3128
export https_proxy=http://10.99.0.130:3128
export HF_HOME=$PROJECT/hf
export HF_LEROBOT_HOME=$PROJECT/lerobot-data     # prefetched; do not let 6 tasks race
export WANDB_DIR=$PROJECT/wandb                  # NOT $HOME — home is quota-limited
export WANDB_CACHE_DIR=$PROJECT/wandb/cache
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$PROJECT/results"

module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PROJECT/envs/lerobot"

source "$PROJECT/repo/configs/hpc/_preflight.sh"

# 89 training episodes = 6 bins. Post-deletion indices (original ep 18 is gone,
# everything after it shifted down by one). DO NOT re-derive from 0..119.
# Built with python, not `seq -s,` — BSD seq appends a TRAILING separator and
# would emit "28,,44" on a Mac while working fine under GNU seq on Explorer.
EPISODES=$(python -c "print(list(range(0,29))+list(range(44,74))+list(range(89,119)))" | tr -d ' ')
python - "$EPISODES" <<'PY' || exit 1
import ast, sys
eps = ast.literal_eval(sys.argv[1])
held = set(range(29, 44)) | set(range(74, 89))          # left bin 3, right bin 2
assert len(eps) == 89, f"expected 89 train episodes, built {len(eps)}"
assert not set(eps) & held, f"train split leaks held-out eval episodes: {sorted(set(eps)&held)}"
assert max(eps) == 118, f"max episode index should be 118, got {max(eps)}"
print(f"train split OK: {len(eps)} episodes, no overlap with the 30 held-out eval episodes")
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
echo " cameras (physical) : $SET_NAME"
echo " cameras (dataset)  : $CAM_A + $CAM_B      <- swapped keys, see header"
echo " chunk_size         : $CHUNK   (open-loop $(python -c "print(f'{$CHUNK/30:.2f}')") s at 30 fps)"
echo " batch/steps        : $BATCH_SIZE / $STEPS"
echo " wandb              : $WANDB_MODE_ARG"
echo " node/GPU           : $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=============================================================="

srun lerobot-train \
  --dataset.repo_id=$REPO_ID \
  --dataset.episodes="$EPISODES" \
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
  --num_workers=8 \
  --seed=1000 \
  --output_dir="$OUT" \
  --job_name="$RUN" \
  --wandb.enable=true \
  --wandb.mode=$WANDB_MODE_ARG \
  --wandb.project=phi-act \
  --wandb.notes="8bin v1. Cameras (physical) $SET_NAME = dataset keys $CAM_A + $CAM_B (SWAPPED, see dataset card). chunk=$CHUNK. 89 train eps; left bin 3 + right bin 2 held out for OOD eval."

echo "task $SLURM_ARRAY_TASK_ID finished rc=$? at $(date)"

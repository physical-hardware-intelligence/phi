#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Φ — π₀.₅ finetune on cubes + cylinder, language-conditioned.
#
#   Base    : lerobot/pi05_base  (~3.5B params, 14.47 GB fp32 on the Hub)
#   Dataset : BrutalCaesar/phi_so101_cubes_cylinder_lang_v1
#             120 episodes · 66,873 frames · 30 fps · 3 cams · state/action 6
#             6 task strings — the reason to use π₀ at all
#
#   Submit the SAME file repeatedly; it resumes itself:
#       sbatch configs/hpc/train_pi05_cubcyl.sbatch          # job 1, fresh
#       sbatch --dependency=afterok:<id> configs/hpc/train_pi05_cubcyl.sbatch
#
# ─────────────────────────────────────────────────────────────────────────────
# 🚨 THE FEATURE NAMES DO NOT MATCH, AND A WRONG MAP WILL NOT RAISE.
#
#    pi05_base was pretrained on base_0_rgb / left_wrist_0_rgb / right_wrist_0_rgb.
#    Our dataset has front / wrist / top. --rename_map bridges them, and passing
#    one DISABLES validate_visual_features_consistency (factory.py:649). Read the
#    first 30 log lines and confirm the three cameras were found.
#
#    front -> base_0_rgb and wrist -> left_wrist_0_rgb are natural.
#    top -> right_wrist_0_rgb is NOT: those weights saw a wrist cam, we feed
#    overhead. Finetuning should adapt it. If you would rather not, drop the top
#    mapping and add --policy.empty_cameras=1.
#
# 🚨 --steps MUST EQUAL scheduler_decay_steps.
#    PI05Config defaults scheduler_decay_steps=30000. Set --steps to anything
#    else and the cosine schedule no longer lands at the end of training. Same
#    trap as docs/theory/diffusion-policy-transformer.md §15.
#
# 🚨 --policy.push_to_hub=false IS NOT OPTIONAL.
#    PI05Config defaults push_to_hub=True with repo_id=None, and cfg.validate()
#    raises "'repo_id' argument missing" at lerobot_train.py:203 — BEFORE any
#    training starts. Caught by the smoke job 10368256 on 2026-09-15, which is
#    the only reason it did not waste a 12-hour H200 queue wait.
#
# 🚨 eval_split HOLDS OUT PER TASK STRING, and this dataset has 6 of them.
#    Verified for 1 task (pen, see train_pen_act.sbatch). NOT verified for 6.
#    0.09 is intended to hold out 2 per task = 12 total. CHECK THE LOGGED COUNT
#    on job 1 before trusting any eval number.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=phi-pi05-cubcyl
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/home/gupta.yashv/phi/logs/pi05-%j.out
#SBATCH --error=/home/gupta.yashv/phi/logs/pi05-%j.err

set -euo pipefail

REAL_HOME=/home/gupta.yashv
PHI=$REAL_HOME/phi
SCRATCH=/scratch/gupta.yashv
ENV_PREFIX=$SCRATCH/envs/phi-cuda
RESULTS=$SCRATCH/phi-results

# 🚨 Without this, ~/.local/lib/python3.12/site-packages/torch (cu121) shadows
#    the env and `import torch` dies with "libnvJitLink.so.12: cannot open
#    shared object file". Reproduced on the login node 2026-09-15.
export PYTHONNOUSERSITE=1
export HF_HOME=$SCRATCH/.cache/huggingface
export WANDB_DIR=$SCRATCH/wandb
export WANDB_CACHE_DIR=$SCRATCH/.cache/wandb
mkdir -p "$PHI/logs" "$WANDB_DIR" "$RESULTS"

module load anaconda3/2024.06 2>/dev/null || true
export PATH="$ENV_PREFIX/bin:$PATH"

# π₀ needs PaliGemma from transformers; lerobot's [pi] extra is transformers+scipy.
# Neither is in the env as of 2026-09-15. Installed once, then skipped.
python -c "import transformers, scipy" 2>/dev/null || {
  echo "installing the [pi] extra into $ENV_PREFIX"
  pip install --no-cache-dir transformers scipy
}

REPO_ID=BrutalCaesar/phi_so101_cubes_cylinder_lang_v1
RUN=pi05_cubcyl_lang
OUT=$RESULTS/$RUN
STEPS=30000                      # must match scheduler_decay_steps
BATCH_SIZE=${BATCH_SIZE:-16}
SAVE_FREQ=${SAVE_FREQ:-2500}     # an 8h job must land checkpoints even if it is cut short
EVAL_STEPS=${EVAL_STEPS:-2500}
EVAL_SPLIT=0.09

RENAME="{\"observation.images.front\": \"observation.images.base_0_rgb\",
         \"observation.images.wrist\": \"observation.images.left_wrist_0_rgb\",
         \"observation.images.top\":   \"observation.images.right_wrist_0_rgb\"}"

echo "──────────────────────────────────────────────────────────────"
echo " run      : $RUN"
echo " dataset  : $REPO_ID"
echo " node     : $(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -c "import torch;print('cuda avail:',torch.cuda.is_available(),'| built for',torch.version.cuda)"
echo "──────────────────────────────────────────────────────────────"

# Resume is decided by what is on disk, so the same file can be submitted N times.
# --output_dir is REQUIRED on resume: without it lerobot resumes into a fresh
# outputs/train/<date>/<time>_resume directory (configs/train.py:203) and the
# chain scatters across directories.
if [ -L "$OUT/checkpoints/last" ]; then
  echo "resuming from $(readlink -f "$OUT/checkpoints/last")"
  srun lerobot-train \
    --config_path="$OUT/checkpoints/last/pretrained_model/train_config.json" \
    --resume=true \
    --policy.push_to_hub=false \
    --output_dir="$OUT"
else
  echo "fresh start from lerobot/pi05_base"
  srun lerobot-train \
    --policy.path=lerobot/pi05_base \
    --policy.device=cuda \
    --policy.dtype=bfloat16 \
    --policy.push_to_hub=false \
    --policy.chunk_size=50 \
    --policy.n_action_steps=50 \
    --rename_map="$RENAME" \
    --dataset.repo_id=$REPO_ID \
    --dataset.eval_split=$EVAL_SPLIT \
    --dataset.video_backend=pyav \
    --batch_size=$BATCH_SIZE \
    --steps=$STEPS \
    --save_freq=$SAVE_FREQ \
    --eval_steps=$EVAL_STEPS \
    --seed=1000 \
    --output_dir="$OUT" \
    --job_name="$RUN" \
    --wandb.enable=true \
    --wandb.project=phi-pi05 \
    --wandb.notes="pi05_base finetuned on the 6-task language cubes/cylinder set. rename_map maps front/wrist/top onto base/left_wrist/right_wrist; the top->right_wrist leg is a deliberate semantic mismatch. Chained across 8h jobs."
fi

echo "finished rc=$? at $(date)"

# ─────────────────────────────────────────────────────────────────────────────
# WHAT TO CHECK ON JOB 1, BEFORE QUEUEING JOBS 2-4
#
# 1. Three cameras found, under the RENAMED keys. A silent rename failure looks
#    like normal training that never learns.
# 2. The held-out episode count. If it is not 12, the per-task-string rounding
#    is not what this file assumes — fix EVAL_SPLIT before the chain runs.
# 3. it/s. Multiply out to 30k steps to size the chain. ACT (51.6M) did 100k
#    steps at batch 8 in 3h03m on an A100; π₀.₅ is ~68x the parameters, so
#    expect 3-4 jobs, not 1. That is an extrapolation, not a measurement.
# 4. Peak VRAM. If it is near 141 GB, add --policy.gradient_checkpointing=true
#    (costs throughput) or halve BATCH_SIZE before chaining.
# ─────────────────────────────────────────────────────────────────────────────

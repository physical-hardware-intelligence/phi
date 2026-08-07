#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Φ — ACT on phi_so101_cubes_cylinder_v1, position holdout, WITH VALIDATION LOSS.
#
#   Dataset : BrutalCaesar/phi_so101_cubes_cylinder_v1  (120 eps, 66,873 frames)
#   Card    : datasets/phi_so101_cubes_cylinder_v1.md
#   Plan    : experiments/2026-08-06_act-cubes-cylinder-splits.md   (Axis 2)
#
#   Two runs:  task 0 = ACT with CVAE (the model)
#              task 1 = ACT with use_vae=false (behavioural-cloning baseline)
#
#   Prefetch first (once, detached):
#     sbatch --partition=short --cpus-per-task=4 --mem=16G --time=00:45:00 \
#       --output=/scratch/$USER/phi/logs/prefetch-%j.out \
#       configs/hpc/_prefetch_dataset.sh BrutalCaesar/phi_so101_cubes_cylinder_v1
#
#   Submit:  sbatch configs/hpc/train_cubes_cylinder.sbatch
#   Watch:   tail -f /scratch/$USER/phi/logs/cubcyl-<jobid>_<n>.err
#            (lerobot-train logs to STDERR; eval lines read `step N: eval_loss=...`)
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=phi-act-cubcyl
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-1
#SBATCH --output=/scratch/gupta.yashv/phi/logs/cubcyl-%A_%a.out
#SBATCH --error=/scratch/gupta.yashv/phi/logs/cubcyl-%A_%a.err

set -uo pipefail
PROJECT=/scratch/$USER/phi
REPO_ID=BrutalCaesar/phi_so101_cubes_cylinder_v1

CHUNK=${CHUNK:-50}
BATCH_SIZE=${BATCH_SIZE:-8}
STEPS=${STEPS:-100000}
SAVE_FREQ=${SAVE_FREQ:-20000}
EVAL_STEPS=${EVAL_STEPS:-10000}
# keys are CORRECT in this dataset — no swap. CAM_C empty = two-camera run.
CAM_A=observation.images.wrist
CAM_B=observation.images.front
CAM_C=

# β sweep. The demonstrations use ~3 distinct grasp approaches per object, so the
# latent needs at least ln(3) ≈ 1.10 nats to represent which one a demo took. At
# the default kl_weight=10 the measured KL is ~0.005 nats — about 0.4% of that —
# so the latent cannot be encoding mode identity and the decoder is averaging the
# three approaches. β is the direct lever: it sets the price the optimizer pays
# per nat of latent information.
#
#   --array=0-1  the original pair (kl_weight=10 default, and the BC baseline)
#   --array=2-3  the β sweep
case $SLURM_ARRAY_TASK_ID in
  0) USE_VAE=true;  KL=10.0; TAG=cvae      ;;   # LeRobot default
  1) USE_VAE=false; KL=10.0; TAG=bc        ;;   # no latent; KL unused
  2) USE_VAE=true;  KL=1.0;  TAG=cvae_kl1  ;;
  3) USE_VAE=true;  KL=0.1;  TAG=cvae_kl0p1;;
  # 4-5: the same CVAE / BC pair but with ALL THREE cameras, so the overhead view
  # is available. 903 encoder tokens instead of 602.
  4) USE_VAE=true;  KL=10.0; TAG=cvae_3cam; CAM_C=observation.images.top ;;
  5) USE_VAE=false; KL=10.0; TAG=bc_3cam;   CAM_C=observation.images.top ;;
  *) echo "bad array index $SLURM_ARRAY_TASK_ID (expected 0-5)"; exit 1 ;;
esac

CAMS="\"$CAM_A\": {\"type\": \"VISUAL\", \"shape\": [3, 480, 640]},
      \"$CAM_B\": {\"type\": \"VISUAL\", \"shape\": [3, 480, 640]}"
if [ -n "$CAM_C" ]; then
  CAMS="$CAMS,
      \"$CAM_C\": {\"type\": \"VISUAL\", \"shape\": [3, 480, 640]}"
fi
FEATS="{\"observation.state\": {\"type\": \"STATE\", \"shape\": [6]}, $CAMS}"

RUN=act_cubcyl_poshold_chunk${CHUNK}_${TAG}
OUT=$PROJECT/results/$RUN

# --- the split, and the trick that makes lerobot produce it --------------------
# LeRobot CAN compute held-out loss natively (`--dataset.eval_split` + `--eval_steps`),
# but `make_train_eval_datasets` holds out the LAST ceil(n * eval_split) episodes
# PER TASK. All 120 of our episodes share one task string, so a plain eval_split
# would hold out 90-119 — the white cube only — not our stratified 30.
#
# 🔑 `LeRobotDataset` PRESERVES the order of `--dataset.episodes` (verified).
#    So we pass the 90 training episodes FIRST and the 30 held-out episodes LAST,
#    with eval_split = 30/120 = 0.25. The "last 30" then are exactly our design.
#    Verified against the real metadata: the resulting eval set == our holdout.
#
# Holdout = 5 episodes from each of the 6 object x container blocks:
#   red cube    0-4    (cardboard)   20-24   (white bin)
#   cylinder    45-49  (cardboard)   65-69   (white bin)
#   white cube  90-94  (cardboard)   110-114 (white bin)
EPISODES=$(python -c "
held  = [*range(0,5), *range(20,25), *range(45,50), *range(65,70), *range(90,95), *range(110,115)]
train = [e for e in range(120) if e not in set(held)]
print(train + held)
" | tr -d ' ')

python - "$EPISODES" <<'PY' || exit 1
import ast, math, sys
eps = ast.literal_eval(sys.argv[1])
held = set([*range(0,5), *range(20,25), *range(45,50), *range(65,70), *range(90,95), *range(110,115)])
assert len(eps) == 120 and len(set(eps)) == 120, "episode list must be all 120, no repeats"
n_eval = math.ceil(len(eps) * 0.25)
assert set(eps[-n_eval:]) == held, "the last 30 entries must be exactly the holdout"
assert not (set(eps[:-n_eval]) & held), "training portion leaks held-out episodes"
print(f"split OK: {len(eps)-n_eval} train + {n_eval} held out, holdout is last in the list")
PY

# --- environment -------------------------------------------------------------
# 🚨 A stale torch in ~/.local shadows the conda env and dies with
# `libnvJitLink.so.12`. Every Φ job on Explorer must set this.
export PYTHONNOUSERSITE=1

export http_proxy=http://10.99.0.130:3128
export https_proxy=http://10.99.0.130:3128
export HF_HOME=$PROJECT/hf
export HF_LEROBOT_HOME=$PROJECT/lerobot-data
export WANDB_DIR=$PROJECT/wandb
export WANDB_CACHE_DIR=$PROJECT/wandb/cache
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$PROJECT/results"

module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PROJECT/envs/lerobot"

source "$PROJECT/repo/configs/hpc/_preflight.sh"

# Dataset must be prefetched. Check the cache ON DISK — do NOT gate this with
# HF_HUB_OFFLINE=1, because LeRobotDataset resolves the v3.0 version TAG through
# the Hub API even when every file is cached, and offline mode then fails on a
# perfectly good cache.
python - "$REPO_ID" <<'PY' || exit 1
import glob, json, os, sys
repo = sys.argv[1]
pat = os.path.join(os.environ["HF_LEROBOT_HOME"], "hub",
                   "datasets--" + repo.replace("/", "--"), "snapshots", "*", "meta", "info.json")
hits = sorted(glob.glob(pat))
if not hits:
    sys.exit(f"dataset not prefetched (nothing matching {pat})\n"
             f"Run configs/hpc/_prefetch_dataset.sh {repo} with sbatch first.")
m = json.load(open(hits[-1]))
print(f"dataset OK: {m['total_episodes']} episodes, {m['total_frames']} frames")
assert m["total_episodes"] == 120, f"expected 120 episodes, found {m['total_episodes']}"
PY

if grep -q "api.wandb.ai" "$HOME/.netrc" 2>/dev/null || [ -n "${WANDB_API_KEY:-}" ]; then
  WANDB_MODE_ARG=online
else
  WANDB_MODE_ARG=offline
  echo "NOTE: no W&B credential -> logging OFFLINE to $WANDB_DIR (sync later from the login node)."
fi

echo "=============================================================="
echo " task $SLURM_ARRAY_TASK_ID  run=$RUN"
echo " cameras       : $CAM_A $CAM_B $CAM_C"
echo " chunk_size    : $CHUNK   (open-loop $(python -c "print(f'{$CHUNK/30:.2f}')") s at 30 fps)"
echo " use_vae       : $USE_VAE   (kl_weight=$KL)"
echo " batch/steps   : $BATCH_SIZE / $STEPS   (save every $SAVE_FREQ, eval every $EVAL_STEPS)"
echo " train epochs  : $(python -c "print(f'{$STEPS*$BATCH_SIZE/49969:.1f}')")  (49,969 train frames)"
echo " node/GPU      : $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=============================================================="

srun lerobot-train \
  --dataset.repo_id=$REPO_ID \
  --dataset.episodes="$EPISODES" \
  --dataset.eval_split=0.25 \
  --dataset.video_backend=pyav \
  --eval_steps=$EVAL_STEPS \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.use_vae=$USE_VAE \
  --policy.kl_weight=$KL \
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
  --wandb.notes="cubes+cylinder, position holdout (5 eps from each of 6 object x bin blocks, 30 held out). use_vae=$USE_VAE, kl_weight=$KL. chunk=$CHUNK. eval_loss on the held-out 30 every $EVAL_STEPS steps."

echo "task $SLURM_ARRAY_TASK_ID finished rc=$? at $(date)"

# --- reading the eval numbers -------------------------------------------------
# W&B: `eval_loss` under mode="eval". Logs: `step N: eval_loss=...` in the .err file.
#   grep -oE "step [0-9]+: eval_loss=[0-9.]+" /scratch/$USER/phi/logs/cubcyl-<jobid>_<n>.err
#
# 🚨 Two things about eval_loss that will mislead you if you skip them:
#
# 1. It is the FULL ACT loss, l1 + kl_weight * kld — not pure L1. So it is NOT
#    directly comparable between task 0 (CVAE) and task 1 (use_vae=false), where
#    the KL term does not exist at all. Our KL measured ~0, so they should be
#    close, but confirm with the train-side `kld_loss` before treating the BC
#    baseline's lower number as better.
#
# 2. `--max_eval_samples` is deliberately NOT set. It selects the FIRST N frames
#    of the eval set, and since frames are ordered by episode that would mean
#    episodes 0-4 only — red cube, cardboard bin. A biased subset of a
#    deliberately stratified holdout. Full eval costs ~2,100 batches (~1-2 min on
#    an H200), which at eval_steps=10000 is ~15 min over the whole run. Pay it.

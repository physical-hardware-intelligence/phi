#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Φ — ACT on phi_so101_cubes_cylinder_recovery_v1. IMAGE AUGMENTATION A/B.
#
#   Dataset : BrutalCaesar/phi_so101_cubes_cylinder_recovery_v1
#             143 eps / 81,943 frames = 120 clean + 23 failure-then-recovery
#   Card    : datasets/phi_so101_cubes_cylinder_recovery_v1.md
#   Control : the `cvae_3cam` run from train_cubes_cylinder.sbatch (task 4).
#             Same policy, same cameras, same holdout — the ONLY differences are
#             the 23 recovery episodes and, on task 1, the augmentation.
#
#   Two runs, one variable:
#     task 0 = image augmentation OFF   (LeRobot default: image_transforms.enable=false)
#     task 1 = image augmentation ON    (the stock 6-transform stack, 3 sampled per frame)
#
#   Prefetch first (once, detached):
#     sbatch --partition=short --cpus-per-task=4 --mem=16G --time=00:45:00 \
#       --output=/scratch/$USER/phi/logs/prefetch-%j.out \
#       configs/hpc/_prefetch_dataset.sh BrutalCaesar/phi_so101_cubes_cylinder_recovery_v1
#
#   Submit:  sbatch configs/hpc/train_recovery.sbatch
#   Watch:   tail -f /scratch/$USER/phi/logs/recovery-<jobid>_<n>.err
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=phi-act-recovery
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-1
#SBATCH --output=/scratch/gupta.yashv/phi/logs/recovery-%A_%a.out
#SBATCH --error=/scratch/gupta.yashv/phi/logs/recovery-%A_%a.err

set -uo pipefail
PROJECT=/scratch/$USER/phi
REPO_ID=BrutalCaesar/phi_so101_cubes_cylinder_recovery_v1

CHUNK=${CHUNK:-50}
BATCH_SIZE=${BATCH_SIZE:-8}
STEPS=${STEPS:-100000}
SAVE_FREQ=${SAVE_FREQ:-20000}
EVAL_STEPS=${EVAL_STEPS:-10000}

# Three cameras. Keys are CORRECT in this dataset (inherited from the parent) —
# no swap. Do NOT reorder these to match a camera INDEX you saw at record time;
# indices are not stable across sessions, the KEYS are what the policy binds to.
CAM_A=observation.images.wrist
CAM_B=observation.images.front
CAM_C=observation.images.top

# --- the single variable -------------------------------------------------------
case $SLURM_ARRAY_TASK_ID in
  0) AUG=false; TAG=recovery_3cam_noaug ;;
  1) AUG=true;  TAG=recovery_3cam_aug   ;;
  *) echo "bad array index $SLURM_ARRAY_TASK_ID (expected 0-1)"; exit 1 ;;
esac

# What `--dataset.image_transforms.enable=true` actually turns on (lerobot 0.6.0,
# ImageTransformsConfig defaults — read them, they are not all harmless):
#
#   max_num_transforms = 3   →  3 of the 6 below are sampled per frame, uniformly
#   brightness   ColorJitter      brightness (0.8, 1.2)
#   contrast     ColorJitter      contrast   (0.8, 1.2)
#   saturation   ColorJitter      saturation (0.5, 1.5)
#   hue          ColorJitter      hue        (-0.05, 0.05)
#   sharpness    SharpnessJitter  sharpness  (0.5, 1.5)
#   affine       RandomAffine     degrees (-5, 5), translate (0.05, 0.05)
#
# 🚨 PRE-REGISTERED PREDICTION, so we do not rationalise afterwards.
#    The five photometric transforms are the ones we actually want: our webcams
#    cannot lock exposure, so lighting genuinely varies between sessions and the
#    policy should be invariant to it.
#    `affine` is the suspect one. translate=0.05 is ±5% of the frame = ±32 px of
#    640 ≈ 2 cm at our workspace scale, about one cube width, and RandomAffine
#    moves the IMAGE while the ACTION LABEL stays put. It therefore teaches
#    "the same joint targets are correct even though the object appears a cube
#    to the left" — which is exactly the invariance a precision grasp must NOT
#    have. If task 1 loses to task 0, re-run with only affine's weight zeroed
#    before concluding that augmentation as a category does not help here.
#
#    🚨 The obvious syntax for that does NOT work. Verified against draccus in
#    lerobot 0.6.0 on 2026-08-10:
#      --dataset.image_transforms.tfs.affine.weight=0.0   -> "unrecognized argument"
#      --dataset.image_transforms.tfs='{"affine": {...}}'  -> parses, but REPLACES
#                                                             the dict: you end up
#                                                             with affine ONLY and
#                                                             lose the other five.
#    `tfs` is a single dict-valued argument, so you must pass all six entries.
#    Set AUG_TFS below and add --dataset.image_transforms.tfs="$AUG_TFS".
#
#    This run is deliberately the plain on/off comparison Yash asked for. The
#    affine-only ablation is the FOLLOW-UP, not a silent modification of it.

# The affine-off ablation, pre-wired but OFF by default. `sbatch --export=ALL,AFFINE_OFF=1 ...`
# re-runs task 1 with the five photometric transforms and affine's weight at 0.
# All six entries must be listed — see the note above about `tfs` replacing the dict.
AUG_TFS='{"brightness": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"brightness": [0.8, 1.2]}},
          "contrast":   {"weight": 1.0, "type": "ColorJitter", "kwargs": {"contrast": [0.8, 1.2]}},
          "saturation": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"saturation": [0.5, 1.5]}},
          "hue":        {"weight": 1.0, "type": "ColorJitter", "kwargs": {"hue": [-0.05, 0.05]}},
          "sharpness":  {"weight": 1.0, "type": "SharpnessJitter", "kwargs": {"sharpness": [0.5, 1.5]}},
          "affine":     {"weight": 0.0, "type": "RandomAffine", "kwargs": {"degrees": [-5.0, 5.0], "translate": [0.05, 0.05]}}}'

EXTRA_ARGS=()
if [ "$AUG" = "true" ] && [ "${AFFINE_OFF:-0}" = "1" ]; then
  EXTRA_ARGS+=(--dataset.image_transforms.tfs="$AUG_TFS")
  TAG=${TAG}_noaffine
  echo "AFFINE_OFF=1 -> photometric transforms only, affine weight 0"
fi

CAMS="\"$CAM_A\": {\"type\": \"VISUAL\", \"shape\": [3, 480, 640]},
      \"$CAM_B\": {\"type\": \"VISUAL\", \"shape\": [3, 480, 640]},
      \"$CAM_C\": {\"type\": \"VISUAL\", \"shape\": [3, 480, 640]}"
FEATS="{\"observation.state\": {\"type\": \"STATE\", \"shape\": [6]}, $CAMS}"

RUN=act_${TAG}_chunk${CHUNK}
OUT=$PROJECT/results/$RUN

# --- the split -----------------------------------------------------------------
# Requirement (Yash, 2026-08-10): ALL 23 recovery episodes go in TRAINING; the
# holdout is drawn from the clean episodes ONLY.
#
# `make_train_eval_datasets` holds out the LAST ceil(n * eval_split) episodes
# PER TASK STRING, and `LeRobotDataset` preserves the order of
# `--dataset.episodes`. So we pass the 113 training episodes first and the 30
# clean holdout episodes last.
#
# 🔑 This is also why the dataset keeps ONE task string. math.ceil is always ≥ 1,
#    so giving the recovery episodes their own task string would FORCE at least
#    one of them into eval — the opposite of the requirement. See the card.
#
# Holdout = the same 30 as the cvae_3cam control, so the two are comparable:
#   red cube    0-4    (cardboard)   20-24   (white bin)
#   cylinder    45-49  (cardboard)   65-69   (white bin)
#   white cube  90-94  (cardboard)   110-114 (white bin)
# All are < 123, so deleting old episodes 123 and 144 did not shift them.
#
# 🚨 eval_split = 0.209 and it is NOT free to round: ceil(143 * s) must equal 30.
#      0.202797 < s <= 0.209790     0.209 -> 30 OK
#                                   0.2098 -> 31 WRONG (one extra episode held out)
#                                   0.21   -> 31 WRONG
EVAL_SPLIT=0.209
EPISODES=$(python -c "
held  = [*range(0,5), *range(20,25), *range(45,50), *range(65,70), *range(90,95), *range(110,115)]
train = [e for e in range(143) if e not in set(held)]
print(train + held)
" | tr -d ' ')

python - "$EPISODES" "$EVAL_SPLIT" <<'PY' || exit 1
import ast, math, sys
eps = ast.literal_eval(sys.argv[1])
split = float(sys.argv[2])
held = set([*range(0,5), *range(20,25), *range(45,50), *range(65,70), *range(90,95), *range(110,115)])
recovery = set(range(120, 143))          # all 23, post-deletion indexing
assert len(eps) == 143 and len(set(eps)) == 143, "episode list must be all 143, no repeats"
n_eval = math.ceil(len(eps) * split)
assert n_eval == 30, f"eval_split {split} holds out {n_eval}, expected exactly 30"
assert set(eps[-n_eval:]) == held, "the last 30 entries must be exactly the clean holdout"
assert not (set(eps[:-n_eval]) & held), "training portion leaks held-out episodes"
assert recovery <= set(eps[:-n_eval]), "ALL 23 recovery episodes must be in training"
assert not (recovery & set(eps[-n_eval:])), "no recovery episode may land in eval"
print(f"split OK: {len(eps)-n_eval} train (incl. all {len(recovery)} recovery) + {n_eval} clean held out")
PY

# --- environment ---------------------------------------------------------------
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
assert m["total_episodes"] == 143, f"expected 143 episodes, found {m['total_episodes']}"
assert m["total_frames"] == 81943, f"expected 81,943 frames, found {m['total_frames']}"
PY

if grep -q "api.wandb.ai" "$HOME/.netrc" 2>/dev/null || [ -n "${WANDB_API_KEY:-}" ]; then
  WANDB_MODE_ARG=online
else
  WANDB_MODE_ARG=offline
  echo "NOTE: no W&B credential -> logging OFFLINE to $WANDB_DIR (sync later from the login node)."
fi

echo "=============================================================="
echo " task $SLURM_ARRAY_TASK_ID  run=$RUN"
echo " dataset       : $REPO_ID  (143 eps, 23 of them recovery)"
echo " cameras       : $CAM_A $CAM_B $CAM_C"
echo " image aug     : $AUG"
echo " chunk_size    : $CHUNK   (open-loop $(python -c "print(f'{$CHUNK/30:.2f}')") s at 30 fps)"
echo " use_vae       : true   (kl_weight=10.0, LeRobot default — same as the cvae_3cam control)"
echo " batch/steps   : $BATCH_SIZE / $STEPS   (save every $SAVE_FREQ, eval every $EVAL_STEPS)"
echo " train frames  : ~64,700 over 113 episodes"
echo " node/GPU      : $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=============================================================="

srun lerobot-train \
  --dataset.repo_id=$REPO_ID \
  --dataset.episodes="$EPISODES" \
  --dataset.eval_split=$EVAL_SPLIT \
  --dataset.video_backend=pyav \
  --dataset.image_transforms.enable=$AUG \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
  --eval_steps=$EVAL_STEPS \
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
  --num_workers=16 \
  --seed=1000 \
  --output_dir="$OUT" \
  --job_name="$RUN" \
  --wandb.enable=true \
  --wandb.mode=$WANDB_MODE_ARG \
  --wandb.project=phi-act \
  --wandb.notes="cubes+cylinder + 23 recovery episodes, 3 cameras. image_transforms.enable=$AUG (the ONLY difference between tasks 0 and 1). All 23 recovery eps in train; holdout = the same clean 30 as the cvae_3cam control. eval_loss every $EVAL_STEPS steps."

echo "task $SLURM_ARRAY_TASK_ID finished rc=$? at $(date)"

# --- 🚨 what these two numbers CAN and CANNOT tell you --------------------------
#
# CAN:  "did adding 23 recovery episodes hurt clean performance?"  — compare
#       either run's eval_loss against the cvae_3cam control on the SAME 30
#       clean episodes. That comparison is valid because the holdout is identical.
#
# CAN:  "did augmentation help fit?"  — task 0 vs task 1 eval_loss. Expect task 1
#       to have HIGHER train loss (it is a harder problem) and that is not a
#       failure; the held-out number is the one that matters.
#
# CANNOT: "did the policy get better at recovering?"  No loss can answer this.
#       The 30 held-out episodes are all clean — they contain no induced failure,
#       so there is nothing in them that tests recovery. Measuring recovery needs
#       a rollout where you deliberately displace or strip the object mid-episode
#       and count whether the policy re-approaches. That protocol is separate
#       work and is the actual gate on this experiment — see docs/evaluation.
#
# BEFORE any rollout: copy configs/calibration/robots/so_follower/phi_follower.json
# onto the machine. A policy trained here emits angles in THAT frame; running it
# against a different calibration mis-grasps silently, with no error at all.
# Confirmed on the arm 2026-08-10 — docs/robots/so-arm101/02-setup.md §3c.

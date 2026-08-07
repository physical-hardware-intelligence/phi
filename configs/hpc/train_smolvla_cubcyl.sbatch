#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Φ — SmolVLA fine-tune on the LANGUAGE-LABELLED cubes+cylinder dataset.
#
#   Dataset : BrutalCaesar/phi_so101_cubes_cylinder_lang_v1
#             Same 120 episodes / 66,873 frames as ..._cubes_cylinder_v1, but with
#             SIX per-episode instructions instead of one shared string:
#               0-19    pick up the red cube        ... cardboard box
#               20-39   pick up the red cube        ... white bin
#               40-59   pick up the yellow cylinder ... cardboard box
#               60-79   pick up the yellow cylinder ... white bin
#               80-99   pick up the white cube      ... cardboard box
#               100-119 pick up the white cube      ... white bin
#
#   WHY a VLA here: the demos contain ~3 grasp approaches per object. ACT's L1 +
#   collapsed latent averages them (which is what the kl_weight sweep is chasing).
#   SmolVLA's action expert is FLOW MATCHING — `sample_noise` / `sample_time` /
#   `denoise_step` — which models multimodal action distributions natively. No
#   latent, no KL, no beta to tune.
#
#   Prereq (once, and ONLY when no phi job is running):
#     sbatch --partition=short --cpus-per-task=4 --mem=16G --time=00:40:00 \
#       --output=/scratch/$USER/phi/logs/install-smolvla-%j.out \
#       configs/hpc/_install_smolvla.sh
#     sbatch --partition=short --cpus-per-task=4 --mem=16G --time=00:45:00 \
#       --output=/scratch/$USER/phi/logs/prefetch-%j.out \
#       configs/hpc/_prefetch_dataset.sh BrutalCaesar/phi_so101_cubes_cylinder_lang_v1
#
#   Submit:  sbatch configs/hpc/train_smolvla_cubcyl.sbatch
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=phi-smolvla
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --array=0-1
#SBATCH --output=/scratch/gupta.yashv/phi/logs/smolvla-%A_%a.out
#SBATCH --error=/scratch/gupta.yashv/phi/logs/smolvla-%A_%a.err

set -uo pipefail
PROJECT=/scratch/$USER/phi
REPO_ID=BrutalCaesar/phi_so101_cubes_cylinder_lang_v1

# SmolVLA fine-tuning recipe: adapt a pretrained model, do not train from scratch.
# Big batch, few steps, lr from the policy default (1e-4). Contrast the ACT runs
# at batch 8 / 100k steps.
BATCH_SIZE=${BATCH_SIZE:-64}
STEPS=${STEPS:-20000}
SAVE_FREQ=${SAVE_FREQ:-5000}
EVAL_STEPS=${EVAL_STEPS:-2500}

case $SLURM_ARRAY_TASK_ID in
  # ── Task 0: POSITION holdout, matched to the ACT runs ──────────────────────
  # Same 30 episodes held out as train_cubes_cylinder.sbatch, so SmolVLA and ACT
  # are answering the same question on the same data.
  0)
    TAG=poshold
    EPISODES=$(python -c "
held  = [*range(0,5), *range(20,25), *range(45,50), *range(65,70), *range(90,95), *range(110,115)]
train = [e for e in range(120) if e not in set(held)]
print(train + held)
" | tr -d ' ')
    EVAL_SPLIT=0.25
    NOTES="position holdout, 30 eps (5 from each of 6 blocks) — matched to the ACT runs"
    ;;
  # ── Task 1: COMPOSITIONAL holdout — the experiment ACT cannot be asked ─────
  # Hold out ALL 20 episodes of "white cube -> cardboard box". The model still
  # sees the white cube (with the white bin) and the cardboard box (with the two
  # other objects), but never that PAIR. At eval you ask for it in words.
  #
  # eval_split cannot express this: it holds out a fraction PER TASK, and we are
  # holding out one task ENTIRELY. So the 20 are simply excluded from
  # --dataset.episodes, and eval_split=0.1 carves a small in-distribution eval
  # set out of the remaining 100 purely for checkpoint selection.
  # The compositional test itself is a ROLLOUT, not a loss.
  1)
    TAG=compositional
    EPISODES=$(python -c "print([e for e in range(120) if not (80 <= e <= 99)])" | tr -d ' ')
    EVAL_SPLIT=0.1
    NOTES="compositional holdout: all 20 'white cube -> cardboard box' episodes (80-99) excluded. Test by asking for that instruction at rollout time."
    ;;
  *) echo "bad array index $SLURM_ARRAY_TASK_ID (expected 0-1)"; exit 1 ;;
esac

RUN=smolvla_cubcyl_${TAG}
OUT=$PROJECT/results/$RUN

python - "$EPISODES" "$SLURM_ARRAY_TASK_ID" <<'PY' || exit 1
import ast, sys
eps = ast.literal_eval(sys.argv[1]); task = int(sys.argv[2])
assert len(eps) == len(set(eps)), "duplicate episodes"
if task == 0:
    held = set([*range(0,5), *range(20,25), *range(45,50), *range(65,70), *range(90,95), *range(110,115)])
    assert len(eps) == 120, f"expected all 120 episodes, got {len(eps)}"
    assert set(eps[-30:]) == held, "the last 30 must be exactly the position holdout"
    print("split OK: 90 train + 30 held out (position), holdout last in the list")
else:
    assert len(eps) == 100, f"expected 100 training episodes, got {len(eps)}"
    assert not (set(eps) & set(range(80, 100))), "white-cube/cardboard episodes leaked into training"
    print("split OK: 100 train, all 20 of 'white cube -> cardboard box' (80-99) held out entirely")
PY

# --- environment -------------------------------------------------------------
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

python - "$REPO_ID" <<'PY' || exit 1
import glob, json, os, sys
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy   # noqa: F401
repo = sys.argv[1]
pat = os.path.join(os.environ["HF_LEROBOT_HOME"], "hub",
                   "datasets--" + repo.replace("/", "--"), "snapshots", "*", "meta", "info.json")
hits = sorted(glob.glob(pat))
if not hits:
    sys.exit(f"dataset not prefetched (nothing matching {pat})\n"
             f"Run configs/hpc/_prefetch_dataset.sh {repo} with sbatch first.")
m = json.load(open(hits[-1]))
print(f"dataset OK: {m['total_episodes']} episodes, {m['total_frames']} frames")
assert m["total_episodes"] == 120
print("smolvla importable: yes")
PY

# The language labels are the entire point — verify SIX tasks, not one. Point this
# at ..._cubes_cylinder_v1 by mistake and it trains on a constant instruction,
# teaching the model that language is noise.
python - "$REPO_ID" <<'PY' || exit 1
import glob, os, sys
import pandas as pd
repo = sys.argv[1]
pat = os.path.join(os.environ["HF_LEROBOT_HOME"], "hub",
                   "datasets--" + repo.replace("/", "--"), "snapshots", "*", "meta", "tasks.parquet")
t = pd.read_parquet(sorted(glob.glob(pat))[-1])
n = len(t)
print(f"distinct instructions: {n}")
for s in list(t.index)[:6]:
    print(f"   - {s}")
assert n == 6, f"expected 6 instructions, found {n} — wrong dataset?"
PY

if grep -q "api.wandb.ai" "$HOME/.netrc" 2>/dev/null || [ -n "${WANDB_API_KEY:-}" ]; then
  WANDB_MODE_ARG=online
else
  WANDB_MODE_ARG=offline
fi

echo "=============================================================="
echo " task $SLURM_ARRAY_TASK_ID  run=$RUN"
echo " split         : $NOTES"
echo " batch/steps   : $BATCH_SIZE / $STEPS  (save $SAVE_FREQ, eval $EVAL_STEPS)"
echo " eval_split    : $EVAL_SPLIT"
echo " node/GPU      : $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=============================================================="

# NOTE: input_features are NOT pinned. Fine-tuning from a pretrained checkpoint,
# lerobot derives them from the dataset — which means ALL THREE cameras, where the
# ACT runs used wrist+front only. That is a real confound in a SmolVLA-vs-ACT
# comparison; record it, and if you need a like-for-like run, add
# --policy.input_features with just the two.
srun lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.repo_id=$REPO_ID \
  --dataset.episodes="$EPISODES" \
  --dataset.eval_split=$EVAL_SPLIT \
  --dataset.video_backend=pyav \
  --eval_steps=$EVAL_STEPS \
  --batch_size=$BATCH_SIZE \
  --steps=$STEPS \
  --save_freq=$SAVE_FREQ \
  --num_workers=16 \
  --seed=1000 \
  --output_dir="$OUT" \
  --job_name="$RUN" \
  --wandb.enable=true \
  --wandb.mode=$WANDB_MODE_ARG \
  --wandb.project=phi-smolvla \
  --wandb.notes="SmolVLA fine-tune from lerobot/smolvla_base. $NOTES"

echo "task $SLURM_ARRAY_TASK_ID finished rc=$? at $(date)"

# --- if it OOMs ---------------------------------------------------------------
# batch 64 with three 512x512 camera streams through a 500M VLM is the upstream
# recipe, not a measured fit for this rig. If it dies on memory, halve the batch
# (BATCH_SIZE=32 sbatch ...) rather than cutting cameras — cameras change the
# experiment, batch size only changes the optimisation.
#
# --- before trusting any of this on hardware ----------------------------------
# MEASURE INFERENCE LATENCY on the Mac before planning rollouts. ACT runs 84 ms
# per chunk there. SmolVLA is 450M params plus a VLM plus `num_steps` denoising
# iterations, and it may not be real-time on MPS. With n_action_steps=50 you only
# query once per 1.67 s, so even ~500 ms is workable — but if it is 3 s the policy
# is undeployable on that machine regardless of how well it trains.

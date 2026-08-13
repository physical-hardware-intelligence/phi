#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Φ — Diffusion Policy with the TRANSFORMER backbone, n_emb=256.
# A/B on REGULARISATION, to answer the question the CNN run left open.
#
#   Submit:  sbatch configs/hpc/train_diffusion_transformer.sbatch
#   Watch:   squeue -u $USER ; tail -f /scratch/$USER/phi/logs/dpt-<jobid>_<n>.err
#
# WHAT THIS IS TESTING, AND WHY
# -----------------------------
# The CNN run (experiments/2026-08-12) ended with train loss 0.002 and held-out
# loss RISING 0.0155 -> 0.0636, a ~32x gap, degrading monotonically from the very
# first measurement. Two explanations fit that evidence and demand completely
# different responses:
#
#   A. THE REGULARISER WAS MISSING. The paper's §3.2 recipe assumes an EMA of the
#      weights -- its stated reason for GroupNorm is the interaction with weight
#      averaging -- and lerobot implements NO EMA (grepped: no EMAModel, no
#      use_ema anywhere in the package). That run also used weight_decay=1e-6,
#      i.e. effectively none. Under A the dataset is fine and the fix is a config
#      change: hours.
#   B. THE DATASET IS TOO SMALL. 113 training episodes / ~62,000 chunks may not
#      support the model regardless of regularisation. Under B the fix is more
#      teleop: days, and it changes what to do next entirely.
#
# The remedies differ by ~two orders of magnitude in cost, which is why this is
# worth a GPU pair rather than a guess.
#
# THE ARMS ARE THE REGULARISATION *PACKAGE*, NOT ONE KNOB
# -------------------------------------------------------
#   task 0  paper  weight_decay 1e-3  attn_dropout 0.3   <- Table 8's recipe
#   task 1  unreg  weight_decay 1e-6  attn_dropout 0.0   <- CNN-style, no reg
#
# This deliberately confounds two variables. The actionable question is binary --
# "does regularisation fix this, yes or no" -- and with two arms the priority is
# detection power. If the whole package changes nothing, neither knob alone will
# either, and that is real evidence for B. If it does, a follow-up isolates which.
#
# 🚨 THIS IS EVIDENCE ABOUT THE CNN, NOT PROOF ABOUT IT. DP-T differs from DP-CNN
#    in more than regularisation: 9,020,934 backbone params vs 68,665,222, plus a
#    different architecture. Strictly this tests whether regularisation controls
#    overfitting in DP-T. Say so when reporting it.
#
# PRE-REGISTERED PREDICTION (recorded so it cannot be retrofitted)
# ---------------------------------------------------------------
# I expect task 0 (paper) to show a materially flatter held-out curve than task 1,
# and I expect task 1 to reproduce the CNN's monotonic degradation. If BOTH arms
# degrade alike, hypothesis B gains real support and the next move is data, not
# hyperparameters.
# Calibration on my own record: of the three predictions pre-registered for the
# CNN A/B, TWO were wrong (the ImageNet arm did not win; DP did not beat ACT on
# the white cube). Weight this accordingly.
#
# 🔑 save_freq=2000, NOT 20000. The CNN run's true optimum was at 10k and was
#    never saved, because save_freq was inherited from the ACT script without
#    re-examination. Disk was never the reason: a DP-T checkpoint is 42,612,198
#    params x 4 B = 170 MB plus Adam's two moments = ~511 MB, so 15 checkpoints
#    per arm is ~7.7 GB, ~15.3 GB for both. That is not a constraint on /scratch.
#
# 🔑 eval every 2000 too. The CNN evaluated every 10k, which is exactly why "the
#    optimum is at or below 10k" could not be resolved. Eval is a forward pass
#    over 30 held-out episodes and costs well under a minute.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=phi-dpt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-1
#SBATCH --output=/scratch/gupta.yashv/phi/logs/dpt-%A_%a.out
#SBATCH --error=/scratch/gupta.yashv/phi/logs/dpt-%A_%a.err

set -uo pipefail
PROJECT=/scratch/$USER/phi
REPO_ID=BrutalCaesar/phi_so101_cubes_cylinder_recovery_v1

# STEPS=30000 rather than the CNN's 100k. That run's held-out optimum was at 10k
# and everything after was monotonic decay -- 70k of those steps bought nothing
# but confirmation. 30k with 15 checkpoints resolves the turn if there is one.
# Override after reading the calibration:  --export=ALL,STEPS=50000
STEPS=${STEPS:-30000}
SAVE_FREQ=${SAVE_FREQ:-2000}
EVAL_STEPS=${EVAL_STEPS:-2000}
LOG_FREQ=200
BATCH=${BATCH:-64}                # §A.4: "64 for all image-based experiments"
NUM_WORKERS=${NUM_WORKERS:-32}    # matches --cpus-per-task=32; the CNN's winning value

# --- the A/B ------------------------------------------------------------------
case $SLURM_ARRAY_TASK_ID in
  0) WDECAY=1e-3; DROP_ATTN=0.3; TAG=paper ;;   # Table 8's transformer recipe
  1) WDECAY=1e-6; DROP_ATTN=0.0; TAG=unreg ;;   # the CNN's regularisation, i.e. none
  *) echo "bad array index $SLURM_ARRAY_TASK_ID (expected 0-1)"; exit 1 ;;
esac
EMB=${EMB:-256}                   # Table 8 sim rows -> 9,020,934 backbone params

# --- architecture, Table 8 + the reference wrapper ----------------------------
# n_layer/n_emb from Table 8; n_head/n_cond_layers/causal_attn/p_drop_emb are the
# reference wrapper's defaults (diffusion_transformer_hybrid_image_policy.py),
# since Table 8 does not list them. p_drop_emb=0.0 confirmed there.
N_LAYER=8; N_HEAD=4; N_COND_LAYERS=0; DROP_EMB=0.0

# To/Ta/Tp inherited UNCHANGED from the CNN runs so DP-T vs DP-CNN is comparable.
# The paper ran at 10 Hz where Ta=8 is 800 ms of open loop; at 30 fps that is 24.
# NOTE Table 8's sim rows use Tp=10 where Table 7's CNN used 16 -- the paper runs
# SHORTER horizons on the transformer. Keeping 48 is a deliberate deviation for
# comparability, flagged rather than silently inherited.
TO=2; TA=24; TP=48
DROP=$((TP - TA - TO + 1))        # 23 -- Tp-Ta-To+1, the lerobot/paper convention

FEATS='{"observation.state": {"type": "STATE", "shape": [6]},
        "observation.images.wrist": {"type": "VISUAL", "shape": [3,480,640]},
        "observation.images.front": {"type": "VISUAL", "shape": [3,480,640]},
        "observation.images.top":   {"type": "VISUAL", "shape": [3,480,640]}}'

# --- the split, identical to the ACT and CNN A/Bs -----------------------------
# eval_split=0.209 is NOT free to round: ceil(143*s) must be 30, and 0.2098 /
# 0.21 both give 31.
EVAL_SPLIT=0.209
EPISODES=$(python -c "
held  = [*range(0,5), *range(20,25), *range(45,50), *range(65,70), *range(90,95), *range(110,115)]
train = [e for e in range(143) if e not in set(held)]
print(train + held)
" | tr -d ' ')

python - "$EPISODES" "$EVAL_SPLIT" <<'PY' || exit 1
import ast, math, sys
eps = ast.literal_eval(sys.argv[1]); split = float(sys.argv[2])
held = set([*range(0,5), *range(20,25), *range(45,50), *range(65,70), *range(90,95), *range(110,115)])
recovery = set(range(120, 143))
assert len(eps) == 143 and len(set(eps)) == 143
n_eval = math.ceil(len(eps) * split)
assert n_eval == 30, f"eval_split {split} holds out {n_eval}, expected 30"
assert set(eps[-n_eval:]) == held, "last 30 must be exactly the clean holdout"
assert recovery <= set(eps[:-n_eval]), "ALL 23 recovery episodes must be in training"
print(f"split OK: {len(eps)-n_eval} train (incl. all {len(recovery)} recovery) + {n_eval} clean held out")
PY

export PYTHONNOUSERSITE=1         # a stale ~/.local torch shadows the conda env
export http_proxy=http://10.99.0.130:3128
export https_proxy=http://10.99.0.130:3128
export HF_HOME=$PROJECT/hf
export HF_LEROBOT_HOME=$PROJECT/lerobot-data
export WANDB_DIR=$PROJECT/wandb
export WANDB_CACHE_DIR=$PROJECT/wandb/cache
# 🔑 the transformer backbone lives in THIS repo, not in lerobot (grep
# lerobot/policies/diffusion for "transformer" -> zero hits). pyproject uses a
# src/ layout, so src must be on the path.
export PYTHONPATH=$PROJECT/repo/src${PYTHONPATH:+:$PYTHONPATH}
mkdir -p "$PROJECT/logs" "$PROJECT/wandb/cache"

module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PROJECT/envs/lerobot"
source "$PROJECT/repo/configs/hpc/_preflight.sh"

# --- fail fast if the port is not synced --------------------------------------
python - <<'PY' || { echo "❌ the transformer port is not importable on this node."; \
                     echo "   Fix: cd $PROJECT/repo && git pull"; exit 1; }
import phi.policies.dp_transformer as M
from lerobot.policies.factory import get_policy_class
cls = get_policy_class("diffusion_transformer")
assert cls.__name__ == "DiffusionTransformerPolicy", cls
assert get_policy_class("diffusion").__name__ == "DiffusionPolicy"   # not clobbered
assert get_policy_class("act").__name__ == "ACTPolicy"
print(f"  port OK: {cls.__name__}; 'diffusion' and 'act' still resolve normally")
PY

if grep -q "api.wandb.ai" "$HOME/.netrc" 2>/dev/null || [ -n "${WANDB_API_KEY:-}" ]; then
  WANDB_MODE_ARG=online
else
  WANDB_MODE_ARG=offline
  echo "⚠️  no W&B credential found; logging OFFLINE to $WANDB_DIR (sync later)"
fi

echo "=============================================================="
echo " DP-TRANSFORMER  task $SLURM_ARRAY_TASK_ID  TAG=$TAG"
echo "   weight_decay = $WDECAY     attn_dropout = $DROP_ATTN"
echo "   n_emb=$EMB n_layer=$N_LAYER n_head=$N_HEAD n_cond_layers=$N_COND_LAYERS causal_attn=true"
echo "   To/Ta/Tp = $TO/$TA/$TP   drop_n_last_frames=$DROP"
echo "   batch=$BATCH  workers=$NUM_WORKERS  steps=$STEPS"
echo "   save_freq=$SAVE_FREQ  eval every $EVAL_STEPS  (15 checkpoints, 15 eval points)"
echo "   DDIM 100 train / 16 eval   resize 240x320 -> crop 216x288"
echo "   vision: 3x ResNet18 from scratch + spatial softmax + GroupNorm"
echo "   wandb project : phi_dp   (same as the CNN runs, so curves overlay)"
echo " node $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=============================================================="

srun python -m phi.train.train_dp_transformer \
  --dataset.repo_id=$REPO_ID \
  --dataset.episodes="$EPISODES" \
  --dataset.eval_split=$EVAL_SPLIT \
  --dataset.video_backend=pyav \
  --eval_steps=$EVAL_STEPS \
  --policy.type=diffusion_transformer \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.n_obs_steps=$TO \
  --policy.horizon=$TP \
  --policy.n_action_steps=$TA \
  --policy.drop_n_last_frames=$DROP \
  --policy.n_emb=$EMB \
  --policy.n_layer=$N_LAYER \
  --policy.n_head=$N_HEAD \
  --policy.n_cond_layers=$N_COND_LAYERS \
  --policy.causal_attn=true \
  --policy.p_drop_emb=$DROP_EMB \
  --policy.p_drop_attn=$DROP_ATTN \
  --policy.optimizer_weight_decay=$WDECAY \
  --policy.resize_shape=[240,320] \
  --policy.crop_ratio=0.9 \
  --policy.use_group_norm=true \
  --policy.pretrained_backbone_weights=null \
  --policy.noise_scheduler_type=DDIM \
  --policy.num_train_timesteps=100 \
  --policy.num_inference_steps=16 \
  --policy.input_features="$FEATS" \
  --batch_size=$BATCH \
  --steps=$STEPS \
  --save_freq=$SAVE_FREQ \
  --log_freq=$LOG_FREQ \
  --num_workers=$NUM_WORKERS \
  --seed=1000 \
  --wandb.enable=true \
  --wandb.mode=$WANDB_MODE_ARG \
  --wandb.project=phi_dp \
  --output_dir=$PROJECT/results/dpt_${TAG}_emb${EMB} \
  --job_name=dpt_${TAG}_emb${EMB}

RC=$?
echo
echo "=============================================================="
echo " DONE  task $SLURM_ARRAY_TASK_ID  TAG=$TAG  rc=$RC"
echo "=============================================================="
echo " checkpoints: $PROJECT/results/dpt_${TAG}_emb${EMB}/checkpoints/"
ls -1 "$PROJECT/results/dpt_${TAG}_emb${EMB}/checkpoints/" 2>/dev/null | tail -20
echo
echo " READ THE eval_loss CURVE, NOT JUST THE LAST NUMBER."
echo " And remember what the CNN rollouts proved on 2026-08-12: a checkpoint whose"
echo " held-out noise-MSE had risen 4x still matched a 100k-step ACT model on the"
echo " arm. eval_loss did NOT predict task success. Treat it as a training-health"
echo " signal for this A/B, not as a reason to pick a checkpoint."
echo
echo " Next: bucket the loss by diffusion timestep k before drawing conclusions."
echo " A single averaged eval_loss blends the near-free k>90 regime (signal is"
echo " 0.0005 of the input, so predicting eps is nearly copying it) with the"
echo " k<5 regime, where recovering eps means dividing by 0.0251 and any residual"
echo " uncertainty about the action is amplified ~40x. Only the second one sets"
echo " whether the gripper closes."

#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Φ — DP-Transformer with causal_attn=FALSE. One arm. Pre-registered.
#
#   Submit:  sbatch configs/hpc/train_dpt_bidirectional.sbatch
#
# WHY THIS RUN EXISTS
# -------------------
# On 2026-08-14 the DP-T `unreg` checkpoint was reported on the arm as visibly
# JITTERY compared with DP-CNN. Measured on a real frame from episode 0, same
# observation, 8 noise draws, 48-step chunks:
#
#     model      mean |step-to-step|   mean |accel|   spread across draws
#     DP-CNN                 0.00384        0.00478              0.05082
#     DP-T                   0.01343        0.01934              0.05750
#
#     DP-T is 3.5x rougher step-to-step and 4.0x in acceleration.
#     Spread across re-plans is nearly IDENTICAL (1.1x), so this is NOT
#     chunk-boundary jumping -- the roughness is INSIDE each chunk.
#
# It is also not latency. Measured per re-plan on MPS:
#     DP-CNN 238.6 ms   DP-T 107.6 ms    -- DP-T is 2.2x FASTER.
# And the two configs are otherwise identical: horizon 48, To 2, Ta 24,
# 16 DDIM steps, resize 240x320, crop 216x288, crop_is_random.
#
# THE HYPOTHESIS
# --------------
# The two backbones DISAGREE about whether the action horizon is bidirectional,
# and it is the only architectural difference left standing.
#
#   DP-CNN   DiffusionConv1dBlock uses nn.Conv1d(..., padding=kernel_size // 2)
#            -> SYMMETRIC. Verified with an impulse at t=3, kernel [1,10,100]:
#               output non-zero at t = [2,3,4]. Every timestep is averaged with
#               its neighbours on BOTH sides, 12 residual blocks deep.
#               Smoothness is not learned, it is structurally unavoidable.
#
#   DP-T     causal_attn=True -> action step t attends only to t' <= t.
#            Nothing forces neighbouring timesteps to resemble each other, and
#            there is not even a backward smoothing path.
#
# PREDICTION (write it down before looking): with causal_attn=false the
# step-to-step roughness of sampled chunks should fall substantially toward
# DP-CNN's 0.00384. If it does not move, the hypothesis is wrong and the jitter
# comes from somewhere else -- most likely `unreg`'s absent regularisation
# (wd 1e-6, dropout 0.0), which is the next thing to test.
#
# ⚠️ ONE FLAG, TWO MASKS -- flagged, not hidden
# ---------------------------------------------
# causal_attn does NOT control self-attention alone. Measured:
#     causal_attn=True   tgt_mask (48,48) causal   memory_mask (48,3) t>=s-1
#     causal_attn=False  tgt_mask None              memory_mask None
# So this run also removes the cross-attention restriction. That restriction
# only ever applied to ACTION TOKEN 0, and generate_actions() slices
# actions[:, start:] with start = n_obs_steps-1 = 1, so token 0 is DISCARDED
# before execution. The extra change therefore touches one throwaway token's
# training signal and nothing that reaches the robot. Close enough to a clean
# self-attention ablation to run as one; noted so the write-up stays honest.
#
# EVERYTHING ELSE IS IDENTICAL TO THE `unreg` ARM
# -----------------------------------------------
# wd 1e-6, p_drop_attn 0.0, seed 1000, same episode list, same eval split,
# same To/Ta/Tp, same scheduler. `unreg` is the control and already exists at
# /scratch/$USER/phi/results/dpt_unreg_emb256 -- no need to retrain it.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=phi-dpt-bidir
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=/scratch/gupta.yashv/phi/logs/dptbi-%j.out
#SBATCH --error=/scratch/gupta.yashv/phi/logs/dptbi-%j.err

set -uo pipefail
PROJECT=/scratch/$USER/phi
REPO_ID=BrutalCaesar/phi_so101_cubes_cylinder_recovery_v1

STEPS=${STEPS:-30000}
SAVE_FREQ=${SAVE_FREQ:-2000}
EVAL_STEPS=${EVAL_STEPS:-2000}
LOG_FREQ=200
BATCH=${BATCH:-64}
NUM_WORKERS=${NUM_WORKERS:-32}

# matched to the `unreg` arm, which is the control
WDECAY=1e-6; DROP_ATTN=0.0; TAG=bidir
EMB=${EMB:-256}
N_LAYER=8; N_HEAD=4; N_COND_LAYERS=0; DROP_EMB=0.0
TO=2; TA=24; TP=48
DROP=$((TP - TA - TO + 1))

FEATS='{"observation.state": {"type": "STATE", "shape": [6]},
        "observation.images.wrist": {"type": "VISUAL", "shape": [3,480,640]},
        "observation.images.front": {"type": "VISUAL", "shape": [3,480,640]},
        "observation.images.top":   {"type": "VISUAL", "shape": [3,480,640]}}'

EVAL_SPLIT=0.209
EPISODES=$(python -c "
held  = [*range(0,5), *range(20,25), *range(45,50), *range(65,70), *range(90,95), *range(110,115)]
train = [e for e in range(143) if e not in set(held)]
print(train + held)
" | tr -d ' ')

export PYTHONNOUSERSITE=1
export http_proxy=http://10.99.0.130:3128
export https_proxy=http://10.99.0.130:3128
export HF_HOME=$PROJECT/hf
export HF_LEROBOT_HOME=$PROJECT/lerobot-data
export WANDB_DIR=$PROJECT/wandb
export WANDB_CACHE_DIR=$PROJECT/wandb/cache
export PYTHONPATH=$PROJECT/repo/src${PYTHONPATH:+:$PYTHONPATH}
mkdir -p "$PROJECT/logs"

module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PROJECT/envs/lerobot"
source "$PROJECT/repo/configs/hpc/_preflight.sh"
WANDB_MODE_ARG=${WANDB_MODE_ARG:-online}

# Fail fast if the port is missing. $PROJECT/repo is an rsync SNAPSHOT, not a
# git checkout -- see docs/hpc/explorer.md. Fix by rsyncing from the Mac.
python - <<'PY' || { echo "❌ transformer port not importable. rsync from the Mac."; exit 1; }
import phi.policies.dp_transformer  # noqa: F401
from lerobot.policies.factory import get_policy_class
assert get_policy_class("diffusion_transformer").__name__ == "DiffusionTransformerPolicy"
print("  port OK")
PY

echo "=============================================================="
echo " DP-T BIDIRECTIONAL   causal_attn=FALSE"
echo "   control = dpt_unreg_emb256 (already trained, causal_attn=true)"
echo "   identical otherwise: wd $WDECAY, p_drop_attn $DROP_ATTN, seed 1000"
echo "   To/Ta/Tp = $TO/$TA/$TP, DDIM 100 train / 16 eval"
echo "   PREDICTION: chunk roughness falls toward DP-CNN's 0.00384"
echo "               (unreg measured 0.01343, DP-CNN 0.00384)"
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
  --policy.causal_attn=false \
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
echo " DONE  rc=$RC"
echo "=============================================================="
grep -oE "step [0-9]+: eval_loss=[0-9.]+" "$PROJECT/logs/dptbi-${SLURM_JOB_ID}.err" 2>/dev/null | tail -20
echo
echo " compare against the control:  unreg best 0.0287 @ 8k"
echo " but eval_loss is NOT the question here -- ROUGHNESS is."
echo " Next: sample chunks from both and measure mean |step-to-step|."

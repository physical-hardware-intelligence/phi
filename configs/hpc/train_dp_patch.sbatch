#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Φ — DP-PATCH: dense frozen DINOv2 patch tokens as cross-attention memory.
#     TWO ARMS, one array job:  causal_attn ON and OFF.
#
#   sbatch configs/hpc/train_dp_patch.sbatch          # both arms
#   sbatch --array=0 configs/hpc/train_dp_patch.sbatch  # bidirectional only
#
# 🚨 PREREQUISITE: DINOv2 weights must already be in $PROJECT/torchhub.
#    Run configs/hpc/_prefetch_dino.sh FIRST (as sbatch -- the login node kills it).
#
# WHAT THIS COMPLETES -- a fully crossed 2x2, encoder x attention mask:
#
#                        | causal_attn=True   | causal_attn=False
#   ---------------------+--------------------+-------------------
#   ResNet18+SpatialSMax | unreg   0.0361 ✓   | bidir   0.0135 ✓    (both done)
#   DINOv2 dense patches | arm 1   ?          | arm 0   ?
#
#   The ResNet row is already paid for. `unreg` overfits (0.0287@8k -> 0.0361@30k);
#   `bidir` does not (flat 0.0132-0.0135 across 20k-30k). Everything below is
#   matched to that row: wd 1e-6, drop 0.0, seed 1000, same episodes, same eval
#   split, same To/Ta/Tp, same DDIM schedule. ONLY the encoder differs.
#
# PREDICTIONS, WRITTEN DOWN BEFORE LOOKING:
#   1. arm 0 (bidir) beats arm 1 (causal), same as the ResNet row. Causal masking
#      has no inference-time justification here -- diffusion denoises all 48
#      positions simultaneously, nothing is decoded left to right.
#   2. NEITHER patch arm overfits. 9.5M trainable vs 42.6M, and `unreg` overfit.
#      If DP-Patch also stays flat, that is the parameter count, NOT the patches --
#      do not retro-fit it as a win for dense features.
#   3. Roughness is NOT expected to improve. Dense patches change what the model
#      sees, not how the loss prices temporal frequency. That problem is still open.
#
# ARMS 2-3 ADDED AFTER READING THE PAPER (arXiv 2607.18236), which the first
# submission was built without. Two things it changed:
#   a) Their ResNet baselines are FROZEN too (Table 3: DP(ResNet-18) 29.35M total,
#      9.09M trainable; S3.5 "we freeze the image encoder and exclusively train the
#      policy"). OUR control trains ResNet-18 from scratch on this task, so arms 0-1
#      vs the control confound DENSITY with FROZEN-vs-TRAINED. Arms 2-3 fix that:
#      identical frozen DINOv2, identical everything, only the bottleneck differs.
#      This is the ONLY comparison in the study that isolates density.
#   b) Their Table 12 mask ablation is arms 0-1: "Full" 0.83/1.10 vs "Block-causal"
#      0.83/1.24 on DP (Push-T/Cube) -- they find block-causal >= full, we measured
#      full 2.28x BETTER on eval_loss. Unreconciled. Candidate: their pred_horizon
#      is 6 (Table 11), ours is 48, and causal self-attention penalises long chunks
#      far harder. [UNVERIFIED]
#
# PREDICTION 4 (their Table 4, Push-T: 256 patches 0.69 -> 1 patch 0.48): avg/cls
# should be MUCH worse than dense. If instead avg ~= dense, density does not matter
# on this task and the whole gap to the ResNet control is frozen-vs-trained.
#
# ── PREDICTION 4 WAS FALSIFIED, IN THE OPPOSITE DIRECTION ────────────────────
#   min eval_loss @30k:  avg 0.0117 · cls 0.0119 · ResNet-bidir 0.0132 ·
#                        dense 0.0163 · ResNet-causal 0.0287 · dense-causal 0.0371
#   POOLED beats DENSE by 28%, ahead at every eval from step 4000. avg and cls
#   agree within 1.7%, so it is the TOKEN COUNT, not the choice of summary.
#
# ARM 4 tests the leading explanation for that inversion. Paper S2.2 says "patches
# maintain full bidirectional attention intra-frame" -- but NEITHER their diffusion
# code nor ours does that: n_cond_layers=0 makes the conditioning encoder a
# per-token MLP, so 1800 patch tokens never attend each other anywhere. Measured:
# perturb one memory token and 1/1803 outputs change at ncl=0, 1803/1803 at ncl=2.
# Hypothesis: dense loses BECAUSE nothing fuses it -- 1800 unfused tokens are worse
# than one averaged token, since the decoder got 601x the input and zero extra
# capacity to combine it, plus 461,568 position rows to learn from 65k frames.
#
# ARM 4 IS UNMASKED (full bidirectional over all 1803 memory tokens), NOT the
# paper's block-causal. Two reasons: (a) bidirectional beat causal in every arm
# measured here, and inter-frame causality is vacuous on the memory anyway since
# both observation frames are already in hand at inference; (b) flash attention
# does not support arbitrary additive masks, so the masked variant falls back to a
# slower kernel and may materialise the 1803^2 matrix (3.33 GB/layer at batch 64) --
# the same SDPA effect that cost arm 1 twenty percent of its throughput.
#
# PREDICTION 5: if arm 4 beats avg (0.0117), dense patches need a fusion stage and
# the paper's S2.2 is right about the architecture even though its code omits it.
# If arm 4 only reaches dense (0.0163), fusion is not the missing piece. If it lands
# between, fusion helps but does not close the gap.
# COST NOTE: whatever it scores, ncl=2 more than DOUBLES inference (the cond encoder
# runs INSIDE the 16-step denoising loop: 177 -> 374 GFLOP/chunk) and forfeits the
# memory-caching optimisation that ncl=0 permits. A win here is not a free win.
#
# 🚨 crop_ratio is 0.875, NOT the control's 0.9. Not a preference: DINOv2's
#    PatchEmbed asserts H%14==0. 240x320*0.875 = 210x280 = exactly 15x20 = 300
#    patches, zero remainder. 0.9 gives 216x288 and CRASHES.
# 🚨 VISUAL normalization is IDENTITY, NOT the control's MEAN_STD. DINOv2 asserts
#    pixels in [0,1] and applies ImageNet stats itself.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=phi-dp-patch
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=07:00:00
#SBATCH --array=0-1
#SBATCH --output=/scratch/gupta.yashv/phi/logs/dpp-%A_%a.out
#SBATCH --error=/scratch/gupta.yashv/phi/logs/dpp-%A_%a.err

set -uo pipefail
PROJECT=/scratch/$USER/phi
REPO_ID=BrutalCaesar/phi_so101_cubes_cylinder_recovery_v1

case $SLURM_ARRAY_TASK_ID in
  0) CAUSAL=false; POOL=none; NCL=0; TAG=bidir   ;;  # dense, matches the winning ResNet arm
  1) CAUSAL=true;  POOL=none; NCL=0; TAG=causal  ;;  # dense, matches patch_policy's hardcoded default
  2) CAUSAL=false; POOL=avg;  NCL=0; TAG=avgpool ;;  # their dino_patch_avg_pool
  3) CAUSAL=false; POOL=cls;  NCL=0; TAG=cls     ;;  # their dino_cls
  4) CAUSAL=false; POOL=none; NCL=2; TAG=fusion  ;;  # dense + a FUSION STAGE (see below)
  *) echo "bad array index $SLURM_ARRAY_TASK_ID (expected 0-4)"; exit 1 ;;
esac

STEPS=${STEPS:-30000}
SAVE_FREQ=${SAVE_FREQ:-2000}
EVAL_STEPS=${EVAL_STEPS:-2000}
LOG_FREQ=200
BATCH=${BATCH:-64}
NUM_WORKERS=${NUM_WORKERS:-32}

WDECAY=1e-6; DROP_ATTN=0.0
EMB=${EMB:-256}
N_LAYER=8; N_HEAD=4; N_COND_LAYERS=$NCL; DROP_EMB=0.0
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
export TORCH_HOME=$PROJECT/torchhub          # where _prefetch_dino.sh put DINOv2
export WANDB_DIR=$PROJECT/wandb
export WANDB_CACHE_DIR=$PROJECT/wandb/cache
export PYTHONPATH=$PROJECT/repo/src${PYTHONPATH:+:$PYTHONPATH}
mkdir -p "$PROJECT/logs"

module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PROJECT/envs/lerobot"
source "$PROJECT/repo/configs/hpc/_preflight.sh"
WANDB_MODE_ARG=${WANDB_MODE_ARG:-online}

# Fail fast. $PROJECT/repo is an rsync SNAPSHOT, not a git checkout.
#
# 🚨 THIS CHECK MUST IMPORT THE TRAIN MODULE, NOT dp_patch DIRECTLY.
#    Registration happens at import time and draccus builds `--policy.type`'s choice
#    list from what is registered IN THE TRAINING PROCESS. A preflight that imports
#    dp_patch itself passes even when the train module forgot to -- which is exactly
#    what happened on 9204468: preflight green, then `invalid choice: 'diffusion_patch'`
#    two minutes later. Import what the real run imports, and assert on the CHOICE LIST.
python - <<'PY' || { echo "❌ preflight failed -- see error above. rsync from the Mac?"; exit 1; }
import os, torch
import phi.train.train_dp_transformer  # noqa: F401  <- the REAL entrypoint
from lerobot.configs.policies import PreTrainedConfig

choices = set(PreTrainedConfig.get_known_choices())
for want in ("diffusion_patch", "diffusion_transformer"):
    assert want in choices, (
        f"'{want}' is NOT in --policy.type's choices after importing the train module. "
        f"Add `import phi.policies.<mod>` to phi/train/train_dp_transformer.py. Got: {sorted(choices)}"
    )
torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
m = torch.hub.load("facebookresearch/dinov2:b48308a", "dinov2_vits14", verbose=False)
n = sum(p.numel() for p in m.parameters())
assert n == 22_056_576, f"DINOv2 param count {n:,} != 22,056,576"
print(f"  port OK | policy.type choices include diffusion_patch | DINOv2 {n:,} params")
print(f"  TORCH_HOME={os.environ.get('TORCH_HOME')}")
PY

echo "=============================================================="
echo " DP-PATCH  task $SLURM_ARRAY_TASK_ID  TAG=$TAG  causal_attn=$CAUSAL"
echo "   encoder : DINOv2 ViT-S/14 FROZEN, 210x280 -> 15x20 = 300 patches/cam"
echo "   pooling : patch_pool=$POOL   (none -> T_cond 1803 | avg,cls -> T_cond 9)"
echo "   params  : none 9,532,038 | avg/cls 9,072,774 trainable, + 22,056,576 frozen"
echo "   control : dpt_${TAG}_emb256 (ResNet row, already trained)"
echo "   To/Ta/Tp = $TO/$TA/$TP, DDIM 100 train / 16 eval, wd $WDECAY, seed 1000"
echo " node $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=============================================================="

srun python -m phi.train.train_dp_transformer \
  --dataset.repo_id=$REPO_ID \
  --dataset.episodes="$EPISODES" \
  --dataset.eval_split=$EVAL_SPLIT \
  --dataset.video_backend=pyav \
  --eval_steps=$EVAL_STEPS \
  --policy.type=diffusion_patch \
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
  --policy.causal_attn=$CAUSAL \
  --policy.patch_pool=$POOL \
  --policy.p_drop_emb=$DROP_EMB \
  --policy.p_drop_attn=$DROP_ATTN \
  --policy.optimizer_weight_decay=$WDECAY \
  --policy.resize_shape=[240,320] \
  --policy.crop_ratio=0.875 \
  --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"MIN_MAX","ACTION":"MIN_MAX"}' \
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
  --output_dir=$PROJECT/results/dpp_${TAG}_emb${EMB} \
  --job_name=dpp_${TAG}_emb${EMB}

RC=$?
echo
echo "=============================================================="
echo " DONE  task $SLURM_ARRAY_TASK_ID  TAG=$TAG  rc=$RC"
echo "=============================================================="
grep -oE "step [0-9]+: eval_loss=[0-9.]+" "$PROJECT/logs/dpp-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err" 2>/dev/null | tail -20
echo
echo " ResNet row for comparison:   unreg (causal) 0.0361 @30k   bidir 0.0135 @30k"
echo " Check prediction 2 FIRST: did eval_loss RISE after its minimum?"
echo "   unreg did (overfit). If neither patch arm does, that is 9.5M vs 42.6M"
echo "   trainable params, not evidence that dense patches help."
echo " Then roughness -- and expect NO improvement there (prediction 3)."

# 🚨 Propagate srun's status. Without this the script exits on the last `echo` and
# sacct reports COMPLETED 0:0 for a run that died in 2 minutes -- which is how
# 9204468 looked green while both arms had crashed.
exit $RC

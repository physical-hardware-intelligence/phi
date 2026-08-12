#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# Φ — Diffusion Policy on phi_so101_cubes_cylinder_recovery_v1.
#     A/B on the vision encoder: the paper's recipe vs lerobot's default.
#
#   Dataset : BrutalCaesar/phi_so101_cubes_cylinder_recovery_v1
#             143 eps / 81,943 frames = 120 clean + 23 failure-then-recovery
#   Paper   : Chi et al., Diffusion Policy (arXiv 2303.04137), Table 7 real-world rows
#   W&B     : project phi_dp   (NOT phi-act -- that project houses the ACT models)
#
#   SIZED FROM CALIBRATION (jobs 9088440, 9088457), measured 2026-08-12:
#
#     batch  workers   compute   loader    total   100k steps   fits 8h wall
#        32       16    156.4     17.0     173.4      4.82 h    yes, big margin
#        64       16    145.7    156.4     302.1      8.39 h    NO
#        64       32    139.2    106.4     245.7      6.82 h    yes, 1.18 h margin  <-- THIS
#
#   Chose batch 64 + 32 workers: it is the paper's batch size (section A.4, "64 for
#   all image-based experiments") and sees 2x the samples in 100k steps, which
#   matters most for task 0, whose encoder trains from scratch.
#
#   Two things the calibration showed that are worth not forgetting:
#     * COMPUTE PER STEP IS FLAT in batch size -- 156 ms at 32, 139 ms at 64. The
#       H200 is nowhere near saturated (12% memory at batch 64), so a bigger batch
#       is nearly free in GPU time. The cost is entirely the dataloader.
#     * THE LOADER CEILING IS NOT WORKER COUNT. Doubling 16->32 workers moved
#       throughput only 1,271 -> 1,563 frames/s (1.23x for 2x workers). Probably
#       /scratch read bandwidth or memory contention. Do not expect more workers
#       to keep buying time.
#
#   🚨 MARGIN IS 1.18 h ON AN 8 h WALL (15%). If the node is busier than during
#      calibration, data_s rises and this can time out near ~95k steps. That is
#      survivable, not fatal: save_freq=20000 means the worst case loses the last
#      20k steps, and lerobot supports --resume=true to continue from the last
#      checkpoint. Do not "fix" it by cutting save_freq.
#
#   Prefetch: already done for this dataset by job 9066698.
#   Submit:  sbatch configs/hpc/train_diffusion_recovery.sbatch
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=phi-dp-recovery
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-1
#SBATCH --output=/scratch/gupta.yashv/phi/logs/dprec-%A_%a.out
#SBATCH --error=/scratch/gupta.yashv/phi/logs/dprec-%A_%a.err

set -uo pipefail
PROJECT=/scratch/$USER/phi
REPO_ID=BrutalCaesar/phi_so101_cubes_cylinder_recovery_v1

# ── SET FROM CALIBRATION 9088440 / 9088457 (see header table) ───────────────
BATCH=${BATCH:-64}                  # paper section A.4: 64 for all image-based experiments
NUM_WORKERS=${NUM_WORKERS:-32}   # matches --cpus-per-task=32 above
# ────────────────────────────────────────────────────────────────────────────

STEPS=100000
SAVE_FREQ=20000
EVAL_STEPS=10000

# ── Horizon / action horizon, and why they are NOT the paper's numbers ───────
# Table 7 real-world rows: To=2, Ta=8, Tp=16 -- at 10 Hz, i.e. Ta=8 is 800 ms of
# open loop. We run at 30 fps, so the same 800 ms needs Ta=24 and Tp=48.
#
# This is not a preference, it is a deadline. Measured on the Mac cockpit
# 2026-08-11: one chunk costs 294.4 ms (74.6 encoders + 219.8 UNet at DDIM-16,
# Tp=48). The Ta=8 budget is 267 ms, so the paper's step counts at our control
# rate would MISS the deadline every chunk. At Ta=24 there is 506 ms headroom.
# Full numbers: docs/deployment/README.md section 4.
#
# n_action_steps does NOT enter compute_loss (verified against the source), so Ta
# stays tunable at rollout across 1..48 with no retraining. Only Tp is baked into
# the weights. Hence: train long, sweep Ta on the arm. Minimum viable Ta on the
# Mac is ~10; Ta ~= Tp/2 is the standard split so you execute the better half.
TO=2; TA=24; TP=48
DROP=$((TP - TA - TO + 1))          # 23 -- Tp-Ta-To+1, the lerobot/paper convention
                                    # NB lerobot's default 7 is correct for Tp=16/Ta=8,
                                    # i.e. it matches the PAPER, not lerobot's own
                                    # horizon=64/n_action_steps=32 defaults.

# ── The A/B: the vision encoder. This pairing is FORCED. ─────────────────────
# lerobot raises ValueError if use_group_norm is combined with pretrained weights,
# and it is right to: GroupNorm discards BatchNorm's running stats while the conv
# filters were fitted to be consumed WITH them.
#
# Paper section 3.2: "a standard ResNet-18 (without pretraining)", global average
# pooling -> spatial softmax, BatchNorm -> GroupNorm. Its stated reason for
# GroupNorm is stability, emphasised as important in conjunction with an EMA of
# the weights. lerobot implements NO EMA (grepped: no EMAModel, no use_ema), so
# that justification does not transfer -- and at batch 64 with n_obs_steps=2 each
# encoder sees 128 images, so BatchNorm's statistics are well estimated anyway.
# Hence a real question rather than a foregone conclusion.
#
# Note: the GroupNorm/BatchNorm choice applies ONLY to the image encoder. The
# UNet is hardcoded Conv1d -> GroupNorm -> Mish with no option, which is correct:
# training draws one random diffusion timestep PER SAMPLE, so a batch spans many
# noise levels and normalising across it would blend them.
case $SLURM_ARRAY_TASK_ID in
  0) GN=true;  PRETRAINED=null;                                TAG=paper   ;;
  1) GN=false; PRETRAINED=ResNet18_Weights.IMAGENET1K_V1;      TAG=lerobot ;;
  *) echo "bad array index $SLURM_ARRAY_TASK_ID (expected 0-1)"; exit 1 ;;
esac

RUN=dp_recovery_3cam_${TAG}_tp${TP}_b${BATCH}
OUT=$PROJECT/results/$RUN

FEATS='{"observation.state": {"type": "STATE", "shape": [6]},
        "observation.images.wrist": {"type": "VISUAL", "shape": [3,480,640]},
        "observation.images.front": {"type": "VISUAL", "shape": [3,480,640]},
        "observation.images.top":   {"type": "VISUAL", "shape": [3,480,640]}}'

# ── The split: identical to the ACT runs so the comparison holds ─────────────
# All 23 recovery episodes in TRAIN; the 30-episode holdout is clean-only and is
# the same one the ACT A/B used. eval_split=0.209 is NOT free to round:
# ceil(143*s) must be 30, and 0.2098 / 0.21 both give 31.
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

export PYTHONNOUSERSITE=1           # a stale ~/.local torch shadows the conda env
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
repo = sys.argv[1]
pat = os.path.join(os.environ["HF_LEROBOT_HOME"], "hub",
                   "datasets--" + repo.replace("/", "--"), "snapshots", "*", "meta", "info.json")
hits = sorted(glob.glob(pat))
if not hits:
    sys.exit(f"dataset not prefetched (nothing matching {pat})")
m = json.load(open(hits[-1]))
assert m["total_episodes"] == 143, f"expected 143 episodes, found {m['total_episodes']}"
assert m["total_frames"] == 81943, f"expected 81,943 frames, found {m['total_frames']}"
print(f"dataset OK: {m['total_episodes']} episodes, {m['total_frames']} frames")
PY

if grep -q "api.wandb.ai" "$HOME/.netrc" 2>/dev/null || [ -n "${WANDB_API_KEY:-}" ]; then
  WANDB_MODE_ARG=online
else
  WANDB_MODE_ARG=offline
  echo "NOTE: no W&B credential -> logging OFFLINE to $WANDB_DIR (sync later)."
fi

echo "=============================================================="
echo " task $SLURM_ARRAY_TASK_ID  run=$RUN"
echo " encoder       : group_norm=$GN  pretrained=$PRETRAINED   ($TAG recipe)"
echo " To/Ta/Tp      : $TO/$TA/$TP   drop_n_last_frames=$DROP"
echo " open loop     : Ta $TA @30fps = $(python -c "print(f'{$TA/30*1000:.0f}')") ms   (measured chunk cost 294 ms on the Mac)"
echo " down_dims     : (256,512,1024)  ~68.7M UNet   (paper #D-Params 67M)"
echo " images        : resize 240x320, crop_ratio 0.9 -> 216x288"
echo " sched         : DDIM, train T=100, infer T=16   (Table 7 D-Iters 100/16)"
echo " batch/steps   : $BATCH / $STEPS  (save $SAVE_FREQ, eval $EVAL_STEPS)"
echo " images/step   : $((BATCH * TO * 3))"
echo " lr/wd/warmup  : 1e-4 / 1e-6 / cosine 500   (Table 7 + section A.4)"
echo " wandb project : phi_dp"
echo " node/GPU      : $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=============================================================="

srun lerobot-train \
  --dataset.repo_id=$REPO_ID \
  --dataset.episodes="$EPISODES" \
  --dataset.eval_split=$EVAL_SPLIT \
  --dataset.video_backend=pyav \
  --eval_steps=$EVAL_STEPS \
  --policy.type=diffusion \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.n_obs_steps=$TO \
  --policy.horizon=$TP \
  --policy.n_action_steps=$TA \
  --policy.drop_n_last_frames=$DROP \
  --policy.down_dims=[256,512,1024] \
  --policy.resize_shape=[240,320] \
  --policy.crop_ratio=0.9 \
  --policy.crop_is_random=true \
  --policy.use_group_norm=$GN \
  --policy.pretrained_backbone_weights=$PRETRAINED \
  --policy.noise_scheduler_type=DDIM \
  --policy.num_train_timesteps=100 \
  --policy.num_inference_steps=16 \
  --policy.input_features="$FEATS" \
  --batch_size=$BATCH \
  --steps=$STEPS \
  --save_freq=$SAVE_FREQ \
  --num_workers=$NUM_WORKERS \
  --seed=1000 \
  --output_dir="$OUT" \
  --job_name="$RUN" \
  --wandb.enable=true \
  --wandb.mode=$WANDB_MODE_ARG \
  --wandb.project=phi_dp \
  --wandb.notes="DP on cubes+cylinder+23 recovery eps, 3 cams. Encoder A/B: $TAG (group_norm=$GN, pretrained=$PRETRAINED). To/Ta/Tp=$TO/$TA/$TP for 800ms open loop at 30fps, matching the paper's DURATION not its step count. down_dims=(256,512,1024) per Table 7's 67M real-world size. crop 216x288. DDIM 100/16. batch $BATCH. Same 30 clean holdout as the ACT A/B."

echo "task $SLURM_ARRAY_TASK_ID finished rc=$? at $(date)"

# ── Reading the result ───────────────────────────────────────────────────────
# 🚨 DP's eval_loss is MSE on predicted noise. ACT's is L1 + KL on actions.
#    Different quantities, different scales. They are NOT comparable, and unlike
#    ACT there is no conversion to degrees. DP vs ACT is a ROLLOUT question only:
#    python -m phi.utils.eval_rollouts --model <hf-id> --dataset recovery
#
# Confounds to record with any DP-vs-ACT number, not discover later:
#   - DP sees 216x288 (7x9 ResNet feature grid); ACT sees 480x640 (15x20)
#   - DP re-plans every 800 ms at Ta=24; our ACT runs commit for 1670 ms
#     (free control: re-run ACT with --n-action-steps 24, no retraining)
#   - DP encoder is from scratch (task 0); ACT's is ImageNet-pretrained
#   This is method-vs-method as published, not a controlled decoder comparison.

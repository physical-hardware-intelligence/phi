#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Φ — CALIBRATION ONLY. Measures how fast Diffusion Policy actually trains on
# this cluster, so the real 8-hour run is sized from a number instead of my
# arithmetic. Trains nothing worth keeping: 250 steps, no checkpoint, no W&B.
#
#   Submit:  sbatch configs/hpc/calib_diffusion.sbatch
#   Read:    /scratch/$USER/phi/logs/calibdp-<jobid>_<n>.out   (summary at the end)
#
# WHY THIS EXISTS
# ---------------
# Two forces pull in opposite directions and I cannot predict the winner:
#   vision cost UP   384 images/step at batch 64 (64 x n_obs 2 x 3 cams) vs
#                    ACT's 24 -- but at 216x288, which is 0.2025x the pixels
#   UNet cost DOWN   68.7M params vs ACT-comparable 259M, on a 48-long time axis
# ACT measured 14.3 steps/s on an H200. A pixel-ratio extrapolation says DP lands
# near 3.2x ACT's vision cost, i.e. ~6.5h for 100k steps -- inside the 8h wall,
# but not by enough to bet 8 hours of GPU on.
#
# 🔑 IT SPLITS updt_s FROM data_s, WHICH IS THE POINT. The dataloader decodes AV1
#    video and resizes 384 frames per step. If data_s dominates, the fix is
#    num_workers / prefetch, NOT a smaller batch, and a synthetic-tensor benchmark
#    would have hidden that completely.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=phi-calib-dp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --array=0-1
#SBATCH --output=/scratch/gupta.yashv/phi/logs/calibdp-%A_%a.out
#SBATCH --error=/scratch/gupta.yashv/phi/logs/calibdp-%A_%a.err

set -uo pipefail
PROJECT=/scratch/$USER/phi
REPO_ID=BrutalCaesar/phi_so101_cubes_cylinder_recovery_v1

STEPS=250            # enough for the averages to settle, cheap enough to be free
LOG_FREQ=25
TARGET_STEPS=100000  # what the real run will do, for the projection

# Dataloader workers. Override to test the loader ceiling:
#   sbatch --cpus-per-task=32 --array=0 --export=ALL,NUM_WORKERS=32 configs/hpc/calib_diffusion.sbatch
# Measured 2026-08-12 at NUM_WORKERS=16: the loader saturates near ~1,200
# decoded frames/s REGARDLESS of batch size (192 x 5.77 = 1,108 at batch 32;
# 384 x 3.31 = 1,271 at batch 64). So batch 64 is not intrinsically slower --
# compute per step was actually LOWER at 64 (145.7 vs 156.4 ms) because the H200
# is nowhere near saturated. The whole penalty is loader throughput.
NUM_WORKERS=${NUM_WORKERS:-16}

case $SLURM_ARRAY_TASK_ID in
  0) BATCH=64; TAG=b64 ;;   # the paper's image-based batch size
  1) BATCH=32; TAG=b32 ;;   # the fallback if 64 will not fit the wall clock
  *) echo "bad array index $SLURM_ARRAY_TASK_ID (expected 0-1)"; exit 1 ;;
esac

# --- the parameters under test (Table 7 real-world rows, adjusted for 30 fps) --
# To/Ta/Tp = 2/24/48 rather than the paper's 2/8/16. The paper ran at 10 Hz, so
# its Ta=8 was 800 ms of open loop; at 30 fps we need Ta=24 for the same 800 ms.
# Copying the step counts would have made us 3x MORE conservative than they were,
# and -- measured on the Mac 2026-08-11 -- undeployable: one chunk costs 294 ms
# (74.6 encoders + 219.8 UNet at DDIM-16), which BLOWS the 267 ms budget Ta=8
# implies. At Ta=24 there is 506 ms of headroom.
# n_action_steps does NOT enter compute_loss (verified), so Ta stays tunable at
# rollout; only Tp is baked into the weights. Hence train long, sweep Ta later.
TO=2; TA=24; TP=48
DROP=$((TP - TA - TO + 1))          # 23, the lerobot/paper convention

FEATS='{"observation.state": {"type": "STATE", "shape": [6]},
        "observation.images.wrist": {"type": "VISUAL", "shape": [3,480,640]},
        "observation.images.front": {"type": "VISUAL", "shape": [3,480,640]},
        "observation.images.top":   {"type": "VISUAL", "shape": [3,480,640]}}'

export PYTHONNOUSERSITE=1           # a stale ~/.local torch shadows the env
export http_proxy=http://10.99.0.130:3128
export https_proxy=http://10.99.0.130:3128
export HF_HOME=$PROJECT/hf
export HF_LEROBOT_HOME=$PROJECT/lerobot-data
mkdir -p "$PROJECT/logs"

module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PROJECT/envs/lerobot"
source "$PROJECT/repo/configs/hpc/_preflight.sh"

echo "=============================================================="
echo " CALIBRATION  task $SLURM_ARRAY_TASK_ID  batch=$BATCH"
echo " To/Ta/Tp = $TO/$TA/$TP   drop_n_last_frames=$DROP"
echo " down_dims=(256,512,1024)  resize=240x320 crop_ratio=0.9 -> 216x288"
echo " group_norm=true  pretrained=None  DDIM 100/16"
echo " images per step: $((BATCH * TO * 3)) at 216x288   num_workers=$NUM_WORKERS"
echo " node $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=============================================================="

# Poll GPU memory so we know whether batch 64 actually fits, not just whether it ran.
MEMLOG=$PROJECT/logs/calibdp-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.mem
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$MEMLOG"; sleep 2; done ) &
MEMPID=$!
trap 'kill $MEMPID 2>/dev/null' EXIT

srun lerobot-train \
  --dataset.repo_id=$REPO_ID \
  --dataset.video_backend=pyav \
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
  --policy.use_group_norm=true \
  --policy.pretrained_backbone_weights=null \
  --policy.noise_scheduler_type=DDIM \
  --policy.num_inference_steps=16 \
  --policy.input_features="$FEATS" \
  --batch_size=$BATCH \
  --steps=$STEPS \
  --log_freq=$LOG_FREQ \
  --num_workers=$NUM_WORKERS \
  --seed=1000 \
  --save_checkpoint=false \
  --wandb.enable=false \
  --output_dir=$PROJECT/results/calib_dp_$TAG \
  --job_name=calib_dp_$TAG

RC=$?
kill $MEMPID 2>/dev/null

echo
echo "=============================================================="
echo " RESULT  batch=$BATCH  rc=$RC"
echo "=============================================================="
ERRLOG=$PROJECT/logs/calibdp-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err
python - "$ERRLOG" "$MEMLOG" "$BATCH" "$TARGET_STEPS" <<'PY'
import re, sys, pathlib
err, mem, batch, target = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
text = pathlib.Path(err).read_text(errors="replace") if pathlib.Path(err).exists() else ""
# lerobot logs e.g.  step:250 smpl:16K ep:.. epch:0.20 loss:0.123 updt_s:0.512 data_s:0.031
upd = [float(x) for x in re.findall(r"updt_s:([0-9.]+)", text)]
dat = [float(x) for x in re.findall(r"data_s:([0-9.]+)", text)]
if not upd:
    print("  Could not find updt_s in the log. Read it by hand:")
    print(f"    grep -oE 'step:[^ ]+.*data_s:[0-9.]+' {err} | tail")
    raise SystemExit(0)
# drop the first log line: it carries compile/warmup and would flatter or spoil the average
u = sum(upd[1:]) / max(len(upd) - 1, 1)
d = sum(dat[1:]) / max(len(dat) - 1, 1)
per = u + d
print(f"  update  (compute) : {u*1000:8.1f} ms/step")
print(f"  data    (loader)  : {d*1000:8.1f} ms/step")
print(f"  total             : {per*1000:8.1f} ms/step  ->  {1/per:6.2f} steps/s")
print(f"  {target:,} steps    : {target*per/3600:8.2f} h")
share = d / per * 100
print(f"\n  dataloader is {share:.0f}% of step time -> "
      + ("🚨 DATALOADER-BOUND: raise num_workers/prefetch_factor before cutting batch size"
         if share > 35 else "compute-bound, the loader is keeping up"))
try:
    vals = [int(x) for x in pathlib.Path(mem).read_text().split() if x.strip().isdigit()]
    if vals:
        print(f"  peak GPU memory   : {max(vals):,} MiB of 143,771 MiB "
              f"({max(vals)/143771*100:.0f}%)")
except FileNotFoundError:
    pass
fits = target * per / 3600 < 8
print(f"\n  VERDICT at batch {batch}: 100k steps {'FITS' if fits else 'DOES NOT FIT'} an 8 h wall"
      + ("" if fits else "  -> use batch 32, or fewer steps, or checkpoint+resume across two jobs"))
PY
echo
echo " Compare the two array tasks before submitting the real run."

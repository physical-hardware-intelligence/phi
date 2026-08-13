#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Φ — CALIBRATION ONLY, for the TRANSFORMER backbone (src/phi/policies).
# Measures how fast DP-T actually trains here so the real run is sized from a
# number instead of arithmetic. Trains nothing worth keeping: 250 steps, no
# checkpoint, no W&B.
#
#   Submit:  sbatch configs/hpc/calib_diffusion_transformer.sbatch
#   Read:    /scratch/$USER/phi/logs/calibdpt-<jobid>_<n>.out   (summary at the end)
#
# WHY THIS EXISTS, AND WHY THE ANSWER IS NOT "7.6x FASTER"
# --------------------------------------------------------
# The transformer backbone is 9,020,934 params against the UNet's 68,665,222 --
# 7.6x smaller. It is tempting to divide the CNN's runtime by 7.6. That is wrong,
# for two measured reasons:
#
#   1. THE VISION ENCODER IS UNCHANGED. 3x ResNet18+spatial-softmax = 33,591,264
#      params chewing 384 images/step at 216x288. It is now the LARGER half of
#      the model (33.6M vs 9.0M) and nothing about it got cheaper.
#   2. THE CNN RUN WAS ALREADY NEAR DATALOADER-BOUND. Calibration 9088457
#      measured batch 64 / 32 workers at 245.7 ms/step = 139.2 compute + 106.4
#      loader. Shrinking the denoiser cuts into the 139.2 only; the 106.4 is
#      untouched, and a smaller network makes the loader a LARGER share.
#
# 🔑 SO THIS SPLITS updt_s FROM data_s, WHICH IS THE POINT. Do not predict the
#    ratio -- on 2026-08-12 I predicted doubling workers would halve data_s and
#    it fell 32% (1,271 -> 1,563 decoded frames/s), because the ceiling is more
#    likely /scratch read bandwidth than worker count.
#
# THE TWO ARMS ARE THE TWO WIDTHS TABLE 8 ACTUALLY USES
# -----------------------------------------------------
#   task 0  n_emb=256  -> 9,020,934 backbone   Table 8 "#D-params 9", all sim rows
#   task 1  n_emb=768  -> 80,540,166 backbone  Table 8 "80", Kitchen + Real Push-T
# Both parameter counts were reproduced exactly on the Mac before this ran, which
# is the evidence the port is wired correctly. Task 1 is the fallback if 256
# underfits -- knowing now whether it fits the wall clock costs nothing extra.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=phi-calib-dpt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --array=0-1
#SBATCH --output=/scratch/gupta.yashv/phi/logs/calibdpt-%A_%a.out
#SBATCH --error=/scratch/gupta.yashv/phi/logs/calibdpt-%A_%a.err

set -uo pipefail
PROJECT=/scratch/$USER/phi
REPO_ID=BrutalCaesar/phi_so101_cubes_cylinder_recovery_v1

STEPS=250            # enough for the averages to settle, cheap enough to be free
LOG_FREQ=25
BATCH=64             # §A.4: "64 for all image-based experiments"
NUM_WORKERS=${NUM_WORKERS:-32}   # the CNN's winning value; override to probe the ceiling

case $SLURM_ARRAY_TASK_ID in
  0) EMB=256; TAG=emb256 ;;   # Table 8 simulation rows
  1) EMB=768; TAG=emb768 ;;   # Table 8 Kitchen + Real Push-T
  *) echo "bad array index $SLURM_ARRAY_TASK_ID (expected 0-1)"; exit 1 ;;
esac

# --- parameters under test (Table 8, adjusted for 30 fps) ---------------------
# To/Ta/Tp = 2/24/48 rather than the paper's 2/8/16: the paper ran at 10 Hz, so
# its Ta=8 was 800 ms of open loop; at 30 fps we need Ta=24 for the same 800 ms.
# Inherited unchanged from the CNN runs so DP-T vs DP-CNN stays comparable.
# NOTE Table 8's sim rows use Tp=10 where Table 7's CNN used 16 -- the paper runs
# SHORTER horizons on the transformer. Keeping 48 is a deliberate deviation for
# comparability; flagged rather than silently inherited.
TO=2; TA=24; TP=48
DROP=$((TP - TA - TO + 1))          # 23, the lerobot/paper convention

# 🚨 weight_decay 1e-3, NOT the CNN's 1e-6. Table 8 vs Table 7, a 1000x gap, and
# Table 8's caption reads "WDecay: weight decay (for transformer only)". Does not
# affect throughput; set here so the calibration runs the real configuration.
WDECAY=1e-3
DROP_ATTN=0.3

FEATS='{"observation.state": {"type": "STATE", "shape": [6]},
        "observation.images.wrist": {"type": "VISUAL", "shape": [3,480,640]},
        "observation.images.front": {"type": "VISUAL", "shape": [3,480,640]},
        "observation.images.top":   {"type": "VISUAL", "shape": [3,480,640]}}'

export PYTHONNOUSERSITE=1           # a stale ~/.local torch shadows the env
export http_proxy=http://10.99.0.130:3128
export https_proxy=http://10.99.0.130:3128
export HF_HOME=$PROJECT/hf
export HF_LEROBOT_HOME=$PROJECT/lerobot-data
# 🔑 the transformer backbone lives in THIS repo, not in lerobot. pyproject uses
# a src/ layout (packages = ["src/phi"]), so src must be on the path.
export PYTHONPATH=$PROJECT/repo/src${PYTHONPATH:+:$PYTHONPATH}
mkdir -p "$PROJECT/logs"

module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PROJECT/envs/lerobot"
source "$PROJECT/repo/configs/hpc/_preflight.sh"

# --- fail fast if the port is not synced -------------------------------------
# Without this, a stale `git pull` on Explorer surfaces as a confusing draccus
# error about an unknown policy type, minutes into the job.
python - <<'PY' || { echo "❌ the transformer port is not importable on this node."; \
                     echo "   Fix: rsync from the Mac -- \$PROJECT/repo is an rsync SNAPSHOT,"; \
                     echo "        not a git checkout (see docs/hpc/explorer.md):"; \
                     echo "        rsync -a --exclude .git/ --exclude __pycache__/ --exclude '._*' \\"; \
                     echo "          ./ explorer:\$PROJECT/repo/"; exit 1; }
import sys
import phi.policies.dp_transformer as M
from lerobot.policies.factory import get_policy_class
cls = get_policy_class("diffusion_transformer")
assert cls.__name__ == "DiffusionTransformerPolicy", cls
assert get_policy_class("diffusion").__name__ == "DiffusionPolicy"   # not clobbered
print(f"  port OK: {cls.__name__}, and 'diffusion' still resolves normally")
PY

echo "=============================================================="
echo " CALIBRATION (transformer)  task $SLURM_ARRAY_TASK_ID  n_emb=$EMB"
echo " To/Ta/Tp = $TO/$TA/$TP   drop_n_last_frames=$DROP   batch=$BATCH"
echo " n_layer=8 n_head=4 n_cond_layers=0 causal_attn=true"
echo " weight_decay=$WDECAY  attn_dropout=$DROP_ATTN  DDIM 100/16"
echo " resize=240x320 crop_ratio=0.9 -> 216x288   group_norm=true  scratch"
echo " images per step: $((BATCH * TO * 3)) at 216x288   num_workers=$NUM_WORKERS"
echo " node $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=============================================================="

MEMLOG=$PROJECT/logs/calibdpt-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.mem
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$MEMLOG"; sleep 2; done ) &
MEMPID=$!
trap 'kill $MEMPID 2>/dev/null' EXIT

srun python -m phi.train.train_dp_transformer \
  --dataset.repo_id=$REPO_ID \
  --dataset.video_backend=pyav \
  --policy.type=diffusion_transformer \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.n_obs_steps=$TO \
  --policy.horizon=$TP \
  --policy.n_action_steps=$TA \
  --policy.drop_n_last_frames=$DROP \
  --policy.n_emb=$EMB \
  --policy.n_layer=8 \
  --policy.n_head=4 \
  --policy.n_cond_layers=0 \
  --policy.causal_attn=true \
  --policy.p_drop_attn=$DROP_ATTN \
  --policy.optimizer_weight_decay=$WDECAY \
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
  --output_dir=$PROJECT/results/calib_dpt_$TAG \
  --job_name=calib_dpt_$TAG

RC=$?
kill $MEMPID 2>/dev/null

echo
echo "=============================================================="
echo " RESULT  n_emb=$EMB  rc=$RC"
echo "=============================================================="
ERRLOG=$PROJECT/logs/calibdpt-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err
python - "$ERRLOG" "$MEMLOG" "$EMB" <<'PY'
import re, sys, pathlib
err, mem, emb = sys.argv[1], sys.argv[2], int(sys.argv[3])
text = pathlib.Path(err).read_text(errors="replace") if pathlib.Path(err).exists() else ""
upd = [float(x) for x in re.findall(r"updt_s:([0-9.]+)", text)]
dat = [float(x) for x in re.findall(r"data_s:([0-9.]+)", text)]
if not upd:
    print("  Could not find updt_s in the log. Read it by hand:")
    print(f"    grep -oE 'step:[^ ]+.*data_s:[0-9.]+' {err} | tail")
    raise SystemExit(0)
# drop the first log line: it carries compile/warmup and would spoil the average
u = sum(upd[1:]) / max(len(upd) - 1, 1)
d = sum(dat[1:]) / max(len(dat) - 1, 1)
per = u + d
print(f"  update  (compute) : {u*1000:8.1f} ms/step")
print(f"  data    (loader)  : {d*1000:8.1f} ms/step")
print(f"  total             : {per*1000:8.1f} ms/step  ->  {1/per:6.2f} steps/s")
print()
for target in (20_000, 30_000, 100_000):
    h = target * per / 3600
    print(f"  {target:>7,} steps    : {h:6.2f} h   {'fits 8h' if h < 8 else '🚨 DOES NOT FIT 8h'}")
share = d / per * 100
print(f"\n  dataloader is {share:.0f}% of step time -> "
      + ("🚨 DATALOADER-BOUND: raise num_workers/prefetch before touching the model"
         if share > 35 else "compute-bound, the loader is keeping up"))
print("\n  REFERENCE, the CNN/UNet at the same batch and workers (job 9088457):")
print("    update 139.2 ms | data 106.4 ms | total 245.7 ms | 43% loader-bound")
print(f"    -> compute change: {u*1000/139.2:5.2f}x   loader change: {d*1000/106.4:5.2f}x")
print("    a 7.6x smaller backbone will NOT give 7.6x; the vision encoder and the")
print("    loader are unchanged. That is what this job is measuring.")
try:
    vals = [int(x) for x in pathlib.Path(mem).read_text().split() if x.strip().isdigit()]
    if vals:
        print(f"\n  peak GPU memory   : {max(vals):,} MiB of 143,771 MiB "
              f"({max(vals)/143771*100:.0f}%)")
except FileNotFoundError:
    pass
PY
echo
echo " Compare the two array tasks before submitting the real run."

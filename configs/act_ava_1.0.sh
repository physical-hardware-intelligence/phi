#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Φ — ACT baseline on Ava (SO-101), wrist-camera only.
#
#   Dataset : Parv-09/Ava_1.0_20260730_172156   (60 eps · 21,522 frames · 30 fps
#             · 1 cam observation.images.wrist_cam 480x640 · state/action dim 6)
#   Task    : pick up the orange and put it in the plate
#   Plan    : experiments/2026-07-30_act-wrist-only-so101.md
#
# Usage:
#   bash configs/act_ava_1.0.sh smoke     # 200 steps, verifies the pipeline
#   bash configs/act_ava_1.0.sh bench     # 300 steps, prints measured it/s
#   bash configs/act_ava_1.0.sh train     # the real run
#
# Everything below is deliberate. Do not "tidy" values without a note in the
# experiment write-up — a run is only reproducible if the config is pinned.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

DATASET="Parv-09/Ava_1.0_20260730_172156"
JOB="act_ava_1.0"
SEED=1000

# Device: mps on the Mac cockpit, cuda on Explorer / HF Jobs / the OMEN box.
DEVICE="${DEVICE:-mps}"
# batch 8 = LeRobot's ACT default. 21,522 frames / 8 = ~2,690 steps per epoch.
BATCH="${BATCH:-8}"
# 50k steps ~= 18.6 epochs. Reduce only with a reason; raise if eval loss is still falling.
STEPS="${STEPS:-50000}"

WANDB_PROJECT="${WANDB_PROJECT:-phi-act}"

common=(
  --dataset.repo_id="$DATASET"
  --policy.type=act
  --policy.device="$DEVICE"
  --seed="$SEED"
  --num_workers=4
  # policy.push_to_hub defaults to TRUE, and validate() then hard-requires a
  # repo_id. Keep it off; upload deliberately once a run is worth keeping.
  --policy.push_to_hub=false
)

case "${1:-train}" in

  smoke)   # Cheap correctness check. Loss should fall. No wandb, no upload.
    lerobot-train "${common[@]}" \
      --steps=200 --batch_size=2 \
      --save_checkpoint=false \
      --output_dir="outputs/train/${JOB}_smoke" \
      --job_name="${JOB}_smoke" \
      --wandb.enable=false
    ;;

  bench)   # Measure real throughput so the Mac-vs-GPU call is evidence-based.
    start=$(date +%s)
    lerobot-train "${common[@]}" \
      --steps=300 --batch_size="$BATCH" \
      --save_checkpoint=false \
      --output_dir="outputs/train/${JOB}_bench" \
      --job_name="${JOB}_bench" \
      --wandb.enable=false
    end=$(date +%s); el=$((end-start))
    ips=$(awk -v s=300 -v e="$el" 'BEGIN{printf "%.2f", s/e}')
    eta=$(awk -v ips="$ips" -v st="$STEPS" 'BEGIN{printf "%.1f", st/ips/3600}')
    echo
    echo "──────── benchmark ────────"
    echo " 300 steps in ${el}s  ->  ${ips} it/s"
    echo " projected ${STEPS} steps: ~${eta} h on ${DEVICE}"
    echo " rule of thumb: under ~1.5 it/s, train on GPU instead."
    echo "───────────────────────────"
    ;;

  train)
    lerobot-train "${common[@]}" \
      --steps="$STEPS" \
      --batch_size="$BATCH" \
      --save_freq=5000 \
      --output_dir="outputs/train/${JOB}" \
      --job_name="${JOB}" \
      --wandb.enable=true \
      --wandb.project="$WANDB_PROJECT" \
      --wandb.notes="ACT baseline, wrist-cam only. 60 eps / 4 bins; one bin is NOT in the wrist FOV at episode start (expected failure mode, measured on purpose)."
    ;;

  *) echo "usage: $0 {smoke|bench|train}"; exit 1 ;;
esac

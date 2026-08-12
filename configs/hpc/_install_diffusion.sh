#!/bin/bash
# One-off: add the Diffusion Policy extra to the Explorer env.
#
#   sbatch --partition=short --cpus-per-task=4 --mem=16G --time=00:40:00 \
#     --job-name=phi-install-diffusion \
#     --output=/scratch/$USER/phi/logs/install-diffusion-%j.out \
#     configs/hpc/_install_diffusion.sh
#
# WHY: lerobot's DiffusionPolicy calls require_package("diffusers") in its
# __init__ and dies immediately without it. ACT never needed it, so the env has
# never had it, and calibration job 9088308 failed in 21 s on exactly this:
#
#   ImportError: 'diffusers' is required but not installed.
#   Install it with: pip install 'lerobot[diffusion]'
#
# 🚨 REFUSES TO RUN while any phi job is queued or running. Installing into
#    site-packages under a live job can swap torch or lerobot out from under it
#    mid-run, and the failure surfaces hours later as something unrelated.
#
# 🚨 lerobot on Explorer is a PyPI install, NOT editable — so it is
#    `pip install "lerobot[diffusion]==0.6.0"`. The `==0.6.0` pin is load-bearing:
#    without it pip may upgrade lerobot itself and invalidate every model card and
#    checkpoint we have produced.
#
# 🚨 AND IT HARD-FAILS IF TORCH MOVES. diffusers declares a torch dependency, so
#    pip is free to "helpfully" reinstall it — which would silently undo the
#    cu126 fix (see fix_torch_cu126.sbatch) and break every GPU job in a way that
#    looks like a CUDA problem rather than a pip problem. Printing the versions is
#    not enough; this exits non-zero.
set -uo pipefail
PROJECT=/scratch/$USER/phi

# Count other phi jobs, EXCLUDING this one. Matching "^phi-" on the name alone
# makes the guard count itself -- this job is called phi-install-* -- so it
# refuses to run 100% of the time. That is what happened on 2026-08-12 (job
# 9088434, FAILED in 1 s). Filter by job id, not by name.
OTHERS=$(squeue -u "$USER" -h -o "%i %j" \
         | awk -v me="${SLURM_JOB_ID:-0}" '$1 != me && $2 ~ /^phi-/' | wc -l | tr -d ' ')
if [ "$OTHERS" -gt 0 ]; then
  echo "REFUSING: $OTHERS other phi job(s) queued/running. Installing now could break them."
  squeue -u "$USER" -o "%.14i %.18j %.8T"
  exit 1
fi

export PYTHONNOUSERSITE=1
export http_proxy=http://10.99.0.130:3128
export https_proxy=http://10.99.0.130:3128
export HF_HOME=$PROJECT/hf

module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PROJECT/envs/lerobot"

BEFORE=$(python -c "import torch; print(torch.__version__, torch.version.cuda)")
LR_BEFORE=$(python -c "import lerobot; print(lerobot.__version__)" 2>/dev/null || echo "unknown")
echo "=== BEFORE ==="
echo "  torch   $BEFORE"
echo "  lerobot $LR_BEFORE"

pip install "lerobot[diffusion]==0.6.0" || exit 1

AFTER=$(python -c "import torch; print(torch.__version__, torch.version.cuda)")
LR_AFTER=$(python -c "import lerobot; print(lerobot.__version__)" 2>/dev/null || echo "unknown")
echo "=== AFTER ==="
echo "  torch   $AFTER"
echo "  lerobot $LR_AFTER"

if [ "$BEFORE" != "$AFTER" ]; then
  echo "🚨 TORCH CHANGED: '$BEFORE' -> '$AFTER'"
  echo "   Every GPU job will now behave differently, and it will look like a CUDA fault."
  echo "   Repair with configs/hpc/fix_torch_cu126.sbatch before running anything."
  exit 1
fi
if [ "$LR_BEFORE" != "$LR_AFTER" ]; then
  echo "🚨 LEROBOT CHANGED: '$LR_BEFORE' -> '$LR_AFTER' — every model card and checkpoint reference is now suspect."
  exit 1
fi

python - <<'PY' || exit 1
import diffusers
print(f"diffusers {diffusers.__version__} imports OK")
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy  # noqa: F401
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.configs.types import FeatureType, PolicyFeature

# Build the exact 3-camera config the real run uses, so an incompatibility shows
# up HERE and not eight hours into a GPU allocation.
feats = {"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,))}
for c in ("wrist", "front", "top"):
    feats[f"observation.images.{c}"] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640))
cfg = DiffusionConfig(
    input_features=feats,
    output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
    n_obs_steps=2, horizon=48, n_action_steps=24, drop_n_last_frames=23,
    down_dims=(256, 512, 1024), resize_shape=(240, 320), crop_ratio=0.9,
    use_group_norm=True, pretrained_backbone_weights=None,
    noise_scheduler_type="DDIM", num_train_timesteps=100, num_inference_steps=16,
)
from lerobot.policies.diffusion.modeling_diffusion import DiffusionModel
m = DiffusionModel(cfg)
unet = sum(p.numel() for p in m.unet.parameters())
tot = sum(p.numel() for p in m.parameters())
print(f"  To/Ta/Tp = {cfg.n_obs_steps}/{cfg.n_action_steps}/{cfg.horizon}  drop={cfg.drop_n_last_frames}")
print(f"  crop_shape {cfg.crop_shape}  group_norm {cfg.use_group_norm}  pretrained {cfg.pretrained_backbone_weights}")
print(f"  sched {cfg.noise_scheduler_type} {cfg.num_train_timesteps}/{cfg.num_inference_steps}")
print(f"  UNet {unet:,}  (paper Table 7 real-world #D-Params = 67M)")
print(f"  TOTAL {tot:,}")
assert 60e6 < unet < 80e6, f"UNet size {unet} is not near the paper's 67M — check down_dims"
# and the lerobot-default arm of the A/B must build too
cfg2 = DiffusionConfig(input_features=feats,
    output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
    n_obs_steps=2, horizon=48, n_action_steps=24, drop_n_last_frames=23,
    down_dims=(256, 512, 1024), resize_shape=(240, 320), crop_ratio=0.9)
DiffusionModel(cfg2)
print("  lerobot-default arm (BatchNorm + ImageNet) builds OK")
PY

echo "install finished rc=$? at $(date)"

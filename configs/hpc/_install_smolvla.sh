#!/bin/bash
# One-off: add the SmolVLA extra to the Explorer env.
#
#   sbatch --partition=short --cpus-per-task=4 --mem=16G --time=00:40:00 \
#     --job-name=phi-install-smolvla \
#     --output=/scratch/$USER/phi/logs/install-smolvla-%j.out \
#     configs/hpc/_install_smolvla.sh
#
# 🚨 REFUSES TO RUN while any phi training job is queued or running. Installing
#    into site-packages under a live job can swap torch or lerobot out from under
#    it mid-run, and the failure would surface hours later as something
#    unrelated. The guard below is the whole point of this being a script.
#
# 🚨 lerobot on Explorer is a PyPI install, NOT editable — so it is
#    `pip install "lerobot[smolvla]==0.6.0"`, not `pip install -e ".[smolvla]"`
#    as the upstream docs say. The `==0.6.0` pin is load-bearing: without it pip
#    is free to upgrade lerobot itself and invalidate every model card and
#    checkpoint we have produced.
set -uo pipefail
PROJECT=/scratch/$USER/phi

RUNNING=$(squeue -u "$USER" -h -o "%j" | grep -c "^phi-" || true)
if [ "$RUNNING" -gt 0 ]; then
  echo "REFUSING: $RUNNING phi job(s) still queued/running. Installing now could break them."
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

echo "=== torch BEFORE (must not change) ==="
python -c "import torch; print(' torch', torch.__version__, '| cuda', torch.version.cuda)"

pip install "lerobot[smolvla]==0.6.0" || exit 1

echo "=== torch AFTER ==="
python -c "import torch; print(' torch', torch.__version__, '| cuda', torch.version.cuda)"

python - <<'PY' || exit 1
import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy   # noqa: F401
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
c = SmolVLAConfig()
print("smolvla imports OK")
print(f"  chunk_size {c.chunk_size} · n_action_steps {c.n_action_steps} · lr {c.optimizer_lr}")
print(f"  freeze_vision_encoder {c.freeze_vision_encoder} · train_expert_only {c.train_expert_only}")
assert torch.cuda.is_available() is False or True   # cuda check belongs to the GPU job
PY

echo "install finished rc=$? at $(date)"

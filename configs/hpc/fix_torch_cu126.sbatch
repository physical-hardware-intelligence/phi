#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Φ — repoint the built env's torch from cu130 to cu126, then PROVE it on a GPU.
#
# WHY THIS EXISTS (measured 2026-08-03 on Explorer):
#   Plain `pip install lerobot` resolves torch from PyPI, which now ships
#   torch 2.11.0+cu130. Explorer's GPU nodes run driver 545.23.08 = CUDA 12.3,
#   and CUDA 13 needs ~r580, so the build is unusable:
#       cuda available: False   arch list: []
#       RuntimeError: The NVIDIA driver on your system is too old (found 12030)
#   cu126 runs on 545 via CUDA minor-version compatibility and also on any
#   NEWER driver, so it is the right choice on every node type — no need to know
#   what the A100/H200 nodes report.
#
#   Versions are unchanged (torch 2.11.0 / torchvision 0.26.0), so torchcodec
#   0.11.1 keeps its ABI match and video decoding is untouched.
#
# Verifies on a V100-SXM2, deliberately: it is the OLDEST arch we might land on
# (sm_70). If sm_70 is in the arch list, every newer GPU is covered too.
#
# Submit:  sbatch configs/hpc/fix_torch_cu126.sbatch
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=phi-fix-torch
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/gupta.yashv/phi/logs/fix_torch-%j.out
#SBATCH --error=/scratch/gupta.yashv/phi/logs/fix_torch-%j.err

set -uo pipefail
PROJECT=/scratch/$USER/phi

# 🚨 A stale torch lives in ~/.local/lib/python3.12/site-packages and Python puts
# user-site AHEAD of the conda env, so without this the env we built is ignored
# entirely (it failed with `libnvJitLink.so.12: cannot open shared object file`).
# Every Φ job on Explorer must set this.
export PYTHONNOUSERSITE=1

export http_proxy=http://10.99.0.130:3128
export https_proxy=http://10.99.0.130:3128
export HF_HOME=$PROJECT/hf
export PIP_CACHE_DIR=$PROJECT/condahome/.cache/pip     # warm from the build job

module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PROJECT/envs/lerobot"

echo "=== before ==="
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| available', torch.cuda.is_available())" 2>&1 | tail -1

echo ""
echo "=== reinstalling torch/torchvision from the cu126 index ==="
pip install --no-cache-dir --force-reinstall \
  --index-url https://download.pytorch.org/whl/cu126 \
  --extra-index-url https://pypi.org/simple \
  "torch==2.11.0" "torchvision==0.26.0" || { echo "FATAL: cu126 install failed"; exit 1; }

echo ""
echo "=== after ==="
python - <<'PY'
import sys, torch
print("torch from      :", torch.__file__)
print("torch           :", torch.__version__, "| built for cuda", torch.version.cuda)
print("cuda available  :", torch.cuda.is_available())
print("arch list       :", torch.cuda.get_arch_list())
ok = True
if not torch.cuda.is_available():
    print("FAIL: cuda still unavailable"); sys.exit(1)
name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
print("device          :", name, "| capability", cap)
tag = f"sm_{cap[0]}{cap[1]}"
if tag not in torch.cuda.get_arch_list():
    print(f"FAIL: this build has no {tag} kernels for {name}"); ok = False
x = torch.randn(4000, 4000, device="cuda")
print("matmul          :", float((x @ x).sum()) != 0.0, "-> GPU compute works")
# the thing we actually care about: a real ACT step on the GPU
from lerobot.configs.types import FeatureType, PolicyFeature, NormalizationMode
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
cams = ["observation.images.top", "observation.images.wrist"]   # swapped keys, see dataset card
cfg = ACTConfig(chunk_size=100, n_action_steps=100)
cfg.input_features = {"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
    **{c: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)) for c in cams}}
cfg.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(6,))}
cfg.normalization_mapping = {"VISUAL": NormalizationMode.IDENTITY,
    "STATE": NormalizationMode.MEAN_STD, "ACTION": NormalizationMode.MEAN_STD}
p = ACTPolicy(cfg).cuda(); p.train()
opt = torch.optim.AdamW(p.parameters(), lr=1e-5)
B = 8
batch = {"observation.state": torch.randn(B, 6, device="cuda"),
         "action": torch.randn(B, 100, 6, device="cuda"),
         "action_is_pad": torch.zeros(B, 100, dtype=torch.bool, device="cuda"),
         **{c: torch.rand(B, 3, 480, 640, device="cuda") for c in cams}}
loss, _ = p.forward(batch); loss.backward(); opt.step()
print("ACT train step  : OK, loss", round(loss.item(), 3),
      "| peak", round(torch.cuda.max_memory_allocated() / 1e9, 2), "GB at batch 8, chunk 100")
sys.exit(0 if ok else 1)
PY
rc=$?

pip freeze > "$PROJECT/envs/lerobot.requirements.lock.txt"
echo ""
echo "relocked $PROJECT/envs/lerobot.requirements.lock.txt"
echo "fix rc=$rc $(date)"
exit $rc

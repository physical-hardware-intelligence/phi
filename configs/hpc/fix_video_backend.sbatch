#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Φ — decide the video decode backend for Explorer, with measurements.
#
# THE PROBLEM (bench 8935296, 2026-08-04): torchcodec cannot load.
#   OSError: libnppicc.so.12: cannot open shared object file
#     -> could not load torchcodec/libtorchcodec_core{4..7}.so
# libnppicc is NVIDIA Performance Primitives. torch's cu126 wheels do NOT pull
# the NPP pip package, so torchcodec's dlopen fails and NO video decodes at all.
# Every batch size in the bench died before touching the GPU (peak 0 MiB).
#
# TWO CANDIDATE FIXES, and we pick on evidence not preference:
#   A) pip install nvidia-npp-cu12  -> keeps torchcodec (the faster path)
#   B) --dataset.video_backend=pyav -> drops torchcodec entirely (fewer moving
#      parts, but AV1 on CPU may make the dataloader the bottleneck)
#
# Deliberately NOT doing: adding /shared/.../cuda/12.8.0/lib64 to
# LD_LIBRARY_PATH. It does contain libnppicc.so.12, but it would also shadow
# torch's bundled cu126 cublas/cudnn with 12.8 builds — a much larger blast
# radius than one missing leaf library deserves.
#
# Submit:  sbatch configs/hpc/fix_video_backend.sbatch
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=phi-fix-video
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:1
#SBATCH --exclude=d1025
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/gupta.yashv/phi/logs/fix_video-%j.out
#SBATCH --error=/scratch/gupta.yashv/phi/logs/fix_video-%j.err

set -uo pipefail
PROJECT=/scratch/$USER/phi
export PYTHONNOUSERSITE=1
export http_proxy=http://10.99.0.130:3128
export https_proxy=http://10.99.0.130:3128
export HF_HOME=$PROJECT/hf
export HF_LEROBOT_HOME=$PROJECT/lerobot-data
export PIP_CACHE_DIR=$PROJECT/condahome/.cache/pip

module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PROJECT/envs/lerobot"
source "$PROJECT/repo/configs/hpc/_preflight.sh"

echo ""
echo "=== A) install NPP so torchcodec can load ==="
pip install "nvidia-npp-cu12" 2>&1 | tail -3
NPP_LIB=$(python -c "import os,nvidia; print(os.path.join(os.path.dirname(nvidia.__file__),'npp','lib'))" 2>/dev/null)
echo "npp lib dir: $NPP_LIB"
ls "$NPP_LIB" 2>/dev/null | head -5
# narrow addition: ONLY the npp dir, so torch's own cu126 libs are not shadowed
export LD_LIBRARY_PATH="$NPP_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo ""
echo "=== decode a REAL frame from our dataset with each backend ==="
python - <<'PY'
import time, torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

REPO = "BrutalCaesar/phi_so101_8bin_v1"

print("torchcodec importable?", end=" ")
try:
    import torchcodec  # noqa: F401
    from lerobot.datasets.video_utils import decode_video_frames  # noqa: F401
    print("YES")
    tc_ok = True
except Exception as e:
    print("NO ->", type(e).__name__, str(e)[:90])
    tc_ok = False

def timed(backend, n=40):
    ds = LeRobotDataset(REPO, episodes=[0], video_backend=backend)
    _ = ds[0]                                    # warm the decoder
    t0 = time.perf_counter()
    for i in range(n):
        item = ds[i]
    dt = (time.perf_counter() - t0) / n
    keys = [k for k in item if "image" in k]
    shp = tuple(item[keys[0]].shape) if keys else None
    return dt, len(keys), shp

for backend in (["torchcodec"] if tc_ok else []) + ["pyav"]:
    try:
        dt, ncam, shp = timed(backend)
        print(f"  {backend:11s} {dt*1000:7.1f} ms/sample  ({ncam} cams, {shp})")
        # batch 32 with 8 workers needs a sample every ~dt/8 s of wall time;
        # print the implied ceiling so the number is actionable
        print(f"              -> {8/dt:6.1f} samples/s with 8 workers "
              f"= {8/dt/32:5.2f} batches/s at batch 32")
    except Exception as e:
        print(f"  {backend:11s} FAILED: {type(e).__name__}: {str(e)[:120]}")
PY

echo ""
echo "=== relock ==="
pip freeze > "$PROJECT/envs/lerobot.requirements.lock.txt"
echo "wrote $(wc -l < "$PROJECT/envs/lerobot.requirements.lock.txt") pkgs"
echo "DONE $(date)"

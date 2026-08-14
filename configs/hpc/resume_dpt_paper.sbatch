#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Φ — RESUME the `paper` arm of job 9145402 from 30k to 60k steps.
#
#   Submit:  sbatch configs/hpc/resume_dpt_paper.sbatch
#
# WHY ONLY THIS ARM, AND WHY MORE STEPS AT ALL
# --------------------------------------------
# `unreg` (wd 1e-6, drop 0.0) is FINISHED. It bottomed at step 8,000 and climbed
# monotonically to 30k. Extending it would spend hours confirming a curve already
# measured -- exactly the mistake the CNN run made with its last 70k steps.
#
# `paper` (wd 1e-3, drop 0.3) is a different case. Reported at 30k as "converged",
# which was WRONG: the last three eval points span 0.0005, and point-to-point
# noise in that curve is +/- 0.0010-0.0016 (there are +0.0016 up-ticks mid-descent),
# so three flat points cannot distinguish a plateau from a slow decline. Block
# means, which average the noise out, still fall:
#
#     steps  2000-6000   0.06867
#     steps  8000-12000  0.06030
#     steps 14000-18000  0.05703
#     steps 20000-24000  0.05427
#     steps 26000-30000  0.05330   <- block delta still -0.00097, still NEGATIVE
#
# 🔑 THE REAL ARGUMENT IS TRAIN LOSS, NOT eval_loss. `paper` ended at train loss
#    0.061 against `unreg`'s 0.013 -- after 29 epochs it is nowhere near fitting
#    its own training data. That is direct evidence of UNDERFITTING, independent
#    of a held-out metric we already know does not predict task success. Rolling
#    out an undertrained checkpoint would be an unfair test of the regularised
#    recipe, and arm time is the scarce resource here.
#
# ⚠️ DO NOT EXPECT THIS TO BEAT `unreg` ON eval_loss. Naive linear extrapolation
#    of the recent rate -- which OVERstates it, the curve is decelerating --
#    reaches ~0.042 at 100k against `unreg`'s 0.0287. The gap does not close.
#    The point is convergence, not winning the metric.
#
# WHY 60k AND NOT 100k
# --------------------
# Measured: 30k took 2h36m = 5.20 min per 1k steps. 100k straight is 8.67 h and
# BLOWS the 8h MaxTime; it would need eval relaxed to every 5000 to fit. 60k is
# 30k more steps = ~2.6 h, comfortable margin, and it keeps eval at every 2000 so
# the extension has the SAME resolution as the run it continues. If the curve is
# still falling at 60k, extend again -- cheaper than a timeout.
#
# EITHER OUTCOME IS INFORMATIVE
# -----------------------------
#   still falling at 60k  -> `paper` was undertrained at 30k; the 30k comparison
#                            understated the regularised recipe.
#   turns somewhere 30-60k -> regularisation DELAYS overfitting rather than
#                            preventing it, which is a stronger and more useful
#                            claim than 30k alone can support.
#   flat                  -> converged; the 30k reading stands.
#
# HOW RESUME WORKS HERE
# ---------------------
# `--config_path` points at the checkpoint's train_config.json and `--resume=true`
# rehydrates optimizer state, scheduler, and the dataloader offset. Config comes
# FROM THE CHECKPOINT, so every parameter is identical by construction -- only
# the CLI `--steps` override changes. That is the point: no chance of silently
# resuming with a different weight decay.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=phi-dpt-resume
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=/scratch/gupta.yashv/phi/logs/dptres-%j.out
#SBATCH --error=/scratch/gupta.yashv/phi/logs/dptres-%j.err

set -uo pipefail
PROJECT=/scratch/$USER/phi
RUN=$PROJECT/results/dpt_paper_emb256
STEPS=${STEPS:-60000}

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

CFG=$RUN/checkpoints/last/pretrained_model/train_config.json
if [ ! -f "$CFG" ]; then
  echo "❌ no checkpoint config at $CFG"
  echo "   Available: $(ls -1 $RUN/checkpoints 2>/dev/null | tr '\n' ' ')"
  exit 1
fi

# Fail fast if the port is missing -- resume loads a diffusion_transformer policy,
# which does not exist without src/phi/policies on the path.
python - <<'PY' || { echo "❌ transformer port not importable."; \
                     echo "   Fix: rsync from the Mac (\$PROJECT/repo is an rsync SNAPSHOT,"; \
                     echo "        not a git checkout -- see docs/hpc/explorer.md)"; exit 1; }
import phi.policies.dp_transformer  # noqa: F401  registers diffusion_transformer
from lerobot.policies.factory import get_policy_class
assert get_policy_class("diffusion_transformer").__name__ == "DiffusionTransformerPolicy"
print("  port OK")
PY

echo "=============================================================="
echo " RESUME dpt_paper_emb256   30,000 -> $STEPS steps"
echo "   config from : $CFG"
echo "   (weight_decay / dropout / horizon all inherited from the checkpoint)"
echo " node $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo " eval_loss so far, for reference:"
grep -oE "step [0-9]+: eval_loss=[0-9.]+" $PROJECT/logs/dpt-9145402_0.err 2>/dev/null | tail -5
echo "=============================================================="

srun python -m phi.train.train_dp_transformer \
  --config_path="$CFG" \
  --resume=true \
  --steps=$STEPS

RC=$?
echo
echo "=============================================================="
echo " DONE  rc=$RC"
echo "=============================================================="
grep -oE "step [0-9]+: eval_loss=[0-9.]+" "$PROJECT/logs/dptres-${SLURM_JOB_ID}.err" 2>/dev/null | tail -20
echo
echo " Compare against the 30k run before concluding anything:"
echo "   unreg best 0.0287 @ 8k   |   paper best 0.0531 @ 26k"
echo " And remember eval_loss did NOT predict rollout success on 2026-08-12."
echo " Use phi.eval.eval_by_k to see WHICH noise regime moved."

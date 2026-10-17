#!/usr/bin/env bash
# Track S — one simulation evaluation of Ilia's lehome_sim, fully recorded.
# Usage: run_sim_eval.sh <label> [scripts/run_eval.py args...]
#   e.g. run_sim_eval.sh smoke --num_workers 1 --num_episodes 2 --all --metrics_only --no_wandb
# Writes ~/phi/logs/sim/<timestamp>-<label>/: gpu_guard.txt, run.txt, gpu.csv (every 5 s), eval.log, gpu_after.txt
set -euo pipefail
label=$1; shift
out=~/phi/logs/sim/$(date +%Y%m%d-%H%M%S)-$label
mkdir -p "$out"
bash ~/phi/sim/gpu_guard.sh 12 | tee "$out/gpu_guard.txt"      # NO-GO stops here (pipefail)
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,temperature.gpu,power.draw --format=csv -l 5 > "$out/gpu.csv" &
mon=$!
trap 'kill $mon 2>/dev/null || true' EXIT
R=~/phi/lehome_solution
{ echo "label: $label"; echo "start: $(date -Is)"; echo "args: $*"
  echo "lehome_solution $(git -C $R rev-parse HEAD)"
  echo "lehome-challenge $(git -C $R/lehome-challenge rev-parse HEAD)"
  echo "IsaacLab $(git -C $R/lehome-challenge/third_party/IsaacLab rev-parse HEAD)"; } > "$out/run.txt"
cd "$R"
export PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin"
set +e
uv run python scripts/run_eval.py --checkpoint_dir outputs/checkpoints/lehome_sim --config_name pi_modified_bc_rl \
  --video_dir "$out/videos" "$@" 2>&1 | tee "$out/eval.log"
rc=${PIPESTATUS[0]}
set -e
echo "end: $(date -Is) · exit $rc" >> "$out/run.txt"
sleep 5
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader > "$out/gpu_after.txt"
echo "GPU after the run:"; cat "$out/gpu_after.txt"; echo "run folder: $out"
exit "$rc"

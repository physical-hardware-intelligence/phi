#!/usr/bin/env bash
# Go/no-go for GPU work on the shared box. Exit 0 = GO, 1 = NO-GO.
# NO-GO if free memory < the GiB asked for, or anyone else runs a compute job over 1 GiB.
# Usage: gpu_guard.sh [required_free_GiB, default 12]
need=${1:-12}
read -r used total < <(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | tr -d ',')
free=$(( (total - used) / 1024 ))
echo "[$(date '+%F %H:%M:%S')] GPU ${used}/${total} MiB used · ${free} GiB free · need ${need} GiB"
heavy=0
while IFS=', ' read -r pid mem name; do
  [ -z "$pid" ] && continue
  user=$(ps -o user= -p "$pid" 2>/dev/null || echo '?')
  echo "  compute app: pid $pid · $user · ${mem} MiB · $name"
  [ "$user" != "$(whoami)" ] && [ "$mem" -gt 1024 ] && heavy=1
done < <(nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader,nounits)
if [ "$free" -lt "$need" ] || [ "$heavy" -eq 1 ]; then echo "NO-GO"; exit 1; fi
echo "GO"

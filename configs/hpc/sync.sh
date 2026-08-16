#!/bin/bash
# Push this repo to Explorer. $PROJECT/repo is an rsync SNAPSHOT, not a git checkout,
# so this is the only way code reaches the cluster.
#
#   ./configs/hpc/sync.sh
#
# 🚨 USE THIS INSTEAD OF A BARE `rsync -az --delete`.
#
# Jobs WRITE into the synced tree -- eval_by_k.sbatch emits
# $PROJECT/repo/outputs/k_eval_<run>.csv. A plain `rsync -az --delete` treats the
# Mac as authoritative and DELETES anything the cluster produced that the Mac does
# not have. That happened on 2026-08-15: four completed k-eval CSVs
# (dpt_bidir, dpp_bidir, dpp_causal, dpp_avgpool) were wiped seconds after the jobs
# finished writing them, by a sync run to deploy an unrelated one-line change.
#
# `--filter='protect outputs/***'` still UPLOADS local files under outputs/, but
# forbids the receiver from deleting remote-only ones. Fetch results back with
# ./configs/hpc/sync.sh pull
set -euo pipefail
cd "$(dirname "$0")/../.."

REMOTE=explorer:/scratch/gupta.yashv/phi/repo/
COMMON=(-az --exclude='.git' --exclude='__pycache__' --exclude='._*' --exclude='.DS_Store')

if [ "${1:-push}" = pull ]; then
  rsync "${COMMON[@]}" "${REMOTE}outputs/" ./outputs/
  echo "pulled outputs/ from Explorer"
else
  rsync "${COMMON[@]}" --delete --filter='protect outputs/***' ./ "$REMOTE"
  echo "pushed to Explorer (remote outputs/ protected from --delete)"
fi

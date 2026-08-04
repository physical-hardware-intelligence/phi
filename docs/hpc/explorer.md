# Explorer (Northeastern HPC) — what bites, and the fix

Everything below was hit and measured on 2026-08-04 getting ACT training up. Scripts live in [`configs/hpc/`](../../configs/hpc/). Read this before writing a new job.

## Layout

```
/scratch/$USER/phi/
  envs/        conda env (prefix install)      logs/     sbatch output
  conda-pkgs/  project-local package cache     results/  run outputs
  condahome/   job-local $HOME                 hf/       HF_HOME
  lerobot-data/ HF_LEROBOT_HOME                repo/     rsync of this repo
```

Home is capped at **76 GB** (69.9 used) — too small for a 6.7 GB env plus caches. No `/projects` allocation. Scratch is the only option.

## Partitions

| Partition | Max time | GPUs |
|---|---|---|
| `gpu` | **8 h** | h200 ×8/node, a100, v100-sxm2/pcie (all 32 GB), t4 (16 GB) |
| `gpu-short`, `gpu-interactive` | 2 h | same pool |
| `short` | 2 days | none — use for installs and downloads |
| `courses-gpu` | 24 h | p100 (16 GB), v100 only. Needs group `rc` or `courses` |

**Do not pin a GPU type.** Node features are `(null)` on most GPU nodes so `--constraint` is useless. `d1025` (t4) is the only sub-32 GB node in `gpu`, so:

```bash
#SBATCH --gres=gpu:1
#SBATCH --exclude=d1025
```

That stays eligible for h200 **and** a100 **and** both v100s — SLURM starts you on whichever frees first. Size the batch for 32 GB and it runs anywhere.

## 🚨 The five traps

**1. `/scratch` returns stale file handles (ESTALE).** A working env lost every stdlib `.py` file between two jobs; `lib/python3.12/encodings/` was left holding an empty `__pycache__`. The same fault poisons the conda package cache and produces `CondaVerificationError: the path ... cannot be found`. **If you have ever blamed conda for random verification errors on this cluster, this is probably why.** Recovery: rename the env and the pkgs cache aside, rebuild. Report recurrences to RC.

**2. `python -V` lies.** On a gutted env it still prints `Python 3.12.13`, because `-V` short-circuits before interpreter init. Health-check by **importing**, never by version or by `ls`:

```bash
python -c "import encodings, codecs, ssl, sqlite3"
```

[`configs/hpc/_preflight.sh`](../../configs/hpc/_preflight.sh) does this plus torch/lerobot/CUDA, and every job sources it. It turns a 40-line codec error into one line naming the fix.

**3. PyPI torch is unusable here.** Default resolution gives `torch 2.11.0+cu130`; the GPU nodes run driver **545.23.08 (CUDA 12.3)**, and CUDA 13 needs ~r580:

```
cuda available: False   arch list: []
RuntimeError: The NVIDIA driver on your system is too old (found version 12030)
```

Pin **cu126** — works on 545 via minor-version compatibility *and* on any newer driver, and ships `sm_70` through `sm_90` so one env covers V100 → H200. Keep the same torch/vision versions the default resolved, so `torchcodec`'s ABI match survives. See [`env/environment.cuda.yml`](../../env/environment.cuda.yml).

**4. `~/.local` shadows the conda env.** Python puts user-site ahead of the env, so a stale `~/.local/.../torch` loads instead and dies on `libnvJitLink.so.12`. Every job needs:

```bash
export PYTHONNOUSERSITE=1
```

**5. Compute nodes have no internet.** Proxy is `http://10.99.0.130:3128`. conda reads it from `.condarc`; **pip does not** — export `http_proxy`/`https_proxy` explicitly.

## Conda

`~/.condarc` may pin `pkgs_dirs` at a project directory that no longer exists, and `CONDA_PKGS_DIRS`/`CONDARC` only **merge** rather than replace. Fix: a **job-local `$HOME`** holding a rewritten `.condarc` (keep the proxy lines, replace `pkgs_dirs`). Build on a **compute node** — the login node kills `conda env create`. Pattern is in [`build_env.sbatch`](../../configs/hpc/build_env.sbatch).

Two rules learned the hard way:

- **A directory existing is not an env working.** A failed create leaves a skeleton (`conda-meta/`, `etc/`); reusing it silently activated the *base* anaconda 3.9 and the job reported success with a 0-package lock file. Health-check, assert `sys.executable == $PREFIX/bin/python`, and fail if the lock has < 50 packages (a real build resolves ~118).
- **Never `rm -rf` a multi-GB env.** It outlasts an interactive timeout and ESTALE makes it fail partway. **Rename**, then delete from a batch job.

## Video decoding: use pyav

`torchcodec` will not load — `libnppicc.so.12: cannot open shared object file`. torch's cu126 wheels don't pull NVIDIA NPP, and installing `nvidia-npp-cu12` only moves the error on to FFmpeg. Measured alternative:

```
pyav  20.4 ms/sample (3 cameras) -> 392 samples/s with 8 workers
```

Pass `--dataset.video_backend=pyav`. In a real run this shows up as **`data_s: 0.005`** — decode is fully hidden behind the GPU step, so it costs nothing. Don't add `/shared/.../cuda/12.8.0/lib64` to `LD_LIBRARY_PATH` to chase NPP: it would shadow torch's bundled cu126 cuBLAS/cuDNN with 12.8 builds.

## Measured ACT throughput

V100-SXM2-32GB · 2 cameras @ 640×480 · chunk 100 · `num_workers=8` · pyav:

| Batch | `updt_s` | `data_s` | steps/s | `smpl/s` | VRAM |
|---|---|---|---|---|---|
| 16 | 0.299 | 0.005 | 3.29 | 53 | 6.91 GB |
| 32 | 0.577 | 0.009 | 1.71 | 55 | 13.2 GB |
| 48 | 0.851 | 0.012 | 1.16 | 56 | 19.5 GB |

**`smpl/s` is flat across batch size** → runs are GPU-bound, so wall time depends on *samples processed*, not batch size. Fit: **≈ 0.62 GB + 0.393 GB/sample** (linear to 3 decimal places across all three points), so batch 71 fits 32 GB. Batch 48 buys +1.8% throughput for +48% VRAM — the GPU is already saturated at batch 16. Shape-level detail in [act-shapes](../theory/act-shapes.md).

Step budget:

```
steps_that_fit = steps/s × 3600 × 7.5      # 7.5 h of the 8 h wall
```

**Choosing epochs:** there is no formula. Anchor on LeRobot's default (`100k steps × batch 8` = **14.6 epochs** on 54,770 frames), then let the asymmetry decide — training resumes from `last`, so under-shooting costs one resubmission while over-running the wall **loses the whole run**. Take the shorter run and save often.

## W&B

`wandb login` writes `~/.netrc`. Point caches at scratch (`WANDB_DIR`, `WANDB_CACHE_DIR`) — W&B data in a quota-limited home will kill a long job. Jobs auto-detect: online if a credential exists, else `--wandb.mode=offline` and sync afterwards **from the login node** (which does have internet):

```bash
wandb sync /scratch/$USER/phi/wandb/offline-run-*
```

## Job order

```bash
sbatch configs/hpc/build_env.sbatch                      # short, ~18 min, no GPU
sbatch configs/hpc/prefetch_dataset.sbatch               # short, ~2 min — once, so array tasks don't race
sbatch configs/hpc/bench_act.sbatch                      # gpu-short — measure, never guess
sbatch configs/hpc/train_8bin_v1.sbatch                  # gpu, array 0-5
```

Chain them so a failure stops the line:

```bash
B=$(sbatch --parsable configs/hpc/build_env.sbatch)
P=$(sbatch --parsable --dependency=afterok:$B configs/hpc/prefetch_dataset.sbatch)
sbatch --dependency=afterok:$P configs/hpc/bench_act.sbatch
```

⚠️ **Never pipe a job's command through `grep` without checking `PIPESTATUS`.** A bench run exited 0 while every batch size had died before touching the GPU (`peak GPU memory: 0 MiB`).

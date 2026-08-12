# Model zoo

Trained checkpoints live on the **Hugging Face Hub**. This folder holds one **card** per model.

Card = a `<slug>.md` with: HF id + revision · policy type · dataset trained on · steps/batch/seed · eval success (k/N) on which task · known failure modes · the exact train + eval commands (or link the experiment).

🚨 **Uploading from Crucial_X9 (exFAT): scrub `._*` first.** macOS writes AppleDouble sidecars next to
every file, and `upload_folder` happily pushes them — we shipped `._model.safetensors` to the Hub on
2026-08-05. Run `find <dir> -name "._*" -delete` before uploading.

🚨 **A repo can exist with zero files for the whole upload.** `upload_folder` pushes the LFS blob
first and creates the commit last, so for ~10 minutes a 200 MB model looks like a broken/empty repo
to anyone who tries to pull it. Confirm with `HfApi().list_repo_files(...)` before telling someone
it is ready.

| Model (HF id) | Policy | Dataset | Steps | Eval (k/N) | Task | Notes |
|---|---|---|---|---|---|---|
| [act_so101_8bin_wrist_front_chunk50](act_so101_8bin_wrist_front_chunk50.md) | ACT, chunk 50 | phi_so101_8bin_v1 (89 ep) | 100K | _not run_ | 8-bin pick & place | wrist+front; **keys swapped** — wrist feeds `observation.images.top` |
| [act_so101_8bin_wrist_top_chunk50](act_so101_8bin_wrist_top_chunk50.md) | ACT, chunk 50 | phi_so101_8bin_v1 (89 ep) | 100K | _not run_ | 8-bin pick & place | wrist+top; **both keys transposed** — wrist→`.top`, top→`.wrist` |
| [act_so101_cubcyl_recovery_chunk50_noaug_3cam](act_so101_cubcyl_recovery_chunk50_noaug_3cam.md) | ACT, chunk 50 | phi_so101_cubes_cylinder_recovery_v1 (143 ep) | 100K | **6/11 held out** (55%, 0.636) | cubes/cylinder → box | 3 cams, keys **correct**; image aug **OFF** (A/B control). White cube **0/4 → 2/4** vs baseline |
| [act_so101_cubcyl_recovery_chunk50_aug_3cam](act_so101_cubcyl_recovery_chunk50_aug_3cam.md) | ACT, chunk 50 | phi_so101_cubes_cylinder_recovery_v1 (143 ep) | 100K | _not run_ | cubes/cylinder → box | 3 cams, keys **correct**; image aug **ON**; 80k weights on branch `step-80000` |
| [dp_so101_cubcyl_recovery_tp48_paper_3cam](dp_so101_cubcyl_recovery_tp48_paper_3cam.md) | **Diffusion**, Tp48/Ta24 | phi_so101_cubes_cylinder_recovery_v1 (143 ep) | **20K** | **6/12 held out** (50%, 0.650) | cubes/cylinder → box | paper encoder (GroupNorm + scratch); **WON the A/B**; overfit past 20k **and rolled out fine anyway** — ties ACT |
| [dp_so101_cubcyl_recovery_tp48_lerobot_3cam](dp_so101_cubcyl_recovery_tp48_lerobot_3cam.md) | **Diffusion**, Tp48/Ta24 | phi_so101_cubes_cylinder_recovery_v1 (143 ep) | **20K** | _not run_ | cubes/cylinder → box | lerobot encoder (BatchNorm + ImageNet); lost the A/B |

🚨 **Diffusion Policy models have a real-time deadline that ACT models do not.** DP runs its UNet `num_inference_steps` times per chunk, not once. Measured 294.4 ms/chunk on the Mac, so `n_action_steps` must stay ≳10 at 30 fps and `num_inference_steps` must stay well below lerobot's default 100. See [docs/deployment §4](../docs/deployment/README.md). Missing the deadline raises no error.

🚨 **Calibration is a precondition of every number in the Eval column.** A policy emits angles in the calibration frame of the machine that recorded its training data. Run it against a different one and it mis-grasps with no error and no warning — confirmed on the arm 2026-08-10. Copy `configs/calibration/robots/so_follower/phi_follower.json` into `~/.cache/huggingface/lerobot/calibration/robots/so_follower/` first, and diff with `python -m phi.utils.compare_calibration`. Any result taken under a different calibration is **void**, not merely noisy.

**Rollout scores live in [`outputs/rollout_scores.csv`](../outputs/rollout_scores.csv)** — aggregate with `python -m phi.utils.eval_rollouts --report`. As of **2026-08-12** it holds **46 scored rollouts across 3 models**:

| held-out | `cvae_3cam` (no recovery) | `recovery_noaug` (ACT) | `dp_..._paper` (Diffusion) |
|---|---:|---:|---:|
| n | 11 | 11 | 12 |
| success | 27% | **55%** | 50% |
| mean progress | 0.418 | 0.636 | **0.650** |
| white cube (45 mm) | 0/4 | **2/4** | 1/4 |

🔑 **ACT and DP are tied.** Paired over the 11 episodes both ran, DP − ACT = −0.018 (t(10) = −0.13; 7 ties). Both beat the no-recovery baseline by a consistent margin (+0.240 and +0.218) that **neither test resolves at n≈11** (p = 0.375 / 0.453). Detecting an effect this size at 80% power needs **n ≈ 40**. The scores are nearly binary, so extra *repeat trials on already-staged scenes* buy far more power per minute than new scenes — `--episodes` accepts duplicates.

🔑 **DP's held-out noise-MSE did not predict task success.** The DP checkpoint was knowingly past its own optimum on a loss that rose 4× monotonically, and still matched a 100k ACT model. Do not use DP `eval_loss` for model selection. See [the experiment](../experiments/2026-08-12_dp-recovery-encoder-ab.md).

The two `8bin` models and the DP `lerobot` arm remain unscored.

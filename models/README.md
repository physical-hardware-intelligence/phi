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
| [act_so101_cubcyl_recovery_chunk50_noaug_3cam](act_so101_cubcyl_recovery_chunk50_noaug_3cam.md) | ACT, chunk 50 | phi_so101_cubes_cylinder_recovery_v1 (143 ep) | 100K | _not run_ | cubes/cylinder → box | 3 cams, keys **correct**; image aug **OFF** (A/B control) |
| [act_so101_cubcyl_recovery_chunk50_aug_3cam](act_so101_cubcyl_recovery_chunk50_aug_3cam.md) | ACT, chunk 50 | phi_so101_cubes_cylinder_recovery_v1 (143 ep) | 100K | _not run_ | cubes/cylinder → box | 3 cams, keys **correct**; image aug **ON**; 80k weights on branch `step-80000` |

🚨 **Calibration is a precondition of every number in the Eval column.** A policy emits angles in the calibration frame of the machine that recorded its training data. Run it against a different one and it mis-grasps with no error and no warning — confirmed on the arm 2026-08-10. Copy `configs/calibration/robots/so_follower/phi_follower.json` into `~/.cache/huggingface/lerobot/calibration/robots/so_follower/` first, and diff with `python -m phi.utils.compare_calibration`. Any result taken under a different calibration is **void**, not merely noisy.

🚨 **Zero of the models in this table have a scored rollout.** That is the bottleneck, not model count.

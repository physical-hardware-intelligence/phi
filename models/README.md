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

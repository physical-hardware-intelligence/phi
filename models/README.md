# Model zoo

Trained checkpoints live on the **Hugging Face Hub**. This folder holds one **card** per model.

Card = a `<slug>.md` with: HF id + revision · policy type · dataset trained on · steps/batch/seed · eval success (k/N) on which task · known failure modes · the exact train + eval commands (or link the experiment).

| Model (HF id) | Policy | Dataset | Steps | Eval (k/N) | Task | Notes |
|---|---|---|---|---|---|---|
| [act_so101_8bin_wrist_front_chunk50](act_so101_8bin_wrist_front_chunk50.md) | ACT, chunk 50 | phi_so101_8bin_v1 (89 ep) | 100K | _not run_ | 8-bin pick & place | wrist+front; **keys swapped** — wrist feeds `observation.images.top` |

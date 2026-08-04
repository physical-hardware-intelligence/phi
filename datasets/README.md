# Dataset registry

Datasets live on the **Hugging Face Hub**, not in git. This folder holds one small **card** per dataset so the next student can find and trust it.

Card = a `<slug>.md` with: HF id · robot · task string · #episodes · cameras + resolutions · who collected it · date · known issues · visualize link (`https://huggingface.co/spaces/lerobot/visualize_dataset`).

| Dataset (HF id) | Task | #ep | Cameras | Collected by | Notes |
|---|---|---|---|---|---|
| [`BrutalCaesar/phi_so101_8bin_v1`](phi_so101_8bin_v1.md) | pick up the duck and place it in the box | 119 | wrist + front + top, 640×480 | Yash / Parv / Sai · 2026-08-03 | 🚨 **`wrist` and `top` keys hold each other's footage** · 8 bins, 2 held out for OOD eval (89 train / 30 eval) |

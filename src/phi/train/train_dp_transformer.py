"""Train Diffusion Policy with the TRANSFORMER backbone via lerobot's own pipeline.

lerobot ships no transformer backbone, and its `get_policy_class` is a hardcoded
if/elif chain with no plugin hook. Importing `phi.policies.dp_transformer`
registers the config and patches that function; everything after that is stock
`lerobot-train` -- same dataset loading, optimizer, scheduler, wandb, checkpoint
and resume logic the UNet runs used. That is deliberate: it keeps a DP-T vs
DP-CNN comparison honest, differing only in the denoising backbone.

    python -m phi.train.train_dp_transformer \
        --dataset.repo_id=BrutalCaesar/phi_so101_cubes_cylinder_recovery_v1 \
        --policy.type=diffusion_transformer \
        --policy.n_emb=256 --policy.n_layer=8 \
        ...

🚨 Use --policy.type=diffusion_transformer, NOT diffusion. The latter silently
   gives you the UNet with none of these hyperparameters applied.
"""

import phi.policies.dp_transformer  # noqa: F401  -- import registers the policy
import phi.policies.dp_patch  # noqa: F401  -- ditto; `diffusion_patch` needs this HERE
from lerobot.scripts.lerobot_train import train

# 🚨 BOTH imports must live in THIS module, not merely in a preflight check.
# `@PreTrainedConfig.register_subclass` fires at import time, and draccus builds
# argparse's `--policy.type` choice list from whatever is registered in THIS process.
# A preflight heredoc that imports the module proves nothing: it is a different
# interpreter. Symptom is `invalid choice: 'diffusion_patch'` listing every other
# policy, which reads like a typo rather than a missing import. Cost 2 GPU jobs
# on 2026-08-15; same root cause as e731502 for the rollout scripts.


def main() -> None:
    train()


if __name__ == "__main__":
    main()

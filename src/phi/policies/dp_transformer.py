"""Diffusion Policy with the paper's TIME-SERIES TRANSFORMER backbone.

lerobot ships only `DiffusionConditionalUnet1d` -- grep `lerobot/policies/diffusion/`
for "transformer" and you get zero hits. But Chi et al. §3.1 describe a second
backbone, and Table 8 gives its hyperparameters. This adds it WITHOUT forking
lerobot.

WHY A DROP-IN ADAPTER WORKS
---------------------------
`DiffusionModel._prepare_global_conditioning` ends with

    torch.cat(global_cond_feats, dim=-1).flatten(start_dim=1)

i.e. it builds (B, n_obs_steps, obs_feat) and then flattens to (B, To*obs_feat)
because that is what the UNet's FiLM wants. The transformer wants the UNFLATTENED
form -- and a row-major flatten is losslessly reversible (verified: reshape
recovers the original exactly). So the adapter only has to reshape back.

Everything else already lines up: lerobot calls
`self.unet(sample, timesteps, global_cond=...)` with sample (B, horizon, action_dim)
and timesteps (B,) long -- exactly what TransformerForDiffusion.forward accepts.

⚠️ THE HYPERPARAMETERS ARE NOT THE CNN's. Table 8 vs Table 7:

    weight_decay   transformer 1e-3   CNN 1e-6      <- 1000x MORE
    attn dropout   transformer 0.3    CNN n/a
    layers/width   8 / 256 (sim rows), 8 / 768 (Kitchen + Real Push-T)

Table 8's caption says "WDecay: weight decay (for transformer only)". §3.1 warns
the transformer is "more sensitive to hyperparameters" and recommends starting
with the CNN. Do NOT inherit weight_decay=1e-6 from the UNet run -- that is the
single most likely way to reproduce the 4x overfit we already measured.

🔑 THE NO-DECAY PARAMETER SPLIT MATTERS HERE. At wd=1e-3, decaying LayerNorm
weights and the learned position embeddings actively hurts. The reference
implementation separates them (`TransformerForDiffusion.get_optim_groups`);
`optim_groups()` below reproduces that split for lerobot's optimizer.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from lerobot.policies.diffusion.modeling_diffusion import DiffusionModel

from phi.policies.transformer_for_diffusion import TransformerForDiffusion


class DiffusionTransformerBackbone(nn.Module):
    """Presents TransformerForDiffusion behind DiffusionConditionalUnet1d's signature.

    The ONLY work done here is undoing lerobot's flatten. Keeping the call
    signature identical is what lets `DiffusionModel` stay untouched -- loss,
    sampling, normalisation and the whole training loop are inherited.
    """

    def __init__(self, config, global_cond_dim: int, *, n_layer: int = 8,
                 n_head: int = 4, n_emb: int = 256, p_drop_emb: float = 0.0,
                 p_drop_attn: float = 0.3, causal_attn: bool = True,
                 n_cond_layers: int = 0) -> None:
        super().__init__()
        if global_cond_dim % config.n_obs_steps != 0:
            raise ValueError(
                f"global_cond_dim {global_cond_dim} is not divisible by "
                f"n_obs_steps {config.n_obs_steps}; cannot un-flatten the conditioning"
            )
        self.n_obs_steps = config.n_obs_steps
        self.obs_feat_dim = global_cond_dim // config.n_obs_steps
        action_dim = config.action_feature.shape[0]

        self.net = TransformerForDiffusion(
            input_dim=action_dim,
            output_dim=action_dim,
            horizon=config.horizon,
            n_obs_steps=config.n_obs_steps,
            cond_dim=self.obs_feat_dim,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_emb,
            p_drop_emb=p_drop_emb,
            p_drop_attn=p_drop_attn,
            causal_attn=causal_attn,
            time_as_cond=True,
            obs_as_cond=True,
            n_cond_layers=n_cond_layers,
        )

    def forward(self, x: torch.Tensor, timestep: torch.Tensor,
                global_cond: torch.Tensor | None = None) -> torch.Tensor:
        if global_cond is None:
            raise ValueError("the transformer backbone requires global_cond")
        cond = global_cond.reshape(x.shape[0], self.n_obs_steps, self.obs_feat_dim)
        return self.net(x, timestep, cond)


class DiffusionTransformerModel(DiffusionModel):
    """DiffusionModel with `unet` replaced by the transformer.

    Subclassing rather than editing lerobot keeps the install upgradeable and
    keeps every other behaviour -- `compute_loss`, `conditional_sample`,
    `generate_actions` -- byte-identical to the UNet runs, so a DP-T vs DP-CNN
    comparison differs ONLY in the denoising backbone.
    """

    def __init__(self, config, **backbone_kwargs) -> None:
        super().__init__(config)
        global_cond_dim = self.unet.global_cond_dim if hasattr(self.unet, "global_cond_dim") else None
        if global_cond_dim is None:
            # Recompute it the way DiffusionModel does, so we do not depend on a private attr.
            global_cond_dim = config.robot_state_feature.shape[0]
            if config.image_features:
                num_images = len(config.image_features)
                if config.use_separate_rgb_encoder_per_camera:
                    global_cond_dim += self.rgb_encoder[0].feature_dim * num_images
                else:
                    global_cond_dim += self.rgb_encoder.feature_dim * num_images
            if config.env_state_feature:
                global_cond_dim += config.env_state_feature.shape[0]
            global_cond_dim *= config.n_obs_steps
        del self.unet
        self.unet = DiffusionTransformerBackbone(config, global_cond_dim, **backbone_kwargs)


def optim_groups(model: nn.Module, weight_decay: float = 1e-3) -> list[dict]:
    """Split parameters into decayed / not-decayed, per the reference implementation.

    Decayed:     Linear and MultiheadAttention WEIGHTS.
    Not decayed: every bias, every LayerNorm/GroupNorm/embedding weight, and the
                 learned position embeddings (`pos_emb`, `cond_pos_emb`).

    This is load-bearing at Table 8's wd=1e-3: shrinking a learned position
    embedding toward zero degrades the only signal that distinguishes timestep 5
    from timestep 40, and shrinking a LayerNorm gain fights the normalisation.
    The vision encoder's parameters are included so the whole policy is covered.
    """
    decay, no_decay = set(), set()
    whitelist = (nn.Linear, nn.MultiheadAttention, nn.Conv1d, nn.Conv2d)
    blacklist = (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d, nn.Embedding)
    for mn, m in model.named_modules():
        for pn, _ in m.named_parameters(recurse=False):
            fpn = f"{mn}.{pn}" if mn else pn
            if pn.endswith("bias") or pn.startswith("bias"):
                no_decay.add(fpn)
            elif isinstance(m, whitelist):
                decay.add(fpn)
            elif isinstance(m, blacklist):
                no_decay.add(fpn)
    named = dict(model.named_parameters())
    for n in named:
        if n.endswith("pos_emb") or n.endswith("cond_pos_emb") or n.endswith("_dummy_variable"):
            no_decay.add(n)
    leftover = named.keys() - decay - no_decay
    no_decay |= leftover                      # anything unclassified is safest undecayed
    overlap = decay & no_decay
    if overlap:
        raise RuntimeError(f"parameters in both buckets: {sorted(overlap)[:5]}")
    return [
        {"params": [named[n] for n in sorted(decay)], "weight_decay": weight_decay},
        {"params": [named[n] for n in sorted(no_decay)], "weight_decay": 0.0},
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Registration, so `lerobot-train`'s whole pipeline (dataset, wandb, checkpoints,
# resume, eval) can be reused with `--policy.type=diffusion_transformer`.
#
# lerobot's `get_policy_class` is a hardcoded if/elif chain with no plugin hook,
# so it is patched below. Everything else is stock lerobot.
# ─────────────────────────────────────────────────────────────────────────────

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy


@PreTrainedConfig.register_subclass("diffusion_transformer")
@dataclass
class DiffusionTransformerConfig(DiffusionConfig):
    """Diffusion Policy with the transformer backbone. Defaults are Table 8's.

    🚨 THE DEFAULTS DIFFER FROM THE CNN'S ON PURPOSE. Table 7 vs Table 8:
          weight_decay   CNN 1e-6   ->  transformer 1e-3   (1000x)
          attn dropout   CNN none   ->  transformer 0.3
    Table 8's caption even reads "WDecay: weight decay (for transformer only)".
    §3.1 warns the transformer is "more sensitive to hyperparameters" and
    recommends trying the CNN first. Inheriting 1e-6 here is the most likely way
    to reproduce the 4x overfit measured on the CNN run (experiments/2026-08-12).

    n_layer/n_emb are Table 8's simulation rows (8 layers, 256 wide -> 9.0M
    backbone params, matching its "#D-params 9"). Kitchen and Real Push-T use
    768 instead, which measures 80.5M against Table 8's "80".
    """

    n_layer: int = 8
    n_head: int = 4              # not listed in Table 8; the reference wrapper's default
    n_emb: int = 256
    p_drop_emb: float = 0.0
    p_drop_attn: float = 0.3     # Table 8, most rows
    causal_attn: bool = True     # reference wrapper default
    n_cond_layers: int = 0       # 0 -> the cond encoder is an MLP, not a transformer

    optimizer_weight_decay: float = 1e-3          # Table 8, NOT the CNN's 1e-6
    optimizer_betas: tuple = (0.9, 0.95)          # reference configure_optimizers()


class DiffusionTransformerPolicy(DiffusionPolicy):
    config_class = DiffusionTransformerConfig
    name = "diffusion_transformer"

    def __init__(self, config: DiffusionTransformerConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.diffusion = DiffusionTransformerModel(
            config,
            n_layer=config.n_layer,
            n_head=config.n_head,
            n_emb=config.n_emb,
            p_drop_emb=config.p_drop_emb,
            p_drop_attn=config.p_drop_attn,
            causal_attn=config.causal_attn,
            n_cond_layers=config.n_cond_layers,
        )

    def get_optim_params(self):
        """Return param GROUPS, not a flat list, so the no-decay split survives.

        lerobot builds the optimizer as `cfg.optimizer.build(policy.get_optim_params())`
        and torch optimizers accept an iterable of dicts, where a per-group
        `weight_decay` overrides the optimizer-level one. Without this override
        lerobot would apply 1e-3 uniformly -- including to `pos_emb`, whose 48
        rows are the ONLY thing distinguishing timestep 5 from timestep 40, and
        to every LayerNorm gain.
        """
        return optim_groups(self.diffusion, self.config.optimizer_weight_decay)


def _register() -> None:
    """Teach lerobot's factory about the new policy type (no plugin hook exists)."""
    from lerobot.policies import factory

    _orig = factory.get_policy_class

    def patched(name: str):
        if name == "diffusion_transformer":
            return DiffusionTransformerPolicy
        return _orig(name)

    if getattr(factory.get_policy_class, "__name__", "") != "patched":
        factory.get_policy_class = patched


_register()

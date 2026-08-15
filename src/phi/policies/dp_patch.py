"""Diffusion Policy with DENSE FROZEN ViT PATCHES as the cross-attention memory.

The denoiser is byte-identical to `dp_transformer`. The ONLY thing that changes is
what reaches it: instead of 3 ResNet-18s pooled to 64 numbers each, the decoder gets
every DINOv2 patch token.

    DP-T      3x ResNet-18 -> 1x1 conv -> SpatialSoftmax -> 64/cam ->    3 memory tokens
    DP-Patch  DINOv2 ViT-S/14 (frozen) -> every patch kept       -> 1803 memory tokens

Measured, per camera per frame: ResNet-18 emits (512,7,9) = 32,256 numbers and
SpatialSoftmax discards 99.8% of them to keep 64. DINOv2 emits 300x384 = 115,200 and
keeps all of them. That is the entire hypothesis under test.

    trainable   DP-T 42,612,198  ->  DP-Patch 9,532,038   (4.5x fewer)
    frozen      DP-T          0  ->  DP-Patch 22,056,576

Paper: Patch Policy, arXiv 2607.18236. Code: github.com/gaoyuezhou/patch_policy (MIT).
Their diffusion head is the SAME real-stanford TransformerForDiffusion we vendored, at
the same geometry (n_layer=8, n_head=4, n_emb=256), so this is a port, not a redesign.

🚨 THREE TRAPS, ALL MEASURED -- see docs/theory/diffusion-policy-patch.md
  1. `crop_ratio` MUST be 0.875. DINOv2's PatchEmbed asserts `H % 14 == 0`; it does not
     pad and it does not drop the remainder, it refuses to run. 240x320 * 0.875 =
     210x280 = exactly 15x20 = 300 patches. The DP-T control's 0.9 -> 216x288 CRASHES.
  2. `VISUAL` normalization MUST be IDENTITY. DINOv2 asserts `0 <= x <= 1` and applies
     its own ImageNet stats. lerobot's default MEAN_STD would feed it garbage.
  3. `causal_attn=True` REQUIRES the patch-aware memory mask (now in
     transformer_for_diffusion.py). The stock `t >= s-1` rule compares an action
     timestep against a flattened (step, camera, patch) slot index -- at T_cond=1803 it
     blocks 98.59% of cells and strands 1754 of 1803 memory tokens. A hard check below
     refuses to build a model that would silently do that.
"""

from __future__ import annotations

from dataclasses import dataclass

import einops
import torch
import torch.nn as nn
import torchvision

from lerobot.configs.policies import PreTrainedConfig
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE
from lerobot.policies.diffusion.modeling_diffusion import DiffusionModel, DiffusionPolicy

from phi.policies.dp_transformer import DiffusionTransformerConfig, DiffusionTransformerPolicy
from phi.policies.transformer_for_diffusion import TransformerForDiffusion

# DINOv2's own preprocessing constants (models/encoder/dino.py in the reference).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class PatchEncoder(nn.Module):
    """Frozen DINOv2 over lerobot's resize/crop, returning EVERY patch token.

    Resize and crop are reproduced from `DiffusionRgbEncoder` verbatim (including
    `torchvision.transforms.Resize`'s antialias default) so the ONLY difference from
    the DP-T control after the crop is the backbone. Deliberately not switching to
    `mode='area'` here: it is a fidelity improvement, but it would be a second
    uncontrolled variable in an A/B whose whole point is the encoder swap.
    """

    def __init__(self, config: "DiffusionPatchConfig") -> None:
        super().__init__()
        if config.resize_shape is not None:
            self.resize = torchvision.transforms.Resize(config.resize_shape)
        else:
            self.resize = None

        crop_shape = config.crop_shape
        if crop_shape is not None:
            self.do_crop = True
            self.center_crop = torchvision.transforms.CenterCrop(crop_shape)
            self.maybe_random_crop = (
                torchvision.transforms.RandomCrop(crop_shape)
                if config.crop_is_random
                else self.center_crop
            )
            h, w = crop_shape
        else:
            self.do_crop = False
            h, w = config.resize_shape

        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        self.backbone = torch.hub.load(config.dino_repo, config.dino_model, verbose=False)
        patch = self.backbone.patch_size
        if h % patch or w % patch:
            raise ValueError(
                f"DINOv2 patch size is {patch} and the post-crop image is {h}x{w}, which is "
                f"not divisible ({h}%{patch}={h % patch}, {w}%{patch}={w % patch}). PatchEmbed "
                f"ASSERTS on this -- it will not pad or truncate. With resize_shape="
                f"{tuple(config.resize_shape)} use crop_ratio=0.875 to land on 210x280."
            )
        # `pool` is THE ablation: it holds the backbone, the resize, the crop and every
        # downstream tensor fixed and changes only the bottleneck. Mirrors the reference's
        # configs/encoder/dino_patch{,_avg_pool,_cls}.yaml.
        self.pool = config.patch_pool
        self.n_patches = 1 if self.pool != "none" else (h // patch) * (w // patch)
        self.feature_dim = self.backbone.embed_dim

        # Frozen means frozen: no grad, and never leaves eval mode (see `train` below).
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        self.register_buffer("_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def train(self, mode: bool = True):
        """Keep the backbone in eval forever; only the crop's randomness follows `mode`."""
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(N, C, H, W) in [0, 1]  ->  (N, n_patches, feature_dim)."""
        if self.resize is not None:
            x = self.resize(x)
        if self.do_crop:
            x = self.maybe_random_crop(x) if self.training else self.center_crop(x)
        # DINOv2 asserts this. Failing loudly here beats silently degraded features.
        if x.min() < -1e-3 or x.max() > 1 + 1e-3:
            raise ValueError(
                f"PatchEncoder expects pixels in [0,1], got [{x.min():.3f}, {x.max():.3f}]. "
                "Set normalization_mapping VISUAL=IDENTITY -- MEAN_STD breaks DINOv2."
            )
        x = (x.clamp(0, 1) - self._mean) / self._std
        with torch.no_grad():
            feats = self.backbone.forward_features(x)
        if self.pool == "avg":
            # Mean over the patch axis: keeps WHAT is in the frame, discards WHERE.
            # dino.py:54 `emb = torch.mean(emb, dim=-2)`.
            return feats["x_norm_patchtokens"].mean(dim=-2, keepdim=True)
        if self.pool == "cls":
            # The CLS token instead -- a learned summary rather than an average.
            return feats["x_norm_clstoken"].unsqueeze(1)
        return feats["x_norm_patchtokens"]


class PatchBackbone(nn.Module):
    """`TransformerForDiffusion` behind the UNet's call signature, WITHOUT the reshape.

    `dp_transformer`'s adapter has to un-flatten lerobot's `(B, To*obs_feat)`. Here
    `_prepare_global_conditioning` is overridden to hand back `(B, S, D)` already, so
    there is nothing to undo.
    """

    def __init__(self, config, cond_dim: int, tokens_per_step: int, **kw) -> None:
        super().__init__()
        self.net = TransformerForDiffusion(
            input_dim=config.action_feature.shape[0],
            output_dim=config.action_feature.shape[0],
            horizon=config.horizon,
            n_obs_steps=config.n_obs_steps,
            cond_dim=cond_dim,
            n_emb=kw["n_emb"],
            n_layer=kw["n_layer"],
            n_head=kw["n_head"],
            p_drop_emb=kw["p_drop_emb"],
            p_drop_attn=kw["p_drop_attn"],
            causal_attn=kw["causal_attn"],
            n_cond_layers=kw["n_cond_layers"],
            time_as_cond=True,
            obs_as_cond=True,
            n_patches=tokens_per_step,
        )

    def forward(self, x, timestep, global_cond=None):
        if global_cond is None:
            raise ValueError("the patch backbone requires global_cond")
        if global_cond.ndim != 3:
            raise ValueError(f"expected unflattened (B,S,D) conditioning, got {global_cond.shape}")
        return self.net(x, timestep, global_cond)


class DiffusionPatchModel(DiffusionModel):
    """`DiffusionModel` with the encoder AND the denoiser replaced.

    `compute_loss`, `conditional_sample` and `generate_actions` are inherited untouched,
    so the diffusion process is identical to every other arm. Verified safe: `global_cond`
    is pure pass-through -- produced in `generate_actions`/`compute_loss`, consumed only
    by `self.unet(...)`. Nothing indexes it, so returning 3-D instead of 2-D is fine.
    """

    def __init__(self, config: "DiffusionPatchConfig", **backbone_kwargs) -> None:
        super().__init__(config)
        # The parent built 3 ResNet-18s to compute a global_cond_dim we do not use.
        # Cheap and transient, and it keeps us from forking lerobot's __init__.
        del self.rgb_encoder
        del self.unet

        self.rgb_encoder = PatchEncoder(config)
        self.n_cameras = len(config.image_features)
        d = self.rgb_encoder.feature_dim
        # State rides as ONE MORE TOKEN PER TIMESTEP rather than a separate block, so the
        # causal memory mask -- which groups tokens into per-timestep windows -- gives it
        # exactly the same treatment as the patches recorded at that instant.
        self.state_emb = nn.Linear(config.robot_state_feature.shape[0], d)
        self.tokens_per_step = self.n_cameras * self.rgb_encoder.n_patches + 1

        if config.causal_attn and self.rgb_encoder.n_patches > 1:
            probe = TransformerForDiffusion(
                input_dim=1, output_dim=1, horizon=config.horizon,
                n_obs_steps=config.n_obs_steps, cond_dim=d, n_layer=1, n_head=1, n_emb=8,
                causal_attn=True, time_as_cond=True, obs_as_cond=True,
                n_patches=self.tokens_per_step,
            )
            reachable = int((probe.memory_mask == 0.0).any(dim=0).sum())
            if reachable != probe.T_cond:
                raise RuntimeError(
                    f"causal memory mask strands {probe.T_cond - reachable} of {probe.T_cond} "
                    "memory tokens. The patch-aware mask in transformer_for_diffusion.py is "
                    "missing or wrong -- refusing to train a model that cannot see its input."
                )

        self.unet = PatchBackbone(config, d, self.tokens_per_step, **backbone_kwargs)

    def _prepare_global_conditioning(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """(B, To, cams, C, H, W) + (B, To, state) -> (B, To*(cams*P + 1), D), UNFLATTENED."""
        b, to = batch[OBS_STATE].shape[:2]
        imgs = einops.rearrange(batch[OBS_IMAGES], "b t n ... -> (b t n) ...")
        patches = self.rgb_encoder(imgs)                                    # (b*t*n, P, D)
        patches = einops.rearrange(patches, "(b t n) p d -> b t (n p) d", b=b, t=to)
        state = self.state_emb(batch[OBS_STATE]).unsqueeze(2)               # (b, t, 1, D)
        tokens = torch.cat([patches, state], dim=2)                         # (b, t, n*P+1, D)
        return einops.rearrange(tokens, "b t p d -> b (t p) d")


@PreTrainedConfig.register_subclass("diffusion_patch")
@dataclass
class DiffusionPatchConfig(DiffusionTransformerConfig):
    """Inherits DP-T's Table 8 defaults; overrides only what DINOv2 forces.

    `crop_ratio` and the VISUAL normalization are NOT free choices here -- see the three
    traps in the module docstring. `__post_init__` refuses the combinations that would
    fail late or, worse, train quietly on broken features.
    """

    dino_repo: str = "facebookresearch/dinov2:b48308a"
    dino_model: str = "dinov2_vits14"

    # "none" = every patch token (dense, the paper's method)
    # "avg"  = mean over patches -> 1 token/camera   (their dino_patch_avg_pool)
    # "cls"  = the CLS token instead -> 1 token/cam  (their dino_cls)
    # This is the ONLY knob that isolates DENSITY: backbone, resize, crop and decoder
    # are byte-identical across the three, so a difference cannot be attributed to
    # anything else. Compare against the ResNet arms with care -- SpatialSoftmax keeps
    # 32 keypoint COORDINATES per camera, so it is a spatial bottleneck, not a global one.
    patch_pool: str = "none"

    crop_ratio: float = 0.875          # -> 210x280 -> exactly 15x20 patches, no remainder
    resize_shape: tuple = (240, 320)   # same as the DP-T control, so only the crop differs
    causal_attn: bool = False          # measured 2.67x better + no overfitting on our data
    optimizer_weight_decay: float = 1e-6   # matches the `unreg`/`bidir` arms, not Table 8
    p_drop_attn: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        if self.patch_pool not in ("none", "avg", "cls"):
            raise ValueError(
                f"patch_pool must be one of 'none' | 'avg' | 'cls', got {self.patch_pool!r}. "
                "A typo here would silently fall through to dense patches and quietly "
                "invalidate the density ablation."
            )
        visual = self.normalization_mapping.get("VISUAL")
        if visual is not None and getattr(visual, "name", str(visual)) != "IDENTITY":
            raise ValueError(
                f"normalization_mapping VISUAL must be IDENTITY for DINOv2, got {visual}. "
                "DINOv2 asserts pixels in [0,1] and applies ImageNet stats itself; lerobot's "
                "MEAN_STD would hand it out-of-range values and silently degrade the features."
            )


class DiffusionPatchPolicy(DiffusionPolicy):
    config_class = DiffusionPatchConfig
    name = "diffusion_patch"

    def __init__(self, config: DiffusionPatchConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.diffusion = DiffusionPatchModel(
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
        """DP-T's no-decay split, minus the 22M frozen parameters.

        Without the `requires_grad` filter AdamW would allocate exp_avg/exp_avg_sq for
        every frozen DINOv2 weight -- ~176 MB of optimizer state that can never update.
        """
        from phi.policies.dp_transformer import optim_groups

        groups = optim_groups(self.diffusion, self.config.optimizer_weight_decay)
        for g in groups:
            g["params"] = [p for p in g["params"] if p.requires_grad]
        return groups


def _register() -> None:
    """Same patch-the-factory trick as dp_transformer; lerobot still has no plugin hook.

    🚨 The sentinel name must differ from dp_transformer's. Both modules guard with
    `__name__ != "patched"`, so reusing that name means importing dp_transformer first
    makes THIS registration a silent no-op and `--policy.type=diffusion_patch` dies with
    a DecodingError that points nowhere near the cause.
    """
    from lerobot.policies import factory

    _orig = factory.get_policy_class

    def patched_with_diffusion_patch(name: str):
        if name == "diffusion_patch":
            return DiffusionPatchPolicy
        return _orig(name)

    if getattr(factory.get_policy_class, "__name__", "") != "patched_with_diffusion_patch":
        factory.get_policy_class = patched_with_diffusion_patch


_register()

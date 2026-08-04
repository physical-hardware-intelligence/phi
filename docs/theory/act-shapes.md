# ACT, shape by shape

What every tensor looks like from dataloader to action, and which of those shapes decides your batch size. Everything here was **traced from the installed `lerobot` 0.6.0**, not read off the paper — reproduce it yourself with the snippet at the bottom.

Config traced: **2 cameras @ 640×480**, `state_dim=6`, `action_dim=6`, `chunk_size=50`, `batch_size=8`, `dim_model=512`. This is the Φ lab's rig ([01-hardware](../robots/so-arm101/01-hardware.md#what-this-lab-actually-runs)); ACT's full default table lives in [the wrist-only experiment §5.1](../../experiments/2026-07-30_act-wrist-only-so101.md).

## The trace

| Stage | Module | Output |
|---|---|---|
| Dataloader | — | `state (8,6)` · `images 2×(8,3,480,640)` · `action (8,50,6)` · `action_is_pad (8,50)` |
| VAE state proj | `Linear 6→512` | `(8, 512)` |
| VAE action proj | `Linear 6→512` | `(8, 50, 512)` |
| **VAE encoder** | 4 layers, self-attn | `(52, 8, 512)` ← `cls + state + 50 actions` |
| Latent params | `Linear 512→64` | `(8, 64)` → `mu (8,32)`, `log_sigma_x2 (8,32)` |
| Reparameterize | — | `latent_sample (8, 32)` |
| Backbone stem | `conv1` s2 → `maxpool` s2 | `(8,64,240,320)` → `(8,64,120,160)` |
| Backbone layer4 | ResNet18 | `(8, 512, 15, 20)` |
| Feature proj | `Conv2d 512→512` 1×1 | `(8, 512, 15, 20)` |
| Flatten `(h w) b c` | — | `(300, 8, 512)` per camera |
| **Encoder** | 4 layers, self-attn | **`(602, 8, 512)`** |
| **Decoder** | 1 layer, cross-attn | `(50, 8, 512)` |
| Action head | `Linear 512→6` | **`(8, 50, 6)`** |

**51.57 M params**: 11.17 M backbone + 40.40 M transformer and heads.

> ⚠️ **Batch is the *middle* dimension inside the transformer.** ACT is batch-first at the boundaries and **sequence-first** in between, because `nn.MultiheadAttention` defaults to `batch_first=False`. In `(602, 8, 512)` the `8` is your batch; `602` and `50` are sequence lengths. Misread this and you will think the model emits 50 batches.

## Where 602 comes from

```
ResNet18 output stride = 32        3×480×640 → 512×15×20
15 × 20 = 300 tokens per camera
300 + 300 + 1 latent + 1 state = 602
```

**ACT contains no resize, crop or interpolate** — grep `policies/act/` and you get zero hits. It feeds the backbone your native resolution. Every other policy in the [zoo](../training/README.md) downsizes (VQ-BeT 84², pi0 224², SmolVLA 512²); ACT alone does not.

| Cameras @ 640×480 | Encoder length | Attn matrix / sample / layer |
|---|---|---|
| 1 | 302 | 2.9 MB |
| **2** | **602** | 11.6 MB |
| 3 | 902 | 26.0 MB |

At 224×224 the same 2-camera setup gives `7×7=49` per camera → **100 tokens, 6× shorter**.

## Where the memory actually goes

Per camera, batch 8, fp32:

| Stage | Shape per sample | @ B=8 |
|---|---|---|
| input | `(3, 480, 640)` | 29.5 MB |
| **conv1** | `(64, 240, 320)` | **157.3 MB** |
| bn1 | `(64, 240, 320)` | 157.3 MB |
| relu | `(64, 240, 320)` | 157.3 MB |
| maxpool | `(64, 120, 160)` | 39.3 MB |
| layer1 | `(64, 120, 160)` | 39.3 MB |
| layer2 | `(128, 60, 80)` | 19.7 MB |
| layer3 | `(256, 30, 40)` | 9.8 MB |
| layer4 | `(512, 15, 20)` | 4.9 MB |

**`conv1 + bn1 + relu` = 472 MB, ~81% of the 585 MB per-camera total.** Two cameras = 1,170 MB, versus **371 MB** for all four encoder attention matrices combined.

⚠️ **So the ResNet stem, not attention, is what fills the GPU.** `conv1` holds 32× the activation memory of `layer4` — it is still at half resolution with 64 channels. The 602 is the memorable number; the stem is the expensive one. Resizing inputs helps mostly by shrinking the stem.

`conv1` is `Conv2d(3, 64, 7×7, stride 2, pad 3)`, 9,408 params. It lifts RGB to 64 channels, takes the first 2× downsample, and uses a wide 7×7 receptive field to grab edges and colour gradients cheaply before the residual blocks. Its BatchNorm is **frozen** (`FrozenBatchNorm2d`) so small training batches don't inject noise through BN statistics.

## Measured VRAM, and how to pick a batch size

Fitted from batch 1/2/4 (MPS, fp32, 2 cameras):

```
≈ 1.04 GB fixed + 0.428 GB per sample
```

| GPU | max batch @ 90% VRAM |
|---|---|
| 24 GB | **~47** |
| A100 40 GB | ~81 |
| A100 80 GB | ~165 |
| H200 141 GB | ~293 |

> ⚠️ **This is an MPS fp32 extrapolation, not a CUDA measurement.** CUDA's allocator, cuDNN workspaces and bf16/AMP all shift it. Use it to size `--mem` and a first `--batch_size`, then confirm with one short run on the real GPU.

The 24 GB row explains a failure you will read about in other people's writeups: **ACT at 640×480 with 2 cameras OOMs at `batch_size=64` on 24 GB.** The fix people reach for is halving the batch, which then needs 2× the steps for the same sample count. That is arithmetic, not an ACT defect — and on Explorer's H200s the constraint does not apply.

## What `chunk_size` does and does not change

Actions never enter the encoder, so the expensive part is **invariant**:

| chunk | VAE seq | encoder seq | params | peak GB (B=2) | open-loop @ 30 fps |
|---|---|---|---|---|---|
| 50 | 52 | **602** | 51.57 M | 1.70 | 1.67 s |
| 75 | 77 | **602** | 51.58 M | 1.71 | 2.50 s |
| 100 | 102 | **602** | 51.60 M | 1.73 | 3.33 s |

**One memory request covers every chunk value.** What the axis actually trades is *reactivity*: with `n_action_steps = chunk_size` the arm commits to the whole chunk before looking again.

⚠️ **A confound to control.** L1 loss is masked by `action_is_pad`, and each episode contributes exactly `chunk−1` partially-padded samples. Over a 89-episode / 54,770-frame split that is **8.0% (chunk 50) · 12.0% (75) · 16.1% (100)** of samples with reduced supervision. Larger chunks get measurably less effective supervision near episode ends — don't attribute that to horizon length.

## Two structural facts worth knowing

**The decoder queries are literally zeros.** `decoder_in = torch.zeros(chunk_size, B, 512)`. All content arrives via learned positional embeddings and cross-attention: query slot *k* asks the 602 observation tokens what to do at step *k*.

**`n_decoder_layers = 1`, deliberately.** LeRobot's config comment cites [tonyzhaozh/act#25](https://github.com/tonyzhaozh/act/issues/25) — the paper says 7 but only the first was ever used, so LeRobot matches real behaviour. ACT is a deep encoder with a **single** cross-attention layer.

**The VAE branch is training-only.** At inference `latent_sample` is **zeros**, so the CVAE never samples at test time — it exists purely to absorb demonstrator multimodality during training. Four of the nine transformer layers do not run on the robot.

## Reproduce it

```python
from lerobot.policies.act.modeling_act import ACTPolicy
# register forward hooks on policy.model.named_modules(), run policy.forward(batch),
# print each module's output shape. Token count = 300*n_cameras + 2 at 640x480.
```
Token arithmetic needs no GPU: `(480//32) * (640//32) = 300` per camera, plus one latent and one state token.

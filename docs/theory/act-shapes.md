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

## Why the backbone is in memory at all

The backbone is **pretrained, not frozen** — a distinction worth being clear about, because it is the reason activations cost anything.

ACT initializes from `ResNet18_Weights.IMAGENET1K_V1` and then **fine-tunes all of it**:

```
backbone params      : 11,166,912
  requires_grad=True : 11,166,912
  requires_grad=False: 0
optimizer groups     : 2 -> [40,404,678 main] + [11,166,912 backbone @ optimizer_lr_backbone=1e-5]
```

Gradients flow through every convolution, so autograd must retain each layer's activations for the backward pass. ImageNet features classify objects; manipulation needs precise spatial localization of *your* object in *your* lighting, so the low-LR fine-tune is deliberate.

**What is frozen is the BatchNorm.** 20 `FrozenBatchNorm2d` modules, whose `weight`/`bias` are registered **buffers, not Parameters** — normalization statistics stay fixed so small training batches can't destabilize them. The DETR recipe.

`conv1` is `Conv2d(3, 64, 7×7, stride 2, pad 3)`, 9,408 params: lifts RGB to 64 channels, takes the first 2× downsample, and uses a wide receptive field to grab edges and colour gradients cheaply before the residual blocks.

Freezing the backbone (`requires_grad_(False)`) measures at **5.118 → 3.731 GB at batch 8, a 27% saving** — only 0.13 GB of which is freed optimizer state; the rest is activations no longer needed. A frozen backbone would also let you **precompute features once** and skip the ResNet during training entirely. That is exactly [Patch Policy](../training/README.md)'s design with a frozen DINOv2 — a real experiment to run, but a deviation from the ACT baseline, so not something to do to a baseline run.

## Where the backbone memory goes

Measured by hooking **all 59 leaf modules** and summing `out.numel() × batch × 4 bytes`. Per camera, batch 8, fp32:

| Group | Tensors at | MB @ B=8 | Share |
|---|---|---|---|
| conv1 | 240×320 | 157.3 | 10.7% |
| bn1 | 240×320 | 157.3 | 10.7% |
| relu | 240×320 | 157.3 | 10.7% |
| maxpool | 120×160 | 39.3 | 2.7% |
| **layer1** | 120×160 | **471.9** | **32.2%** |
| layer2 | 60×80 | 275.3 | 18.8% |
| layer3 | 30×40 | 137.6 | 9.4% |
| layer4 | 15×20 | 68.8 | 4.7% |

**Resolution dominates, not any single layer.** Everything at ≥120×160 — the stem plus `layer1` — is **67%** of backbone activations; `layer3 + layer4` together are 14%. `layer1` alone beats the entire stem because it runs 8 more tensors at 120×160 (each `BasicBlock` is conv→bn→relu→conv→bn→relu). This is why *resizing inputs* is the lever, not swapping the backbone.

Compare with attention: all four encoder attention matrices at B=8 total **371 MB**, against 2,929 MB of backbone leaf activations for 2 cameras. **The convolutional frontend, not the 602-token attention, is what fills the GPU.**

> ⚠️ **That 2,929 MB is an upper bound.** torchvision's ResNet uses `nn.ReLU(inplace=True)`, so ReLU outputs overwrite their input and cost nothing extra, and conv/BN pairs share buffers. The measured saving from freezing the backbone is 1,387 MB at B=8 — roughly half the naive sum. Treat the per-tensor table as *where* memory goes, and the measured deltas below as *how much*.

## Measured VRAM, and how to pick a batch size

Measured **one batch size per process** — this matters, see the warning below. MPS, fp32, 2 cameras, chunk 50:

| Batch | GB | Marginal per sample |
|---|---|---|
| 1 | 1.469 | — |
| 2 | 1.701 | 0.232 |
| 4 | 2.695 | 0.497 |
| 8 | 5.118 | 0.606 |

Strongly nonlinear at small batch, where fixed cost dominates. Fitting B = 2, 4, 8:

```
≈ 0.49 GB fixed + 0.575 GB per sample
```

| GPU | max batch @ 90% VRAM |
|---|---|
| 24 GB | **~36** |
| A100 40 GB | ~61 |
| A100 80 GB | ~124 |
| H200 141 GB | ~219 |

> 🚨 **Do not measure several batch sizes in one process.** `torch.mps.driver_allocated_memory()` (and CUDA's `memory_allocated`) report a **monotonic high-water mark**, so the second reading inherits the first's cached blocks. An earlier version of this page reported `1.04 GB + 0.428 GB/sample` and a 24 GB ceiling of ~47 for exactly this reason. **One process per data point.**

> ⚠️ **This is an MPS fp32 extrapolation, not a CUDA measurement.** CUDA's allocator, cuDNN workspaces and bf16/AMP all shift it. Use it to size `--mem` and a first `--batch_size`, then confirm with one short run on the real GPU.

The 24 GB row explains a failure you will read about in other people's writeups: **ACT at 640×480 with 2 cameras OOMs at `batch_size=64` on 24 GB.** The fix people reach for is halving the batch, which then needs 2× the steps for the same sample count. That is arithmetic, not an ACT defect — and on Explorer's H200s the constraint does not apply.

## Should you resize before the backbone?

ACT has no resize, but you can add one — and it is the single biggest lever on both memory and speed. Measured end-to-end training steps (2 cameras, chunk 50, batch 8, MPS fp32):

| Input | layer4 grid | Tokens/cam | 2-cam encoder | Backbone act/cam | Step time |
|---|---|---|---|---|---|
| **480×640** | 15×20 | 300 | **602** | 1,465 MB | 3,229 ms · 0.31 it/s |
| 360×480 | 12×15 | 180 | 362 | 828 MB | — |
| **240×320** | 8×10 | 80 | **162** | 367 MB | **931 ms · 1.07 it/s** |
| 224×224 | 7×7 | 49 | 100 | 239 MB | **622 ms · 1.61 it/s** |
| 120×160 | 4×5 | 20 | 42 | 92 MB | — |

Halving to 240×320 gives **3.7× fewer tokens, 2.3× less memory, 3.5× faster steps**.

> ⚠️ `H//32` only gives the grid when H and W are multiples of 32. 240×320 yields **8×10, not 7×10**, because conv padding rounds up. Read the grid off this table.

**How to do it.** LeRobot's transform factory accepts any `torchvision.transforms.v2` class, so the augmentation pipeline can be coerced into a deterministic resize: set `enable=true`, `max_num_transforms=1`, and a single `tfs` entry of `type: Resize`. It is deterministic because weight-0 transforms are skipped and `n_subset = min(len(tfs), max_num_transforms) = 1`, so the one transform is always drawn. Verified: 50/50 draws returned `(3, 240, 320)`.

> 🚨 **Never list Resize alongside other transforms.** With resize as 1-of-4 and `max_num_transforms=3`, draws returned *both* `(3,240,320)` and `(3,480,640)` — mixed shapes in one batch crash collation. Safe only when every other weight is 0. (Which means **resize and augmentation are mutually exclusive** through this mechanism.)
>
> Not yet verified: whether draccus accepts a new `tfs` key from the command line, since the default dict has no `resize` entry. Prefer a config file.

### 🚨 A resolution mismatch at inference does not raise

Build a policy for 240×320 and feed it something else:

```
trained@240x320, fed 240x320: RAN, out (1, 50, 6)
trained@240x320, fed 480x640: RAN, out (1, 50, 6)   <- 602 tokens instead of 162
trained@240x320, fed 224x224: RAN, out (1, 50, 6)
```

**All of them run silently and return a plausible action chunk.** The camera positional embedding is sinusoidal and generated *from the feature map*, so it adapts to any grid; the learned 1D embedding is only 2 rows (latent + state), independent of image size. There is no shape assertion anywhere in the path — the code even comments that "H and W may vary but H\*W is constant."

So a policy trained resized and rolled out un-resized operates far outside its training distribution **with no error and no warning**. It just moves badly. Two ways to match:

1. **Apply the identical resize in the rollout pipeline.** Correct by construction.
2. **Capture at that resolution** (`--robot.cameras="{ top: {..., width: 320, height: 240} }"`), which also cuts USB bandwidth — see [02-setup §5c](../robots/so-arm101/02-setup.md#5c-bandwidth-set-fourcc-mjpg). ⚠️ But **many UVC webcams crop rather than scale at lower resolutions**, changing the field of view — then it is not the same scene and matching resolution does not save you. Check with `phi.utils.camera_align` first; if the FOV shifts, option 1 is the only correct path.

### When to reach for it
**Not for a baseline run.** It deviates from the ACT configuration the literature reports, and on an 80–141 GB GPU memory is not the constraint. There is also a specific reason to expect an accuracy cost: a token covers 32×32 px of *input* either way, so at 240×320 each token spans **64×64 px of original scene** — the overhead camera's spatial precision halves, which is exactly what a held-out-position eval measures.

Reach for it when **wall-clock** binds — Explorer's `gpu` partition caps around 8 h, and 3.5× decides whether a run fits its window. Do it as a clean extra run at the best chunk size, after the baseline, and write it up as an ablation in [`experiments/`](../../experiments/).

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

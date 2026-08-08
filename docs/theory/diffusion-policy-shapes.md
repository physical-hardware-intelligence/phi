# Diffusion Policy, shape by shape

Every tensor from dataloader to action, traced from the installed `lerobot` 0.6.0 — not read off the paper. Companion to [act-shapes](act-shapes.md); the comparison table at the end is the short version.

> 📖 **Read [diffusion-policy-why](diffusion-policy-why.md) first if you want the argument rather than the trace.** It derives the whole architecture from one problem (regression averages multimodal demonstrations into invalid actions) and explains why noise, why 100 steps, and why the multimodality lives in sampling instead of in a latent. This page is the reference; that page is the story.

Config traced: **LeRobot defaults**, 2 cameras @ 3×480×640, state/action dim 6, batch 8.

> Needs the optional extra: `pip install "lerobot[diffusion]==0.6.0"` (pulls `diffusers>=0.27.2,<0.36.0`). Without it `DiffusionPolicy` will not import.

## The framing that makes the rest make sense

ACT is a **function**: observation in, action chunk out, one pass.

Diffusion Policy is **a learned repair operation on trajectories**. Input and output have the *same* shape — `(8, 64, 6)` in, `(8, 64, 6)` out. It takes a corrupted 64-step trajectory and predicts which part of it is garbage. The observation never enters as content; it is a **global style knob** on that repair operation.

Everything odd about the architecture follows: why the image is crushed to 64 numbers, why convolution runs along *time*, why the same shape appears at both ends, and why inference runs the network 100 times.

## Two time axes. Never confuse them.

| Symbol | Size | Meaning |
|---|---|---|
| `B` | 8 | batch |
| **`S`** | **2** | **observation steps — the PAST** (`n_obs_steps`) |
| `C,H,W` | 3,480,640 | colour and space |
| **`T`** | **64** | **horizon — the FUTURE** (`horizon`), the thing being denoised |
| `A` | 6 | action dims (six joints) |

`S` is history and gets **destroyed early**. `T` is prediction and **survives to the output**.

## The measured trace

| Stage | Module | Output |
|---|---|---|
| Dataloader | — | `images 2×(8,2,3,480,640)` · `state (8,2,6)` · `action (8,64,6)` |
| Fold `S` into `B` | `rearrange "b s n ... -> n (b s) ..."` | `(16, 3, 480, 640)` per camera |
| Backbone | ResNet18, **one per camera** | `(16, 512, 15, 20)` |
| Spatial softmax | `Conv2d 512→32` + soft-argmax | `(16, 32, 2)` |
| Projection | `Linear 64→64` + ReLU | `(16, 64)` |
| Global condition | concat state + both cams, flatten `S` | **`(8, 268)`** |
| Step embedding | sinusoidal → `128→512→Mish→128` | `(8, 128)` |
| Condition vector | concat | **`(8, 396)`** |
| **Axis flip** | `rearrange "b t d -> b d t"` | `(8, 6, 64)` |
| down 0 | 2× residual, `Conv1d(k=3,s=2)` | `(8,512,64)` → `(8,512,32)` |
| down 1 | 2× residual, `Conv1d(k=3,s=2)` | `(8,1024,32)` → `(8,1024,16)` |
| down 2 | 2× residual, `Identity` | `(8, 2048, 16)` |
| mid | 2× residual | `(8, 2048, 16)` |
| up 0 | `cat` skip → 4096, 2× residual, `ConvTranspose1d` | `(8,1024,16)` → `(8,1024,32)` |
| up 1 | `cat` skip → 2048, 2× residual, `ConvTranspose1d` | `(8,512,32)` → `(8,512,64)` |
| final | `Conv1dBlock(k=5)` + `Conv1d(512→6, k=1)` | **`(8, 64, 6)`** = predicted noise |

**277.82 M params**: `unet` **255.43 M** + `rgb_encoder` 22.39 M (2 × 11.2, **no sharing**). 92% of the model generates; 8% sees.

## The observation path

**Folding `S` into `B` is free.** The encoder is a function of one image with no opinion about batch or time, so merging the axes lets one conv pass handle all 16 images. No information crosses.

`use_separate_rgb_encoder_per_camera` defaults to **True** — two independent ResNet18s. ACT shares one backbone; this doubles the vision parameters on the belief that different viewpoints deserve different extractors.

**ResNet18 trades space for semantics.** Stride 32: each of the 15×20 cells summarises a 32×32 input patch, and channels go 3 → 512, changing meaning from "colour" to "512 learned detectors". Identical to ACT to this point.

### SpatialSoftmax — the pivotal transformation

Three sub-operations, all in `SpatialSoftmax.forward`:

**1. `Conv2d(512, 32, kernel_size=1)`.** A 1×1 conv is a per-cell linear map *across channels*, touching no neighbours. It asks: which 32 combinations of the 512 detectors are worth **locating**? → `(16,32,15,20)`

**2. `softmax` over the flattened spatial axis.**
```python
features = features.reshape(-1, self._in_h * self._in_w)   # (16*32, 300)
attention = F.softmax(features, dim=-1)                    # over the 300 cells
```
Each channel becomes a **probability distribution over grid positions**. Channel *k* stops meaning "how strong is feature *k*" and starts meaning "**where** is feature *k*".

**3. Expectation against a coordinate grid — soft-argmax.**
```python
pos_x, pos_y = np.meshgrid(np.linspace(-1,1,W), np.linspace(-1,1,H))   # (300, 2)
expected_xy = attention @ self.pos_grid                                 # (16*32, 2)
```
`(N,300) @ (300,2)` is `Σ p(cell)·coord(cell)` — the mean position under that belief, in normalised `[-1,1]` coordinates.

```
(16, 32, 15, 20)  →  (16, 32, 2)
```

**`H` and `W` are consumed and replaced by a coordinate axis of size 2.** The output is not features-at-locations; it *is* locations. Thirty-two `(x,y)` pairs.

- **Compression is 14,400×**: `3×480×640 = 921,600` numbers → 64. Appearance — colour, texture, *which* of two similar objects — is gone. Only geometry survives.
- **Precision beats the grid.** Soft-argmax is a weighted mean, not an argmax, so a keypoint lands between cells. Resolution is set by softmax peakedness, not by the 15×20 spacing.

Then `Linear(64→64)` + ReLU lets the network recombine raw coordinates into useful quantities (e.g. *differences* between keypoints — relative rather than absolute position).

### Then `S` is destroyed

The camera axis is absorbed into features (`64+64=128`), the 6 joint values are concatenated (**134 per observation step**), and `.flatten(start_dim=1)` collapses `S`:

```
(8, 2, 134)  ->  (8, 268)
```

**The past stops being a time axis.** Nothing tells the network that features 0–133 are "now" and 134–267 are "33 ms ago" — it learns that from the fixed ordering, which is why `n_obs_steps` must match between training and deployment.

⚠️ Note what `S=2` buys: **velocity is implicitly available**. ACT at `n_obs_steps=1` sees only current positions.

**The entire visual world is now 268 numbers**, versus ACT's 602 tokens × 512 dims.

## Telling the network how broken the input is

`DiffusionSinusoidalPosEmb` turns scalar `t ∈ {0..99}` into 128 numbers — sin/cos at 64 geometrically spaced frequencies — then `Linear(128→512) → Mish → Linear(512→128)`.

**Why not pass `t` raw?** The network must behave very differently at `t=95` (nearly pure noise: make coarse structural decisions) than at `t=5` (nearly clean: polish only). Sinusoids give a smooth multi-scale code where adjacent `t` are similar and distant `t` differ. Same trick as transformer positional encoding, different quantity.

```python
global_feature = torch.cat([timesteps_embed, global_cond], axis=-1)   # (8, 396)
```

**`t` and the observation become the same kind of thing** — one global context vector. Neither has any temporal address inside the chunk.

## The axis flip that defines the architecture

```python
x = einops.rearrange(x, "b t d -> b d t")     # (8, 64, 6) -> (8, 6, 64)
```

`Conv1d` wants `(B, C, L)`, so **the 6 joints become channels and the 64 future timesteps become the length the kernel slides along.**

**Why convolve along time?** A `k=5` kernel spanning 5 consecutive future steps produces a feature describing *a short stretch of motion* — a local velocity, a curvature, a grasp closing. Sharing weights along `T` encodes: *what makes a stretch of trajectory plausible is the same at step 3 as at step 40.* Motion is translation-invariant in time. Strong and correct prior.

**Why aren't joints a second conv axis?** Joints are *not* translation-invariant — shoulder-pan and gripper are not interchangeable, so sliding a kernel across them is meaningless. Channels are the axis you mix fully; length is the axis you share weights along.

> **Share weights over time. Mix everything over joints.**

The mirror image of a transformer, which makes each timestep a token (no sharing over time, attention instead) and shares nothing over channels.

## One residual block, exactly

```python
out = self.conv1(x)                                    # Conv1d(k=5,pad=2) -> GroupNorm(8) -> Mish
cond_embed = self.cond_encoder(cond).unsqueeze(-1)     # Mish -> Linear(396, 2C) -> (B, 2C, 1)
scale, bias = cond_embed[:, :C], cond_embed[:, C:]
out = scale * out + bias                               # FiLM
out = self.conv2(out)                                  # Conv1d -> GroupNorm -> Mish
out = out + self.residual_conv(x)                      # skip (1x1 conv if channels differ)
```

Five load-bearing details:

**`padding = kernel_size // 2` → length preserved, and the convolution is NON-CAUSAL.** Position `t` sees `t−2 … t+2`, including *later* steps. Deliberate: the model revises the whole 64-step chunk simultaneously. **There is no autoregression anywhere in Diffusion Policy.**

**`GroupNorm(8)`, not BatchNorm.** Per-sample normalisation over `(channel-group × time)`, so it is **batch-independent** — the function at inference with batch 1 is identical to training at batch 64. Critical here, because inference runs batch 1 a hundred times.

**`Mish`** = `x·tanh(softplus(x))`, smooth everywhere. The target is a smooth continuous field; ReLU's kink doesn't help.

**🚨 FiLM is `(B, 2C, 1)` and broadcasts over `T`.** The deepest point in the architecture: **the same scale and bias apply to all 64 timesteps.** The observation *cannot* say "at step 12 do this" — it can only globally re-weight which feature channels matter. ⇒ **The observation has no temporal address.** All temporal structure must come from the convolutions reading the noisy trajectory; the conditioning only selects *which kind* of trajectory is being produced.

**The residual** makes each block learn a *correction*, which is what makes a 14-block stack trainable.

## The U: why downsample time

**Receptive field.** A stack of `k=5` convs at full resolution grows its view 4 steps per layer — ~16 layers just to see 64 steps. Downsampling makes it cheap: at `T=16`, one kernel step covers 4 original timesteps, so `k=5` spans 20 real steps. Computed over the real stack: **by the middle of the UNet one position already sees more than the entire 64-step chunk.**

That is what the U buys — **global temporal coherence**: step 60 can be consistent with step 2.

**The skips** restore what downsampling destroyed. Coarse levels decide the *shape* of the motion; skips hand back high-resolution *timing*. Concatenated along the **channel** axis, hence `dim_in * 2` in the up blocks.

> ⚠️ **Verified oddity: one skip is computed and never used.** Three skips are collected (`T=64,32,16`) but `up_modules` iterates `in_out[1:]` — only two entries — so only two are popped. The `T=64` skip is dead; `final_conv` covers that stage. Harmless, but relevant if you change `down_dims`.

## What the shapes do, dynamically

### Training — exactly one forward pass

```python
eps = torch.randn(trajectory.shape)                          # (8, 64, 6)
timesteps = torch.randint(0, num_train_timesteps, (B,))      # one t per sample
noisy_trajectory = self.noise_scheduler.add_noise(trajectory, eps, timesteps)
pred = self.unet(noisy_trajectory, timesteps, global_cond)
loss = F.mse_loss(pred, eps, reduction="none") * (~action_is_pad).unsqueeze(-1)
```

`add_noise` is elementwise per `(b,t,d)`: `x_t = sqrt(ᾱ_t)·x₀ + sqrt(1−ᾱ_t)·ε`. Computed from this config (`squaredcos_cap_v2`, `T=100`):

| `t` | `ᾱ_t` | signal `sqrt(ᾱ)` | noise `sqrt(1−ᾱ)` |
|---|---|---|---|
| 0 | 0.999369 | **100.0%** | 2.5% |
| 25 | 0.835621 | 91.4% | 40.5% |
| 50 | 0.478265 | 69.2% | 72.2% |
| 75 | 0.133495 | 36.5% | 93.1% |
| 99 | 0.000000 | **0.0%** | 100.0% |

Each step lands one sample at one random rung. Over training the network learns the denoising operation **at every corruption level at once** — one network, 100 jobs, selected by the `t` embedding.

**Why predict `ε` and not the action?** Algebraically equivalent (`x₀ = (x_t − sqrt(1−ᾱ)·ε)/sqrt(ᾱ)`), but far better conditioned: `ε` has unit variance at *every* `t`, whereas recovering `x₀` at `t=99` means dividing by ~0.0005. **Constant-scale target ⇒ stable gradients across the whole ladder.**

⇒ **Sanity check: initial loss must be ≈ 1.0** (measured 1.0569). An untrained net outputs ≈0 and the target is unit-variance noise. If it isn't ≈1 at init, something is miswired.

### Inference — 100 passes, but only one vision pass

```python
global_cond = self._prepare_global_conditioning(batch)     # ONCE, outside the loop
sample = torch.randn(size=(B, horizon, action_dim))        # pure noise
self.noise_scheduler.set_timesteps(self.num_inference_steps)
for t in self.noise_scheduler.timesteps:                   # 99, 98, ... 0
    model_output = self.unet(sample, t, global_cond=global_cond)
    sample = self.noise_scheduler.step(model_output, t, sample).prev_sample
```

`num_inference_steps` defaults to `None`, which **falls back to `num_train_timesteps` = 100**. Each `step` applies the DDPM posterior update, with `clip_sample=True` clamping the implied `x₀` to `[-1,1]` every iteration.

**Vision runs once and is reused identically 100 times** — so the ResNets and SpatialSoftmax are not in the loop, and `num_inference_steps` is the only real latency lever.

**Then most of the answer is discarded**: `horizon=64` generated, `n_action_steps=32` executed. The tail exists so the convolutions have context beyond the part you use.

## Measured cost, and the deployment verdict

On the Mac cockpit (MPS), one action chunk:

```
100 DDPM steps = 100 UNet forwards
latency 9551 ms   (95.5 ms per denoise step)
executes 32 actions -> replan every 1.07 s at 30 fps
budget 1067 ms -> DOES NOT FIT (9.0x over)
```

**Diffusion Policy at LeRobot defaults cannot run on the Mac.** Fixes, in order of preference: `noise_scheduler_type=DDIM` with `num_inference_steps≈10` (built for exactly this; network unchanged, bigger strides down the ladder), shrink `down_dims`, or run inference on a GPU box.

> ⚠️ **Apparent upstream inconsistency:** `drop_n_last_frames` defaults to **7**, but its own comment gives `horizon − n_action_steps − n_obs_steps + 1 = 64 − 32 − 2 + 1 = 31`. The 7 matches an older `horizon=16, n_action_steps=8` config. Not fatal (`action_is_pad` masks short windows) but it means more padded supervision near episode ends than intended. Set it to 31 for a serious run.

## Versus ACT

| | [ACT](act-shapes.md) | Diffusion Policy |
|---|---|---|
| Obs steps | 1 | **2** (velocity implicit) |
| Vision backbone | **one shared** ResNet18 | **one per camera** |
| Image → representation | 300 tokens/cam, 602 total × 512 dims | **64 numbers/cam** |
| Core network | transformer, 4 enc + 1 dec | **1D conv UNet over time** |
| Conditioning | tokens in the sequence | **FiLM**, no temporal address |
| Action output | 100-step chunk, all executed | 64-step horizon, **32 executed** |
| Produced by | one forward, direct regression | **100 denoising steps** |
| Loss | L1 + 10 × KL | **MSE on noise** |
| Default LR | 1e-5 | **1e-4** |
| Params | 51.6 M | **277.8 M** (5.4×) |

**ACT spends capacity on seeing; Diffusion Policy spends it on generating.**

**The multimodality difference matters most.** ACT needs a CVAE latent to represent "several valid ways to do this" — and on the Φ dataset that latent **fully collapsed** (`kld = 6.7e-5`, output sensitivity 0.067%; see [the 8-bin experiment](../../experiments/2026-08-04_act-8bin-6run.md)). Diffusion Policy needs no such device: **different starting noise gives a different valid trajectory.** Multimodality lives in the sampling process, so there is no auxiliary variable to collapse and nothing gets zeroed at test time.

**The honest risk for a Φ task:** 64 numbers per camera means the policy knows *where* things are and almost nothing about *what* they are. Fine for one duck and one box. Add a second, similar-looking object and keypoints may not distinguish them — where ACT's 300 full-dimensional tokens per camera still could.

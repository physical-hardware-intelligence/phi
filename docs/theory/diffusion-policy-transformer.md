# Diffusion Policy — the transformer backbone

The same diffusion process as [diffusion-policy-why](diffusion-policy-why.md), with the denoiser swapped: a **1-D conv UNet over time** becomes an **8-layer transformer decoder**.

> 📖 **Read [diffusion-policy-why](diffusion-policy-why.md) first.** It derives *why diffusion at all* — why noise, why 100 small steps, why the target is the noise, why multimodality lives in sampling. None of that changes here and none of it is repeated. This page is only about **what changes when the backbone becomes a transformer**, and what we measured when we trained one.

Everything below is measured from our port on this repo's data, not read off the paper. Where a number comes from the paper it says so.

---

## The whole thing on one page

The UNet and the transformer solve the same problem — *given a noisy 48-step trajectory and a description of the scene, say which part is garbage* — but they disagree about **what a trajectory is**.

| | conv UNet | transformer |
|---|---|---|
| a trajectory is… | a **signal** with 6 channels and 48 time samples | a **sequence** of 48 tokens |
| shares parameters over… | time (a kernel slides) | nothing; attention relates positions explicitly |
| receptive field | grows with depth, local first | global at layer 1 |
| the diffusion step `k` enters as | **FiLM** — scale+shift on every residual block, unavoidable | **one memory token** each action token may choose to query |
| the observation enters as | FiLM, folded into the same conditioning vector | **2 more memory tokens** read by cross-attention |
| inductive bias | smoothness, locality | none; must be learned |

> Shapes here use our trained config (`horizon=48`). The CNN trace in [diffusion-policy-shapes](diffusion-policy-shapes.md) uses lerobot's default of 64 — same architecture, different horizon.

The conv UNet *assumes* nearby timesteps are related. The transformer assumes nothing and has to learn it. That single difference explains every result on this page: the transformer needs more regularisation, is more sensitive to how much data you have, and does not get smoothness for free.

---

# Part I — the port

## 1. Where the code came from, and why we didn't fork lerobot

`lerobot` ships only the CNN backbone. Rather than fork it, we **vendored three files unmodified** from the original [real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy) (MIT, verified against the GitHub API on 2026-08-13) and registered a new policy type beside lerobot's:

```
src/phi/policies/transformer_for_diffusion.py   vendored, UNMODIFIED
src/phi/policies/positional_embedding.py        vendored, UNMODIFIED
src/phi/policies/module_attr_mixin.py           vendored, UNMODIFIED
src/phi/policies/dp_transformer.py              ours — adapter + config + registration
```

Attribution headers are retained in all three. `_register()` patches `lerobot.policies.factory.get_policy_class`, so `--policy.type=diffusion_transformer` works everywhere lerobot's own types do, and **`lerobot` itself is untouched** — it stays a clean pip dependency you can upgrade.

> ⚠️ The port lives on `PYTHONPATH`, not in `lerobot`. Every sbatch script must `export PYTHONPATH=$PROJECT/repo/src` **and** fail fast on a missing import, or a job dies twenty minutes in with `KeyError: diffusion_transformer`. See any script in `configs/hpc/`.

## 2. The adapter is three lines, because of one property of `flatten`

lerobot builds its conditioning vector and ends with:

```python
return global_cond.flatten(start_dim=1)     # (B, S, D) -> (B, S*D)
```

The transformer wants the un-flattened `(B, S, D)` so it can treat the `S` observation steps as separate memory tokens. Row-major `flatten` is **losslessly reversible** — it interleaves nothing — so the adapter only has to reshape back:

```python
def forward(self, x, timestep, global_cond=None):
    cond = global_cond.reshape(x.shape[0], self.n_obs_steps, self.obs_feat_dim)
    return self.net(x, timestep, cond)
```

No re-derivation, no second encoder pass, no risk of a silent axis swap. **The whole port is this reshape plus a config class.**

## 3. One trap: `get_optim_params` must return *groups*

The transformer recipe splits parameters into decayed and non-decayed sets (biases, LayerNorms and embeddings get **no** weight decay). If `get_optim_params()` returns a flat iterable, that split is silently discarded and every parameter gets the same decay.

Ours returns groups, measured:

```
42,528,832 params @ weight_decay 1e-3
    83,366 params @ weight_decay 0
```

Sanity check: `42,528,832 + 83,366 = 42,612,198` = transformer `9,020,934` + three ResNet-18 encoders `33,591,264`. Nothing is missing and nothing is double-counted.

---

# Part II — the architecture, dimension by dimension

## 4. Parameter arithmetic that closes exactly

**The config we actually trained** — note `horizon=48`, not the class default of 64. That is the `tp48` in our model-card names.

```
horizon H = 48        n_obs_steps S = 2        n_emb E = 256
n_layer = 8           n_head = 4               causal_attn = True
cond_dim = 198        = 3 cameras × 64 keypoint features + 6 joint values
input_dim = output_dim = 6
```

Which gives **T = 48 action tokens** and **T_cond = 3 memory tokens**.

```
Per decoder layer                                              1,053,440
  ├─ self-attention                                              263,168
  │     in_proj_weight (768, 256)   = 3 × 256 × 256 = 196,608
  │     in_proj_bias   (768,)                            768
  │     out_proj       (256, 256)                     65,536
  │     out_proj bias  (256,)                            256
  ├─ cross-attention   (identical shape)                         263,168
  ├─ feed-forward                                                525,568
  │     Linear(256 → 1024)  262,144 + 1,024
  │     Linear(1024 → 256)  262,144 +   256
  └─ 3 × LayerNorm     3 × (256 + 256)                             1,536

Full ledger:
  input_emb     Linear(6 → 256)                      1,792
  pos_emb       Parameter(1, 48, 256)               12,288
  cond_obs_emb  Linear(198 → 256)                   50,944
  cond_pos_emb  Parameter(1, 3, 256)                   768
  encoder MLP   256 → 1024 → 256                   525,568
  decoder       8 × 1,053,440                    8,427,520
  ln_f + head   LayerNorm(256) + Linear(256 → 6)     2,054
  ─────────────────────────────────────────────────────────
  TOTAL                                          9,020,934      ✓ closes exactly
```

**9,020,934 ≈ the "9" in the Diffusion Policy paper's Table 8.** Widened to `n_emb=768` we measured **80,540,166**, matching that table's "80". The port reproduces the paper's own parameter counts, which is the strongest available evidence it is wired correctly.

> 🔑 **`in_proj_weight` packs Q, K and V into one matrix.** `(3d, d)` is not three heads — it is three *projections* stacked, so one matmul produces all of Q, K, V. A common misreading; check the shape before assuming head count.

## 5. The action block — how a trajectory becomes 48 tokens

This is the `tgt` of the decoder: the thing being denoised.

```
sample                       (B, 48, 6)     the noisy trajectory
  │ input_emb  Linear(6 → 256)              1,792 params
  ▼
token_embeddings             (B, 48, 256)
  │ + pos_emb[:, :48, :]     Parameter(1,48,256), LEARNED, ADDED not concatenated
  ▼
  │ drop(p_drop_emb = 0.0)   ← a no-op at our config
  ▼
x                            (B, 48, 256)   → decoder tgt
```

Three things worth sitting with:

**`input_emb` is an expansion, 6 → 256, and per-timestep.** It touches one timestep at a time and mixes nothing across time — all temporal mixing is attention's job. Why expand 6 numbers into 256? Because the token is not just storage for the payload; it is the **workspace** attention will write into. Six numbers is what the token *carries*; 256 is the room it has to think. Cost is 1,792 parameters — 0.02% of the model.

**The position embedding is added, not concatenated.** Concatenating would grow the width and force every downstream matrix to be bigger. Adding keeps `E=256` and relies on the network learning to separate "which timestep am I" from "what action am I" inside the same 256 dimensions. This is standard transformer practice and it works because the two signals occupy different learned subspaces.

**The output is symmetric with the input.** After 8 decoder layers, `ln_f` then `head: Linear(256 → 6)` collapses back to `(B, 48, 6)`. **Same shape in, same shape out** — because Diffusion Policy is a repair operation on trajectories, not a function from observation to action. See [diffusion-policy-shapes §The framing](diffusion-policy-shapes.md#the-framing-that-makes-the-rest-make-sense).

## 5a. Where the 198 comes from — the vision encoder, measured

The decoder receives `(B, 2, 198)`. Here is every step that produces it, measured on our trained checkpoint. **This is the same encoder DP-CNN uses** — it is the one part the two backbones share, which is what makes DP-CNN vs DP-T a clean backbone comparison.

### Three independent ResNets

```
use_separate_rgb_encoder_per_camera = True
ModuleList of 3   weights shared between cam0 and cam1? False   (checked data_ptr)
params each 11,197,088   total 33,591,264
```

Three genuinely separate ResNet-18s — the bet being that a wrist view and an overhead view deserve different filters. ACT bets the opposite and shares one backbone. Never A/B'd here.

### The chain, one camera

```
raw                                (2, 3, 480, 640)
Resize(240,320) antialias=True     (2, 3, 240, 320)
CenterCrop(216,288)                (2, 3, 216, 288)     RandomCrop during training
ResNet-18 body, stride 32          (2, 512,   7,   9)   216/32=6.75→7,  288/32=9.00→9
Conv2d(512 → 32, kernel 1×1)       (2,  32,   7,   9)
SpatialSoftmax                     (2,  32,   2)        ← H,W GONE
flatten → Linear(64→64) → ReLU     (2,  64)
```

> 🚨 **The feature map is 7×9 = 63 cells.** Not the 15×20 = 300 of [diffusion-policy-shapes](diffusion-policy-shapes.md), which traces lerobot's *defaults* (no resize, no crop). Our config resizes to 240×320 and crops to 216×288 first, so the grid is **4.8× coarser**. Each of the 63 cells summarises a 32×32 patch of the crop — undoing the 2× resize, a **64×64 pixel region of the original camera frame**.

### SpatialSoftmax — the index/value swap

Three sub-operations. The middle one is the conceptual jump.

**1. `Conv2d(512 → 32, kernel 1×1)`** — a per-cell mix *across channels only*, touching no neighbours. ResNet gave 512 learned detectors at each of 63 cells; this asks **which 32 combinations are worth locating**.

**2. `softmax` over the flattened 63 cells** — each of the 32 channels independently becomes a **probability map over grid positions**. Verified: `channel 0 sums to 1.000000, peak cell 0.1031`.

**3. Expectation against a fixed coordinate grid** — `(32, 63) @ (63, 2)`, i.e. `Σ p(cell)·coord(cell)`. Measured: `keypoint 0 landed at (x,y) = (+0.2367, −0.4906)` in `[-1,1]`. The grid is a fixed buffer, **not learned**.

> 🔑 Before: *channel k says how strongly feature k fires, at every cell.* After: *channel k says **where** feature k is.* The `H,W` axes are consumed and replaced by an axis of size 2. The output is not features-at-locations — it **is** locations.

Because it is a weighted mean rather than an argmax, a keypoint can land *between* cells: precision is set by softmax peakedness, not by the 7×9 spacing. That matters a lot at 63 cells.

**Compression: 186,624 numbers → 64 per camera, i.e. 2,916×.** Lossy by construction — colour, texture, and *which* of two similar cubes are all gone. Only geometry survives.

### And that is the 198

```
3 cams × 64 = 192   +   state 6   =   198
unfold S=2                        →   (B, 2, 198)
```

## 6. The memory block — how the observation becomes 3 tokens

This is what cross-attention reads. It is assembled from two very different sources.

```
timestep k        (B,)                  a single integer per sample
  │ SinusoidalPosEmb(256)               PARAMETER-FREE
  ▼
time_emb          (B, 1, 256)  ──────┐
                                     │
cond              (B, 2, 198)        │   2 observation steps,
  │ cond_obs_emb Linear(198 → 256)   │   each 3×64 keypoints + 6 joints
  ▼                                  │
cond_obs_emb      (B, 2, 256)  ──────┤
                                     ▼
                        torch.cat(dim=1)   ← the CONCATENATION, along SEQUENCE
                                     ▼
cond_embeddings   (B, 3, 256)
  │ + cond_pos_emb  Parameter(1,3,256), learned          768 params
  │ encoder                                          525,568 params
  ▼
memory            (B, 3, 256)      → decoder memory
```

**The concatenation is along the sequence axis, not the feature axis.** `(B,1,256) ⊕ (B,2,256) → (B,3,256)`. The widths already match at 256, which is exactly why both branches project to `E` first. Nothing is summed; the three tokens sit side by side, and the decoder queries them like any other sequence.

So the memory is literally **three tokens**:

| token | is | width |
|---|---|---|
| 0 | "how noisy is this trajectory" (diffusion step `k`) | 256 |
| 1 | observation at `t−1` | 256 |
| 2 | observation at `t` | 256 |

> 🚨 **`self.encoder` is not a transformer encoder at our config.** With `n_cond_layers=0` the code takes this branch:
>
> ```python
> self.encoder = nn.Sequential(
>     nn.Linear(n_emb, 4 * n_emb), nn.Mish(), nn.Linear(4 * n_emb, n_emb))
> ```
>
> A 2-layer MLP applied **independently to each of the 3 tokens**. There is **no attention in the encoder** — the diffusion-step token and the two observation tokens never mix with each other. They only meet inside the decoder's cross-attention, filtered through whatever each action token chooses to query.
>
> At **525,568 parameters** this MLP is the same size as one decoder feed-forward and **5.8% of the whole model** — a large budget for a position-wise map that does no mixing. Setting `n_cond_layers > 0` swaps it for real `nn.TransformerEncoderLayer`s, which is an untested knob for us.

**Why is the timestep a memory token rather than a modulation?** This is the sharpest architectural difference from the CNN backbone. The UNet injects `k` through **FiLM** — a per-channel scale and shift applied at every residual block, so `k` conditions *everything, everywhere, unavoidably* (see [diffusion-policy-why §14](diffusion-policy-why.md#14-film-the-deepest-choice-in-the-architecture)). The transformer makes `k` **one token among three** and lets each action token *choose* how much to attend to it. More expressive; also easier to ignore. That freedom is part of why the transformer needs more regularisation.

## 7. The two masks — and one surprising consequence

`causal_attn=True` (the paper's Table 8 value) creates **two** masks. With `causal_attn=False` both are `None` and everything is bidirectional.

**`tgt_mask` — `(48, 48)`, causal over action tokens.** Additive `-inf` above the diagonal. Measured: row 0 attends to 1 token, row 47 attends to 48.

An action token can see earlier timesteps of the trajectory but not later ones. That is a real design choice, not an obvious one: the whole chunk is denoised *jointly*, so bidirectional attention was available and was not taken. The cost is that early tokens have very little context — token 0 sees only itself.

**`memory_mask` — `(48, 3)`.**

```python
t, s = torch.meshgrid(torch.arange(T), torch.arange(S), indexing='ij')
mask = t >= (s - 1)      # "add one dimension since time is the first token in cond"
```

| `s` | memory token | rule | effect |
|---|---|---|---|
| 0 | diffusion step `k` | `t ≥ −1` | always visible |
| 1 | observation `t−1` | `t ≥ 0` | always visible |
| 2 | observation `t` | `t ≥ 1` | **invisible to action token 0** |

This looks like an off-by-one and is not. It is correct temporal causality, and the reason is §7a.

### 7a. 🔑 The horizon does not start at "now" — and that is why action 0 is discarded

Straight from the config, not interpretation:

```
observation_delta_indices = [-1, 0]
action_delta_indices      = [-1, 0, 1, 2, ... , 46]
```

A delta index is an offset in frames from the current moment `n`. **Both lists start at −1.** So the observation window and the action window are *aligned*, and they both begin one frame in the past:

| action index | moment | executable? |
|---:|---|---|
| 0 | **n−1** | no — already happened |
| 1 | **n** (now) | **yes, the first real one** |
| 47 | n+46 | yes |

`generate_actions` therefore slices:

```python
start = n_obs_steps - 1        # = 1
actions = actions[:, start : start + n_action_steps]
```

Verified across configs — the first executed action is always `n+0`:

```
 n_obs      obs moments     action moments   start   discarded   first executed
     1         n+0..n+0          n+0..n+47       0        none            n+0
     2         n-1..n+0          n-1..n+46       1    idx 0..0            n+0
     3         n-2..n+0          n-2..n+45       2    idx 0..1            n+0
     4         n-3..n+0          n-3..n+44       3    idx 0..2            n+0
```

**Number discarded = number of past observation steps = `n_obs_steps − 1`.** Not a magic constant — "skip however many frames of history you asked for."

**So the mask is right.** Action token 0 predicts the action at moment `n−1`. Observation index 1 is from moment `n`, which is *after* that action. Blocking it refuses to let a token see its own future. And the proof that this is principled rather than accidental — the restricted set and the discarded set are identical at every `n_obs_steps`:

```
  n_obs_steps=2   blocked memory access: [0]        discarded: [0]        identical
  n_obs_steps=3   blocked memory access: [0, 1]     discarded: [0, 1]     identical
  n_obs_steps=4   blocked memory access: [0, 1, 2]  discarded: [0, 1, 2]  identical
```

Both fall out of the same anchoring, so they track automatically at any `n_obs_steps`. That is not what coincidence looks like.

> ⚠️ **Corrected 2026-08-14.** An earlier version of this section called this a "surprising consequence" and claimed *"action token 0 — the very next action to execute — cannot attend to the most recent observation."* **Wrong twice over.** Action 0 is not the next action to execute (it is discarded), and its restriction is correct, not suspect. The suggested experiment that followed — testing `causal_attn=False` for a first-action artefact — was chasing nothing.

**Why anchor the horizon in the past at all?** Three reasons, and the third is the real one:

1. **One timeline.** Action index `i` and observation index `i` are the same moment, which is what lets the memory rule be a one-liner.
2. **Free training data.** Grab 48 consecutive actions and the aligned observations; no offsetting anywhere.
3. **It anchors the trajectory.** Action 0 is a *reconstruction of something the model can check against an observation it was given.* Like drawing a curve through known points, it makes the continuation start in the right place pointing the right way. Action 0 is not waste — it is a **run-up**, costing one of 48 slots to make step 1, the one that reaches the motors, better anchored.

> 🔑 **Consequence for the `causal_attn` ablation:** `causal_attn` is one flag controlling *two* masks (verified: `False` sets both to `None`). But the memory mask only ever restricted action token 0, and action token 0 never reaches the robot — so turning the flag off is *effectively* a clean self-attention-only experiment. See [experiments/2026-08-14](../../experiments/) for the pre-registered run.

## 8. The three sublayers, with every dimension

The decoder is `nn.TransformerDecoder(decoder_layer, num_layers=8)` — **8 identical-in-shape, independently-parameterised layers**, applied in sequence. Not one layer run 8 times: 8 separate sets of weights, `8 × 1,053,440 = 8,427,520` parameters.

Each layer applies three sublayers to `x` of shape `(B, 48, 256)`, with the **same `(B, 3, 256)` memory** at every layer:

| sublayer | reads | writes | what it can do |
|---|---|---|---|
| self-attention | `(B,48,256)` | `(B,48,256)` | relate timestep *i* to timestep *j* **within the trajectory**, masked causally |
| cross-attention | `(B,48,256)` queries, **`(B,3,256)` memory** | `(B,48,256)` | let each timestep query **`k`, obs `t−1`, obs `t`** |
| feed-forward | `(B,48,256)` | `(B,48,256)` | per-timestep nonlinearity, 256 → 1024 → 256 |

> 🔑 **The memory is computed once and reused by all 8 layers.** The encoder runs a single time; `memory` is passed unchanged into every `TransformerDecoderLayer`. Only the action tokens evolve with depth — the observation representation the decoder reads is identical at layer 0 and layer 7.
>
> This is the opposite of SmolVLA, where the prefix is *also* refined by every layer, so the expert reads increasingly abstract features as it goes deeper (see [smolvla §11a](smolvla.md#11a-mode-a-even-rows-join-attend-split)). Here the ladder does not exist.

Inside attention the two matmuls contract **different axes** — this is the part worth slowing down on. With 4 heads, `head_dim = 256/4 = 64`:

```
self-attention  (48 queries, 48 keys)
  scores = Q @ Kᵀ   (B,4,48,64) @ (B,4,64,48)  contracts head_dim → (B,4,48,48)
  scores /= √64  = /8
  output = P @ V    (B,4,48,48) @ (B,4,48,64)  contracts n_keys   → (B,4,48,64)

cross-attention (48 queries, 3 keys)          ← note the asymmetry
  scores = Q @ Kᵀ   (B,4,48,64) @ (B,4,64,3)   contracts head_dim → (B,4,48,3)
  output = P @ V    (B,4,48,3)  @ (B,4,3,64)   contracts n_keys   → (B,4,48,64)
```

The first collapses the *feature* axis to produce similarity; the second collapses the *key* axis to produce a weighted average. Same operation, opposite meaning.

**Cross-attention's map is `(48, 3)`** — every action token distributes 3 softmax weights over "how noisy am I", "what did I see at `t−1`", "what did I see at `t`". That is the entire channel through which the observation reaches the trajectory.

**The `√d` scaling is not decoration.** We measured the standard deviation of `Q@Kᵀ` against `d` and it tracks `√d` to three digits. Without the division, softmax saturates as `d` grows and gradients vanish.

The causal mask is **additive `-inf`**, not multiplicative — applied before softmax so masked positions receive exactly zero probability rather than a small one.

## 9. Pre-LN, and the sublayers are sequential — not parallel

```python
x = x + attn(ln1(x))
x = x + cross(ln2(x))
x = x + ff(ln3(x))
```

A natural misreading is that all three sublayers read the same `x` in parallel. **They do not.** We verified with forward hooks that `ln1`, `ln2` and `ln3` receive *different* tensors — each sees the output of the previous sublayer. Information from self-attention is already present when cross-attention runs.

### Why LayerNorm at all — measured

Residual streams grow. We measured the growth of `‖x‖` through 8 layers:

```
without LayerNorm   866×
with    LayerNorm   4.3×
```

Pre-LN normalises the *input* to each sublayer while leaving the residual highway untouched, so the sublayer always sees a well-scaled input no matter how large the stream has grown.

> ⚠️ **A claim we made and had to walk back.** We first said "normalization layers absorb a lot," which is too loose to be useful. Measured, the decay of an injected scale perturbation is `1.789 → 1.111 → 1.013 → 1.000` — it is gone within about **two blocks**, and the final output ratio is `1.010`. LayerNorm absorbs input-scale differences quickly and almost completely. That is a much narrower statement than the one we started with, and it is the one the measurement supports. It also killed a recommendation we had been about to make, to switch normalisation to `QUANTILES`.

## 10. Two different position encodings, and why

| what | encoding | size | why |
|---|---|---|---|
| action timestep (0…63) | **learned lookup table** | 48 × 256 = 12,288 | small, fixed, densely trained vocabulary |
| diffusion step `k` (0…99) | **sinusoidal** | 128 dims | see below |

**Why a learned table for positions.** When the vocabulary is small, fixed, and every entry is visited in every batch, a free lookup table is strictly more expressive than any formula — it can represent *any* function of position, including ones no sinusoid can. The cost is trivial: 12,288 parameters, 0.14% of the model. A formula only earns its place when you need to generalise to positions you never trained on. Here you never do — the horizon is always 64.

**Why sinusoidal for `k`.**

```python
emb = exp(arange(128) · −log(10000)/127)
```

`arange(128)` is `0…127`; multiplying by `−log(10000)/127` and exponentiating gives a **geometric sweep from 1.0 down to 1/10000**. `exp(0)=1` at one end, `exp(−log 10000)=1e−4` at the other, geometrically spaced between. Each of the 128 dims oscillates at its own rate, so a single `k` becomes a multi-scale code.

Two measured properties earn it:

- **Smoothness** — cosine similarity between adjacent `k` is **0.9723**. Neighbouring noise levels get neighbouring codes, so the network can interpolate behaviour across `k` instead of memorising 100 unrelated cases.
- **Constant norm** — every embedding has norm `√128`. The conditioning signal never grows or shrinks with `k`.

> 🚨 **A justification we got wrong, recorded so nobody repeats it.** We originally claimed sinusoids were needed because DDIM samples *non-integer* timesteps at inference. **That is false.** We measured `set_timesteps(16)` and it returns `[90, 84, …, 0]` — all integers, all within the trained range. Sinusoids are justified by smoothness and constant norm, **not** by interpolation to unseen values. The wrong reason would have led us to a wrong conclusion about which encodings are substitutable.

---

# Part III — what we measured training it

## 11. The action scale is not what the config implies

Actions are normalised `MIN_MAX`, which maps them to `[−1, 1]`. It is easy to read that as "unit variance". It is not.

```
measured variance of normalised actions:  0.313
```

Min-max puts the *extremes* at ±1; the bulk sits well inside. This matters because `clip_sample_range=1.0` is its **matched pair** — clipping at ±1 is only correct because min-max guarantees the data lives there. Change one and you must change the other.

Compare SmolVLA, which uses `MEAN_STD` and genuinely does get unit variance. See [smolvla §14 flow matching vs DDPM](smolvla.md#14-flow-matching-vs-ddpm).

## 12. Width is nearly free

```
n_emb 256 →  9,020,934 params
n_emb 768 → 80,540,166 params    8.9× the parameters
wall-clock cost                  +0.5%
```

Vision dominates. Three ResNet-18 encoders are **33,591,264** parameters run on 384 images per batch, against a 9 M denoiser run on tensors of shape `(B, 64, 256)`. If you are choosing where to spend, **widening the transformer is close to free and widening the vision path is not.**

## 13. The regularisation A/B — and what it actually showed

Two arms, identical except for regularisation. Pre-registered in `experiments/2026-08-13_dp-transformer-regularisation-ab.md` before either was run.

| arm | weight decay | attn dropout | best `eval_loss` |
|---|---|---|---|
| `unreg` | 1e-6 | 0.0 | **0.0287 @ 8k** |
| `paper` (Table 8 values) | 1e-3 | 0.3 | 0.0531 @ 26k → **0.0492 @ 50k** |

**`unreg` wins on `eval_loss` by 1.7×, and the gap never closes.** But the useful finding is not the winner — it is the *shape*.

- `unreg` bottoms at **8k steps** and climbs monotonically to 30k. It overfits, and it overfits early.
- `paper` descends slowly for 50k+ steps and never reaches `unreg`'s floor. It **underfits** — train loss ended at 0.061 against `unreg`'s 0.013 after 29 epochs.

So Table 8's regularisation **controls the overfit shape correctly but overshoots in strength for our dataset size**. That is a statement about *our 120 episodes*, not about the paper — the paper's recipe was tuned for larger datasets.

> 🔑 **What this implies for the CNN backbone.** The CNN's own 4× overfit was previously ascribed to a missing regulariser. This A/B says the more likely cause is **dataset size**: even the paper's full regularisation cannot make a transformer fit 120 episodes without underfitting, so the CNN's overfit is probably data-limited too. Adding regularisation to the CNN is unlikely to be the fix; adding episodes is.

## 14. Bucketed evaluation — `eval_loss` averages over a variable that matters

`compute_loss` draws **one `k` per sample, uniformly** from `[0, 100)`, and `eval_loss` averages over that draw. But the difficulty of the job is wildly different at each `k`:

```
k=0    signal 0.9997  noise 0.0251   recovering ε means dividing by 0.0251
k=50   signal 0.6916  noise 0.7223   balanced
k=99   signal 0.0005  noise 1.0000   the input essentially IS the noise
```

`src/phi/eval/eval_by_k.py` pins `k` and reports **variance explained** (`1 − mse`), because raw MSE mostly reproduces the schedule shape and says nothing about the model. Two design points make it a measurement rather than a plot:

1. **The same `ε` is reused across every bucket and every checkpoint**, seeded per batch index. Without this the noise draw confounds both comparisons.
2. **`global_cond` is hoisted out of the bucket loop** — it has no `k` dependence, so 15 buckets cost `1× vision + 15× denoiser` rather than `15× (vision + denoiser)`. Vision is the dominant cost.

### The measured result

At 30k steps, `unreg` wins **15 of 15 buckets** — and the gap is not uniform:

```
   k    unreg     paper     diff
   0  0.65653   0.56571  +0.09082
   2  0.78092   0.67286  +0.10806
   5  0.85334   0.81676  +0.03658
  ...
  95  0.99977   0.99828  +0.00149
  99  0.99981   0.99751  +0.00231

  mean gap  k ≤ 5   +0.07849
  mean gap  k ≥ 90  +0.00207      38× smaller
```

**The regularisation penalty is concentrated almost entirely at low `k`** — the fine-action-detail regime, which is the one that decides whether the gripper closes on a 25 mm cube. That is a far sharper statement than "paper's `eval_loss` is higher".

We also measured that `unreg`'s buckets **move in opposite directions** as it overfits: `k=0` degrades by 18% while `k=99` *improves* 7.2×. A single averaged number cannot show this, and would report a mild net change.

> ⚠️ **A claim we made backwards.** We wrote that "the average hides what matters" on the theory that `eval_loss` is a blend dominated by easy high-`k` buckets. Measured on our schedule, **`k=0–9` contributes 49.8% of the uniform average and `k=80–99` contributes 1.2%.** `eval_loss` is already close to a *low-`k`* metric. It still cannot tell you *which* regime moved — that is the real justification for bucketing — but the original reasoning was wrong.

## 15. The resume trap: `--steps` is the cosine horizon

We resumed `paper` from 30k to 60k with `--config_path` pointing at the checkpoint and `--resume=true`, and the sbatch banner claimed:

> Config comes FROM THE CHECKPOINT, so every parameter is identical by construction — only the CLI `--steps` override changes.

**That is wrong, and it is a trap worth internalising.** The scheduler is `cosine`, and `--steps` *is* its horizon:

```
  step 30000, cosine over 30000 total → lr = 0.000e+00
  step 30000, cosine over 60000 total → lr = 5.066e-05
  measured lr in the resume log       → 5.0e-05   ✓
```

Extending the run **re-raised the learning rate from ~0 back to 5e-5** and re-annealed over a fresh 60k cosine. That fully explains the `eval_loss` bump at the resume boundary (`0.0536 → 0.0583 → 0.0598`, then recovery). The model was kicked out of the minimum the original schedule had annealed it into.

> 🔑 **Rule: with any horizon-dependent scheduler, changing `--steps` on resume changes the entire LR trajectory, not just where you stop.** A resumed run is a *different recipe*, not a longer one. If you want a clean extension, use a schedule that does not depend on the total (constant, or step decay).

## 16. What `eval_loss` does and does not tell you

This section exists because we got it wrong twice, in opposite directions, and the second error is the more instructive one.

### The claim we propagated, and why it was false

Several places in this repo stated that a DP checkpoint whose `eval_loss` had **"risen 4×"** still scored 50% on the arm — and concluded from that that DP noise-MSE was *"worthless"* for model selection.

Two separate errors, both now corrected repo-wide.

**Error 1 — it conflated the run with the checkpoint.** From `experiments/2026-08-12`:

```
task 0 `paper`:  10k 0.0155 (optimum, never saved) │ 20k 0.0173 │ 100k 0.0636
                 0.0636 / 0.0155 = 4.10×   ← the RUN rose 4×
                 0.0173 / 0.0155 = 1.12×   ← the ROLLED-OUT checkpoint
```

The run's loss did rise 4×. The checkpoint actually put on the arm was **20k, only 12% past its own optimum**. The 4×-degraded 100k checkpoint was **never rolled out**.

**Error 2 — and this is the one worth internalising — one point cannot refute a correlation.** Exactly **one** DP checkpoint was ever rolled out. To test whether `eval_loss` *ranks* checkpoints you need several DP checkpoints spanning a range of `eval_loss`, rolled out under identical conditions. That experiment has never been run.

Nor does comparing DP against ACT test it: their `eval_loss` values are **different quantities** — noise-prediction MSE versus L1 + KL — and are not comparable numbers. A cross-architecture tie says nothing about whether either metric ranks its own checkpoints.

### What is actually supported

```
n = 12 episodes → 29% power
n = 30 episodes → 64% power
n = 40 episodes → 77% power
```

At n=12 we cannot detect anything but a huge effect. DP and ACT tied at a paired difference of **−0.018** with **7 of 11 episodes identical** — at 29% power, that result carries almost no information either way.

> 🔑 **The supportable statement: DP `eval_loss` is UNVALIDATED as a model-selection metric on this task — not disproven.** Untested is not the same as useless. Ranking checkpoints by it remains the only cheap option we have, and there is no evidence against doing so. Chasing the 10k optimum is still the reasonable default.
>
> The generalisable lesson is about *us*, not the metric: **a single underpowered observation was used to retire a measurement**, and that conclusion then propagated into eight files, a script docstring, and job banners where it shaped later decisions. The failure mode is not "we measured wrong" — it is "we concluded too hard from one point, and wrote the conclusion down as fact."

### The experiment that would actually settle it

Roll out DP checkpoints at 20k / 40k / 60k / 100k — spanning `eval_loss` 0.0173 → 0.0636, a genuine 4× range — on the same held-out episodes, `n ≥ 30`. If success rate is flat across a 4× `eval_loss` spread, the metric is uninformative *and we would know it*. That is one afternoon of arm time and it retires the question permanently.

---

## See also

- [diffusion-policy-why](diffusion-policy-why.md) — the diffusion process itself; read first
- [diffusion-policy-shapes](diffusion-policy-shapes.md) — the CNN backbone, every tensor measured
- [smolvla](smolvla.md) — flow matching instead of DDPM, and a frozen VLM instead of a trained encoder
- [act-shapes](act-shapes.md) — the CVAE baseline both are compared against
- `experiments/2026-08-13_dp-transformer-regularisation-ab.md` — the pre-registration and full results
- `src/phi/eval/eval_by_k.py` — the bucketed evaluator

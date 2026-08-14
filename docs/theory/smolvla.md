# SmolVLA — a frozen VLM with an action expert bolted on

Every number on this page was **measured**, not quoted. The architecture was built from `config.json` alone (`load_vlm_weights=False`), which needs no checkpoint download, and then instrumented with forward hooks. Traced against `lerobot` 0.6.0, `transformers` 5.5.4.

> 📖 **Prerequisites.** [diffusion-policy-why](diffusion-policy-why.md) for why generative action heads exist at all, and [diffusion-policy-transformer](diffusion-policy-transformer.md) for attention mechanics (`√d` scaling, pre-LN residuals, packed QKV). This page assumes both and does not repeat them.

Paper: <https://huggingface.co/papers/2506.01844>

---

## The whole thing on one page

SmolVLA is **two transformer towers running in lockstep**, 16 rows tall.

```
        LEFT COLUMN                    RIGHT COLUMN
        "the VLM"                      "the action expert"
        960 wide                       720 wide
        FROZEN                         TRAINABLE
        241 tokens                     50 tokens
        (images + language + state)    (the noisy action chunk)
```

Both do row 0, then both do row 1, and so on. At every row they talk. **How they talk alternates**, and that alternation is the entire architecture.

Three claims that make the rest follow:

1. **Almost nothing is new.** Of 450 M parameters, only **1.6 M** are modules SmolVLA invented — five `Linear` layers. Everything else is a frozen SmolVLM2-500M or a Llama-shaped expert.
2. **The towers meet in head space.** They have different widths (960 vs 720) but both project into **15 heads × 64 = 960**. That shared geometry is load-bearing; without it they could not attend jointly.
3. **The objective is flow matching, not DDPM**, and the difference is not cosmetic — it changes *what the loss measures* at every noise level.

---

# Part I — the parameter ledger

## 1. Forced vs chosen

Every dimension is either **forced** by the SmolVLM2-500M checkpoint or **chosen** by SmolVLA. Nothing else exists.

| dim | value | origin |
|---|---:|---|
| VLM hidden `H` | **960** | FORCED — checkpoint |
| VLM heads / head_dim | **15 × 64** | FORCED (15·64 = 960) |
| VLM **kv** heads | **5** | FORCED — GQA |
| VLM FFN | **2560** | FORCED |
| vocab | **49280** | FORCED |
| vision: hidden / layers / patch / image | **768 / 12 / 16 / 512** | FORCED |
| VLM layers kept | **16** of 32 | **CHOSEN** (`num_vlm_layers`) |
| expert hidden `E` | **720** | **CHOSEN** = `int(960 × 0.75)` |
| expert FFN | **2048** | **COMPUTED** — see below |
| expert heads / head_dim | **15 × 64** | **INHERITED — deliberately not scaled** |
| `max_state_dim` / `max_action_dim` | **32 / 32** | CHOSEN |
| `chunk_size` / `n_action_steps` | **50 / 50** | CHOSEN |
| `n_obs_steps` | **1** | CHOSEN |

**The expert FFN is computed, not looked up.** `get_intermediate_size(720)`:

```
int(2·720/3) = 480  →  int(4·480) = 1920  →  256·ceil(1920/256) = 2048
```

The `2/3` then `×4` is the standard SwiGLU correction — a gated FFN has *three* matrices instead of two, so you shrink by 2/3 to match the parameter count of a 4× vanilla FFN. Rounding to a multiple of 256 is tensor-core alignment.

## 2. 🔑 The dimension that is NOT scaled, and why the architecture depends on it

`expert_width_multiplier=0.75` scales hidden size (960→720) and FFN (2560→2048). It does **not** scale `head_dim`. The expert config is a `deepcopy` with only three fields overwritten, so `head_dim=64` and `num_attention_heads=15` survive untouched.

So `720 / 15 = 48`, but head_dim stays **64** — the expert's attention runs in a **960-dim space wider than its own 720-dim residual stream**:

```
expert q_proj : Linear(720 → 960)     ← projects UP into the shared head space
expert o_proj : Linear(960 → 720)     ← projects back DOWN
```

This is not sloppiness. In the self-attention rows the two towers' queries are concatenated **along the sequence axis**:

```python
query_states = torch.cat(query_states, dim=1)   # (B,241,15,64) ⊕ (B,50,15,64) → (B,291,15,64)
```

That `cat` is legal **only** if head count and head_dim match exactly. The width multiplier can shrink the residual stream and the FFN; it can never touch the head geometry.

> **Head space is the shared language.** In: project up. Out: project back down to your own width.

## 3. The ledger, closing exactly

```
One VLM text layer (H=960):
  q_proj  (960,960)   921,600     o_proj  (960,960)   921,600
  k_proj  (320,960)   307,200     v_proj  (320,960)   307,200
  gate/up/down (2560) 7,372,800   2 × RMSNorm            1,920
                                  ───────────────────────────
                                  TOTAL              9,832,320

One EXPERT layer, even index (self-attn):     6,268,320
One EXPERT layer, odd  index (cross-attn):    6,012,320
```

Odd layers are exactly **256,000** smaller because their `k_proj`/`v_proj` are rebuilt as `Linear(320 → 320)` instead of `Linear(720 → 320)`.

```
  vision_model          86,433,024   = 12 × 7,087,872 + 1,377,024 + 1,536
  connector             11,796,480   = 12288 × 960          (no bias)
  text_model           204,626,880   = 47,308,800 + 16 × 9,832,320 + 960
  lm_head               47,308,800   ← dead, see §4
  lm_expert             98,245,840   = 8×6,268,320 + 8×6,012,320 + 720
  ──────────────────────────────────
  SmolVLMWithExpert    448,411,024   ✓ matches measured exactly

  + lerobot projections  1,635,152
      state_proj          Linear(  32 →  960)     31,680
      action_in_proj      Linear(  32 →  720)     23,760
      action_out_proj     Linear( 720 →   32)     23,072
      action_time_mlp_in  Linear(1440 →  720)  1,037,520
      action_time_mlp_out Linear( 720 →  720)    519,120
  ──────────────────────────────────
  GRAND TOTAL          450,046,176
  TRAINABLE             99,880,992   (22.2%)
```

## 4. 🚨 47.3 M parameters — 10.5% of the model — are dead

`vlm.lm_head` is a `(49280, 960)` matrix. `tie_word_embeddings=False`, and we checked `data_ptr()` — it is a genuine second copy of the vocabulary projection, not a view of `embed_tokens`.

The forward pass uses `models = [text_model, lm_expert]` and **never touches `lm_head`**. SmolVLA never generates a token. The matrix is loaded, moved to GPU, and never called.

Practical consequence: ~190 MB of bf16 VRAM you can reclaim by deleting the module after load, if you are memory-constrained on a small GPU.

---

# Part II — the input pipeline

## 5. `resize_with_pad`, and the 25% of every frame that is dead

```python
ratio  = max(cur_width/width, cur_height/height)
resized = F.interpolate(img, size=(rh, rw), mode="bilinear", align_corners=False)
padded  = F.pad(resized, (pad_width, 0, pad_height, 0), value=-1)
```

`max` of the two ratios means **fit-inside**: the larger shrink factor is used, so neither side overflows and aspect ratio is exactly preserved. For a 480×640 camera into 512×512:

```
ratio   = max(640/512, 480/512) = 1.2500     ← width binds
resized = (384, 512)
pad     = (128 rows, 0 cols)
```

`F.pad`'s tuple is `(left, right, top, bottom)`, so `(pad_width, 0, pad_height, 0)` puts **all 128 rows on top and nothing on the bottom**. Verified: row 0 is all `−1`, the last row is not.

> 🚨 **Measured: 65,536 of 262,144 pixels — exactly 25.0% of every frame — are the constant `−1`.** In patch terms, **8 of the 32 patch rows are pure padding**, which survive the ViT and occupy **16 of the 64 output tokens per camera**. With 3 cameras that is **48 of 192 image tokens carrying no information**, at full attention cost.
>
> This is a property of *our rig*: 4:3 cameras into a 1:1 model. Worth knowing before attributing a result to the architecture.

**What `bilinear` does.** Each output pixel is a distance-weighted average of the 4 nearest inputs (2 per axis). Measured 1-D:

```
upsample   [0,1,2,3] ×2  →  [0.0, 0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.0]
downsample [0,1,2,3] ×½  →  [0.5, 2.5]        (pairwise means)
```

Going 640→512 is a 1.25× reduction, but bilinear only ever reads a 2×2 neighbourhood. With no anti-aliasing prefilter, input pixels are effectively **skipped** and high-frequency detail **aliases** rather than averaging away. For a 25 mm cube on a textured table this is not academic — fine edges become sampling-dependent. `align_corners=False` treats pixels as areas rather than points, which is the correct choice for resizing.

## 6. SigLIP → connector: 1024 patches become 64 tokens

512/16 = 32, so **32×32 = 1024 patches**. Patch embedding is a strided conv, `(768, 3, 16, 16)` = 589,824 params — a 16×16 conv with stride 16, i.e. "flatten each patch and project". Position embedding is a learned `(1024, 768)` table — exactly 1024 rows, so the tower is **fixed-resolution by construction**.

The connector is the interesting part:

```
measured:  (B, 1024, 768)  →  (B, 64, 960)
one module: modality_projection.proj.weight = (960, 12288)
```

`scale_factor=4` groups each **4×4 block of patches** into one token: 32×32 ÷ 4×4 = **8×8 = 64 tokens**, each carrying 16 patches × 768 = **12,288 features**. Then one `Linear(12288 → 960)`. That is `12288 × 960 = 11,796,480`, matching the measured count exactly, with no bias.

> 🔑 **This is not pooling.** No averaging, no attention, no learned query — a pure reshape plus one projection. Spatial detail inside a 4×4 block is **folded into the channel dimension**, not discarded, and the projection decides what to keep. Contrast the DP-CNN's `AvgPool` head, which throws spatial structure away irrecoverably. Each output token covers a **64×64 pixel** region.

## 7. Language, state, and the √d asymmetry

**Language** is a plain `(49280, 960)` lookup. `tokenizer_max_length=48` is a cap; `pad_language_to="longest"` means the real length is the longest prompt in the batch — for "pick up the red cube" that is ~8 tokens, not 48.

**Both image and language embeddings are multiplied by √960 = 30.9839.** `state_emb` and `action_time_emb` are **not**.

> 🔑 The asymmetry is precise: **the two streams coming out of the frozen VLM get the VLM's expected scale; the two streams produced by newly-trained `Linear`s do not**, because those layers learn their own scale.

**State** is zero-padded from 6 joints to 32 (`pad_vector` writes `[..., :6]`, leaves 26 zeros), then `Linear(32 → 960)`. This is the cross-embodiment slot. On our arm, **26 of 32 input columns receive no gradient signal**.

> ⚠️ **`n_obs_steps = 1`.** `prepare_state` takes `batch[OBS_STATE][:, -1, :]` — the last frame only. **There is no temporal axis anywhere in SmolVLA.** Our DP-T sees 2 frames; Patch Policy sees 10.

## 8. The prefix, measured on our rig

```
images   3 cams × 64 tok = 192   indices [0, 191]
language                    48   indices [192, 239]
state                        1   index    240
                         ─────
prefix                     241
+ actions                   50   indices [241, 290]
                         ─────
TOTAL SEQUENCE             291
```

---

# Part III — the suffix

## 9. Time is concatenated, not added

```python
action_emb  = action_in_proj(noisy_actions)          # (B,50,32) → (B,50,720)
time_emb    = create_sinusoidal_pos_embedding(...)   # (B,720)
time_emb    = time_emb[:, None, :].expand_as(action_emb)   # view, not a copy
action_time = torch.cat([action_emb, time_emb], dim=2)     # (B,50,1440)
x = action_time_mlp_out(F.silu(action_time_mlp_in(action_time)))   # → (B,50,720)
```

Our DP-T **adds** the timestep embedding into a shared conditioning vector. SmolVLA **concatenates** it to every action token and fuses with a 2-layer MLP. `Linear(1440→720)` is 1,037,520 params — by far the largest of the five new modules.

Concatenation lets the MLP learn an arbitrary interaction between "which action step" and "how noisy"; addition forces them into one shared subspace. **SmolVLA pays 1 M parameters for that freedom.**

## 10. The time embedding differs from DP-T's on purpose

```python
fraction = torch.linspace(0.0, 1.0, 360)
period   = min_period * (max_period/min_period) ** fraction    # 4e-3 · 1000^f
pos_emb  = cat([sin(2π·t/period), cos(2π·t/period)])
```

Measured:

```
period[0]   = 0.004000    fastest: 1570.8 rad per unit t
period[359] = 4.000000    slowest:    1.571 rad per unit t
ratio period[k+1]/period[k] = 1.019428, constant   → GEOMETRIC spacing
‖emb‖ = 18.9737 = √360 for every t                 → constant norm
cos-sim(t=0.50, t=0.55) = 0.2842
```

Three deliberate differences from DP-T's `1/10000^(2i/d)`:

1. **`t` is a real in [0,1]**, not an integer in [0,100). The frequencies resolve a continuum.
2. **`max_period = 4.0 > 1`** — the slowest component completes only **¼ of a cycle** across the whole range of `t`, so it is monotone with no wraparound. DP-T's 10000-based formula wastes most of its band on periods far longer than its input range (see [diffusion-policy-why §13](diffusion-policy-why.md#13-the-timestep-embedding), where we measured roughly half its dims carrying almost no signal).
3. **`min_period = 4e-3`** resolves ~0.4% steps in `t`, comfortably finer than the 0.1 spacing of the 10 inference steps.

---

# Part IV — the mask

```python
cumsum       = torch.cumsum(att_masks, dim=1)
att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]    # i attends j iff block(j) ≤ block(i)
```

This is the `big_vision` / PaliGemma `mask_ar` trick. `att_masks` is built by `+=` statements: `[0]` per image token, `[0]` per language token, `[1]` for state, `[1]` **per action**. Measured block ids:

```
images   [0..191]   → block 0
language [192..239] → block 0        ← SAME block as images
state    [240]      → block 1
actions  [241..290] → blocks 2,3,4,…,51   ← EACH ITS OWN BLOCK
```

Verified consequences:

| relation | result |
|---|---|
| image token → actions | **False** — so prefix KV is cacheable |
| image token → language | **True** — same block, fully bidirectional |
| state → images | True |
| action 0 → state | True |
| **action 0 → action 1** | **False** — strictly causal among themselves |
| action 49 → action 0 | True |

> ⚠️ **An unstated side effect.** The source comment only claims "image, language and state inputs do not attend to action tokens." The per-token block increment for actions (`att_masks += [1] * chunk_size`) *also* makes the action chunk strictly causal internally. Nothing documents this.

It is **block-causal, not causal end to end** — the prefix is a bidirectional blob, then actions are causal after it. Same mathematical object as Patch Policy's mask, but blocked by **modality** rather than by **timestep**.

`position_ids = cumsum(pad_masks) - 1` gives RoPE positions that correctly **skip padded slots**, so positions stay dense when a camera is missing.

---

# Part V — the forward pass

Routing, with `fill_kv_cache=False` and `attention_mode="cross_attn"`, reduces to `layer_idx % 2 == 0`. Traced across 16 layers: **24 attention calls, not 16.**

| layers | calls | mask shapes | actions see each other? |
|---|---|---|---|
| **even** (0,2,…,14) — 8 | 1 joint | `(291,291)` | **yes, causally** |
| **odd** (1,3,…,15) — 8 | 2 separate | `(241,241)` then `(50,241)` | **no** |

## 11a. Mode A — even rows: *join, attend, split*

```
  LEFT (VLM)                             RIGHT (expert)
  in  (1, 241, 960)                      in  (1, 50, 720)
  q_proj → (1,241,960)                   q_proj → (1,50,960)      ← UP
  k_proj → (1,241,320)                   k_proj → (1,50,320)
  v_proj → (1,241,320)                   v_proj → (1,50,320)
  view   → q (1,241,15,64) k (1,241,5,64)   view → q (1,50,15,64) k (1,50,5,64)
        └──────────────────┬──────────────────┘
                           ▼  torch.cat(dim=1)
                  q (1,291,15,64)  k,v (1,291,5,64)

  apply_rope(q), apply_rope(k)
  GQA expand k,v :  (1,291,5,64) → (1,291,15,64)
  q @ kᵀ         :  (1,15,291,64) @ (1,15,64,291) → (1,15,291,291)
  × head_dim^-0.5 = 1/8 ; mask with finfo(fp32).min ; softmax
  probs @ v      :                                → (1,15,291,64)
  reshape        :                                → (1, 291, 960)

  split: rows   0:241 → VLM    o_proj (960→960) → (1,241,960)
         rows 241:291 → expert o_proj (960→720) → (1, 50,720)   ← DOWN
```

Then `+residual → post_attention_layernorm → mlp → +residual`, with the VLM's MLP at 960→2560→960 and the expert's at 720→2048→720.

## 11b. Mode B — odd rows: *the expert borrows keys*

The measured hook output gives the game away in one line:

```
L1 EXP.k_proj  (1, 241, 320) → (1, 241, 320)
                   ↑↑↑
           241, not 50 — it is eating the VLM's keys, not its own tokens
```

```python
expert_key_states = expert_layer.self_attn.k_proj(_key_states)   # Linear(320 → 320)
```

The expert does **not** re-read the prefix hidden states. It re-projects the VLM's *already-computed keys*. It cannot change what the VLM extracted — only **how it reads it**. A learned lens on frozen features.

The attention map is **(50, 241)** — 50 rows, 241 columns, **no column for any action token**. That is the concrete reason actions are invisible to each other in these layers.

## 12. Why 320 and not 960 — GQA

Queries and keys need the same **head width**, not the same total width.

```
num_attention_heads  = 15  →  q is 15 × 64 = 960
num_key_value_heads  =  5  →  k is  5 × 64 = 320
```

15 people asking questions, 5 filing cabinets, each consulted by 3. Before the dot product the 5 kv-heads are expanded ×3 — verified consecutive grouping (q-heads 0,1,2 read kv-head 0).

> ⚠️ **GQA saves memory, not (much) compute.** A common misstatement. Measured MACs per token at L=291:
>
> ```
>                       MHA (15 kv)   GQA (5 kv)      saved
>   projections          3,686,400     2,457,600    1,228,800   (33%)
>   attention core         558,720       558,720            0   ( 0%)  ← identical
>   ─────────────────────────────────────────────────────────
>   total                4,245,120     3,016,320    1,228,800   (29%)
> ```
>
> Once expanded back to 15 heads, `q @ kᵀ` is identical either way. The saving is entirely in the smaller `k_proj`/`v_proj`. The **real** payoff is the KV cache: 29.6 MB → **9.9 MB**, and bandwidth is what bottlenecks the denoising loop.

## 13. The KV cache — all 16 layers, both modes

Decisive test: delete one layer's cache entry and re-run a denoise step.

```
   deleted layer    mode              result
               0  A even      CRASHED: KeyError
               1   B odd      CRASHED: KeyError
               2  A even      CRASHED: KeyError
              14  A even      CRASHED: KeyError
              15   B odd      CRASHED: KeyError
```

Every layer, both modes. **Nothing is stored unused.** The modes differ only in *how* they use it:

```python
# EVEN — forward_attn_layer
key_states = torch.cat([past_key_values[layer_idx]["key_states"], key_states], dim=1)
#            cached 241 prefix keys ⊕ fresh 50 action keys = 291

# ODD — forward_cross_attn_layer
key_states = past_key_values[layer_idx]["key_states"]
#            cached 241 prefix keys alone = 241
```

> 🔑 **The cache exists because the prefix is constant across all 10 denoising steps.** Mode B additionally drops the action keys — which is what makes actions invisible to each other — but that is a *consequence* of the mode, not the reason the cache exists.

---

# Part VI — the objective

## 14. Flow matching vs DDPM

```python
time = Beta(1.5, 1.0).sample() * 0.999 + 0.001
x_t  = time·noise + (1 - time)·actions
u_t  = noise - actions
loss = F.mse_loss(u_t, v_t)
```

### 14a. The path is not variance-preserving

```
   t   | SmolVLA coeffs      Var(x_t) | DP coeffs           Var(x_k)
  0.25 | sig 0.750 noi 0.250    0.625 | sig 0.920 noi 0.391    1.000
  0.50 | sig 0.500 noi 0.500    0.500 | sig 0.703 noi 0.711    1.000
  0.75 | sig 0.250 noi 0.750    0.625 | sig 0.380 noi 0.925    1.000
```

DP satisfies `sig² + noi² = 1` at every step — a rotation. SmolVLA's coefficients are **linear**, so input variance dips to **0.5** at `t=0.5`. It can afford this because `ACTION: MEAN_STD` gives genuinely unit-variance actions, where DP's `MIN_MAX` measured **0.313** (see [diffusion-policy-transformer §11](diffusion-policy-transformer.md#11-the-action-scale-is-not-what-the-config-implies)).

### 14b. 🔑 The action never drops out of the target

Under a toy world where the observation predicts the action with residual sd `r = 0.3`, best achievable MSE:

| t / k | SmolVLA | DP (ε-pred) |
|---:|---:|---:|
| 0.50 | 0.3297 | 0.0806 |
| 0.99 | **0.0916** | **0.0000** |

Because the target is `noise − actions`, **the action term never vanishes** — SmolVLA floors at `r² = 0.09`, i.e. the loss at the most-sampled end *is* the action-prediction error. DP's `ε` target lets the action vanish entirely: at `k=99` the input *is* the noise and predicting it is copying.

This is exactly our k-bucket finding seen from the objective side. Our DP's `k=99` bucket explained **99.98%** of variance and carried **1.2%** of the uniform average — it was measuring almost nothing.

### 14c. Time is sampled toward the informative end

```
Beta(1.5,1.0)·0.999+0.001   mean 0.601   median 0.631
  t ∈ [0.0,0.2):  8.9%
  t ∈ [0.8,1.0): 28.6%
```

Density ∝ √t. **28.6% of samples land where the loss measures action prediction.** Our DP samples `k` uniformly, spending equal budget on a regime that is nearly free to solve.

---

# Part VII — inference

Two distinct passes, both measured.

**Pass 1 — prefix, run ONCE.** `fill_kv_cache=True` short-circuits routing, so **all 16 layers use self-attention** on the 241-token prefix. 16 calls, all `(241,241)`.

```
cache = 16 layers × (key+value) × (1,241,5,64)
      = 2,467,840 floats = 9.9 MB fp32 / 4.9 MB bf16
```

**Pass 2 — denoise, run 10 times.** 16 calls per step: 8 masked `(50,291)`, 8 masked `(50,241)`. **Only 50 tokens are ever forwarded** — the vision tower and the 241-token prefix are computed once for the whole chunk, not once per step.

## 15. Euler integration is exact

```python
dt = -1.0/10
for step in range(10):
    time = 1.0 + step*dt          # 1.0, 0.9, …, 0.1
    x_t  = x_t + dt * denoise_step(x_t, time)
```

With an oracle field `v_t = noise − actions`, measured `‖x_t − actions‖`:

```
step 0  t=1.00  56.7707
step 5  t=0.50  28.3853
end     t=0.00   0.0000   ← EXACT
```

> 🔑 **10 Euler steps are exact**, because the training path is a **straight line** and its velocity is constant in `t`. All error comes from `v_t ≠ u_t` — model error — never from discretisation. This is why SmolVLA needs 10 steps where our DP needs 16 DDIM steps on a curved variance-preserving path.

---

# Part VIII — caveats for our rig

Concrete, measured, and worth knowing before attributing any result to "the architecture":

- **25% of every camera frame is dead padding**, consuming 48 of 192 image tokens. 4:3 cameras into a 1:1 model (§5).
- **47.3 M parameters (10.5%) are a never-called `lm_head`** (§4).
- **Only 1.6 M parameters are genuinely new** — the entire adaptation surface between SmolVLM2 and our arm is five `Linear` layers (§3).
- **26 of 32 state and action slots are always zero** for a 6-DoF arm (§7).
- **Action tokens cannot see each other in half the depth** — 8 of 16 layers are prefix-only (§11b).
- **`n_obs_steps=1` with all 50 steps executed** = **1.67 s of open loop from a single frame** at 30 fps, versus our DP-T's 800 ms. Same open-loop exposure as our ACT.

## 16. A `lerobot` bug to work around

`SmolVLMWithExpertModel.train()` does not `return self`, breaking the `nn.Module` contract that `.train()` / `.eval()` are chainable. So:

```python
m = SmolVLMWithExpertModel(...).eval()   # ← m is None
```

Use two statements instead. This will bite any script using the standard idiom.

---

## Still to measure

Everything above is architecture, measured from `config.json` and forward hooks with **randomly initialised weights**. Not yet measured, and requiring the ~2 GB checkpoint:

- real image-token counts on our actual camera stream
- training throughput and VRAM on the H200
- whether the 25% padding waste shows up as a measurable success-rate cost

---

## See also

- [diffusion-policy-why](diffusion-policy-why.md) — why generative action heads exist
- [diffusion-policy-transformer](diffusion-policy-transformer.md) — DDPM + transformer, the closest comparison
- [diffusion-policy-shapes](diffusion-policy-shapes.md) — the CNN backbone
- [act-shapes](act-shapes.md) — the CVAE baseline
- Source: `lerobot/policies/smolvla/` — `modeling_smolvla.py` (916 lines), `smolvlm_with_expert.py` (561), `configuration_smolvla.py` (159), `processor_smolvla.py` (103)

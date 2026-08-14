# ACT — why it is shaped like that

Everything here is **measured** from `lerobot` 0.6.0 on our config (3 cameras @ 480×640, 6 joints, `chunk_size=50`), not read off the paper.

> 📖 **Companion: [act-shapes](act-shapes.md)** — the reference. Token counts, VRAM, batch-size guidance, the resize trap, and the `kld_loss` reading guide live there and are **not repeated here**. This page is the argument: what problem each piece solves, what is load-bearing, and what is historical accident.

Paper: <https://arxiv.org/abs/2304.13705>

---

## The whole thing on one page

ACT answers the same question as Diffusion Policy — *how do you avoid averaging multimodal demonstrations into an invalid action?* — with a completely different bet.

| | Diffusion Policy | ACT |
|---|---|---|
| where multimodality lives | **in sampling** — run the denoiser 100 times from different noise | **in a latent** `z` — one CVAE draw picks a mode |
| observation reaches the action via | a **global conditioning vector** (192 numbers) | **cross-attention over 902 tokens** |
| output | denoised trajectory, same shape in and out | direct regression, `Linear(512 → 6)` |
| forward passes at inference | 100 (or 16 DDIM) | **1** |
| loss | noise-MSE | **L1** + KL |

**And here is the finding that should reframe the whole architecture: ACT's latent collapses, on purpose.** At inference `z` is set to **zeros** and the CVAE encoder — **33.7% of the model** — never runs. So the mechanism ACT introduced to handle multimodality is switched off exactly when the robot is moving.

What actually runs on the arm is: a shared ResNet-18, a 4-layer transformer encoder over 902 tokens, and **one** cross-attention layer. That is much closer to Patch Policy's "frozen features + thin readout head" than the CVAE framing suggests, and it is the most interesting thing about ACT for anyone trying to build something new.

---

# Part I — the parameter ledger

## 1. It closes exactly

```
  backbone (ResNet-18, SHARED across all 3 cameras)     11,166,912
  vae_encoder            4 layers × 4,333,184           17,332,736
  encoder                4 layers × 4,333,184           17,332,736
  decoder                1 layer  × 5,384,832 + norm     5,385,856
  vae_encoder_cls_embed              (1, 512)                  512
  vae_encoder_robot_state_input_proj (512, 6)                3,584
  vae_encoder_action_input_proj      (512, 6)                3,584
  vae_encoder_latent_output_proj     (64, 512)              32,832
  encoder_robot_state_input_proj     (512, 6)                3,584
  encoder_latent_input_proj          (512, 32)              16,896
  encoder_img_feat_input_proj        Conv2d(512,512,1,1)   262,656
  encoder_1d_feature_pos_embed       (2, 512)                1,024
  decoder_pos_embed                  (50, 512)              25,600
  action_head                        (6, 512)                3,078
  ────────────────────────────────────────────────────────────────
  TOTAL                                                 51,571,590    ✓
```

Per-layer, and these close too:

```
encoder layer                                     4,333,184
  self_attn   in_proj 3×512×512 + 1,536 bias
              out_proj 512×512 + 512                1,050,624
  FFN         Linear(512→3200) + Linear(3200→512)   3,280,512
  2 × LayerNorm                                         2,048

decoder layer                                     5,384,832
  self_attn                                         1,050,624
  multihead_attn  (cross)  identical shape          1,050,624
  FFN                                               3,280,512
  3 × LayerNorm                                         3,072
```

A decoder layer costs exactly `1,051,648` more than an encoder layer: one extra attention block plus one extra LayerNorm.

## 2. 🚨 One third of the model does not run on the robot

```
vae_encoder + cls_embed + 3 projections = 17,373,248 = 33.7% of 51,571,590
```

Verified by hook: `m.train()` → `vae_encoder` called **1×**; `m.eval()` → called **0×**. At inference the code takes this branch unconditionally:

```python
latent_sample = torch.zeros([batch_size, self.config.latent_dim])
```

**Four of the nine transformer layers are training-only.** They are loaded, checkpointed, and moved to GPU, and they never see a robot.

## 3. The backbone is shared, unlike Diffusion Policy's

`11,166,912` is **one** ResNet-18 for all three cameras. Diffusion Policy defaults to `use_separate_rgb_encoder_per_camera=True` — three separate encoders, `33,591,264` params (see [diffusion-policy-shapes](diffusion-policy-shapes.md#the-observation-path)).

Two defensible philosophies, and the repo runs both:

- **ACT (shared)** — a wrist view and an overhead view are both *images*; one extractor generalises across viewpoints and gets 3× the gradient signal per parameter.
- **DP (separate)** — different viewpoints have different statistics and deserve different filters.

Nobody has A/B'd this on our rig. It is a clean, cheap experiment: flip DP to a shared encoder and you free 22 M parameters.

---

# Part II — the CVAE, and why it is switched off

## 4. What the VAE encoder actually sees

This is the part most descriptions get vague about. The VAE encoder's input sequence is:

```
[ CLS ,  robot_state ,  action_0 , action_1 , … , action_49 ]
    1        1                       50                        = 52 tokens
```

Measured: `vae_encoder_pos_enc` is a buffer of shape `(1, 52, 512)` — **fixed sinusoidal, not a parameter**.

> 🔑 **The VAE encoder reads the ground-truth future actions.** That is what makes it a *conditional* VAE: at training time it gets to look at the answer, compress it into 32 numbers, and hand those to the decoder as a hint. `[0]` selects the CLS token output, and `vae_encoder_latent_output_proj: Linear(512 → 64)` splits into `mu` (32) and `log_sigma_x2` (32).

```python
latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
```

The reparameterisation trick: sampling is moved outside the gradient path so `mu` and `sigma` stay differentiable.

**Why this is supposed to help.** If two demonstrations show different valid ways to grasp, a plain regressor must output their average — which is often invalid (the mean of two grasp approaches is a collision). Giving the decoder a latent that *identifies which mode this demonstration is* removes the ambiguity: the decoder is no longer asked to predict a multimodal distribution, only one mode given `z`.

**Why it fails.** At inference there is no ground-truth action to encode, so you must sample `z` from the prior. ACT does not even do that — it uses `z = 0`, the prior's mean. So any mode information the latent learned is unavailable exactly when it is needed.

## 5. The KL term forces the collapse, measured

```python
mean_kld = (-0.5 * (1 + log_sigma_x2 - mu.pow(2) - log_sigma_x2.exp())).sum(-1).mean()
loss = l1_loss + mean_kld * 10.0
```

Standard KL( N(μ,σ) ‖ N(0,I) ), **summed over 32 dims**. Measured cost of actually using the latent:

| latent state | `kld` | × `kl_weight=10` |
|---|---:|---:|
| collapsed, μ=0 σ=1 | 0.0000 | 0.00 |
| barely used, μ=0.1 σ=1 | 0.1600 | 1.60 |
| genuinely informative, μ=1 σ=0.5 | **26.1807** | **261.81** |

The L1 term is of order 0.1–1 in normalised action units. **An informative latent costs ~262 against an L1 term that could at most improve by ~1.** The optimiser is not making a subtle trade-off — collapse is overwhelmingly the cheaper solution, and `kl_weight=10.0` guarantees it.

> 🔑 **This is not a bug and near-zero `kld_loss` is the healthy state** — [act-shapes §kld_loss](act-shapes.md#kld_loss-is-supposed-to-be-0-dont-treat-it-as-a-bug) explains why (high KLD would mean the decoder leans on information that does not exist at rollout). But it does mean the honest description of ACT is: **a CVAE whose latent is regularised into carrying nothing, plus 17 M parameters of scaffolding to achieve that.**
>
> **The design-space question this opens:** if the latent must collapse for train/test consistency, what is the CVAE *for*? Two coherent answers, both untested here — (a) drop it entirely and see if anything changes, which is a free experiment; or (b) keep it and actually **sample `z` from the prior at inference**, turning ACT into a genuine generative policy. ACT's own authors chose neither.

---

# Part III — the encoder: 902 tokens

## 6. Where the observation tokens come from

```
one camera:  (B,3,480,640) → ResNet-18 → (B,512,15,20) → Conv2d(512,512,1,1) → 300 tokens
3 cameras                                                                       900
+ latent token   encoder_latent_input_proj:      Linear(32 → 512)                 1
+ state token    encoder_robot_state_input_proj: Linear(6  → 512)                 1
                                                                              ─────
measured encoder input: torch.Size([902, 2, 512])                               902
```

ResNet-18's stride is 32, so `480/32 × 640/32 = 15 × 20 = 300`. Each token summarises a 32×32 input patch.

**`encoder_img_feat_input_proj` is a 1×1 convolution**, 262,656 params. A 1×1 conv is a per-position linear map across channels — it touches no neighbours. Its only job is to move 512 ResNet channels into the 512-dim transformer space. It happens to be a no-op in width here; it is not a no-op in function.

> 🔑 **This is the single biggest architectural difference from Diffusion Policy.** DP crushes each camera through SpatialSoftmax to **32 (x,y) keypoints = 64 numbers**, then concatenates everything into one 192-number conditioning vector. ACT keeps **900 spatial tokens** and lets attention decide what matters.
>
> DP throws away appearance and keeps geometry, by construction. ACT keeps everything and pays 902× the attention cost. That axis — **how much observation bandwidth reaches the action decoder** — is the cleanest single dimension along which all these policies differ.

## 7. Two position embeddings, and why one of them is 2-D

| what | encoding | params |
|---|---|---:|
| latent + state tokens | **learned** `Embedding(2, 512)` | 1,024 |
| image tokens | **2-D sinusoidal**, parameter-free | 0 |
| VAE encoder positions | fixed 1-D sinusoidal buffer | 0 |
| decoder queries | **learned** `Embedding(50, 512)` | 25,600 |

`ACTSinusoidalPositionEmbedding2d(256)` produces `(B, 512, 15, 20)`: **256 dims encode the row index, 256 encode the column, concatenated.**

Why not a flat 1-D embedding over 300 positions? Because a flat index cannot express *"same column, different row"* — the two axes would be entangled in one scalar, and the encoder would have to learn that position 20 is directly below position 0. The 2-D split makes row and column **independently decodable** by construction.

Measured neighbour similarity, confirming smoothness in both axes:

```
cos-sim (0,0) vs (0,1)  [adjacent column] = 0.9986
cos-sim (0,0) vs (1,0)  [adjacent row]    = 0.9975
```

Parameter-free also means **resolution-independent** — the same embedding works at any feature-map size, which a learned table could not do. That is what makes the inference-time resize trap in [act-shapes §resolution mismatch](act-shapes.md#a-resolution-mismatch-at-inference-does-not-raise) *silent* rather than a crash.

---

# Part IV — the decoder

## 8. The queries are zeros. All of them.

```python
decoder_in = torch.zeros((chunk_size, batch_size, dim_model))
```

Verified by pre-hook on decoder layer 0: input `(50, 2, 512)`, **all zeros, max|x| = 0.000000**.

The 50 action tokens enter carrying **no information whatsoever**. Everything they become comes from `decoder_pos_embed` — a learned `Embedding(50, 512)`, 25,600 parameters — which is added to the *queries* inside the attention call, never written into the token stream.

> 🔑 **The right mental model: `decoder_pos_embed` row *k* is a learned QUESTION, not a token.** "What should the action be 
> at step *k* of a chunk?" Cross-attention carries that question to the 902 observation tokens and the answer comes back as the value-weighted sum. The 50 rows are 50 learned probes into the observation.
>
> This is inherited from DETR, where the same trick makes object queries into "learned slots that go find an object". ACT's contribution was recognising that a temporal chunk index works the same way as an object slot.

## 9. Position goes into Q and K, never into V

```python
q = k = self.maybe_add_pos_embed(x, decoder_pos_embed)
x = self.self_attn(q, k, value=x)[0]              # ← value is RAW x

x = self.multihead_attn(
    query = self.maybe_add_pos_embed(x, decoder_pos_embed),
    key   = self.maybe_add_pos_embed(encoder_out, encoder_pos_embed),
    value = encoder_out,                           # ← value is RAW encoder_out
)[0]
```

**Position steers *where* to look; it never contaminates *what* is retrieved.** Queries and keys decide the attention weights, so position belongs there. Values are the payload that gets summed and written into the residual stream — adding position there would inject coordinates into the content.

Note also `maybe_add_pos_embed` is called **inside every layer**, not once at the input. With `n_decoder_layers=1` that distinction is invisible; it matters if you ever raise the layer count. Contrast [DP-T §5](diffusion-policy-transformer.md#5-the-action-block-how-a-trajectory-becomes-48-tokens), which adds `pos_emb` once and lets the residual stream carry it.

## 10. The shapes

```
self-attention   (50 queries, 50 keys)
  q,k = zeros + decoder_pos_embed        (50,B,512) → 8 heads × 64
  scores  (B,8,50,50)      ← the 50 chunk steps coordinate with each other

cross-attention  (50 queries, 902 keys)          ← the asymmetry that defines ACT
  scores  (B,8,50,902)
  output  (B,8,50,64) → (50,B,512)

FFN              512 → 3200 → 512               (a 6.25× expansion, unusually wide)

action_head      Linear(512 → 6)                → (B,50,6)
```

`dim_feedforward=3200` against `dim_model=512` is **6.25×**, where the transformer convention is 4×. Nobody has ablated it.

## 11. 🚨 The decoder is one layer, because of a bug that was never fixed

```python
# Note: Although the original ACT implementation has 7 for `n_decoder_layers`, there is a bug
# in the code that means only the first layer is used. Here we match the original
# implementation by setting this to 1.
# See https://github.com/tonyzhaozh/act/issues/25
n_decoder_layers: int = 1
```

lerobot's own config says it. **Every published ACT number was produced with a 1-layer decoder**, because the reference implementation silently discarded layers 2–7. lerobot faithfully reproduces the bug so its results match the paper.

This is not a footnote — it is the most load-bearing fact about ACT's design space:

- ACT is a **deep encoder (4 layers over 902 tokens) with a single cross-attention readout.** The heavy lifting is representation, not decoding.
- The paper's claimed decoder depth was never tested. Setting `n_decoder_layers=7` is an **untested, one-line experiment** that no published number covers.
- It reframes what ACT is close to: a big visual encoder plus a thin head — architecturally the same bet as **Patch Policy** (frozen ViT + small action head), reached by accident rather than design.

---

# Part V — inference

## 12. One forward pass, then 50 actions

`n_action_steps=50` with `chunk_size=50` means the whole chunk is executed open-loop. At 30 fps that is **1.67 s between observations** — identical exposure to SmolVLA (see [smolvla §Part VIII](smolvla.md#part-viii-caveats-for-our-rig)), and more than double our DP-T's 800 ms.

## 13. Temporal ensembling, and its sign convention

Default `temporal_ensemble_coeff = None` — **off**. When enabled, overlapping predictions for the same wall-clock timestep are averaged with `exp(-coeff * arange(chunk))`, normalised online:

```
coeff = +0.01   [1.0, 0.99, 0.9802, 0.9704, …]   ← weights SHRINK with index
coeff = -0.01   [1.0, 1.0101, 1.0202, 1.0305, …] ← weights GROW with index
```

Each row *i* is the weight given to a prediction made *i* chunks ago. So:

- **positive `coeff` → the newest prediction dominates** (most responsive, least smooth)
- **negative `coeff` → the oldest prediction dominates** (most smooth, laggiest)

Enabling it forces `n_action_steps=1` — a full chunk is predicted every single step, so inference cost rises **50×**. That is the real reason it is off by default, and it is worth knowing before blaming jitter on the policy.

---

# Part VI — what is load-bearing, and what is accident

The reason to read all of this is to know which parts you may change. Measured, on our config:

**Load-bearing — change these and ACT stops being ACT:**

- **Action chunking.** Predicting 50 steps at once is the idea that every later method adopted (DP, SmolVLA, Patch Policy all chunk).
- **Learned query embeddings as addresses.** 25,600 parameters turn "step *k*" into a probe. This is the mechanism, and it is remarkably cheap.
- **High observation bandwidth.** 902 tokens reaching the decoder through cross-attention, versus DP's 192 numbers.

**Accident or unexamined — fair game:**

- **`n_decoder_layers=1`** — a reproduced upstream bug (§11).
- **The CVAE** — 33.7% of parameters, guaranteed to collapse by `kl_weight=10`, unused at inference (§2, §5).
- **`dim_feedforward=3200`** (6.25×, not the conventional 4×) — never ablated.
- **Shared vs separate camera backbones** — ACT shares, DP does not; never A/B'd here (§3).
- **`kl_weight=10.0`** — chosen to force collapse. Whether a lower weight plus prior sampling at inference would beat it is untested.

**The honest summary.** Strip the collapsed CVAE and ACT is: *shared ResNet-18 → 4-layer transformer encoder over 902 tokens → 50 learned queries → one cross-attention → `Linear(512→6)`.* Roughly 34 M running parameters doing direct L1 regression, with all the capacity in the encoder.

That it ties Diffusion Policy on our arm ([experiments/2026-08-12](../../experiments/2026-08-12_dp-recovery-encoder-ab.md), paired difference −0.018, 7/11 episodes identical — though at n=12 and ~29% power, see [DP-T §16](diffusion-policy-transformer.md#16-what-eval_loss-does-and-does-not-tell-you)) is a real datapoint about how much the generative machinery is buying.

---

## See also

- [act-shapes](act-shapes.md) — the reference: token counts, VRAM, batch sizing, config traps
- [diffusion-policy-why](diffusion-policy-why.md) — the same multimodality problem, solved by sampling instead of a latent
- [diffusion-policy-transformer](diffusion-policy-transformer.md) — a transformer decoder conditioned on 3 memory tokens instead of 902
- [smolvla](smolvla.md) — a frozen VLM in place of a trained encoder
- Source: `lerobot/policies/act/modeling_act.py` (748 lines), `configuration_act.py` (176)

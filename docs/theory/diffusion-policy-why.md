# Diffusion Policy — why it is shaped like that

**Read this one start to finish.** It derives the whole architecture from a single problem, in the order the ideas actually depend on each other, so you can rebuild it at a whiteboard without looking anything up. [diffusion-policy-shapes](diffusion-policy-shapes.md) is the companion **reference** — every tensor measured, costs, and the ACT comparison table. Story here, lookup there.

Config as installed (`lerobot` 0.6.0, all verified against source): `horizon=64`, `n_obs_steps=2`, `n_action_steps=32`, `num_train_timesteps=100`, `beta_schedule=squaredcos_cap_v2`, `prediction_type=epsilon`, `noise_scheduler_type=DDPM`, `clip_sample=True`, `lr=1e-4`.

| Part | Question |
|---|---|
| **[I](#part-i--the-problem-and-the-answer)** | Why does this architecture exist at all? |
| **[II](#part-ii--how-it-sees)** | How do 921,600 pixels become 64 numbers? |
| **[III](#part-iii--how-it-generates)** | Why is every piece of the UNet the piece it is? |
| **[IV](#part-iv--what-the-story-predicts)** | What can you now predict without running anything? |

Each section follows the same shape: **the problem stated exactly → the candidate designs → why the winner won → the shapes.**

## The whole thing on one page

```
 ═══ COMPUTED ONCE per chunk ══════════════════════════════════════════════

   images (8,2,2,3,480,640)      state (8,2,6)          t  (one rung of 100)
            │                          │                        │
     2× ResNet18                       │                    sinusoid
     (16, 512, 15, 20)                 │                    (8, 128)
            │                          │                        │
     1×1 conv 512→32                   │                 MLP 128→512→128
     (16, 32, 15, 20)                  │                    (8, 128)
            │                          │                        │
     soft-argmax                       │                        │
     (16,32,2) → (16,64)               │                        │
            └────────────┬─────────────┘                        │
                    (8, 268)  global_cond                       │
                         └─────────────────┬──────────────────────┘
                                 the condition  (8, 396)
                                           │
                                           │  FiLM into all 14 blocks
 ═══ ×100  the generator ═══════════════════▼══════════════════════════════

     sample xₜ  ──────►  UNet, conv along TIME  ──────►  noise ε̂
     (8, 64, 6)          T: 64 → 32 → 16 → 32 → 64       (8, 64, 6)
          ▲                                                   │
          └──────────────  scheduler.step  ◄──────────────────┘
                            xₜ → xₜ₋₁

 ══════════════════════════════════════════════════════════════════════════

     slice [:, 1:33]  ──►  unnormalize  ──►  action queue
     (8, 32, 6)            MIN_MAX → deg     one (8, 6) per control tick
```

**Read it as one asymmetry.** Everything above the divide runs **once per chunk**; everything below runs **100 times**. That single split explains the latency profile, why adding a camera is nearly free, and why `num_inference_steps` is the only real speed lever.

Three inputs, three fates: the images are crushed **14,400×** into 64 numbers each, the state passes through untouched, and the timestep is *expanded* from one number to 128. All three flatten into one 396-vector that has **no time axis at all** — which is precisely why FiLM (§14) is the only sensible way to inject it. The condition is then computed once and reused identically on every one of the 100 iterations.

The loop is a **cycle, not a pipeline**: input and output are both `(8, 64, 6)`, because this is a repair operation on a trajectory, not a function from observation to action. And then half the answer is discarded — 64 generated, 32 executed.

---
---

# Part I — the problem and the answer

## 1. Averaging destroys behaviour

Our `phi_so101_cubes_cylinder_v1` demonstrations contain roughly **three distinct grasp approaches per object**. That is not noise in the data. It is the data being honest: there really are several correct ways to pick up a cube.

Now train a policy that maps observation → action by regression:

- Minimising **L2** makes the network output the **conditional mean** of the actions seen in that state.
- Minimising **L1** (what ACT uses) makes it output the **conditional median**.

Both are *summary statistics of a distribution*. And here is the alarming part:

> **The average of several valid actions is usually not a valid action.**

If half the demonstrations swing left around the cube and half swing right, the mean is *straight into the cube*. Asked which way around the lamp post, the policy answers "through it." This is not a small-error problem you can fix with more data or a bigger network — **more data makes the average more precisely wrong.** It is a category error: you asked a question with several right answers and forced a format that permits one.

**This is the entire reason Diffusion Policy exists.** Everything below is machinery in service of it.

## 2. The fixes that don't work, including the one we ran

**Add a latent that picks the mode.** This is ACT's CVAE. A latent `z` encodes *which* approach this demo took; the decoder is conditioned on it. In principle the multimodality moves into `z` and the decoder becomes single-valued.

In practice it collapses. The decoder learns to ignore `z`, the KL term drives the posterior to the prior, and you are back to averaging with extra steps. **We measured exactly this:** `kld = 6.7e-5`, output sensitivity **0.067%**, and a β sweep down to 0.1 left KL under **0.02 nats** when representing 3 modes needs at least `ln(3) ≈ 1.10`. Worse, at inference ACT **sets `z` to zero by construction** — so even a healthy latent is discarded when it matters.

**GANs.** No averaging (a discriminator won't accept a blurry compromise), but unstable training and mode collapse — the same disease in a different costume.

**Discretise the action space** (BeT, VQ-BeT). Genuinely works: a classifier over action tokens can put mass on two separate modes without putting any in between. You pay quantisation error and a codebook to maintain.

What you actually want is none of these. You want a **sampler** — something you can ask twice and get two different, individually-valid answers from.

## 3. The reframe that makes sampling easy

Learning the full distribution `p(trajectory | observation)` is hard. But you never needed the distribution itself — you needed to **draw from it**. So change the question:

> Instead of *"what does the distribution look like?"*, learn *"from wherever I am in trajectory space, which direction is more like real data?"*

That is a **direction field** — an arrow at every point. With the arrows, sampling is just: start somewhere random, follow arrows.

Why this is the good trick: **learning a field of arrows is plain supervised regression with MSE.** No adversary, no latent, no KL, no codebook. It is the single thing neural networks are best at. We have converted "model a complicated multimodal distribution" into "fit a smooth vector field," and the multimodality will come out in the *walking*, not in the network.

## 4. Why noise: the arrows must exist *everywhere*

A 64-step, 6-joint trajectory is a point in **ℝ³⁸⁴**. Real trajectories occupy a vanishingly thin sheet in that space — almost every point in ℝ³⁸⁴ is not merely a bad trajectory, it is not a trajectory at all.

Place yourself at a random point. Which way is the sheet? You have no idea, and neither does the network, because it never saw a training example anywhere near you. **The arrows are undefined almost everywhere, which is exactly where sampling has to start.**

The fix is to **blur the data**. Add Gaussian noise of scale σ and the infinitely thin sheet becomes a cloud with actual thickness; now nearby points have a well-defined "which way is home."

- **Small σ**: arrows precise, but only exist very close to real data.
- **Large σ**: arrows exist everywhere, but point only at the coarse blob.

Neither alone works. So use **a ladder of noise levels** — 100 rungs, nearly-clean to pure noise. Each rung's arrows are trustworthy in its own neighbourhood, and consecutive rungs overlap, so you can walk down.

| `t` | signal `√ᾱ` | noise `√(1−ᾱ)` |
|---|---|---|
| 0 | 100.0% | 2.5% |
| 25 | 91.4% | 40.5% |
| 50 | 69.2% | 72.2% |
| 75 | 36.5% | 93.1% |
| 99 | 0.0% | 100.0% |

`squaredcos_cap_v2` (cosine) rather than linear, because a linear schedule destroys the signal too early: by mid-ladder there is nothing left to learn from and half your rungs teach the network to map noise to noise.

## 5. Why the training target is *the noise*

```python
eps = torch.randn(trajectory.shape)                       # (8, 64, 6)
timesteps = torch.randint(0, 100, (B,))                   # one rung per sample
noisy = sqrt(ᾱ_t) * trajectory + sqrt(1-ᾱ_t) * eps        # corrupt it
loss = mse(unet(noisy, t, obs), eps)                      # "which part is garbage?"
```

We corrupt a known-good trajectory by a known amount and ask the network to name the corruption. **Why does naming the noise give the arrows?** Because the corruption is exactly what stands between you and the data:

```
x₀ = (x_t − √(1−ᾱ_t)·ε) / √ᾱ_t
```

"Which part of what I'm looking at is garbage" and "which way is home" are the same question. Formally `∇ₓ log p_t(x) = −ε / √(1−ᾱ_t)` — the predicted noise **is** the score function, up to a known scale.

**Why predict `ε` rather than `x₀`?** Algebraically identical, numerically not. `ε` has unit variance at *every* rung, so the target is the same size whether `t=1` or `t=99` — constant-scale target, stable gradients. Recovering `x₀` at `t=99` means dividing by `√ᾱ ≈ 0.0005`.

**Free sanity check:** an untrained network outputs ≈0 against a unit-variance target, so **initial loss must be ≈ 1.0** (we measured 1.0569). If a run doesn't start there, it is miswired — catch it before burning GPU hours.

## 6. Why sampling takes many small steps — the part that actually solves multimodality

> We can predict `ε`. From pure noise, why not predict it once, solve for `x₀`, and be done in one pass?

Because an MSE-optimal denoiser returns **`E[x₀ | x_t]`** — the *expectation* over every clean trajectory consistent with what it sees. At `t=99` the input carries no information, so that expectation is the average of every trajectory in the dataset.

**A one-shot jump lands on the mean. That is precisely the problem we started with.**

So take a **small** step: nudge toward the estimate, then **add a little noise back**, landing at `t=98` — a slightly more specific place. Ask again. Repeat 100 times, committing a little more each step.

```python
for t in scheduler.timesteps:               # 99, 98, ... 0
    eps_hat = unet(sample, t, global_cond)
    sample  = scheduler.step(eps_hat, t, sample).prev_sample   # step + re-noise
```

The randomness injected at each step is not a wart. **It is what picks which mode you land in.** Two runs from different starting noise walk different paths and arrive at the left-hand grasp and the right-hand grasp — each a real trajectory, neither an average.

> ### 🔑 The multimodality lives in the sampling process, not in the network.
> No latent to collapse. No KL. No β. The failure mode we spent a week chasing on ACT **cannot occur here**, because nothing compresses mode identity into a bottleneck the decoder is free to ignore.

**DDIM** sets the re-injected noise to zero — deterministic, so you can take much larger strides down the ladder (~10 steps instead of 100) at the cost of sample diversity. That is the fix for the Mac, where DDPM defaults measured **9551 ms per chunk, 9.0× over budget**.

**`clip_sample=True`** clamps the implied `x₀` to `[-1, 1]` every step. Safe because actions are `MIN_MAX`-normalised to exactly that box — the two numbers are the same number for a reason. It stops one bad early estimate throwing the chain off the manifold.

---
---

# Part II — how it sees

The observation path turns **two cameras × 3×480×640** into **268 numbers**, and then never looks at an image again. Almost all of the compression happens in one 20-line module.

## 7. The trilemma: what do you do with a feature map?

ResNet18 (minus its last two layers) gives you `(512, 15, 20)` per image. The policy needs a vector. Three honest options:

| Option | Output size | Keeps | Destroys |
|---|---|---|---|
| Flatten | 153,600 | everything | nothing, but unusable — position entangled with identity |
| Global average pool | 512 | *what* is present | **all position** |
| **Spatial softmax** | **64** | ***where* things are** | *what they look like* |

Average pooling is what image classifiers do, and it is **catastrophic for manipulation**. "There is a red cube somewhere in this image" tells an arm nothing. The whole task is *where*.

So spatial softmax is not a compression trick that happens to be small. It is a **deliberate bet that manipulation is a geometry problem.**

## 8. The 1×1 conv — choosing what deserves a coordinate

```python
features = self.nets(features)     # Conv2d(512, 32, kernel_size=1)
```

**You already know a 1×1 convolution: RGB → greyscale is one.** `gray = 0.299·R + 0.587·G + 0.114·B` is a weighted sum across channels, applied independently at every pixel, same weights everywhere — exactly `Conv2d(3, 1, kernel_size=1)`. This is that, with 512 inputs, 32 outputs, and learned weights.

A `kernel_size=1` conv has **no spatial extent** — its window is one cell — but a conv kernel always spans **every input channel**. At each cell it reads that cell's 512-vector and emits a 32-vector, then slides on with the **identical** weights:

```
   at cell (7,11):  a 512-vector  ──[ W : 32 × 512 ]──►  a 32-vector
   …same W at all 300 cells, nothing crosses between them
```

⇒ **A 1×1 conv is a `Linear` layer applied per position, with weights shared across positions.** `H` and `W` come out unchanged.

**Where does the 32 live?** Stop picturing a solid cuboid. A `(512, 10, 12)` tensor is a **flat 10×12 grid with a skewer standing at each of the 120 positions**, every skewer holding 512 beads. The conv walks the grid and swaps each 512-bead skewer for a 32-bead one. No square moves, merges or disappears — only the *height* of the skewers changes. The 32 lives down the depth axis, exactly where the 512 was.

And the reduction at one skewer is **32 dot products**, not squeezing or pooling. `W` is `(32, 512)` = 32 rows, each 512 long; each row is a *recipe* that multiplies the whole column and sums to one number.

> **It is not compression by throwing things away. It is compression by asking fewer, better questions** — all 512 values are consulted every time, 32 times over.

Shrunk to hand-checkable size (4 channels → 2, on a 2×3 grid), with `recipe 0 = [1, 1, 0, 0]` and `recipe 1 = [0, 0, 1, −1]`:

```
INPUT   (1, 4, 2, 3)      every square hides a column of 4
   (0,0) [1, 2, 3, 4]     (0,1) [10, 11, 12, 13]     (0,2) [100, 101, 102, 103]
OUTPUT  (1, 2, 2, 3)      grid UNCHANGED, depth 4 -> 2
   (0,0) [3, -1]          (0,1) [21, -1]             (0,2) [201, -1]

square (0,0), column [1,2,3,4]:
   recipe 0:  1·1 + 1·2 + 0·3 +  0·4  =  3
   recipe 1:  0·1 + 0·2 + 1·3 + -1·4  = -1
```

Note the second output channel reads `−1` at *every* square. Not a bug: recipe 1 computes a **difference**, and in that test data channels 2 and 3 always differ by 1. A difference-recipe is blind to the overall level and reports only contrast — which is precisely the sort of invariance the real layer learns ("reddish minus gripper-coloured," firing on the cube regardless of room brightness), and why negative weights matter. A recipe that can only add can never build an invariance.

**Weights are few; the work is many.** `16,384` weights = 32 recipes × 512 inputs, but `4,915,200` multiply-adds because the recipe book is consulted at all 300 squares. That **300× ratio is weight sharing**, and it is also why this is called a convolution at all: the sliding and the sharing are both present, the window just happens to be 1×1. All of the weight sharing, none of the neighbourhood.

It doesn't resemble a matmul, it *is* one (measured):

```
conv output      (4, 32, 15, 20)
matmul output    (4, 32, 15, 20)      W = conv.weight.view(32, 512)
max |difference| 8.94e-07             y = W @ x.reshape(B,512,300) + b

weight shape             (32, 512, 1, 1)   -- the trailing 1,1 carry no information
params  1×1 conv         16,416
params  3×3 conv (same)  147,488   = 9× more
```

**What "32 combinations" means.** Output channel *k* is `Σ_c W[k,c]·feature_c` — a learned recipe over the 512 backbone detectors. Weights can be **negative**, so a recipe can subtract: *"reddish and roundish but not gripper-coloured."* It can synthesise detectors the backbone lacks, including by cancellation. And since only 32 survive, it is a **learned budget**: of everything ResNet noticed, these 32 things are worth a coordinate.

**It also sets the precision.** This conv feeds the softmax directly, so the *magnitude* of its output is the softmax temperature. Larger weights → peakier distribution → tighter keypoint. The same 16,416 parameters choose both **what** gets located and **how sharply** (see §10).

## 9. SpatialSoftmax — the index/value swap

### The deep idea

In an ordinary CNN feature map:

```
feature[k][i][j]  =  "how much of detector k is at position (i, j)"
  ^index  ^index         ^value
```

Spatial softmax **promotes position from an index to a value**:

```
keypoint[k]  =  (x, y)  =  "where detector k is"
  ^index         ^value
```

`H` and `W` are not pooled or summarised. They are **consumed and re-emitted as the output itself.** That is why the output is `(B, K, 2)` and not `(B, K)`: **the trailing axis of size 2 is not a feature dimension, it is a coordinate frame.** Once you see that, everything else follows.

### The shapes

Take the docstring's `(512, 10, 12)`. Think **512 sheets of paper, each 10×12** — not "a 3D tensor."

**(a) Reshape — the cuboid becomes a rug of rows.**

```python
features = features.reshape(-1, self._in_h * self._in_w)   # (512, 10, 12) -> (512, 120)
```

`reshape` moves no data. Sheet *k* is read **row-major** — row 0 left to right, then row 1 — and laid out as one line of 120 numbers, becoming row *k*.

```
  cuboid (512, 10, 12)                 mat (512, 120)

  ┌──────────────┐ sheet 0      row   0 │ r0 │ r1 │ r2 │ … │ r9 │   ← 10 runs of 12
  ├──────────────┤ sheet 1      row   1 │ r0 │ r1 │ r2 │ … │ r9 │
  ├──────────────┤   ⋮            ⋮
  └──────────────┘ sheet 511    row 511 │ r0 │ r1 │ r2 │ … │ r9 │
```

**(b) Softmax — one belief per channel.** `F.softmax(features, dim=-1)`; `dim=-1` is the spatial axis. **One softmax per channel**, each row summing to 1. Rows never look at each other (see the docstring trap in §11).

**(c) `pos_grid` — a lookup table, not an image.** Built once, constant, `(120, 2)`, read as: *"cell i is at x = `pos_grid[i][0]`, y = `pos_grid[i][1]`."* It is not data, it is the ruler.

**(d) The matmul — the shared axis is summed away.**

```python
expected_xy = attention @ self.pos_grid     # (512, 120) @ (120, 2) -> (512, 2)
```

```
        attention                pos_grid            keypoints
        (512, 120)         @      (120, 2)     =      (512, 2)

     ┌────────────────┐        ┌─────┬─────┐       ┌─────┬─────┐
 k → │ p p p  …  p    │        │  x  │  y  │   k → │E[x] │E[y] │
     │                │        │  ⋮  │  ⋮  │       │     │     │
512  │                │   120  │  ⋮  │  ⋮  │  512  │     │     │
     └────────────────┘        └─────┴─────┘       └─────┴─────┘
      └───── 120 ─────┘         └── 2 ──┘           └── 2 ──┘
              │                     │
              └──── shared ─────────┘   summed over, disappears
```

```
keypoints[k][0] = Σᵢ  attention[k][i] · pos_grid[i][0]     ← E[x]
keypoints[k][1] = Σᵢ  attention[k][i] · pos_grid[i][1]     ← E[y]
```

The index `i` appears on **both** inputs and on **neither** output — the definition of a contracted axis. Equivalently `torch.einsum('ki,id->kd', attention, pos_grid)`: `i` missing from the output means `i` is summed, so **120 vanishes.**

> **The frame worth keeping.** The last two operations contract **perpendicular axes**:
>
> | | contracts | leaves alone | einsum |
> |---|---|---|---|
> | **1×1 conv** | the **channel** axis, 512 → 32 | `H`, `W` | `'chw,kc->khw'` |
> | **SpatialSoftmax** | the **spatial** axis, 300 → gone | channels | `'ki,id->kd'` |
>
> Same contraction, rotated 90°. **The conv decides *what* to track; the softmax reports *where* it is.**

**(e) Where the batch went.** The real call is `(B, K, H, W)`, and `reshape(-1, H*W)` folds **both** B and K into rows:

```
(4, 32, 15, 20)  ->  (128, 300)  @  (300, 2)  ->  (128, 2)  ->  view(4, 32, 2)
```

**Batch and channel are both passengers.** The operation only ever acts on the spatial axis, which is why it is four lines.

### 🚨 The invariant that makes it correct

Both flattenings must use the **same cell ordering**: `features.reshape(-1, H*W)` on a contiguous `(…, H, W)` gives index `h*W + w`, and `pos_x.reshape(H*W, 1)` on the `(H, W)` meshgrid gives the same. If they ever diverge, every keypoint is silently mirrored or transposed **with no error raised.**

> Line 443 uses `np.meshgrid` rather than `torch.linspace` because the latter "causes a small degradation in pc_success of pre-trained models." Someone chased a real regression to a float difference in a coordinate grid.

## 10. A 2×3 example you can check by hand

Six cells, so `pos_grid` is `(6, 2)`:

```
cell 0 (r0,c0)  x=-1.0  y=-1.0        cell 3 (r1,c0)  x=-1.0  y=+1.0
cell 1 (r0,c1)  x=+0.0  y=-1.0        cell 4 (r1,c1)  x=+0.0  y=+1.0
cell 2 (r0,c2)  x=+1.0  y=-1.0        cell 5 (r1,c2)  x=+1.0  y=+1.0
```

One channel with a single bump at cell 1:

```
activations = [[0, 2, 0],       flattened -> [0, 2, 0, 0, 0, 0]
               [0, 0, 0]]
softmax     -> [0.081, 0.596, 0.081, 0.081, 0.081, 0.081]
keypoint    -> (+0.000, -0.516)      but the bump is at (0.0, -1.0)
```

**Look at y.** The peak sits at `y = -1`; the answer is `-0.516`. The five background cells, 8% of the mass each, dragged the keypoint **half way to the image centre.** Sharpen the bump from 2 to 6:

```
softmax  -> [0.002, 0.988, 0.002, 0.002, 0.002, 0.002]
keypoint -> (+0.000, -0.985)
```

⇒ **Localisation accuracy is entirely a function of how peaked the distribution is.** There is no learned temperature in this port (robomimic has one; LeRobot hardcodes it to 1), so the network controls its own precision *implicitly* by learning larger 1×1-conv weights. **Confidence and precision are the same parameter here.**

**Why *soft* and not `argmax`?** Two reasons. `argmax` has zero gradient almost everywhere, so you could never train through it — the expectation is smooth and reaches **all 300 cells** weighted by probability. And **sub-pixel precision**: hard argmax returns one of 300 grid positions, each covering a 32×32 pixel patch, so you would eat ±16 px of quantisation error. The weighted mean lands *between* cells.

## 11. Two docstring traps, and three ways keypoints lie

**"applying channel-wise softmax"** reads like softmax *across* the 512 channels. It is not — `dim=-1` is space. "Channel-wise" means *done separately per channel*. Consequence: **channels never compete.** All 32 keypoints fire on every frame, independently, each always returning a coordinate. There is no way to say "not me."

**"a learnable linear mapping (in_channels, H, W) -> (num_kp, H, W)"** is true but hides the structure — it is a 1×1 conv, per-cell and neighbour-blind, not a dense layer over the flattened map. See §8.

Three lies:

**The dumbbell.** If a channel fires on two objects, its expected position is the midpoint — where there is nothing. **The centre of mass of a dumbbell is in empty air.** This is the exact averaging pathology from §1, one layer earlier in the same network.

**Silence looks like the centre.** Flat activations — object absent, occluded, out of frame — give a uniform distribution, whose expectation over a `[-1,1]` grid is exactly **(0, 0)**. *"I see nothing"* and *"it is dead centre"* are the same output. §10 shows this is a continuum, not a special case.

**Magnitude is discarded.** Softmax normalises, so a faint smudge and a blazing detection give the same confident coordinate. **The keypoint carries no confidence.**

## 12. Then `S` dies, and what all this costs us

The camera axis is absorbed into features (`64 + 64 = 128`), the 6 joint values are concatenated (**134 per observation step**), and `.flatten(start_dim=1)` collapses the history axis:

```
(8, 2, 134)  ->  (8, 268)
```

**The past stops being a time axis.** Nothing tells the network that features 0–133 are "now" and 134–267 are "33 ms ago" — it learns that from fixed ordering, which is why `n_obs_steps` must match between training and deployment. What `S=2` buys: **velocity is implicitly available.** ACT at `n_obs_steps=1` sees only positions.

**The cost on Φ data.** Keypoints encode *where*, not *what*. Our dataset has three objects — 25 mm red cube, 45 mm white cube, yellow cylinder. A channel tuned to "cube-shaped thing" returns the same coordinate whichever cube it sees. Identity must be encoded **indirectly, as a pattern across the 32 channels**. Possible, but far weaker than ACT's 300 full-dimensional tokens per camera, which carry appearance outright.

The dumbbell risk is *low* for us by design — we stage one object and one container at a time. **The identity risk is not.**

---
---

# Part III — how it generates

`(8, 64, 6)` in, `(8, 64, 6)` out, conditioned on 396 numbers. 95.5% of the model lives here.

## 13. The timestep embedding

**The problem.** One network must perform **100 different jobs**, selected by an integer. And neighbouring rungs need *nearly identical* behaviour while distant ones need different behaviour — so the encoding must express *"how far apart, at every scale."*

**❌ Raw scalar.** Every other input is unit-variance; a number ranging to 99 either dominates the first layer or forces its weights to vanish. And one number times one weight is one direction of influence — the network must carve 100 regions out of a single axis.

**❌ One-hot, 100 dims.** Now every `t` is perfectly distinguishable, which is the problem (measured):

```
t=47 vs t=48:   one-hot cos-sim 0.0        sinusoid cos-sim 0.9706
t=47 vs t=55:   one-hot cos-sim 0.0        sinusoid cos-sim 0.7187
t=47 vs t=95:   one-hot cos-sim 0.0        sinusoid cos-sim 0.5002
```

**Every pair is equally far apart.** Rung 48 is as unrelated to 47 as 95 is. You had the most valuable prior available — *adjacent noise levels are almost the same job* — and binned it. Integers only, too.

**❌ Learned embedding table.** Dense and learnable, but starts with **no notion that 47 and 48 are close**; it must spend capacity discovering what you already knew.

**⚠️ Binary encoding.** *Genuinely the right idea* — bit 0 flips every step, bit 6 every 64, so you get **multi-scale structure for free**. But it is **discontinuous**: 31 → 32 flips six bits, so adjacent `t` can be maximally distant in code space.

**✅ Sinusoidal — binary made continuous.** Replace each bit with a wave. Fast waves are the low bits, slow waves the high bits, and everything is smooth. Similarity decays with distance, at every scale. It also takes real-valued `t`, which is why the whole diffusion and flow-matching field standardised on it.

```python
half_dim = self.dim // 2                                # 64
emb = math.log(10000) / (half_dim - 1)
emb = torch.exp(torch.arange(half_dim) * -emb)          # 64 frequencies
```

```
freq[0]  = 1.000000      wavelength      6.28     ← fastest hand
freq[31] = 0.010758      wavelength    583
freq[63] = 0.00010000    wavelength 62,832        ← slowest hand
```

**A clock face with 64 hands**, each ticking at a different rate. Read all 64 and you know the time at every resolution at once.

**Why `sin` *and* `cos`?** `sin` alone is not injective over a period:

```
t=2.000   sin(f·t)=+0.8415   cos(f·t)=+0.5403
t=4.283   sin(f·t)=+0.8415   cos(f·t)=−0.5403
```

Same sine, different `t`. The pair is a point on the unit circle, which **does** identify the phase uniquely.

**Shapes:**

```
t                        (8,)          one integer per sample
  .unsqueeze(-1)         (8, 1)
  × freqs (1, 64)        (8, 64)       outer product
  cat(sin, cos)          (8, 128)      64 angles -> 128 numbers
  Linear(128,512)→Mish→Linear(512,128) (8, 128)
  cat with global_cond   (8, 396)      -> feeds every FiLM encoder
```

**Why the MLP on top?** The sinusoid is a fixed handcrafted **code** — it says *which rung*, not what to do there. The MLP turns *rung* into *behaviour* and lets the network rotate out of the sinusoidal basis. Cost **131,712 params, 0.05% of the UNet.**

> ⚠️ **A real finding about our config.** The slowest wavelength is **62,832** and `num_train_timesteps` is **100** — those bands traverse 0.16% of a cycle across the entire schedule. Measured, `|emb(1) − emb(0)|` averages **0.579** in the fast bands and **1.76e-04** in the slow ones. This encoding was designed for transformer sequences of ~10,000 positions and reused for `t ∈ [0,100]`, so **roughly the upper half of the 128 dims carries almost no signal.** Harmless at 0.05% of parameters, but expect nothing from raising `diffusion_step_embed_dim`.

## 14. FiLM — the deepest choice in the architecture

**The problem, stated exactly.** The UNet is a trajectory → trajectory function, and every operation inside it is defined *along time*. Now make the answer depend on what the robot sees, given `cond` = 396 numbers.

> **The trajectory has a time axis. The conditioning does not.**
> `(8, 6, 64)` ← has a `64`.  `(8, 396)` ← has no time at all.

A camera frame is a **snapshot**. So: how do you inject something with no time axis into a pipeline where time is the only axis anything happens along? Every design below answers that one question.

**❌ Concatenate to the input.** Tile to `(8, 396, 64)`, concat → `(8, 402, 64)`. Three failures. **It costs 67×** — the first conv goes from `6·512·5 = 15,360` weights to `402·512·5 = 1,029,120`. **It only enters at the front door**, so block 10 sees it only after ten convolutions have smeared it. And **it tells the network a lie**: you handed it 64 identical copies and labelled them a time-varying signal.

**❌ Cross-attention.** The conditioning is one vector, not a sequence. Attention with a single key means `softmax` over one element, which is always `1.0` — the mechanism collapses to a learned projection. **Attention only earns its cost when there is something to select *among*.**

**⚠️ Add a bias at every layer.** Cheap, enters everywhere. **This is a real option** (`use_film_scale_modulation=False`). It fails as follows (measured):

```
                  t0    t1     t2    t3
out[c]           2.0   5.0   -1.0   3.0     the raw feature
× 0.3, + 1.0     1.6   2.5    0.7   1.9     kept, compressed
× 0.0, + 1.0     1.0   1.0    1.0   1.0     pattern ERASED
+ (−10), no ×   -8.0  -5.0  -11.0  -7.0     intact, just shifted
```

`a.diff() == (a-10).diff()` is `True`; with `scale=0` it is `False`.

> **Addition is invertible. Multiplication by zero is not.** Add a constant and you can always subtract it back — no information is lost. The observation can *bury* a feature under an offset, but never *remove* it.

**✅ FiLM — a per-channel scale *and* bias.** `scale = 0` deletes a channel. But the mute is only the visible symptom; the real reason is structural:

```
additive:   out = f(x) + g(c)          ∂out/∂x = f′(x)          ← no c anywhere
FiLM:       out = g(c)·f(x) + h(c)     ∂out/∂x = g(c)·f′(x)     ← c sets the sensitivity
```

Measured, sweeping `c` over 0, 1, 5:

```
additive  f(x)+g(c)       →  [2.0,  2.0,  2.0]
FiLM      g(c)·f(x)+h(c)  →  [0.0,  3.0, 15.0]
```

> **With additive conditioning, how strongly the network responds to the trajectory is fixed — it cannot depend on what the robot sees.** The two contributions are *separable*: you could compute `f(x)` in one room and `g(c)` in another and post the answers.

Additive conditioning can express *"given what I see, add this offset."* It can **never** express *"this trajectory feature matters **only if** the cube is on the left."* That sentence contains an **if**, and an *if* is a product. You already know this pattern: attention's core is `QKᵀ`, a **product**, not `Q + K`, for exactly this reason. **FiLM is the cheap version of the same idea.**

**Shapes** (mid block, `C=2048`, `T=16`, `B=8`):

```
cond                     (8, 396)       one vector per sample, no time axis
  Linear(396, 4096)      (8, 4096)      4096 = 2 × 2048: one scale + one bias per channel
  .unsqueeze(-1)         (8, 4096, 1)   ← manufactures a length-1 TIME slot
  split at 2048          scale (8, 2048, 1)   bias (8, 2048, 1)

out from conv trunk      (8, 2048, 16)
scale * out + bias       (8, 2048, 16)  ← the 1 was stretched to 16
```

Broadcasting aligns from the right and **stretches any axis of size 1**:

```
scale[b,c] is ONE number s.        out[b,c] is 16 numbers.

   out:     [ a₀    a₁    a₂   …   a₁₅ ]
   result:  [ s·a₀  s·a₁  s·a₂  …  s·a₁₅ ]  + b
               ↑     ↑     ↑
             the same s, all 16 times
```

**Why per-channel granularity?** It was a choice:

| Granularity | Knobs per block | Verdict |
|---|---|---|
| one scalar for the tensor | 2 | too coarse — a single master volume |
| **per channel `(C, 1)`** | **4,096** | **chosen** — a channel *is* a feature |
| per (channel, timestep) `(C, T)` | 65,536 | rejected |

Per-(channel, timestep) is the obvious "more expressive" option and would give the conditioning a temporal address. Rejected for two reasons. **Principled:** the observation genuinely *is* a snapshot, so 16 different values along time means inventing information never measured. **Arithmetic:** `Linear(396, 2·2048·16)` = **25,952,256 params in one block**; across 14 blocks, **363M** — larger than the entire 255M UNet. Not a trade-off, an impossibility.

> 🚨 **The mixing desk has its faders taped down.**
> The conv trunk is the band, producing 2048 tracks. FiLM is the desk: 2048 faders (`scale`) and 2048 DC offsets (`bias`). The observation sets the desk **once, before the song**, and cannot ride the faders during it. A fader at zero mutes a track permanently; an offset can only bury one.
>
> ⇒ **The observation selects *which kind* of trajectory is produced. It cannot say "at step 12, do this."** All temporal structure must come from the convolutions reading the noisy trajectory.

And the price is small:

```
UNet total            255,425,670
FiLM cond_encoders     11,382,784   (4.5%)
timestep encoder          131,712   (0.05%)
conv trunk + rest     243,911,174   (95.5%)
```

## 15. GroupNorm

**Two problems.** *Why normalise at all?* Activation scale drifts through layers; too large and gradients explode, too small and they vanish, and either way no single learning rate works everywhere. *Then the real question:* normalising means subtract a mean and divide by a std, both computed by pooling numbers. So —

> **Which numbers share a mean and std?**

That single question generates the entire family. Nothing else differs. On `(B, C, T) = (8, 512, 64)`:

| | pools over | one statistic per | batch-dependent? |
|---|---|---|---|
| **BatchNorm** | `(B, T)` | channel | **yes** ← disqualifying |
| **InstanceNorm** | `(T,)` | (sample, channel) | no |
| **LayerNorm** | `(C, T)` | sample | no |
| **GroupNorm(8)** | `(C/8, T)` | (sample, group) | no |

**❌ BatchNorm.** Excellent in its native setting (large-batch image classification). Here it is disqualifying for *deployment*, not accuracy. Inference runs **batch 1, one hundred times per chunk**. Measured:

```
GroupNorm : batch-1 output == batch-8 output for the same sample?   True
BatchNorm : same check                                              False
```

BatchNorm papers over this with running averages at eval time — meaning **train-time and eval-time are literally different functions**, and any drift in those statistics degrades you silently, on hardware, with no error.

**❌ InstanceNorm.** Batch-independent, so the fatal flaw is gone. But it normalises **every channel to unit scale independently**, destroying *relative* magnitude between channels — and "this channel is firing hard and that one isn't" is real information.

**⚠️ LayerNorm.** Batch-independent ✓, relative magnitudes preserved ✓. But it pools **all 512 channels into one statistic**, and channels are not interchangeable. One loud channel inflates the shared std and **shrinks the other 511**.

**✅ GroupNorm(8).** Batch-independent ✓, magnitudes preserved within a group ✓, contamination confined to 64 channels instead of 512 ✓. And it is exactly the **dial between the two failures**: `G=1` *is* LayerNorm, `G=C` *is* InstanceNorm.

Shape unchanged, `(8, 512, 64) → (8, 512, 64)`. One detail: `n_groups` is fixed at 8, so **group size grows** as the UNet widens —

```
GroupNorm(8,  512) -> 8 × 64      GroupNorm(8, 2048) -> 8 × 256
```

Honestly: 8 is the config default and performance is known to be flat across a wide range of `G`. Not a tuned number.

> **Grading on a curve, and choosing the class size.** BatchNorm grades you against whoever happened to sit the exam with you — unacceptable when you sit it alone. LayerNorm grades you against every subject you took. InstanceNorm grades each subject only against itself. GroupNorm splits subjects into departments and grades within department.

## 16. The convolution arithmetic

Three different length behaviours are needed, each a separate requirement:

| | Requirement | Why |
|---|---|---|
| inside a block | length **preserved** | the residual add needs matching shapes; skips must align later |
| going down | length **halved** | `k=5` at full resolution needs ~16 layers to span 64 steps |
| coming up | length **exactly doubled** | must land back on the skip's length, to the element |

**Preserve — `Conv1d(k=5, padding=2)`.** Measured `L=64 -> 64`. `padding = k // 2` preserves length for any **odd** `k`; even kernels are off by one because they have no centre tap.

And **non-causality is not a separate decision.** Pad symmetrically and centre the window, and position `t` necessarily reads `t−2 … t+2`, **including later steps**. Choosing "preserve length" *is* choosing "this is a brush that revises the whole chunk," not "a pen that walks forward." The absence of autoregression in Diffusion Policy is encoded in one `//`.

**Halve — strided conv, not pooling.** `Conv1d(k=3, s=2, p=1)`: `floor((64 + 2 − 3)/2) + 1 = 32`. Pooling is parameter-free but discards by a **fixed rule**; a strided conv **learns what to keep** for the price of one small kernel.

**Double — `ConvTranspose1d(k=4, s=2, p=1)`:** `(32−1)·2 − 2 + 4 = 64`.

**Why `k=4` and not `k=3`?** A transposed conv **scatters** each input across `s` output positions, and coverage is uneven unless `k` divides `s`. Set all weights and inputs to 1 and the output *is* the tap count:

```
k=4 s=2 p=1:  taps [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1]   uniform
k=3 s=2 p=1:  taps [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1]     <-- ALTERNATES
k=5 s=2 p=2:  taps [2, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 2]     <-- ALTERNATES
k=2 s=2 p=0:  taps [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  uniform
```

> 🚨 **The rule: `k` must be divisible by `s`.** That alternating `1, 2, 1, 2` is the **checkerboard artifact** (Odena et al., 2016). In images it is a visible grid. **Here the axis is time, so it is a period-2 ripple in the generated trajectory** — every other joint command systematically over- or under-weighted, at 30 Hz. Not cosmetic: that is the arm buzzing. `k=4` is the smallest kernel satisfying the constraint at `s=2`. **Check this before editing `down_dims` or the up-block kernels.**

## 17. Residuals, skips, and Mish

**The residual.** `out = out + self.residual_conv(x)`, where `residual_conv` is `Identity` when channels match and otherwise **a 1×1 conv** — the same operation from §8, used purely to remap channel count so the addition is legal. Each block therefore learns a **correction** rather than a replacement, and the gradient gets a highway. A 14-block stack is only trainable because of this.

**Skips concatenate, they don't add.** Addition requires matching channels and *asserts the two tensors live in the same space*. Concatenation lets the following conv **learn** the combination, including learning to ignore one. The deciding question is whether the merged things *are* the same kind of thing:

> **ResNet adds because the skip is a correction to the same quantity. U-Net concatenates because the two branches mean different things** — one is the plan, the other is the detail.

```
mid output    (8, 2048, 16)
skip[2]       (8, 2048, 16)
cat(dim=1)    (8, 4096, 16)      <- hence `dim_in * 2` in the up blocks
ResBlock      (8, 1024, 16)
```

**Mish, honestly.** `Mish(x) = x·tanh(softplus(x))` — smooth everywhere, self-gating, no dead units. The target here is a **smooth continuous vector field**, and ReLU is piecewise-linear so its outputs carry creases. But the gap to GELU or SiLU is small and largely empirical. This is the paper's choice, kept by LeRobot. **Of everything on this page it is the one I would expect to swap with no measurable effect.**

---
---

# Part IV — what the story predicts

If you understand the above, you can call these before running anything:

- **Initial training loss ≈ 1.0.** Not approximately-ish — unit-variance noise against a near-zero output.
- **Latency scales linearly in `num_inference_steps` and not at all in camera count** past the single vision pass. Measured 95.5 ms per denoise step on MPS.
- **Sampling the same observation twice gives different trajectories** — a free diagnostic ACT structurally cannot offer. Run it 10× from one observation and *look* at the spread. A tight spread on multimodal data means something is wrong.
- **Object-identity confusion arrives before trajectory failure**, because of the keypoint bottleneck (§12).
- **`n_action_steps=32` of `horizon=64` executes**, and it is steps **1–32**, not 0–31 — see shapes.
- **No latent to collapse**, so `kld` diagnostics are meaningless here and the β sweep has no analogue.

## The whiteboard version

1. Demonstrations are multimodal; regression outputs the mean of the modes; **the mean of two valid grasps is a collision.**
2. So don't predict the action — learn a **direction field** toward real trajectories and sample by following it.
3. That field is undefined in empty space, so **blur the data across 100 noise levels**, learning all 100 with one network selected by a timestep embedding.
4. Train by corrupting a real trajectory and asking **"which part is noise?"** — plain MSE, and that answer *is* the field.
5. Sample by walking down in small steps, **re-injecting noise each time**; one big jump lands on the mean again, and **the per-step randomness is what commits to one mode.**

And two companions for the components:

> **The feature map becomes a rug of rows; softmax turns each row into a belief over cells; the coordinate table says where each cell is; the matmul is a weighted average that consumes the cell axis and leaves one (x, y) per row.** — and the centre of mass of a dumbbell is in the empty middle.

> **The mixing desk has its faders taped down.** FiLM multiplies rather than adds because addition cannot delete, and because `f(x) + g(c)` makes the network's sensitivity to the trajectory independent of what it sees — which forbids any *if*.

---

## See also

- [diffusion-policy-shapes](diffusion-policy-shapes.md) — the measured tensor-by-tensor trace, VRAM and latency, config traps, and the ACT comparison table
- [act-shapes](act-shapes.md) — the same treatment for ACT, including the CVAE collapse
- [`experiments/2026-08-04_act-8bin-6run.md`](../../experiments/2026-08-04_act-8bin-6run.md) — where that collapse was measured on our own data

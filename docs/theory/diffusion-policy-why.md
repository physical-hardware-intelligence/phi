# Diffusion Policy — why it is shaped like that

[diffusion-policy-shapes](diffusion-policy-shapes.md) traces every tensor and answers *what*. This one answers *why*, in the order the ideas actually depend on each other, so you can rebuild the argument at a whiteboard without looking anything up.

Config as installed (`lerobot` 0.6.0, verified): `horizon=64`, `n_obs_steps=2`, `n_action_steps=32`, `num_train_timesteps=100`, `beta_schedule=squaredcos_cap_v2`, `prediction_type=epsilon`, `noise_scheduler_type=DDPM`, `clip_sample=True`, `lr=1e-4`.

---

## 1. The problem: averaging destroys behaviour

Our `phi_so101_cubes_cylinder_v1` demonstrations contain roughly **three distinct grasp approaches per object**. That is not noise in the data. It is the data being honest: there really are several correct ways to pick up a cube.

Now train a policy that maps observation → action by regression:

- Minimising **L2** makes the network output the **conditional mean** of the actions seen in that state.
- Minimising **L1** (what ACT uses) makes it output the **conditional median**.

Both are *summary statistics of a distribution*. And here is the thing that should feel alarming:

> **The average of several valid actions is usually not a valid action.**

If half the demonstrations swing left around the cube and half swing right, the mean is *straight into the cube*. Asked which way around the lamp post, the policy answers "through it". This is not a small-error problem you can fix with more data or a bigger network — more data makes the average **more precisely wrong**. It is a category error: you asked a question with several right answers, and forced a format that permits one.

**This is the entire reason Diffusion Policy exists.** Everything else in the architecture is machinery in service of it.

---

## 2. The fixes that don't work, including the one we ran

**Add a latent that picks the mode.** This is ACT's CVAE. A latent `z` encodes *which* of the several approaches this demo took; the decoder is conditioned on it. In principle the multimodality moves into `z` and the decoder becomes single-valued.

In practice it collapses. The decoder learns to ignore `z`, the KL term drives the posterior to the prior, and you are back to averaging with extra steps. **We measured exactly this:** `kld = 6.7e-5`, output sensitivity **0.067%**, and a β sweep down to 0.1 left KL under **0.02 nats** when representing 3 modes needs at least `ln(3) ≈ 1.10`. Worse, at inference ACT **sets `z` to zero by construction** — so even a healthy latent is discarded when it matters.

**GANs.** No averaging (the discriminator won't accept a blurry compromise), but unstable training and mode collapse, which is the same disease in a different costume.

**Discretise the action space** (BeT, VQ-BeT). Genuinely works — a classifier over action tokens can put mass on two separate modes without putting any in between. You pay quantisation error and a codebook to maintain.

What you actually want is none of these. You want a **sampler**: something you can ask twice and get two different, individually-valid answers from.

---

## 3. The reframe that makes sampling easy

Learning the full distribution `p(trajectory | observation)` is hard. But notice you never needed the distribution itself. You needed to **draw from it**.

So change the question:

> Instead of *"what does the distribution look like?"*, learn *"from wherever I am in trajectory space, which direction is more like real data?"*

That is a **direction field** — an arrow at every point. And if you have the arrows, sampling is just: start somewhere random, follow arrows.

Why this is the good trick: **learning a field of arrows is plain supervised regression with MSE.** No adversary, no latent, no KL, no codebook. It is the single thing neural networks are best at. We have converted "model a complicated multimodal distribution" into "fit a smooth vector field", and the multimodality will come out in the *walking*, not in the network.

Everything from here is engineering to make those arrows well-defined.

---

## 4. Why noise: the arrows have to exist *everywhere*

A 64-step, 6-joint trajectory is a point in **ℝ³⁸⁴**. Real trajectories occupy a vanishingly thin sheet in that space — almost every point in ℝ³⁸⁴ is not merely a bad trajectory, it is not a trajectory at all.

So place yourself at a random point. Which way is the sheet? You have no idea, and neither does the network, because it never saw a single training example anywhere near you. **The arrows are undefined almost everywhere, which is exactly where sampling has to start.**

The fix is to **blur the data**. Take real trajectories and add Gaussian noise of scale σ. The infinitely thin sheet becomes a fuzzy cloud with actual thickness, and now points near it have a well-defined "which way is home".

- **Small σ**: cloud is still thin. Arrows are precise but only exist very close to real data.
- **Large σ**: cloud fills space. Arrows exist everywhere, but they only point at the coarse blob — they can tell you "trajectories are roughly over there", not which grasp to use.

Neither alone works. So use **a ladder of noise levels** — 100 rungs, from nearly clean to pure noise. Each rung's arrows are trustworthy in its own neighbourhood, and the rungs overlap, so you can walk from the top of the ladder to the bottom.

That ladder is the `ᾱ` schedule:

| `t` | signal `√ᾱ` | noise `√(1−ᾱ)` |
|---|---|---|
| 0 | 100.0% | 2.5% |
| 25 | 91.4% | 40.5% |
| 50 | 69.2% | 72.2% |
| 75 | 36.5% | 93.1% |
| 99 | 0.0% | 100.0% |

`squaredcos_cap_v2` (cosine) rather than linear because a linear schedule destroys the signal too early: by the middle of the ladder there is nothing left to learn from, and half your rungs are wasted teaching the network to map noise to noise. The cosine schedule lingers in the useful high-signal region.

---

## 5. Why the training target is *the noise*

The recipe is three lines:

```python
eps = torch.randn(trajectory.shape)                       # (8, 64, 6)
timesteps = torch.randint(0, 100, (B,))                   # one rung per sample
noisy = sqrt(ᾱ_t) * trajectory + sqrt(1-ᾱ_t) * eps        # corrupt it
loss = mse(unet(noisy, t, obs), eps)                      # "which part is garbage?"
```

We corrupt a known-good trajectory by a known amount and ask the network to name the corruption. **Why does naming the noise give us the arrows?** Because the corruption is exactly the thing standing between you and the data:

```
x₀ = (x_t − √(1−ᾱ_t)·ε) / √ᾱ_t
```

"Which part of what I'm looking at is garbage" and "which way is home" are the same question. Formally, `∇ₓ log p_t(x) = −ε / √(1−ᾱ_t)` — the predicted noise **is** the score function, up to a known scale. The arrows, exactly.

**Why predict `ε` rather than the clean trajectory `x₀`?** Algebraically identical, numerically not. `ε` has unit variance at *every* rung, so the target is the same size whether `t=1` or `t=99` — constant-scale target, stable gradients across the whole ladder. Recovering `x₀` at `t=99` means dividing by `√ᾱ ≈ 0.0005`.

**Free sanity check:** an untrained network outputs ≈0, and the target is unit-variance noise, so **initial loss must be ≈ 1.0** (we measured 1.0569). If your run doesn't start there, something is miswired — check that before burning GPU hours.

One network learns all 100 denoising jobs at once; the timestep embedding tells it which one it's doing right now. That's why `t` is fed in at all, and why it's a **sinusoidal** embedding rather than a raw scalar: the behaviour needed at `t=95` (decide coarse structure) is genuinely different from `t=5` (polish), so the network needs a rich, smooth, multi-scale code for "how broken is this".

---

## 6. Why sampling takes many small steps — the part that actually solves multimodality

Here is the question that unlocks the whole design:

> We can predict `ε`. From pure noise, why not predict `ε` once, solve for `x₀`, and be done in a single pass?

Because an MSE-optimal denoiser returns **`E[x₀ | x_t]`** — the *expectation* over every clean trajectory consistent with what it's looking at. At `t=99` the input carries no information, so that expectation is the average of every trajectory in the dataset.

**A one-shot jump lands you on the mean. That is precisely the averaging problem we started with.** All that work, same failure.

So take a **small** step instead. Nudge toward the estimate, then **add a little noise back**, landing at `t=98`: a slightly more specific place. Ask again. Now the network has marginally more to work with, and its expectation is over a marginally narrower set. Repeat 100 times, and each step commits a little more.

```python
for t in scheduler.timesteps:               # 99, 98, ... 0
    eps_hat = unet(sample, t, global_cond)
    sample  = scheduler.step(eps_hat, t, sample).prev_sample   # step + re-noise
```

The randomness injected at each step is not a wart. **It is the thing that picks which mode you end up in.** Two runs from different starting noise walk down different paths and land on the left-hand grasp and the right-hand grasp respectively — each one a real trajectory, neither one an average.

> ### 🔑 The multimodality lives in the sampling process, not in the network.
> There is no latent variable to collapse. No KL term. No β to tune. The failure mode we spent a week chasing on ACT **cannot occur here**, because nothing in this architecture is trying to compress mode identity into a bottleneck that the decoder is free to ignore.

Two consequences worth holding onto:

**DDIM** sets the re-injected noise to zero. Deterministic, so you can take much larger strides down the ladder — ~10 steps instead of 100 — at the cost of sample diversity. That is the recommended fix for the Mac, where DDPM defaults measured **9551 ms per chunk, 9.0× over budget**.

**`clip_sample=True`** clamps the implied `x₀` to `[-1, 1]` at every step. Safe because actions are normalised, and it stops one bad early estimate from throwing the chain off the data manifold entirely — a guardrail on a 100-step walk.

---

## 7. Now every architectural choice is forced

With the argument in hand, the shapes doc stops being a list and becomes a set of consequences.

| Design choice | Why it could not be otherwise |
|---|---|
| **Input and output are the same shape** `(8,64,6)` | It is a repair operation on a trajectory, not a function from observation to action. |
| **Observation enters via FiLM, not as tokens** | The arrows must be conditioned on what you see — but the observation says *which kind* of trajectory is wanted, not what to do at step 12. Hence one global scale/bias broadcast over all 64 steps, with **no temporal address**. |
| **Vision runs once, reused 100×** | The conditioning doesn't change as you descend the ladder. Only the trajectory does. So the ResNets are outside the loop, and `num_inference_steps` is the only real latency lever. |
| **Convolution along *time*** | What makes a stretch of motion plausible is the same at step 3 as at step 40. Motion is translation-invariant in time, so share weights along `T`. Joints are *not* interchangeable, so mix them fully as channels. **Share over time, mix over joints.** |
| **Non-causal kernels** (`padding = k//2`) | You are revising a whole chunk simultaneously, not rolling one forward. There is no autoregression anywhere in Diffusion Policy. |
| **A UNet, not a flat stack** | Step 60 must be consistent with step 2, so the receptive field has to span all 64 steps. Downsampling time gets there in a few layers instead of ~16. |
| **GroupNorm, not BatchNorm** | Inference is batch-1, run 100 times. The function must be batch-independent or training and deployment diverge. |
| **92% of parameters in the UNet** | Generating is the job. Seeing is a side condition. |

And one choice that is **not** forced by any of this:

**SpatialSoftmax crushing each image to 32 (x,y) keypoints — 64 numbers, a 14,400× compression.** That is a Diffusion Policy *paper* decision, not a diffusion requirement, and it is the architecture's soft spot for us. The policy learns **where** things are and almost nothing about **what** they are. One duck and one box: fine. **A red cube and a white cube of different sizes** — our actual dataset — is close to the adversarial case for a keypoint bottleneck, where ACT's 300 full-dimensional tokens per camera still carry appearance.

---

## 8. What the story lets you predict

If you understand the above, you can call these before running anything:

- **Initial training loss ≈ 1.0.** Not approximately-ish — it is unit-variance noise against a near-zero output.
- **Latency scales linearly in `num_inference_steps` and not at all in camera count** (past the single vision pass). Measured: 95.5 ms per denoise step on MPS.
- **Sampling the same observation twice gives different trajectories.** This is a free diagnostic ACT structurally cannot offer: run it 10 times from one observation and *look* at the spread. Tight spread on multimodal data means something is wrong; a healthy model should show you the three grasp approaches.
- **`n_action_steps=32` of `horizon=64` is executed.** The discarded tail exists so the convolutions have context beyond the part you use — it is not waste.
- **No latent to collapse**, so `kld` diagnostics are meaningless here and the β sweep has no analogue.
- **Expect object-identity confusion before you expect trajectory failure**, because of the keypoint bottleneck.

---

## 9. The whiteboard version

1. Demonstrations are multimodal; regression outputs the mean of the modes; **the mean of two valid grasps is a collision.**
2. So don't predict the action — learn a **direction field** pointing toward real trajectories, and sample by following it.
3. That field is undefined in empty space, so **blur the data across 100 noise levels** to give it definition everywhere, and learn all 100 levels with one network selected by a timestep embedding.
4. Train by corrupting a real trajectory and asking **"which part is noise?"** — plain MSE, and that answer is the direction field.
5. Sample by walking down the ladder in small steps, **re-injecting noise each time.** One big jump would land on the mean again; **the per-step randomness is what commits to one mode**, which is why there is no latent here to collapse.

---

## See also

- [diffusion-policy-shapes](diffusion-policy-shapes.md) — the tensor-by-tensor trace, measured costs, and the ACT comparison table
- [act-shapes](act-shapes.md) — the same treatment for ACT, including the CVAE collapse
- [`experiments/2026-08-04_act-8bin-6run.md`](../../experiments/2026-08-04_act-8bin-6run.md) — where the collapse was measured on our own data

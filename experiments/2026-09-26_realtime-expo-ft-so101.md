# Real-Time EXPO-FT on the SO-101 — RL post-training a VLA on a moving target

**Status**: 📋 **Planned — Phase 0 next.** No code, no runs. · **Owner**: Yash · **Created**: 2026-09-26

> The club's second RL track. [RL in sim](2026-08-02_rl-in-sim-so101.md) trains SAC from scratch in simulation. This one post-trains a **pretrained VLA on the real arm**. Different question; they may share learner code (M8).

**How to use this doc.** Work through the checklist in [§6](#6-plan-step-by-step) top to bottom. Tick each box in the PR that completes it, and put every number in [Results](#results) as it arrives.

**Every number carries one tag:**
`[paper §x]` Real-Time EXPO-FT, arXiv 2609.18207 · `[code: file]` the authors' repo, [pd-perry/expo-ft](https://github.com/pd-perry/expo-ft) · `[M#]` measured in Phase 0 · `[derived]` computed on this page · `[estimate]` replace with a measurement · `[my reading]` interpretation, stated nowhere

---

## The question

**Does Real-Time EXPO-FT's Dynamic Picking result hold on a five-joint, $400 arm — and is the critic's ranking what actually does the work?**

## Why do this

1. **It tests the paper's core claim where the paper isolates it.** Reading the camera frame at execution time instead of the stale one is individually significant on one task only: Dynamic Picking, 24 → 30 of 30, p = 0.024 `[derived §2.14]`. So that is the task we copy.
2. **It is a different arm.** The paper used a Franka. Ours has five plastic joints, backlash, and a servo P-gain of 16 against a default of 32 ([control-loop §3](../docs/theory/control-loop.md)).
3. **It builds what the laundry goal is missing.** Larchenko's prize-winning folding recipe ran its RL in simulation only. His real-robot round was plain BC *"because there was no real-side reward or value"* ([arXiv 2606.27163](https://arxiv.org/abs/2606.27163) §10.5). This experiment builds that reward + value loop on our arm.

---

## 1. One decision, on a timeline

```
control step (33.3 ms)    t     t+1    t+2    t+3    t+4   …   t+10
                          │
VLA reads s_t ────────────┤ inference, d = 3 steps
                          │◄──────── d ──────►│
arm executes              last 3 actions of the OLD chunk (the committed prefix)
                                              │
editor + critic read s_{t+3} ─────────────────┤ pick 1 of 64, inside one step
arm executes                                  8 actions of the NEW chunk: t+3 … t+10
```

The whole method is in that picture. The expensive model plans from an old frame; the cheap models correct from the newest one.

---

## 2. The method: every concept, with its maths

Each symbol is defined before it is used. Each formula is followed by a number on our arm.

### 2.1 Time

```
f       = 30 Hz              control rate                              [paper V-C]
Δ       = 1/f = 33.3 ms      one control step
H       = 16                 actions predicted per chunk               [paper Tab III]
C       = 8                  actions executed before replanning        [paper Tab IV]
t_inf                        wall time of one VLA inference            [M1]
d       = ⌊t_inf · f⌋ + 1    inference delay, in steps                 [paper IV-A]

constraint:   1 ≤ d ≤ C      if d > C the committed actions run out before the new chunk exists
```

Worked: `t_inf = 67 ms` → `d = ⌊0.067 × 30⌋ + 1 = ⌊2.01⌋ + 1 = 3`. That is the paper's value. The three clocks involved (policy, action playback, servo PID) are in [control-loop §1](../docs/theory/control-loop.md).

### 2.2 Staleness — the reason fresh frames matter

The action executed at step `t + d + k`, for `k = 0 … C−1`, was computed from:

```
VLA:      s_t        staleness  d + k  steps
editor:   s_{t+d}    staleness  k      steps

mean over one chunk:   VLA   d + (C−1)/2          editor   (C−1)/2
```

With `C = 8`:

| d | VLA staleness | editor staleness | removed by reading the fresh frame |
|---|---|---|---|
| 3 | 6.5 steps = 217 ms | 3.5 steps = 117 ms | 46% |
| 5 | 8.5 steps = 283 ms | 3.5 steps = 117 ms | 59% |
| 8 | 11.5 steps = 383 ms | 3.5 steps = 117 ms | 70% |

`[derived]` **This gives a prediction:** the fresh-frame gain should grow with `d` (H3).

**What staleness costs on a turntable.** A block at radius `r` on a plate turning at `ω` rad/s moves at `v = ω · r`. If the policy aims where the block *was*:

```
miss distance  ≈  v × staleness        an upper bound: a policy that leads the target does better
```

Worked, `ω = 0.6 rad/s`, `r = 0.10 m`, so `v = 60 mm/s`, at `d = 3`: VLA misses by 13 mm, editor by 7 mm. With a grasp tolerance near ±10 mm `[M6]`, that is the difference between a miss and a grasp.

**Choosing the plate speed.** The task is only "dynamic" if the VLA's miss exceeds the tolerance `T` while the editor's does not:

```
T / 0.217 s  <  v  <  T / 0.117 s
T = 10 mm:   46 mm/s  <  v  <  85 mm/s
r = 0.10 m:  0.46     <  ω  <  0.85 rad/s     =  4.4 to 8.2 RPM
```

`[derived, T from M6]` That is the band where this method can show anything at all. A real policy partly leads the target, so the true band sits somewhat higher. Phase 3's success gate is what actually decides the speed.

One more fact makes the task learnable from single frames: at constant `ω`, a block's velocity is fixed by its position on the plate. One image is enough to predict where it will be.

### 2.3 The VLA: flow matching

```
A      the clean action chunk, H × 6 normalised joint targets
ε      noise of the same shape, ε ~ N(0, I)
t      flow time in [0, 1]

x_t  = t·ε + (1 − t)·A          t = 0 is clean, t = 1 is pure noise
u    = ε − A                    the velocity the network must predict
loss = ‖ v_θ(x_t, t, s) − u ‖²
```

`[code: lerobot/policies/smolvla/modeling_smolvla.py, forward]` · architecture: [theory/smolvla](../docs/theory/smolvla.md)

**Sampling.** Start at `x = ε` with `t = 1`. Take 10 Euler steps of `Δt = −0.1`, each `x ← x + Δt · v_θ(x, t, s)`. At `t = 0`, `x ≈ A`. `[smolvla num_steps = 10; paper Tab III]`

**Why a flow and not ACT.** The method needs *many different chunks from one frame*: 32 noise seeds give 32 chunks. ACT zeroes its latent at inference (`lerobot/policies/act/modeling_act.py:456`) and returns the same chunk every time.

> ⚠️ **Sign trap.** The paper writes `A^τ = τA + (1−τ)ε`, so its `τ = 1 − t`. Where the paper says "prefix at flow time 1", LeRobot needs **`t = 0`**. Copying the number literally labels clean actions as pure noise, and nothing raises an error.

### 2.4 Training-time RTC: teaching the VLA to continue a chunk

At inference the first `d` actions of every new chunk are already committed, because they are executing while it is computed. Train for exactly that situation:

```
pick   d ~ Uniform{0, …, d_max}              d_max = 10 during SFT     [code: scripts/dynamic_pick/offline_train.sh]

positions 0 … d−1:     x = A                 time = 0       clean, given
positions d … H−1:     x = t·ε + (1−t)·A     time = t       noised, predicted

mask   m_k = 0 for k < d,  1 otherwise
loss   = Σ_k m_k ‖ v_θ(x)_k − u_k ‖²  /  Σ_k m_k
```

- **Online** (during RL), `d` is fixed to the deployment delay, and set to 0 for the first chunk of each episode `[paper VII-E2]`.
- **At inference**, feed the committed `d` actions as the clean prefix, denoise positions `d … H−1`, and execute positions `d … d+C−1`.
- **Why:** the arm never pauses at a chunk boundary, and never jumps, because the new chunk is trained to agree with the motion already under way.

**What exists, and what must be written.** LeRobot 0.6.0 ships only *inference-time* RTC, which steers sampling and changes no training. This loss has to be written. SmolVLA already attaches a time embedding to each action token separately ([smolvla §9](../docs/theory/smolvla.md)), so per-position time is a small change: pass `time` with shape `[B, H]` instead of `[B]`.

> Two different things share one name. The paper's "w/ RTC" baselines use **training-time** RTC. LeRobot's `--inference.type=rtc` ([deployment](../docs/deployment/README.md)) is **inference-time** RTC. Do not mix them.

### 2.5 Candidates

32 noise seeds, so 32 chunks from one frame. The VLM prefix (images, state, text) is computed once and cached; only the action expert runs 32 times `[paper VII-E2, Tab III]`.

Keep positions `d … d+C−1` of each. One candidate `a_i` is then `C × 6 = 48` numbers.

### 2.6 The editor

```
input    the latest frame s_{t+d}, and one candidate a_i
network  MLP 3 × 256 on [ critic image embedding (512), proprio embedding (64), a_i (48) ]
output   μ, log σ ∈ R^48

sample   u   = μ + σ ⊙ ξ,           ξ ~ N(0, I)
edit     â_i = 0.1 · tanh(u)        edit_scale = 0.1                     [paper Tab IV]
edited   ã_i = a_i + â_i
```

`tanh` keeps every entry in `[−1, 1]`, so each of the 48 numbers moves by **at most ±0.1**. What 0.1 means in degrees, and in millimetres at the gripper, is measured in `[M4]`.

The paper edits only position and gripper for picking (`edit_action_xyzg = True`). In joint space there is no clean position/rotation split, so we edit all six joints `[deviation]`.

### 2.7 Selection

```
candidates   = { a_1 … a_32 } ∪ { ã_1 … ã_32 }              64 chunks
Q_min2(s, a) = min( Q_j(s, a), Q_k(s, a) )                 j, k drawn at random from 10 heads
execute      a* = argmax over the 64 of  Q_min2(s_{t+d}, ·)     no softmax    [paper Eq 6, VII-E2]
```

The image goes through the critic's encoder once. The 64 candidates only pass through the small heads.

**Why min-of-2: the maximization bias, computed.** Suppose each Q estimate is the truth plus independent noise of size `σ`. The argmax of 64 noisy scores preferentially picks the one with the luckiest noise:

| scoring | average error of the chosen candidate |
|---|---|
| one head | **+2.34 σ** |
| min of 2 heads | **+1.29 σ** |

`[derived: 400k-sample simulation, iid Gaussian head errors]` Min-of-2 removes about 45% of the bias, not all of it. What remains is why the critic has to be validated (H4). The idea is from [TD3](https://arxiv.org/abs/1802.09477) and [REDQ](https://arxiv.org/abs/2101.05982).

### 2.8 The critic

**Architecture.** Its own encoder: pre-activation ResNetV2, stages `(3, 4, 6, 3)` of basic blocks, GroupNorm with 4 groups, 64 → 512 filters, on two camera views stacked as 6 channels at 224². That gives a 512-d image embedding; proprioception is embedded to 64-d; both are concatenated with the flattened chunk and fed to **10 heads**, each 3 × 256 with LayerNorm `[paper VII-E2; code: configs/model/expo_ft_pi_config.py]`.

**One training example is one chunk:**

```
(s_t,  a_{t:t+C},  r,  s_{t+C},  done)

r    = 1 if the success detector fired inside this chunk, else 0
y    = r + γ^C · (1 − done) · Q'_min2( s_{t+C}, ã*_{t+C} )
loss = ( Q_φ(s_t, a) − y )²          all 10 heads regress onto the same y
```

- `γ = 0.99` per step `[code: configs/model/td_config.py]`, so per chunk `γ^C = 0.99⁸ = 0.9227`. The per-chunk exponent is implied by the paper's note that RLPD, unlike the method, "uses γ per environment step rather than γ^C" `[paper VII-F]`.
- `ã*_{t+C}` is the chunk the policy *would* pick at the next boundary, found through the noise filter (§2.10).
- `Q'` is a slow copy: `φ' ← 0.995 φ' + 0.005 φ` after each step, which halves the old weights' share every 138 steps `[paper Tab III; derived]`.

**What a Q value means here.** The only reward is a 1 at success, so Q is "how soon will I succeed", squeezed into (0, 1]:

| success arrives at chunk k | Q at chunk 0 = 0.9227^k |
|---|---|
| 6 (1.6 s) | 0.62 |
| 12 (3.2 s) | 0.38 |
| 24 (6.4 s) | 0.15 |

`[derived]` Episodes time out at 200 steps, which is 25 chunks `[code: configs/task/dynamic_pick.py]`.

### 2.9 Training the editor, and its temperature

```
L_edit = E[ α · log π_edit(â | s, a)  −  Q_min2(s, a + â) ]
```

Minimising this raises Q, while the `α log π` term keeps the editor from collapsing to one fixed nudge.

**How the gradient gets there.** `â = 0.1 · tanh(μ + σξ)` is a differentiable function of `μ` and `σ` once `ξ` is drawn. This is the reparameterisation trick, and it lets `∂Q/∂â` flow back into the editor. **The VLA receives none of this gradient.**

`log π_edit` needs the change-of-variables correction for tanh ([SAC](https://arxiv.org/abs/1801.01290), Appendix C):

```
log π(â) = log N(u; μ, σ)  −  Σ_i log(1 − tanh²(u_i))
```

Compute it on the `[−1, 1]` output, before the ×0.1. That is the standard SAC convention, and it is the one in which the entropy target below means what it says. Including the scale would shift `log π` by `−48 · log 0.1 = +110`, and α would run away. Check which convention the reference code uses (§10 #7).

**Temperature α** is learned:

```
L_α = −α · ( log π_edit(â) + H_target ),      H_target = −D/2 = −24,   D = 48
```

If the editor's entropy falls below the target, α rises and pushes it back up, so it keeps exploring. If entropy is above target, α falls. Start at `α₀ = 0.01`. Entropy enters `L_edit` only, **never the critic's target `y`** `[paper VII-E2]`.

### 2.10 The noise filter (training only)

Building `ã*_{t+C}` for each critic target would cost 32 full VLA decodes. Instead:

```
Q_f(s, ε)      scores 32 raw noise seeds (H × 32, the padded action shape) without decoding
ε*             = argmax Q_f
decode ε* once → a;   draw one edit â;   ã* = argmax over {a, a + â} of Q'_min2
L_f            = ( Q_f(s, ε*)  −  stopgrad[ Q'_min2(s, ã*) ] )²
```

One decode per target instead of 32. Rollouts never use it `[paper Eq 10–11, from FASTER]`.

### 2.11 The VLA keeps learning

Each update call makes one step of the §2.4 loss, on **successful episodes only**: the demos plus successful rollouts `[code: actor_success_only = True; paper VII-E2]`.

This is the only way the VLA's weights change. The critic never sends it a gradient. Successes only, because BC copies whatever it is shown, including failing.

### 2.12 The update schedule, and what "UTD 20" means

```
K = 25        new environment steps per update call               [paper Tab IV]
one call:     20 critic steps on disjoint minibatches of 64
              → 1 VLA step → 1 editor step → 1 α step
calls are queued during an episode and run at its end; learning starts after 10 episodes   [paper VII-E4]
```

So "UTD 20" means 20 critic steps **per call**, not per environment step. Per environment step it is `20 / 25 = 0.8` `[derived]`.

Over the 10-minute budget, `10 × 60 × 30 = 18,000` environment steps, which matches the paper's "~18k":

```
18,000 / 25 = 720 calls   →   14,400 critic steps · 720 VLA steps · 720 editor steps
```

### 2.13 Reward: the success detector

```
held(t)    = gripper_pos(t) > g_empty + margin      a jaw closed on a cube stops wider than one closed on air
lifted(t)  = z_tool(t) > z_plate + h                z_tool from forward kinematics
success    = held ∧ lifted, for 6 consecutive steps (200 ms)     [code: success_consecutive_steps = 6]
reward     = 1 on that step, and the episode ends.  Timeout at 200 steps → 0
```

`g_empty`, `margin` and `h` come from `[M5, M6]`. The paper's 0.30 m is measured in the DROID base frame; ours comes from [`simulation/so101_forward_kinematics.py`](../simulation/so101_forward_kinematics.py). The paper's text says 5 steps and its code says 6; we follow the code.

**Reset is mostly free.** After a success the arm carries the cube to a random radius and angle on the plate and releases it (IK), as the reference does with its randomised drop `[code: success_reset_randomize_magnitude]`. After a timeout the cube is still on the plate. A human is needed only when it is knocked off, and that count is logged.

### 2.14 Statistics

Every arm gets **30 trials**, the paper's protocol. Arms are compared with **Fisher's exact test**, which stays exact at small `n` and near 0% or 100%, exactly where the usual normal approximation breaks.

Worked, on the paper's own Dynamic Picking row: 30/30 against 24/30 gives `p = 0.024` `[derived]`.

What 30 trials can detect at 80% power, two-sided α = 0.05, with `p̄` the average success rate of the two arms:

```
Δ_min ≈ (1.96 + 0.84) · √( 2 · p̄(1 − p̄) / 30 )

p̄ = 0.50  →  2.80 × 0.129  =  36 points
p̄ = 0.85  →  2.80 × 0.092  =  26 points       (and less at the ceiling; the exact test handles that)
```

**Pre-registered, written before any data exists.** The primary endpoint is H1 at the trained speed and delay, with at most two looks at the data:

```
look 1, n = 30 per arm:   p < 0.0294 → supported     p > 0.30 → not supported     otherwise → look 2
look 2, n = 60 per arm:   p < 0.0294 → supported     otherwise → not supported
```

Why 0.0294 and not 0.05: every extra look is another chance for noise to cross the line. 0.0294 at each of two looks keeps the overall false-positive rate at 0.05 (Pocock's boundary for two looks). The early stop at `p > 0.30` only makes the test more conservative. Checked by simulation: with no real difference between arms, this rule with Fisher's test declares one **2.3–2.6%** of the time (20k runs each at true rates 0.5 and 0.8), under the 5% limit `[derived]`. Fixed now, so it cannot be tuned after looking.

### 2.15 Is the critic right? Two separate questions

| question | measurement | why |
|---|---|---|
| Does Q know which **states** are good? | AUROC of `Q(s, a*)` at each decision against the episode's outcome, on held-out episodes. 0.5 means no information | calibration |
| Does Q know which **actions** are good? | probe P1: the same checkpoint, argmax vs a uniformly random pick among the same 64 | the causal test, and the question Larchenko's action-ranking head failed (*"effectively zero"* correlation) |

AUROC can be high while P1 is null. A critic can learn "block near the gripper is good" without ever ranking actions.

---

## 3. Fidelity: what we copy, what we change

| | paper | us | why · risk |
|---|---|---|---|
| Arm | DROID Franka | SO-101: 5 joints + jaw | what we have · backlash, P-gain 16 |
| Actions | end-effector Cartesian + gripper **velocity**, 7-d | **absolute joint positions**, 6-d | LeRobot runs position mode · ±0.1 means something different, see M4 |
| Edited dims | position + gripper | all 6 joints | no position/rotation split in joint space |
| Base | π₀.₅ LoRA (Gemma 2B + 300M expert) | **SmolVLA**, ~450M | their learner uses 4 GPUs, 1 sampling and 3 updating `[code: scripts/dynamic_pick/run_server_async.sh]`; we have one 16 GB RTX 4080 · different prior |
| Views | 2 × 224²: side + wrist, plus end-effector pose | 2 × 224²: one exterior + wrist, plus joints | match the count. Pick top or front in Phase 1, whichever sees the plate unoccluded |
| H / C / d | 16 / 8 / 3 in the paper, 5 in the released script | 16 / 8 / from M1 | — |
| Demos | 25 successes `[code: scripts/dynamic_pick/collect_data.sh]` | 25 | the paper's text never states the count |
| SFT | 4000 steps, batch 64, `d_max = 10`; RL starts from the **step-2000** checkpoint `[code: offline_train.sh, run_server_async.sh]` | same | — |
| SFT gate | "around 30% or higher" `[paper IV-C]` | ≥ 9/30 | — |
| Online budget | 10 min ≈ 18k steps | same | — |
| VLA optimiser | AdamW 2.5e-5, clip 1.0, on LoRA π₀.₅ `[Tab III]` | same, on full SmolVLA | untested at this scale · watch for collapse |
| Plate | "a block placed on a rotating plate" `[paper V-C]` | **stepper-driven plate** (variable speed). A microwave motor, commonly 5–6 RPM, works for the core arms only | the released docstring disagrees, §10 #3 |

---

## 4. Hypotheses

| | claim | the paper predicts | falsified if |
|---|---|---|---|
| **H1** primary | Real-Time EXPO-FT beats EXPO-FT w/ RTC | 30 vs 24 of 30 | the §2.14 rule says not supported |
| H2 | RL with a *stale* editor adds little on this task | 24 vs 22 (p = 0.76, from the paper's counts) | EXPO-FT w/ RTC significantly beats SFT w/ RTC |
| H3 | the fresh-frame gain grows with `d` | the §2.2 table: 46% → 70% of staleness removed | the gain is flat or shrinks |
| H4 | the critic ranks actions | argmax beats random (P1) | P1 shows no difference |
| H5 | the edit's gain is local to the trained speed | shrinks away from `ω_train`, stays ≥ 0 | goes **negative** off-distribution |

---

## 5. Arms and probes

**Four arms, two of which train online for 10 minutes:**

| arm | VLA | editor + critic read | online RL? |
|---|---|---|---|
| **A0** SFT | the A1 checkpoint, run **synchronously** with `d = 0`: the arm holds still while it thinks | — | no |
| **A1** SFT w/ RTC | training-time RTC, asynchronous | — | no |
| **A2** EXPO-FT w/ RTC | training-time RTC | `s_t`, the stale frame the VLA saw | yes |
| **A3** Real-Time EXPO-FT | training-time RTC | `s_{t+d}`, the fresh frame | yes |

A2 and A3 differ in exactly one thing: which frame the editor and critic read. `[my reading: the paper never says which frame EXPO-FT w/ RTC's editor sees]` A0 reuses A1's weights, which is valid because `d = 0` is inside A1's training range; that isolates the execution scheme `[my choice]`.

**Three probes on A3's final weights.** 30 trials each, no training:

| probe | change | question |
|---|---|---|
| **P1** | pick uniformly at random among the same 64 | does the ranking matter? |
| **P2** | no editor: argmax over the 32 VLA chunks only | does editing add anything beyond selecting? |
| **P3** | VLA alone: 1 chunk, no editor, no selection | how much of the gain moved into the VLA's own weights? |

Together they split A3's score into VLA, selection and edit. The paper does not report this split.

---

## 6. Plan, step by step

Each phase has a gate. Do not start a phase until the previous gate is met.

### Phase 0 — measure; no learning, no training code
- [ ] **M1** SmolVLA inference latency at N = 1 and N = 32 (prefix cached), on Mac MPS and on the 4080. 5 warm-up calls, then 50 timed, synchronising the device before reading the clock → sets `d`
- [ ] **M2** Mac ↔ 4080 round trip: ping, then echo a 30 KB payload → decides the topology below
- [ ] **M3** critic encoder on one frame + heads on 64 candidates, timed like M1 → must be well under 33 ms
- [ ] **M4** what ±0.1 means: per-joint normalisation stats → degrees per joint → mm at the gripper via FK → is the edit budget meaningful?
- [ ] **M5** gripper `Present_Position`, 20 reads each: open · closed on air · closed on the cube → `g_empty`, `margin`
- [ ] **M6** hand-guided: tool height on the plate and lifted, and the grasp tolerance → `h`, `T`
- [ ] **M7** one SmolVLA step with the prefix loss at batch 64, plus one critic step, on the 4080 → `torch.cuda.max_memory_allocated`: fits in 16 GB?
- [ ] **M8** read `lerobot/rl/` (actor, learner, buffer, `gym_manipulator`) and decide what §7 can reuse
- **Gate:** `d ≤ 8` · fast step < 33 ms · training fits in memory

### Phase 1 — plate, detector, auto-reset
- [ ] stepper plate with a speed control; choose the exterior camera
- [ ] detector (§2.13) → **20/20** on staged grasps and **20/20** on near-misses (empty lift, pinch-and-drop)
- [ ] auto-reset → 18/20 clean drops

### Phase 2 — demos
- [ ] 25 successful teleop episodes at `ω_train`, first estimate from §2.2 · **gate:** a human can do it at that speed

### Phase 3 — SFT (A1; A0 reuses it)
- [ ] training-time RTC loss (§2.4), with a unit test for the mask and the `t = 0` prefix
- [ ] 4000 steps, checkpoints at 2000 and 4000 · **gate:** A1 ≥ 9/30, else slow the plate and redo Phase 2

### Phase 4 — baselines
- [ ] A0: 30 trials · [ ] A1: 30 trials

### Phase 5 — the primary result
- [ ] A2: train 10 min, 30 trials · [ ] A3: train 10 min, 30 trials · apply the §2.14 rule. **Reported whatever it is**

### Phase 6 — why it worked, or didn't
- [ ] P1 · [ ] P2 · [ ] P3 · [ ] AUROC from the logs

### Phase 7 — off-distribution
- [ ] A1 and A3 at 0.5×, 1.5×, 2× `ω_train`, 30 trials each

### Phase 8 — delay sweep
- [ ] A2 and A3 from `d_real` up to `d = 8`, e.g. {3, 5, 8}, by injected sleep (the paper's own method). `d` stops at 8 = C; there `d + C = 16 = H`, so the chunk is used to its last action

### Phase 9 — ablations, only if Phase 5 worked
- [ ] **A. Reward source.** Train A3 with the reward from a learned image classifier (`lerobot/rewards/classifier`) and from a VLM judge, instead of §2.13's detector. Evaluation stays human-verified. This is the paper's limitation 2, and the one that matters for laundry, where "folded" cannot be read off joint angles
- [ ] **B. Dense reward.** Larchenko's scheme on this task: 0.5 when `held` first holds for 6 steps, 0.5 more at success. On failure, withdraw what was paid, spread evenly from the peak to the end, so every return stays 0 or 1. It fits EXPO because updates run at episode end, when the outcome is known. This is the paper's limitation 3. LeRobot's per-frame progress models (`robometer`, `sarm`, `topreward`) are ready-made alternatives

### Where the models run — decided by M1–M3

| topology | inference | learner | cost |
|---|---|---|---|
| a. arm on the Mac, all models on the 4080 | 4080 | 4080 | a network round trip inside every decision |
| b. arm on the Mac, inference on the Mac, learner on the 4080 | MPS | 4080 | weights synced every update call; MPS speed |
| c. arm plugged into the 4080 workstation | 4080 | 4080 | no network. It isn't ours; ask its owner first |

The paper runs (a) on a LAN: GPU 0 samples actions, GPUs 1–3 update, and the robot talks to it over WebSocket `[code: run_server_async.sh]`. LeRobot's async inference ([deployment](../docs/deployment/README.md)) is the same client/server split.

---

## 7. What must be built

Keep `src/phi/` thin: call LeRobot where it already does the job (CONTRIBUTING). New code goes in `src/phi/rl/`.

| piece | build on (LeRobot 0.6.0; fit confirmed in M8) | to write |
|---|---|---|
| Flow VLA, inference-time RTC, `so_follower`, recording | `policies/smolvla`, `policies/rtc` | — |
| Robot client ↔ GPU server | async inference | the fast edit-and-select step inside the server |
| Training-time RTC loss for SmolVLA | — | §2.4: time per token, prefix mask, and the unit test |
| Asynchronous executor with delay `d` | the inference-RTC action queue | start inference `d` steps early · commit the prefix · fresh-frame edit and select at the boundary |
| Actor ↔ learner transport, replay storage | `rl/learner_service.py` (gRPC), `rl/buffer.py` | a `C`-step window sampler, since the buffer stores single steps |
| Critic, editor, α, noise filter | reference in JAX, [pd-perry/expo-ft](https://github.com/pd-perry/expo-ft) — read for meaning | PyTorch, §2.6–2.10 |
| Learner loop | `rl/learner.py`, which is SAC-specific | demos seeded in the buffer · K = 25 · 20 critic steps per call · start after 10 episodes · VLA on successes only |
| Real-arm env, detector, auto-reset | `rl/gym_manipulator.py` · [`simulation/so101_forward_kinematics.py`](../simulation/so101_forward_kinematics.py) | §2.13, and commit the IK |
| Logs | — | every decision: all 64 Q values, chosen index, VLA or edited, latency, detector state. P1–P3 and AUROC come from these |

---

## 8. Budget

```
episodes per online arm  ≥ 18,000 steps / 200 steps per episode = 90     (successes end early, so more)
wall time per episode    ≈ rollout (≤ 6.7 s) + queued update calls + reset
```

The update time is unknown until M7. If one call takes ~2 s, a 200-step episode queues 8 calls, about 16 s, and one online arm takes roughly an hour `[estimate]`.

Core, A0–A3 plus P1–P3: about 3 hours of arm time. Off-distribution: +30 min. Delay sweep: +4 h `[estimate]`.

---

## 9. What this gives the laundry goal

The laundry goal follows [Learning to Fold](https://arxiv.org/abs/2606.27163) (Larchenko; RECAP + AWR, 1st of 62 at LeHome 2026). The two methods differ at the root: **EXPO-FT changes the action** (edit, then select) at every decision, while **RECAP + AWR changes the data** (which frames the model imitates, and labelled how).

| carries over | why |
|---|---|
| real-robot reward → critic → learner → robot client | the piece Larchenko names as missing |
| P1: does a separate TD critic rank actions? | his action-ranking head did not. The answer decides whether laundry keeps best-of-N |
| training-time RTC for SmolVLA and π₀.₅ | same chunk-boundary problem; the principled version of his soft inpainting |
| ablation B: his dense reward, inside EXPO | the controlled comparison his report never ran |

| does not carry over | why |
|---|---|
| the fresh-frame edit | cloth is nearly static between frames `[reasoned, not measured]` |
| the joint-angle detector | "folded" is not in the joint angles. Laundry needs a vision detector, which is ablation A |
| chunk-level TD as it stands | a 30 s fold would be ~112 chunks, and `0.9227¹¹² ≈ 1e-4` at the start. Laundry needs dense reward |

---

## 10. Paper vs code, and what is not yet verified

Resolve every row before Phase 5. Where they disagree, the launch scripts are what produced the released runs.

| # | paper | released code | we |
|---|---|---|---|
| 1 | Dynamic Picking `d = 3` | `--delay=5` in `scripts/dynamic_pick/run_server_async.sh` | measure ours; log both |
| 2 | success held 5 steps | `success_consecutive_steps = 6` | 6 |
| 3 | "a block placed on a rotating plate" | docstring: *"what makes it dynamic is the reset"* (a randomised drop pose) | turntable. A plate is hardware and would not appear in a config, so both may be true. Ask the authors |
| 4 | critic "ResNet-50" (main text) | stages (3,4,6,3) of basic blocks, which is the ResNet-34 layout | follow the code |
| 5 | backup edit and Q read `s_{t+C}` (Eq 10) | `q_edit_use_main_obs = False`: *"backup edit/Q-selection on the delayed (main-actor) obs"* | read the learner before coding |
| 6 | terminal windows for Kick and Balance only | `valids_keep_terminal_windows=True` for dynamic_pick too | follow the code |
| 7 | is the edit added in normalised or raw action space? Is `log π` taken before or after the ×0.1? | not stated in the configs read | read the learner |
| 8 | object-speed sweep: trained per speed, or evaluated only? | the figure was not extractable | not cited here as evidence about generalisation |
| 9 | which frame EXPO-FT w/ RTC's editor reads | not stated | `s_t`, my reading |
| 10 | number of demos | not in the text | 25, from the scripts |

---

## What counts as a result

Any of these is a club result, including the negative ones:
- **A2 vs A3 on our arm**, with the §2.14 rule applied, whatever it says
- **P1**: whether a separate TD critic ranks actions. A null here is as useful to the laundry goal as a positive
- **A working real-robot reward + critic + learner loop** that another task can reuse

⚠️ **Training reward is not evaluation.** Every evaluation success is confirmed by a person watching, as in the paper. The detector trains the policy; it does not grade it.

## Results

_Filled in as each phase lands. One row per arm or probe._

| phase | arm / probe | successes / 30 | vs | p | date · commit |
|---|---|---|---|---|---|
| | | | | | |

## References

**Papers**: [Real-Time EXPO-FT](https://arxiv.org/abs/2609.18207) (Dong, Hung, Sadigh, Finn) · [EXPO-FT](https://arxiv.org/abs/2605.25477) · [EXPO](https://arxiv.org/abs/2507.07986) · [Training-time RTC](https://arxiv.org/abs/2512.05964) (Black, Ren, Equi, Levine) · [FASTER](https://arxiv.org/abs/2604.19730) · [SAC](https://arxiv.org/abs/1801.01290) · [TD3](https://arxiv.org/abs/1802.09477) · [REDQ](https://arxiv.org/abs/2101.05982) · [Learning to Fold](https://arxiv.org/abs/2606.27163)

**Code**: [pd-perry/expo-ft](https://github.com/pd-perry/expo-ft) (`configs/` and `scripts/dynamic_pick/` read directly, 2026-09-25) · LeRobot 0.6.0 source in the `phi` env

**Repo**: [theory/smolvla](../docs/theory/smolvla.md) · [theory/control-loop](../docs/theory/control-loop.md) · [deployment](../docs/deployment/README.md) · [evaluation](../docs/evaluation/README.md) · [RL in sim](2026-08-02_rl-in-sim-so101.md) · [π₀.₅ cubes/cylinder](2026-09-15_pi05-cubcyl-lang.md)

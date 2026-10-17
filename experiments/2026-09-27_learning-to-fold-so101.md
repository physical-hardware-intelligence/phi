# Learning to Fold on our SO-101s — reproduce Ilia Larchenko's LeHome policy, then beat it

**Status**: 🟡 **Phase 0 running.** **Gate B green 2026-09-28**: server runs, wiring verified (B6 run 2 PASS), M1 679 ms. Tracks A and C open. · **Owner**: Yash · **Created**: 2026-09-27 · **Ladder**: L4
**Tracks**: A Knowledge — *open, no hardware needed* · B Server — Yash · C Rig — Sai, Parv

> The club's laundry goal. We start from the strongest public bimanual folding policy for **our exact arm**, prove we can run it, measure it on our rig, then improve it. The sister experiment, [Real-Time EXPO-FT](2026-09-26_realtime-expo-ft-so101.md), builds the real-robot reward + critic loop that Phase G here will need.

**How to use this doc.** Work top to bottom. Every phase ends in a **gate**: a check that is allowed to fail. Don't start a phase until its inputs are green. Tick boxes in the PR that completes them; every number goes in [Results](#results).

**Every number carries one tag.**
`[paper §x]` [Learning to Fold](https://arxiv.org/abs/2606.27163), Larchenko 2026 · `[code: file:line]` [IliaLarchenko/lehome_solution](https://github.com/IliaLarchenko/lehome_solution) @ `ca73828`, read 2026-09-27 · `[HF]` read from the Hugging Face Hub API, 2026-09-27 · `[mac]` run in our Mac `phi` env, 2026-09-27 · `[box]` verified on the lab GPU box, 2026-08-22 · `[M#]` to measure here · `[derived]` computed on this page · `[estimate]` replace with a measurement · `[my reading]` interpretation, stated nowhere

---

## The question

**Does the 2nd-place LeHome real-round policy fold clothes on a rig it has never seen — and can we raise its success rate with our own data?**

## Why this

1. **It is our arm.** Bimanual SO-101, 12 joints, three cameras. No other public folding policy is trained for this hardware.
2. **The checkpoints are public.** Both rounds, 10.5 GB each `[HF]`. We start from a policy that took one engineer three months on an H200 `[paper §9]`.
3. **He names what is missing.** His real round was plain imitation *"because there was no real-side reward or value"*, and he guesses 90%+ is reachable if that gap is closed `[paper §10.5]`. That is the research, once we match him.

---

## 0. The whole system in two pictures

**How one decision happens on the real robot** (20 Hz, one request per 250 ms of motion):

```
 ROBOT CLIENT (laptop + arms)                         POLICY SERVER (GPU)
 ──────────────────────────                           ──────────────────
 read 12 joint angles (degrees)
 read 3 cameras (raw frames)      ── ~6.1 MB ──►      resize each view to 224² (with padding)
   right_front 1280×720                                SigLIP  → image tokens            (frozen)
   left_wrist   640×480                                Gemma-2B → reads images + joints + garment type
   right_wrist  640×480                                Gemma-300M × 10 flow steps → 30 future moves
                                  ◄── 5 targets ──     keep the first 5, as absolute joint angles
 execute 5 × 50 ms = 250 ms
 arm is FROZEN while waiting for the next reply  (the loop is synchronous)
```

`[code: scripts/record_real_dagger.py:352, :703; README serve flags; paper §4]`

**How the checkpoint was made, and where we start:**

```
π₀.₅ base ─► 20k steps BC ─► RL flywheel in Isaac Sim ─────────► lehome_sim   1st of 62 · 79.63%
                             ~300k steps · 12.5k rollouts          │
                                                                    │ late-but-not-latest checkpoint
                                                                    ▼
                             strip the sim-only parts ─► fine-tune on real data ─► lehome_real   2nd · 865/1080
                                                                                        │
                                                                          WE START HERE ┘
```

`[paper §2.6, §8, §9.4–9.6, Tab 7]`

**Why `lehome_real`, not `lehome_sim`:** the sim policy failed zero-shot on real hardware even for him `[paper §9.1]`. The real one has already seen 500 organizer + 792 of his own real episodes `[paper Tab 6]`.

---

## 1. How it works — every idea, with its intuition and its number

### 1.1 The model

**In one line:** a vision-language model with the language removed and a small action generator attached.

```
3 camera views ─► 224² each ─► SigLIP-So400m (frozen) ─┐
12 joints ─► 256 bins each ────────────────────────────┼─► Gemma-2B ─► cached keys/values
garment type (1 of 4, inferred by the model itself) ───┘                   │
noise (30 × 12) ─► Gemma-300M action expert, 10 Euler steps ──────────────┘─► 30 × 12 joint deltas
```

| fact | value |
|---|---|
| lineage | π₀.₅ → his BEHAVIOR-1K winner → this `[paper §4]` |
| size | π₀.₅ class, ~3.5B parameters (`lerobot/pi05_base` is 3.5B, [pi05 experiment](2026-09-15_pi05-cubcyl-lang.md)) |
| output | 30 actions × 12 joints, predicted as **deltas** from the current state `[code: training/config.py:810]` |
| horizon | 30 steps = **1.5 s at 20 Hz** (real), 1.0 s at 30 Hz (sim) `[derived]` |
| executed | first **5** of 30, then re-plan `[README]` |
| language | **none**. Text input dropped entirely `[paper §4]` |
| generator | flow matching: start from noise, walk toward a valid chunk in 10 steps `[paper §7.1]`. Background: [diffusion-policy-why](../docs/theory/diffusion-policy-why.md), [smolvla §15](../docs/theory/smolvla.md) |

**Why generate instead of regress:** there are many valid ways to fold a sleeve. Averaging two valid grasps gives an invalid one. A generator samples *one* of them.

### 1.2 Four problems → four decisions

The whole design falls out of four constraints.

| problem | decision | what it costs |
|---|---|---|
| PPO/GRPO need action log-probabilities; flow models don't give them cheaply, and *"pushing probability away"* from bad actions pushes predictions **off** the thin manifold of valid ones `[paper §2.2]` | only **reweight** and **condition** on good actions: AWR + RECAP | **no exploration.** It sharpens what it has seen and cannot invent a recovery `[paper §10.4]` |
| reward is binary, only at the end `[paper §1.2]` | densify with the challenge's own success checker; claw it all back on failure `[paper §6.1]` | reward design is *"still an art"* `[paper §10.1]` |
| a separate critic is a second model to train, sync and serve `[paper §5]` | put the value heads **inside** the policy | the heads overfit and need extra regularisation `[paper §5.3]` |
| rollouts go stale while the policy keeps changing `[paper §6.5]` | record predictions at collection time; decay old data 0.98 per iteration; blend toward an outcome-only baseline | *"over-engineered"*, by his own account `[paper §6.8]` |

### 1.3 Reward: pay for progress, repay on failure

**In one line:** reward measurable progress, but if the fold fails, take all of it back — so the total always equals "did it fold".

```
first fold done        +0.5     (tops: paid gradually as the main keypoint gap closes)
full success           +0.5  →  total 1.0
failure                everything paid is withdrawn, spread evenly from the peak to the end
⇒   Σ r_t = 1[success]   exactly
```

Worked: if folding the first sleeve closes 60% of the gap, it earns 0.3 `[paper §6.1]`. Success itself is geometric: keypoint pairs that must meet or stay apart, 5 conditions for tops, 4 for pants `[paper §1.1]`.

### 1.4 The policy is its own value function

**In one line:** the network that decides what to do also predicts how the episode will end.

| head | predicts | used for |
|---|---|---|
| success | P(this episode succeeds) | **the value function**, V `[paper §5.1]` |
| completion | fraction of the fold done | progress signal; barely moves as the policy changes, so it stays valid on old data |
| garment type | 1 of 4 | the inference-time type bootstrap (§1.6) |
| keypoint distances | the checker's own ratios | a cheap world-model substitute, **sim-only** |
| Δ success | outcome minus V, given *these* actions | Q-like; ranks candidate chunks `[paper §5.2]` |

**The trick is the attention mask.** All heads read one query token that sits in the image group, so it sees **pixels only** — no joint angles, no garment label `[paper §4.1]`. The value cannot cheat off proprioception, and the garment head cannot copy its own input.

### 1.5 Advantage, consumed twice

**Advantage** `A` = how much better this moment went than the value expected `[paper §6]`.

**AWR, through the sampler.** `P(load frame) ∝ e^{clip(A, −2, 2)}`. A frame at `A = −2` is loaded **1.8%** as often as one at `A = +2` `[derived: e⁻⁴]`. It isn't down-weighted after loading — it is simply almost never decoded, so compute goes to good actions `[paper §2.3]`.
⚠️ The heads must still learn the *real* data distribution, so every frame carries an importance weight `w = 1/(N·p·T_ep)` on head losses only `[paper §2.3]`.

**RECAP, as an input.** A learned advantage token tells the model "this was a good action". Negative-advantage frames are always shown as neutral; positive ones are masked with probability 0.5 → 0.1 as `A` goes 0 → 2 `[paper §4.3]`. At inference: `v̂ = v_uncond + α (v_cond − v_uncond)`, with `α = 7–9` `[paper §7.3, Tab 4]`.
**Intuition:** ask for "the good version", then ask louder.

🔑 **None of 1.4–1.5's RL machinery is in `lehome_real`.** It was stripped for the real round `[paper §9.5]`. The client still sends `cfg_scale: 5.0`, but the model only runs guidance when an advantage embedding exists, and the real checkpoint has none — so it is ignored, at no cost `[code: models/pi_modified.py:1641; training/config.py:800]`.

### 1.6 Inference knobs: accuracy for free

The same weights score differently depending on how they are run `[paper §7]`.

| knob | meaning | sim, tuned per garment `[Tab 4]` | real round `[README; code: record_real_dagger.py:305]` |
|---|---|---|---|
| executed `n_e` | moves sent before re-planning | 3–5 | 5 |
| playback | stretch those moves over more steps (slower) | = `n_e` | 5 (no stretch) |
| anchor `n_a` | moves kept to seed the next chunk | 3–6 | 1 |
| inpaint onset | how early the anchor stops pulling | 0.4–0.5 | 0.5 |
| guidance `α` | advantage amplification | 7–9 | off |
| temperature `τ` | noise scale; < 1 = nearer the mode | 0.7–0.9 | 0.9 |
| candidates `N` | chunks drawn, best one kept | 2–3 | 1 |

In sim he tuned these **online during data collection** with a Thompson-sampling bandit, one Beta posterior per value, for free `[paper §7.7]`. What converged is informative: short execution wins (near moves are predicted better than far ones), and more than 3 candidates never helped.

**Garment bootstrap.** The garment is not labelled at test time. The first call reads the model's own guess, the robot holds still for that one chunk, then the guess is latched `[code: record_real_dagger.py:386]`. Late in training the guess was >99% right `[paper §4.2]`.

### 1.7 Sim → real in one week `[paper §9]`

| step | what he did |
|---|---|
| start | a strong but **not the latest** sim checkpoint — the latest was most overfit to the renderer `[§9.4]` |
| strip | kept action + garment + completion heads; removed keypoint and world-model heads, advantage, guidance, best-of-N `[§9.5]` |
| mix | organizer 60% · his teleop + DAgger 30% · sim replays 10% of every batch `[Tab 6]` |
| align | camera-overlay tool, calibration, units `[§9.7]` |
| diversify | heavy per-camera augmentation, **state noise and dropout** so it trusts images over a miscalibrated arm `[§9.8]` |
| speed | resample each source's time axis: organizer ×1.0 · teleop ×1.5 · DAgger ×2.0 · sim ×0.65 `[§9.9]` |
| DAgger | human frames weighted highest; the **5 s before a takeover ramped to zero**, so the moves that caused the failure aren't reinforced `[§9.10]` |

**Why speed factors exist:** actions are deltas over a fixed horizon. A slow source teaches small deltas, a fast one large deltas. Mixed naively, the model learns an inconsistent notion of "one step".

### 1.8 Where he landed, and what he says is open

| round | result |
|---|---|
| online (Isaac Sim) | **1st of 62, 79.63%**, 6.1 points clear. Long top 74.5 · short top 70.0 · long pants 80.5 · shorts 93.5 `[Tab 5]` |
| real final (ICRA) | **2nd, 865 of 1080**, behind 895 `[Tab 7]` |

His own open problems: recovery without a human `[§10.4]`, and one pipeline with a real-side value `[§10.5]`.

---

## 2. What we reproduce, exactly

| | in scope | out of scope, and why |
|---|---|---|
| policy | `lehome_real`, run at its intended rate on a matched rig | the sim RL flywheel: needs Isaac Sim and months of rollouts |
| score | our success rate under a **pre-registered rubric** (§9) | his 865/1080: their rig, their garments, their jury. Not reproducible by construction |
| data | the organizer's 500 real episodes + our own | his 792 home episodes are not among the released assets: the README links only the organizer datasets and the two checkpoints `[code: README]` |

**Every garment we buy is unseen for `lehome_real`.** The organizer's physical garments aren't ours.

---

## 3. Verified facts, as of 2026-09-27

Nothing here is assumed. Each row says how it was checked.

### 3.1 The checkpoint `[HF]`

| | value | how |
|---|---|---|
| `IliaLarchenko/lehome_real` | 10.49 GB, Orbax OCDBT, public | `GET /api/models/IliaLarchenko/lehome_real` |
| ships with | `assets/real_bc/norm_stats.json`, FAST tokenizer | same |
| units | **degrees.** `wrist_roll` spans −138.4° to +159.0° (left) — impossible in a ±100 encoding. Gripper 0–100 | read `norm_stats.json` q01/q99 |
| stats also carry | per-timestep mean/std, action correlation Cholesky factors | same keys |

### 3.2 The organizer dataset `[HF]`

`lehome/dataset_challenge_real`, subset `four_types_merged`: LeRobot v3.0, `bi_so_follower`, **20 fps**, **500 episodes, 187,135 frames**, 4.48 GB — matching his Table 6 exactly.

| key | shape | note |
|---|---|---|
| `observation.state`, `action` | 12 | `left_shoulder_pan.pos … left_gripper.pos, right_shoulder_pan.pos … right_gripper.pos` |
| `observation.images.right_front` | 720 × 1280, AV1 | the overhead/front view. **16:9** |
| `observation.images.left_wrist` | 480 × 640, AV1 | 4:3 |
| `observation.images.right_wrist` | 480 × 640, AV1 | 4:3 |

| garment | episodes | mean fold, s `[derived]` |
|---|---|---|
| long-sleeve top | 125 | 23.9 |
| short-sleeve top | 125 | 20.4 |
| long pants | 125 | 16.9 |
| shorts | 125 | 13.6 |

### 3.3 The robot ↔ server contract `[code]`

| | fact | where |
|---|---|---|
| transport | WebSocket, port 8000, stateless `infer_chunk` | `scripts/serve.py:65` |
| real rate | `--fps 20` | `record_real_dagger.py:837` |
| units | `use_degrees=True` on all four arms | `record_real_dagger.py:886` |
| key mapping on the wire | `right_front → top_rgb` · `left_wrist → left_rgb` · `right_wrist → right_rgb` | `record_real_dagger.py:358` |
| images sent | **raw uint8, base64, no resize, no compression** → 6.14 MB per request `[derived]` | `record_real_dagger.py:352` |
| resize | server-side `resize_with_pad` to 224² — keeps aspect ratio, pads the rest | `models/observation.py:434` |
| loop | **synchronous**: the arm holds still while the server thinks | `record_real_dagger.py:703` |
| memory | ~~JAX takes 80% of GPU memory (12.8 of 16 GB)~~ **Corrected 2026-09-28:** on demand. `serve.py` also sets `XLA_PYTHON_CLIENT_ALLOCATOR=platform`, which turns preallocation off. **Measured: 5.7 GB loaded, ~6.0 GB peak while serving** | `scripts/serve.py:37–38` · `[box]` |

### 3.4 Our stack

| | value | how |
|---|---|---|
| Mac lerobot | **0.6.0**; `SOFollowerConfig.use_degrees` defaults to **True** | `[mac]` `config_so_follower.py:42` |
| bimanual classes | `bi_so_follower`, `bi_so_leader` exist | `[mac]` |
| arm ids | sub-arms are `{id}_left`, `{id}_right`; bimanual-level cameras keep **unprefixed** keys | `[mac]` `bi_so_follower.py:56, :43` |
| runner imports vs 0.6.0 | of the **13 lerobot symbols** it needs, **11 resolve**; `build_dataset_frame`, `combine_feature_dicts` moved to `lerobot.utils.feature_utils` | `[mac]` import test |
| keyboard | runner uses `evdev` → **Linux only**. `pynput` is available on the Mac | `[mac]` |
| missing | `websockets` | `[mac]` |
| robot-side JAX? | **none.** The runner's import chain has no jax/openpi | `[code]` grep |
| GPU box | RTX 4080 SUPER 16 GB, driver 580.173.02, Ubuntu 22.04, **no sudo**, 1.1 TB free, Tailscale relay ~29 ms | `[box]` |
| training GPU | Explorer H200 (141 GB), `gpu` partition, 8 h wall | [pi05 experiment](2026-09-15_pi05-cubcyl-lang.md) |
| server env | Python **3.11 exactly**, `jax[cuda12]==0.5.3`, `torch==2.7.1` (cu126) | `[code: pyproject.toml]` |
| pinned deps | openpi `c23745b`, lehome-challenge `5ea947e` | `[code]` `git ls-tree HEAD` |
| lockfile | **none**: `uv.lock` is gitignored, so `uv sync` resolves fresh. Only the pins above are fixed | `[code: .gitignore:5]` |

---

## 4. Fidelity: his rig vs ours

| | Ilia, real round | us | gap → fix |
|---|---|---|---|
| arms | 2 followers **46 cm apart** + 2 leaders `[README]` | 2 pairs ✅ | build the mount |
| robot OS | Linux | macOS | port the runner (§8) |
| robot lerobot | 0.4.3 (fork pin) `[code]` | 0.6.0 | 2 import moves |
| top camera | RealSense D435, RGB only, 1280×720 | webcam | **set 1280×720, 16:9.** Depth was never used `[paper §1.1]`, so no loss |
| wrist cameras | 2 × 640×480 | 1 wrist + 1 front | remount `front` as the second wrist |
| rate | 20 Hz | our datasets are 30 fps | record and run at 20 |
| units | degrees | degrees by default ✅ | check calibration (C6) |
| inference | local RTX PRO 6000 | 4080 across a relay | the topology gate (§6) |
| garments | organizer's **kids' clothes** `[paper §9.1]` | none yet | buy, matched to organizer frames |
| operator | 3-pedal foot switch `[paper §9.10]` | keyboard | pedal optional |
| training | 1 × H200, batch 192 | Explorer H200 | chain 8 h jobs with `--resume` |

---

## 5. Traps — symptom → cause → fix

| # | symptom | cause | fix | source |
|---|---|---|---|---|
| 1 | every joint ~10% off, looks fine on screen | units: normalised ±100 vs degrees | `use_degrees=True` everywhere; `wrist_roll` beyond ±100 proves degrees | `[paper §9.11]` |
| 2 | `wrist_roll` range halved | follower and leader in different units | same units on all four arms | `[code: record_real_dagger.py:882]`, lerobot #3176 |
| 3 | policy erratic, **no error raised** | camera key wrong | keys exactly `right_front / left_wrist / right_wrist`; replay-gate controls catch it | `[code]`; our [pi05 run](2026-09-15_pi05-cubcyl-lang.md) hit the silent version |
| 4 | top view geometry off | a 4:3 top camera pads differently from their 16:9 | capture the top at **1280×720** | `[code: observation.py:434]` |
| 5 | arm frozen most of the time | 6.1 MB of raw frames per request across a relay | §6 | `[derived]` |
| 6 | `No keyboard found under /dev/input/event*` | `evdev` is Linux-only | replace with `pynput` | `[code: :212]`, `[mac]` |
| 7 | `ImportError: build_dataset_frame` | moved in lerobot 0.6.0 | import from `lerobot.utils.feature_utils` | `[mac]` |
| 8 | setup takes an hour and fills disk | `setup.sh` has **no flag to skip** the Isaac Sim step | run steps 2–4 by hand (B1) | `[code: setup.sh:41–48, :253]` |
| 9 | env differs from his | `setup.sh` runs `git pull origin main` on openpi | stay on the pinned `c23745b` | `[code: setup.sh:179]` |
| 10 | `cannot find -lcuda` | the symlink fix needs sudo | userspace symlink + `LIBRARY_PATH` (B1) | `[code: setup.sh:130]` |
| 11 | first chunk of an episode does nothing | garment bootstrap holds still for one chunk | by design | `[code: :386]` |
| 12 | a garment gets the wrong fold | ambiguous garment (long pants that look like shorts) | log it; not a policy bug | `[paper §10.1]` |
| 13 | fine-tune LR never reaches peak | `warmup_steps: 100000` > `num_train_steps: 50000` if the step counter restarts at 0 | read the weight loader before F4 | `[code: configs/train_real_bc.yaml]` `[my reading]` |
| 14 | random camera crashes on the Mac | `av` and `cv2` both bundle `libavdevice`; macOS warns *"may cause spurious casting failures"* | known, pre-existing in `phi` | `[mac]` |
| 15 | `import torchcodec` fails on the box | it needs system FFmpeg and CUDA 12 NPP libraries, which `setup.sh` installs with apt (sudo). **Not an issue on our box** (2026-09-28): CUDA 12.4 NPP and FFmpeg 4.4.2 are system-installed, and torchcodec imports | the server never decodes video; only B6 does. Fallback: PyAV (`video_backend="pyav"`), whose build here has AV1 (`libdav1d`) | `[code: setup.sh:71, :227]` · `[box]` |
| 16 | a `uv pip` command changes the wrong environment, no error | an active conda env, even `base`, can outrank `.venv` for `uv pip` | build with conda deactivated; use only `uv sync` / `uv run` | `[my reading]` of uv's environment discovery |
| 17 | `hf download` fails in seconds: `LocalEntryNotFoundError`, caused by `ConnectionResetError: [Errno 104]` | the network filters `huggingface.co` by name: TCP connects, then the TLS hello is reset | download on another machine at a pinned `--revision`, copy over SSH, check file count and bytes against the Hub API | `[box]` 2026-09-28 |
| 18 | on the Mac, `hf download` receives data but almost none reaches disk (2.66 GB in, 0.10 GB written); the Xet log repeats *"connection struggling"* with success ratio 1.000 | not isolated. Seen writing to an exFAT drive, which has no sparse files `[my reading]` | `HF_HUB_DISABLE_XET=1`: plain HTTP wrote 18 MB/s to the same drive | `[mac]` 2026-09-28 |
| 19 | a dataset download exits 0 but a subset is missing | `hf` ≥ 1.0 takes one pattern per `--include` and reads the rest as file names, then warns *"Ignoring `--include` since filenames have been explicitly set"*. `hf` 0.35 wanted space-separated patterns instead, so no single spelling works on both | one `--include` per call (B3), then verify the counts | `[mac]` 2026-09-28: 3 of 4 subsets arrived · `huggingface_hub/cli/download.py`: 0.35.3 `nargs="*"`, 1.25.1 a typer `list[str]` option |
| 20 | policy moves oddly, **no error**; one arm or joint is ignored | the runner's `state_dict_to_vec12` uses `obs.get(name, 0.0)`: a joint name our robot doesn't emit is sent as **0.0**, silently | in the Mac port (D1), assert all 12 `MOTOR_NAMES` are present before the first request | `[code: real_robot_config.py:52]` |
| 21 | every request takes ~0.4 s longer than the model needs | `websockets` compresses each 6 MB message by default (`compression="deflate"`) | `compression=None` in the client's `connect()`; the server falls back to plain frames | `[box]` 2026-10-03: 667 → 279 ms |

---

## 6. Where the model runs

**The problem, in one line:** the client ships 6.1 MB of raw frames for every 250 ms of motion, and the arm stands still until the reply comes back.

```
t_req     = upload + inference + reply + round trip
arm duty  = 250 ms / (250 ms + t_req)
```

Upload alone, before any inference `[derived]`:

| link | upload of 6.14 MB | arm moving |
|---|---|---|
| 10 Mbps | 4.9 s | 5% |
| 25 Mbps | 2.0 s | 11% |
| 50 Mbps | 0.98 s | 20% |
| 100 Mbps | 0.49 s | 34% |
| 500 Mbps | 0.10 s | 72% |

Ilia's loop is synchronous too, so *his* arm also paused — but only for local inference. Ours adds the network.
`[my reading]` Cloth barely moves while the arm is still, so a long freeze mostly costs time rather than success. Untested.

**Topologies:**

| | robot side | server | cost |
|---|---|---|---|
| **a** | Mac | GPU box, across Tailscale | 6.1 MB per request |
| **a′** | Mac, resizing to 224² *before* sending | GPU box | 602 KB (9.8%), but the model input must be proven identical |
| **b** | the GPU box itself (Linux: `evdev`, v4l2 work natively) | same box | no network. The box isn't ours; ask its owner |

**Decision rule** (after M2, M3):
1. `t_req ≤ 250 ms` with raw frames → **a**.
2. Otherwise → **a′**, only if B6 re-run with client-side resize gives **identical** predictions at the same seed. The paper's own warning: a resize path invisible to the eye changed sim success, and the model could tell the two apart perfectly `[paper §9.1]`.
3. Still slow → **b**.

The 250 ms threshold is `[estimate]`. Record `t_req` on every episode either way, so results stay interpretable.

**Measured, 2026-09-28 `[mac]`: the protocol itself costs ~130–146 ms per request before any inference.** That's with a stand-in server that does no inference, over localhost, from JSON + base64 of 6.14 MB. It's in the message format, so a, a′ and b all pay it; a′'s 224² frames would cut it ~10×. The box's CPU may be faster; B5 on the box gives the number that counts.

**Measured on the GPU box, 2026-09-28 `[box]`: worse. The same no-inference stand-in takes 516 ms median (p95 550).** The real server took 695–763 ms on a GPU it shared with another job. So on that box **~0.5 s of each ~0.7 s request is the message format and ~0.2 s is the model** `[derived]`. JSON decode + base64 account for only ~70 ms; the other ~450 ms of the receive path is unexplained so far `[UNVERIFIED]`. **The format, not the network, is the first cost to attack.** The levers are a′ (client-side 224²) or binary frames, each after an "identical predictions" check.
Re-measured on a quiet box with an idle GPU: **stand-in 488 ms, real server 679 ms.** The transport cost is the stack itself, not CPU load or GPU sharing.

**Cause found 2026-10-03: websocket compression.** `websockets` deflates every message by default. Our 6.14 MB barely compresses (to 4.65 MB) yet costs ~0.35–0.45 s per request. **Fix: `compression=None` on the client only.** The server falls back, its code is untouched, and the model gets identical bytes. Measured `[box]`: stand-in 466 → **112 ms**, real server 667 → **279 ms** (target 250). Next lever: 224² frames from the client.

---

## 7. The plan

Three tracks run **in parallel**; they share nothing until Phase D.

```
Track A  knowledge ───────────── Gate A ─┐
Track B  server ─── Gate B ──────────────┼─► D integrate ─► E zero-shot ─► F fine-tune ─► G research
Track C  rig ───────────── Gate C ───────┘
```

| | target | why this date |
|---|---|---|
| Gate A | Sat Oct 3 | ~19 h of reading, two people, one week `[estimate]` |
| Gate B | env Thu Oct 1 · replay Sat Oct 3 | env is the highest-variance item; fail fast |
| Gate C | Sat Oct 10 | two Saturdays + weekday sessions |
| Gate D | Sat Oct 17 | |
| **Gate E** | **Sat Oct 31** | **the club goal date.** Midterms Oct 21, guest lecture Oct 27 sit inside this window |
| Gate F | November | |

### Track A — Knowledge · no hardware, a good first project (L1+)

Read in this order. Each row says why it cannot be skipped.

| # | read | why | est. |
|---|---|---|---|
| A0 | his videos: [overview](https://youtu.be/_LBU3ue0CpU), [laundry RL](https://youtu.be/mTohakalKz8), the playlist linked from the [README](https://github.com/IliaLarchenko/lehome_solution) | he narrates the whole system; start here | 2 h |
| A1 | [**Learning to Fold** 2606.27163](https://arxiv.org/abs/2606.27163): §1, §2, §10 first, then the rest | the map; §10 is his list of open problems | 3 h |
| A2 | 🔴 [**BEHAVIOR-1K solution** 2512.06951](https://arxiv.org/abs/2512.06951) | **mandatory.** Correlated noise, soft inpainting, cross-layer KV mixing and multi-sample flow matching are explained **only** here `[paper §4]` | 2 h |
| A3 | [π₀.₅ 2504.16054](https://arxiv.org/abs/2504.16054) | the base model | 2 h |
| A4 | [Flow matching 2210.02747](https://arxiv.org/abs/2210.02747) + our [diffusion-policy-why](../docs/theory/diffusion-policy-why.md) | the generator | 2 h |
| A5 | [AWR 1910.00177](https://arxiv.org/abs/1910.00177) · [π*₀.₆ / RECAP 2511.14759](https://arxiv.org/abs/2511.14759) | the two RL halves | 3 h |
| A6 | [GAE 1506.02438](https://arxiv.org/abs/1506.02438) · [CFG 2207.12598](https://arxiv.org/abs/2207.12598) · [FAST 2501.09747](https://arxiv.org/abs/2501.09747) | the machinery | 2 h |
| A7 | [LeHome env 2604.22363](https://arxiv.org/abs/2604.22363) · [his blog](https://ilialarchenko.com/projects/lehome2026) | the benchmark and the checker | 2 h |

**🚪 Gate A — answer these in writing, one paragraph each.** Answers are in the cited section.

1. Why not PPO or GRPO? `[§2.2]`
2. Why apply AWR through the sampler, and what correction does that force? `[§2.3]`
3. Why is all reward withdrawn on failure? `[§6.1]`
4. Why can't the value head see the joint angles? `[§4.1]`
5. What does guidance on the advantage do, and why is α = 7–9 surprising? `[§4.3, §7.3]`
6. Why does each data source need a speed factor? `[§9.9]`
7. Why transfer from a *late but not latest* checkpoint? `[§9.4]`
8. Why did the real round run as plain imitation, and what would fix it? `[§9.5, §10.5]`

### Track B — Server · Yash (box access)

- [x] **B0** `nvidia-smi` first, always — it's a shared box. Work inside `tmux`. ✅ 2026-09-28: another lab job held 3.5 of 16 GB, which is fine for B1–B3 but not for B4 (see B4).
- [x] **B1** Build the server env **by hand** (steps 2–4 of `setup.sh`; step 5 would install Isaac Sim and has no skip flag). ✅ 2026-09-28, exactly these commands, 1 min 40 s for 283 packages, no warnings:

```bash
conda deactivate                                                  # until the prompt shows no (base): trap 16
curl -LsSf https://astral.sh/uv/install.sh | sh                   # userspace, into ~/.local/bin
export PATH="$HOME/.local/bin:$PATH"
git clone https://github.com/IliaLarchenko/lehome_solution ~/phi/lehome_solution
cd ~/phi/lehome_solution
git checkout ca73828                                              # the commit this plan was read at
git submodule update --init --recursive                           # submodules at its pins — do NOT pull
git -C openpi rev-parse --short=7 HEAD                            # must print c23745b
uv sync --python 3.11 --extra dev                                 # uv downloads its own 3.11 if none is found
cp uv.lock ~/phi/lehome_uv.lock                                   # no upstream lockfile: record what resolved
```

  **Resolved** (2026-09-28): uv 0.12.19 · CPython 3.11.16 · jax / jaxlib / jax-cuda12-plugin 0.5.3 · flax 0.10.2 · orbax-checkpoint 0.11.13 · torch 2.7.1+cu126 · torchcodec 0.5+cu126 · numpy 1.26.4 · **lerobot 0.4.4** (`>=0.4.3` in his pyproject, so not his 0.4.3 exactly) · huggingface-hub 0.35.3.

  **Which Python runs:** `uv sync` builds `.venv/` with its own 3.11 (the box's system Python is 3.10), and every `uv run` uses `.venv/`. No conda, same as `setup.sh`.
  **Skipped from `setup.sh` step 4, on purpose:** the `lehome` sim package (nothing in `src/` or `scripts/` imports it) and the torchcodec reinstall (`pyproject.toml` already pins 0.5 from the cu126 index).

  Only if you hit `cannot find -lcuda`:
```bash
mkdir -p ~/lib && ln -sf /lib/x86_64-linux-gnu/libcuda.so.1 ~/lib/libcuda.so
export LIBRARY_PATH=$HOME/lib:$LIBRARY_PATH
```

- [x] **B2** **The moment of truth:** `XLA_PYTHON_CLIENT_PREALLOCATE=false uv run python -c "import jax; print(jax.devices())"` shows the 4080. Preallocation off, because by default JAX grabs 75% of a card other people are using.
  **Why it should work:** `jax[cuda12]` ships its own CUDA runtime; the 580 driver is newer than it needs, and drivers are backward compatible — the same reason Squint's cu124 torch ran on this box `[box]`.
  ✅ **2026-09-28:** `[CudaDevice(id=0)]`, `NVIDIA GeForce RTX 4080 SUPER`, and a 1024² matmul returned the right answer. Three days before the kill date.
- [ ] **B3** Download, pinned to the Hub commits we verified against, one `--include` per call (trap 19):
```bash
uv run hf download IliaLarchenko/lehome_real --revision 03b26d31bfe6859b438cb793bb911d607226eef6 \
    --local-dir outputs/checkpoints/lehome_real                                            # 10.49 GB
for g in top_long top_short pant_long pant_short; do
  uv run hf download lehome/dataset_challenge_real --repo-type dataset \
      --revision 5a28286fb60db8bf9fab9a552a001645722752cd --local-dir data/lehome_real --include "${g}_merged/*"
done
```
  **Then prove it:** [`scripts/lehome/verify_hub_download.py`](../scripts/lehome/verify_hub_download.py) checks every file's size and hash against the Hub's listing. `fetch` saves the listing into the folder wherever the Hub is reachable; `check` runs wherever the files are.
  Use the **per-garment subsets**: the merged set ships one generic task string and no garment label `[README]`, and B6 needs the true label. Sizes `[HF]`: 1.38 + 1.31 + 0.95 + 0.84 = 4.48 GB, the same total as the merged set; ~15 GB with the checkpoint.
  ⚠️ **2026-09-28: our GPU box can't reach `huggingface.co`** (trap 17). The files are downloaded on a laptop at pinned Hub revisions, model `03b26d3` and dataset `5a28286`, then copied over SSH with the box owner's OK. Check after copying: 27 files / 10,492,307,528 B and 51 files / 4,479,323,893 B.
- [ ] **B4** Start the server with the real-round flags `[README]`:
```bash
uv run python scripts/serve.py \
  --actions_to_execute 5 --actions_to_keep 1 --execute_in_n_steps 5 --num_steps 10 \
  policy:checkpoint --policy.config pi_modified_real_bc --policy.dir outputs/checkpoints/lehome_real
```
  **On a shared box, check `nvidia-smi` first, as its own step, then decide.** The server needs ~6 GB, not the 12.8 GB first assumed (§3.3), so it can share a 16 GB card. The other job's load still skews latency, so take M1 on an idle GPU.
  ✅ **2026-09-28:** up in ~1 min. `Config: execute=5, keep=1, steps=5, num_steps=10, best_of_n=1`; norm stats and the 360×360 noise correlation matrix loaded from the checkpoint. It listens on `0.0.0.0:8000`, hard-coded (`serve.py:423`), so stop it when you're done.
- [x] **B5 · M1** Latency of one `infer_chunk` on the box, client on `localhost`: 5 warm-up calls, then 50 timed. Record median and p95.
  ✅ **M1, 2026-09-28, idle GPU: median 679 ms, p95 696 ms.** The same requests to a stand-in with no model take 488 ms, so **~0.49 s is the message and ~0.19 s the model** `[derived]` (§6).
  **2026-10-03, after the compression fix: 279 ms median, p95 284 ms** (§6).
  **2026-09-28, GPU shared with another job (13–25% load): median 763 ms, p95 785 ms, min 507 ms.** Not the official M1; re-run on an idle GPU. If it holds, the arm moves ~25% of the time even with no network, 3× over the §6 budget. Still to separate: the other job's load, the on-demand `platform` allocator (§3.3), and the protocol's own ~130–146 ms (§6).
- [x] **B6 · The replay gate** — the software is only correct if it predicts what the organizer's operator actually did.

  **In one line:** feed the server frames from the organizer's own recordings and check it predicts their next moves.

```
for 20 episodes (5 per garment type), 10 evenly spaced frames each:
    send   state[t] + the three images under their wire keys + the true garment id
           (no initial_actions: each frame is a fresh session, so no anchor from a previous chunk)
    get    5 actions â[t … t+4]
    error  = mean |â − action[t … t+4]|, per joint      (degrees; gripper in 0–100 units)
baseline   "hold": repeat state[t] five times — a predictor that does nothing
```

  **The test must be able to fail.** Each control injects one bug the gate exists to catch:

| control | change                                                                   | expected                                         |
| ------- | ------------------------------------------------------------------------ | ------------------------------------------------ |
| C1      | swap the two wrist images                                                | error rises                                      |
| C2      | rotate the top image 180° (the *sim* convention `[code: config.py:801]`) | error rises                                      |
| C3      | pair each state with another episode's images                            | error rises sharply                              |
| C4      | scale arm joints ×0.9 (the units-bug size `[paper §9.11]`)               | **informative only**: how much it leans on state |

  **Pass, fixed before the first run** `[estimate]`: policy error < 0.5 × hold · each of C1–C3 raises error > 1.5× · holds on the mean of 3 seeds.
  **Scope:** these are frames it trained on. This gate tests the plumbing — keys, units, resize, normalisation — not generalisation.
  **No arms, no network.** It can also run as a batch job on Explorer if the box is busy.
  **Tool:** [`scripts/lehome/replay_gate.py`](../scripts/lehome/replay_gate.py), built 2026-09-28. It sends exactly what Ilia's robot client sends, with each piece cited to his code, and reads the organizer files with pyarrow + PyAV, so it doesn't depend on a lerobot version. Episodes 0, 31, 62, 93, 124 per garment; 10 frames each; 3,000 requests in all. Tested so far `[mac]`:
  - `dry-run`: 200 samples, **6.14 MB per request (the §6 figure, now measured)**, hold baseline **3.841**, so the policy must reach **< 1.92**.
  - Against a stand-in server that only holds still: **FAIL, exit 1**, with policy = hold (1.000×) and C1–C3 unchanged (1.00×). The gate rejects a do-nothing server.
  - **Run against the real server, 2026-09-28 `[box]`: FAIL by the rule above.** 3,000 requests. Policy **1.538 = 0.40× hold** ✅. C3 **2.19×** ✅. **C1 1.18× ❌ and C2 1.28× ❌**, both below the 1.5× bar. Pass means 1.545 / 1.537 / 1.531. C4 2.88×.
  - **Post-hoc, labelled as such** (`replay_gate.py report`): every control makes predictions worse, 95% bootstrap CIs C1 [1.13, 1.24], C2 [1.17, 1.38], C3 [1.93, 2.49], sign tests p ≤ 4×10⁻⁷. So the wiring beats each wrong alternative: a swapped or rotated default would have *improved* under C1/C2. **The 1.5× bar was mis-set**: over 250 ms the policy leans on joint state and task phase more than on camera identity. Per garment, policy/hold: top_long 0.29 · top_short 0.33 · pant_long 0.47 · pant_short 0.56.
  - **Rule v2, fixed 2026-09-28 19:40 PDT, before run 2 (the confirmation run).** It was chosen after seeing run 1, which is why it's tested on episodes run 1 didn't use.
    - **Samples:** episodes **15, 46, 77, 108** per garment (none used in run 1), 10 frames each, so 160 samples, 3 passes, fresh server (noise from `key(0)`).
    - **Pass = all of:**
      1. policy error < 0.5 × hold.
      2. for **each** of C1, C2, C3, all three of:
         - the 95% bootstrap CI of (control error / policy error) lies entirely above 1.0 (10,000 resamples over samples, seed 0);
         - an exact two-sided sign test on the per-sample differences gives p < 0.01;
         - more samples come out worse than better.

      Per-sample errors are averaged over the 3 passes. C4 is reported, not graded.
    - **Why this rule:** the gate asks whether our wiring beats each wrong alternative, which is a question of direction and reliability. Run 1's 1.5× demanded a size of effect we had no basis for.
    - **If it fails:** treat it as a real wiring problem. No robot runs until it's found.
    - Run 1's FAIL stays on the record.
  - ✅ **Run 2, 2026-09-28: PASS under rule v2.** Policy **0.385× hold** · C1 **1.12** [1.05, 1.20] · C2 **1.29** [1.16, 1.42] · C3 **2.27** [1.99, 2.58], sign tests p ≤ 1.5×10⁻⁹ · C4 2.89×. **It replicates run 1 on unseen episodes** (0.40 / 1.18 / 1.28 / 2.19 / 2.88).

**🚪 Gate B:** B2 green · B6 passes · M1 recorded.
**Status 2026-09-28: 🟢 green**, five days early. B2 ✅ · B6 ✅ (run 2, rule v2) · M1 ✅ 679 ms. **Deviation on the record:** B6's rule was corrected after run 1 failed it, fixed in writing first, then confirmed on unused episodes.
**Kill rule:** if JAX can't see the GPU by **Thu Oct 1**, stop and choose: a userspace CUDA toolkit from conda-forge, Explorer as the server for B6, or ask the box owner. Don't grind silently.

### Track C — Rig · Sai, Parv

- [ ] **C1** Calibrate under the ids **his runner hardcodes**, so one calibration serves plain teleop now and his runner later: followers **`bimanual_follower_left`** / **`_right`**, leaders **`bimanual_leader_left`** / **`_right`** (`--robot_id` defaults to `bimanual_follower`, the leader id is fixed at `bimanual_leader` `[code: record_real_dagger.py:847, :890]`; sub-arms are `{id}_left/_right` `[mac]`). The existing pair keeps its calibration: copy `robots/so_follower/phi_follower.json` → `bimanual_follower_left.json` and `teleoperators/so_leader/phi_leader.json` → `bimanual_leader_left.json`. Only the new pair needs `lerobot-calibrate`. The assembled Kit Pro ships with motor ids set, so skip `setup-motors` ([02-setup](../docs/robots/so-arm101/02-setup.md))
- [ ] **C2** Mount the followers **46 cm apart** on one rigid board `[README]`. Clamps drift.
- [ ] **C3** Cameras, named exactly as the organizer's:

| key | camera | mode | placement |
|---|---|---|---|
| `right_front` | overhead webcam | **1280×720** | ~65 cm above the table `[README]`; match the organizer frames |
| `left_wrist` | wrist webcam | 640×480, MJPG | left gripper |
| `right_wrist` | today's `front` camera, remounted | 640×480, MJPG | right gripper — needs a second wrist mount (print it) |

  Record each camera's model; our notes name only two (EMEET C960, Logitech Brio 101). Confirm each supports its mode.
- [ ] **C4** 🔴 **Buy the garments this week:** kid-sized long-sleeve top, short-sleeve top, long pants, shorts — matched to the organizer's overhead frames. He spent months assuming adult clothes `[paper §9.1]`. At least 2 instances per type.
- [ ] **C5** Record one manual bimanual episode at 20 Hz with `bi_so_follower` + `bi_so_leader`. Load it back. Diff its `meta/info.json` features against §3.2: names, shapes, fps must match.
- [ ] **C6 · The overlay gate** — `scripts/real_camera_align.py` drives the arms to an organizer joint state and overlays our live cameras on their frame `[paper §9.7]`. It runs on the Mac: its keyboard uses POSIX `termios`, not `evdev`, and RealSense is only imported if configured `[code]`.
  **Why this checks calibration too:** a joint angle means "degrees from *this arm's* calibrated zero". If our zero sits elsewhere, the arm lands elsewhere even with perfect cameras.

**🚪 Gate C:** C5's schema matches · overlay screenshots show the arms coinciding in all three views (saved in the PR).
**Then do the opposite on purpose:** move cameras slightly, change lighting, re-calibrate between sessions. His policy survived rig drift because no exact geometry was ever load-bearing `[paper §9.7, §10.3]`.

### Track S — Simulation · side track, started 2026-10-17

**Why:** reproduce Ilia's own simulation result (~80% success, 1st of 62) with `lehome_sim`, the round-1 policy that `lehome_real` was fine-tuned from. That proves we run his full system, whatever state the rig is in, and gives a sandbox for §12 research without arm time. It needs the GPU, not the arms, so it runs beside Track C.

**Constraints, verified:** the challenge pins **Isaac Sim 5.1.0**, whose **minimum GPU is the RTX 4080** (16 GB). Our GPU box meets it exactly; A100/H100/H200 can't (no RT cores). On 16 GB, plan on **1 simulator worker** (Ilia ran 4 on 96 GB) `[estimate]`.

| gate | passes when | by |
|---|---|---|
| **S1** | Isaac Sim + lehome installed without sudo; the simulator starts headless | **Fri Oct 24 (kill date)** |
| **S2** | 2 episodes × every garment complete, no crash; GPU peak + time per episode recorded | Oct 27 |
| **S3** | his success rate on his evaluation setting. **The pass rule is written here before the run** | Nov 7 `[estimate]` |

**Scripts** (`scripts/lehome/sim/`): `setup_sim_box.sh` (replaces setup.sh step 5: no hidden errors, IsaacLab fork pinned to `69f6fa5`, NVIDIA's license accepted only by explicit opt-in) · `gpu_guard.sh` (go/no-go on a shared GPU) · `run_sim_eval.sh` (one recorded run: versions, GPU trace every 5 s, log).
**Pins:** `lehome_sim` @ `79dcf9e` · `lehome/asset_challenge` @ `bea65fd` · lehome-challenge @ `5ea947e` · warp-lang 1.12.1 · flatdict 4.0.1.

### Phase D — Integrate

- [x] **D1** Port the runner to macOS + lerobot 0.6.0 (§8). Apache-2.0: keep the notice, list every change.
  ✅ **2026-10-03:** [`scripts/lehome/record_real_dagger_mac.py`](../scripts/lehome/README.md), 10 marked changes, plus a `--dry_run` mode in which nothing moves. Tested from the Mac against the real server with demo frames: correct moves, garment auto-detect right for all 4 types, and a missing joint is refused. Not yet run with arms and cameras.
- [ ] **D2 · M2** Mac → box throughput: `time scp` a 100 MB file, both directions.
- [ ] **D3 · M3** End-to-end `t_req` from the Mac. Apply the §6 decision rule.
  **First reading, 2026-10-03, via the Tailscale relay (off campus): 2.46 s median** with full-size frames. Rule 1 fails, so **a′ (224² frames from the client)** is next; re-measure on campus with a direct path.
- [ ] **D4** First autonomous episode. Start with **shorts** — shortest fold (13.6 s) and his easiest type in sim (93.5%). Safety: torque is already capped at 50% ([control-loop §3](../docs/theory/control-loop.md)); set `max_relative_target` on both followers `[mac]`; hand on `SPACE` to take over.

**🚪 Gate D:** one episode runs to success or timeout with no faults · `t_req` logged · video saved.

### Phase E — Zero-shot baseline · the Oct 31 target

Run the §9 protocol: **4 garments × 20 trials = 80 trials**, no training.

**🚪 Gate E:** the number exists, every trial on video — **whatever it is.** Low is expected: he never achieved zero-shot sim → real, and ours is real → *different* real `[paper §9.1]`. But his policy was unusually robust to rig drift `[paper §10.3]`, so it's genuinely unknown.

### Phase F — Fine-tune · November

- [ ] **F1** Collect with the ported runner, DAgger mode. **Easy starts first**: his stated regret is chasing hard initial states when the final used mostly easy ones `[paper §9.10]`.
- [ ] **F2** Explorer env in `/scratch`: Python 3.11, `uv`, `jax[cuda12]`, `PYTHONNOUSERSITE=1` ([pi05 lessons](2026-09-15_pi05-cubcyl-lang.md)). Smoke job on `sharing` first — our π₀.₅ run found two blockers that way.
- [ ] **F3** **Decision: normalisation stats.** Default: keep his `real_bc` stats, so step 0 sees the inputs it was trained on. Recomputing over the union is the A/B.
- [ ] **F4** Train from `lehome_real` params. Our speed factor = our mean fold time ÷ the organizer's, per garment (his method `[paper §9.9]`). Resolve trap 13 first. Chain 8 h jobs with `scripts/train.py --resume`.
- [ ] **F5** Re-run the §9 protocol. Apply the pre-registered comparison.

**🚪 Gate F:** the comparison, reported whatever it says.

### Phase G — Research
See §12. Starts only after F.

---

## 8. What must be built

| piece | reuse | write |
|---|---|---|
| policy server | `scripts/serve.py`, unchanged | — |
| replay gate | organizer dataset + the client's message format | `scripts/lehome/replay_gate.py`, with C1–C4 · ✅ written 2026-09-28, self-tested, not yet run live |
| download check | the Hub's own file listing | `scripts/lehome/verify_hub_download.py` · ✅ 2026-09-28 |
| Mac robot client ✅ 2026-10-03 | `scripts/record_real_dagger.py` | move 2 imports · `evdev` → `pynput` (grant the terminal *Input Monitoring*) · `pip install websockets` · macOS ports (`/dev/tty.usbmodem*`) and camera indices in `configs/lehome/real_robot_mac.yaml` · optional 224² client-side resize (§6) |
| overlay tool | `scripts/real_camera_align.py` | only the Mac yaml |
| second wrist mount | the existing wrist mount | print a mirror |
| Explorer training | `scripts/train.py --resume`, `configs/train_real_bc.yaml` | `configs/hpc/train_lehome_real.sbatch` |
| results | this doc | the per-trial sheet (§9) |

---

## 9. Evaluation protocol — fixed before any data exists

| | rule |
|---|---|
| trial | garment laid flat in a marked start zone (reference photo); runs to done, **90 s timeout** `[estimate: ~4× the longest mean fold]`, or unsafe stop |
| design | 20 per garment; order randomised in **blocks of 4** (one of each type) so drift — light, fatigue, a wearing gripper — spreads evenly |
| primary | binary success on our rubric: the fold matches the organizer's end state for that garment (photographed reference). *Their* jury's exact rubric isn't published |
| secondary | fold steps completed · time to success · `t_req` · failure mode: missed grab, imprecise fold, double grab, poor flattening, drop, bad placement ([Dyna's production taxonomy](https://www.dyna.co/research/scaling-customer-deployments)) |
| scoring | from the final overhead photo + video; for F comparisons, videos are labelled by hash so the scorer doesn't know the condition |
| stats | Wilson 95% interval per rate; comparisons by Fisher's exact test, two-sided α = 0.05 (a stratified Cochran–Mantel–Haenszel test by garment is the stricter alternative) |

**What 20 per garment can and cannot tell us** `[derived: Wilson intervals; Fisher power by 4,000-run simulation]`:

| result | 95% interval |
|---|---|
| 4/20 | 0.08 – 0.42 |
| 10/20 | 0.30 – 0.70 |
| 16/20 | 0.58 – 0.92 |

| true change | power at n = 20 per arm | 40 | 80 (pooled, 4 × 20) |
|---|---|---|---|
| 0.2 → 0.5 | 0.41 | 0.74 | **0.97** |
| 0.5 → 0.8 | 0.39 | 0.75 | **0.97** |
| 0.4 → 0.6 | 0.14 | 0.37 | 0.67 |

**So:** per-garment numbers are descriptive only. The **pooled 80-trial** comparison reliably detects a 30-point change. A 20-point change needs far more trials than we can afford — say so rather than claim it.

---

## 10. Budget

| | amount |
|---|---|
| one 80-trial evaluation | ~2 h of arm time `[estimate: 18.7 s mean fold + reset + scoring ≈ 1.5 min]`, plus inference freezes (§6) |
| garments | ~$40 `[estimate]` |
| second wrist mount | printed, ~free |
| foot pedal (optional) | ~$30 `[estimate]` |
| box disk | 10.49 GB checkpoint + ~4.5 GB data of 1.1 TB free `[HF, box]` |
| Explorer | queue waits: ~12 h estimated start seen on our π₀.₅ run |

---

## 11. Paper vs code, and what is not yet verified

Resolve each row before the phase named.

| # | question | status | before |
|---|---|---|---|
| 1 | real rate | paper 20 Hz · runner default 20 ✅ | — |
| 2 | guidance on real | paper: removed · client sends 5.0 · code ignores it ✅ | — |
| 3 | real inference knobs | not tabulated in the paper; README + client give n_e 5, n_a 1, τ 0.9, onset 0.5 | B4 |
| 4 | the jury's rubric | not published | E — ours is stated in §9 |
| 5 | overhead camera pose | organizer calls it `right_front`, he calls it overhead | C3 — match the frames, not the names |
| 6 | "46 cm apart" measured between what | assumed base centre to base centre | C2 — confirm with the overlay |
| 7 | does inference touch a gated asset? | FAST is training-only `[paper §4]`; unverified end to end | B4 |
| 8 | 4080 latency | unknown | M1 |
| 9 | LR schedule on a params-only restore | trap 13 | F4 |
| 10 | keep vs recompute norm stats | decision F3 | F4 |

---

## 12. Research, once we match him — ranked for raising accuracy

| # | idea | why it's worth it | needs |
|---|---|---|---|
| 1 | **Bandit-tuned inference on the real robot** | in sim he tuned 7 knobs online by Thompson sampling and called it *"a big win that cost no extra training"* `[paper §10.1]`. In the real round he ran them **fixed**. Zero training cost; no real-robot version exists in his work `[my reading]` | a success label per episode (a human, at our scale) |
| 2 | **A real-side value function** → advantage conditioning + best-of-N on real | his biggest regret; his 90%+ guess `[paper §10.5]`. Candidate signal: `Present_Load` / `Present_Current`, free on our servo bus ([control-loop §2](../docs/theory/control-loop.md)) — mind the 30 Hz aliasing caveat there | EXPO-FT's reward + critic loop |
| 3 | **DAgger weighted by value**, not his hand-tuned ramp | he calls his scheme *"deliberately crude"* `[paper §9.10]` | #2 |
| 4 | **Recovery without a human** | the stated open problem `[paper §10.4]` | — |
| 5 | **A model small enough for the Mac** | *"this task could be solved with a much smaller model"* `[paper §10.2]`. Removes the network problem entirely. We already train SmolVLA | Phase F data |
| 6 | **Steerable placement** | [Dyna found](https://www.dyna.co/research/scaling-customer-deployments) precise instruction-following was the missing capability for bin placement. His model takes no text | language labels |
| 7 | **Sim puzzles**: resize fingerprinting `[§9.1]`; best-of-N helping at ~zero ranking correlation `[§7.4]` | cheap, publishable | Isaac Sim |

---

## What counts as a result

Any of these, including the negative ones:
- **A zero-shot number** on a rig the policy never saw, with its interval
- **The replay gate** — a reusable check that a VLA server is wired correctly, before it touches hardware
- **The Mac port**, upstreamed if he wants it
- **Any pre-registered comparison**, whatever it says

⚠️ **The success rubric grades; nothing learned grades itself.** Every evaluation success is confirmed by a person.

## Results

_Filled in as each gate lands._

| phase | condition | garment | successes / n | interval | notes | date · commit |
|---|---|---|---|---|---|---|
| | | | | | | |

## Log

- **2026-09-27** Created. Every §3 fact verified from source, the Hub, or the Mac env on this date.
- **Correction, 2026-09-27:** an earlier plan (chat, 2026-09-24) said fine-tuning "will not fit 16 GB" and deferred it. True for the 4080 — but Explorer's H200, the same GPU Ilia used, is already in club use ([pi05](2026-09-15_pi05-cubcyl-lang.md)). Fine-tuning moves to Explorer.
- **2026-09-28** B1 re-checked line by line against `setup.sh` and `pyproject.toml` before first run. No conda (uv supplies 3.11), save the resolved `uv.lock` (upstream has none), two unneeded installs dropped. Traps 15–16 added; three `setup.sh` line citations corrected.
- **2026-09-28** **B0–B2 done.** Upstream `main` is still `ca73828`, so our pin is his latest. B1 built cleanly in 1 min 40 s; **B2 passes**: JAX sees the RTX 4080 SUPER. Traps 10 and 15 don't apply on our box: the `libcuda` link, CUDA 12 NPP and FFmpeg with AV1 are already installed. Next: B3 downloads; B4 waits for an idle GPU, since another lab job held 3.5 GB and the server wants 12.8.
- **2026-09-28** B3: the GPU box's network resets TLS connections to `huggingface.co` (trap 17). Downloads go through a laptop at pinned Hub revisions, then SSH, pending the box owner's OK.
- **2026-09-28** B3 on the laptop: **both downloads verified** by size and hash against the Hub listing. Checkpoint 27/27 files, 10,492,307,528 B; dataset 51/51 files, 4,479,323,893 B. Found on the way: traps 18 (Xet stall) and 19 (`--include` parsing changed in `hf` 1.x, which silently dropped one subset). New tool: `scripts/lehome/verify_hub_download.py`.
- **2026-09-28** B5/B6 tool `scripts/lehome/replay_gate.py` built and self-tested on the laptop. The dry run measured the 6.14 MB request; against a hold-still stand-in the gate FAILs as it should. The protocol alone costs ~130–146 ms per request (§6). Trap 20 found while reading the runner: missing joint names become 0.0 silently.
- **2026-09-28** Box owner OK'd GPU use and the file copy. Files copied and re-verified on the box by hash. **B4** server up in ~1 min; it uses **~6 GB, not 12.8** (§3.3 corrected). **B5**, shared GPU: 763 ms median. **B6: FAIL by its pre-set rule.** Policy 0.40× hold ✅ and C3 ✅, but C1 1.18× and C2 1.28× missed the 1.5× bar. Post-hoc, every control is worse with CIs clear of 1.0, so the plumbing looks right and the bar was mis-set. A corrected criterion is written into B6 before any re-run. Transport alone on the box: 516 ms (§6).
- **2026-09-28** Rule v2 written into B6 at 19:40, before run 2, with a hashed record. **Run 2 PASS** on episodes 15/46/77/108, replicating run 1. Clean **M1: 679 ms** (488 ms of it transport). **Gate B green**, five days early.
- **2026-10-17** Track S (simulation) started: pins recorded, sim checkpoint + assets downloading at pinned revisions, box install scripts written. S1 kill date Oct 24.
- **2026-10-03** D1: runner ported to the Mac and link-tested against the real server (no arms yet). From the Mac via the Tailscale relay: 2.46 s per request, so client-side 224² frames are next.
- **2026-10-03** Speed cause found: websocket compression (trap 21). Client-side `compression=None` cuts a request from **667 to 279 ms**. `replay_gate.py` now defaults to it (`--compression`).

## References

**Papers**: [Learning to Fold](https://arxiv.org/abs/2606.27163) · [BEHAVIOR-1K solution](https://arxiv.org/abs/2512.06951) · [π₀.₅](https://arxiv.org/abs/2504.16054) · [π*₀.₆ / RECAP](https://arxiv.org/abs/2511.14759) · [AWR](https://arxiv.org/abs/1910.00177) · [GAE](https://arxiv.org/abs/1506.02438) · [CFG](https://arxiv.org/abs/2207.12598) · [FAST](https://arxiv.org/abs/2501.09747) · [Flow matching](https://arxiv.org/abs/2210.02747) · [LeHome env](https://arxiv.org/abs/2604.22363) · [XSA](https://arxiv.org/abs/2603.09078) · [DAgger](https://arxiv.org/abs/1011.0686)

**Code & data**: [lehome_solution](https://github.com/IliaLarchenko/lehome_solution) @ `ca73828` (Apache-2.0) · [lehome_real](https://huggingface.co/IliaLarchenko/lehome_real) · [lehome_sim](https://huggingface.co/IliaLarchenko/lehome_sim) · [dataset_challenge_real](https://huggingface.co/datasets/lehome/dataset_challenge_real) · [official challenge](https://github.com/lehome-official/lehome-challenge)

**Repo**: [control-loop](../docs/theory/control-loop.md) · [smolvla](../docs/theory/smolvla.md) · [diffusion-policy-why](../docs/theory/diffusion-policy-why.md) · [Real-Time EXPO-FT](2026-09-26_realtime-expo-ft-so101.md) · [π₀.₅ cubes/cylinder](2026-09-15_pi05-cubcyl-lang.md)

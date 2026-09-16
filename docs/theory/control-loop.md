# The control loop, and where force lives

**Applies to**: every policy we run on the SO-ARM101. Nothing here is model-specific.
**Prerequisites**: none. If you have read [act-why](act-why.md) you will recognise action chunking; it is re-derived here from the controller's side.
**Every number below is measured from the installed `lerobot` on our own rig, or quoted from a paper.** Where a number is not published, this document says so instead of guessing.

---

## 1. There is no "the" control loop. There are three.

A policy does not drive a motor. It writes a number that something else turns into current. On our arm that chain has three nested loops running at wildly different rates:

| Loop | Rate | Where it runs | How we know |
|---|---|---|---|
| **Policy decision** | **~0.6 Hz** | host GPU/CPU | 48-step chunk ÷ 30 fps = one decision per 1.6 s |
| **Action playback** | **30 Hz** | host → serial bus | `fps: 30` in every `meta/info.json` we have recorded |
| **Servo position PID** | 🔴 **unpublished** | inside each STS3215's own MCU | not in the control table, not in `lerobot`, not in Feetech's e-manual |

**The third row is the honest answer to "what frequency does the controller run at."** Nobody publishes it. It is firmware on the servo's microcontroller. All we can say with confidence is that it must be much faster than 30 Hz, because it has to close position between our commands. **Do not put a number on it in a talk or a blog post.**

**The bus is not the bottleneck.** `DEFAULT_BAUDRATE = 1_000_000` in `lerobot/motors/feetech/feetech.py`. A six-motor sync-read is tiny against 1 Mbps. We run at 30 Hz because of the **cameras**, not the servos. Faster proprioception is available; faster vision is not.

### Why the chunk makes this worse than it looks
Action chunking is what made behaviour cloning work (ACT: chunk 1 → 1% success, chunk 100 → 44%). It is also what opens the gap. Our policy commits **48 steps, 1.6 seconds**, then stops looking. So the *effective* decision rate is not 30 Hz, it is **0.6 Hz**.

---

## 2. What the servo actually is

From `lerobot/motors/feetech/tables.py`, `STS_SMS_SERIES_CONTROL_TABLE`. Addresses are `(byte_address, size)`.

**Writable, and this is the interesting part:**

| Register | Addr | Section | What it does |
|---|---|---|---|
| `P_Coefficient` | 21 | **EPROM** | ⭐ **the stiffness of the arm** |
| `D_Coefficient` | 22 | **EPROM** | damping |
| `I_Coefficient` | 23 | **EPROM** | integral term |
| `Operating_Mode` | 33 | EPROM | see §3 |
| `Max_Torque_Limit` | 16 | EPROM | blunt global effort ceiling |
| `Protection_Current` | 28 | EPROM | over-current cutoff |
| `Overload_Torque` | 36 | EPROM | torque held when overloaded |
| `Goal_Position` | 42 | **SRAM** | what we write every frame |
| `Torque_Limit` | 48 | SRAM | runtime torque cap |

**Read-only feedback available on the bus:**

`Present_Position` (56) · `Present_Velocity` (58) · **`Present_Load` (60)** · `Present_Voltage` (62) · `Present_Temperature` (63) · **`Present_Current` (69)**

🔑 **Force information already exists on our hardware.** `Present_Load` and `Present_Current` are sitting on the bus, free to read. See §6 for why we do not read them and what it would cost.

> **Two registers we cannot explain.** `DTs (81,1) # (ms)` and `Hts (83,1) # (ns)`, both in the factory section. The names suggest timing constants and `Hts` is annotated *"valid for firmware >= 2.54"*. `lerobot` never touches them. **Unknown. Do not build anything on a guess about them.**

---

## 3. Our arm is a stiff position servo, by explicit configuration

`lerobot/robots/so_follower/so_follower.py` writes this on **every connect**:

```python
self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
# Set P_Coefficient to lower value to avoid shakiness (Default is 32)
self.bus.write("P_Coefficient", motor, 16)
self.bus.write("I_Coefficient", motor, 0)
self.bus.write("D_Coefficient", motor, 32)
self.bus.write("Max_Torque_Limit",   motor, 500)  # 50% of max torque to avoid burnout
self.bus.write("Protection_Current", motor, 250)  # 50% of max current to avoid burnout
self.bus.write("Overload_Torque",    motor, 25)   # 25% torque when overloaded
```

**`P_Coefficient = 16` is the single number that sets how hard our arm fights you.** Feetech's default is 32; lerobot halves it, and the code comment tells you why: *"to avoid shakiness."* That comment is a compressed statement of the whole speed/precision tradeoff, see §5.

### The available modes, and the one that is missing
`OperatingMode` in `feetech.py`:

| Value | Mode | Comment in source |
|---|---|---|
| 0 | **POSITION** | what we use |
| 1 | VELOCITY | *"constant speed mode"* |
| 2 | **PWM** | *"PWM open-loop speed regulation mode"* |
| 3 | STEP | *"step servo mode"* |

🔴 **There is no torque or current mode.** You cannot command force on this hardware. `PWM = 2` is the closest thing and it gives up position closure entirely. This is the hard constraint everything else in this document works around.

---

## 4. Why a closed position loop is still force-blind

A position PID is **stiff by construction**. Block the arm, position error grows, P multiplies that error into more current, and the arm pushes *harder*. It does not yield to contact, it rejects it as a disturbance. That is the correct behaviour for a position servo and the wrong behaviour for picking up an egg.

`Max_Torque_Limit = 500` is the only thing stopping it, and a global ceiling is not compliance: it caps the worst case identically for every task, every moment.

**So "our policy ignores force" is two separate failures, and it is worth keeping them apart:**

1. **The policy never perceives force.** `Present_Load` and `Present_Current` are not in the observation.
2. **The controller senses load but is tuned to reject it.** Compliance is not a sensor, it is a *commanded parameter*, and ours is a constant.

---

## 5. Speed and precision trade against each other, and P is the dial

This is the part members get wrong, so state it plainly.

Commanding positions faster does **not** make the joint arrive faster. With a soft gain the joint **lags further behind** under speed, so tracking error grows. To move faster *and* stay accurate you must raise `P_Coefficient`, which buys tighter tracking and costs compliance and adds vibration. Which is exactly why lerobot lowered it to 16.

Dyna hit the identical wall in production and said it in product terms: *"Speed and quality pulled against each other… Neat and fast were competing for the same seconds."* Their 2.7× throughput gain (35 → 95 napkins/hour, 75% → 93% quality) came from **better models and tooling, not faster hardware**.

**Corollary for camera upgrades.** Raising camera fps lowers observation staleness (33 ms → 8 ms at 120 fps) and improves velocity estimates. It does **not** buy speed, because the servo gain is the limit, and it does **not** buy contact sensing, because contact resolves in **1–10 ms** and vision can only ever see deformation, never force.

---

## 6. Where force could enter, and what blocks each route

| Route | Verdict |
|---|---|
| **`Present_Load` + `Present_Current` in the observation** | ✅ possible, but see the timing caveat below |
| **Stiffness as a per-timestep action** (SoftMimic-style) | 🔴 **blocked.** `P_Coefficient` is in **EPROM**, not SRAM |
| **Stiffness as a per-episode condition** | ✅ real experiment, see §9 |
| **Direct force/torque command** | 🔴 no such operating mode |
| **PWM mode with our own loop** | ⚠️ possible, needs a microcontroller, see §8 |

### The EPROM blocker, in full
`P_Coefficient` sits at address 21, under the `# EPROM` comment in the control table. `Goal_Position` sits at 42, under `# SRAM`. Flash has finite write endurance (this servo class is typically rated around 10⁵ writes), and there is a `Lock` register at 55 of the kind that normally guards EPROM writes.

Writing stiffness at 30 Hz would exhaust ~10⁵ writes in **under an hour** and kill the servo. **A per-step stiffness action is not a tuning problem on this hardware, it is impossible.**

### 🔴 We currently read load and current at 0 Hz
The entire per-loop read, from `so_follower.py`:

```python
def get_observation(self) -> RobotObservation:
    # Read arm position
    obs_dict = self.bus.sync_read("Present_Position")
```

Only `Present_Position`. And `sync_read(data_name: str, ...)` takes **one register name per call**, so adding load and current means **three bus round trips per loop instead of one.**

**Measure the cost before committing to it.** The code already logs it: `logger.debug(f"{self} read state: {dt_ms:.1f}ms")`. Enable debug logging, read the number, multiply by three, compare against the 33.3 ms budget that 30 fps gives you. The cameras already eat most of it.

### ⚠️ And temper the expectation: force in the observation buys much less than it sounds
Two timing facts, neither about the sensor:
- **Sample period.** Contact resolves in 1–10 ms; we sample every 33 ms. Load appears in an observation *after* the damage.
- **The chunk, which is worse.** The policy commits 1.6 s and stops looking. Even if it saw the spike, it is not re-planning.

**So force-in-observation is not a route to contact reactivity on this stack.** Its real value is **offline**, where rate does not matter: labelling which grasps slipped, a continuous quality metric instead of binary success, and the ground-truth current-versus-motion data that **ServoGap** needs (internal project: fitting a MuJoCo actuator model to our own STS3215). Plus one narrow runtime case that does survive the chunk: **slow-varying context** ("am I carrying something heavy") changes over seconds, not milliseconds.

---

## 7. The frequency ladder, across the field

| System | Decision | Middle | Control |
|---|---|---|---|
| **Ours (SO-ARM101)** | ~0.6 Hz | **30 Hz** | servo PID, **unpublished** |
| **ACT** (paper) | — | **50 Hz**, 100-step chunk = 2 s | unspecified |
| **Diffusion Policy** (paper) | — | **10 Hz** | *"interpolation controller"*, rate not given |
| **Figure Helix 02** | **7–9 Hz** | **200 Hz** | **1 kHz**, learned, 10 M params |
| **Dyna-2** | **~1 Hz** | **~10 Hz** | **~1 kHz** |
| **π0** | — | 50-step chunks | ⚠️ execution rate **not verified by us** |
| **Human** | **~5 Hz** (visual reaction 200–250 ms) | ~20–30 Hz (spinal reflex 30–50 ms) | mechanical: muscle is a tunable spring |

Two things fall out of that table.

**Humans are slow at vision and fast at force.** Our arm already beats human visual reaction by roughly six times. What we lack is the bottom two rows. Dexterity is not a perception-rate problem.

**Our gap is the widest in the table, with nothing in the middle.** Figure bridges 7–9 Hz to 1 kHz with a 200 Hz reactive layer. We go from one decision per 1.6 s straight into a fixed-gain PID we never retune per task. That is not a misconfiguration; it is what the SO-101 stack *is*. It is also why "add force" is not a feature you switch on: **there is no layer to put it in.**

---

## 8. "Modelling the controller" means three different things

The phrase gets used loosely and the three meanings have completely different feasibility for us.

| | What it is | Where it runs | Ours? |
|---|---|---|---|
| **(a) Learned actuator model** | a network mapping command + state → what the motor *actually did*, replacing the analytical motor model | **in simulation** | ✅ this is **ServoGap** (internal project, no public write-up yet) |
| **(b) Learned controller** | a network emitting actuator commands at kHz, replacing a hand-written PD | **on the robot** | 🔴 see below |
| **(c) System identification** | measure what `P = 16` actually does | offline analysis | ✅ cheap, undone |

**(a) is Hwangbo et al. 2019** (*Science Robotics*): a learned actuator model, built precisely because the analytical model was what broke sim-to-real transfer. Legged locomotion crossed over to reliable hardware years before manipulation partly because of this.

**(b) is Figure's System 0**: 10 M parameters at 1 kHz, trained on over a thousand hours of joint-level retargeted human motion, and by their own account it *"replaces 109,504 lines of hand-engineered C++ with a single neural prior."*

### 🔴 We cannot do (b) on this arm, because the controller is not ours
The PID runs on the STS3215's own sealed microcontroller. There is no register that replaces its loop. Figure builds their own actuators. Eka builds their own wrists and grippers — and argues the *opposite* of Figure, that a backdrivable low-inertia arm gives *"natural backdrivability for safe contact **without complex compliant controllers**"*, i.e. mechanism instead of policy. We are using a hobby servo with closed firmware and neither option is open to us as shipped.

### ⭐ Except one door, and we have already built this pattern
`OperatingMode.PWM = 2` turns off the servo's position loop and takes raw duty cycle. **Then the loop is ours.**

At a 30 Hz host rate that yields a hopeless position controller; a stable one needs hundreds of Hz. So the loop has to live closer to the motor — which is **exactly the [SockBot] architecture**: an ESP32 taking high-level commands over serial at 115200 while the fast loop runs locally.

**Put a microcontroller between the host and the arm bus, run the servos in PWM mode, and there is somewhere to put a System 0.** That is the only route on this hardware from "position targets into a sealed PID" to a control layer we own, and it costs about $10 rather than a new arm. It is unbuilt and unproven here. Treat it as the most interesting open project in this document, not as a recommendation.

---

## 9. What is load-bearing, and what is accident

Per this section's convention elsewhere in `theory/`:

**Load-bearing (do not change casually):**
- `Operating_Mode = POSITION`. Everything in the stack assumes position targets.
- `Max_Torque_Limit = 500` and `Protection_Current = 250`. These exist to stop burnout. The comments say so.
- 30 fps. It is the camera rate and it sets the dataset clock.

**Accident, or at least a choice you may revisit:**
- **`P_Coefficient = 16`.** It is halved from Feetech's default purely to stop visible shake. Nothing about our task chose it. It is a single byte and it is the arm's compliance.
- **`I_Coefficient = 0`.** No integral term at all, so steady-state error under a constant load is never corrected.
- **Reading only `Present_Position`.** Four other feedback registers are free on the bus.

---

## 10. Ordered things to try

1. **System ID on `P_Coefficient`** (§8c). Sweep P over a few values, measure tracking error and contact force on a fixed trajectory. Cheapest experiment in the repo, needs no purchase, and gives the number the rest of this depends on. *"Stiff versus compliant grasp on an egg"* is a demo, a blog post and a real result.
2. **Log `Present_Load` + `Present_Current`.** Measure the added loop cost first (§6). Bank it as offline labelling data and ServoGap ground truth, **not** as reactivity.
3. **Shrink the open-loop window.** The real unlock, and it is software. Smaller chunks, or real-time chunking. Going from 0.6 Hz to ~30 Hz effective replanning is what would give force-in-observation somewhere to land.
4. **Per-episode stiffness conditioning.** Collect episodes at several fixed P values, record P as a dataset field, train a policy conditioned on it. SoftMimic's idea within the EPROM constraint.
5. **ServoGap** — the learned actuator model. In sim there is no EPROM, so per-step stiffness is free. Train there, deploy at the nearest achievable fixed gain.
6. **PWM + microcontroller** (§8). Speculative, highest ceiling.

---

## Open questions

- 🔴 What rate does the STS3215's internal PID actually run at? Not published anywhere we have looked.
- 🔴 What are `DTs` and `Hts`?
- ⚠️ What is the real EPROM write endurance on this specific part? Our ~10⁵ figure is the class norm, not a datasheet reading for the STS3215.
- ⚠️ What does one extra `sync_read` cost on our bus? Measurable today with the existing debug log; nobody has measured it.
- ⚠️ π0's execution frequency. We cite 50-step chunks from the paper but have not verified the Hz.

## References

- ACT — chunk 1 → 100 gives 1% → 44%: https://arxiv.org/abs/2304.13705
- Diffusion Policy — 10 Hz command, interpolation controller: https://arxiv.org/abs/2303.04137
- Figure, Helix 02 — three layers, learned System 0: https://www.figure.ai/news/helix-02
- SoftMimic (MIT) — policy outputs joint-space position targets for a PD controller; **stiffness is an input at deployment**, one policy spans many stiffnesses: https://arxiv.org/abs/2510.17792
- DexWrist (MIT) — backdrivability without complex compliant controllers: https://arxiv.org/abs/2507.01008
- Hwangbo et al. 2019 — learned actuator model, learned control at control rate with proprioception, *Science Robotics* 4(26)
- Our own argument, written up: [A VLA is not the policy](https://physical-hardware-intelligence.github.io/blog/a-vla-is-not-the-policy.html)
- Source of truth for every measured number here: `lerobot/motors/feetech/tables.py`, `lerobot/motors/feetech/feetech.py`, `lerobot/robots/so_follower/so_follower.py`

[SockBot]: https://makerworld.com/en/models/2758393-lerobot-on-tracks

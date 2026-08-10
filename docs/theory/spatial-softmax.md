# SpatialSoftmax — the pivotal transformation

`lerobot/policies/diffusion/modeling_diffusion.py:402`, ~20 lines of code, and the single biggest commitment Diffusion Policy makes about what a robot needs to see. It compresses **921,600 numbers into 64** and decides that geometry matters and appearance does not.

This page is the deep dive. [diffusion-policy-shapes](diffusion-policy-shapes.md) traces where it sits in the pipeline; [diffusion-policy-why](diffusion-policy-why.md) is the argument for the surrounding architecture.

---

## 1. The trilemma it resolves

You have a `(512, 15, 20)` feature map and the policy needs a vector. There are three honest options:

| Option | Output size | Keeps | Destroys |
|---|---|---|---|
| Flatten | 153,600 | everything | nothing, but unusable — position is entangled with identity |
| Global average pool | 512 | *what* is present | **all position** |
| **Spatial softmax** | **64** | ***where* things are** | *what they look like* |

Average pooling is what image classifiers do, and it is **catastrophic for manipulation**. "There is a red cube somewhere in this image" tells an arm nothing. The whole task is *where*.

So spatial softmax is not a compression trick that happens to be small. It is a **deliberate bet that manipulation is a geometry problem.**

---

## 2. The deep idea: it swaps what is an index and what is a value

In an ordinary CNN feature map:

```
feature[k][i][j]  =  "how much of detector k is at position (i, j)"
  ^index  ^index         ^value
```

Channel `k` is an index (*which* feature), `(i, j)` is an index (*where*), the number is a value (*how much*).

Spatial softmax **promotes position from an index to a value**:

```
keypoint[k]  =  (x, y)  =  "where detector k is"
  ^index         ^value
```

`H` and `W` are not pooled or summarised. They are **consumed and re-emitted as the output itself.** The channel index survives as *which question*; the answer is a coordinate.

That is why the output is `(B, K, 2)` and not `(B, K)`. **The trailing axis of size 2 is not a feature dimension — it is a coordinate frame.** Once you see that, every other property follows.

---

## 3. The shapes, step by step

Take the docstring's example: a cuboid `(512, 10, 12)`. Think **512 sheets of paper, each 10×12** — not "a 3D tensor".

### 3a. Reshape — the cuboid becomes a rug of rows

```python
features = features.reshape(-1, self._in_h * self._in_w)   # (512, 10, 12) -> (512, 120)
```

`reshape` moves no data. It stops treating each sheet as 2D. Sheet *k* is read **row-major** — row 0 left to right, then row 1, then row 2 — and laid out as one line of 120 numbers, which becomes row *k*.

```
  cuboid (512, 10, 12)                 mat (512, 120)

  ┌──────────────┐ sheet 0      row   0 │ r0 │ r1 │ r2 │ … │ r9 │   ← 10 runs of 12
  ├──────────────┤ sheet 1      row   1 │ r0 │ r1 │ r2 │ … │ r9 │
  ├──────────────┤   ⋮            ⋮
  └──────────────┘ sheet 511    row 511 │ r0 │ r1 │ r2 │ … │ r9 │
```

### 3b. Softmax — one belief per channel

```python
attention = F.softmax(features, dim=-1)     # along each row, over the 120 cells
```

`dim=-1` is the spatial axis. **One softmax per channel**, each over its own 120 numbers, each row now summing to 1. Rows never look at each other — see the docstring trap in §7.

### 3c. `pos_grid` — a lookup table, not an image

Built once in `__init__`, constant, shared by all channels. `(120, 2)`, read as:

> **"Cell *i* is at x = `pos_grid[i][0]`, y = `pos_grid[i][1]`."**

It is not data. It is the ruler.

### 3d. The matmul — the shared axis is summed away

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

Written as sums:

```
keypoints[k][0] = Σᵢ  attention[k][i] · pos_grid[i][0]     ← E[x] for channel k
keypoints[k][1] = Σᵢ  attention[k][i] · pos_grid[i][1]     ← E[y] for channel k
```

The index `i` (over the 120 cells) appears on **both** inputs and on **neither** output. That is the definition of a contracted axis. Equivalently:

```python
torch.einsum('ki,id->kd', attention, pos_grid)
```

`i` is missing from the output → `i` is summed → **120 vanishes**. What remains is `k` (which channel) and `d` (which coordinate).

### 3e. Where the batch went

The real call is `(B, K, H, W)`, and `reshape(-1, H*W)` folds **both** B and K into rows:

```
(4, 32, 15, 20)  ->  (128, 300)  @  (300, 2)  ->  (128, 2)  ->  view(4, 32, 2)
```

**Batch and channel are both passengers.** The operation genuinely only acts on the spatial axis, which is why the whole thing is four lines.

---

## 4. 🚨 The invariant that makes it correct

Both flattenings must use the **same cell ordering**:

- `features.reshape(-1, H*W)` on a contiguous `(…, H, W)` tensor → index `h*W + w`
- `pos_x.reshape(H*W, 1)` on the `(H, W)` meshgrid output → index `h*W + w`

Both row-major, identically. If they ever diverge every keypoint is silently mirrored or transposed, **with no error raised**. Worth knowing before you touch this code.

> Related: line 443 uses `np.meshgrid` rather than `torch.linspace` because the latter "behaves slightly differently than numpy and causes a small degradation in pc_success of pre-trained models." Someone chased a real regression down to a float difference in a coordinate grid.

---

## 5. A 2×3 example you can check by hand

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

**Look at y.** The peak sits at `y = -1`, the answer is `-0.516`. The five background cells, 8% of the mass each, dragged the keypoint **half way to the image centre.**

Sharpen the same bump from 2 to 6:

```
softmax  -> [0.002, 0.988, 0.002, 0.002, 0.002, 0.002]
keypoint -> (+0.000, -0.985)
```

⇒ **Localisation accuracy is entirely a function of how peaked the distribution is.** There is no learned temperature in this port (robomimic has one; LeRobot hardcodes it to 1), so the network controls its own precision *implicitly*, by learning larger 1×1-conv weights. Bigger outputs → peakier softmax → tighter keypoint. **Confidence and precision are the same parameter here.**

---

## 6. Why *soft* and not `argmax`

**Differentiability.** `argmax` has zero gradient almost everywhere and is undefined at ties — you could never train through it. The expectation is smooth, and gradient reaches **all 300 cells** weighted by probability, not just the winner. Every cell is told "you should have been more or less confident."

**Sub-pixel precision, which is the part people miss.** Hard argmax returns one of 300 grid positions. Each cell covers a 32×32 pixel patch of a 640×480 image, so you would eat ±16 px of quantisation error. The weighted mean **lands between cells** — precision is set by peakedness, not by grid spacing. A 15×20 grid localises far finer than 32 pixels.

---

## 7. Two places the docstring will mislead you

**"applying channel-wise softmax"** reads like softmax *across* the 512 channels. It is not — `dim=-1` is space. "Channel-wise" means *done separately per channel*. The consequence matters: **channels never compete.** All 32 keypoints fire on every frame, independently, and each always returns a coordinate. There is no winner-take-all across features, and no way to say "not me".

**"a learnable linear mapping (in_channels, H, W) -> (num_kp, H, W)"** is true but hides the structure. It is `nn.Conv2d(512, 32, kernel_size=1)` — a **per-cell** linear map across channels, shared across all 300 positions, touching zero neighbours. Not a dense layer over the flattened map. It asks one question everywhere: *of my 512 detectors, which 32 combinations are worth locating?* Being learned, it can synthesise detectors the backbone lacks — "reddish **and** roundish" as a single keypoint.

---

## 8. Three ways it lies to you

**The dumbbell.** If a channel fires on two objects, its expected position is the midpoint — where there is nothing. The centre of mass of a dumbbell is in empty air. This is exactly the averaging pathology that motivates [diffusion over regression](diffusion-policy-why.md), showing up one layer earlier in the same network.

**Silence looks like the centre.** Flat activations — object absent, occluded, out of frame — give a uniform distribution, whose expectation over a `[-1, 1]` grid is exactly **(0, 0)**. *"I see nothing"* and *"it is dead centre"* produce an identical output. There is no absence signal. §5 shows this is a continuum, not a special case: background mass always pulls toward centre.

**Magnitude is discarded.** Softmax normalises, so a faint smudge and a blazing detection give the same confident coordinate. **The keypoint carries no confidence.** The policy cannot tell "the cube is definitely here" from "I am guessing."

---

## 9. What this costs on Φ data

Keypoints encode *where*, not *what*. Our `phi_so101_cubes_cylinder_v1` has three objects — 25 mm red cube, 45 mm white cube, yellow cylinder. A channel tuned to "cube-shaped thing" returns the same coordinate regardless of which cube it sees. Object identity has to be encoded **indirectly, as a pattern across the 32 channels** (colour-tuned channels drifting to centre for the wrong object; size inferred from the spread between channels). Possible, but a far weaker channel than ACT's 300 full-dimensional tokens per camera, which carry appearance outright.

The dumbbell risk is *low* for us by design — we deliberately stage one object and one container at a time. The identity risk is not.

⇒ **Prediction: on this architecture expect object-identity confusion before trajectory failure.**

---

## 10. Reproduce it

```bash
cd /Volumes/Crucial_X9/Projects/phi && python -c "
import torch, numpy as np, torch.nn.functional as F
H,W=2,3
px,py=np.meshgrid(np.linspace(-1,1,W),np.linspace(-1,1,H))
pos=torch.tensor(np.stack([px.reshape(-1),py.reshape(-1)],1),dtype=torch.float)
for peak in (2.,6.,20.):
    a=torch.tensor([[0.,peak,0.],[0.,0.,0.]])
    p=F.softmax(a.reshape(1,H*W),dim=-1)
    print(f'peak={peak:5} -> keypoint {tuple(round(v,3) for v in (p@pos)[0].tolist())}')
"
```

---

## The sentence to keep

> The feature map becomes a rug of rows, one row per channel. Softmax turns each row into a belief over cells. The coordinate table says where each cell is. The matmul is a weighted average that consumes the cell axis and leaves one (x, y) per row.

And the companion, for the failure mode:

> The centre of mass of a dumbbell is in the empty middle.

---

## See also

- [diffusion-policy-shapes](diffusion-policy-shapes.md) — where this sits in the full tensor trace
- [diffusion-policy-why](diffusion-policy-why.md) — why the rest of the architecture is shaped the way it is
- [act-shapes](act-shapes.md) — the contrast: 300 tokens × 512 dims per camera, appearance retained

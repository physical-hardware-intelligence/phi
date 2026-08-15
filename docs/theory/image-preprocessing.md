# Making a picture smaller, without lying about it

Every policy here starts by shrinking a 480×640 camera frame. That step looks trivial and isn't. This page explains it from the ground up — every term defined before it is used — and records what we measured on our own frames.

> **Who should read this:** anyone adding a new vision backbone, changing `resize_shape`, or wondering why a camera behaves differently from the others.

---

## 1. The problem

A camera gives 480×640 = 307,200 pixels. A backbone wants fewer. Four out of every five pixels have to go. The only question is **how you decide what the survivors say.**

Two methods sound equally reasonable. One of them lies.

## 2. Pick, or average

A one-line picture, 12 pixels. `1` = white, `0` = black. Shrink to 6.

**Fine stripes, one pixel each:**

```
original             [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
PICK every 2nd       [1, 1, 1, 1, 1, 1]          "it's solid white"
AVERAGE each pair    [.5, .5, .5, .5, .5, .5]    "it's uniform grey"
```

Six pixels cannot show twelve alternating bands — the stripes cannot survive either way. What matters is *how each method fails*. Averaging says grey: *"too fine for me, here is the true average brightness."* Picking says solid white, which is not a blurry version of the truth — it is a confident report of something that was never there.

**Nudge the camera one pixel:**

```
original             [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
PICK every 2nd       [0, 0, 0, 0, 0, 0]          "it's solid BLACK"
AVERAGE each pair    [.5, .5, .5, .5, .5, .5]    unchanged
```

Same scene, one pixel of shake, and picking flips from all-white to all-black.

**The dangerous case — stripes 3 px wide, shrink by 2:**

```
original             [1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0]
PICK every 2nd       [1, 0, 1, 1, 0, 1]      stripes of a DIFFERENT width
AVERAGE each pair    [1, .5, .5, 1, .5, .5]
```

Picking did not lose the pattern. It **invented a new one** — different spacing, entirely fictitious, and perfectly plausible-looking.

> 🔑 **That invented pattern is what "aliasing" means.** Detail too fine to represent does not fade out politely; it disguises itself as coarser detail that is wrong. Everything else on this page is bookkeeping.

## 3. How fine is "too fine"

To show a stripe you need at least **two pixels** — one for the light band, one for the dark. One pixel cannot be both.

That threshold has a formal name (Nyquist). The name adds nothing. It is "you need two pixels per stripe."

**Worked for our rig, 480 → 210:**

```
480 old rows into 210 new slots  ->  each slot covers 480/210 = 2.2857 old rows

   new row      covers old rows
         0       0.000 to  2.286
         1       2.286 to  4.571
       209     477.714 to 480.000
   check: 210 × 2.2857 = 480. Every old row accounted for, none twice.

a stripe needs 2 NEW pixels
2 × 2.2857 = 4.57 OLD pixels
=> anything in the original finer than 4.57 px has nowhere to live
```

Sanity check with a round number: shrinking 480 → 240 is a factor of exactly 2, so a 2-new-pixel stripe is 4 old pixels. Same logic, easier arithmetic.

## 4. How much of *our* pictures is too fine

Measured on real frames from episode 0 — the share of image detail finer than the 4.57 px threshold:

```
  wrist    0.79%
  front    0.94%
  top      3.75%     <- 4x more than the others
```

The top camera looks down at a textured table, so it carries the most fine grain. **Prediction: if aliasing hurts anywhere, it hurts the top camera most.** (It does — §6.)

## 5. What the three methods actually do

| method | what it does | aliases? |
|---|---|---|
| `bilinear` (default) | blends the **4 nearest** source pixels, always, regardless of shrink factor | **yes** |
| `bilinear, antialias=True` | blurs away too-fine detail first, then shrinks | no |
| `area` | each output pixel = honest average of exactly the source pixels it covers | no |

At a 2.29× shrink each output pixel should summarise ~5.2 source pixels. Plain bilinear looks at 4 — it is mostly picking with a light smear.

```
max |F.interpolate(mode='area') − adaptive_avg_pool2d| = 0.000e+00
```

> 🔑 **`mode='area'` is not an approximation of the ideal — it IS the ideal.** It is exactly the image a camera built at that resolution would record. Use it for downscaling. (It is only correct downward; for upsampling it degenerates.)

## 6. What it costs, measured on real frames

Using `area` as the reference (any deviation from it *is* resampling error, nothing else differs), and using "how far apart are two moments 4 seconds apart, with the robot moving" as the yardstick:

```
camera  method                    error, as % of 4 seconds of real motion
wrist   bilinear                                35.7%
wrist   antialias=True                          33.7%
front   bilinear                                45.4%
front   antialias=True                          32.4%
top     bilinear                               112.7%   <-
top     antialias=True                          53.1%
```

> 🚨 **On the top camera, plain bilinear injects more fake change than four seconds of real robot motion produces.** The artefact is larger than the signal.

The §4 spectrum predicted exactly this: top had 4× the fine detail and took ~2× the damage. Two independent measurements agreeing is why we trust it.

## 7. 🚨 Which of OUR pipelines actually alias

**This corrected an earlier claim in this repo that "everything aliases." It does not.**

```
model                        spatial op                          aliased?
ACT (ours)                   none — raw 480×640 to ResNet        NO
DP-CNN / DP-T (ours)         tv.Resize(240,320) antialias=True   NO
SmolVLA                      F.interpolate to 512×512, no aa     YES
```

- **ACT** has zero hits for `resize`, `Resize`, `crop`, `interpolate`. It feeds the raw frame to the backbone. There is no resampling step to get wrong.
- **DP-CNN / DP-T** use `torchvision.transforms.Resize`, which since torchvision 0.17 defaults to `antialias=True`. Verified bit-identical to `F.interpolate(antialias=True)`; `maxdiff 0.00e+00`.
- **SmolVLA**'s hand-rolled `resize_with_pad` calls `F.interpolate(... mode="bilinear", align_corners=False)` with **no `antialias` argument**, and `F.interpolate`'s default is `False`. Confirmed aliased.

> **Lesson recorded:** the original claim generalised from SmolVLA's file to "lerobot" as a whole without reading the other configs. Before asserting a pipeline has a property, read *that* pipeline's config, not a sibling's.

## 8. Choosing a target size: aspect ratio for free

If the target's aspect ratio matches the source, there is no padding and no distortion. For a 3:4 camera and a patch-14 ViT:

```
H = 42m, W = 56m  ->  grid 3m × 4m  ->  12m² patches

  m         size      grid   patches/cam
  4      168×224     12×16           192
  5      210×280     15×20           300     <- our choice
  6      252×336     18×24           432
  8      336×448     24×32           768
```

**210×280 verified:** `210/14 = 15.0`, `280/14 = 20.0`, and `480/210 = 640/280 = 2.2857` exactly. No padding, no crop, no stretch — every one of the 300 patches carries signal.

Compare SmolVLA's 512×512 on the same camera: **25% of every frame is dead `−1` padding**, wasting 16 of its 64 tokens per camera.

## 9. Non-square input to DINOv2: verified safe

DINOv2 ViT-S/14 has `pos_embed` of shape `(1, 1370, 384)` = 1 cls + **37×37**, pretrained on square 518×518. A 15×20 grid has no learned position tags, so `interpolate_pos_encoding` bicubic-resizes the 37×37 tag grid down to (20, 15) and asserts the result matches.

"Does not crash" is not "is not degraded", so we checked feature health directly:

```
                        patches   ‖f‖    mean pairwise cos   rank
  210×280 (ours)            300  45.52          0.378        300/300
  224×224 squashed          256  45.16          0.381        256/256
  518×518 native square    1369  46.46          0.363        383/384
```

Degeneracy would show as collapsed norms, pairwise cosine near 1.0, or rank deficiency. **None appear; 300/300 full rank.** Non-square is safe.

Related flags: `interpolate_offset = 0.1`, `interpolate_antialias = False`.

Also corrected here: **DINOv2 ViT-S/14 is 22,056,576 parameters**, not the 21,629,184 an earlier architectural estimate gave.

---

## The rule

```python
# downscaling, always
img = F.interpolate(img, size=(210, 280), mode="area")
```

Pick a target whose aspect ratio matches the source and whose sides divide by the patch size. Then you need no padding, no crop, and no compromise.

**Known imperfection:** 480/210 = 2.2857 is not an integer, so `area` uses floor/ceil cell boundaries — some output pixels average a 2×2 source block, others 3×3. It is a true area average with mildly non-uniform blur. Small next to the ~3× improvement over plain bilinear, but not literally perfect.

**Not measured:** whether any of this changes task success. We measured picture fidelity, an intermediate quantity. The link from "the top camera's features are wrong by more than four seconds of motion" to "the gripper misses the cube" is a plausible mechanism, not a demonstrated one.

## See also

- [smolvla §5](smolvla.md#5-resize_with_pad-and-the-25-of-every-frame-that-is-dead) — the padding waste, measured
- [diffusion-policy-shapes](diffusion-policy-shapes.md) — where DP's resize and crop sit in the chain
- [act-why](act-why.md) — why ACT skips resizing entirely

# phi_so101_cubes_cylinder_recovery_v1

**HF id**: [`BrutalCaesar/phi_so101_cubes_cylinder_recovery_v1`](https://huggingface.co/datasets/BrutalCaesar/phi_so101_cubes_cylinder_recovery_v1) (public, `v3.0` tag present)
**Robot**: SO-ARM101 leader–follower (`so_follower`) · **Task string**: `"pick up the cubes/cylinder and place it in the box"` (one string — see [Why one task string](#why-one-task-string-and-not-per-episode-labels))
**Derived from**: [phi_so101_cubes_cylinder_v1](phi_so101_cubes_cylinder_v1.md) — the original 120 episodes are **byte-identical** and unshifted
**Collected**: recovery episodes 2026-08-10 · **Operator**: Yash
**Local root**: `~/.cache/huggingface/lerobot/BrutalCaesar/phi_so101_cubes_cylinder_recovery_v1`

| | |
|---|---|
| Episodes | **143** (120 clean + 23 recovery) |
| Frames | **81,943** (45.5 min) |
| fps | 30 |
| Cameras | 3 × 640×480 (`wrist`, `front`, `top`) |
| Size | 1.29 GB |

`verify_dataset` ✅ all checks pass: every referenced video shard present for all three cameras, **zero dropped frames**, metadata agrees with data.

Camera keys are **correct** (no transposition) — inherited unchanged from the parent dataset.

---

## What "recovery" means here, and what it is not

Each recovery episode is a **failure followed by a successful human correction** — the arm misses or drops, then recovers and completes the placement.

> 🚨 **This is NOT a set of negative examples.** ACT's loss is L1 on demonstrated actions; the gradient always pulls the prediction *toward* the data, and there is no sign available for "avoid this." Appending an episode that **ends** in failure would make the policy **more** likely to reproduce that failure.
>
> What this data actually provides is **positive supervision at off-distribution states** — the DAgger/RaC mechanism. The value is not "learning what not to do," it is "learning how to get back to the expert manifold from states a clean demonstrator never visits."
>
> ⇒ **Any episode that ends in failure with no completed recovery must be excluded, not merely down-weighted.** See [the three at-cap episodes](#-three-episodes-ended-at-the-recording-time-cap) below, which are unresolved on exactly this point.

## Episode map

Original 120 clean episodes at indices **0–119**, unchanged from the parent dataset (all deletions were above index 119, so nothing shifted).

### Recovery episodes — 23 at indices 120–142

| Object | Container | Episodes | n |
|---|---|---|---:|
| yellow cylinder | white bin | `120–124` | **5** ⚠️ |
| yellow cylinder | cardboard box | `125–130` | 6 |
| white cube | white bin | `131–136` | 6 |
| white cube | cardboard box | `137–142` | 6 |

⚠️ **The first block has 5, not 6.** 25 recovery episodes were recorded (originally 120–144); two were deleted (below). The design intent was 6/6/6/6 = 24. **This asymmetry is deliberate and accepted** — do not "fix" it by rebalancing, and account for it in any per-condition success rate, since that block has 17% fewer samples than the others.

Recovery episodes are **longer** than clean ones — mean 21.8 s vs 18.6 s — which is the expected signature of failure + recovery.

## Deletions

Two episodes removed with `lerobot-edit-dataset --operation.type=delete_episodes`, which **reindexes**:

| Original index | Length | Reason |
|---|---|---|
| `123` | 248 frames (8.3 s) | far below the 21.4 s recovery median; accepted as unusable, leaving block 1 at 5 |
| `144` | 261 frames (8.7 s) | the 25th episode — a stray extra take beyond the intended 24 |

Reindexing verified by matching episode lengths against `old − {123, 144}` in order: **exact match**. Frames removed = 509 = 248 + 261 ✅

The **145-episode pre-deletion state still exists locally** at `~/.cache/huggingface/lerobot/BrutalCaesar/phi_so101_cubes_cylinder_v1_20260806_161432` and is the only copy of those two episodes. Do not clear that directory until you are certain they are not wanted.

## Three episodes ended at the recording time cap — ✅ resolved, keep them

Episodes **128, 134, 140** (new indexing) are exactly **750 frames = 25.000 s** — the `episode_time_s` cap, not a natural end. Those three plus the deleted `123` sat at positions **3, 9, 15, 21** of the recording sequence: the **4th take of every block**, spaced exactly 6 apart, which is a periodic grid rather than four independent accidents.

**Resolved 2026-08-10 (Yash, from the video): all three completed their recovery.** They ran long, they did not truncate mid-recovery, so they end on the expert manifold like every other recovery episode. **All 23 are in training; nothing is excluded.**

## Why one task string, and not per-episode labels

Per-episode object/container/recovery labels would be more useful metadata — but they are **incompatible with the required eval split**, and the requirement wins.

`make_train_eval_datasets` (`lerobot/datasets/factory.py`) splits **per task string**:

```python
for eps in task_to_episodes.values():
    n_eval = math.ceil(len(eps) * cfg.dataset.eval_split)
    train_episodes.extend(eps[: len(eps) - n_eval])
    eval_episodes.extend(eps[len(eps) - n_eval :])
```

`math.ceil` of any positive number is ≥ 1, so **giving the recovery episodes their own task string guarantees at least one of them lands in eval** — and the design calls for all 23 in training, holdout drawn from clean episodes only.

Keeping one task string also keeps the holdout **identical to the `cvae_3cam` control run**, which is what makes the two directly comparable.

⇒ The labels live in this card. Add them as a separate `..._recovery_lang_v1` variant when a VLA run actually needs language conditioning, the same way [`phi_so101_cubes_cylinder_lang_v1`](phi_so101_cubes_cylinder_v1.md) relates to its parent.

## Training split

All 23 recovery episodes → **training**. Holdout = 30 episodes drawn only from the original clean blocks (5 from each of the 6), unchanged from the parent experiment:

```
0–4 · 20–24 · 45–49 · 65–69 · 90–94 · 110–114
```

All are < 123, so the deletions did not shift them.

> 🚨 **`eval_split = 0.209`, and the value is not free to round.** It must satisfy `ceil(143 × s) == 30`:
> ```
> 0.202797 < s <= 0.209790
>   s = 0.209   -> 30   OK
>   s = 0.2098  -> 31   WRONG   (one extra episode silently held out)
>   s = 0.21    -> 31   WRONG
> ```
> Train = 113 episodes, of which 23 are recovery.

**What this split can and cannot tell you.** `eval_loss` on a clean holdout answers *"did adding recovery data damage clean performance?"* It **cannot** measure whether recovery ability improved — no loss can, because the held-out episodes contain no induced failures. That requires a rollout where you deliberately displace or strip the object mid-episode and check whether the policy retries. Plan that protocol separately.

## Related

- [phi_so101_cubes_cylinder_v1](phi_so101_cubes_cylinder_v1.md) — the 120-episode parent
- [experiments/2026-08-06_act-cubes-cylinder-splits.md](../experiments/2026-08-06_act-cubes-cylinder-splits.md) — the split design this inherits
- [docs/theory/diffusion-policy-why.md](../docs/theory/diffusion-policy-why.md) §1–2 — why an L1 objective cannot represent "don't"

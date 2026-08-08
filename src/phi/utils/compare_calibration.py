"""Diff two SO-101 calibration files and say, in degrees, how far apart they are.

    python -m phi.utils.compare_calibration phi_follower parv
    python -m phi.utils.compare_calibration ~/mine.json /path/to/theirs.json --explain

WHY THIS EXISTS
---------------
A policy is trained on ONE person's calibration and then run on someone else's
laptop. Nothing errors. The arm moves smoothly. It just reaches to slightly the
wrong place, and the only symptom is a success rate nobody measured. This turns
that invisible failure into a number you can read in five seconds.

Pure stdlib on purpose — no torch, no lerobot, no serial port, no robot. It runs
in ~50 ms on any machine, including one that has never had the env installed, so
there is no excuse not to run it before a handoff.

WHAT A CALIBRATION FILE ACTUALLY IS
-----------------------------------
Each joint is a Feetech STS3215 servo with a 12-bit encoder: one full turn is
split into 4096 numbered positions ("ticks"). One tick = 360/4095 = 0.0879 deg.
Where tick 0 physically points is arbitrary — it depends on how the horn happened
to be pressed onto the output shaft at assembly. Nobody chose it.

    homing_offset       Feetech firmware: Present_Position = Actual_Position - Homing_Offset
                        `set_half_turn_homings()` picks it so that whatever pose you were
                        holding at the "move to the middle of its range" prompt reads 2047.
                        It is a record of YOUR POSE, not a property of the arm.

    range_min/max       the two physical hard stops, recorded AFTER homing, by
                        `record_ranges_of_motion()` while you sweep the joint by hand.

LeRobot then feeds the policy (`use_degrees=True` is the default in SOFollowerConfig,
so joints 1-5 are MotorNormMode.DEGREES and the gripper is RANGE_0_100):

    joints 1-5   degrees = (Present_Position - mid) * 360/4095,  mid = (range_min+range_max)/2
    gripper      percent = (Present_Position - range_min) / (range_max - range_min) * 100

THE SELF-CANCELLING PROPERTY (why any of this works at all)
-----------------------------------------------------------
Substitute Present_Position = Actual - homing into the degrees formula. `mid` was
measured in the same homed frame, so it carries the same homing term, and the two
cancel exactly:

    degrees = (Actual - mid_absolute) * 360/4095      mid_absolute = midpoint of the two
                                                       PHYSICAL hard stops

The homing pose does not survive the algebra. What is left is "how far am I from
the centre of my mechanical range" — a fact about plastic and metal, not about the
person. Two people who both push each joint into its stops get IDENTICAL degrees.
That is a good design, and it is why cross-machine deployment mostly works.

It has exactly one hole, and this tool exists to find it:

  🚨 A FULL-TURN JOINT HAS NO STOPS TO ANCHOR TO. `wrist_roll` is excluded from
     `record_ranges_of_motion()` and its range is HARDCODED to [0, 4095] in
     so_follower.py. So `mid` is 2047.5 for everybody, there is nothing for the
     homing term to cancel against, and the calibration pose survives in full.
     Whatever wrist angle you were holding at the ENTER prompt becomes permanent
     law. Two people can differ by tens of degrees here while every other joint
     agrees to a fraction of one.

  🚨 THE GRIPPER HAS GAIN AS WELL AS OFFSET. RANGE_0_100 divides by your measured
     span, so a different span rescales every command. It is anchored (the jaws do
     have stops) but it is the only joint where the error is not a pure shift.

HOW THE COMPARISON WORKS
------------------------
Two calibrations are only comparable in the arm's own absolute encoder frame:

    Actual = Present + homing_offset          (mod 4096)

So `range_min + homing_offset` is the physical low stop in absolute ticks, and
that number is a property of the HARDWARE. Which gives a free identity test:

  * stops agree within a few ticks   -> same physical arm (or identically indexed
                                        builds), and every offset below is real.
  * stops disagree by hundreds       -> different arms. Horns sit on different
                                        spline teeth. The offsets below are
                                        MEANINGLESS and the tool says so.

The measured joints are the control group. If they agree and one joint does not,
that joint has a calibration problem, not a hardware difference.

NOT COVERED (deliberately)
--------------------------
Camera framing, arm placement on the table, and lighting are at least as likely to
break a handoff as calibration, and none of them are in this file. Use
`camera_realign.py` for the cameras. This tool answers exactly one question.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from pathlib import Path

# 12-bit encoder: 4096 discrete positions per revolution. lerobot's DEGREES mode
# divides by `model_resolution_table[model] - 1`, so 4095, not 4096.
RESOLUTION = 4096
DEG_PER_TICK = 360.0 / (RESOLUTION - 1)

CAL_ROOT = Path(
    os.environ.get("HF_LEROBOT_HOME", str(Path.home() / ".cache" / "huggingface" / "lerobot"))
) / "calibration"

# Joint order as the policy sees it (observation.state / action vector order).
ORDER = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

# Does an angular error on this joint MOVE the gripper through space?
#   shoulder_pan swings the entire arm, so its lever really is the full reach.
#   The others pivot partway down the chain — the reach is an upper bound.
#   wrist_roll spins the jaws about their own axis: it changes the ANGLE OF
#   ATTACK and nothing else. Converting it to millimetres would be a lie, and a
#   flattering one, since a roll error is not something the policy can servo away.
TRANSLATES = {"shoulder_pan": "exact", "shoulder_lift": "bound",
              "elbow_flex": "bound", "wrist_flex": "bound", "wrist_roll": None}

# How far apart two calibrations of the SAME arm may sit before it matters.
# 0.5 deg at a 250 mm reach is ~2 mm — below the repeatability of the arm itself.
# 2.0 deg is ~9 mm, which is a third of the 25 mm cube: that is where it starts
# deciding grasps. Above 10 deg the approach geometry is simply wrong.
FINE, NOTABLE, SEVERE = 0.5, 2.0, 10.0

# Median disagreement between two files' physical hard stops, in ticks. Repeatedly
# pushing the same 3D-printed stop varies by a few ticks; a different arm's horn
# sits on a different spline tooth, which is ~164 ticks (25-tooth spline) or more.
# 25 leaves generous room for the former without admitting the latter.
SAME_ARM_TICKS = 25


def signed_circular(a: float, b: float) -> float:
    """a - b on a 4096-tick ring, returned in (-2048, +2048].

    Encoder positions wrap: tick 4095 and tick 0 are the same physical place, one
    tick apart. A plain subtraction would report 4095. Every difference in this
    file goes through here.
    """
    return ((a - b + RESOLUTION / 2) % RESOLUTION) - RESOLUTION / 2


def is_full_turn(joint: dict) -> bool:
    """True for a joint whose range was hardcoded rather than measured.

    so_follower.py sets range_min=0 / range_max=4095 for `wrist_roll` instead of
    recording it. That is the signature, and it is what removes the anchor.
    """
    return joint["range_min"] == 0 and joint["range_max"] == RESOLUTION - 1


def absolute(joint: dict) -> tuple[float, float, int, float]:
    """Re-express one joint in the arm's own absolute encoder frame.

    Returns (low_stop, high_stop, span, mid) — the first, second and fourth in
    absolute ticks, which is the only frame in which two files can be compared.
    """
    h = joint["homing_offset"]
    lo = (joint["range_min"] + h) % RESOLUTION
    hi = (joint["range_max"] + h) % RESOLUTION
    span = joint["range_max"] - joint["range_min"]   # homing-free: it subtracts out
    # Walk UP from the low stop by half the span. Doing it this way (rather than
    # averaging lo and hi) stays correct when the range crosses the 4095/0 seam,
    # which it does on four of our six joints.
    mid = (lo + span / 2) % RESOLUTION
    return lo, hi, span, mid


def load(spec: str) -> tuple[Path, dict]:
    """Accept a path, or a bare calibration id to look up in the LeRobot cache."""
    p = Path(spec).expanduser()
    if p.suffix == ".json":
        if not p.exists():
            sys.exit(f"no such file: {p}")
    else:
        hits = sorted(CAL_ROOT.glob(f"*/*/{spec}.json"))
        if not hits:
            avail = sorted({f.stem for f in CAL_ROOT.glob("*/*/*.json")})
            sys.exit(f"no calibration named '{spec}' under {CAL_ROOT}\n"
                     f"available: {', '.join(avail) or '(none)'}")
        # A leader mis-calibrated with --robot.type lands in robots/ too, so the
        # same id can exist twice. The follower is what a policy runs on; prefer it.
        robots = [h for h in hits if "/robots/" in h.as_posix()]
        if len(hits) > 1:
            print(f"note: '{spec}' matches {len(hits)} files; using the follower one",
                  file=sys.stderr)
        p = (robots or hits)[0]

    data = json.load(open(p))
    missing = [j for j in ORDER if j not in data]
    if missing:
        sys.exit(f"{p} is missing joints {missing} — is this an SO-101 calibration?")
    return p, data


CALIBRATION_PROTOCOL = """
  RECALIBRATION PROTOCOL (the two steps that produce every offset above)

    a) At "Move to the middle of its range of motion and press ENTER", set
       wrist_roll to a REPEATABLE PHYSICAL LANDMARK — jaws level and untwisted,
       parallel to the table edge. This one pose is the entire calibration of
       that joint: its range is hardcoded, so nothing else constrains it, and
       every degree you are off here is a degree the policy is off forever.
       (It also sets your ±180° of usable travel, so an off-centre wrist here
       spends teleop headroom too — see troubleshooting.md.)

    b) Push every OTHER joint firmly into BOTH hard stops and hold a beat at
       each. Stop short on one side only and the midpoint between the stops
       moves by half your shortfall, which lands directly on the policy.
"""


def effect_at_gripper(joint: str, deg: float, reach: float) -> str:
    """Turn a joint-angle error into the thing you would actually see go wrong."""
    if TRANSLATES.get(joint) is None:
        return f"{abs(deg):.0f}° roll"
    mm = reach * math.sin(math.radians(abs(deg)))
    return f"{mm:.1f} mm"


def verdict(deg: float) -> str:
    a = abs(deg)
    if a < FINE:
        return "ok"
    if a < NOTABLE:
        return "minor"
    if a < SEVERE:
        return "SIGNIFICANT"
    return "SEVERE"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Diff two SO-101 calibrations and report the disagreement in degrees.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Names are looked up under " + str(CAL_ROOT) + "; paths ending in .json are used as-is.",
    )
    ap.add_argument("a", help="calibration id or path — the one the POLICY WAS TRAINED WITH")
    ap.add_argument("b", help="calibration id or path — the one it will RUN ON")
    ap.add_argument("--reach", type=float, default=250.0,
                    help="horizontal distance from the base to the gripper, mm, for converting "
                         "degrees into millimetres of error (default 250)")
    ap.add_argument("--explain", action="store_true",
                    help="print the full derivation of what these numbers mean")
    args = ap.parse_args(argv)

    if args.explain:
        print(__doc__)
        print("=" * 78 + "\n")

    pa, A = load(args.a)
    pb, B = load(args.b)
    label_a, label_b = pa.stem, pb.stem
    if label_a == label_b:
        label_a, label_b = label_a + " (A)", label_b + " (B)"

    print(f"A  {label_a:<24s} {pa}")
    print(f"B  {label_b:<24s} {pb}")
    print(f"\n1 tick = {DEG_PER_TICK:.4f} deg   ·   end-effector reach assumed {args.reach:.0f} mm\n")

    # ── 1. Is this even the same arm? ────────────────────────────────────────
    # Absolute stop positions are hardware. If they match, everything downstream
    # is meaningful; if they do not, nothing downstream is.
    print("=" * 78)
    print("1. SAME ARM?   physical hard stops in absolute encoder ticks")
    print("=" * 78)
    print(f"{'joint':14s} {'stop':>6s} {label_a[:9]:>10s} {label_b[:9]:>10s} {'diff':>7s} {'deg':>7s}")
    print("-" * 78)

    stop_diffs: list[float] = []
    for j in ORDER:
        if is_full_turn(A[j]) or is_full_turn(B[j]):
            print(f"{j:14s} {'--':>6s} {'':>10s} {'':>10s}      "
                  f"(full-turn joint: range hardcoded, no stops to compare)")
            continue
        la, ha, _, _ = absolute(A[j])
        lb, hb, _, _ = absolute(B[j])
        for name, x, y in (("low", la, lb), ("high", ha, hb)):
            d = signed_circular(x, y)
            stop_diffs.append(abs(d))
            print(f"{j if name == 'low' else '':14s} {name:>6s} {x:10.0f} {y:10.0f} "
                  f"{d:7.0f} {d * DEG_PER_TICK:7.2f}")

    med = statistics.median(stop_diffs) if stop_diffs else float("nan")
    same_arm = med <= SAME_ARM_TICKS
    print(f"\n  median disagreement: {med:.0f} ticks ({med * DEG_PER_TICK:.2f} deg)")
    if same_arm:
        print(f"  -> SAME PHYSICAL ARM (or identically indexed builds). Under {SAME_ARM_TICKS} ticks is")
        print("     just how hard each person leaned on a 3D-printed stop. Section 2 is meaningful.")
    else:
        print("  -> ⚠️  DIFFERENT ARMS. Each servo horn sits on its own spline tooth, so these two")
        print("     files describe different absolute frames and SECTION 2 IS MEANINGLESS —")
        print("     do not read offsets from it, and do NOT copy one file onto the other machine.")
        print("     Recalibrate on the target arm instead (see the protocol at the end).")

    # ── 2. The offsets that actually reach the policy ────────────────────────
    print("\n" + "=" * 78)
    print("2. FRAME OFFSET   where each file thinks 'zero degrees' is")
    print("=" * 78)
    if not same_arm:
        print("   (suppressed — different arms, see above)\n")
        worst: list[tuple[str, float]] = []
    else:
        print(f"{'joint':14s} {'mid A':>9s} {'mid B':>9s} {'ticks':>7s} {'DEGREES':>9s} "
              f"{'at gripper':>12s}  verdict")
        print("-" * 78)
        worst = []
        for j in ORDER:
            if j == "gripper":
                continue                       # different formula, handled in §3
            _, _, _, ma = absolute(A[j])
            _, _, _, mb = absolute(B[j])
            d = signed_circular(ma, mb)
            deg = d * DEG_PER_TICK
            effect = effect_at_gripper(j, deg, args.reach)
            flag = "  🚨 full-turn: unanchored" if is_full_turn(A[j]) else ""
            print(f"{j:14s} {ma:9.1f} {mb:9.1f} {d:7.1f} {deg:9.2f} {effect:>12s}  "
                  f"{verdict(deg)}{flag}")
            worst.append((j, deg))

        print("\n  Sign: B reads this many degrees HIGHER than A at the same physical pose, so a")
        print("  command from a policy trained on A lands that far off when executed on B.")
        print(f"  'at gripper' converts the angle into displacement at a {args.reach:.0f} mm reach —")
        print("  exact for shoulder_pan (it swings the whole arm), an upper bound for the joints")
        print("  further down the chain. wrist_roll is reported as ROLL because it spins the jaws")
        print("  about their own axis: it changes the angle of attack without moving the gripper.")

    # ── 3. The gripper, which is scaled, not shifted ─────────────────────────
    print("\n" + "=" * 78)
    print("3. GRIPPER   RANGE_0_100: gain AND offset, so check it across the whole travel")
    print("=" * 78)
    ga, gb = A["gripper"], B["gripper"]
    a0, a1, aspan, _ = absolute(ga)
    b0, b1, bspan, _ = absolute(gb)
    print(f"  {label_a:<22s} closed {a0:6.0f}   open {a1:6.0f}   span {aspan:5d}")
    print(f"  {label_b:<22s} closed {b0:6.0f}   open {b1:6.0f}   span {bspan:5d}")
    print(f"  span differs by {bspan - aspan:+d} ticks ({(bspan - aspan) / aspan * 100:+.1f}% gain)\n")
    print(f"  {'command':>9s} {'A -> tick':>11s} {'B -> tick':>11s} {'error':>9s}")
    grip_err = 0.0
    for n in (0, 25, 50, 75, 100):
        ta = n / 100 * aspan + a0
        tb = n / 100 * bspan + b0
        e = signed_circular(tb, ta)
        grip_err = max(grip_err, abs(e))
        print(f"  {n:9d} {ta:11.1f} {tb:11.1f} {e:+9.1f}")
    print(f"\n  worst {grip_err:.0f} ticks = {grip_err * DEG_PER_TICK:.2f} deg of jaw travel — "
          f"{'negligible' if grip_err < 25 else '⚠️  enough to change whether a grasp closes'}")

    # ── 4. What to do about it ───────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("4. VERDICT")
    print("=" * 78)
    bad = sorted([w for w in worst if abs(w[1]) >= NOTABLE], key=lambda x: -abs(x[1]))
    if not same_arm:
        print("  Different arms — the offsets could not be computed, because there is no shared")
        print("  frame to measure them in. What you CAN still do:")
        print("    * compare the spans in section 1: they ARE comparable across arms, and if they")
        print("      differ by more than ~40 ticks somebody did not reach both hard stops")
        print("    * recalibrate on the target arm, holding to the protocol below")
        print(CALIBRATION_PROTOCOL)
    elif not bad and grip_err < 25:
        print("  ✅ These two calibrations agree. Every joint is inside "
              f"{NOTABLE:.1f} deg and the gripper")
        print("     scaling is negligible. Calibration is NOT your problem — look at camera")
        print("     framing and object/table placement instead (camera_realign.py).")
    else:
        print("  ⚠️  These calibrations disagree enough to change where the arm reaches:\n")
        for j, deg in bad:
            effect = effect_at_gripper(j, deg, args.reach)
            print(f"     {j:14s} {deg:+7.2f} deg  ({effect} at the gripper)  {verdict(deg)}")
            if is_full_turn(A[j]):
                print("                    cause: full-turn joint, range hardcoded [0,4095], so the")
                print("                    calibration POSE never cancels. Whoever pressed ENTER with")
                print("                    the wrist at a different angle set a different zero.")
                print("                    The jaws approach the object rotated by this much for the")
                print("                    whole episode, and vision cannot servo a roll error away.")
            else:
                sa, sb = absolute(A[j])[2], absolute(B[j])[2]
                print(f"                    cause: swept ranges differ by {sb - sa:+d} ticks "
                      f"({abs(sb - sa) * DEG_PER_TICK:.1f} deg), so the")
                print("                    midpoint between the stops moved by half of it.")
        if grip_err >= 25:
            print(f"     {'gripper':14s} {grip_err:7.0f} ticks of travel error — check grasps specifically")

        print("\n  FIX, in order of preference:")
        print("    1. It is the SAME ARM, so just copy A's file onto B's machine. Every offset")
        print("       above goes to exactly ZERO — it reproduces the frame the policy trained in.")
        print("       This beats recalibrating, which only re-rolls the dice on wrist_roll.")
        print(f"         scp {pa} <them>:{pa.parent}/")
        print(f"       Then connect once with --robot.id={pa.stem}; LeRobot writes it to the motors.")
        print("    2. Only if they must recalibrate, follow the protocol below.")
        print("    3. Either way, confirm with `lerobot-replay` on the target machine BEFORE you")
        print("       trust a rollout. If the replay lands off, calibration is still wrong.")
        print(CALIBRATION_PROTOCOL)

    print("\n  Whatever you conclude here, a calibration difference is invisible to watching a")
    print("  few rollouts — it costs success RATE, not smoothness. Score it (eval_rollouts.py),")
    print("  do not eyeball it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

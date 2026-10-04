"""Eval records: one human judgement per episode, saved after every judgement, with a Wilson 95%
interval on the success rate."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from phi.studio.evals import EvalError, EvalStore, wilson


def score_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """The Wilson interval from its definition, independent of the closed form: the set of p whose
    score test |k/n - p| / sqrt(p(1-p)/n) <= z, with each edge found by bisection."""
    ph = k / n

    def inside(p: float) -> bool:
        return abs(ph - p) <= z * math.sqrt(p * (1 - p) / n)

    low, high = 0.0, 1.0
    if ph > 0:  # p near 0 is outside, p = ph is inside
        out, inn = 0.0, ph
        for _ in range(200):
            mid = (out + inn) / 2
            out, inn = (out, mid) if inside(mid) else (mid, inn)
        low = inn
    if ph < 1:
        inn, out = ph, 1.0
        for _ in range(200):
            mid = (inn + out) / 2
            inn, out = (mid, out) if inside(mid) else (inn, mid)
        high = inn
    return low, high


@pytest.mark.parametrize("k,n", [(0, 1), (1, 1), (8, 10), (0, 10), (10, 10), (13, 20), (1, 50)])
def test_wilson_matches_the_score_test_definition(k: int, n: int) -> None:
    lo, hi = wilson(k, n)
    elo, ehi = score_interval(k, n)
    assert lo == pytest.approx(elo, abs=1e-9) and hi == pytest.approx(ehi, abs=1e-9)
    assert 0 <= lo <= k / n <= hi <= 1


def test_wilson_of_no_episodes_is_the_whole_range() -> None:
    assert wilson(0, 0) == (0.0, 1.0)


def test_every_judgement_is_on_disk_before_the_call_returns(tmp_path: Path) -> None:
    store = EvalStore(tmp_path)
    rec = store.begin(policy="mock-reach", task="put the cube in the box", planned=3, limit_s=20)
    store.mark("success", note="clean grasp", duration_s=12.5, run_id="r1")
    store.mark("failure", note="dropped it")
    on_disk = json.loads((tmp_path / "evals" / f"{rec['id']}.json").read_text())
    assert [e["outcome"] for e in on_disk["episodes"]] == ["success", "failure"]
    assert on_disk["episodes"][0] == {"n": 1, "outcome": "success", "note": "clean grasp",
                                      "duration_s": 12.5, "run_id": "r1",
                                      "at": on_disk["episodes"][0]["at"]}  # fmt: skip
    assert on_disk["successes"] == 1 and on_disk["n"] == 2 and on_disk["rate"] == 0.5
    assert on_disk["ci95"] == pytest.approx(list(wilson(1, 2)))


def test_a_restart_resumes_the_unfinished_eval(tmp_path: Path) -> None:
    a = EvalStore(tmp_path)
    rec = a.begin(policy="mock-reach", task="t", planned=5, limit_s=10)
    a.mark("success")
    b = EvalStore(tmp_path)  # Studio crashed and came back
    assert b.current is not None and b.current["id"] == rec["id"] and b.current["n"] == 1
    b.mark("failure")
    assert b.current["n"] == 2


def test_undo_removes_only_the_last_judgement(tmp_path: Path) -> None:
    s = EvalStore(tmp_path)
    s.begin(policy="p", task="t", planned=3, limit_s=10)
    s.mark("success")
    s.mark("success")
    rec = s.undo()
    assert rec["n"] == 1 and rec["successes"] == 1
    s.undo()
    with pytest.raises(EvalError):
        s.undo()  # nothing left


def test_end_closes_the_record_and_lists_it_newest_first(tmp_path: Path) -> None:
    s = EvalStore(tmp_path, now=iter(range(1_000_000, 2_000_000, 100)).__next__)
    first = s.begin(policy="p", task="a", planned=1, limit_s=10)
    s.mark("success")
    s.end()
    assert s.current is None
    second = s.begin(policy="p", task="b", planned=1, limit_s=10)
    s.mark("failure")
    s.end()
    listed = s.list()
    assert [r["id"] for r in listed] == [second["id"], first["id"]]
    assert all(r["ended_at"] for r in listed) and "episodes" not in listed[0]  # summaries only


def begin(s: EvalStore) -> dict:
    return s.begin(policy="p", task="t", planned=2, limit_s=5)


@pytest.mark.parametrize("call", [
    lambda s: s.mark("success"),  # no eval running
    lambda s: s.end(),
    lambda s: (begin(s), begin(s)),
    lambda s: (begin(s), s.mark("maybe")),
    lambda s: (begin(s), s.mark("success", note="x" * 2001)),
    lambda s: s.begin(policy="p", task="t", planned=0, limit_s=5),
    lambda s: s.begin(policy="p", task=5, planned=2, limit_s=5),
])  # fmt: skip
def test_bad_eval_calls_raise_eval_error(tmp_path: Path, call) -> None:
    with pytest.raises(EvalError):
        call(EvalStore(tmp_path))


def test_a_corrupt_record_is_skipped_not_fatal(tmp_path: Path) -> None:
    (tmp_path / "evals").mkdir()
    (tmp_path / "evals" / "bad.json").write_text("{not json")
    s = EvalStore(tmp_path)
    assert s.list() == [] and s.current is None


def test_one_policy_run_is_judged_once(tmp_path: Path) -> None:
    # A double click or a second window must not count one episode twice.
    s = EvalStore(tmp_path)
    begin(s)
    s.mark("success", run_id="a")
    with pytest.raises(EvalError, match="already judged"):
        s.mark("failure", run_id="a")
    s.mark("failure", run_id="b")
    s.undo()
    s.mark("success", run_id="b")  # undone, so it can be judged again
    assert s.current is not None and s.current["n"] == 2


def test_a_run_from_before_the_eval_cannot_be_judged_in_it(tmp_path: Path) -> None:
    # A policy run on the Run policy page, before the eval began, is not one of its episodes.
    s = EvalStore(tmp_path, now=lambda: 1000.0)
    begin(s)
    with pytest.raises(EvalError, match="before the eval"):
        s.mark("success", run_id="old", run_started_at=999.0)
    s.mark("success", run_id="new", run_started_at=1000.5)
    assert s.current is not None and s.current["n"] == 1

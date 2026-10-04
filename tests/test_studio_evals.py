"""Eval records: one human judgement per episode, saved after every judgement, with a Wilson 95%
interval on the success rate."""

from __future__ import annotations

import json
import math
from itertools import count
from pathlib import Path

import pytest

from phi.studio.evals import EvalError, EvalStore, wilson

_ids = count(1)


def ran(s: EvalStore, rid: str, **kw) -> dict:
    """The worker's view of an ended policy run of the running eval (worker.PolicyRun.view)."""
    rec = s.current
    assert rec is not None
    return {"run_id": rid, "id": rec["policy"], "task": rec["task"], "limit_s": rec["limit_s"],
            "started_at": rec["started_at"], "running": False, "episode_s": 4.0, **kw}  # fmt: skip


def judge(s: EvalStore, outcome: str, rid: str | None = None, note: str = "", **kw) -> dict:
    rid = rid or f"run{next(_ids)}"
    return s.mark(outcome, note, ran(s, rid, **kw) if s.current else None, rid)


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
    judge(store, "success", "r1", note="clean grasp", episode_s=12.5)
    judge(store, "failure", note="dropped it")
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
    judge(a, "success")
    b = EvalStore(tmp_path)  # Studio crashed and came back
    assert b.current is not None and b.current["id"] == rec["id"] and b.current["n"] == 1
    judge(b, "failure")
    assert b.current["n"] == 2


def test_undo_removes_only_the_last_judgement(tmp_path: Path) -> None:
    s = EvalStore(tmp_path)
    s.begin(policy="p", task="t", planned=3, limit_s=10)
    judge(s, "success")
    judge(s, "success")
    rec = s.undo()
    assert rec["n"] == 1 and rec["successes"] == 1
    s.undo()
    with pytest.raises(EvalError):
        s.undo()  # nothing left


def test_end_closes_the_record_and_lists_it_newest_first(tmp_path: Path) -> None:
    s = EvalStore(tmp_path, now=iter(range(1_000_000, 2_000_000, 100)).__next__)
    first = s.begin(policy="p", task="a", planned=1, limit_s=10)
    judge(s, "success")
    s.end()
    assert s.current is None
    second = s.begin(policy="p", task="b", planned=1, limit_s=10)
    judge(s, "failure")
    s.end()
    listed = s.list()
    assert [r["id"] for r in listed] == [second["id"], first["id"]]
    assert all(r["ended_at"] for r in listed) and "episodes" not in listed[0]  # summaries only


def begin(s: EvalStore) -> dict:
    return s.begin(policy="p", task="t", planned=2, limit_s=5)


@pytest.mark.parametrize("call", [
    lambda s: judge(s, "success"),  # no eval running
    lambda s: s.end(),
    lambda s: (begin(s), begin(s)),
    lambda s: (begin(s), judge(s, "maybe")),
    lambda s: (begin(s), judge(s, "success", note="x" * 2001)),
    lambda s: (begin(s), s.mark("success", "", None, None)),  # no run to judge
    lambda s: (begin(s), judge(s, "success", running=True)),  # still running
    lambda s: (begin(s), s.mark("success", "", ran(s, "a"), "b")),  # a newer run replaced it
    lambda s: (begin(s), judge(s, "success", id="other-policy")),
    lambda s: (begin(s), judge(s, "success", task="another task")),
    lambda s: (begin(s), judge(s, "success", limit_s=99.0)),
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
    judge(s, "success", "a")
    with pytest.raises(EvalError, match="already judged"):
        judge(s, "failure", "a")
    judge(s, "failure", "b")
    s.undo()
    judge(s, "success", "b")  # undone, so it can be judged again
    assert s.current is not None and s.current["n"] == 2


def test_a_run_from_before_the_eval_cannot_be_judged_in_it(tmp_path: Path) -> None:
    # A policy run on the Run policy page, before the eval began, is not one of its episodes.
    s = EvalStore(tmp_path, now=lambda: 1000.0)
    begin(s)
    with pytest.raises(EvalError, match="before the eval"):
        judge(s, "success", "old", started_at=999.0)
    judge(s, "success", "new", started_at=1000.5)
    assert s.current is not None and s.current["n"] == 1


@pytest.mark.parametrize("bad", [
    {"episodes": []},  # no id
    {"id": "x", "policy": "p", "task": "t", "planned": 1, "limit_s": 5, "started_at": None,
     "ended_at": None, "episodes": []},
    {"id": "x", "policy": "p", "task": "t", "planned": 1, "limit_s": 5, "started_at": 1.0,
     "ended_at": None, "episodes": [{"outcome": "maybe"}]},
    ["not", "a", "record"],
])  # fmt: skip
def test_a_malformed_record_is_skipped_not_fatal(tmp_path: Path, bad) -> None:
    # Before: a missing id raised in list(), and a null start time raised while Studio started.
    (tmp_path / "evals").mkdir()
    (tmp_path / "evals" / "bad.json").write_text(json.dumps(bad))
    s = EvalStore(tmp_path)
    assert s.current is None and s.list() == []


@pytest.mark.parametrize("step", ["mark", "undo", "end"])
def test_a_failed_save_changes_nothing(tmp_path: Path, step: str) -> None:
    # Before: the episode was added in memory first, so a judgement reported as not saved was
    # written with the next one.
    s = EvalStore(tmp_path)
    begin(s)
    judge(s, "success", "r1")
    before = json.dumps(s.current)

    def broken(rec: dict) -> None:
        raise PermissionError("read-only")

    s._save = broken  # type: ignore[method-assign]
    with pytest.raises(OSError):
        {"mark": lambda: judge(s, "failure", "r2"), "undo": s.undo, "end": s.end}[step]()
    assert json.dumps(s.current) == before
    del s._save  # the disk is writable again
    judge(s, "failure", "r2")  # r2 was never counted, so it can still be judged
    on_disk = json.loads(next((tmp_path / "evals").glob("*.json")).read_text())
    assert [e["run_id"] for e in on_disk["episodes"]] == ["r1", "r2"] and not on_disk["ended_at"]


def test_awaiting_is_true_only_for_an_ended_unjudged_run_of_this_eval(tmp_path: Path) -> None:
    s = EvalStore(tmp_path)
    assert not s.awaiting({"run_id": "a"})  # no eval
    begin(s)
    assert s.awaiting(ran(s, "a"))
    assert not s.awaiting(ran(s, "a", running=True))
    assert not s.awaiting(ran(s, "a", started_at=s.current["started_at"] - 1))
    assert not s.awaiting(None)
    judge(s, "success", "a")
    assert not s.awaiting(ran(s, "a"))

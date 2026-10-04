"""Eval records: a person judges each policy episode a success or a failure.

WHY the server owns the record and saves it after every judgement: a browser reload or a Studio
crash must not lose judged episodes, and an eval left unfinished is picked up on the next start.

The success rate comes with a Wilson 95% score interval, which stays inside [0, 1] and is not
degenerate at 0 or n successes, unlike the normal (Wald) interval. With 10 episodes, 8 successes
gives roughly 0.49 to 0.94: small evals say little, and the interval shows it.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from phi.studio.policy import is_finite_number

OUTCOMES = ("success", "failure")
MAX_NOTE = 2000
MAX_EPISODES = 1000


class EvalError(ValueError):
    pass


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials: every p the score test does not reject
    at level z. (0, 1) when there are no trials."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _well_formed(rec: Any) -> bool:
    """A record this module wrote. Anything else on disk is skipped, so one bad file cannot stop
    Studio from starting or listing."""
    return (
        isinstance(rec, dict)
        and isinstance(rec.get("id"), str)
        and isinstance(rec.get("policy"), str)
        and isinstance(rec.get("task"), str)
        and is_finite_number(rec.get("started_at"))
        and is_finite_number(rec.get("limit_s"))
        and (rec.get("ended_at") is None or is_finite_number(rec.get("ended_at")))
        and isinstance(rec.get("planned"), int)
        and isinstance(rec.get("episodes"), list)
        and all(isinstance(e, dict) and e.get("outcome") in OUTCOMES for e in rec["episodes"])
    )


def _summarise(rec: dict[str, Any]) -> dict[str, Any]:
    eps = rec["episodes"]
    k = sum(e["outcome"] == "success" for e in eps)
    lo, hi = wilson(k, len(eps))
    rec.update(successes=k, n=len(eps), rate=k / len(eps) if eps else None, ci95=[lo, hi])
    return rec


class EvalStore:
    def __init__(self, data_dir: Path | str, now: Callable[[], float] = time.time) -> None:
        self.dir = Path(data_dir) / "evals"
        self.now = now
        self.current: dict[str, Any] | None = None
        for rec in self._load_all():  # newest first
            if not rec.get("ended_at"):
                self.current = rec
                break

    # -- reading ------------------------------------------------------------------------------
    def _load_all(self) -> list[dict[str, Any]]:
        out = []
        if self.dir.is_dir():
            for p in self.dir.glob("*.json"):
                try:
                    rec = json.loads(p.read_text())
                except (OSError, ValueError):
                    continue  # a damaged file must not stop Studio; the others still list
                if _well_formed(rec):
                    out.append(_summarise(rec))
        return sorted(out, key=lambda r: r["started_at"], reverse=True)

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        """Summaries without episodes, newest first, not counting the running eval."""
        cur = self.current["id"] if self.current else None
        return [{k: v for k, v in r.items() if k != "episodes"}
                for r in self._load_all() if r["id"] != cur][:limit]  # fmt: skip

    # -- writing ------------------------------------------------------------------------------
    def _save(self, rec: dict[str, Any]) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / f"{rec['id']}.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(rec, indent=2))
        os.replace(tmp, path)  # WHY: a crash mid-write leaves the previous file, never half of one

    def _commit(self, rec: dict[str, Any]) -> dict[str, Any]:
        """Save, then adopt. WHY this order: if the save fails, memory still matches the disk, so a
        judgement reported as not saved is not counted later either."""
        self._save(_summarise(rec))
        self.current = None if rec["ended_at"] else rec
        return rec

    def _need(self) -> dict[str, Any]:
        if self.current is None:
            raise EvalError("No eval is running. Start one first.")
        return self.current

    def begin(self, policy: Any, task: Any, planned: Any, limit_s: Any) -> dict[str, Any]:
        if self.current is not None:
            raise EvalError("An eval is already running. End it before starting another.")
        if not isinstance(policy, str) or not policy or len(policy) > 100:
            raise EvalError("Choose a policy.")
        if not isinstance(task, str) or len(task) > 300:
            raise EvalError("The task must be text of at most 300 characters.")
        if not isinstance(planned, int) or isinstance(planned, bool) or not 1 <= planned <= 500:
            raise EvalError("Plan between 1 and 500 episodes.")
        if not is_finite_number(limit_s) or not 1 <= limit_s <= 600:
            raise EvalError("The episode time limit must be between 1 and 600 seconds.")
        t = self.now()
        base = rid = time.strftime("%Y%m%d-%H%M%S", time.localtime(t))
        n = 2
        while (self.dir / f"{rid}.json").exists():
            rid, n = f"{base}-{n}", n + 1
        rec = {"id": rid, "policy": policy, "task": task, "planned": planned,
               "limit_s": float(limit_s), "started_at": t, "ended_at": None,
               "episodes": []}  # fmt: skip
        return self._commit(rec)

    # -- judging ------------------------------------------------------------------------------
    # `run` is the worker's view of its newest policy run (worker.PolicyRun.view), taken by the
    # server from telemetry. WHY not fields from the browser: a window could then judge a run
    # Studio never ran, or one from before the eval.
    @staticmethod
    def _why_not_mine(rec: dict[str, Any], run: Any) -> str | None:
        if not isinstance(run, dict) or not isinstance(run.get("run_id"), str):
            return "There is no policy run to judge. Run an episode first."
        ran = (run.get("id"), run.get("task"), run.get("limit_s"))
        if ran != (rec["policy"], rec["task"], rec["limit_s"]):
            return "The last run used a different policy, task or time limit from this eval."
        started = run.get("started_at")
        if not is_finite_number(started) or started < rec["started_at"]:
            return "The last run started before the eval. Run a new episode to judge."
        return None

    def awaiting(self, run: Any) -> bool:
        """True while `run` is an ended episode of the running eval that nobody has judged."""
        rec = self.current
        return (
            rec is not None
            and self._why_not_mine(rec, run) is None
            and not run.get("running")
            and all(e.get("run_id") != run["run_id"] for e in rec["episodes"])
        )

    def mark(
        self, outcome: Any, note: Any = "", run: Any = None, run_id: Any = None
    ) -> dict[str, Any]:
        """Judge `run`. `run_id` is the run the window showed: if a newer run replaced it, the
        window is out of date and the judgement is refused."""
        rec = self._need()
        if outcome not in OUTCOMES:
            raise EvalError("The outcome must be success or failure.")
        if not isinstance(note, str) or len(note) > MAX_NOTE:
            raise EvalError(f"A note must be text of at most {MAX_NOTE} characters.")
        why = self._why_not_mine(rec, run)
        if why:
            raise EvalError(why)
        if run_id != run["run_id"]:
            raise EvalError("A newer run replaced the one shown. Judge the newest run.")
        if run.get("running"):
            raise EvalError("The episode is still running. Judge it when it ends.")
        if any(e.get("run_id") == run_id for e in rec["episodes"]):
            raise EvalError("This run is already judged. Undo the last judgement to change it.")
        if len(rec["episodes"]) >= MAX_EPISODES:
            raise EvalError(f"An eval holds at most {MAX_EPISODES} episodes.")
        dur = run.get("episode_s")
        episode = {"n": len(rec["episodes"]) + 1, "outcome": outcome, "note": note,
                   "duration_s": dur if is_finite_number(dur) else None, "run_id": run_id,
                   "at": self.now()}  # fmt: skip
        return self._commit({**rec, "episodes": [*rec["episodes"], episode]})

    def undo(self) -> dict[str, Any]:
        rec = self._need()
        if not rec["episodes"]:
            raise EvalError("There is no judgement to undo.")
        return self._commit({**rec, "episodes": rec["episodes"][:-1]})

    def end(self) -> dict[str, Any]:
        return self._commit({**self._need(), "ended_at": self.now()})

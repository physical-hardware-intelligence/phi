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
                    if isinstance(rec, dict) and isinstance(rec.get("episodes"), list):
                        out.append(_summarise(rec))
                except (OSError, ValueError, KeyError, TypeError):
                    continue  # a damaged file must not stop Studio; the others still list
        return sorted(out, key=lambda r: r.get("started_at", 0), reverse=True)

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
        self._save(_summarise(rec))
        self.current = rec
        return rec

    def mark(
        self, outcome: Any, note: Any = "", duration_s: Any = None, run_id: Any = None
    ) -> dict[str, Any]:
        rec = self._need()
        if outcome not in OUTCOMES:
            raise EvalError("The outcome must be success or failure.")
        if not isinstance(note, str) or len(note) > MAX_NOTE:
            raise EvalError(f"A note must be text of at most {MAX_NOTE} characters.")
        if duration_s is not None and (not is_finite_number(duration_s) or duration_s < 0):
            raise EvalError("The episode duration must be a number of seconds.")
        if run_id is not None and (not isinstance(run_id, str) or len(run_id) > 64):
            raise EvalError("The run id must be short text.")
        if run_id is not None and any(e.get("run_id") == run_id for e in rec["episodes"]):
            raise EvalError("This run is already judged. Undo the last judgement to change it.")
        if len(rec["episodes"]) >= MAX_EPISODES:
            raise EvalError(f"An eval holds at most {MAX_EPISODES} episodes.")
        rec["episodes"].append({"n": len(rec["episodes"]) + 1, "outcome": outcome, "note": note,
                                "duration_s": duration_s, "run_id": run_id,
                                "at": self.now()})  # fmt: skip
        self._save(_summarise(rec))
        return rec

    def undo(self) -> dict[str, Any]:
        rec = self._need()
        if not rec["episodes"]:
            raise EvalError("There is no judgement to undo.")
        rec["episodes"].pop()
        self._save(_summarise(rec))
        return rec

    def end(self) -> dict[str, Any]:
        rec = self._need()
        rec["ended_at"] = self.now()
        self._save(_summarise(rec))
        self.current = None
        return rec

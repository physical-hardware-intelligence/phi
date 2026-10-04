"""The assistant's process handling, against a stand-in CLI (tests/support/fake_claude.py)."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from phi.studio import assistant as A
from phi.studio.assistant import ClaudeCLI, Conversation

FAKE = [sys.executable, str(Path(__file__).parent / "support" / "fake_claude.py")]
CONTEXT = '{"session": {"state": "FAULT"}}'


def run(coro):
    return asyncio.run(coro)


def convo(tmp: Path, cmd: list[str] | None = FAKE) -> tuple[Conversation, list[dict]]:
    events: list[dict] = []
    return Conversation(ClaudeCLI(cwd=tmp, cmd=cmd), events.append), events


async def until(events: list[dict], kind: str, after: int = 0, timeout: float = 10) -> dict:
    end = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < end:
        for e in events[after:]:
            if e["kind"] == kind:
                return e
        await asyncio.sleep(0.02)
    raise AssertionError(f"no {kind} event; got {[e['kind'] for e in events]}")


def text(events: list[dict], after: int = 0) -> str:
    return "".join(e["text"] for e in events[after:] if e["kind"] == "delta")


def test_the_cli_runs_isolated_and_read_only(tmp_path: Path) -> None:
    a = ClaudeCLI(cwd=tmp_path, dirs=[Path("/cal")], cmd=["claude"]).args()
    assert a[a.index("--setting-sources") + 1] == ""  # no global CLAUDE.md, no hooks
    assert a[a.index("--tools") + 1] == "Read,Grep,Glob" and "Bash" not in " ".join(a)
    assert "--no-session-persistence" in a and "--strict-mcp-config" in a
    assert a[a.index("--add-dir") + 1] == "/cal"


def test_an_answer_streams_with_its_tool_lines_and_sees_the_context(tmp_path: Path) -> None:
    async def go() -> None:
        c, ev = convo(tmp_path)
        await c.ask("why did it fault?", CONTEXT)
        await until(ev, "done")
        kinds = [e["kind"] for e in ev]
        assert kinds[0] == "start" and "tool" in kinds and kinds[-1] == "done"
        assert next(e for e in ev if e["kind"] == "tool")["text"] == "Read robot-config.yaml"
        got = text(ev)
        assert "You asked: why did it fault?" in got and "Context has state: True" in got
        assert not c.busy and c.turns[0][0] == "why did it fault?"
        await c.aclose()

    run(go())


def test_a_second_question_reaches_the_same_process(tmp_path: Path) -> None:
    async def go() -> None:
        c, ev = convo(tmp_path)
        await c.ask("one", CONTEXT)
        await until(ev, "done")
        pid = c.proc.pid
        n = len(ev)
        await c.ask("two", CONTEXT)
        await until(ev, "done", after=n)
        assert c.proc.pid == pid and "Turn 2" in text(ev, n) and "Replayed: False" in text(ev, n)
        await c.aclose()

    run(go())


def test_stop_kills_and_the_next_question_replays_the_earlier_turns(tmp_path: Path) -> None:
    async def go() -> None:
        c, ev = convo(tmp_path)
        await c.ask("slow", CONTEXT)
        await until(ev, "delta")
        proc = c.proc
        c.stop()
        assert ev[-1]["kind"] == "done" and ev[-1]["stopped"] and not c.busy
        await asyncio.wait_for(proc.wait(), 5)
        assert c.turns[-1][1].endswith("[stopped]")
        n = len(ev)
        await c.ask("again", CONTEXT)
        await until(ev, "done", after=n)
        assert "Turn 1" in text(ev, n) and "Replayed: True" in text(ev, n)  # a new process
        await c.aclose()

    run(go())


@pytest.mark.parametrize("question,expect", [
    ("crash", "exit 3"),
    ("signout", "not signed in"),
])  # fmt: skip
def test_a_failing_cli_ends_in_an_error_with_a_fix(tmp_path: Path, question, expect) -> None:
    async def go() -> None:
        c, ev = convo(tmp_path)
        await c.ask(question, CONTEXT)
        err = await until(ev, "error")
        assert expect in err["message"] and err["fix"] and not c.busy
        if question == "crash":
            assert "simulated crash" in err["message"]  # the CLI's own words reach the window
        await c.aclose()

    run(go())


def test_a_silent_cli_is_stopped_after_the_timeout(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(A, "SILENCE_S", 1.0)

    async def go() -> None:
        c, ev = convo(tmp_path)
        await c.ask("silent", CONTEXT)
        err = await until(ev, "error", timeout=8)
        assert "silent" in err["message"] and not c.busy and c.proc is None
        await c.aclose()

    run(go())


def test_one_question_at_a_time_and_none_empty(tmp_path: Path) -> None:
    async def go() -> None:
        c, ev = convo(tmp_path)
        await c.ask("   ", CONTEXT)
        assert ev[-1]["kind"] == "error"
        await c.ask("slow", CONTEXT)
        await c.ask("another", CONTEXT)
        assert ev[-1]["kind"] == "error" and "still answering" in ev[-1]["message"]
        await c.aclose()

    run(go())


def test_status_says_whether_claude_is_usable(tmp_path: Path) -> None:
    async def go() -> None:
        ok = await ClaudeCLI(cwd=tmp_path, cmd=FAKE).status()
        assert ok["available"] and ok["signed_in"]
        out = await ClaudeCLI(cwd=tmp_path, cmd=[*FAKE, "--signed-out"]).status()
        assert not out["available"] and "claude auth login" in out["fix"]
        none = await ClaudeCLI(cwd=tmp_path, cmd=None).status()
        assert not none["available"] and "not installed" in none["error"]
        c, ev = convo(tmp_path, cmd=None)
        await c.ask("hi", CONTEXT)
        assert ev[-1]["kind"] == "error" and "not installed" in ev[-1]["message"]
        await c.aclose()

    run(go())

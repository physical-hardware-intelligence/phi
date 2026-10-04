"""Claude inside Studio. One `claude` CLI process per conversation, read-only tools, and Studio's
live state attached to every question, so nobody copies an error into another app.

    window --assist_ask--> server --stdin JSON line--> claude -p --input-format stream-json
    window <--assist events-- server <--stdout JSON events--

WHY the CLI and not an API key: it uses the Claude login already on this Mac, so Studio stores no
secret. WHY one long-lived process per conversation: earlier turns, including the files Claude read,
stay in its context. If the process dies or is stopped, the next question starts a new one and
replays the earlier turns as text.

Safety: Claude gets Read, Grep and Glob, nothing else. Its process has no channel to the worker, so
it cannot move an arm; it can only tell the person what to do.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shlex
import shutil
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MAX_QUESTION = 8000
REPLAY_CHARS = 12_000  # earlier turns replayed into a new process, newest kept
SILENCE_S = 120.0  # no output for this long during an answer: give up and say so
TOOLS = "Read,Grep,Glob"
# Permission rules on top of the file view's own checks (files.SECRET).
DENY = ("Read(**/.env*)", "Read(**/*token*)", "Read(**/*secret*)", "Read(**/*credential*)",
        "Read(**/*.pem)", "Read(**/*.key)", "Read(~/.ssh/**)", "Read(**/.git/**)")  # fmt: skip
SIGN_IN_FIX = "Sign in once in a terminal: claude auth login. Then ask again."

SYSTEM = """You are the assistant inside Phi Studio, a local web app on the user's Mac for LeRobot
SO-101 robot arms: connect, calibrate, teleoperate leader to follower (one or two pairs), run
policies, and evaluate them.

How you work here:
- Every question arrives with a <studio_context> block: what Studio sees right now (session state,
  arms, ports, calibration matches, servo health, the policy run, the eval, recent log, the page,
  and the error the user asked about). Trust it over assumptions, and name the field you used.
- You can read files with Read, Grep and Glob, inside the folders listed in the context. You
  cannot run commands, change files, or move the arms. When a fix needs an action, say exactly
  which Studio button or which terminal command, and why.
- Cite files as path:line, with the path relative to its folder (for example
  src/phi/studio/worker.py:303) or absolute. Studio turns these into links.
- Answer briefly: the cause, the evidence, the fix, in that order. Short paragraphs or a numbered
  list. If you are unsure, say what you checked and what would settle it.
- Safety: never suggest bypassing Stop, the torque-off confirmation, or a calibration check.
  Motion needs a person at the rig.

Facts about this setup:
- macOS names an arm's serial port /dev/tty.usbmodem<serial> (also listed as /dev/cu.*). Names
  change with the USB socket and after reboots. robot-config.yaml in the main checkout, if
  present, is this Mac's record of ports, arm ids and cameras. LeRobot's lerobot-find-port finds a
  port by unplugging the arm.
- LeRobot keeps one calibration JSON per arm id in its calibration folder, with id, drive_mode,
  homing_offset, range_min and range_max per joint. Present_Position = Actual - Homing_Offset.
- Studio's code: src/phi/studio/ (server.py, worker.py, session.py, identity.py, evals.py,
  files.py) and the web UI in studio/web/src."""


def find_cli() -> list[str] | None:
    """The command that runs Claude. PHI_STUDIO_CLAUDE overrides it (tests use a stand-in)."""
    if os.getenv("PHI_STUDIO_CLAUDE"):
        return shlex.split(os.environ["PHI_STUDIO_CLAUDE"])
    path = shutil.which("claude") or str(Path("~/.local/bin/claude").expanduser())
    return [path] if Path(path).is_file() else None


def child_env() -> dict[str, str]:
    env = dict(os.environ)
    # WHY: Studio may itself run under Claude Code; these mark a nested session.
    for k in ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT"):
        env.pop(k, None)
    return env


@dataclass
class ClaudeCLI:
    cwd: Path
    dirs: list[Path] = field(default_factory=list)
    model: str | None = None
    cmd: list[str] | None = field(default_factory=find_cli)

    def args(self) -> list[str]:
        assert self.cmd is not None
        a = [*self.cmd, "-p", "--input-format", "stream-json", "--output-format", "stream-json",
             "--verbose", "--include-partial-messages", "--setting-sources", "",
             "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}', "--system-prompt", SYSTEM,
             "--tools", TOOLS, "--allowedTools", TOOLS, "--disallowedTools", " ".join(DENY),
             "--no-session-persistence"]  # fmt: skip
        for d in self.dirs:
            a += ["--add-dir", str(d)]
        if self.model:
            a += ["--model", self.model]
        return a

    async def status(self) -> dict[str, Any]:
        """Is Claude usable on this Mac? Runs `claude auth status`, which prints JSON."""
        base = {"model": self.model, "signed_in": False, "method": None}
        if self.cmd is None:
            return {**base, "available": False,
                    "error": "Claude Code is not installed on this Mac.",
                    "fix": "Install Claude Code, then sign in: claude auth login"}  # fmt: skip
        try:
            p = await asyncio.create_subprocess_exec(
                *self.cmd,
                "auth",
                "status",
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=child_env(),
            )
            out, _ = await asyncio.wait_for(p.communicate(), 15)
            info = json.loads(out.decode() or "{}")
        except (OSError, ValueError, TimeoutError) as e:
            return {**base, "available": False,
                    "error": f"Could not ask Claude Code its status: {e}",
                    "fix": "Check that `claude --version` runs in a terminal."}  # fmt: skip
        if not info.get("loggedIn"):
            return {**base, "available": False, "error": "Claude is not signed in on this Mac.",
                    "fix": SIGN_IN_FIX}  # fmt: skip
        return {**base, "available": True, "signed_in": True, "method": info.get("authMethod"),
                "error": None, "fix": None}  # fmt: skip


def describe_tool(name: str, inp: dict[str, Any]) -> str:
    """One short line for a tool call, as the window shows it."""
    if name == "Read":
        return f"Read {inp.get('file_path', '')}"
    if name == "Grep":
        where = inp.get("path") or inp.get("glob") or "the folders"
        return f"Searched for “{inp.get('pattern', '')}” in {where}"
    if name == "Glob":
        return f"Listed {inp.get('pattern', '')}"
    return name


class Conversation:
    """One window's conversation with Claude. `emit` sends an event to that window only."""

    def __init__(self, cli: ClaudeCLI, emit: Callable[[dict[str, Any]], None]) -> None:
        self.cli = cli
        self.emit = emit
        self.id = uuid.uuid4().hex[:12]
        self.proc: asyncio.subprocess.Process | None = None
        self.reader: asyncio.Task[None] | None = None
        self.turns: list[tuple[str, str]] = []  # (question, answer), for replay into a new process
        self.busy = False
        self.question = ""
        self.answer = ""
        self._streamed = False  # text arrived as deltas for the current message
        self._stderr: list[str] = []
        self._heard = 0.0  # loop time of the last output, or of the question
        self._tasks: set[asyncio.Task[None]] = set()  # readers of every process started, to reap

    def _send(self, kind: str, **kw: Any) -> None:
        self.emit({"type": "assist", "conversation": self.id, "kind": kind, **kw})

    # -- process ----------------------------------------------------------------------------------
    async def _ensure(self) -> bool:
        if self.proc is not None and self.proc.returncode is None:
            return False
        if self.cli.cmd is None:
            raise OSError("Claude Code is not installed on this Mac.")
        self.proc = await asyncio.create_subprocess_exec(
            *self.cli.args(), cwd=str(self.cli.cwd), stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=child_env(),
            limit=16 * 1024 * 1024)  # fmt: skip
        self._stderr = []
        self.reader = asyncio.create_task(self._read(self.proc))
        for t in (self.reader, asyncio.create_task(self._drain_stderr(self.proc))):
            self._tasks.add(t)
            t.add_done_callback(self._tasks.discard)
        return True

    async def _drain_stderr(self, proc: asyncio.subprocess.Process) -> None:
        assert proc.stderr is not None
        async for line in proc.stderr:
            self._stderr = [*self._stderr[-19:], line.decode(errors="replace").rstrip()]

    def _replay(self) -> str:
        if not self.turns:
            return ""
        parts, size = [], 0
        for q, a in reversed(self.turns):
            block = f"User: {q}\nYou: {a}\n"
            if size + len(block) > REPLAY_CHARS:
                break
            parts.append(block)
            size += len(block)
        return ("<earlier_turns>\nThe conversation so far, restored after a restart:\n"
                + "\n".join(reversed(parts)) + "</earlier_turns>\n\n")  # fmt: skip

    async def ask(self, question: Any, context: str) -> None:
        if not isinstance(question, str) or not question.strip():
            self._send("error", message="Type a question first.", fix=None)
            return
        if len(question) > MAX_QUESTION:
            self._send("error", message=f"Keep a question under {MAX_QUESTION} characters.",
                       fix=None)  # fmt: skip
            return
        if self.busy:
            self._send("error", message="Claude is still answering.", fix="Stop it, or wait.")
            return
        try:
            fresh = await self._ensure()
        except OSError as e:
            self._send("error", message=f"Could not start Claude: {e}", fix=SIGN_IN_FIX)
            return
        body = (self._replay() if fresh else "") + (
            f"<studio_context>\n{context}\n</studio_context>\n\n{question}"
        )
        line = json.dumps({"type": "user", "message": {"role": "user", "content": body}}) + "\n"
        self.busy, self.question, self.answer, self._streamed = True, question, "", False
        self._heard = asyncio.get_running_loop().time()
        self._send("start", question=question)
        try:
            assert self.proc is not None and self.proc.stdin is not None
            self.proc.stdin.write(line.encode())
            await self.proc.stdin.drain()
        except (OSError, ConnectionError) as e:
            self._fail(f"Could not send the question to Claude: {e}")

    def stop(self) -> None:
        """Kill the process. The partial answer is kept, marked as stopped."""
        if self.busy:
            self.turns.append((self.question, self.answer + " [stopped]"))
            self.busy = False
            self._send("done", stopped=True, cost_usd=None, duration_ms=None)
        self._kill()

    def reset(self) -> None:
        self.stop()
        self.turns.clear()
        self.id = uuid.uuid4().hex[:12]

    def close(self) -> None:
        self._kill()

    async def aclose(self) -> None:
        """Kill and wait until every process this conversation started has exited."""
        self._kill()
        if self._tasks:
            await asyncio.wait(set(self._tasks), timeout=5)

    def _kill(self) -> None:
        p, self.proc = self.proc, None
        if p is not None and p.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                p.kill()

    def _fail(self, message: str, fix: str | None = None) -> None:
        if self.busy:
            self.turns.append((self.question, self.answer + " [failed]"))
        self.busy = False
        self._send("error", message=message, fix=fix)
        self._kill()

    # -- events -----------------------------------------------------------------------------------
    async def _read(self, proc: asyncio.subprocess.Process) -> None:
        assert proc.stdout is not None
        loop = asyncio.get_running_loop()
        while True:
            try:  # WHY a short poll: a question can arrive while a read is already waiting
                raw = await asyncio.wait_for(proc.stdout.readline(), 2.0)
            except TimeoutError:
                if proc is self.proc and self.busy and loop.time() - self._heard > SILENCE_S:
                    self._fail(f"Claude was silent for {SILENCE_S:.0f} s, so Studio stopped it.",
                               "Ask again. If it keeps happening, check your network.")  # fmt: skip
                    return
                continue
            except ValueError:  # one line over the stream limit: skip it, keep reading
                continue
            if not raw:
                break
            self._heard = loop.time()
            try:
                ev = json.loads(raw)
            except ValueError:
                continue
            if proc is self.proc:
                self._on_event(ev)
        await proc.wait()
        if proc is self.proc and self.busy:
            tail = " ".join(self._stderr[-3:])[-400:]
            self._fail(f"Claude stopped unexpectedly (exit {proc.returncode}). {tail}".strip(),
                       "Ask again. If it repeats, run `claude -p hi` in a terminal.")  # fmt: skip
        if proc is self.proc:
            self.proc = None

    def _on_event(self, ev: dict[str, Any]) -> None:
        kind = ev.get("type")
        if kind == "stream_event":
            e = ev.get("event") or {}
            if e.get("type") == "message_start":
                self._streamed = False
            d = e.get("delta") or {}
            if e.get("type") == "content_block_delta" and d.get("type") == "text_delta":
                self._streamed = True
                self.answer += d.get("text", "")
                self._send("delta", text=d.get("text", ""))
        elif kind == "assistant":
            for block in (ev.get("message") or {}).get("content") or []:
                if block.get("type") == "tool_use":
                    name, inp = block.get("name", ""), block.get("input") or {}
                    self._send("tool", name=name, text=describe_tool(name, inp))
                elif block.get("type") == "text" and not self._streamed:
                    self.answer += block.get("text", "")
                    self._send("delta", text=block.get("text", ""))
        elif kind == "result":
            text = ev.get("result") if isinstance(ev.get("result"), str) else ""
            if not self.busy:
                return
            if "Failed to authenticate" in text or "not logged in" in text.lower():
                self._fail("Claude is not signed in on this Mac.", SIGN_IN_FIX)
                return
            if ev.get("is_error") or ev.get("subtype") not in (None, "success"):
                self._fail(f"Claude could not answer: {text or ev.get('subtype')}",
                           "Ask again. If it repeats, start a new conversation.")  # fmt: skip
                return
            self.turns.append((self.question, self.answer))
            self.busy = False
            self._send("done", stopped=False, cost_usd=ev.get("total_cost_usd"),
                       duration_ms=ev.get("duration_ms"))  # fmt: skip
        elif kind == "rate_limit_event":
            info = ev.get("rate_limit_info") or {}
            if info.get("status") not in (None, "allowed"):
                self._send("limit", status=info.get("status"), resets_at=info.get("resetsAt"))

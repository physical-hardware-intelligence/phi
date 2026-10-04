"""Server: auth, control, frames, and the heartbeat path end to end with a real worker process
(mock rig). SPEC X-2, TEL-4, TEL-5."""

from __future__ import annotations

import asyncio
import json
import socket
import struct
import sys
import time
from pathlib import Path
from typing import Any

import pytest

aiohttp = pytest.importorskip("aiohttp")
pytest.importorskip("PIL")
from aiohttp.test_utils import TestServer  # noqa: E402

import phi.studio.server as server_mod  # noqa: E402
from phi.studio.server import Studio  # noqa: E402


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


async def started(pairs: int = 1) -> tuple[Studio, TestServer, aiohttp.ClientSession]:
    port = free_port()
    studio = Studio({"kind": "mock", "pairs": pairs, "cameras": ["front"]}, port, token="t0k")
    server = TestServer(studio.app(), host="127.0.0.1", port=port)
    await server.start_server()
    return studio, server, aiohttp.ClientSession()


async def ws(
    session: aiohttp.ClientSession, port: int, token: str = "t0k", origin: str | None = None
) -> Any:
    origin = origin or f"http://127.0.0.1:{port}"
    return await session.ws_connect(
        f"http://127.0.0.1:{port}/ws?token={token}", headers={"Origin": origin}
    )


async def until(sock: Any, pred, timeout: float = 5.0, beat: bool = False) -> dict[str, Any]:
    """Read JSON messages until pred(msg) holds. Optionally keep sending heartbeats."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if beat:
            await sock.send_str(json.dumps({"cmd": "heartbeat"}))
        try:
            m = await sock.receive(timeout=0.2)
        except TimeoutError:
            continue
        if m.type == aiohttp.WSMsgType.TEXT:
            d = json.loads(m.data)
            if pred(d):
                return d
    raise AssertionError("condition not met in time")


def run(coro):
    return asyncio.run(coro)


def test_refuses_bad_token_origin_and_host() -> None:
    async def go() -> None:
        studio, server, session = await started()
        port = server.port
        try:
            for kwargs in ({"token": "wrong"}, {"origin": "https://evil.example"}):
                with pytest.raises(aiohttp.WSServerHandshakeError) as e:
                    await ws(session, port, **kwargs)
                assert e.value.status == 403
            r = await session.get(
                f"http://127.0.0.1:{port}/ws?token=t0k",
                headers={"Host": f"attacker.example:{port}", "Origin": f"http://127.0.0.1:{port}"},
            )
            assert r.status == 403 and "host" in await r.text()
        finally:
            await session.close()
            await server.close()

    run(go())


def test_connect_identify_and_state_reach_the_window() -> None:
    async def go() -> None:
        studio, server, session = await started(pairs=2)
        try:
            w = await ws(session, server.port)
            hello = await until(w, lambda d: d["type"] == "hello")
            assert hello["control"] is True and hello["mock"] is True
            await w.send_str(json.dumps({"cmd": "connect"}))
            ident = await until(w, lambda d: d["type"] == "identity")
            assert len(ident["arms"]) == 4 and all(a["exact"] for a in ident["arms"])
            await until(w, lambda d: d["type"] == "state" and d["state"] == "IDENTIFIED")
        finally:
            await session.close()
            await server.close()

    run(go())


def test_camera_frames_arrive_as_binary_packets() -> None:
    async def go() -> None:
        studio, server, session = await started()
        try:
            w = await ws(session, server.port)
            end = time.monotonic() + 5
            while time.monotonic() < end:
                m = await w.receive(timeout=2)
                if m.type == aiohttp.WSMsgType.BINARY:
                    assert m.data[:1] == b"F"
                    (n,) = struct.unpack(">H", m.data[1:3])
                    head = json.loads(m.data[3 : 3 + n])
                    assert head["key"] == "front" and head["w"] == 640
                    assert m.data[3 + n : 3 + n + 2] == b"\xff\xd8"  # JPEG magic
                    return
            raise AssertionError("no frame")
        finally:
            await session.close()
            await server.close()

    run(go())


def test_second_window_may_stop_but_not_arm() -> None:
    async def go() -> None:
        studio, server, session = await started()
        try:
            a = await ws(session, server.port)
            await until(a, lambda d: d["type"] == "hello")
            b = await ws(session, server.port)
            hb = await until(b, lambda d: d["type"] == "hello")
            assert hb["control"] is False
            await b.send_str(json.dumps({"cmd": "connect"}))
            err = await until(b, lambda d: d["type"] == "error")
            assert "control" in err["message"]
            for c in ("connect", "confirm", "arm"):
                await a.send_str(json.dumps({"cmd": c}))
            await until(a, lambda d: d.get("state") == "ARMED", beat=True)
            await b.send_str(json.dumps({"cmd": "stop"}))
            await until(b, lambda d: d.get("state") == "STOPPED")
        finally:
            await session.close()
            await server.close()

    run(go())


def test_closing_the_controlling_window_stops_motion() -> None:
    async def go() -> None:
        studio, server, session = await started()
        try:
            a = await ws(session, server.port)
            await until(a, lambda d: d["type"] == "hello")
            watcher = await ws(session, server.port)
            for c in ("connect", "confirm", "arm"):
                await a.send_str(json.dumps({"cmd": c}))
            await a.send_str(json.dumps({"cmd": "start", "activity": "teleop"}))
            await until(a, lambda d: d.get("state") == "MOVING", beat=True)
            t0 = time.monotonic()
            await a.close()
            s = await until(watcher, lambda d: d.get("state") == "STOPPED", timeout=3)
            assert s["stop_reason"] == "window closed" and time.monotonic() - t0 < 0.5
        finally:
            await session.close()
            await server.close()

    run(go())


def test_worker_death_is_reported_to_every_window() -> None:
    async def go() -> None:
        studio, server, session = await started()
        try:
            w = await ws(session, server.port)
            await until(w, lambda d: d["type"] == "hello")
            studio.proc.kill()
            d = await until(w, lambda d: d["type"] == "worker_exit")
            assert "Restart" in d["message"]
        finally:
            await session.close()
            await server.close()

    run(go())


def test_a_reload_inside_the_heartbeat_window_still_stops() -> None:
    # Cmd+R mid-teleop: the new window takes control and beats within 1 s, which the heartbeat
    # watchdog alone would not catch.
    async def go() -> None:
        studio, server, session = await started()
        try:
            a = await ws(session, server.port)
            await until(a, lambda d: d["type"] == "hello")
            for c in ("connect", "confirm", "arm"):
                await a.send_str(json.dumps({"cmd": c}))
            await a.send_str(json.dumps({"cmd": "start", "activity": "teleop"}))
            await until(a, lambda d: d.get("state") == "MOVING", beat=True)
            await a.close()
            b = await ws(session, server.port)
            await until(b, lambda d: d.get("state") == "STOPPED", timeout=1.5, beat=True)
        finally:
            await session.close()
            await server.close()

    run(go())


def test_taking_control_mid_motion_stops_first() -> None:
    async def go() -> None:
        studio, server, session = await started()
        try:
            a = await ws(session, server.port)
            await until(a, lambda d: d["type"] == "hello")
            b = await ws(session, server.port)
            for c in ("connect", "confirm", "arm"):
                await a.send_str(json.dumps({"cmd": c}))
            await a.send_str(json.dumps({"cmd": "start", "activity": "teleop"}))
            await until(a, lambda d: d.get("state") == "MOVING", beat=True)
            await b.send_str(json.dumps({"cmd": "take_control"}))
            s = await until(b, lambda d: d.get("state") == "STOPPED", timeout=1.0)
            assert s["stop_reason"] == "control moved"
        finally:
            await session.close()
            await server.close()

    run(go())


class _Fake:
    """A Client stand-in that records what the server pushes to it."""

    def __init__(self) -> None:
        self.got: list[dict[str, Any]] = []

    def push(self, msg: dict[str, Any]) -> None:
        self.got.append(msg)


def offline_studio() -> tuple[Studio, list[dict[str, Any]]]:
    studio = Studio({"kind": "mock"}, 1, token="t0k")
    forwarded: list[dict[str, Any]] = []
    studio.to_worker = forwarded.append  # type: ignore[method-assign]
    return studio, forwarded


def test_a_window_without_control_cannot_arm_and_its_heartbeats_are_ignored() -> None:
    studio, forwarded = offline_studio()
    a, b = _Fake(), _Fake()
    studio.controller = a  # type: ignore[assignment]
    for cmd in ("heartbeat", "arm", "start", "release", "disconnect"):
        studio._from_client(b, json.dumps({"cmd": cmd}))  # type: ignore[arg-type]
    assert forwarded == []
    studio._from_client(b, json.dumps({"cmd": "stop"}))  # type: ignore[arg-type]
    assert forwarded == [{"cmd": "stop"}]


@pytest.mark.parametrize("raw", ['{"cmd": ["stop"]}', '{"cmd": {"a": 1}}', "[1, 2]", '"stop"',
                                 '{"cmd": 7}', "not json"])  # fmt: skip
def test_malformed_messages_get_an_error_not_a_dropped_socket(raw: str) -> None:
    studio, forwarded = offline_studio()
    a = _Fake()
    studio.controller = a  # type: ignore[assignment]
    studio._from_client(a, raw)  # type: ignore[arg-type]
    assert forwarded == [] and a.got[-1]["type"] == "error"


def test_non_ascii_token_is_refused_not_a_server_error() -> None:
    async def go() -> None:
        studio, server, session = await started()
        try:
            with pytest.raises(aiohttp.WSServerHandshakeError) as e:
                await ws(session, server.port, token="%C3%A9")
            assert e.value.status == 403
        finally:
            await session.close()
            await server.close()

    run(go())


def test_a_busy_port_is_a_one_line_message_not_a_traceback() -> None:
    from typer.testing import CliRunner

    from phi.cli import app

    with socket.socket() as held:
        held.bind(("127.0.0.1", 0))
        held.listen()
        port = held.getsockname()[1]
        r = CliRunner().invoke(app, ["studio", "--mock", "--no-browser", "--port", str(port)])
    assert r.exit_code == 1
    assert f"Port {port} is in use" in r.output and "--port" in r.output
    assert "Traceback" not in r.output


async def started_with_data(tmp: Any) -> tuple[Studio, TestServer, aiohttp.ClientSession]:
    port = free_port()
    studio = Studio({"kind": "mock", "pairs": 1, "cameras": ["front"]}, port, token="t0k",
                    data_dir=tmp)  # fmt: skip
    server = TestServer(studio.app(), host="127.0.0.1", port=port)
    await server.start_server()
    return studio, server, aiohttp.ClientSession()


def test_the_rig_description_reaches_every_window(tmp_path: Any) -> None:
    async def go() -> None:
        studio, server, session = await started_with_data(tmp_path)
        try:
            a = await ws(session, server.port)
            rig = await until(a, lambda d: d["type"] == "rig")
            assert any(p["id"] == "mock-reach" and p["available"] for p in rig["policies"])
            b = await ws(session, server.port)  # a window opened later gets it replayed
            await until(b, lambda d: d["type"] == "rig")
        finally:
            await session.close()
            await server.close()

    run(go())


def test_evals_are_kept_by_the_server_saved_and_shown_to_every_window(tmp_path: Any) -> None:
    async def go() -> None:
        studio, server, session = await started_with_data(tmp_path)
        try:
            a = await ws(session, server.port)
            await until(a, lambda d: d["type"] == "hello")
            send = lambda m: a.send_str(json.dumps(m))  # noqa: E731
            await send({"cmd": "eval_begin", "policy": "mock-reach", "task": "cube in box",
                        "planned": 2, "limit_s": 1})  # fmt: skip
            ev = await until(a, lambda d: d["type"] == "eval" and d["current"] is not None)
            rid = ev["current"]["id"]
            await send({"cmd": "eval_mark", "outcome": "success", "run_id": "made-up"})
            err = await until(a, lambda d: d["type"] == "error")
            assert "no policy run" in err["message"]  # the server judges only runs it saw

            for c in ("connect", "confirm", "arm"):
                await send({"cmd": c})
            await until(a, lambda d: d["type"] == "state" and d["state"] == "ARMED", beat=True)
            await send({"cmd": "start", "activity": "policy", "policy": "mock-reach",
                        "task": "cube in box", "limit_s": 1})  # fmt: skip
            t = await until(a, lambda d: d["type"] == "telemetry" and d["policy"]
                            and not d["policy"]["running"], beat=True)  # fmt: skip
            run_id = t["policy"]["run_id"]

            await send({"cmd": "resume"})  # a new run now would replace the unjudged one
            await until(a, lambda d: d["type"] == "state" and d["state"] == "ARMED", beat=True)
            await send({"cmd": "start", "activity": "policy", "policy": "mock-reach",
                        "task": "cube in box", "limit_s": 1})  # fmt: skip
            err = await until(a, lambda d: d["type"] == "error", beat=True)
            assert "not judged" in err["message"]

            await send({"cmd": "eval_mark", "outcome": "success", "note": "ok", "run_id": run_id})
            ev = await until(a, lambda d: d["type"] == "eval" and d["current"]["n"] == 1, beat=True)
            assert ev["current"]["ci95"][1] == 1.0
            assert ev["current"]["episodes"][0]["duration_s"] == pytest.approx(1.0, abs=0.1)
            assert (tmp_path / "evals" / f"{rid}.json").exists()

            b = await ws(session, server.port)  # view-only window
            ev_b = await until(b, lambda d: d["type"] == "eval")
            assert ev_b["current"]["id"] == rid
            await b.send_str(json.dumps({"cmd": "eval_mark", "outcome": "failure"}))
            err = await until(b, lambda d: d["type"] == "error")
            assert "control" in err["message"]

            await send({"cmd": "eval_mark", "outcome": "success", "run_id": run_id})
            err = await until(a, lambda d: d["type"] == "error", beat=True)
            assert "already judged" in err["message"]
            await send({"cmd": "eval_end"})
            ev = await until(a, lambda d: d["type"] == "eval" and d["current"] is None, beat=True)
            assert ev["past"][0]["id"] == rid and ev["past"][0]["n"] == 1
        finally:
            await session.close()
            await server.close()

    run(go())


def test_a_bad_eval_file_does_not_break_new_windows(tmp_path: Any) -> None:
    # Before: a record without an id made every new window's handler raise after it was added to
    # the clients, leaking it (and possibly control) on each reconnect.
    (tmp_path / "evals").mkdir()
    (tmp_path / "evals" / "bad.json").write_text(json.dumps({"episodes": []}))

    async def go() -> None:
        studio, server, session = await started_with_data(tmp_path)
        try:
            a = await ws(session, server.port)
            ev = await until(a, lambda d: d["type"] == "eval")
            assert ev["current"] is None and ev["past"] == []
            await a.close()
            for _ in range(20):
                if not studio.clients:
                    break
                await asyncio.sleep(0.05)
            assert not studio.clients and studio.controller is None
        finally:
            await session.close()
            await server.close()

    run(go())


def test_evals_without_a_data_directory_say_so() -> None:
    async def go() -> None:
        studio, server, session = await started()
        try:
            a = await ws(session, server.port)
            await until(a, lambda d: d["type"] == "hello")
            await a.send_str(json.dumps({"cmd": "eval_begin", "policy": "p", "task": "t",
                                         "planned": 1, "limit_s": 5}))  # fmt: skip
            err = await until(a, lambda d: d["type"] == "error")
            assert "data directory" in err["message"]
        finally:
            await session.close()
            await server.close()

    run(go())


# -- assistant and file view ---------------------------------------------------------------------
FAKE_CLI = f"{sys.executable} {Path(__file__).parent / 'support' / 'fake_claude.py'}"


async def started_with_code(tmp: Path) -> tuple[Studio, TestServer, aiohttp.ClientSession]:
    code = tmp / "code"
    code.mkdir()
    (code / "robot-config.yaml").write_text("robot:\n  port: /dev/tty.usbmodem123\n")
    (code / ".env").write_text("HF_TOKEN=hf_x\n")
    port = free_port()
    studio = Studio({"kind": "mock", "pairs": 1, "cameras": ["front"]}, port, token="t0k",
                    data_dir=tmp / "data", code_root=code)  # fmt: skip
    server = TestServer(studio.app(), host="127.0.0.1", port=port)
    await server.start_server()
    return studio, server, aiohttp.ClientSession()


def test_any_window_can_read_files_but_not_secrets(tmp_path: Path, monkeypatch) -> None:
    async def go() -> None:
        studio, server, session = await started_with_code(tmp_path)
        try:
            a = await ws(session, server.port)  # takes control
            b = await ws(session, server.port)  # view only
            await until(b, lambda d: d["type"] == "hello" and not d["control"])
            await b.send_str(json.dumps({"cmd": "files_index"}))
            ix = await until(b, lambda d: d["type"] == "files")
            assert [n["path"] for n in ix["notes"]] == ["robot-config.yaml"]
            await b.send_str(json.dumps({"cmd": "file_read", "path": "robot-config.yaml"}))
            f = await until(b, lambda d: d["type"] == "file")
            assert "usbmodem123" in f["text"] and f["root"] == "code"
            await b.send_str(json.dumps({"cmd": "file_read", "root": "code", "path": ".env"}))
            err = await until(b, lambda d: d["type"] == "file_error")
            assert "secrets" in err["message"]
            await b.send_str(json.dumps({"cmd": "files_search", "query": "usbmodem"}))
            res = await until(b, lambda d: d["type"] == "search")
            assert [h["path"] for h in res["hits"]] == ["robot-config.yaml"]
            await b.send_str(json.dumps({"cmd": "ports"}))
            await until(b, lambda d: d["type"] == "ports")
            await a.close()
        finally:
            await session.close()
            await server.close()

    run(go())


def test_the_assistant_answers_only_the_window_that_asked(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PHI_STUDIO_CLAUDE", FAKE_CLI)

    async def go() -> None:
        studio, server, session = await started_with_code(tmp_path)
        try:
            a = await ws(session, server.port)
            b = await ws(session, server.port)
            await until(b, lambda d: d["type"] == "hello")
            await b.send_str(json.dumps({"cmd": "assist_status"}))
            st = await until(b, lambda d: d["type"] == "assist_status")
            assert st["available"]
            await a.send_str(json.dumps({"cmd": "connect"}))  # gives the log a state change
            await until(a, lambda d: d["type"] == "state" and d["state"] == "IDENTIFIED")
            await b.send_str(json.dumps({"cmd": "assist_ask", "text": "what state?",
                                         "page": "overview",
                                         "focus": {"message": "boom"}}))  # fmt: skip
            done = await until(b, lambda d: d["type"] == "assist" and d["kind"] == "done")
            assert done["stopped"] is False
            await b.send_str(json.dumps({"cmd": "assist_context", "page": "overview",
                                         "focus": {"message": "boom"}}))  # fmt: skip
            ctx = (await until(b, lambda d: d["type"] == "assist_context"))["text"]
            assert '"IDENTIFIED"' in ctx and '"boom"' in ctx and "t0k" not in ctx
            assert '"robot-config.yaml"' not in ctx  # files are read on demand, not pushed
            with pytest.raises(AssertionError):  # the other window heard nothing
                await until(a, lambda d: d["type"] == "assist", timeout=0.5)
            proc = next(c for c in studio.clients if c.conversation).conversation.proc
            await b.close()
            await asyncio.wait_for(proc.wait(), 5)  # closing the window ends its Claude process
            await a.close()
        finally:
            await session.close()
            await server.close()

    run(go())


def test_checks_run_from_any_window_and_reach_the_assistant(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PHI_STUDIO_CLAUDE", f"{FAKE_CLI} --signed-out")

    async def go() -> None:
        studio, server, session = await started_with_code(tmp_path)
        try:
            a = await ws(session, server.port)
            b = await ws(session, server.port)  # view only
            await until(b, lambda d: d["type"] == "hello" and not d["control"])
            await b.send_str(json.dumps({"cmd": "checks_run"}))
            first = await until(b, lambda d: d["type"] == "checks")
            before = {r["id"]: r for r in first["results"]}
            assert len(before) == 21 and before["arms_answer"]["status"] == "skip"
            await a.send_str(json.dumps({"cmd": "connect"}))
            await until(b, lambda d: d["type"] == "telemetry" and d["arms"], beat=False)
            await b.send_str(json.dumps({"cmd": "checks_run"}))
            got = await until(b, lambda d: d["type"] == "checks", timeout=10)
            res = {r["id"]: r for r in got["results"]}
            assert res["arms_answer"]["status"] == "pass" and res["faults"]["status"] == "pass"
            assert res["assistant"]["status"] == "warn"
            assert "claude auth login" in res["assistant"]["fix"]
            assert res["rig_config"]["status"] == "warn"  # this config names a port but no id
            assert res["ports_identity"]["status"] == "skip"  # mock ports are not real
            await b.send_str(json.dumps({"cmd": "assist_context", "page": "checks"}))
            ctx = json.loads((await until(b, lambda d: d["type"] == "assist_context"))["text"])
            titles = [r["title"] for r in ctx["checks"]["not_passing"]]
            assert "Claude assistant" in titles and "Rig config" in titles
            assert ctx["checks"]["minutes_ago"] == 0 and "not live" in ctx["checks"]["note"]
            await a.close()
            await b.close()
        finally:
            await session.close()
            await server.close()

    run(go())


def test_a_read_only_command_that_crashes_still_answers_its_window(
    tmp_path: Path, monkeypatch
) -> None:
    # WHY: an uncaught error here left the window's spinner up for good (review, 2026-10-04).
    def boom(_):
        raise RuntimeError("probe exploded")

    monkeypatch.setattr(server_mod, "run_checks", boom)

    async def go() -> None:
        studio, server, session = await started_with_code(tmp_path)
        try:
            b = await ws(session, server.port)
            await b.send_str(json.dumps({"cmd": "checks_run"}))
            got = await until(b, lambda d: d["type"] == "checks")
            assert "probe exploded" in got["error"] and "results" not in got
            await b.send_str(json.dumps({"cmd": "files_search", "query": ""}))
            err = await until(b, lambda d: d["type"] == "file_error")
            assert err["op"] == "search"
            await b.close()
        finally:
            await session.close()
            await server.close()

    run(go())

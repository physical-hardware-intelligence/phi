"""Server: auth, control, frames, and the heartbeat path end to end with a real worker process
(mock rig). SPEC X-2, TEL-4, TEL-5."""

from __future__ import annotations

import asyncio
import json
import socket
import struct
import time
from typing import Any

import pytest

aiohttp = pytest.importorskip("aiohttp")
pytest.importorskip("PIL")
from aiohttp.test_utils import TestServer  # noqa: E402

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

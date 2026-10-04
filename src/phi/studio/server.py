"""The Studio server: static UI plus one WebSocket, bound to 127.0.0.1 (ARCHITECTURE ADR-S7, S12).

    browser  <--WebSocket-->  this process  <--Pipe-->  robot worker process

Security, because this socket can move motors:
  * binds 127.0.0.1 only
  * the WebSocket needs the per-launch token, an allowed Origin, and an allowed Host (DNS rebinding)
  * no CORS headers, so no other site's page can read responses

Evals live here, not in the worker: judging an episode moves nothing, and the record must survive a
worker crash. Every judgement is on disk before the windows hear about it.

Control: one window holds control; any window may Stop. Control changing hands stops the rig:
closing or reloading the controlling window, or another window taking control, sends Stop at
once. The worker's heartbeat timeout is the backstop for a window that hangs without closing.
"""

from __future__ import annotations

import asyncio
import json
import multiprocessing as mp
import os
import secrets
import socket
import struct
import threading
from pathlib import Path
from typing import Any

from aiohttp import WSMsgType, web

from phi.studio.evals import EvalError, EvalStore

STATIC = Path(__file__).parent / "static"
COMMANDS = {"heartbeat", "connect", "identify", "confirm", "arm", "start", "stop", "resume",
            "release", "clear", "disconnect", "inject",
            "cal_start", "cal_middle", "cal_finish", "cal_save", "cal_cancel"}  # fmt: skip
EVAL_COMMANDS = {"eval_begin", "eval_mark", "eval_undo", "eval_end"}  # answered by the server
ANYONE = {"stop", "take_control"}  # allowed from a window without control
REPLAYED = ("rig", "state", "identity", "worker_exit")  # newest of each, sent to a new window


def frame_packet(msg: dict[str, Any]) -> bytes:
    """b'F' + uint16 header length + JSON header + JPEG bytes."""
    head = json.dumps({k: msg[k] for k in ("key", "t", "seq", "w", "h")}).encode()
    return b"F" + struct.pack(">H", len(head)) + head + msg["jpeg"]


class Client:
    """One browser window. Holds the newest telemetry and the newest frame per camera, so a slow
    window drops stale data instead of queueing it."""

    def __init__(self, ws: web.WebSocketResponse) -> None:
        self.ws = ws
        self.events: list[dict[str, Any]] = []  # state, identity, errors: every one delivered
        self.telemetry: dict[str, Any] | None = None
        self.frames: dict[str, dict[str, Any]] = {}
        self.wake = asyncio.Event()

    def push(self, msg: dict[str, Any]) -> None:
        kind = msg.get("type")
        if kind == "telemetry":
            self.telemetry = msg
        elif kind == "frame":
            self.frames[msg["key"]] = msg
        else:
            self.events.append(msg)
        self.wake.set()

    async def run(self) -> None:
        while not self.ws.closed:
            await self.wake.wait()
            self.wake.clear()
            events, self.events = self.events, []
            for e in events:
                await self.ws.send_str(json.dumps(e))
            if self.telemetry is not None:
                t, self.telemetry = self.telemetry, None
                await self.ws.send_str(json.dumps(t))
            frames, self.frames = self.frames, {}
            for f in frames.values():
                await self.ws.send_bytes(frame_packet(f))


class Studio:
    def __init__(
        self,
        spec: dict[str, Any],
        port: int,
        token: str | None = None,
        data_dir: Path | str | None = None,
    ) -> None:
        self.spec = {**spec, "data_dir": str(data_dir)} if data_dir else dict(spec)
        self.port = port
        self.data_dir = Path(data_dir) if data_dir else None
        self.evals = EvalStore(self.data_dir) if self.data_dir else None
        self.token = token or secrets.token_urlsafe(24)
        self.allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        self.allowed_origins = {f"http://{h}" for h in self.allowed_hosts}
        self.clients: set[Client] = set()
        self.controller: Client | None = None
        self.last: dict[
            str, dict[str, Any]
        ] = {}  # newest state / identity, replayed to new windows
        self.loop: asyncio.AbstractEventLoop | None = None
        self.proc: Any = None
        self.conn: Any = None
        self._send_lock = threading.Lock()

    # -- worker ---------------------------------------------------------------------------------
    def start_worker(self) -> None:
        from phi.studio.worker import run_worker

        ctx = mp.get_context("spawn")
        self.conn, child = ctx.Pipe()
        self.proc = ctx.Process(target=run_worker, args=(child, self.spec), daemon=True)
        self.proc.start()
        child.close()  # WHY: otherwise recv() never sees EOF when the worker dies
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        """Worker -> windows. Runs in a thread because Pipe.recv blocks."""
        while True:
            try:
                msg = self.conn.recv()
            except (EOFError, OSError):
                code = self.proc.exitcode if self.proc else None
                msg = {"type": "worker_exit", "code": code,
                       "message": "The robot worker stopped. Restart Studio."}  # fmt: skip
                self._dispatch(msg)
                return
            self._dispatch(msg)

    def _dispatch(self, msg: dict[str, Any]) -> None:
        if msg.get("type") in REPLAYED:
            self.last[msg["type"]] = msg
        if self.loop is not None:
            self.loop.call_soon_threadsafe(self._fanout, msg)

    def _fanout(self, msg: dict[str, Any]) -> None:
        for c in list(self.clients):
            c.push(msg)

    def to_worker(self, msg: dict[str, Any]) -> None:
        with self._send_lock:
            try:
                self.conn.send(msg)
            except (BrokenPipeError, OSError):
                pass

    def stop_worker(self) -> None:
        if self.conn is not None:
            self.conn.close()
        if self.proc is not None:
            self.proc.join(3)
            if self.proc.is_alive():
                self.proc.terminate()

    # -- web --------------------------------------------------------------------------------------
    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/ws", self.ws_handler)
        app.router.add_get("/api/health", self.health)
        app.router.add_get("/", self.index)
        if (STATIC / "assets").is_dir():
            app.router.add_static("/assets", STATIC / "assets")
        app.on_startup.append(self._on_startup)
        app.on_shutdown.append(self._on_shutdown)
        return app

    async def _on_startup(self, app: web.Application) -> None:
        self.loop = asyncio.get_running_loop()
        if self.proc is None:
            self.start_worker()

    async def _on_shutdown(self, app: web.Application) -> None:
        for c in list(self.clients):
            await c.ws.close()
        self.stop_worker()

    async def health(self, request: web.Request) -> web.Response:
        return web.json_response(
            {"ok": True, "worker_alive": bool(self.proc and self.proc.is_alive())}
        )

    async def index(self, request: web.Request) -> web.StreamResponse:
        page = STATIC / "index.html"
        if page.exists():
            return web.FileResponse(page, headers={"Cache-Control": "no-store"})
        return web.Response(text="Studio UI is not built. Run: make studio-web", status=503)

    def _refuse(self, request: web.Request) -> str | None:
        if request.host not in self.allowed_hosts:
            return f"host {request.host!r} not allowed"
        if request.headers.get("Origin") not in self.allowed_origins:
            return "origin not allowed"
        # WHY bytes: compare_digest raises TypeError on non-ASCII str, which would be a 500.
        if not secrets.compare_digest(request.query.get("token", "").encode(), self.token.encode()):
            return "bad or missing token"
        return None

    async def ws_handler(self, request: web.Request) -> web.StreamResponse:
        why = self._refuse(request)
        if why:
            return web.Response(status=403, text=why)
        ws = web.WebSocketResponse(heartbeat=10)
        await ws.prepare(request)
        client = Client(ws)
        self.clients.add(client)
        if self.controller is None:
            self.controller = client
        sender = asyncio.create_task(client.run())
        client.push({"type": "hello", "control": self.controller is client,
                     "mock": self.spec.get("kind", "mock") == "mock"})  # fmt: skip
        for kind in REPLAYED:
            if kind in self.last:
                client.push(self.last[kind])
        if self.evals is not None:
            client.push(self._eval_msg())
        try:
            async for m in ws:
                if m.type == WSMsgType.TEXT:
                    self._from_client(client, m.data)
        finally:
            self.clients.discard(client)
            if self.controller is client:
                # WHY stop now, not on the heartbeat timeout: a reload reconnects and beats again
                # within 1 s, so the timeout alone would let motion run on.
                self.to_worker({"cmd": "stop", "reason": "window closed"})
                self.controller = None
                for other in self.clients:
                    other.push({"type": "control", "available": True})
            sender.cancel()
        return ws

    def _from_client(self, client: Client, data: str) -> None:
        try:
            msg = json.loads(data)
            cmd = msg["cmd"]
        except (ValueError, KeyError, TypeError):
            cmd = None
        if not isinstance(cmd, str):
            client.push({"type": "error", "message": "malformed message", "fix": ""})
            return
        if cmd == "take_control":
            if self.controller is not None and self.controller is not client:
                self.to_worker({"cmd": "stop", "reason": "control moved"})
                self.controller.push({"type": "control", "control": False})
            self.controller = client
            client.push({"type": "control", "control": True})
            return
        if cmd not in COMMANDS and cmd not in EVAL_COMMANDS:
            client.push({"type": "error", "message": f"unknown command {cmd!r}", "fix": ""})
            return
        if client is not self.controller and cmd not in ANYONE:
            if cmd != "heartbeat":
                client.push({"type": "error", "message": "Another window has control.",
                             "fix": "Take control to operate the rig here."})  # fmt: skip
            return
        if cmd in EVAL_COMMANDS:
            self._eval(client, cmd, msg)
            return
        self.to_worker({k: v for k, v in msg.items() if isinstance(k, str)})

    # -- evals ----------------------------------------------------------------------------------
    def _eval_msg(self) -> dict[str, Any]:
        assert self.evals is not None
        return {"type": "eval", "current": self.evals.current, "past": self.evals.list(20),
                "dir": str(self.evals.dir)}  # fmt: skip

    def _eval(self, client: Client, cmd: str, msg: dict[str, Any]) -> None:
        if self.evals is None:
            client.push({"type": "error", "message": "Evals need a data directory.",
                         "fix": "Start Studio with --data-dir."})  # fmt: skip
            return
        e = self.evals
        try:
            if cmd == "eval_begin":
                e.begin(msg.get("policy"), msg.get("task", ""), msg.get("planned"),
                        msg.get("limit_s"))  # fmt: skip
            elif cmd == "eval_mark":
                e.mark(msg.get("outcome"), msg.get("note", ""), msg.get("duration_s"),
                       msg.get("run_id"), msg.get("run_started_at"))  # fmt: skip
            elif cmd == "eval_undo":
                e.undo()
            else:
                e.end()
        except EvalError as err:
            client.push({"type": "error", "message": str(err), "fix": ""})
            return
        except OSError as err:  # the judgement is not saved, so do not show it as saved
            client.push({"type": "error", "message": f"Could not save the eval: {err}",
                         "fix": f"Check that {e.dir} is writable."})  # fmt: skip
            return
        self._fanout(self._eval_msg())


class PortInUse(RuntimeError):
    def __init__(self, port: int) -> None:
        super().__init__(f"Port {port} is in use, probably by another Studio.")
        self.port = port


def port_free(port: int) -> bool:
    """Whether Studio could bind 127.0.0.1:port. Same SO_REUSEADDR setting as aiohttp, so a port
    in TIME_WAIT after a recent Studio counts as free."""
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def serve(
    spec: dict[str, Any],
    port: int = 8765,
    open_browser: bool = True,
    data_dir: Path | None = None,
) -> None:
    if not port_free(port):  # before printing a URL or opening a browser at the wrong server
        raise PortInUse(port)
    # WHY an env override: a fixed token lets a dev preview reload with the same URL.
    # The default is a fresh random token per launch.
    studio = Studio(spec, port, token=os.environ.get("PHI_STUDIO_TOKEN") or None, data_dir=data_dir)
    url = f"http://127.0.0.1:{port}/#token={studio.token}"
    print(f"Phi Studio on {url}\nCtrl+C stops Studio and releases torque.", flush=True)
    if open_browser:
        import webbrowser

        threading.Timer(0.8, webbrowser.open, args=(url,)).start()
    web.run_app(studio.app(), host="127.0.0.1", port=port, print=None)

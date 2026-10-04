// Connection to the Studio server and the app state it feeds.
// One WebSocket: JSON for state, identity, telemetry and errors; binary for camera frames.
import { useRef, useSyncExternalStore } from "react";

export type Tone = "neutral" | "info" | "ok" | "warn" | "danger";
export type SessionState =
  | "DISCONNECTED" | "CONNECTED" | "IDENTIFIED" | "READY" | "CALIBRATING"
  | "ARMED" | "MOVING" | "STOPPED" | "FAULT";

export interface StateMsg {
  type: "state";
  state: SessionState;
  label: string;
  tone: Tone;
  next_action: string;
  activity: string | null;
  stop_reason: string | null;
  fault: string | null;
}

export interface ArmIdentity {
  name: string;
  role: "leader" | "follower";
  port: string;
  serial: string;
  expected: string; // the calibration file this arm is registered with
  match: string | null;
  max_deg: number | null;
  worst_joint: string | null;
  exact: boolean;
  ok: boolean; // exact, and the match is this arm's own file
}

export interface JointHealth { load: number; temp: number; volt: number; faults: string[] }
export interface ArmTelemetry {
  role: "leader" | "follower";
  online: boolean;
  torque: boolean;
  pos: Record<string, number>;
  health: Record<string, JointHealth>;
}
export interface JointCal { id: number; drive_mode: number; homing_offset: number; range_min: number; range_max: number }
export type CalStep = "middle" | "ranges" | "review";
export interface CalView {
  arm: string;
  role: "leader" | "follower";
  step: CalStep;
  joints: Record<string, { min: number; pos: number; max: number; fixed: boolean }>; // raw ticks
  old: Record<string, JointCal>;
  new: Record<string, JointCal> | null;
}
export interface PolicyView {
  id: string;
  name: string;
  task: string;
  run_id: string; // one per episode: an eval judges each run once
  started_at: number; // unix seconds; an eval only judges runs that started after it
  limit_s: number;
  episode_s: number;
  step: number;
  chunk: number;
  running: boolean;
  ended: string | null;
  chunk_ms: number;
  chunk_ms_p50: number;
  chunk_ms_max: number;
  action: Record<string, Record<string, number>>; // follower -> joint -> target
}
export interface Telemetry {
  type: "telemetry";
  t: number;
  arms: Record<string, ArmTelemetry>;
  loop: { hz: number; p50_ms: number; p99_ms: number };
  calibration: CalView | null;
  policy: PolicyView | null;
}

export interface PolicyInfo { id: string; name: string; available: boolean; note: string }
export interface RigInfo {
  mock: boolean;
  bimanual: boolean;
  // id: the LeRobot id; file: its calibration file under a LeRobot calibration root
  arms: { name: string; role: "leader" | "follower"; side: "left" | "right" | null; id: string; file: string }[];
  policies: PolicyInfo[];
  cal_dir: string | null;
}

export interface Episode {
  n: number; outcome: "success" | "failure"; note: string; duration_s: number | null; run_id: string | null; at: number;
}
export interface EvalSummary {
  id: string;
  policy: string;
  task: string;
  planned: number;
  limit_s: number;
  started_at: number;
  ended_at: number | null;
  successes: number;
  n: number;
  rate: number | null;
  ci95: [number, number];
}
export interface EvalRecord extends EvalSummary { episodes: Episode[] }
export interface EvalState { current: EvalRecord | null; past: EvalSummary[]; dir: string }

export interface StudioError { id: number; message: string; fix: string; at: number }
export interface ActivityEntry { id: number; at: number; tone: Tone; text: string; detail?: string; ask?: boolean } // ask: a problem Claude can help with

export type Link = "connecting" | "open" | "closed" | "refused" | "down"; // down: nothing answers at all

// -- assistant and files ---------------------------------------------------------------------------
export interface Focus { message: string; fix?: string } // the error a question is about
export interface AssistStatus { available: boolean; signed_in: boolean; method: string | null; model: string | null; version: string | null; error: string | null; fix: string | null }
export type Part = { kind: "text"; text: string } | { kind: "tool"; text: string };
export interface Turn {
  id: number;
  question: string;
  focus: Focus | null;
  page: string;
  parts: Part[];
  state: "waiting" | "streaming" | "done" | "stopped" | "error";
  error?: { message: string; fix: string | null };
  ms?: number | null;
}
export interface Assist {
  open: boolean;
  status: AssistStatus | null; // null: not checked yet
  turns: Turn[];
  focus: Focus | null; // attached to the next question
  context: string | null; // the last context block the server showed us
}
export interface FileRoot { key: string; label: string; path: string }
export interface FileRef { root: string; path: string; label?: string; id?: string; kind?: string; mtime: number }
export interface FilesIndex { roots: FileRoot[]; notes: FileRef[]; calibrations: FileRef[]; lerobot: LeRobotView | null }
/** robot-config.yaml as LeRobot reads it (src/phi/studio/rigspec.py). */
export interface LeRobotArm {
  key: string; role: "leader" | "follower"; side: "left" | "right" | null; type: string; id: string | null;
  port: string | null; line: number | null; calibration: string | null; calibrated: boolean;
  use_degrees: boolean; max_relative_target: number | Record<string, number> | null;
}
export interface LeRobotCommand { id: string; title: string; why: string; cmd: string }
export interface LeRobotView {
  file: { root: string; path: string }; error?: string; bimanual: boolean; arms: LeRobotArm[];
  cameras: { key: string; side: string | null; feature: string; type: unknown; source: unknown }[];
  features: string[]; problems: string[]; commands: LeRobotCommand[];
}
export interface OpenFile { root: string; path: string; abs: string; text: string; size: number; mtime: number; truncated: boolean; line: number | null }
export interface SearchHit { root: string; path: string; line: number; text: string }
export interface SearchResult { query: string; hits: SearchHit[]; scanned: number; stopped: boolean }
export interface SerialPort { device: string; tty: string | null; description: string | null; vid: number | null; pid: number | null; serial: string | null; manufacturer: string | null; usb: boolean; arm: string | null }
export interface Ports { ports: SerialPort[]; error: string | null; fix: string | null; at: number }
export interface FilesState {
  index: FilesIndex | null;
  open: OpenFile | null;
  loading: string | null; // the path being opened
  error: { message: string; path: string | null } | null;
  search: SearchResult | null;
  searching: string | null;
  ports: Ports | null;
}

// -- checks ----------------------------------------------------------------------------------------
export type CheckStatus = "pass" | "info" | "warn" | "fail" | "skip";
export interface CheckResult {
  id: string;
  group: "This Mac" | "Rig" | "Studio";
  title: string;
  status: CheckStatus;
  detail: string;
  fix: string | null;
  file: { root: string; path: string; line: number | null } | null;
}
export interface ChecksState { running: boolean; at: number | null; ms: number | null; results: CheckResult[] | null }

export interface Snapshot {
  link: Link;
  mock: boolean;
  control: boolean;
  state: StateMsg | null;
  identity: ArmIdentity[];
  telemetry: Telemetry | null;
  errors: StudioError[];
  workerExit: string | null;
  cameras: Record<string, { online: boolean; message?: string }>;
  activity: ActivityEntry[]; // newest first, last 200
  rig: RigInfo | null;
  evals: EvalState | null; // null: this Studio has no data directory
  calibrated: { arm: string; path: string | null; at: number } | null; // the last save, for the Calibrate page
  assist: Assist;
  files: FilesState;
  checks: ChecksState;
}

export interface Frame { key: string; t: number; seq: number; w: number; h: number; jpeg: Blob }

const JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"] as const;
export { JOINTS };

type Listener = () => void;

class Studio {
  private ws: WebSocket | null = null;
  private listeners = new Set<Listener>();
  private frameListeners = new Map<string, Set<(f: Frame) => void>>();
  private beat: Worker | null = null;
  private errorId = 0;
  private retry = 0;
  private retryTimer: number | null = null;
  private started = false;
  snap: Snapshot = {
    link: "connecting", mock: false, control: false, state: null, identity: [],
    telemetry: null, errors: [], workerExit: null, cameras: {}, activity: [], rig: null, evals: null,
    calibrated: null,
    assist: { open: false, status: null, turns: [], focus: null, context: null },
    files: { index: null, open: null, loading: null, error: null, search: null, searching: null, ports: null },
    checks: { running: false, at: null, ms: null, results: null },
  };
  private turnId = 0;
  private activityId = 0;

  token(): string {
    // The launch URL carries the token in the fragment, which never reaches a server log.
    // Keep it for this tab and remove it from the address bar.
    const m = location.hash.match(/token=([\w-]+)/);
    if (m) {
      sessionStorage.setItem("phi-studio-token", m[1]);
      history.replaceState(null, "", location.pathname + location.search);
    }
    return sessionStorage.getItem("phi-studio-token") ?? "";
  }

  start(): void {
    if (this.started) return;
    this.started = true;
    // A link pasted into an already-open tab only changes the hash: pick the token up from there.
    window.addEventListener("hashchange", () => {
      if (/token=/.test(location.hash) && this.snap.link !== "open") { this.retry = 0; this.ws?.close(); this.connect(); }
    });
    // Esc stops from any page and any focus, including inside dialogs: capture phase, before anything else.
    // WHY here and not in a React effect: if rendering ever crashes, React unmounts the tree and its
    // effects' listeners with it. This one lives as long as the page.
    window.addEventListener("keydown", (e) => { if (e.key === "Escape") this.stop(); }, true);
    this.connect();
  }

  stop(): void {
    if (this.send({ cmd: "stop" })) return;
    // WHY say so: a silent no-op Stop is the worst failure a stop button can have.
    this.localError(
      "Stop did not reach Studio: this window is offline",
      "If this window had control, the rig stopped when the link dropped. Otherwise stop from the window that has control, or cut the followers' power.",
    );
  }

  connect(): void {
    // WHY drop the pending retry and the old socket: a link pasted while a retry was queued used to
    // open a second socket. The first stayed the server's controller, so this window read "view only".
    if (this.retryTimer !== null) { window.clearTimeout(this.retryTimer); this.retryTimer = null; }
    const old = this.ws;
    if (old) { old.onopen = old.onmessage = old.onclose = null; old.close(); this.stopHeartbeat(); }
    const tok = this.token();
    if (!tok) { this.ws = null; this.set({ link: "refused" }); return; }
    const ws = new WebSocket(`ws://${location.host}/ws?token=${encodeURIComponent(tok)}`);
    ws.binaryType = "arraybuffer";
    this.ws = ws;
    let opened = false;
    if (this.snap.link !== "down") this.set({ link: "connecting" }); // WHY: no flicker while Studio is down
    ws.onopen = () => {
      opened = true;
      this.retry = 0;
      this.set({ link: "open" });
      this.startHeartbeat();
      if (this.snap.assist.open) this.send({ cmd: "assist_status" });
    };
    ws.onmessage = (e) => (typeof e.data === "string" ? this.onJson(JSON.parse(e.data)) : this.onFrame(e.data));
    ws.onclose = () => {
      this.stopHeartbeat();
      if (this.snap.link !== "down") this.set({ link: "closed" });
      this.set({ control: false });
      this.dropPending();
      if (!opened && this.retry >= 2) void this.diagnose(ws);
      else this.reconnectSoon();
    };
  }

  private reconnectSoon(): void {
    this.retryTimer = window.setTimeout(() => { this.retryTimer = null; this.connect(); }, Math.min(4000, 500 * 2 ** this.retry++));
  }

  /** Why the link keeps failing. A WebSocket cannot say; the health endpoint, which needs no token, can.
   * No answer: Studio is not running, so keep trying. An answer: Studio refused this window's token. */
  private async diagnose(ws: WebSocket): Promise<void> {
    let refused = false;
    try { refused = (await fetch("/api/health", { cache: "no-store" })).ok; } catch { /* nothing listening */ }
    if (this.ws !== ws) return; // a new link was opened meanwhile; it owns the retries now
    if (refused) { this.set({ link: "refused" }); return; }
    this.set({ link: "down" });
    this.reconnectSoon();
  }

  // WHY a Worker for the heartbeat clock: browsers throttle timers in hidden or occluded tabs to
  // about once a second, which would trip the 1 s motion watchdog while the operator looks away.
  // Worker timers are not throttled the same way. [Unverified on every browser: test in Stage 5.]
  private startHeartbeat(): void {
    this.stopHeartbeat();
    const src = "setInterval(() => postMessage(0), 250);";
    this.beat = new Worker(URL.createObjectURL(new Blob([src], { type: "text/javascript" })));
    this.beat.onmessage = () => this.send({ cmd: "heartbeat" });
  }
  private stopHeartbeat(): void { this.beat?.terminate(); this.beat = null; }

  /** Replies that will never come once the link drops: clear what waits for them, so no spinner sticks. */
  private dropPending(): void {
    const f = this.snap.files;
    if (this.snap.assist.status) this.setAssist({ status: null }); // re-checked on reconnect
    if (this.snap.checks.running) this.set({ checks: { ...this.snap.checks, running: false } });
    if (f.loading || f.searching) this.setFiles({ loading: null, searching: null });
    const t = this.snap.assist.turns.at(-1);
    if (t && (t.state === "waiting" || t.state === "streaming")) {
      this.lastTurn((t) => ({
        ...t, state: "error",
        error: { message: "The link to Studio dropped during the answer.", fix: "Ask again once this window reconnects." },
      }));
    }
  }

  /** False when this window has no live link, so the message went nowhere. */
  send(msg: Record<string, unknown>): boolean {
    if (this.ws?.readyState !== WebSocket.OPEN) return false;
    this.ws.send(JSON.stringify(msg));
    return true;
  }

  localError(message: string, fix: string): void {
    const e = { id: ++this.errorId, message, fix, at: Date.now() };
    this.set({ errors: [...this.snap.errors.slice(-4), e], activity: this.logged("warn", message, fix, true) });
  }

  /** A new activity entry. A danger entry is always a problem; a warn entry is one only when the caller says
   * so, because state changes such as "Confirm arms" use the warn tone for a normal next step. */
  private logged(tone: Tone, text: string, detail?: string, ask = tone === "danger"): ActivityEntry[] {
    const e = { id: ++this.activityId, at: Date.now(), tone, text, detail, ask };
    return [e, ...this.snap.activity].slice(0, 200);
  }

  dismissError(id: number): void { this.set({ errors: this.snap.errors.filter((e) => e.id !== id) }); }

  // -- assistant ---------------------------------------------------------------------------------
  private setAssist(p: Partial<Assist>): void { this.set({ assist: { ...this.snap.assist, ...p } }); }
  private setFiles(p: Partial<FilesState>): void { this.set({ files: { ...this.snap.files, ...p } }); }
  private lastTurn(fn: (t: Turn) => Turn): void {
    const turns = this.snap.assist.turns;
    if (turns.length) this.setAssist({ turns: [...turns.slice(0, -1), fn(turns[turns.length - 1])] });
  }

  /** Open the panel, optionally about one error. Checks Claude's sign-in the first time. */
  openAssistant(focus?: Focus): void {
    this.setAssist({ open: true, focus: focus ?? this.snap.assist.focus });
    if (!this.snap.assist.status) this.send({ cmd: "assist_status" });
  }
  closeAssistant(): void { this.setAssist({ open: false }); }
  toggleAssistant(): void { if (this.snap.assist.open) this.closeAssistant(); else this.openAssistant(); }
  clearFocus(): void { this.setAssist({ focus: null }); }
  checkAssistant(): void { this.setAssist({ status: null }); this.send({ cmd: "assist_status" }); }
  showContext(): void { this.send({ cmd: "assist_context", page: readRoute(), focus: this.snap.assist.focus }); }

  ask(question: string): boolean {
    const a = this.snap.assist;
    const busy = a.turns.some((t) => t.state === "waiting" || t.state === "streaming");
    if (busy || !question.trim()) return false;
    const page = readRoute();
    if (!this.send({ cmd: "assist_ask", text: question, page, focus: a.focus })) {
      this.localError("Claude cannot be asked while this window is offline", "Wait for it to reconnect.");
      return false;
    }
    const turn: Turn = { id: ++this.turnId, question, focus: a.focus, page, parts: [], state: "waiting" };
    this.setAssist({ turns: [...a.turns, turn], focus: null, context: null });
    return true;
  }
  stopAnswer(): void { this.send({ cmd: "assist_stop" }); }
  newChat(): void { this.send({ cmd: "assist_reset" }); this.setAssist({ turns: [], focus: null, context: null }); }

  private onAssist(m: any): void {
    switch (m.kind) {
      case "start": this.lastTurn((t) => ({ ...t, state: "streaming" })); break;
      case "delta":
        this.lastTurn((t) => {
          const parts = [...t.parts];
          const end = parts[parts.length - 1];
          if (end?.kind === "text") parts[parts.length - 1] = { kind: "text", text: end.text + m.text };
          else parts.push({ kind: "text", text: m.text });
          return { ...t, parts, state: "streaming" };
        });
        break;
      case "tool": this.lastTurn((t) => ({ ...t, parts: [...t.parts, { kind: "tool", text: m.text }] })); break;
      case "done": this.lastTurn((t) => ({ ...t, state: m.stopped ? "stopped" : "done", ms: m.duration_ms })); break;
      case "error":
        if (/not signed in|not installed|too old/i.test(m.message)) this.send({ cmd: "assist_status" });
        // An echoed error streamed in as answer text first; the error box says it once.
        this.lastTurn((t) => ({
          ...t, state: "error", error: { message: m.message, fix: m.fix },
          parts: m.echoed ? t.parts.filter((p) => p.kind !== "text") : t.parts,
        }));
        break;
      case "limit": {
        // WHY the 1e12 test: whether resetsAt is in seconds or milliseconds is not documented; both are handled.
        const at = typeof m.resets_at === "number" ? new Date(m.resets_at > 1e12 ? m.resets_at : m.resets_at * 1000) : null;
        const when = at ? ` It resets at ${at.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", hour12: false })}.` : "";
        const what = m.status === "allowed_warning" ? "Close to the Claude usage limit."
          : m.status === "rejected" ? "The Claude usage limit is reached." : `Claude usage: ${m.status}.`;
        this.lastTurn((t) => ({ ...t, parts: [...t.parts, { kind: "tool", text: what + when }] }));
        break;
      }
    }
  }

  // -- checks ------------------------------------------------------------------------------------
  /** Run every pre-flight check. Read-only, so any window may. */
  runChecks(): void {
    if (this.snap.checks.running) return;
    if (!this.send({ cmd: "checks_run" })) {
      this.localError("Checks cannot run while this window is offline", "Wait for it to reconnect.");
      return;
    }
    this.set({ checks: { ...this.snap.checks, running: true } });
  }

  // -- files -------------------------------------------------------------------------------------
  loadFiles(): void { this.send({ cmd: "files_index" }); this.refreshPorts(); }
  refreshPorts(): void { this.send({ cmd: "ports" }); }
  /** Open a file in the Files page. `root` omitted: the server finds which root holds `path`. */
  // WHY check send(): a request that never left would leave its spinner, and the disabled Search button, forever.
  openFile(path: string, line?: number | null, root?: string): void {
    go("files");
    if (!this.send({ cmd: "file_read", path, line: line ?? null, ...(root ? { root } : {}) })) {
      this.localError("Files cannot open while this window is offline", "Wait for it to reconnect.");
      return;
    }
    this.setFiles({ loading: path, error: null });
  }
  search(query: string): void {
    if (!this.send({ cmd: "files_search", query })) {
      this.localError("Search cannot run while this window is offline", "Wait for it to reconnect.");
      return;
    }
    this.setFiles({ searching: query, error: null });
  }

  private onJson(m: any): void {
    switch (m.type) {
      // WHY clear these: the server replays its own identity and worker_exit right after hello, and
      // telemetry and camera status stream again within a tick, so a value kept from an earlier server
      // (a restarted Studio) would otherwise stay on screen for good. Checks keep their own timestamp.
      case "hello":
        this.set({ mock: m.mock, control: m.control, workerExit: null, identity: [], telemetry: null, cameras: {} });
        break;
      case "control":
        if ("control" in m) this.set({ control: m.control });
        break;
      case "state": {
        const prev = this.snap.state;
        const changed = !prev || prev.state !== m.state || prev.fault !== m.fault;
        const why = m.state === "FAULT" ? m.fault : m.state === "STOPPED" ? `reason: ${m.stop_reason}` : m.activity ?? undefined;
        this.set({ state: m, activity: changed ? this.logged(m.tone, m.label, why ?? undefined) : this.snap.activity });
        break;
      }
      case "identity": {
        const bad = (m.arms as ArmIdentity[]).filter((a) => !a.ok).length;
        this.set({
          identity: m.arms,
          activity: this.logged(bad ? "warn" : "ok", `Read ${m.arms.length} arms`, bad ? `${bad} need attention` : "every arm matches its own calibration"),
        });
        break;
      }
      case "telemetry": this.set({ telemetry: m }); break;
      case "rig": this.set({ rig: { mock: m.mock, bimanual: !!m.bimanual, arms: m.arms, policies: m.policies, cal_dir: m.cal_dir } }); break;
      case "eval": this.set({ evals: { current: m.current, past: m.past, dir: m.dir } }); break;
      case "calibrated":
        this.set({
          calibrated: { arm: m.arm, path: m.path, at: Date.now() },
          activity: this.logged("ok", `Calibrated ${m.arm}`, m.path ? `saved to ${m.path}` : "registers written; no file saved"),
        });
        break;
      case "camera": this.set({ cameras: { ...this.snap.cameras, [m.key]: { online: m.online, message: m.message } } }); break;
      case "worker_exit": this.set({ workerExit: m.message, activity: this.logged("danger", m.message) }); break;
      case "error": this.localError(m.message, m.fix ?? ""); break;
      case "assist": this.onAssist(m); break;
      case "assist_status": this.setAssist({ status: m }); break;
      case "assist_context": this.setAssist({ context: m.text }); break;
      case "files": this.setFiles({ index: { roots: m.roots, notes: m.notes, calibrations: m.calibrations, lerobot: m.lerobot ?? null } }); break;
      case "file": this.setFiles({ open: { root: m.root, path: m.path, abs: m.abs, text: m.text, size: m.size, mtime: m.mtime, truncated: m.truncated, line: m.line }, loading: null, error: null }); break;
      case "file_error": // the server says which request failed; a search and an open can be in flight together
        this.setFiles({ ...(m.op === "search" ? { searching: null } : { loading: null }), error: { message: m.message, path: m.path } });
        break;
      case "search": this.setFiles({ search: m, searching: null }); break;
      case "ports": this.setFiles({ ports: { ...m, at: Date.now() } }); break;
      case "checks": {
        if (m.error) {
          this.set({ checks: { ...this.snap.checks, running: false } });
          this.localError(m.error, "Run them again. If it repeats, ask Claude: the error is in Studio's log.");
          break;
        }
        const bad = (m.results as CheckResult[]).filter((r) => r.status === "fail").length;
        const warn = (m.results as CheckResult[]).filter((r) => r.status === "warn").length;
        this.set({
          checks: { running: false, at: m.at * 1000, ms: m.ms, results: m.results },
          activity: this.logged(bad ? "danger" : warn ? "warn" : "ok", "Ran checks",
            [bad && `${bad} failed`, warn && `${warn} ${warn === 1 ? "warning" : "warnings"}`].filter(Boolean).join(", ")
              || "everything Studio can check passed", Boolean(bad || warn)),
        });
        break;
      }
    }
  }

  private onFrame(buf: ArrayBuffer): void {
    const v = new DataView(buf);
    if (v.getUint8(0) !== 0x46) return; // "F"
    const n = v.getUint16(1);
    const head = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 3, n)));
    const frame: Frame = { ...head, jpeg: new Blob([new Uint8Array(buf, 3 + n)], { type: "image/jpeg" }) };
    if (!this.snap.cameras[frame.key]?.online) {
      this.set({ cameras: { ...this.snap.cameras, [frame.key]: { online: true } } });
    }
    this.frameListeners.get(frame.key)?.forEach((f) => f(frame));
  }

  onFrames(key: string, fn: (f: Frame) => void): () => void {
    if (!this.frameListeners.has(key)) this.frameListeners.set(key, new Set());
    this.frameListeners.get(key)!.add(fn);
    return () => this.frameListeners.get(key)!.delete(fn);
  }

  private set(p: Partial<Snapshot>): void {
    this.snap = { ...this.snap, ...p };
    this.listeners.forEach((l) => l());
  }

  subscribe = (l: Listener): (() => void) => {
    this.listeners.add(l);
    return () => this.listeners.delete(l);
  };
}

export const studio = new Studio();

function shallowEqual(a: unknown, b: unknown): boolean {
  if (Object.is(a, b)) return true;
  if (typeof a !== "object" || typeof b !== "object" || a === null || b === null || Array.isArray(a) !== Array.isArray(b)) return false;
  const ka = Object.keys(a), kb = Object.keys(b);
  return ka.length === kb.length && ka.every((k) => Object.is((a as Record<string, unknown>)[k], (b as Record<string, unknown>)[k]));
}

// WHY keep the last value when the new one is shallow-equal: useSyncExternalStore treats every new
// object as a change. A selector such as `s.rig?.policies ?? []` returns a fresh [] while rig is null,
// so React re-rendered until it threw #185 (maximum update depth) on every reload of Run policy.
export function useStudio<T>(select: (s: Snapshot) => T): T {
  const last = useRef<{ value: T } | null>(null);
  return useSyncExternalStore(studio.subscribe, () => {
    const value = select(studio.snap);
    if (last.current && shallowEqual(last.current.value, value)) return last.current.value;
    last.current = { value };
    return value;
  });
}

// -- routing and theme -------------------------------------------------------------------------
export type Route = "overview" | "checks" | "calibrate" | "teleop" | "policy" | "evaluate" | "setup" | "files" | "guide";
export const ROUTES: Route[] = ["overview", "checks", "calibrate", "teleop", "policy", "evaluate", "setup", "files", "guide"];

// A route is the hash's first segment; the guide also takes a section, as in #/guide/teleop.
function hashParts(): string[] { return location.hash.replace(/^#\/?/, "").split("/"); }

function readRoute(): Route {
  const r = hashParts()[0] as Route;
  return ROUTES.includes(r) ? r : "overview";
}

export function useSection(): string | null {
  return useSyncExternalStore(
    (l) => { window.addEventListener("hashchange", l); return () => window.removeEventListener("hashchange", l); },
    () => hashParts()[1] ?? null,
  );
}

export function useRoute(): Route {
  return useSyncExternalStore(
    (l) => { window.addEventListener("hashchange", l); return () => window.removeEventListener("hashchange", l); },
    readRoute,
  );
}

export function go(r: Route, section?: string): void { location.hash = section ? `#/${r}/${section}` : `#/${r}`; }

export type Theme = "light" | "dark";
const THEME_KEY = "phi-studio-theme";
const themeListeners = new Set<() => void>();

export function getTheme(): Theme {
  try { return localStorage.getItem(THEME_KEY) === "dark" ? "dark" : "light"; } catch { return "light"; }
}

export function setTheme(t: Theme): void {
  try { localStorage.setItem(THEME_KEY, t); } catch { /* private window: theme lasts this page only */ }
  document.documentElement.dataset.theme = t;
  themeListeners.forEach((l) => l());
}

export function useTheme(): Theme {
  return useSyncExternalStore((l) => { themeListeners.add(l); return () => themeListeners.delete(l); }, () => document.documentElement.dataset.theme === "dark" ? "dark" : "light");
}

/** Wilson score interval, the same formula the server uses (phi/studio/evals.py). */
export function wilson(k: number, n: number, z = 1.96): [number, number] {
  if (n === 0) return [0, 1];
  const p = k / n;
  const d = 1 + (z * z) / n;
  const c = (p + (z * z) / (2 * n)) / d;
  const h = (z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n))) / d;
  return [Math.max(0, c - h), Math.min(1, c + h)];
}

export const pct = (x: number) => `${Math.round(x * 100)}%`;
export const TICKS_PER_DEG = 4096 / 360;

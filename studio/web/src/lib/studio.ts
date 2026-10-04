// Connection to the Studio server and the app state it feeds.
// One WebSocket: JSON for state, identity, telemetry and errors; binary for camera frames.
import { useSyncExternalStore } from "react";

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
  arms: { name: string; role: "leader" | "follower" }[];
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
export interface ActivityEntry { id: number; at: number; tone: Tone; text: string; detail?: string }

export type Link = "connecting" | "open" | "closed" | "refused";

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
  snap: Snapshot = {
    link: "connecting", mock: false, control: false, state: null, identity: [],
    telemetry: null, errors: [], workerExit: null, cameras: {}, activity: [], rig: null, evals: null,
    calibrated: null,
  };
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
    // A link pasted into an already-open tab only changes the hash: pick the token up from there.
    window.addEventListener("hashchange", () => {
      if (/token=/.test(location.hash) && this.snap.link !== "open") { this.retry = 0; this.ws?.close(); this.connect(); }
    });
    this.connect();
  }

  connect(): void {
    const tok = this.token();
    if (!tok) { this.set({ link: "refused" }); return; }
    const ws = new WebSocket(`ws://${location.host}/ws?token=${encodeURIComponent(tok)}`);
    ws.binaryType = "arraybuffer";
    this.ws = ws;
    this.set({ link: "connecting" });
    ws.onopen = () => { this.retry = 0; this.set({ link: "open" }); this.startHeartbeat(); };
    ws.onmessage = (e) => (typeof e.data === "string" ? this.onJson(JSON.parse(e.data)) : this.onFrame(e.data));
    ws.onclose = (e) => {
      this.stopHeartbeat();
      // 1006 right after open usually means the handshake was refused (bad or stale token).
      const refused = this.snap.link === "connecting" && this.retry >= 2;
      this.set({ link: refused ? "refused" : "closed", control: false });
      if (!refused) setTimeout(() => this.connect(), Math.min(4000, 500 * 2 ** this.retry++));
      void e;
    };
  }

  // WHY a Worker for the heartbeat clock: browsers throttle timers in hidden or occluded tabs to
  // about once a second, which would trip the 1 s motion watchdog while the operator looks away.
  // Worker timers are not throttled the same way. [Unverified on every browser: test in Stage 5.]
  private startHeartbeat(): void {
    const src = "setInterval(() => postMessage(0), 250);";
    this.beat = new Worker(URL.createObjectURL(new Blob([src], { type: "text/javascript" })));
    this.beat.onmessage = () => this.send({ cmd: "heartbeat" });
  }
  private stopHeartbeat(): void { this.beat?.terminate(); this.beat = null; }

  /** False when this window has no live link, so the message went nowhere. */
  send(msg: Record<string, unknown>): boolean {
    if (this.ws?.readyState !== WebSocket.OPEN) return false;
    this.ws.send(JSON.stringify(msg));
    return true;
  }

  localError(message: string, fix: string): void {
    const e = { id: ++this.errorId, message, fix, at: Date.now() };
    this.set({ errors: [...this.snap.errors.slice(-4), e], activity: this.logged("warn", message, fix) });
  }

  private logged(tone: Tone, text: string, detail?: string): ActivityEntry[] {
    const e = { id: ++this.activityId, at: Date.now(), tone, text, detail };
    return [e, ...this.snap.activity].slice(0, 200);
  }

  dismissError(id: number): void { this.set({ errors: this.snap.errors.filter((e) => e.id !== id) }); }

  private onJson(m: any): void {
    switch (m.type) {
      case "hello": this.set({ mock: m.mock, control: m.control }); break;
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
      case "rig": this.set({ rig: { mock: m.mock, arms: m.arms, policies: m.policies, cal_dir: m.cal_dir } }); break;
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

export function useStudio<T>(select: (s: Snapshot) => T): T {
  return useSyncExternalStore(studio.subscribe, () => select(studio.snap));
}

// -- routing and theme -------------------------------------------------------------------------
export type Route = "overview" | "calibrate" | "teleop" | "policy" | "evaluate";
export const ROUTES: Route[] = ["overview", "calibrate", "teleop", "policy", "evaluate"];

function readRoute(): Route {
  const r = location.hash.replace(/^#\/?/, "") as Route;
  return ROUTES.includes(r) ? r : "overview";
}

export function useRoute(): Route {
  return useSyncExternalStore(
    (l) => { window.addEventListener("hashchange", l); return () => window.removeEventListener("hashchange", l); },
    readRoute,
  );
}

export function go(r: Route): void { location.hash = `#/${r}`; }

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

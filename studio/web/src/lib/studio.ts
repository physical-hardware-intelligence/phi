// Connection to the Studio server and the app state it feeds.
// One WebSocket: JSON for state, identity, telemetry and errors; binary for camera frames.
import { useSyncExternalStore } from "react";

export type Tone = "neutral" | "info" | "ok" | "warn" | "danger";
export type SessionState =
  | "DISCONNECTED" | "CONNECTED" | "IDENTIFIED" | "READY"
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
export interface Telemetry {
  type: "telemetry";
  t: number;
  arms: Record<string, ArmTelemetry>;
  loop: { hz: number; p50_ms: number; p99_ms: number };
}

export interface StudioError { id: number; message: string; fix: string; at: number }

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
    telemetry: null, errors: [], workerExit: null, cameras: {},
  };

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
    this.set({ errors: [...this.snap.errors.slice(-4), e] });
  }

  dismissError(id: number): void { this.set({ errors: this.snap.errors.filter((e) => e.id !== id) }); }

  private onJson(m: any): void {
    switch (m.type) {
      case "hello": this.set({ mock: m.mock, control: m.control }); break;
      case "control":
        if ("control" in m) this.set({ control: m.control });
        break;
      case "state": this.set({ state: m }); break;
      case "identity": this.set({ identity: m.arms }); break;
      case "telemetry": this.set({ telemetry: m }); break;
      case "camera": this.set({ cameras: { ...this.snap.cameras, [m.key]: { online: m.online, message: m.message } } }); break;
      case "worker_exit": this.set({ workerExit: m.message }); break;
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

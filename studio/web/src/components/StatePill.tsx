import { Circle, CircleOff, Hand, LoaderCircle, OctagonAlert, Plug, Ruler, Square, TriangleAlert } from "lucide-react";
import type { Link, SessionState, StateMsg } from "../lib/studio";

// Shape as well as colour, so state never depends on colour alone (WCAG 1.4.1).
const SHAPE: Record<SessionState, typeof Circle> = {
  DISCONNECTED: Plug,
  CONNECTED: LoaderCircle,
  IDENTIFIED: TriangleAlert,
  READY: Hand,
  CALIBRATING: Ruler,
  ARMED: Circle,
  MOVING: Circle,
  STOPPED: Square,
  FAULT: OctagonAlert,
};

const ACTIVITY: Record<string, string> = { teleop: "teleop", policy: "policy", replay: "replay", calibration: "calibration" };

// Without a live link the rig state is unknown; say why instead of showing the last one.
const NO_LINK: Record<Exclude<Link, "open">, { label: string; tone: string; spin: boolean }> = {
  connecting: { label: "Connecting", tone: "neutral", spin: true },
  closed: { label: "Reconnecting", tone: "warn", spin: true },
  down: { label: "Studio not running", tone: "danger", spin: false },
  refused: { label: "Not connected", tone: "danger", spin: false },
};

export function StatePill({ s, link }: { s: StateMsg | null; link: Link }) {
  if (link !== "open" || !s) {
    const n = link === "open" ? { label: "Starting", tone: "neutral", spin: true } : NO_LINK[link];
    return (
      <span className={`state-pill tone-${n.tone}`} role="status">
        {n.spin ? <LoaderCircle className="spin" aria-hidden /> : <CircleOff aria-hidden />}
        <span>{n.label}</span>
      </span>
    );
  }
  const Icon = SHAPE[s.state];
  const moving = s.state === "MOVING";
  return (
    <span className={`state-pill tone-${s.tone}${moving ? " is-moving" : ""}`} role="status" aria-live="polite">
      <Icon className={s.state === "CONNECTED" ? "spin" : undefined} fill={moving ? "currentColor" : "none"} aria-hidden />
      <span>{s.label}</span>
      {moving && s.activity && <span className="state-pill-sub">{ACTIVITY[s.activity] ?? s.activity}</span>}
    </span>
  );
}

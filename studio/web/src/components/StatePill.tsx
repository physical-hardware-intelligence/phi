import { Circle, Hand, LoaderCircle, OctagonAlert, Plug, Square, TriangleAlert } from "lucide-react";
import type { SessionState, StateMsg } from "../lib/studio";

// Shape as well as colour, so state never depends on colour alone (WCAG 1.4.1).
const SHAPE: Record<SessionState, typeof Circle> = {
  DISCONNECTED: Plug,
  CONNECTED: LoaderCircle,
  IDENTIFIED: TriangleAlert,
  READY: Hand,
  ARMED: Circle,
  MOVING: Circle,
  STOPPED: Square,
  FAULT: OctagonAlert,
};

const ACTIVITY: Record<string, string> = { teleop: "teleop", policy: "policy", replay: "replay" };

export function StatePill({ s }: { s: StateMsg | null }) {
  if (!s) {
    return (
      <span className="state-pill tone-neutral">
        <LoaderCircle className="spin" aria-hidden />
        <span>Starting</span>
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

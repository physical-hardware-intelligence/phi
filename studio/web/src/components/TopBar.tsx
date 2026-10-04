import * as Tooltip from "@radix-ui/react-tooltip";
import { OctagonX } from "lucide-react";
import { studio, useStudio } from "../lib/studio";
import { StatePill } from "./StatePill";

// The page header. Left: where you are. Right, identical on every page: session state, loop rate,
// control, and Stop (Franka and UR keep state and stop in one fixed place).
export function TopBar({ title, sub, onStop }: { title: string; sub: string; onStop: () => void }) {
  const state = useStudio((s) => s.state);
  const loop = useStudio((s) => s.telemetry?.loop);
  const control = useStudio((s) => s.control);
  const link = useStudio((s) => s.link);
  const anyTorque = useStudio((s) => Object.values(s.telemetry?.arms ?? {}).some((a) => a.torque));
  // Stop is never disabled while any servo holds torque.
  const live = anyTorque || state?.state === "ARMED" || state?.state === "MOVING";

  return (
    <header className="topbar">
      <div className="topbar-title">
        <h1 className="t-page">{title}</h1>
        <p className="topbar-sub">{sub}</p>
      </div>
      <div className="topbar-right">
        {loop && state && state.state !== "DISCONNECTED" && (
          <Tooltip.Root>
            <Tooltip.Trigger asChild>
              <span className={`loop num ${loop.hz < 27 ? "is-slow" : ""}`} tabIndex={0}>
                <span className="loop-hz">{loop.hz.toFixed(1)} Hz</span>
                <span className="loop-p99">p99 {loop.p99_ms.toFixed(1)} ms</span>
              </span>
            </Tooltip.Trigger>
            <Tooltip.Portal>
              <Tooltip.Content className="tooltip" sideOffset={8}>
                Control loop rate, and the 99th-percentile time one cycle takes. Target 30 Hz.
              </Tooltip.Content>
            </Tooltip.Portal>
          </Tooltip.Root>
        )}
        <StatePill s={link === "open" ? state : null} />
        {link === "open" && !control && (
          <button className="btn btn-sm" onClick={() => studio.send({ cmd: "take_control" })}>Take control</button>
        )}
        <Tooltip.Root>
          <Tooltip.Trigger asChild>
            <button className={`btn stop-btn ${live ? "is-live" : ""}`} onClick={onStop} disabled={!live} aria-keyshortcuts="Escape">
              <OctagonX aria-hidden /> Stop <span className="kbd">Esc</span>
            </button>
          </Tooltip.Trigger>
          <Tooltip.Portal>
            <Tooltip.Content className="tooltip" sideOffset={8}>
              Holds every follower where it is. Torque stays on. This is a software stop; the physical cut is
              the followers' power.
            </Tooltip.Content>
          </Tooltip.Portal>
        </Tooltip.Root>
      </div>
    </header>
  );
}

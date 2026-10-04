import * as Tooltip from "@radix-ui/react-tooltip";
import { studio, useStudio } from "../lib/studio";
import { StatePill } from "./StatePill";

export function TopBar({ onStop }: { onStop: () => void }) {
  const state = useStudio((s) => s.state);
  const loop = useStudio((s) => s.telemetry?.loop);
  const control = useStudio((s) => s.control);
  const mock = useStudio((s) => s.mock);
  const link = useStudio((s) => s.link);
  const anyTorque = useStudio((s) => Object.values(s.telemetry?.arms ?? {}).some((a) => a.torque));
  // Stop is never disabled while any servo holds torque.
  const canMove = anyTorque || state?.state === "ARMED" || state?.state === "MOVING";

  return (
    <header className="topbar">
      <div className="topbar-left">
        <span className="wordmark">Phi Studio</span>
        {mock && <span className="badge tone-info">Mock rig</span>}
      </div>

      <div className="topbar-center">
        <StatePill s={link === "open" ? state : null} />
      </div>

      <div className="topbar-right">
        {loop && state && state.state !== "DISCONNECTED" && (
          <Tooltip.Root>
            <Tooltip.Trigger asChild>
              <span className={`loop num t-xs ${loop.hz < 27 ? "is-slow" : ""}`} tabIndex={0}>
                {loop.hz.toFixed(1)} Hz
                <span className="faint">p99 {loop.p99_ms.toFixed(1)} ms</span>
              </span>
            </Tooltip.Trigger>
            <Tooltip.Portal>
              <Tooltip.Content className="tooltip" sideOffset={6}>
                Control loop rate, and the 99th-percentile time one cycle takes. Target 30 Hz.
              </Tooltip.Content>
            </Tooltip.Portal>
          </Tooltip.Root>
        )}
        {link === "open" && !control && (
          <button className="btn btn-sm" onClick={() => studio.send({ cmd: "take_control" })}>
            Take control
          </button>
        )}
        <Tooltip.Root>
          <Tooltip.Trigger asChild>
            <button
              className={`btn stop-btn ${canMove ? "is-live" : ""}`}
              onClick={onStop}
              disabled={!canMove}
              aria-keyshortcuts="Escape"
            >
              Stop <span className="kbd">Esc</span>
            </button>
          </Tooltip.Trigger>
          <Tooltip.Portal>
            <Tooltip.Content className="tooltip" sideOffset={6}>
              Holds every follower where it is. Torque stays on. This is a software stop; the
              physical cut is the follower's power or USB cable.
            </Tooltip.Content>
          </Tooltip.Portal>
        </Tooltip.Root>
      </div>
    </header>
  );
}

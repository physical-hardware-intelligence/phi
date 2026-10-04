import * as Dialog from "@radix-ui/react-dialog";
import { Play, Power, PowerOff, RotateCcw } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { studio, useStudio } from "../lib/studio";

const COUNTDOWN_S = 3;

// The named actions that move the rig forward, one primary at a time, following the session
// state machine. Motion starts only from "Start teleop", after a cancellable countdown.
export function ActionBar() {
  const s = useStudio((x) => x.state);
  const control = useStudio((x) => x.control);
  const identity = useStudio((x) => x.identity);
  const anyTorque = useStudio((x) => Object.values(x.telemetry?.arms ?? {}).some((a) => a.torque));
  const [count, setCount] = useState<number | null>(null);
  const [torqueOff, setTorqueOff] = useState(false);
  const timer = useRef<number | null>(null);

  const cancel = () => {
    if (timer.current) window.clearInterval(timer.current);
    timer.current = null;
    setCount(null);
  };
  const startCountdown = () => {
    setCount(COUNTDOWN_S);
    timer.current = window.setInterval(() => {
      setCount((c) => {
        if (c === null) return null;
        if (c <= 1) {
          cancel();
          studio.send({ cmd: "start", activity: "teleop" });
          return null;
        }
        return c - 1;
      });
    }, 1000);
  };
  // Esc cancels a countdown too, and any state change away from Holding cancels it.
  useEffect(() => { if (s?.state !== "ARMED") cancel(); }, [s?.state]);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") cancel(); };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, []);

  if (!s) return null;
  const send = (cmd: string) => () => studio.send({ cmd });
  const off = !control;
  const mismatched = identity.some((a) => !a.exact);

  return (
    <div className="actionbar">
      <div className="actionbar-text">
        <span className="t-md">
          {s.state === "FAULT"
            ? anyTorque ? "Followers are holding position. Fix the cause, then clear." : "Torque is off. Fix the cause, then clear."
            : s.state === "MOVING" ? "Teleop is running. Each leader drives its follower." : s.next_action}
        </span>
      </div>
      <div className="actionbar-buttons">
        {s.state === "DISCONNECTED" && (
          <button className="btn btn-primary" onClick={send("connect")} disabled={off}>
            <Power aria-hidden /> Connect
          </button>
        )}
        {s.state === "IDENTIFIED" && (
          <>
            <button className="btn" onClick={send("identify")} disabled={off}>
              <RotateCcw aria-hidden /> Read again
            </button>
            <button className="btn btn-primary" onClick={send("confirm")} disabled={off || mismatched}
              title={mismatched ? "Every arm must match its calibration file exactly" : undefined}>
              Confirm {identity.length} arms
            </button>
          </>
        )}
        {s.state === "READY" && (
          <button className="btn btn-primary" onClick={send("arm")} disabled={off}>
            <Power aria-hidden /> Enable torque
          </button>
        )}
        {s.state === "ARMED" && (
          count === null ? (
            <>
              <button className="btn" onClick={() => setTorqueOff(true)} disabled={off}>
                <PowerOff aria-hidden /> Torque off
              </button>
              <button className="btn btn-primary" onClick={startCountdown} disabled={off}>
                <Play aria-hidden /> Start teleop
              </button>
            </>
          ) : (
            <div className="countdown" role="timer" aria-live="assertive">
              <span className="t-md">Teleop starts in <b className="num">{count}</b></span>
              <button className="btn" onClick={cancel}>Cancel <span className="kbd">Esc</span></button>
            </div>
          )
        )}
        {s.state === "STOPPED" && (
          <>
            <button className="btn" onClick={() => setTorqueOff(true)} disabled={off}>
              <PowerOff aria-hidden /> Torque off
            </button>
            <button className="btn btn-primary" onClick={send("resume")} disabled={off}>
              Resume
            </button>
          </>
        )}
        {s.state === "FAULT" && anyTorque && (
          <button className="btn" onClick={() => setTorqueOff(true)} disabled={off}>
            <PowerOff aria-hidden /> Torque off
          </button>
        )}
        {s.state === "FAULT" && (
          <button className="btn btn-primary" onClick={send("clear")} disabled={off}>
            <RotateCcw aria-hidden /> Clear and re-check arms
          </button>
        )}
        {s.state !== "DISCONNECTED" && s.state !== "MOVING" && (
          <button className="btn btn-ghost" onClick={send("disconnect")} disabled={off}>
            Disconnect
          </button>
        )}
      </div>

      <Dialog.Root open={torqueOff} onOpenChange={setTorqueOff}>
        <Dialog.Portal>
          <Dialog.Overlay className="dialog-overlay" />
          <Dialog.Content className="dialog" aria-describedby="torque-off-desc">
            <Dialog.Title className="t-lg">Turn torque off?</Dialog.Title>
            <p id="torque-off-desc" className="t-md muted">
              The followers go limp and fall under their own weight. Support each follower by hand,
              or lower it to its rest pose first.
            </p>
            <div className="dialog-buttons">
              <Dialog.Close asChild>
                <button className="btn" autoFocus>Keep torque on</button>
              </Dialog.Close>
              <button className="btn btn-danger" onClick={() => { studio.send({ cmd: "release" }); setTorqueOff(false); }}>
                Turn torque off
              </button>
            </div>
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>
    </div>
  );
}

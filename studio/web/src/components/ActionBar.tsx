import * as Dialog from "@radix-ui/react-dialog";
import { Play, Power, PowerOff, RotateCcw, Unplug } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { go, studio, useStudio } from "../lib/studio";

const COUNTDOWN_S = 3;

interface Props {
  activity: "teleop" | "policy";
  startLabel?: string;
  startMsg?: Record<string, unknown>; // extra fields for the start command (policy config)
  canStart?: boolean;
  startBlocked?: string; // why Start is disabled, shown as its title
}

// One primary action at a time, following the session state machine. Motion starts only after a
// cancellable countdown; both torque-off actions ask first, because the followers drop.
export function ActionBar({ activity, startLabel, startMsg, canStart = true, startBlocked }: Props) {
  const s = useStudio((x) => x.state);
  const control = useStudio((x) => x.control);
  const identity = useStudio((x) => x.identity);
  const anyTorque = useStudio((x) => Object.values(x.telemetry?.arms ?? {}).some((a) => a.torque));
  const [count, setCount] = useState<number | null>(null);
  const [confirmOff, setConfirmOff] = useState<null | "release" | "disconnect">(null);
  const start = useRef(() => {});
  start.current = () => studio.send({ cmd: "start", activity, ...startMsg });

  const cancel = () => setCount(null);
  const startCountdown = () => setCount(COUNTDOWN_S);
  // One timeout per tick, cleared by the effect cleanup. WHY: leaving the page unmounts this bar,
  // and a countdown that outlived it would start motion on a page that never showed it.
  useEffect(() => {
    if (count === null) return;
    const t = window.setTimeout(() => {
      if (count > 1) { setCount(count - 1); return; }
      setCount(null);
      start.current();
    }, 1000);
    return () => window.clearTimeout(t);
  }, [count]);
  // Any state change away from Holding cancels a countdown, and so does Esc.
  useEffect(() => { if (s?.state !== "ARMED") cancel(); }, [s?.state]);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") cancel(); };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, []);

  if (!s) return null;
  const send = (cmd: string) => () => studio.send({ cmd });
  const off = !control;
  const mismatched = identity.some((a) => !a.ok);
  const mine = s.activity === activity;
  const other = s.state === "MOVING" && !mine;
  const disconnect = () => (anyTorque ? setConfirmOff("disconnect") : studio.send({ cmd: "disconnect" }));
  const verb = activity === "teleop" ? "Teleop" : "The policy";

  const message =
    s.state === "FAULT" ? (anyTorque ? "Followers are holding position. Fix the cause, turn torque off, then clear." : "Torque is off. Fix the cause, then clear.")
    : s.state === "MOVING" ? (mine ? `${verb} is running.` : `${s.activity} is running on another page.`)
    : s.state === "IDENTIFIED" && mismatched ? "Some arms do not match their own calibration file."
    : s.state === "CALIBRATING" ? "Calibration is in progress."
    : s.state === "ARMED" ? `Followers are holding. ${startLabel ?? "Start teleop"} when ready.`
    : s.next_action;

  return (
    <div className="sessionbar panel" data-state={s.state}>
      <div className="sessionbar-text">
        <span className="sessionbar-msg">{message}</span>
        {!control && <span className="sessionbar-note">View only: another window has control.</span>}
      </div>
      <div className="sessionbar-buttons">
        {s.state === "DISCONNECTED" && (
          <button className="btn btn-primary" onClick={send("connect")} disabled={off}><Power aria-hidden /> Connect</button>
        )}
        {s.state === "IDENTIFIED" && (
          <>
            <button className="btn" onClick={send("identify")} disabled={off}><RotateCcw aria-hidden /> Read again</button>
            {mismatched ? (
              <button className="btn btn-primary" onClick={() => go("calibrate")}>Go to Calibrate</button>
            ) : (
              <button className="btn btn-primary" onClick={send("confirm")} disabled={off}>Confirm {identity.length} arms</button>
            )}
          </>
        )}
        {(s.state === "IDENTIFIED" || s.state === "READY") && anyTorque && (
          <button className="btn" onClick={() => setConfirmOff("release")} disabled={off}><PowerOff aria-hidden /> Torque off</button>
        )}
        {s.state === "READY" && (
          <button className="btn btn-primary" onClick={send("arm")} disabled={off}><Power aria-hidden /> Enable torque</button>
        )}
        {s.state === "ARMED" && (count === null ? (
          <>
            <button className="btn" onClick={() => setConfirmOff("release")} disabled={off}><PowerOff aria-hidden /> Torque off</button>
            <button className="btn btn-primary" onClick={startCountdown} disabled={off || !canStart} title={!canStart ? startBlocked : undefined}>
              <Play aria-hidden /> {startLabel ?? "Start teleop"}
            </button>
          </>
        ) : (
          <div className="countdown" role="timer" aria-live="assertive">
            <span>{verb} starts in <b className="num">{count}</b></span>
            <button className="btn" onClick={cancel}>Cancel <span className="kbd">Esc</span></button>
          </div>
        ))}
        {s.state === "STOPPED" && (
          <>
            <button className="btn" onClick={() => setConfirmOff("release")} disabled={off}><PowerOff aria-hidden /> Torque off</button>
            <button className="btn btn-primary" onClick={send("resume")} disabled={off}>Resume</button>
          </>
        )}
        {s.state === "FAULT" && anyTorque && (
          <button className="btn btn-primary" onClick={() => setConfirmOff("release")} disabled={off}><PowerOff aria-hidden /> Torque off</button>
        )}
        {s.state === "FAULT" && (
          <button className={`btn ${anyTorque ? "" : "btn-primary"}`} onClick={send("clear")} disabled={off || anyTorque}
            title={anyTorque ? "Turn torque off first: clearing re-checks every arm" : undefined}>
            <RotateCcw aria-hidden /> Clear and re-check arms
          </button>
        )}
        {other && <span className="faint">Stop it first to start {activity === "teleop" ? "teleop" : "a policy"}.</span>}
        {s.state !== "DISCONNECTED" && s.state !== "MOVING" && s.state !== "CALIBRATING" && (
          <button className="btn btn-ghost" onClick={disconnect} disabled={off}><Unplug aria-hidden /> Disconnect</button>
        )}
      </div>

      <Dialog.Root open={confirmOff !== null} onOpenChange={(o) => { if (!o) setConfirmOff(null); }}>
        <Dialog.Portal>
          <Dialog.Overlay className="dialog-overlay" />
          <Dialog.Content className="dialog" aria-describedby="torque-off-desc">
            <Dialog.Title className="dialog-title">
              {confirmOff === "disconnect" ? "Disconnect and turn torque off?" : "Turn torque off?"}
            </Dialog.Title>
            <p id="torque-off-desc" className="dialog-text">
              The followers go limp and fall under their own weight. Support each follower by hand, or lower it to its
              rest pose first.
            </p>
            <div className="dialog-buttons">
              <Dialog.Close asChild><button className="btn" autoFocus>Keep torque on</button></Dialog.Close>
              <button className="btn btn-danger" onClick={() => { studio.send({ cmd: confirmOff ?? "release" }); setConfirmOff(null); }}>
                {confirmOff === "disconnect" ? "Disconnect" : "Turn torque off"}
              </button>
            </div>
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>
    </div>
  );
}

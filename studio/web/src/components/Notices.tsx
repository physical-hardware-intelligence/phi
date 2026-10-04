import { MessageSquareText, OctagonAlert, TriangleAlert, X } from "lucide-react";
import { studio, useStudio } from "../lib/studio";

const FAULT_FIX: Array<[RegExp, string]> = [
  [/overload/, "Something is blocking the joint or the gripper is squeezing too hard. Free the joint, then clear."],
  [/overheat/, "Let the servo cool for a few minutes before clearing. Check that nothing is stalling it."],
  [/voltage/, "Check the power supply: the right adapter for this arm, plugged in, cable seated."],
  [/no status packet|not answering|unplugged/i, "Check the arm's USB cable and power, then clear. The other arms are holding."],
];

function fixFor(fault: string): string {
  return FAULT_FIX.find(([re]) => re.test(fault))?.[1] ?? "Read the details, fix the cause, then clear.";
}

// Faults and errors stay in the page next to the work until resolved or dismissed: never a toast.
export function Notices() {
  const s = useStudio((x) => x.state);
  const errors = useStudio((x) => x.errors);
  const exit = useStudio((x) => x.workerExit);
  const link = useStudio((x) => x.link);

  return (
    <div className="notices">
      {link === "refused" && (
        <Notice tone="danger" title="This page is not connected to Studio"
          fix="Open the link Studio printed in the terminal. Each launch makes a new one." />
      )}
      {link === "closed" && <Notice tone="warn" title="Connection lost. Reconnecting." fix="" />}
      {link === "down" && (
        <Notice tone="danger" title="Studio is not running"
          fix="Start it again in a terminal with make studio. A new launch prints a new link; open that one." />
      )}
      {exit && <Notice tone="danger" title={exit} fix="Run phi studio again. Motion stopped when the worker ended." ask />}
      {s?.state === "FAULT" && s.fault && <Notice tone="danger" title={s.fault} fix={fixFor(s.fault)} ask />}
      {s?.state === "STOPPED" && s.stop_reason === "heartbeat" && (
        <Notice tone="warn" title="Motion stopped because this window stopped answering"
          fix="Studio stops the arm when the controlling window is closed, frozen, or loses its connection. Resume when ready." ask />
      )}
      {errors.map((e) => (
        <Notice key={e.id} tone="warn" title={e.message} fix={e.fix} onDismiss={() => studio.dismissError(e.id)} ask />
      ))}
    </div>
  );
}

function Notice({ tone, title, fix, onDismiss, ask }: {
  tone: "warn" | "danger"; title: string; fix: string; onDismiss?: () => void; ask?: boolean;
}) {
  const Icon = tone === "danger" ? OctagonAlert : TriangleAlert;
  return (
    <div className={`notice tone-${tone}`} role={tone === "danger" ? "alert" : "status"}>
      <Icon className="notice-icon" aria-hidden />
      <div className="notice-body">
        <div className="notice-title">{title}</div>
        {fix && <div className="notice-fix">{fix}</div>}
      </div>
      <div className="notice-actions">
        {ask && (
          <button className="btn btn-sm notice-ask" onClick={() => studio.openAssistant({ message: title, fix })}>
            <MessageSquareText aria-hidden /> Ask Claude
          </button>
        )}
        {onDismiss && (
          <button className="btn btn-ghost btn-sm btn-icon" onClick={onDismiss} aria-label="Dismiss">
            <X aria-hidden />
          </button>
        )}
      </div>
    </div>
  );
}

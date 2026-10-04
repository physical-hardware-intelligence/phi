import { Camera, CameraOff, Check, ChevronRight, CircleAlert } from "lucide-react";
import type { ReactNode } from "react";
import { go, studio, useStudio, type Route, type Tone } from "../lib/studio";
import { Notices } from "../components/Notices";
import { label } from "../lib/labels";

type StepStatus = "done" | "next" | "blocked" | "open";
interface Step { n: number; title: string; status: StepStatus; detail: string; action?: ReactNode }

const CONFIRMED = new Set(["READY", "ARMED", "MOVING", "STOPPED"]);

// What to do next, read from live state. Each row says what the step is, whether it is done, and the one
// action that moves it forward.
function useSteps(): Step[] {
  const s = useStudio((x) => x.state);
  const identity = useStudio((x) => x.identity);
  const control = useStudio((x) => x.control);
  const st = s?.state ?? "DISCONNECTED";
  const connected = st !== "DISCONNECTED";
  const bad = identity.filter((a) => !a.ok);
  const confirmed = CONFIRMED.has(st);
  const goBtn = (r: Route, label: string, primary = false) => (
    <button className={`btn btn-sm ${primary ? "btn-primary" : ""}`} onClick={() => go(r)}>{label}<ChevronRight aria-hidden /></button>
  );

  return [
    {
      n: 1, title: "Connect the rig",
      status: connected ? "done" : "next",
      detail: connected ? `${identity.length} arms answered on their ports` : "Studio reads each arm's identity from its servos. Nothing moves.",
      action: !connected && <button className="btn btn-sm btn-primary" onClick={() => studio.send({ cmd: "connect" })} disabled={!control}>Connect</button>,
    },
    {
      n: 2, title: "Check calibration and confirm arms",
      status: !connected ? "blocked" : confirmed ? "done" : "next",
      detail: !connected ? "Needs a connected rig"
        : confirmed ? "Every arm matches its own calibration file"
        : bad.length ? `${bad.length} of ${identity.length} arms do not match their own calibration file`
        : "Every arm matches its own file. Confirm to allow torque.",
      action: connected && !confirmed && (bad.length ? goBtn("calibrate", "Calibrate", true)
        : <button className="btn btn-sm btn-primary" onClick={() => studio.send({ cmd: "confirm" })} disabled={!control || st !== "IDENTIFIED"}>Confirm arms</button>),
    },
    {
      n: 3, title: "Teleoperate",
      status: !confirmed ? "blocked" : s?.activity === "teleop" ? "done" : "open",
      detail: s?.state === "MOVING" && s.activity === "teleop" ? "Teleop is running" : "Drive each follower with its leader and check the cameras",
      action: goBtn("teleop", "Open"),
    },
    {
      n: 4, title: "Run a policy",
      status: !confirmed ? "blocked" : "open",
      detail: "Run a trained policy on the followers and watch its latency",
      action: goBtn("policy", "Open"),
    },
    {
      n: 5, title: "Evaluate",
      status: !confirmed ? "blocked" : "open",
      detail: "Run several episodes, judge each one, and get a success rate with its uncertainty",
      action: goBtn("evaluate", "Open"),
    },
  ];
}

export function Overview() {
  const steps = useSteps();
  return (
    <div className="page overview">
      <Notices />
      <section className="panel workflow">
        <div className="panel-head">
          <h2 className="panel-title">Workflow</h2>
        </div>
        <ol className="steps">
          {steps.map((st) => (
            <li key={st.n} className={`step is-${st.status}`}>
              <span className="step-mark" aria-hidden>{st.status === "done" ? <Check /> : st.n}</span>
              <div className="step-text">
                <div className="step-title">{st.title}</div>
                <div className="step-detail">{st.detail}</div>
              </div>
              <div className="step-action">{st.status !== "blocked" && st.action}</div>
            </li>
          ))}
        </ol>
      </section>
      <RigHealth />
      <Activity />
    </div>
  );
}

function RigHealth() {
  const identity = useStudio((s) => s.identity);
  const tele = useStudio((s) => s.telemetry);
  const cams = useStudio((s) => s.cameras);
  const live = useStudio((s) => s.link === "open");
  const state = useStudio((s) => s.state?.state);

  return (
    <section className="panel health">
      <div className="panel-head">
        <h2 className="panel-title">Rig health</h2>
      </div>
      {identity.length === 0 ? (
        <p className="empty">{state === "DISCONNECTED" || !state ? "Connect the rig to read its arms." : "Reading arms."}</p>
      ) : (
        <table className="table">
          <thead><tr><th>Arm</th><th>Calibration</th><th>Torque</th><th className="r">Max temp</th><th className="r">Max load</th></tr></thead>
          <tbody>
            {identity.map((a) => {
              const t = tele?.arms[a.name];
              const h = t ? Object.values(t.health) : [];
              const faults = h.flatMap((x) => x.faults);
              const maxT = h.length ? Math.max(...h.map((x) => x.temp)) : null;
              const maxL = h.length ? Math.max(...h.map((x) => Math.abs(x.load))) : null; // the sign is direction
              return (
                <tr key={a.name}>
                  <td>
                    <div className="arm-cell">
                      <span className="strong">{label(a.name)}</span>
                      <span className="faint t-cap">{a.role === "leader" ? "Leader" : "Follower"}</span>
                    </div>
                  </td>
                  <td>
                    {t && !t.online ? <Badge tone="danger">Not answering</Badge>
                      : faults.length ? <Badge tone="danger">{faults[0]}</Badge>
                      : a.ok ? <Badge tone="ok">Matches</Badge>
                      : a.exact ? <Badge tone="danger">Swapped</Badge>
                      : <Badge tone="warn">{a.match ? `${a.max_deg?.toFixed(1)}° off` : "No file"}</Badge>}
                  </td>
                  <td>{t?.torque ? <Badge tone="warn">On</Badge> : <span className="faint">Off</span>}</td>
                  <td className={`r num ${maxT !== null && maxT >= 60 ? "text-warn" : ""}`}>{maxT === null ? "" : `${maxT.toFixed(0)} °C`}</td>
                  <td className={`r num ${maxL !== null && maxL > 80 ? "text-warn" : ""}`}>{maxL === null ? "" : `${maxL.toFixed(0)}%`}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
      {Object.keys(cams).length > 0 && (
        <>
          <hr className="divider" />
          <ul className="cam-list">
            {Object.entries(cams).map(([k, c]) => (
              <li key={k} className={!live ? "" : c.online ? "" : "text-danger"}>
                {c.online && live ? <Camera aria-hidden /> : <CameraOff aria-hidden />}
                <span>{label(k)}</span>
                <span className="faint">{!live ? "unknown while offline" : c.online ? "streaming" : c.message ?? "no signal"}</span>
              </li>
            ))}
          </ul>
        </>
      )}
    </section>
  );
}

function Badge({ tone, children }: { tone: Tone; children: ReactNode }) {
  return <span className={`badge tone-${tone}`}>{children}</span>;
}

function Activity() {
  const items = useStudio((s) => s.activity);
  return (
    <section className="panel activity">
      <div className="panel-head">
        <h2 className="panel-title">Activity</h2>
        <span className="panel-sub">This session, newest first</span>
      </div>
      {items.length === 0 ? (
        <p className="empty">Nothing yet.</p>
      ) : (
        <ul className="log">
          {items.slice(0, 60).map((e) => (
            <li key={e.id} className={`log-row tone-${e.tone}`}>
              <time className="log-time num">{new Date(e.at).toLocaleTimeString([], { hour12: false })}</time>
              {e.tone === "danger" || e.tone === "warn" ? <CircleAlert className="log-icon" aria-hidden /> : <span className="dot" />}
              <span className="log-text">{e.text}</span>
              <span className="log-detail">{e.detail}</span>
              {e.ask ? (
                <button className="btn btn-ghost btn-sm log-ask" onClick={() => studio.openAssistant({ message: e.text, fix: e.detail })}>
                  Ask Claude
                </button>
              ) : <span />}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

import { CircleAlert, CircleCheck, CircleX, Check, Hand, Power, RotateCcw, X } from "lucide-react";
import { useEffect, useState } from "react";
import { Notices } from "../components/Notices";
import { label, labels } from "../lib/labels";
import {
  JOINTS, go, studio, useStudio, TICKS_PER_DEG,
  type ArmIdentity, type CalStep, type CalView, type SessionState,
} from "../lib/studio";

const CAN_START = new Set<SessionState>(["IDENTIFIED", "READY"]);
const STEPS: { key: CalStep | "check"; title: string; detail: string }[] = [
  { key: "check", title: "Check", detail: "Torque off on every arm, cable not swapped" },
  { key: "middle", title: "Set the middle", detail: "Every joint at mid-range; homing makes it read 2047" },
  { key: "ranges", title: "Record ranges", detail: "Sweep each joint end to end, except Wrist Roll" },
  { key: "review", title: "Review and save", detail: "Old against new, then write registers and file" },
];

// LeRobot's calibration, one step per screen (so_follower.py calibrate). The worker owns every step;
// this page shows where it is and sends the next command.
export function Calibrate() {
  const cal = useStudio((s) => s.telemetry?.calibration ?? null);
  const done = useStudio((s) => s.calibrated);
  const [hidden, setHidden] = useState<number | null>(null);
  return (
    <div className="page calibrate">
      <Notices />
      {done && hidden !== done.at && !cal && (
        <div className="notice tone-ok" role="status">
          <CircleCheck className="notice-icon" aria-hidden />
          <div className="notice-body">
            <div className="notice-title">Calibrated {done.arm}</div>
            <div className="notice-fix">
              {done.path ? <>Registers written and saved to <span className="mono">{done.path}</span>.</>
                : "Registers written. No file was saved: this Studio has no calibration directory."}
              {" "}Every arm was read again.
            </div>
          </div>
          <button className="btn btn-ghost btn-sm notice-x" onClick={() => setHidden(done.at)} aria-label="Dismiss"><X aria-hidden /></button>
        </div>
      )}
      <div className="cal-grid">
        <ArmPicker active={cal?.arm ?? null} />
        {cal ? <Wizard cal={cal} /> : <HowItWorks />}
      </div>
    </div>
  );
}

function blockedReason(a: ArmIdentity, st: SessionState | undefined, control: boolean, holding: string[]): string | null {
  if (!control) return "Another window has control";
  if (st === "CALIBRATING") return "Another calibration is running";
  if (st === "FAULT") return "Clear the fault first";
  if (holding.length) return `Turn torque off first: ${labels(holding)} ${holding.length === 1 ? "holds" : "hold"} torque`;
  if (!st || !CAN_START.has(st)) return "Turn torque off first";
  if (a.exact && !a.ok) return "Fix the swapped cables first";
  return null;
}

function ArmPicker({ active }: { active: string | null }) {
  const arms = useStudio((s) => s.identity);
  const st = useStudio((s) => s.state?.state);
  const control = useStudio((s) => s.control);
  const tele = useStudio((s) => s.telemetry);
  const holding = Object.entries(tele?.arms ?? {}).filter(([, a]) => a.torque).map(([n]) => n);

  return (
    <section className="panel cal-arms">
      <div className="panel-head">
        <h2 className="panel-title">Arms</h2>
      </div>
      {st === "DISCONNECTED" || !st ? (
        <div className="empty empty-action">
          <p>Connect to read each arm's calibration from its servos.</p>
          <button className="btn btn-primary" onClick={() => studio.send({ cmd: "connect" })} disabled={!control}>
            <Power aria-hidden /> Connect
          </button>
        </div>
      ) : arms.length === 0 ? (
        <p className="empty">Reading arms.</p>
      ) : (
        <ul className="pick-list">
          {arms.map((a) => {
            const why = blockedReason(a, st, control, holding);
            const isActive = active === a.name;
            return (
              <li key={a.name} className={`pick-row ${isActive ? "is-active" : ""}`}>
                <div className="pick-text">
                  <div className="pick-name">
                    <span className="ellipsis">{label(a.name)}</span>
                    <span className="pick-role">{a.role === "leader" ? "Leader" : "Follower"}</span>
                  </div>
                  <CalBadge a={a} />
                </div>
                {isActive ? (
                  <span className="badge tone-info">In progress</span>
                ) : (
                  <button className="btn btn-sm" disabled={why !== null} title={why ?? undefined}
                    onClick={() => studio.send({ cmd: "cal_start", arm: a.name })}>
                    Calibrate
                  </button>
                )}
              </li>
            );
          })}
        </ul>
      )}
      {arms.length > 0 && !control && st !== "CALIBRATING" && (
        <div className="panel-foot"><span className="text-warn">View only: another window has control.</span></div>
      )}
      {arms.length > 0 && control && st && !CAN_START.has(st) && st !== "CALIBRATING" && (
        <div className="panel-foot panel-foot-action">
          <span>{st === "FAULT" ? "Clear the fault, then calibrate." : "Calibration needs torque off on every arm."}</span>
          {st !== "FAULT" && <button className="btn btn-sm" onClick={() => go("teleop")}>Go to Teleoperate</button>}
        </div>
      )}
    </section>
  );
}

// One line under the arm's name: does the calibration on its servos match the file Studio expects?
function CalBadge({ a }: { a: ArmIdentity }) {
  const [tone, Icon, text] =
    a.ok ? ["ok", CircleCheck, `Matches ${a.expected}.json`]
    : a.exact ? ["danger", CircleX, `Holds ${a.match}.json: cables swapped?`]
    : a.match === null ? ["warn", CircleAlert, "No calibration file"]
    : ["warn", CircleAlert, `${a.max_deg?.toFixed(1)}° off ${a.expected}.json`];
  return <div className={`pick-meta text-${tone}`}><Icon aria-hidden /><span className="ellipsis">{text}</span></div>;
}

function HowItWorks() {
  const rig = useStudio((s) => s.rig);
  return (
    <section className="panel">
      <div className="panel-head">
        <div>
          <h2 className="panel-title">How calibration works</h2>
          <p className="panel-sub">LeRobot's steps, the same ones its calibrate command runs</p>
        </div>
      </div>
      <ol className="steps">
        {STEPS.map((s, i) => (
          <li key={s.key} className="step is-open">
            <span className="step-mark" aria-hidden>{i + 1}</span>
            <div className="step-text">
              <div className="step-title">{s.title}</div>
              <div className="step-detail">{s.detail}</div>
            </div>
            <div />
          </li>
        ))}
      </ol>
      <p className="panel-foot">
        Cancel at any step and Studio writes the arm's old registers back.
        {rig?.mock && rig.cal_dir && <> Mock rig: files save to <span className="mono">{rig.cal_dir}</span>, never to LeRobot's own folder.</>}
      </p>
    </section>
  );
}

function Wizard({ cal }: { cal: CalView }) {
  const control = useStudio((s) => s.control);
  const mock = useStudio((s) => s.mock);
  const rig = useStudio((s) => s.rig);
  const expected = useStudio((s) => s.identity.find((a) => a.name === cal.arm)?.expected ?? cal.arm);
  const [sweeping, setSweeping] = useState(false);
  const at = STEPS.findIndex((s) => s.key === cal.step);
  const send = (cmd: string) => () => studio.send({ cmd });

  // The mock's scripted hand keeps moving a limp arm: let go of it when the recording ends.
  const sweep = (on: boolean) => { studio.send({ cmd: "inject", arm: cal.arm, kind: on ? "hand" : "still" }); setSweeping(on); };
  useEffect(() => { if (cal.step !== "ranges" && sweeping) sweep(false); }, [cal.step]); // eslint-disable-line react-hooks/exhaustive-deps
  const cancel = () => { if (sweeping) sweep(false); studio.send({ cmd: "cal_cancel" }); };

  const unmoved = Object.entries(cal.joints).filter(([, v]) => !v.fixed && v.min === v.max).map(([j]) => j);
  const path = rig?.cal_dir ? `${rig.cal_dir}/${expected}.json` : null;

  return (
    <section className="panel wizard">
      <div className="panel-head">
        <div>
          <h2 className="panel-title">Calibrating {cal.arm}</h2>
          <p className="panel-sub">{cal.role === "leader" ? "Leader" : "Follower"}, torque off. Support it by hand.</p>
        </div>
        <button className="btn btn-ghost" onClick={cancel} disabled={!control}><RotateCcw aria-hidden /> Cancel and restore</button>
      </div>
      <ol className="stepper" aria-label="Calibration steps">
        {STEPS.map((s, i) => (
          <li key={s.key} className={`stepper-item ${i < at ? "is-done" : i === at ? "is-current" : ""}`} aria-current={i === at ? "step" : undefined}>
            <span className="step-mark" aria-hidden>{i < at ? <Check /> : i + 1}</span>
            <span className="stepper-title">{s.title}</span>
          </li>
        ))}
      </ol>
      <hr className="divider" />

      {cal.step === "middle" && (
        <div className="wizard-body">
          <div className="instr">
            <p className="instr-lead">Move every joint of {cal.arm} by hand to roughly the middle of its range, and hold it there.</p>
            <p className="muted">Studio then writes each servo's homing offset so that this pose reads 2047, half of one turn.
              Ranges are recorded around it in the next step.</p>
          </div>
          <div className="wizard-actions">
            <button className="btn btn-primary" onClick={send("cal_middle")} disabled={!control}>Set middle</button>
          </div>
        </div>
      )}

      {cal.step === "ranges" && (
        <div className="wizard-body">
          <div className="instr">
            <p className="instr-lead">Move each joint slowly through its full range, end to end. Leave wrist roll: its range is a full turn.</p>
          </div>
          <RangeBars cal={cal} />
          <div className="wizard-actions">
            {mock && (
              <button className="btn" onClick={() => sweep(!sweeping)} disabled={!control}>
                <Hand aria-hidden /> {sweeping ? "Stop the hand" : "Sweep it by hand (mock)"}
              </button>
            )}
            <span className="wizard-hint">{unmoved.length ? `Not moved yet: ${unmoved.map(label).join(", ")}` : "Every joint has moved."}</span>
            <button className="btn btn-primary" onClick={send("cal_finish")} disabled={!control || unmoved.length > 0}
              title={unmoved.length ? "Move every joint first" : undefined}>
              Finish recording
            </button>
          </div>
        </div>
      )}

      {cal.step === "review" && cal.new && (
        <div className="wizard-body">
          <ReviewTable cal={cal} />
          <div className="wizard-actions">
            <span className="wizard-hint">
              Save writes these registers to {cal.arm}{path ? <> and saves <span className="mono">{path}</span></> : " (no file: Studio has no calibration directory)"}.
            </span>
            <button className="btn btn-primary" onClick={send("cal_save")} disabled={!control}>Save calibration</button>
          </div>
        </div>
      )}
    </section>
  );
}

function RangeBars({ cal }: { cal: CalView }) {
  return (
    <ul className="ranges" aria-label="Recorded ranges">
      {JOINTS.map((j) => {
        const v = cal.joints[j];
        if (!v) return null;
        const pos = (x: number) => `${(Math.min(4095, Math.max(0, x)) / 4095) * 100}%`;
        const span = (v.max - v.min) / TICKS_PER_DEG;
        return (
          <li key={j} className={`range-row ${v.fixed ? "is-fixed" : v.min === v.max ? "is-still" : ""}`}>
            <span className="range-joint">{label(j)}</span>
            <div className="range-track" aria-hidden>
              <span className="range-mid" />
              <span className="range-fill" style={{ left: pos(v.min), width: `calc(${pos(v.max)} - ${pos(v.min)})` }} />
              <span className="range-pos" style={{ left: pos(v.pos) }} />
            </div>
            <span className="range-nums mono">{v.min}<span className="faint"> / </span><b>{v.pos}</b><span className="faint"> / </span>{v.max}</span>
            <span className="range-span num">{v.fixed ? "Full turn" : v.min === v.max ? "Not moved" : `${span.toFixed(0)}°`}</span>
          </li>
        );
      })}
    </ul>
  );
}

function ReviewTable({ cal }: { cal: CalView }) {
  const cell = (o: number, n: number) => (
    <td className="r mono">
      {o === n ? <span className="faint">{n}</span> : <><span className="faint">{o} → </span><b>{n}</b></>}
    </td>
  );
  return (
    <div className="table-scroll">
      <table className="table review">
        <thead>
          <tr><th>Joint</th><th className="r">Homing offset</th><th className="r">Min</th><th className="r">Max</th><th className="r">Range</th></tr>
        </thead>
        <tbody>
          {JOINTS.map((j) => {
            const o = cal.old[j], n = cal.new![j];
            return (
              <tr key={j}>
                <td className="strong">{label(j)}</td>
                {cell(o.homing_offset, n.homing_offset)}
                {cell(o.range_min, n.range_min)}
                {cell(o.range_max, n.range_max)}
                <td className="r num">{((n.range_max - n.range_min) / TICKS_PER_DEG).toFixed(0)}°</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

import { useState, type ReactNode } from "react";
import { ActionBar } from "../components/ActionBar";
import { Cameras } from "../components/Cameras";
import { MockPanel } from "../components/MockPanel";
import { Notices } from "../components/Notices";
import { JOINTS, useStudio, type PolicyInfo } from "../lib/studio";
import { label } from "../lib/labels";

const LIMIT = { min: 1, max: 600 }; // seconds; the worker refuses anything else (worker.py POLICY_LIMIT_S)

// What the operator last chose, kept while they move between pages.
const memory = { policy: "", task: "Pick up the red cube and place it in the box", limit: 30 };

export function usePolicyChoice(policies: PolicyInfo[]) {
  const [policy, setPolicy] = useState(memory.policy || policies.find((p) => p.available)?.id || "");
  const [task, setTask] = useState(memory.task);
  const [limit, setLimit] = useState(memory.limit);
  const chosen = policies.find((p) => p.id === policy) ?? policies.find((p) => p.available);
  const set = {
    policy: (v: string) => { memory.policy = v; setPolicy(v); },
    task: (v: string) => { memory.task = v; setTask(v); },
    limit: (v: number) => { memory.limit = v; setLimit(v); },
  };
  const problem = !chosen ? "No policy is available on this rig"
    : !chosen.available ? chosen.note
    : !(limit >= LIMIT.min && limit <= LIMIT.max) ? `Time limit must be ${LIMIT.min} to ${LIMIT.max} s`
    : task.length > 300 ? "Task text is over 300 characters"
    : null;
  return { policy: chosen?.id ?? "", task, limit, set, problem, chosen };
}

export function Policy() {
  const policies = useStudio((s) => s.rig?.policies ?? []);
  const choice = usePolicyChoice(policies);
  return (
    <div className="page policy">
      <ActionBar activity="policy" startLabel="Run policy" canStart={choice.problem === null}
        startBlocked={choice.problem ?? undefined}
        startMsg={{ policy: choice.policy, task: choice.task, limit_s: choice.limit }} />
      <Notices />
      <div className="work-grid">
        <div className="col">
          <Cameras />
          <ActionVsState />
        </div>
        <div className="col rail">
          <PolicySetup policies={policies} choice={choice} />
          <RunPanel />
          <MockPanel />
        </div>
      </div>
    </div>
  );
}

export function PolicySetup({ policies, choice, title = "Policy", children }: {
  policies: PolicyInfo[]; choice: ReturnType<typeof usePolicyChoice>; title?: string; children?: ReactNode;
}) {
  const moving = useStudio((s) => s.state?.state === "MOVING");
  const off = policies.filter((p) => !p.available);
  return (
    <section className="panel">
      <div className="panel-head"><h2 className="panel-title">{title}</h2></div>
      <fieldset className="panel-body form" disabled={moving}>
        <label className="field">
          <span className="field-label">Policy</span>
          <select className="select" value={choice.policy} onChange={(e) => choice.set.policy(e.target.value)}>
            {policies.map((p) => (
              <option key={p.id} value={p.id} disabled={!p.available}>{p.name}{p.available ? "" : " (not available)"}</option>
            ))}
          </select>
          {off.length > 0 && <span className="field-hint">{off.map((p) => p.name).join(" and ")}: {off[0].note}</span>}
        </label>
        <label className="field">
          <span className="field-label">Task</span>
          <textarea className="textarea" rows={2} value={choice.task} maxLength={300} onChange={(e) => choice.set.task(e.target.value)} />
          <span className="field-hint">Given to the policy as its instruction. The scripted mock policy ignores it.</span>
        </label>
        <label className="field">
          <span className="field-label">Time limit</span>
          <div className="input-suffix">
            <input className="input num" type="number" min={LIMIT.min} max={LIMIT.max} step={1} value={choice.limit}
              onChange={(e) => choice.set.limit(Number(e.target.value))} />
            <span>seconds</span>
          </div>
          <span className="field-hint">The run stops itself at the limit and the followers hold where they are.</span>
        </label>
        {children}
      </fieldset>
    </section>
  );
}

const ENDED: Record<string, string> = {
  "time limit": "Reached its time limit", user: "Stopped by you", heartbeat: "Stopped: this window stopped answering",
  "window closed": "Stopped: the controlling window closed", "control moved": "Stopped: control moved to another window",
  fault: "Stopped by a fault", disconnected: "Stopped: the rig was disconnected",
};
// WHY per state: after a teleop stop torque is already on, and Run policy appears only after Resume.
const FIRST_RUN: Record<string, string> = { ARMED: "Press Run policy above.", STOPPED: "Resume, then Run policy." };

export function RunPanel() {
  const run = useStudio((s) => s.telemetry?.policy ?? null);
  const hz = useStudio((s) => s.telemetry?.loop.hz ?? 0);
  const state = useStudio((s) => s.state?.state ?? "");
  return (
    <section className="panel">
      <div className="panel-head">
        <div>
          <h2 className="panel-title">This run</h2>
          {run && <p className="panel-sub">{run.name}</p>}
        </div>
        {run && <span className={`badge tone-${run.running ? "ok" : run.ended === "time limit" ? "neutral" : "warn"}`}>
          {run.running ? "Running" : "Ended"}</span>}
      </div>
      {!run ? (
        <p className="empty">Nothing has run yet. {FIRST_RUN[state] ?? "Enable torque, then Run policy."}</p>
      ) : (
        <div className="panel-body run">
          <div className="progress" role="progressbar" aria-valuemin={0} aria-valuemax={run.limit_s} aria-valuenow={run.episode_s}
            aria-label="Episode time">
            <span style={{ width: `${Math.min(100, (run.episode_s / run.limit_s) * 100)}%` }} />
          </div>
          <div className="progress-label num">
            <span>{run.episode_s.toFixed(1)} s</span><span className="faint">of {run.limit_s.toFixed(0)} s</span>
          </div>
          {!run.running && run.ended && <p className="run-ended">{ENDED[run.ended] ?? `Stopped: ${run.ended}`}</p>}
          <dl className="kv">
            <dt>Control steps</dt><dd className="num">{run.step}</dd>
            <dt>Actions per inference</dt><dd className="num">{run.chunk}</dd>
            <dt>Inference, last</dt><dd className="num">{fmtMs(run.chunk_ms)}</dd>
            <dt>Inference, median / max</dt><dd className="num">{fmtMs(run.chunk_ms_p50)} / {fmtMs(run.chunk_ms_max)}</dd>
            <dt>Control loop</dt><dd className="num">{hz.toFixed(1)} Hz</dd>
            <dt>Run id</dt><dd className="mono">{run.run_id}</dd>
          </dl>
        </div>
      )}
    </section>
  );
}

function fmtMs(ms: number): string {
  return ms < 1 ? `${(ms * 1000).toFixed(0)} µs` : `${ms.toFixed(1)} ms`;
}

// Where the policy asked each joint to go against where it is. A gap that keeps growing on one joint
// usually means that joint is blocked or overloaded.
export function ActionVsState() {
  const run = useStudio((s) => s.telemetry?.policy ?? null);
  const arms = useStudio((s) => s.telemetry?.arms ?? {});
  const followers = Object.keys(run?.action ?? {});
  return (
    <section className="panel">
      <div className="panel-head">
        <div>
          <h2 className="panel-title">Action and state</h2>
          <p className="panel-sub">Policy target against measured position. Degrees; gripper 0 to 100</p>
        </div>
      </div>
      {!run || followers.length === 0 ? (
        <p className="empty">Run a policy to see what it commands.</p>
      ) : (
        <div className="table-scroll">
          <table className="jt">
            <thead>
              <tr>
                <th />
                {followers.map((f) => <th key={f} colSpan={3} className="jt-arm">{label(f)}</th>)}
              </tr>
              <tr className="jt-sub">
                <th className="jt-joint">Joint</th>
                {followers.map((f) => [<th key={`${f}t`}>Target</th>, <th key={`${f}p`}>Actual</th>, <th key={`${f}e`}>Gap</th>])}
              </tr>
            </thead>
            <tbody>
              {JOINTS.map((j) => (
                <tr key={j}>
                  <th className="jt-joint">{label(j)}</th>
                  {followers.map((f) => <Gap key={f} target={run.action[f]?.[j]} actual={arms[f]?.pos[j]} />)}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

function Gap({ target, actual }: { target?: number; actual?: number }) {
  if (target === undefined || actual === undefined) return <><td /><td /><td /></>;
  const gap = target - actual;
  const big = Math.abs(gap) > 15;
  return (
    <>
      <td className="num">{target.toFixed(1)}</td>
      <td className="num">{actual.toFixed(1)}</td>
      <td className={`num ${big ? "text-warn" : "faint"}`}>{gap >= 0 ? "+" : ""}{gap.toFixed(1)}</td>
    </>
  );
}


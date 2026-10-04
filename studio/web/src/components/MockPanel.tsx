import { useState } from "react";
import { JOINTS, studio, useStudio } from "../lib/studio";

const KINDS = [
  ["overload", "Overload a joint"],
  ["overheat", "Overheat a joint"],
  ["voltage", "Drop the supply voltage"],
  ["unplug", "Unplug the arm"],
  ["swap", "Swap its cable with its pair's"],
] as const;

// Mock rig only: inject the faults a real rig produces, to see how Studio reacts to each.
export function MockPanel() {
  const mock = useStudio((s) => s.mock);
  const arms = useStudio((s) => s.identity);
  const control = useStudio((s) => s.control);
  const [arm, setArm] = useState("");
  const [kind, setKind] = useState<string>("overload");
  const [joint, setJoint] = useState<string>("gripper");
  if (!mock || !arms.length) return null;
  const target = arm || arms.find((a) => a.role === "follower")?.name || arms[0].name;

  return (
    <section className="panel mockpanel">
      <div className="panel-head">
        <div>
          <h2 className="panel-title">Fault drill</h2>
          <p className="panel-sub">Mock rig only: see how Studio reacts to each fault</p>
        </div>
      </div>
      <div className="panel-body mock-form">
        <label className="field">
          <span className="field-label">Arm</span>
          <select className="select" value={target} onChange={(e) => setArm(e.target.value)}>
            {arms.map((a) => <option key={a.name} value={a.name}>{a.name}</option>)}
          </select>
        </label>
        <label className="field">
          <span className="field-label">Fault</span>
          <select className="select" value={kind} onChange={(e) => setKind(e.target.value)}>
            {KINDS.map(([k, label]) => <option key={k} value={k}>{label}</option>)}
          </select>
        </label>
        {kind !== "unplug" && kind !== "swap" && (
          <label className="field">
            <span className="field-label">Joint</span>
            <select className="select" value={joint} onChange={(e) => setJoint(e.target.value)}>
              {JOINTS.map((j) => <option key={j} value={j}>{j}</option>)}
            </select>
          </label>
        )}
        <div className="mock-buttons">
          <button className="btn btn-sm" disabled={!control}
            onClick={() => studio.send({ cmd: "inject", arm: target, kind, joint: kind === "unplug" || kind === "swap" ? undefined : joint })}>
            Inject fault
          </button>
          <button className="btn btn-sm btn-ghost" disabled={!control}
            onClick={() => arms.forEach((a) => {
              studio.send({ cmd: "inject", arm: a.name, kind: "clear" });
              studio.send({ cmd: "inject", arm: a.name, kind: "replug" });
            })}>
            Repair all
          </button>
        </div>
      </div>
    </section>
  );
}

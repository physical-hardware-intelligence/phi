import { Check, CircleAlert, CircleCheck, Copy, FileText, TriangleAlert } from "lucide-react";
import { useEffect, useState } from "react";
import { Notices } from "../components/Notices";
import { label } from "../lib/labels";
import { go, studio, useStudio, type LeRobotArm, type LeRobotView } from "../lib/studio";

// robot-config.yaml read with LeRobot 0.6.0's own rules (src/phi/studio/rigspec.py): each arm's type, id,
// port and calibration file, what LeRobot would refuse, and the CLI commands for this rig in the order a
// new rig needs them. Read-only; tests check every command against LeRobot's own CLI parser.
export function Setup() {
  const link = useStudio((s) => s.link);
  const index = useStudio((s) => s.files.index);
  useEffect(() => { if (link === "open") studio.loadFiles(); }, [link]);
  const lr = index?.lerobot;
  const rig = useStudio((s) => s.rig);

  return (
    <div className="page setup">
      <Notices />
      {!index ? (
        <section className="panel"><p className="empty">Reading robot-config.yaml</p></section>
      ) : !lr ? (
        <section className="panel">
          <p className="empty">No robot-config.yaml in the main checkout. Studio reads the rig from it.</p>
        </section>
      ) : lr.error ? (
        <section className="panel">
          <div className="panel-head"><h2 className="panel-title">Rig</h2><OpenConfig lr={lr} /></div>
          <p className="empty text-danger">robot-config.yaml does not parse: {lr.error}</p>
        </section>
      ) : (
        <>
          {rig?.mock && (
            <p className="setup-note">
              Studio is running a mock {rig.bimanual ? "bimanual" : "single-arm"} rig. This page shows the real rig in
              robot-config.yaml.
            </p>
          )}
          <RigPanel lr={lr} />
          <div className="setup-grid">
            <CommandsPanel lr={lr} />
            <KeysPanel lr={lr} />
          </div>
        </>
      )}
    </div>
  );
}

function OpenConfig({ lr, line }: { lr: LeRobotView; line?: number | null }) {
  return (
    <button className="btn btn-sm" onClick={() => { studio.openFile(lr.file.path, line ?? null, lr.file.root); go("files"); }}>
      <FileText aria-hidden /> Open robot-config.yaml
    </button>
  );
}

function RigPanel({ lr }: { lr: LeRobotView }) {
  return (
    <section className="panel">
      <div className="panel-head">
        <div className="setup-title">
          <h2 className="panel-title">Rig</h2>
          <span className="badge tone-neutral">{lr.bimanual ? "Bimanual" : "Single arm"}</span>
        </div>
        <OpenConfig lr={lr} />
      </div>
      <div className="table-scroll">
        <table className="table setup-arms">
          <thead>
            <tr>
              <th>Arm and LeRobot type</th><th>Port</th><th>Calibration file (id.json)</th>
              <th>Units</th><th>Step limit</th>
            </tr>
          </thead>
          <tbody>{lr.arms.map((a) => <ArmRow key={a.key} a={a} />)}</tbody>
        </table>
      </div>
      {lr.problems.length > 0 && (
        <ul className="setup-problems">
          {lr.problems.map((p) => <li key={p}><TriangleAlert aria-hidden /><span>{p}</span></li>)}
        </ul>
      )}
    </section>
  );
}

function ArmRow({ a }: { a: LeRobotArm }) {
  const cap = a.max_relative_target;
  return (
    <tr>
      <td>
        <div className="strong">{label(a.key)}</div>
        <code className="ident setup-type">{a.type}</code>
      </td>
      <td>{a.port ? <code className="ident setup-port" title={a.port}>{a.port}</code> : <span className="text-warn">Not set</span>}</td>
      <td>
        {!a.id ? <span className="text-warn">No id, so LeRobot would save None.json</span> : (
          <span className={`setup-cal ${a.calibrated ? "text-ok" : "text-warn"}`} title={a.calibration ?? undefined}>
            {a.calibrated ? <CircleCheck aria-hidden /> : <CircleAlert aria-hidden />}
            <code className="ident">{a.id}.json</code>
            {!a.calibrated && <span>missing</span>}
          </span>
        )}
      </td>
      <td>{a.use_degrees ? "Degrees" : "-100 to 100"}</td>
      <td className="num">
        {a.role === "leader" ? <span className="faint">Leader</span>
          : cap === null ? <span className="text-warn">None</span>
          : typeof cap === "number" ? `${cap}` : "Per joint"}
      </td>
    </tr>
  );
}

function CommandsPanel({ lr }: { lr: LeRobotView }) {
  return (
    <section className="panel">
      <div className="panel-head">
        <h2 className="panel-title">Commands for this rig</h2>
        <span className="panel-sub">Run in the phi env, in order</span>
      </div>
      <ol className="cmd-list">
        {lr.commands.map((c, i) => (
          <li key={c.id} className="cmd-item">
            <span className="cmd-n num" aria-hidden>{i + 1}</span>
            <div className="cmd-body">
              <div className="cmd-title">{c.title}</div>
              <p className="cmd-why">{c.why}</p>
              <CommandBlock cmd={c.cmd} />
            </div>
          </li>
        ))}
      </ol>
    </section>
  );
}

/** One flag per line with a trailing backslash, the way LeRobot's docs print them. The copied text is the
 * same, and a shell runs it as one command. */
function CommandBlock({ cmd }: { cmd: string }) {
  const parts = cmd.split(/ (?=--|'--)/);
  const text = parts.join(" \\\n  ");
  const [done, setDone] = useState(false);
  return (
    <div className="cmd-block">
      <pre><code>{text}</code></pre>
      <button className="btn btn-ghost btn-sm btn-icon cmd-copy" aria-label="Copy command" title="Copy"
        onClick={() => { void navigator.clipboard?.writeText(text); setDone(true); setTimeout(() => setDone(false), 1500); }}>
        {done ? <Check aria-hidden /> : <Copy aria-hidden />}
      </button>
    </div>
  );
}

// The names a dataset and a policy see. A policy trained on one rig expects these exact keys, so a camera
// renamed or moved between arms breaks it (bi_so_follower.py:91-98).
function KeysPanel({ lr }: { lr: LeRobotView }) {
  return (
    <section className="panel">
      <div className="panel-head">
        <h2 className="panel-title">Dataset keys</h2>
        <span className="panel-sub">What lerobot-record writes</span>
      </div>
      <div className="panel-body setup-keys">
        <div className="field-label">observation.state and action, {lr.features.length} values</div>
        <div className="key-chips">{lr.features.map((f) => <code key={f} className="ident key-chip">{f}</code>)}</div>
        <div className="field-label">Cameras</div>
        {lr.cameras.length === 0 ? <p className="faint">No cameras in robot-config.yaml.</p> : (
          <ul className="setup-cams">
            {lr.cameras.map((c) => {
              const idx = c.index === null || c.index === undefined || c.index === "" || c.index === "TBD" ? null : String(c.index);
              return (
                <li key={c.feature}>
                  <span className="strong">{label(c.side ? `${c.side}_${c.key}` : c.key)}</span>
                  <code className="ident">{c.feature}</code>
                  {idx === null ? <span className="text-warn">index not set</span> : <span className="faint num">index {idx}</span>}
                </li>
              );
            })}
          </ul>
        )}
      </div>
    </section>
  );
}

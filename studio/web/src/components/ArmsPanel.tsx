import { CircleCheck, TriangleAlert, Unplug } from "lucide-react";
import { useStudio } from "../lib/studio";

// One row per arm: what it is (name, role), where it is (port, adapter serial), and whether its
// servo registers match a calibration file exactly. Matches come from the registers, not the port.
export function ArmsPanel() {
  const arms = useStudio((s) => s.identity);
  const tele = useStudio((s) => s.telemetry);
  const state = useStudio((s) => s.state?.state);

  return (
    <section className="panel arms">
      <div className="panel-head">
        <h2 className="panel-title">Arms</h2>
        <span className="panel-sub">{arms.length ? `${arms.length} found` : ""}</span>
      </div>
      {arms.length === 0 ? (
        <p className="empty">
          {state === "DISCONNECTED" || !state
            ? "Connect to read each arm's identity from its servos."
            : "Reading identities."}
        </p>
      ) : (
        <ul className="arm-list">
          {arms.map((a) => {
            const online = tele?.arms[a.name]?.online ?? true;
            const torque = tele?.arms[a.name]?.torque ?? false;
            return (
              <li key={a.name} className="arm-row">
                <div className="arm-main">
                  <span className="arm-name">{a.name}</span>
                  <span className={`badge ${a.role === "leader" ? "tone-neutral" : "tone-info"}`}>
                    {a.role === "leader" ? "Leader" : "Follower"}
                  </span>
                  {torque && <span className="badge tone-warn">Torque on</span>}
                </div>
                <div className="arm-meta mono faint">{a.port}</div>
                <div className="arm-check">
                  {!online ? (
                    <span className="check tone-danger"><Unplug aria-hidden />Not answering</span>
                  ) : a.ok ? (
                    <span className="check tone-ok"><CircleCheck aria-hidden />Matches {a.match}.json</span>
                  ) : a.exact ? (
                    <span className="check tone-danger">
                      <TriangleAlert aria-hidden />
                      Has {a.match}.json, expected {a.expected}.json. Cables swapped?
                    </span>
                  ) : a.match === null ? (
                    <span className="check tone-warn"><TriangleAlert aria-hidden />No calibration files to compare</span>
                  ) : (
                    <span className="check tone-warn">
                      <TriangleAlert aria-hidden />
                      Nearest {a.match}.json, {a.max_deg?.toFixed(1)}° off on {a.worst_joint}
                    </span>
                  )}
                </div>
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
}

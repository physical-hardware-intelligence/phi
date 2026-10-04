import { JOINTS, useStudio } from "../lib/studio";
import { label } from "../lib/labels";

// Live joint readouts. Followers show position, load, temperature and fault bits; leaders show
// position only. Numbers use tabular figures and fixed decimals so columns never jitter.
export function JointTable() {
  const tele = useStudio((s) => s.telemetry);
  const arms = tele ? Object.entries(tele.arms) : [];

  if (!arms.length) {
    return (
      <section className="panel joints">
        <div className="panel-head"><h2 className="panel-title">Joints</h2></div>
        <p className="empty">Joint readings appear once the rig is connected.</p>
      </section>
    );
  }
  return (
    <section className="panel joints">
      <div className="panel-head">
        <h2 className="panel-title">Joints</h2>
        <span className="panel-sub">Degrees; gripper 0 to 100</span>
      </div>
      <div className="table-scroll">
        <table className="jt">
          <thead>
            <tr>
              <th scope="col" className="jt-joint">Joint</th>
              {arms.map(([name, a]) => (
                <th key={name} scope="colgroup" colSpan={a.role === "follower" ? 4 : 1} className="jt-arm">
                  {label(name)}
                </th>
              ))}
            </tr>
            <tr className="jt-sub">
              <th />
              {arms.map(([name, a]) =>
                a.role === "follower" ? (
                  [<th key={`${name}p`}>Position</th>, <th key={`${name}l`}>Load</th>,
                   <th key={`${name}t`}>Temp</th>, <th key={`${name}f`}>Status</th>]
                ) : (
                  <th key={`${name}p`}>Position</th>
                ),
              )}
            </tr>
          </thead>
          <tbody>
            {JOINTS.map((j) => (
              <tr key={j}>
                <th scope="row" className="jt-joint">{label(j)}</th>
                {arms.map(([name, a]) => {
                  const p = a.pos[j];
                  const h = a.health[j];
                  const cells = [
                    <td key="p" className="num" data-live>{p === undefined ? "" : p.toFixed(1)}</td>,
                  ];
                  if (a.role === "follower") {
                    cells.push(
                      <td key="l" className={`num ${h && Math.abs(h.load) > 80 ? "cell-warn" : ""}`} data-live>
                        {h ? `${h.load.toFixed(0)}%` : ""}
                      </td>,
                      <td key="t" className={`num ${h && h.temp >= 60 ? "cell-warn" : ""}`} data-live>
                        {h ? `${h.temp.toFixed(0)} °C` : ""}
                      </td>,
                      <td key="f" className={h?.faults.length ? "cell-danger" : "faint"}>
                        {h ? (h.faults.length ? h.faults.join(", ") : "OK") : ""}
                      </td>,
                    );
                  }
                  return <Cells key={name}>{cells}</Cells>;
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function Cells({ children }: { children: React.ReactNode }) {
  return <>{children}</>;
}

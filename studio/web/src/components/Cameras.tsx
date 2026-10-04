import { useStudio } from "../lib/studio";
import { CameraTile } from "./CameraTile";

const USUAL = ["front", "wrist", "top"];

// Stable order: the club's usual keys first, anything else after, alphabetically.
export function Cameras({ title = "Cameras" }: { title?: string }) {
  const cams = useStudio((s) => s.cameras);
  const have = Object.keys(cams);
  const keys = [...new Set([...USUAL, ...have.sort()])].filter((k) => !have.length || k in cams);
  return (
    <section className="panel cams-panel">
      <div className="panel-head">
        <h2 className="panel-title">{title}</h2>
        <span className="panel-sub">Preview at 15 fps</span>
      </div>
      <div className="cams">{keys.map((k) => <CameraTile key={k} name={k} />)}</div>
    </section>
  );
}

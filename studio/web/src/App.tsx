import * as Tooltip from "@radix-ui/react-tooltip";
import { useCallback, useEffect } from "react";
import { ActionBar } from "./components/ActionBar";
import { ArmsPanel } from "./components/ArmsPanel";
import { CameraTile } from "./components/CameraTile";
import { JointTable } from "./components/JointTable";
import { MockPanel } from "./components/MockPanel";
import { Notices } from "./components/Notices";
import { TopBar } from "./components/TopBar";
import { studio, useStudio } from "./lib/studio";

const CAMERAS = ["front", "wrist", "top"];

export function App() {
  const stop = useCallback(() => {
    if (studio.send({ cmd: "stop" })) return;
    // WHY say so: a silent no-op Stop is the worst failure a stop button can have.
    studio.localError(
      "Stop did not reach Studio: this window is offline",
      "If this window had control, the rig stopped when the link dropped. Otherwise stop from the window that has control, or cut the followers' power.",
    );
  }, []);

  // Esc stops from any focus, including inside dialogs: capture phase, before anything else.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") stop(); };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [stop]);
  useEffect(() => { studio.start(); }, []);

  const cams = useStudio((s) => s.cameras);
  // Stable order: the club's usual keys first, anything else after, alphabetically.
  const keys = [...new Set([...CAMERAS, ...Object.keys(cams)])].filter(
    (k) => !Object.keys(cams).length || k in cams,
  );

  return (
    <Tooltip.Provider delayDuration={400}>
      <div className="app">
        <TopBar onStop={stop} />
        <main className="main">
          <div className="page-head">
            <h1 className="t-xl">Rig</h1>
          </div>
          <ActionBar />
          <Notices />
          <div className="rig-grid">
            <div className="col">
              <section className="panel cams-panel">
                <div className="panel-head"><h2>Cameras</h2></div>
                <div className="cams">
                  {keys.map((k) => <CameraTile key={k} name={k} />)}
                </div>
              </section>
              <JointTable />
            </div>
            <div className="col">
              <ArmsPanel />
              <MockPanel />
            </div>
          </div>
        </main>
      </div>
    </Tooltip.Provider>
  );
}

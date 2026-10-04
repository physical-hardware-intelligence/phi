import * as Tooltip from "@radix-ui/react-tooltip";
import { Suspense, lazy, useEffect, type ComponentType } from "react";
import { Boundary } from "./components/Boundary";
import { Sidebar } from "./components/Sidebar";
import { TopBar } from "./components/TopBar";
import { getTheme, setTheme, studio, useRoute, useStudio, type Route } from "./lib/studio";
import { Calibrate } from "./pages/Calibrate";
import { Checks } from "./pages/Checks";
import { Evaluate } from "./pages/Evaluate";
import { Files } from "./pages/Files";
import { Overview } from "./pages/Overview";
import { Policy } from "./pages/Policy";
import { Setup } from "./pages/Setup";
import { Teleop } from "./pages/Teleop";

// WHY lazy: the panel carries the markdown renderer, a third of the bundle, and most sessions never open it.
const AssistantPanel = lazy(() => import("./components/Assistant").then((m) => ({ default: m.AssistantPanel })));

const PAGES: Record<Route, { title: string; sub: string; el: ComponentType }> = {
  overview: { title: "Overview", sub: "Rig status, and what to do next", el: Overview },
  checks: { title: "Checks", sub: "Every read-only check of this Mac, the rig and Studio, in one run", el: Checks },
  calibrate: { title: "Calibrate", sub: "Write each arm's homing offsets and joint ranges", el: Calibrate },
  teleop: { title: "Teleoperate", sub: "Drive each follower with its leader", el: Teleop },
  policy: { title: "Run policy", sub: "Run a trained policy on the followers", el: Policy },
  evaluate: { title: "Evaluate", sub: "Judge episodes and measure the success rate", el: Evaluate },
  setup: { title: "LeRobot setup", sub: "Your rig as LeRobot reads robot-config.yaml, and the commands for it", el: Setup },
  files: { title: "Files", sub: "Serial ports, rig notes, calibration files and code on this Mac", el: Files },
};

export function App() {
  const route = useRoute();
  const assist = useStudio((s) => s.assist.open);
  // Esc to stop is registered by studio.start(), outside React, so a render crash cannot remove it.
  // Cmd+J (Ctrl+J off the Mac) opens and closes Claude from anywhere.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && !e.shiftKey && !e.altKey && e.key.toLowerCase() === "j") {
        e.preventDefault();
        studio.toggleAssistant();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
  useEffect(() => { setTheme(getTheme()); studio.start(); }, []);

  const page = PAGES[route];
  const Page = page.el;
  return (
    <Tooltip.Provider delayDuration={400}>
      <div className={`shell ${assist ? "has-assist" : ""}`}>
        <Sidebar />
        <div className="main">
          <TopBar title={page.title} sub={page.sub} onStop={() => studio.stop()} />
          <main className="content" key={route}>
            <Boundary what={`The ${page.title} page`}><Page /></Boundary>
          </main>
        </div>
        {assist && (
          <Boundary what="Claude" className="assist">
            <Suspense fallback={<aside className="assist" aria-busy />}><AssistantPanel /></Suspense>
          </Boundary>
        )}
      </div>
    </Tooltip.Provider>
  );
}

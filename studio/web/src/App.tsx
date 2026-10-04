import * as Tooltip from "@radix-ui/react-tooltip";
import { Suspense, lazy, useCallback, useEffect, type ComponentType } from "react";
import { Sidebar } from "./components/Sidebar";
import { TopBar } from "./components/TopBar";
import { getTheme, setTheme, studio, useRoute, useStudio, type Route } from "./lib/studio";
import { Calibrate } from "./pages/Calibrate";
import { Checks } from "./pages/Checks";
import { Evaluate } from "./pages/Evaluate";
import { Files } from "./pages/Files";
import { Overview } from "./pages/Overview";
import { Policy } from "./pages/Policy";
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
  files: { title: "Files", sub: "Serial ports, rig notes, calibration files and code on this Mac", el: Files },
};

export function App() {
  const route = useRoute();
  const assist = useStudio((s) => s.assist.open);
  const stop = useCallback(() => {
    if (studio.send({ cmd: "stop" })) return;
    // WHY say so: a silent no-op Stop is the worst failure a stop button can have.
    studio.localError(
      "Stop did not reach Studio: this window is offline",
      "If this window had control, the rig stopped when the link dropped. Otherwise stop from the window that has control, or cut the followers' power.",
    );
  }, []);

  // Esc stops from any page and any focus, including inside dialogs: capture phase, before anything else.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") stop(); };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [stop]);
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
          <TopBar title={page.title} sub={page.sub} onStop={stop} />
          <main className="content" key={route}>
            <Page />
          </main>
        </div>
        {assist && <Suspense fallback={<aside className="assist" aria-busy />}><AssistantPanel /></Suspense>}
      </div>
    </Tooltip.Provider>
  );
}

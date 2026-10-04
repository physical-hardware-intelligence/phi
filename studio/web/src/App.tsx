import * as Tooltip from "@radix-ui/react-tooltip";
import { useCallback, useEffect, type ComponentType } from "react";
import { Sidebar } from "./components/Sidebar";
import { TopBar } from "./components/TopBar";
import { getTheme, setTheme, studio, useRoute, type Route } from "./lib/studio";
import { Calibrate } from "./pages/Calibrate";
import { Evaluate } from "./pages/Evaluate";
import { Overview } from "./pages/Overview";
import { Policy } from "./pages/Policy";
import { Teleop } from "./pages/Teleop";

const PAGES: Record<Route, { title: string; sub: string; el: ComponentType }> = {
  overview: { title: "Overview", sub: "Rig status, and what to do next", el: Overview },
  calibrate: { title: "Calibrate", sub: "Write each arm's homing offsets and joint ranges", el: Calibrate },
  teleop: { title: "Teleoperate", sub: "Drive each follower with its leader", el: Teleop },
  policy: { title: "Run policy", sub: "Run a trained policy on the followers", el: Policy },
  evaluate: { title: "Evaluate", sub: "Judge episodes and measure the success rate", el: Evaluate },
};

export function App() {
  const route = useRoute();
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
  useEffect(() => { setTheme(getTheme()); studio.start(); }, []);

  const page = PAGES[route];
  const Page = page.el;
  return (
    <Tooltip.Provider delayDuration={400}>
      <div className="shell">
        <Sidebar />
        <div className="main">
          <TopBar title={page.title} sub={page.sub} onStop={stop} />
          <main className="content" key={route}>
            <Page />
          </main>
        </div>
      </div>
    </Tooltip.Provider>
  );
}

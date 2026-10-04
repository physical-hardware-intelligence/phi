import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import { getTheme } from "./lib/studio";
import "./styles/tokens.css";
import "./styles/base.css";
import "./styles/app.css";

// Before the first paint, so a dark-theme user never sees a light flash.
document.documentElement.dataset.theme = getTheme();

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);

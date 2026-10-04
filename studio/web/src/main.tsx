import "@fontsource-variable/inter";
import "@fontsource/geist-mono/400.css";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import "./styles/tokens.css";
import "./styles/base.css";
import "./styles/app.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);

import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App.jsx";

// Self-hosted fonts (D-12-06): only the two weights the Typography contract
// actually uses (400 body / 600 heading-and-up). Sora is display-only, so
// just 600 is loaded; Inter and JetBrains Mono each need both weights.
// Imported before index.css so the token system's font-family stacks
// resolve against faces already registered.
import "@fontsource/sora/600.css";
import "@fontsource/inter/400.css";
import "@fontsource/inter/600.css";
import "@fontsource/jetbrains-mono/400.css";
import "@fontsource/jetbrains-mono/600.css";
import "./index.css";

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <App />
  </StrictMode>
);

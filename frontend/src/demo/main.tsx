import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import Demo from "./Demo";
import "../styles.css";
import "./demo.css";

// The demo defaults to dark theme (the app's theme toggle isn't shown here).
document.documentElement.dataset.theme = "dark";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <Demo />
  </StrictMode>,
);

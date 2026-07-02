// fal platform extension: cost badges, session/balance/jobs sidebar,
// and endpoint autocomplete for the ComfyUI canvas.

import { app } from "../../scripts/app.js";
import { loadPricingMap, setupNodeBadges } from "./fal_badges.js";
import { registerSidebar } from "./fal_sidebar.js";
import { installAutocomplete } from "./fal_autocomplete.js";

// Start loading the pricing map immediately: node definitions register before
// setup() runs, and the badge drawer looks the map up lazily at draw time.
const pricingReady = loadPricingMap();

function injectStylesheet() {
  try {
    const link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = new URL("./fal_platform.css", import.meta.url).href;
    document.head.appendChild(link);
  } catch (error) {
    console.debug("[fal] stylesheet injection failed", error);
  }
}

app.registerExtension({
  name: "fal.platform",

  beforeRegisterNodeDef(nodeType, nodeData) {
    try {
      setupNodeBadges(nodeType, nodeData);
    } catch (error) {
      console.debug("[fal] badge setup failed", error);
    }
  },

  async setup() {
    injectStylesheet();
    try {
      registerSidebar(app);
    } catch (error) {
      console.debug("[fal] sidebar registration failed", error);
    }
    try {
      installAutocomplete();
    } catch (error) {
      console.debug("[fal] autocomplete install failed", error);
    }
    try {
      await pricingReady;
      app.graph?.setDirtyCanvas?.(true, true);
    } catch (error) {
      console.debug("[fal] pricing map warmup failed", error);
    }
  },
});

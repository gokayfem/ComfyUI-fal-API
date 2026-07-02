// Endpoint autocomplete for free-typed fal nodes (Any Endpoint, Submit, ...).

import { debounce, getJson } from "./fal_api.js";
import { FREE_TYPED_NODES, refreshNodeBadge } from "./fal_badges.js";

const SEARCH_DEBOUNCE_MS = 300;
const RESULT_LIMIT = 25;
const ENDPOINT_WIDGET = "endpoint_id";

let activePopup = null;

function destroyPopup() {
  if (!activePopup) return;
  try {
    activePopup.cleanup?.();
    activePopup.element.remove();
  } catch (error) {
    console.debug("[fal] popup cleanup failed", error);
  }
  activePopup = null;
}

function findTarget(canvas, value) {
  const pair = canvas?.node_widget;
  if (
    Array.isArray(pair) &&
    pair[0] &&
    pair[1]?.name === ENDPOINT_WIDGET &&
    FREE_TYPED_NODES.has(pair[0].type)
  ) {
    return pair;
  }
  const selected = Object.values(canvas?.selected_nodes || {});
  for (const node of selected) {
    if (!FREE_TYPED_NODES.has(node?.type)) continue;
    const widget = (node.widgets || []).find((w) => w?.name === ENDPOINT_WIDGET);
    if (widget && String(widget.value ?? "") === String(value ?? "")) return [node, widget];
  }
  return null;
}

function resultRow(model, apply) {
  const row = document.createElement("div");
  row.className = "fal-suggest-item";
  const title = document.createElement("span");
  title.className = "fal-suggest-title";
  title.textContent = model.title || model.endpoint_id;
  const endpoint = document.createElement("span");
  endpoint.className = "fal-suggest-endpoint";
  endpoint.textContent = model.endpoint_id;
  row.append(title, endpoint);
  if (model.label) {
    const price = document.createElement("span");
    price.className = "fal-suggest-price";
    price.textContent = model.label;
    row.append(price);
  }
  row.addEventListener("mousedown", (event) => {
    event.preventDefault();
    event.stopPropagation();
    apply(model.endpoint_id);
  });
  return row;
}

function positionPopup(popup, dialog) {
  try {
    const rect = dialog.getBoundingClientRect();
    popup.style.left = `${Math.max(4, rect.left)}px`;
    popup.style.top = `${rect.bottom + 4}px`;
  } catch (error) {
    console.debug("[fal] popup positioning failed", error);
  }
}

function attachAutocomplete(dialog, input, node, widget) {
  destroyPopup();
  const popup = document.createElement("div");
  popup.className = "fal-suggest";
  document.body.appendChild(popup);
  positionPopup(popup, dialog);

  const apply = (endpointId) => {
    try {
      widget.value = endpointId;
      input.value = endpointId;
      widget.callback?.(endpointId);
      refreshNodeBadge(node, endpointId);
      node.setDirtyCanvas?.(true, true);
    } catch (error) {
      console.debug("[fal] could not apply endpoint suggestion", error);
    }
    destroyPopup();
  };

  const search = async (query) => {
    try {
      const models = await getJson(
        `/models?q=${encodeURIComponent(query || "")}&limit=${RESULT_LIMIT}`
      );
      if (activePopup?.element !== popup) return;
      popup.replaceChildren(...(models || []).map((model) => resultRow(model, apply)));
      popup.style.display = models?.length ? "block" : "none";
    } catch (error) {
      console.debug("[fal] endpoint search failed", error);
    }
  };
  const debouncedSearch = debounce(() => search(input.value), SEARCH_DEBOUNCE_MS);

  const onInput = () => debouncedSearch();
  const onKeyDown = (event) => {
    if (event.key === "Escape" || event.key === "Enter") destroyPopup();
  };
  const onOutsideDown = (event) => {
    if (!popup.contains(event.target) && event.target !== input) destroyPopup();
  };
  input.addEventListener("input", onInput);
  input.addEventListener("keydown", onKeyDown);
  document.addEventListener("mousedown", onOutsideDown, true);
  const aliveCheck = setInterval(() => {
    if (!input.isConnected) destroyPopup();
  }, 500);

  activePopup = {
    element: popup,
    cleanup: () => {
      input.removeEventListener("input", onInput);
      input.removeEventListener("keydown", onKeyDown);
      document.removeEventListener("mousedown", onOutsideDown, true);
      clearInterval(aliveCheck);
    },
  };
  search(input.value);
}

export function installAutocomplete() {
  const canvasClass = globalThis.LGraphCanvas;
  if (!canvasClass?.prototype?.prompt) {
    console.debug("[fal] LGraphCanvas.prompt unavailable; endpoint autocomplete disabled");
    return;
  }
  const originalPrompt = canvasClass.prototype.prompt;
  canvasClass.prototype.prompt = function (title, value, callback, event, ...rest) {
    const dialog = originalPrompt.call(this, title, value, callback, event, ...rest);
    try {
      const target = findTarget(this, value);
      const input = dialog?.querySelector?.("input, textarea");
      if (target && input) attachAutocomplete(dialog, input, target[0], target[1]);
    } catch (error) {
      console.debug("[fal] autocomplete attach failed", error);
    }
    return dialog;
  };
}

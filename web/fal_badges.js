// Cost badges: a small price pill floating above every priced fal node.

import { debounce, getJson } from "./fal_api.js";

export const FREE_TYPED_NODES = new Set([
  "FalAnyEndpoint_fal",
  "FalSubmit_fal",
  "FalCostEstimator_fal",
  "FalResultByRequestId_fal",
]);

const DYNAMIC_NODE_PREFIX = "FalAPI_";
const ENDPOINT_WIDGET = "endpoint_id";
const LIVE_DEBOUNCE_MS = 500;

// node class key -> {label, per_run}; filled asynchronously.
let pricingMap = {};
// endpoint_id -> label|null for free-typed endpoint lookups.
const liveLabelCache = new Map();

export async function loadPricingMap() {
  try {
    pricingMap = (await getJson("/pricing_map")) || {};
    console.debug(`[fal] pricing map loaded (${Object.keys(pricingMap).length} nodes)`);
  } catch (error) {
    console.debug("[fal] pricing map unavailable", error);
  }
}

function titleHeight() {
  const lg = globalThis.LiteGraph;
  const height = lg && typeof lg.NODE_TITLE_HEIGHT === "number" ? lg.NODE_TITLE_HEIGHT : 30;
  return height;
}

function drawPill(node, ctx, text) {
  if (!text || node?.flags?.collapsed) return;
  ctx.save();
  try {
    ctx.font = "10px Inter, 'Segoe UI', sans-serif";
    const padX = 7;
    const height = 16;
    const width = ctx.measureText(text).width + padX * 2;
    const x = 0;
    const y = -titleHeight() - height - 6;
    ctx.beginPath();
    if (typeof ctx.roundRect === "function") {
      ctx.roundRect(x, y, width, height, height / 2);
    } else {
      ctx.rect(x, y, width, height);
    }
    ctx.fillStyle = "rgba(12, 12, 18, 0.88)";
    ctx.fill();
    ctx.lineWidth = 1;
    ctx.strokeStyle = "rgba(167, 139, 250, 0.45)";
    ctx.stroke();
    ctx.fillStyle = "#ece9fd";
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.fillText(text, x + padX, y + height / 2 + 0.5);
  } finally {
    ctx.restore();
  }
}

function labelForNode(node, typeName) {
  if (FREE_TYPED_NODES.has(typeName)) return node._falPriceLabel || null;
  return pricingMap[typeName]?.label || null;
}

export function refreshNodeBadge(node, endpointValue) {
  const endpoint = String(endpointValue ?? "").trim();
  if (!endpoint) {
    node._falPriceLabel = null;
    node.setDirtyCanvas?.(true, true);
    return;
  }
  if (liveLabelCache.has(endpoint)) {
    node._falPriceLabel = liveLabelCache.get(endpoint);
    node.setDirtyCanvas?.(true, true);
    return;
  }
  getJson(`/pricing?endpoint_id=${encodeURIComponent(endpoint)}`)
    .then((data) => {
      const label = data?.label || null;
      liveLabelCache.set(endpoint, label);
      node._falPriceLabel = label;
      node.setDirtyCanvas?.(true, true);
    })
    .catch((error) => console.debug("[fal] live pricing lookup failed", error));
}

function watchEndpointWidget(node) {
  const widget = (node.widgets || []).find((w) => w?.name === ENDPOINT_WIDGET);
  if (!widget) return;
  const refresh = debounce(() => refreshNodeBadge(node, widget.value), LIVE_DEBOUNCE_MS);
  const previousCallback = widget.callback;
  widget.callback = function (...args) {
    const result = previousCallback?.apply(this, args);
    try {
      refresh();
    } catch (error) {
      console.debug("[fal] endpoint widget watch failed", error);
    }
    return result;
  };
  refreshNodeBadge(node, widget.value);
}

function hookFreeTypedNode(nodeType) {
  const previousCreated = nodeType.prototype.onNodeCreated;
  nodeType.prototype.onNodeCreated = function (...args) {
    const result = previousCreated?.apply(this, args);
    try {
      watchEndpointWidget(this);
    } catch (error) {
      console.debug("[fal] could not watch endpoint widget", error);
    }
    return result;
  };
  const previousConfigure = nodeType.prototype.onConfigure;
  nodeType.prototype.onConfigure = function (...args) {
    const result = previousConfigure?.apply(this, args);
    try {
      const widget = (this.widgets || []).find((w) => w?.name === ENDPOINT_WIDGET);
      if (widget) refreshNodeBadge(this, widget.value);
    } catch (error) {
      console.debug("[fal] badge refresh on configure failed", error);
    }
    return result;
  };
}

export function setupNodeBadges(nodeType, nodeData) {
  const typeName = nodeData?.name;
  if (!typeName || !nodeType?.prototype) return;
  const isFreeTyped = FREE_TYPED_NODES.has(typeName);
  if (!isFreeTyped && !typeName.startsWith(DYNAMIC_NODE_PREFIX)) return;

  const previousDraw = nodeType.prototype.onDrawForeground;
  nodeType.prototype.onDrawForeground = function (ctx, ...args) {
    const result = previousDraw?.apply(this, [ctx, ...args]);
    try {
      drawPill(this, ctx, labelForNode(this, typeName));
    } catch (error) {
      console.debug("[fal] badge draw failed", error);
    }
    return result;
  };
  if (isFreeTyped) hookFreeTypedNode(nodeType);
}

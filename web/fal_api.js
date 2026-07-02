// Fetch helpers and tiny utilities for the fal platform extension. No deps.

let apiRef = null;
try {
  const mod = await import("../../scripts/api.js");
  apiRef = mod?.api ?? null;
} catch (error) {
  console.debug("[fal] scripts/api.js unavailable, falling back to fetch()", error);
}

function rawFetch(path, options) {
  if (apiRef && typeof apiRef.fetchApi === "function") {
    return apiRef.fetchApi(path, options);
  }
  return fetch(path, options);
}

export async function getJson(path) {
  const response = await rawFetch(`/fal_api${path}`);
  if (!response.ok) {
    throw new Error(`GET /fal_api${path} -> HTTP ${response.status}`);
  }
  return await response.json();
}

export async function postJson(path, body) {
  const response = await rawFetch(`/fal_api${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body ?? {}),
  });
  if (!response.ok) {
    throw new Error(`POST /fal_api${path} -> HTTP ${response.status}`);
  }
  return await response.json();
}

export function debounce(fn, delayMs) {
  let timer = null;
  return (...args) => {
    if (timer) clearTimeout(timer);
    timer = setTimeout(() => {
      timer = null;
      fn(...args);
    }, delayMs);
  };
}

export function humanAge(unixSeconds) {
  if (typeof unixSeconds !== "number") return "?";
  const seconds = Math.max(0, Date.now() / 1000 - unixSeconds);
  if (seconds < 60) return `${Math.floor(seconds)}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h`;
  return `${Math.floor(seconds / 86400)}d`;
}

export function shortEndpoint(endpoint) {
  const parts = String(endpoint || "").split("/").filter(Boolean);
  if (parts.length <= 1) return endpoint || "(unknown)";
  return parts.slice(-2).join("/");
}

export function formatUsd(value) {
  if (typeof value !== "number" || !isFinite(value)) return null;
  return `$${value.toFixed(value < 10 ? 4 : 2).replace(/\.?0+$/, "") || "0"}`;
}

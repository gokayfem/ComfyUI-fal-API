// Sidebar panel: session spend, account balance and the async job inbox.

import { formatUsd, getJson, humanAge, postJson, shortEndpoint } from "./fal_api.js";

const REFRESH_MS = 3000;
const JOB_LIMIT = 50;

let refreshTimer = null;
let panelRoot = null;

function element(tag, className, text) {
  const el = document.createElement(tag);
  if (className) el.className = className;
  if (text != null) el.textContent = text;
  return el;
}

function copyText(text, feedbackEl) {
  const done = () => {
    if (!feedbackEl) return;
    const original = feedbackEl.textContent;
    feedbackEl.textContent = "copied!";
    setTimeout(() => {
      feedbackEl.textContent = original;
    }, 900);
  };
  try {
    if (navigator.clipboard?.writeText) {
      navigator.clipboard.writeText(text).then(done, () => {});
      return;
    }
  } catch (error) {
    console.debug("[fal] clipboard copy failed", error);
  }
  done();
}

async function cancelJob(job, refresh) {
  try {
    const result = await postJson("/cancel", {
      endpoint_id: job.endpoint,
      request_id: job.request_id,
    });
    if (!result?.ok) console.debug("[fal] cancel refused", result?.error);
  } catch (error) {
    console.debug("[fal] cancel request failed", error);
  }
  refresh();
}

function jobRow(job, refresh) {
  const row = element("div", "fal-job");
  const pending = job.status === "submitted";
  row.append(element("span", "fal-job-icon", pending ? "⏳" : "✅"));

  const info = element("div", "fal-job-info");
  info.append(element("div", "fal-job-endpoint", shortEndpoint(job.endpoint)));
  const requestId = String(job.request_id || "-");
  const meta = element(
    "div",
    "fal-job-meta",
    `${humanAge(pending ? job.submitted_at : job.collected_at ?? job.submitted_at)} ago · ${requestId}`
  );
  meta.title = "Click to copy request id";
  meta.addEventListener("click", () => copyText(requestId, meta));
  info.append(meta);
  row.append(info);

  if (pending) {
    const cancel = element("button", "fal-job-cancel", "Cancel");
    cancel.addEventListener("click", () => cancelJob(job, refresh));
    row.append(cancel);
  }
  return row;
}

function buildPanel() {
  const root = element("div", "fal-panel");
  root.append(element("div", "fal-panel-title", "fal platform"));

  const stats = element("div", "fal-stats");
  const session = element("div", "fal-stat");
  session.append(element("span", "fal-stat-label", "Session"), element("span", "fal-stat-value", "…"));
  const balance = element("div", "fal-stat");
  balance.append(element("span", "fal-stat-label", "Balance"), element("span", "fal-stat-value", "…"));
  stats.append(session, balance);

  const jobsHeader = element("div", "fal-jobs-header", "Jobs");
  const jobs = element("div", "fal-jobs");
  root.append(stats, jobsHeader, jobs);
  return {
    root,
    sessionValue: session.lastChild,
    balanceValue: balance.lastChild,
    jobsHeader,
    jobs,
  };
}

function renderSession(target, data) {
  const total = formatUsd(data?.total_usd) ?? "$0";
  const calls = data?.calls ?? 0;
  target.textContent = `${total} · ${calls} call${calls === 1 ? "" : "s"}`;
}

function renderBalance(target, data) {
  const balance = formatUsd(data?.balance_usd);
  target.textContent = balance ?? "unavailable";
  target.classList.toggle("fal-muted", balance == null);
}

function renderJobs(view, data, refresh) {
  const jobs = Array.isArray(data?.jobs) ? data.jobs : [];
  const counts = data?.counts || {};
  view.jobsHeader.textContent = `Jobs · ${counts.submitted ?? 0} pending, ${counts.collected ?? 0} collected`;
  view.jobs.replaceChildren(
    ...(jobs.length
      ? jobs.map((job) => jobRow(job, refresh))
      : [element("div", "fal-muted fal-jobs-empty", "No jobs yet — queue one with Fal Submit.")])
  );
}

async function refreshPanel(view) {
  const refresh = () => refreshPanel(view).catch(() => {});
  const [session, balance, jobs] = await Promise.allSettled([
    getJson("/session"),
    getJson("/balance"),
    getJson(`/jobs?limit=${JOB_LIMIT}`),
  ]);
  try {
    if (session.status === "fulfilled") renderSession(view.sessionValue, session.value);
    if (balance.status === "fulfilled") renderBalance(view.balanceValue, balance.value);
    else renderBalance(view.balanceValue, {});
    if (jobs.status === "fulfilled") renderJobs(view, jobs.value, refresh);
  } catch (error) {
    console.debug("[fal] panel render failed", error);
  }
}

function panelVisible() {
  return !!panelRoot && panelRoot.isConnected && panelRoot.offsetParent !== null && !document.hidden;
}

function startRefreshLoop(view) {
  if (refreshTimer) clearInterval(refreshTimer);
  const tick = () => {
    if (!panelVisible()) return;
    refreshPanel(view).catch((error) => console.debug("[fal] panel refresh failed", error));
  };
  refreshTimer = setInterval(tick, REFRESH_MS);
  refreshPanel(view).catch((error) => console.debug("[fal] initial panel refresh failed", error));
}

export function mountPanel(container) {
  const view = buildPanel();
  panelRoot = view.root;
  container.replaceChildren(view.root);
  startRefreshLoop(view);
}

function mountFloatingFallback() {
  const wrapper = element("div", "fal-floating");
  const panelHost = element("div", "fal-floating-panel");
  panelHost.style.display = "none";
  const toggle = element("button", "fal-floating-toggle", "fal");
  toggle.title = "fal: session cost, balance, jobs";
  toggle.addEventListener("click", () => {
    const hidden = panelHost.style.display === "none";
    panelHost.style.display = hidden ? "block" : "none";
    if (hidden) mountPanel(panelHost);
  });
  wrapper.append(panelHost, toggle);
  document.body.appendChild(wrapper);
}

export function registerSidebar(app) {
  try {
    const manager = app?.extensionManager;
    if (manager && typeof manager.registerSidebarTab === "function") {
      manager.registerSidebarTab({
        id: "fal-platform",
        icon: "pi pi-bolt",
        title: "fal",
        tooltip: "fal: session cost, balance, jobs",
        type: "custom",
        render: (el) => {
          try {
            mountPanel(el);
          } catch (error) {
            console.debug("[fal] sidebar mount failed", error);
          }
        },
      });
      return;
    }
  } catch (error) {
    console.debug("[fal] sidebar tab registration failed", error);
  }
  try {
    mountFloatingFallback();
  } catch (error) {
    console.debug("[fal] floating panel fallback failed", error);
  }
}

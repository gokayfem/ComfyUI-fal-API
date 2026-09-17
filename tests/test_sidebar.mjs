// Exercise the real sidebar module without ComfyUI or third-party DOM libraries.
// Run: node --experimental-vm-modules --test tests/test_sidebar.mjs
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { setImmediate } from "node:timers/promises";
import test from "node:test";
import vm from "node:vm";

class Element {
  constructor(tag) {
    this.tag = tag;
    this.children = [];
    this.listeners = {};
    this.isConnected = true;
    this.disabled = false;
    this.textContent = "";
    this.classList = { toggle() {} };
  }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; }
  addEventListener(event, listener) { this.listeners[event] = listener; }
  get lastChild() { return this.children.at(-1); }
  find(className) {
    if (this.className === className) return this;
    return this.children.map((child) => child.find(className)).find(Boolean);
  }
}

async function mount(status, { failPost = false, refreshOK = true } = {}) {
  const calls = [];
  const context = vm.createContext({
    document: { createElement: (tag) => new Element(tag), hidden: false },
    console: { debug() {} },
    setInterval: () => 1,
    clearInterval: () => {},
    setTimeout: (callback) => { callback(); },
  });
  const api = new vm.SyntheticModule(["formatUsd", "getJson", "humanAge", "postJson", "shortEndpoint"], function () {
    this.setExport("formatUsd", () => "$0");
    this.setExport("humanAge", () => "now");
    this.setExport("shortEndpoint", (value) => value);
    this.setExport("getJson", async (path) => {
      if (path === "/registry_status") {
        if (status instanceof Error) throw status;
        return status;
      }
      if (path === "/registry_refresh") return { running: false, finished_at: 1, ok: refreshOK, message: "Validation failed" };
      return {};
    });
    this.setExport("postJson", async (path) => {
      calls.push(path);
      if (failPost) throw new Error("offline");
      return { started: true, running: true };
    });
  }, { context });
  const source = await readFile(new URL("../web/fal_sidebar.js", import.meta.url), "utf8");
  const sidebar = new vm.SourceTextModule(source, { context });
  await sidebar.link(() => api);
  await sidebar.evaluate();
  const root = new Element("div");
  sidebar.namespace.mountPanel(root);
  await setImmediate();
  return { root, calls };
}

for (const status of [{ new_count: 0, new_models: [] }, new Error("catalog unavailable"), { new_count: 1, new_models: [{ title: "New model" }] }]) {
  test(`refresh is available with status ${JSON.stringify(status)}`, async () => {
    const { root, calls } = await mount(status);
    const button = root.find("fal-registry-refresh");
    assert.ok(button, "Existing models need schema refresh even with no new IDs or an unavailable catalog check");
    button.listeners.click();
    await setImmediate();
    assert.deepEqual(calls, ["/registry_refresh"]);
    assert.match(root.find("fal-registry-done").textContent, /updated controls/);
    assert.equal(button.disabled, false);
  });
}

for (const options of [{ failPost: true }, { refreshOK: false }]) {
  test(`failed refresh can be retried: ${JSON.stringify(options)}`, async () => {
    const { root, calls } = await mount({ new_count: 0 }, options);
    const button = root.find("fal-registry-refresh");
    button.listeners.click();
    await setImmediate();
    assert.ok(root.find("fal-registry-error"));
    assert.equal(button.disabled, false);
    button.listeners.click();
    await setImmediate();
    assert.equal(calls.length, 2);
  });
}

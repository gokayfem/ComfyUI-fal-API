import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import vm from "node:vm";

const source = await readFile(new URL("../web/fal_suggestions.js", import.meta.url), "utf8");
const module = new vm.SourceTextModule(source, { context: vm.createContext({ console }) });
await module.link(() => { throw new Error("Unexpected import"); });
await module.evaluate();
const { setupSuggestedWidgets } = module.namespace;

function setup(name = "mode", defaultValue = "balanced") {
  const callbackValues = [];
  const promptCalls = [];
  class Node {
    constructor() {
      this.widgets = [
        { name: "prompt", type: "text", value: "test" },
        { name, type: "text", value: defaultValue, options: {}, callback: (value) => callbackValues.push(value) },
        { name: "seed", type: "number", value: 42 },
      ];
    }
    onNodeCreated() { this.originalCalled = true; return "original-result"; }
    addWidget(type, name, value, callback, options) {
      const widget = { type, name, value, callback, options };
      this.widgets.push(widget);
      return widget;
    }
    setDirtyCanvas() {}
  }
  const data = {
    name: "FalAPI_any-future-model",
    input: { optional: { [name]: ["STRING", { fal_suggestions: ["balanced", "quality"] }] } },
  };
  setupSuggestedWidgets(Node, data, { canvas: { prompt: (...args) => promptCalls.push(args) } });
  const node = new Node();
  assert.equal(node.onNodeCreated(), "original-result");
  return { node, widget: node.widgets[1], promptCalls, callbackValues, data };
}

for (const name of ["mode", "voice", "language", "model_id"]) {
  test(`generic suggested ${name} keeps widget order, default and STRING socket`, () => {
    const { node, widget, data } = setup(name, "custom-default");
    assert.equal(widget.type, "combo");
    assert.equal(widget.value, "custom-default");
    assert.deepEqual(node.widgets.map((w) => w.name), ["prompt", name, "seed"]);
    assert.equal(node.widgets[2].value, 42);
    assert.equal(data.input.optional[name][0], "STRING");
    assert.equal(node.originalCalled, true);
  });
}

test("saved values missing from examples survive load and serialization", () => {
  const { node, widget } = setup();
  widget.value = "older-workflow-value";
  node.onConfigure({});
  assert.ok(widget.options.values.includes("older-workflow-value"));
  assert.equal(node.widgets.map((w) => w.value)[1], "older-workflow-value");
});

test("suggestion selection and custom entry preserve original callbacks", () => {
  const { widget, promptCalls, callbackValues } = setup();
  widget.value = "quality";
  widget.callback("quality");
  assert.deepEqual(callbackValues, ["quality"]);
  const custom = widget.options.values.at(-1);
  widget.value = custom;
  widget.callback(custom);
  assert.equal(widget.value, "quality", "UI-only label must never reach a queued API request");
  assert.equal(promptCalls[0][1], "quality");
  promptCalls[0][2]("new-mode-from-api");
  assert.equal(widget.value, "new-mode-from-api");
  assert.ok(widget.options.values.includes("new-mode-from-api"));
  assert.deepEqual(callbackValues, ["quality", "new-mode-from-api"]);
});

test("canceling custom entry leaves the current value intact", () => {
  const { widget, promptCalls } = setup();
  const custom = widget.options.values.at(-1);
  widget.value = custom;
  widget.callback(custom);
  promptCalls[0][2](null);
  assert.equal(widget.value, "balanced");
});

test("nodes without suggestion metadata are unchanged", () => {
  class Node {}
  setupSuggestedWidgets(Node, { name: "FalAPI_plain-model", input: { required: { prompt: ["STRING", {}] } } }, {});
  assert.equal(Node.prototype.onNodeCreated, undefined);
});

// Suggested string values are an editable dropdown, not an API enum.
// Keep the original input name, STRING socket and serialized widget position.
const CUSTOM = "Enter custom value…";

function replaceStringWidget(node, name, inputOptions, app) {
  const suggestions = inputOptions.fal_suggestions;
  const index = (node.widgets || []).findIndex((widget) => widget.name === name);
  if (index < 0 || !suggestions.length) return;
  const original = node.widgets[index];
  const originalSize = node.size ? [...node.size] : null;
  const values = [...new Set(suggestions)];
  let lastValue = original.value ?? "";
  let widget;
  const choices = (value) => [...new Set([...values, value]), CUSTOM];
  const sync = () => {
    if (widget.value !== CUSTOM) lastValue = widget.value ?? "";
    widget.options.values = choices(lastValue);
  };

  const apply = (value, ...args) => {
    if (value == null) return; // Cancel leaves the previous value intact.
    lastValue = String(value);
    widget.value = lastValue;
    sync();
    original.callback?.call(widget, lastValue, ...args);
    node.setDirtyCanvas?.(true, true);
  };
  const onSelect = (value, ...args) => {
    if (value !== CUSTOM) {
      apply(value, ...args);
      return;
    }
    // Never serialize or submit the UI-only custom-entry label.
    widget.value = lastValue;
    if (typeof app?.canvas?.prompt === "function") {
      app.canvas.prompt(`Custom ${name}`, lastValue, (text) => apply(text, ...args), args.at(-1));
    } else {
      apply(globalThis.prompt?.(`Custom ${name}`, lastValue), ...args);
    }
  };

  const tooltip = `${inputOptions.tooltip || original.tooltip || ""} Suggested values; choose '${CUSTOM}' to enter any other value.`.trim();
  widget = node.addWidget("combo", name, lastValue, onSelect, {
    ...original.options,
    values: choices(lastValue),
    tooltip,
  });
  widget.tooltip = tooltip;
  widget._falSyncSuggestions = sync;
  widget.value = lastValue;
  // addWidget appends. Move it into the old slot so positional workflow values
  // and every following widget keep their existing meaning.
  const appendedIndex = node.widgets.indexOf(widget);
  node.widgets.splice(appendedIndex, 1);
  node.widgets.splice(index, 1, widget);
  if (original.label != null) widget.label = original.label;
  if (original.serializeValue) widget.serializeValue = original.serializeValue.bind(widget);
  original.onRemove?.();
  if (originalSize) node.setSize?.(originalSize);
}

export function setupSuggestedWidgets(nodeType, nodeData, app) {
  if (!nodeData?.name?.startsWith("FalAPI_")) return;
  const fields = Object.entries({ ...nodeData.input?.required, ...nodeData.input?.optional })
    .filter(([, spec]) => spec?.[0] === "STRING" && Array.isArray(spec?.[1]?.fal_suggestions))
    .map(([name, spec]) => [name, spec[1]]);
  if (!fields.length) return;
  const originalCreated = nodeType.prototype.onNodeCreated;
  nodeType.prototype.onNodeCreated = function (...args) {
    const result = originalCreated?.apply(this, args);
    for (const [name, inputOptions] of fields) {
      try {
        replaceStringWidget(this, name, inputOptions, app);
      } catch (error) {
        console.debug(`[fal] suggestion widget failed for ${name}`, error);
      }
    }
    return result;
  };
  const originalConfigure = nodeType.prototype.onConfigure;
  nodeType.prototype.onConfigure = function (...args) {
    const result = originalConfigure?.apply(this, args);
    // Configuration loads the old positional values after node creation.
    for (const widget of this.widgets || []) widget._falSyncSuggestions?.();
    return result;
  };
}

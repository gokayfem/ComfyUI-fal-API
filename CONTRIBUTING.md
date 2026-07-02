# Contributing to ComfyUI-fal-API

Thanks for helping! Before you write anything, read this — it will probably save you the PR entirely.

## A new fal model does NOT need code

Historically, adding a model to this pack meant hand-writing a node class. **That is no longer how it works.** Every live public model on fal gets a node automatically, generated at ComfyUI startup from the committed snapshot at `data/fal_registry.json`. No node class, no mapping entry, no code.

The snapshot stays fresh two ways:

- A **weekly GitHub Action** (`.github/workflows/registry-refresh.yml`) rebuilds the registry and opens a PR.
- Anyone can run it locally: `python scripts/build_registry.py --out data/fal_registry.json` (then `python scripts/build_readme.py` to regenerate [MODELS.md](MODELS.md)).

To check whether a model is already covered:

```bash
grep '"endpoint_id": "fal-ai/your/endpoint"' data/fal_registry.json
# or browse MODELS.md, or search the node browser in ComfyUI
```

If a model is live on [fal.ai/models](https://fal.ai/models) but missing from the snapshot, rerun `scripts/build_registry.py` — if it's *still* missing, open an issue with the endpoint id. And if you need a model **right now**, the **Fal Any Endpoint (fal)** node calls any endpoint by id without any registry entry at all.

So: **please don't open a PR that adds a node class for a new model.** It will be redundant the moment the registry refreshes.

## Want a model promoted or renamed? Edit `data/featured_models.json`

When a model deserves curation — a spot in the **FAL/Featured** menu tier or a friendlier display name — add its endpoint to `data/featured_models.json` (featured tier + display-name override). That's the whole change: one JSON entry, not a new node class.

## When a hand-written node IS justified

A curated node earns its place only when the generated node genuinely can't express the UX:

- **Multi-endpoint orchestration** — one node fanning out to several endpoints (e.g. Combined Video Generation).
- **Special input ergonomics** — first/last-frame image pairing, unified T2V/I2V dispatch, LoRA slots with per-slot scales.

If you're writing one, the rules are non-negotiable:

1. **Import only from the `.fal_utils` facade** (`from .fal_utils import ApiHandler, FalConfig, ImageUtils, ResultProcessor, ...`) — never reach into `nodes/utils/` internals or call `fal_client` directly. The facade gives you the result cache, spend guard, session ledger, and error handling for free.
2. **Raise errors — no silent fallbacks.** Never return blank images or `"Error: ..."` strings; let `ApiHandler` surface fal's actual error message.
3. **Tooltips on every input.** Users should never have to guess a parameter.
4. **Never change existing node keys, input names, or output signatures.** Existing user workflows reference them forever. `tests/legacy_node_keys.json` is the snapshot of keys that must never be removed or renamed, and `tests/test_mappings.py` fails the suite if one disappears. New inputs must be optional with backward-compatible defaults.
5. **Add tests** alongside the existing ones in `tests/`.

## Dev setup

```bash
pip install -r requirements.txt
python -m pytest tests     # the suite MUST pass
ruff check .               # lint, same as CI
```

CI runs both on every PR (Python 3.10 and 3.12). The most important test to understand is the **compatibility snapshot**: `tests/test_mappings.py` asserts that every node key recorded in `tests/legacy_node_keys.json` still registers. If your change makes it fail, the fix is to restore the key — not to edit the snapshot.

## Architecture map

```
scripts/build_registry.py     queries fal's platform APIs → writes the snapshot
data/fal_registry.json        committed model catalog (~1,391 models)
data/featured_models.json     curation: featured tier + display-name overrides
scripts/build_readme.py       renders MODELS.md from the snapshot

nodes/dynamic/                the auto-generated node machinery
  registry_loader.py          reads the snapshot, applies [dynamic_nodes] config;
                              never raises — failures degrade to curated-only
  factory.py                  builds one node class per model, in memory
  schema_to_inputs.py         registry input specs → ComfyUI INPUT_TYPES (+ tooltips)
  arguments.py                widget/socket values → API arguments (uploads media)
  outputs.py                  API result → IMAGE / VIDEO / AUDIO / result_json
  any_endpoint.py             the generic "call any endpoint by id" node

nodes/*.py                    curated hand-written nodes (image, video, llm, vlm,
                              trainer, upscaler, util_*)
nodes/fal_utils.py            import facade — node modules import ONLY from here
nodes/utils/                  the implementations behind the facade: api, config,
                              pricing, result_cache, ledger, billing (spend guard),
                              job_store, media, archive, errors, logger
nodes/platform_node.py        platform nodes (Submit/Collect, costs, request ids)
nodes/inbox_node.py           durable job inbox
nodes/billing_node.py         account balance
nodes/server_routes.py        HTTP endpoints backing the frontend extension
web/                          ComfyUI frontend: cost badges, fal sidebar,
                              endpoint autocomplete
tests/                        pytest suite, incl. the legacy_node_keys.json snapshot
```

## PR checklist

- [ ] Not a hand-written node for a single new model (registry covers it — see above)
- [ ] `python -m pytest tests` passes locally
- [ ] `ruff check .` is clean
- [ ] No existing node keys, inputs, or outputs changed
- [ ] New curated node (if truly justified): uses `.fal_utils`, raises errors, has tooltips and tests
- [ ] No secrets, no `config.ini`, no generated artifacts in the diff

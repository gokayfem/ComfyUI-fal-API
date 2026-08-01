# ComfyUI-fal-API

**Every fal model in ComfyUI, one API key.**

Custom nodes that bring the entire [fal.ai](https://fal.ai) catalog into ComfyUI: ~90 curated hand-written nodes for the most popular models, plus ~1,400 auto-generated nodes covering every live public model on fal — image, video, audio, 3D, LLMs and more. One `FAL_KEY` unlocks all of them. With a persistent result cache (never pay for the same call twice), spend guards, async fan-out, and zero-I/O fal→fal chaining. The exact current catalog lives in [MODELS.md](MODELS.md).

## Table of Contents

- [What's New](#whats-new)
- [Installation](#installation)
- [Configuration](#configuration)
- [Screenshots](#screenshots)
- [Curated Nodes](#curated-nodes)
- [Auto-Generated Nodes](#auto-generated-nodes)
- [Platform Utilities](#platform-utilities)
- [Utility Nodes](#utility-nodes)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## What's New

### 2.5

- **Async execution** — fal calls run asynchronously on modern ComfyUI, so the UI stays responsive while jobs are in flight.
- **Typed builder nodes** — build LoRA lists and reference-element configs with dedicated, connectable nodes instead of hand-typed JSON.
- **Featured tier** — hand-picked models surface under **FAL/Featured** in the node menu, and models superseded by newer versions are flagged.
- **Registry freshness** — the pack warns when the committed model registry snapshot is stale, and you can refresh it from the fal sidebar without leaving ComfyUI.
- **Automatic daily catalog updates** — a guarded GitHub workflow validates and commits new fal endpoints without waiting for a manual PR merge. Missing historical endpoints remain registered under **FAL/Compatibility** so saved workflows still load.
- **MODELS.md** — the full ~1,400-model catalog moved out of this README into [MODELS.md](MODELS.md).

### 2.4

Twenty utility nodes under `FAL/Utils` covering everything between your assets and a fal endpoint — dataset prep (images → captioned training ZIP), media loading from URLs/folders, video trim/concat/mux/frame-extract, image grids and resizing, and JSON/prompt/text data helpers. See [Utility Nodes](#utility-nodes).

### 2.3

Durable **job inbox** (queued jobs survive ComfyUI restarts), **provenance receipts** (saved outputs carry endpoint + request id in a sidecar/PNG chunk and can be re-materialized for free), and the in-canvas web extension: cost badges above nodes, a fal sidebar with spend/balance/live jobs, and endpoint autocomplete.

### 2.2

**Persistent result cache** — identical fal calls are served from disk, free and instant, across restarts; uploads are deduplicated too. **Spend guard** — refuse to submit past a session budget or below a balance floor, before money moves. **URL passthrough** — wire a fal node's URL output into the next node's `*_direct_url` input and intermediate media never touches your machine.

### 2.1

Platform utilities under `FAL/Platform`: **Fal Submit + Fal Collect** for parallel fan-out (five video jobs run concurrently on fal, not back-to-back), **Fal Result by Request ID** to re-fetch any past result without re-paying, **Fal Cost Estimator**, and **Fal Session Costs**.

### 2.0

Every live public model on fal became a node: ~1,400 auto-generated nodes built at startup from the committed `data/fal_registry.json`, with native `IMAGE`/`VIDEO`/`AUDIO` sockets, schema-derived tooltips, and pricing in the help panel. Plus the generic **Fal Any Endpoint** node, visible errors (failed calls raise fal's actual error message instead of silently returning blanks), progress + cancellation, and full backward compatibility for all ~90 curated nodes.

## Installation

The recommended installation method is **ComfyUI Manager**: search for
`ComfyUI-fal-API`, install it, and restart ComfyUI. Manager installs the
dependencies into the same Python environment ComfyUI uses.

For a manual installation:

1. Navigate to your ComfyUI custom-nodes directory and clone this repository:
   ```
   cd custom_nodes
   git clone https://github.com/gokayfem/ComfyUI-fal-API.git
   cd ComfyUI-fal-API
   ```
2. Install the dependencies with **ComfyUI's Python**, not an unrelated system
   `pip`:
   ```
   python -m pip install -r requirements.txt
   ```
   From the root of **ComfyUI Windows Portable**, use its embedded interpreter:
   ```powershell
   .\python_embeded\python.exe -m pip install -r .\ComfyUI\custom_nodes\ComfyUI-fal-API\requirements.txt
   ```
3. Configure your API key (below) and restart ComfyUI. Curated nodes appear under the **FAL** category, auto-generated nodes under **FAL/Models/&lt;category&gt;** (e.g. `FAL/Models/text-to-image`), and hand-picked models under **FAL/Featured** — or just search for any model by name.

## Configuration

1. Get your fal API key from [fal.ai](https://fal.ai/dashboard/keys)
2. Copy `config.ini.example` to `config.ini` inside `custom_nodes/ComfyUI-fal-API` (`config.ini` is gitignored, so your key never ends up in a commit)
3. Replace `<your_fal_api_key_here>` with your actual fal API key — or set the `FAL_KEY` environment variable instead:
   ```bash
   export FAL_KEY=your_actual_api_key
   ```

### config.ini reference

All sections besides `[API]` are optional; defaults shown.

```ini
[API]
FAL_KEY = your_actual_api_key

[dynamic_nodes]
; Set to false to load only the curated hand-written nodes.
enabled = true
; Comma-separated category filter; leave unset to load everything.
; categories = text-to-image,image-to-video

[cache]
; Persistent result cache: identical fal calls are served from disk (free)
; across ComfyUI restarts. Bypass per node with force_rerun.
enabled = true
ttl_days = 7
max_entries = 5000

[spend_guard]
; Refuse to submit jobs once the session's estimated spend reaches the
; budget, or when the account balance falls below the floor. 0 = disabled.
session_budget_usd = 0
min_balance_usd = 0

[archive]
; Safety caps for the dataset/folder → ZIP upload utilities.
max_files = 5000
max_total_mb = 2048

[registry]
; On startup a background thread compares the local model registry
; against fal's live catalog and logs how many new models are available;
; the fal sidebar shows them with a one-click refresh (restart required).
; startup_check = true

[MODELS.md](MODELS.md).**

Find them in the node browser under `FAL/Models/<category>`, or search by model name. Node keys are `FalAPI_<endpoint-id>` (slashes → dashes), so workflows stay stable across registry refreshes. Each generated node gives you:

- **Native inputs/outputs** — `IMAGE`/`VIDEO`/`AUDIO` sockets; connected media is uploaded to fal automatically, and video/audio/image models return native ComfyUI types (JSON-ish models return the raw result string).
- **Tooltips + pricing** — hover any input for fal's own parameter docs; the help panel shows the model's current pricing.
- **Seed semantics** — `seed = -1` means "random / omit seed".
- **force_rerun** — bypass result caching to re-roll identical inputs.

**Fal Any Endpoint (fal)** is the escape hatch: one generic node that calls *any* fal endpoint by id with free-form JSON arguments plus optional image/video/audio inputs (uploaded and merged into the matching keys). Outputs are extracted as `IMAGE`/`VIDEO`/`AUDIO`, with the raw result always available as `result_json`. Even brand-new models work the day they launch.

**Keeping the catalog fresh:** a daily GitHub Action builds a candidate registry, validates its structure and endpoint changes, regenerates [MODELS.md](MODELS.md), and commits the result when it is safe. Large unexpected catalog changes are blocked rather than published. Endpoints missing from a refresh are retained under **FAL/Compatibility** instead of deleting their node keys and breaking saved workflows. You can also run `python scripts/build_registry.py` yourself (then `python scripts/build_readme.py`), or use the refresh button in the fal sidebar. Install updates through ComfyUI Manager to receive the latest committed snapshot; locally refreshed nodes appear after restarting ComfyUI. If ~1,400 extra nodes is more than you want, the `[dynamic_nodes]` config section disables or filters them — see [Configuration](#configuration).

## Platform Utilities

Nodes built on fal's platform primitives (queue, request ids, per-model pricing), under `FAL/Platform`:

| Node | What it does |
| --- | --- |
| **Fal Submit** / **Fal Collect** | Queue a job on any endpoint and collect it later — wire N Submits into N Collects and all N jobs run **in parallel** on fal, so the graph takes as long as the slowest one, not the sum. |
| **Fal Result by Request ID** | Paste any past request id (console log, sidebar, or [fal dashboard](https://fal.ai/dashboard/requests)) to re-fetch its result **without re-generating or re-paying**. |
| **Fal Cost Estimator** | Endpoint id + run count → cost report and `total_usd` float from fal's published pricing, *before* you queue anything. |
| **Fal Session Costs** | Running ledger of every fal call this session (endpoint, duration, request id, estimated cost), with optional reset. |
| **Fal Account Balance** | Your live fal balance (needs an admin-scoped key; scoped keys degrade gracefully). |
| **Fal Job Inbox** | Every Submit is journaled to disk, so queued jobs **survive ComfyUI restarts** — lists pending/collected jobs and outputs the newest pending `request_id` + `endpoint_id`. |
| **Fal Save Media from URL** | fal result URLs eventually expire; this downloads any result into your `output/` directory and writes a provenance receipt (`<file>.fal.json` sidecar + PNG text chunk with endpoint, request id, source URL). |
| **Fal Provenance from File** | Read a saved file's receipt back into `endpoint_id` + `request_id` — re-materialize a generation for free, months later. |

Behind the nodes, three always-on platform features:

- **Persistent result cache** — identical fal calls are served from a disk cache, free and instant, across restarts; input uploads are deduplicated. Bypass per node with `force_rerun`; tune via `[cache]`.
- **Spend guard** — with `[spend_guard]` configured, the pack refuses to submit once estimated session spend hits your budget or your balance drops below the floor — the node errors *before* money moves.
- **URL passthrough** — every generated node's media input has a `*_direct_url` twin and image nodes output `image_urls`; chain fal→fal by URL and intermediate media never touches your machine.

And a web extension (degrades silently on older ComfyUI): **cost badges** above every priced fal node with live estimates on free-typed endpoint fields, a **fal sidebar** (session spend, balance, live job list with per-job Cancel, registry freshness/refresh), and **endpoint autocomplete** with prices when editing any `endpoint_id` field.

## Utility Nodes

Twenty nodes under `FAL/Utils` covering everything between your assets and a fal endpoint — no other packs needed:

| Category | Nodes |
| --- | --- |
| `FAL/Utils/Dataset` | *Images → Training ZIP URL* (standard LoRA caption layout), *Folder → ZIP URL*, *Video → Frame Dataset ZIP URL*, *Batch Caption Images* (parallel VLM captioning) |
| `FAL/Utils/Load` | *Load Image from URL* (multi-URL batching), *Load Audio from URL*, *Load Image Folder*, *Upload Folder as ZIP URL* |
| `FAL/Utils/Video` | *Extract Frames* (efficient last-frame seek — chain into image-to-video for endless extension), *Trim* (keyframe remux, no re-encode), *Concat* (auto resolution/fps normalize), *Mux Audio + Video*, *Video → Audio* |
| `FAL/Utils/Image` | *Image Grid with Labels*, *Resize to fal Preset* (cover/contain/stretch to standard image_size dims), *Image ↔ Base64* |
| `FAL/Utils/Data` | *JSON Extract* (dot/bracket path queries against any `result_json`), *Prompt Lines* (cycling line picker), *Text Template* |

The full LoRA-training pipeline needs nothing else: Load Image Folder → Batch Caption → Images→ZIP → any trainer node.

## Troubleshooting

1. Ensure you have the latest version of ComfyUI installed
2. Update this custom node package:
   ```
   cd custom_nodes/ComfyUI-fal-API
   git pull
   python -m pip install -r requirements.txt
   ```
3. **Windows Portable Python is blocked or no longer starts after installing a node?**
   Do not keep rerunning `pip`. From the portable root, first check the exact
   interpreter and dependency state:
   ```powershell
   .\python_embeded\python.exe -c "import sys; print(sys.executable); print(sys.version)"
   .\python_embeded\python.exe -m pip check
   ```
   This project installs Python packages only; it does not replace or modify
   `python.exe`. If the first command itself is blocked or the executable was
   quarantined, review Windows Security **Protection history**. Restore a file
   only when the portable archive came from the official ComfyUI release, or
   re-extract a clean official portable build and move your `models`, `input`,
   `output`, and `user` data across. Avoid disabling antivirus globally. Then
   reinstall this node through ComfyUI Manager, or use the exact embedded-
   interpreter requirements command from the Installation section.
4. **Dynamic nodes not appearing?** Check the ComfyUI console for a line like `Registered N dynamic fal nodes` at startup. If it says the nodes are disabled, remove `enabled = false` from the `[dynamic_nodes]` section of your `config.ini` (and check the `categories` filter isn't excluding what you're looking for). Any registry loading error is also printed there.
5. **`VIDEO` output is `None` or video sockets are missing?** Update ComfyUI — native `VIDEO`/`AUDIO` types require a recent ComfyUI version.
6. **API calls failing?** Failed fal requests raise visible errors that include fal's actual error message (validation issues, content policy, quota). Read the error text in ComfyUI — it usually tells you exactly which parameter to fix.

## Contributing

Contributions are welcome — but note that **a new fal model usually needs no code at all**: it appears automatically via the registry. Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a PR; it covers when a hand-written node is (and isn't) justified, the compatibility rules, and dev setup.

<details>
<summary><strong>Cite this project</strong></summary>

If ComfyUI-fal-API supports your work, please cite the software. GitHub also
provides ready-to-copy APA and BibTeX entries via **Cite this repository**.

```bibtex
@software{Aydogan_ComfyUI_fal_API_2026,
  author  = {Aydoğan, Gökay},
  title   = {ComfyUI-fal-API},
  version = {2.5.0},
  year    = {2026},
  url     = {https://github.com/gokayfem/ComfyUI-fal-API}
}
```

[ORCID](https://orcid.org/0000-0002-2343-9433) · [Citation metadata](CITATION.cff)

</details>

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

If you encounter any issues or have questions, please open an issue on the [GitHub repository](https://github.com/gokayfem/ComfyUI-fal-API/issues).

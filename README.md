# ComfyUI-fal-API

**Every fal model in ComfyUI, one API key.**

Custom nodes that bring the entire [fal.ai](https://fal.ai) catalog into ComfyUI: ~90 curated hand-written nodes for the most popular models, plus ~1,391 auto-generated nodes covering every live public model on fal — image, video, audio, 3D, LLMs and more. One `FAL_KEY` unlocks all of them.

## Table of Contents

- [What's New in 2.0](#whats-new-in-20)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Curated Nodes](#curated-nodes)
  - [Image Generation](#image-generation)
  - [Video Generation](#video-generation)
  - [Language Models (LLMs)](#language-models-llms)
  - [Vision Language Models (VLMs)](#vision-language-models-vlms)
- [Auto-Generated Nodes](#auto-generated-nodes)
- [Generated Model List](#generated-model-list)
- [Registry Maintenance](#registry-maintenance)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## What's New in 2.0

- **~1,391 auto-generated nodes** — every live public model on fal, generated at startup from the committed `data/fal_registry.json`. Roughly: 537 video, ~565 image, 115 audio, 47 3D/file, 20 text, 107 other/JSON models. Find them in the node browser under `FAL/Models/<category>`.
- **Native ComfyUI types everywhere** — auto-generated nodes take native `IMAGE`/`VIDEO`/`AUDIO` inputs (uploaded to fal automatically) and return native `VIDEO`/`AUDIO` outputs. Tooltips come straight from the API schemas, and each node's help panel shows the model's pricing.
- **Fal Any Endpoint (fal)** — a generic node that calls *any* fal endpoint by id with free-form JSON arguments plus optional image/video/audio inputs. Even brand-new models work the day they launch.
- **Visible errors (breaking-ish change)** — failed API calls now raise real errors in ComfyUI with fal's actual error message (validation issues, content policy, quota), instead of silently returning blank images or `"Error:"` strings. If your workflow relied on failures passing through quietly, it will now stop at the failing node.
- **Progress and cancellation** — queue position and model logs stream to the console, and ComfyUI's Cancel button interrupts in-flight fal requests.
- **All ~90 curated nodes unchanged** — fully backward compatible. Bonus fixes: the GPT-Image 2 / GPT-Image 2 Edit nodes now actually load, and a new **Video from URL → VIDEO (fal)** node bridges URL outputs into ComfyUI's native `VIDEO` type.
- **Config options** — `config.ini.example` is now provided (your `config.ini` stays gitignored), and an optional `[dynamic_nodes]` section lets you disable or filter the auto-generated nodes.
- **Dependencies fixed** — `pyproject.toml` now declares everything needed (`opencv-python` etc.); requires `fal-client>=1.0`.

## Installation

1. Navigate to your ComfyUI custom nodes directory:
   ```
   cd custom_nodes
   ```

2. Clone this repository:
   ```
   git clone https://github.com/gokayfem/ComfyUI-fal-API.git
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Configuration

1. Get your fal API key from [fal.ai](https://fal.ai/dashboard/keys)

2. Copy `config.ini.example` to `config.ini` inside `custom_nodes/ComfyUI-fal-API` (the `config.ini` file is gitignored, so your key never ends up in a commit)

3. Replace `<your_fal_api_key_here>` with your actual fal API key:
   ```ini
   [API]
   FAL_KEY = your_actual_api_key
   ```

4. Alternatively, you can set the FAL_KEY environment variable:
   ```bash
   export FAL_KEY=your_actual_api_key
   ```

### Optional: dynamic node settings

Add a `[dynamic_nodes]` section to `config.ini` to control the auto-generated nodes:

```ini
[dynamic_nodes]
# Set to false to load only the curated hand-written nodes
enabled = true
# Comma-separated category filter; leave unset to load everything
categories = text-to-image,image-to-video
```

## Usage

After installation and configuration, restart ComfyUI. The curated nodes appear in the node browser under the **FAL** category, and the auto-generated nodes under **FAL/Models/&lt;category&gt;** (e.g. `FAL/Models/text-to-image`). You can also just search for a model name — every auto-generated node is named after its fal model.

## Curated Nodes

Hand-written nodes with carefully tuned inputs for the most popular models. All of these are unchanged and backward compatible in 2.0.

### Image Generation

- **Flux Pro (fal)**: Generate high-quality images using the Flux Pro model
- **Flux Dev (fal)**: Use the development version of Flux for image generation
- **Flux Schnell (fal)**: Fast image generation with Flux Schnell
- **Flux Pro 1.1 (fal)**: Latest version of Flux Pro for image generation
- **Flux Pro 1 Fill (fal)**: Image-to-image generation with mask-based fill capabilities
- **Flux Ultra (fal)**: Ultra-high quality image generation with advanced controls
- **Flux General (fal)**: ControlNets, Ipadapters, Loras for Flux Dev
- **Flux LoRA (fal)**: Flux with dual LoRA support for custom styles
- **Flux Pro Kontext (fal)**: Context-aware single image-to-image generation with max_quality toggle
- **Flux Pro Kontext Multi (fal)**: Multi-image composition (2-4 images) with context awareness and max_quality toggle
- **Flux Pro Kontext Text-to-Image (fal)**: Text-to-image with aspect ratio controls and max_quality toggle
- **Recraft V3 (fal)**: Professional design generation with multiple style options
- **Sana (fal)**: High-quality image synthesis with ultra-high resolution support
- **HiDream Full (fal)**: Advanced image generation with comprehensive parameter control
- **Ideogram v3 (fal)**: Advanced text-to-image generation with typography support
- **Clarity Upscaler (fal)**: Clarity upscaler for upscaling images with high very fidelity
- **Seedvr Upscaler (fal)**: Use SeedVR2 to upscale your images
- **Imagen4 Preview (fal)**: Use Imagen4 (Preview version) to generate images
- **Qwen Image Edit (fal)**: Use Qwen to edit images
- **Qwen Image Edit Plus with LoRAs (fal)**: Use Qwen Image Edit Plus with LoRA support to edit images
- **SeedEdit 3.0 (fal)**: Use SeedEdit 3.0 to edit images
- **Seedream 4.0 Edit (fal)**: Use Seedream 4.0 to edit images
- **Nano Banana Text-to-Image (fal)**: Use Nano Banana to generate images
- **Nano Banana Edit (fal)**: Use Nano Banana to edit images
- **Nano Banana Pro (fal)**: Unified node for both text-to-image and image editing with Nano Banana Pro
- **Nano Banana 2 (fal)**: Unified node for text-to-image and image editing with Nano Banana 2 (Gemini 3.1 Flash Image) with multi-resolution (0.5K-4K) and optional web search grounding
- **Reve Text-to-Image (fal)**: Use Reve's image model to generate images
- **Dreamina v3.1 Text-to-Image (fal)**: Use Dreamina v3.1 to generate images
- **GPT-Image 1.5 (fal)**: High-fidelity text-to-image generation with strong prompt adherence
- **GPT-Image 1.5 Edit (fal)**: High-fidelity image editing with strong prompt adherence (supports up to 16 batched images and optional mask)
- **GPT-Image 2.0 (fal)** and **GPT-Image 2 Edit (fal)**: OpenAI GPT-Image 2 generation and editing — *fixed in 2.0: these nodes previously failed to load*
- **DY Wan Fun 2.2 (fal)**: Generate images using DY Wan Fun 2.2 model
- **DY Wan Upscaler (fal)**: Upscale images using DY Wan Upscaler

### Video Generation

- **Infinity Star Text-to-Video (fal)**: Generate videos using Infinity Star and text prompts
- **Kling Video Generation (fal)**: Generate videos using the Kling model
- **Kling Pro v1.0 Video Generation (fal)**: Original version of Kling Pro for video generation
- **Kling Pro v1.6 Video Generation (fal)**: Latest version of Kling Pro with improved quality
- **Kling Master v2.0 Video Generation (fal)**: Advanced video generation with Kling Master
- **Kling Pro 2.1 Video Generation (fal)**: Video Generation with Kling Pro with First Frame Last Frame support
- **Kling v2.5 Turbo Pro Image-to-Video (fal)**: Video Generation with Kling Turbo with First Frame Last Frame support
- **Kling Omni Image-to-Video (fal)**: Kling Omni image-to-video generation with start/end image support
- **Kling Omni Reference-to-Video (fal)**: Generate videos with reference images and elements
- **Kling Omni Video-to-Video Edit (fal)**: Edit videos with prompts and reference images
- **Kling Omni Video-to-Video Reference (fal)**: Video-to-video with reference image control
- **Kling v2.6 Pro Video Generation (fal)**: Unified T2V/I2V with native audio generation
- **Kling V3 Standard Video Generation (fal)**: Kling 3.0 Standard unified T2V/I2V with 1080p, 3-15s duration, native audio, and end frame control
- **Kling V3 Pro Video Generation (fal)**: Kling 3.0 Pro unified T2V/I2V with higher quality cinematic output (3-15s)
- **Kling V3 Standard Motion Control (fal)**: Character animation via motion transfer from reference video
- **Kling V3 Pro Motion Control (fal)**: Pro-quality character animation via motion transfer from reference video
- **Kling O3 Standard Video Generation (fal)**: Kling O3 Standard unified T2V/I2V with 3-15s duration and native audio
- **Kling O3 Pro Video Generation (fal)**: Kling O3 Pro unified T2V/I2V with highest quality output (3-15s)
- **Krea Wan 14b Video-to-Video (fal)**: Video-to-Video generation using Krea Wan 14b model
- **Runway Gen3 Image-to-Video (fal)**: Convert images to videos using Runway Gen3
- **Luma Dream Machine (fal)**: Create videos with Luma Dream Machine
- **MiniMax Video Generation (fal)**: Generate videos using MiniMax model
- **MiniMax Text-to-Video (fal)**: Create videos from text prompts using MiniMax
- **MiniMax Subject Reference (fal)**: Generate videos with subject reference using MiniMax
- **Pixverse Swap (fal)**: Swap a person, object, or background in a video using Pixverse Swap
- **Google Veo2 Image-to-Video (fal)**: Convert images to videos using Google's Veo2 model
- **Veo3 Video Generation (fal)**: Text-to-video generation with Google Veo3 model
- **Veo 3.1 First-Last-Frame-to-Video (fal)**: First Frame - Last Frame (Optional) video generation using the full VEO 3.1 model
- **Veo 3.1 Fast First-Last-Frame-to-Video (fal)**: First Frame - Last Frame (Optional) video generation using the fast VEO 3.1 model
- **Wan Pro Image-to-Video (fal)**: High-quality video generation with Wan Pro model
- **Wan 2.5 Preview Image-to-Video (fal)**: Image-to-video generation with the latest Wan 2.5 preview model
- **Wan VACE Video Edit (fal)**: Video + Reference Images to video generation with Wan VACE
- **Wan 2.2 VACE Fun 14b (fal)**: Video editing with Wan 2.2 VACE Fun 14b model for pose and depth control
- **Wan 2.2 14b Animate: Replace Character (fal)**: Animate video content by replacing the foreground character with a new or augmented character using Wan 2.2 14b
- **Wan 2.2 14b Animate: Move Character (fal)**: Animate video content by moving the foreground character within the scene using Wan 2.2 14b
- **Wan 2.6 Video Generation (fal)**: Unified T2V/I2V node - generates video from text, or from image if provided
- **Wan 2.6 Reference-to-Video (fal)**: Generate videos with subject consistency using up to 3 reference videos (@Video1, @Video2, @Video3 in prompt)
- **Seedance Image-to-Video (fal)**: Convert images to videos using Seedance Lite model
- **Seedance Text-to-Video (fal)**: Generate videos from text prompts using Seedance Lite model
- **Seedance Pro Image-to-Video (fal)**: Convert images to videos using Seedance Pro model with advanced controls
- **Sora 2 Pro Image-to-Video (fal)**: Generate Videos from an image input using OpenAI Sora 2 Pro
- **Video Upscaler (fal)**: Upscale video quality using AI
- **Seedvr Upscale Video (fal)**: Upscale video quality using Seedvr
- **Bria Video Increase Resolution (fal)**: Increase video resolution using Bria
- **Topaz Upscale Video (fal)**: Upscale video quality using Topaz
- **Combined Video Generation (fal)**: Generate videos using multiple services simultaneously
  - Supports Kling Pro v1.6, Kling Master v2.0, MiniMax, Luma, Veo2, and Wan Pro
  - Each service can be individually enabled/disabled
  - Wan Pro runs with safety checker enabled and automatic seed selection
- **Load Video from URL**: Load and process videos from a given URL
- **Video from URL → VIDEO (fal)**: *New in 2.0* — turn any video URL (e.g. from a `result_json` output) into ComfyUI's native `VIDEO` type

### Language Models (LLMs)

- **LLM (fal)**: Large Language Model for text generation and processing VIA openrouter endpoint
  - Available models:
    - google/gemini-2.5-flash
    - anthropic/claude-sonnet-4.5
    - openai/gpt-4.1
    - openai/gpt-oss-120b
    - meta-llama/llama-4-maverick
    - custom (Get model name from openrouter)'

### Vision Language Models (VLMs)

- **VLM (fal)**: Vision Language Model for image understanding and text generation VIA openrouter endpoint
  - Available models:
    - google/gemini-2.5-flash
    - anthropic/claude-sonnet-4.5
    - openai/gpt-4o
    - qwen/qwen3-vl-235b-a22b-instruct
    - x-ai/grok-4-fast
    - custom (Get model name from openrouter)
  - Supports various tasks such as image captioning, visual question answering, and more

## Auto-Generated Nodes

New in 2.0: at startup, this package reads the committed `data/fal_registry.json` and generates a ComfyUI node for every live public model on fal — currently ~1,391 models. No code is written to disk; the node classes are built in memory each time ComfyUI starts, so updating the registry file is all it takes to get new models.

**Where to find them:** the node browser under `FAL/Models/<category>` (e.g. `FAL/Models/image-to-video`, `FAL/Models/text-to-image`), or just search for the model name. Internally the node keys are `FalAPI_<endpoint-id>` (slashes replaced with dashes), so workflows stay stable across registry refreshes.

**What each generated node gives you:**

- **Native inputs** — `IMAGE`, `VIDEO`, and `AUDIO` sockets for media parameters; connected media is uploaded to fal automatically. Every other parameter (prompts, enums, sliders) is a regular widget.
- **Native outputs** — video models return ComfyUI's native `VIDEO` type, audio models return `AUDIO`, image models return `IMAGE` batches. JSON-ish models return the raw result as a string.
- **Tooltips from the API schema** — hover any input to see fal's own parameter documentation.
- **Pricing in the help panel** — each node's description includes the model's current pricing so you know what a queue run costs before you press it.
- **Seed semantics** — `seed = -1` means "random / omit seed"; any other value is sent to the API for reproducibility.
- **force_rerun** — a toggle that bypasses ComfyUI's result caching so you can re-roll the same inputs without changing anything.

### Fal Any Endpoint (fal)

The escape hatch: a single generic node (under `FAL/Models`) that calls **any** fal endpoint by id. Type the endpoint id (e.g. `fal-ai/flux/dev`), provide arguments as a JSON object, and optionally connect image/video/audio inputs — they are uploaded and override the matching keys (`image_url`, `image_urls`, `video_url`, `audio_url`, `seed`) in your JSON. Outputs are extracted opportunistically as `IMAGE`/`VIDEO`/`AUDIO`, and the raw API result is always available as a `result_json` string.

### Disabling or filtering

If ~893 extra nodes is more than you want, use the `[dynamic_nodes]` config section (see [Configuration](#configuration)) to disable them entirely (`enabled = false`) or load only certain categories (`categories = text-to-image,image-to-video`). The curated nodes always load regardless.

## Generated Model List

The full list of auto-generated model nodes, grouped by category (largest first). Click a category to expand it. This section is regenerated with `python scripts/build_readme.py`.

<!-- BEGIN GENERATED MODEL LIST -->

1391 models · newest model 2026-07-01 · refresh with `scripts/build_registry.py`

<details>
<summary><strong>image-to-image</strong> — 376 models</summary>

| Model | Endpoint | Lab | Output |
| --- | --- | --- | --- |
| Age Modify | [`fal-ai/image-apps-v2/age-modify`](https://fal.ai/models/fal-ai/image-apps-v2/age-modify) | — | images |
| AuraSR | [`fal-ai/aura-sr`](https://fal.ai/models/fal-ai/aura-sr) | — | image |
| Bagel | [`fal-ai/bagel/edit`](https://fal.ai/models/fal-ai/bagel/edit) | — | images |
| ben-v2-image | [`fal-ai/ben/v2/image`](https://fal.ai/models/fal-ai/ben/v2/image) | — | image |
| Bernini-R Edit Image | [`fal-ai/bernini-r/edit-image`](https://fal.ai/models/fal-ai/bernini-r/edit-image) | Bytedance | image |
| Birefnet Background Removal | [`fal-ai/birefnet`](https://fal.ai/models/fal-ai/birefnet) | — | image |
| Birefnet Background Removal V2 | [`fal-ai/birefnet/v2`](https://fal.ai/models/fal-ai/birefnet/v2) | — | image |
| Boogu Image | [`fal-ai/boogu-image/edit`](https://fal.ai/models/fal-ai/boogu-image/edit) | — | images |
| Bria | [`fal-ai/bria/reimagine`](https://fal.ai/models/fal-ai/bria/reimagine) | Bria AI | images |
| Bria Background Replace | [`fal-ai/bria/background/replace`](https://fal.ai/models/fal-ai/bria/background/replace) | Bria AI | images |
| Bria Eraser | [`fal-ai/bria/eraser`](https://fal.ai/models/fal-ai/bria/eraser) | Bria AI | image |
| Bria Expand Image | [`fal-ai/bria/expand`](https://fal.ai/models/fal-ai/bria/expand) | Bria AI | image |
| Bria GenFill | [`fal-ai/bria/genfill`](https://fal.ai/models/fal-ai/bria/genfill) | Bria AI | images |
| Bria Product Shot | [`fal-ai/bria/product-shot`](https://fal.ai/models/fal-ai/bria/product-shot) | Bria AI | images |
| Bria RMBG 2.0 | [`fal-ai/bria/background/remove`](https://fal.ai/models/fal-ai/bria/background/remove) | Bria AI | image |
| Bytedance Seedream V4 Edit | [`fal-ai/bytedance/seedream/v4/edit`](https://fal.ai/models/fal-ai/bytedance/seedream/v4/edit) | Bytedance | images |
| Bytedance Seedream V4.5 Edit | [`fal-ai/bytedance/seedream/v4.5/edit`](https://fal.ai/models/fal-ai/bytedance/seedream/v4.5/edit) | Bytedance | images |
| Bytedance Seedream V5 Lite Edit | [`fal-ai/bytedance/seedream/v5/lite/edit`](https://fal.ai/models/fal-ai/bytedance/seedream/v5/lite/edit) | Bytedance | images |
| Cartoonify | [`fal-ai/cartoonify`](https://fal.ai/models/fal-ai/cartoonify) | — | images |
| CCSR Upscaler | [`fal-ai/ccsr`](https://fal.ai/models/fal-ai/ccsr) | — | image |
| Chrono Edit | [`fal-ai/chrono-edit`](https://fal.ai/models/fal-ai/chrono-edit) | — | images |
| Chrono Edit Lora | [`fal-ai/chrono-edit-lora`](https://fal.ai/models/fal-ai/chrono-edit-lora) | — | images |
| Chrono Edit Lora Gallery | [`fal-ai/chrono-edit-lora-gallery/paintbrush`](https://fal.ai/models/fal-ai/chrono-edit-lora-gallery/paintbrush) | — | images |
| Chrono Edit Lora Gallery | [`fal-ai/chrono-edit-lora-gallery/upscaler`](https://fal.ai/models/fal-ai/chrono-edit-lora-gallery/upscaler) | — | images |
| City Teleport | [`fal-ai/image-apps-v2/city-teleport`](https://fal.ai/models/fal-ai/image-apps-v2/city-teleport) | — | images |
| Clarity Upscaler | [`fal-ai/clarity-upscaler`](https://fal.ai/models/fal-ai/clarity-upscaler) | Clarity AI | image |
| CodeFormer | [`fal-ai/codeformer`](https://fal.ai/models/fal-ai/codeformer) | — | image |
| ControlLight | [`fal-ai/control-light`](https://fal.ai/models/fal-ai/control-light) | — | images |
| ControlNet SDXL | [`fal-ai/fast-sdxl-controlnet-canny/image-to-image`](https://fal.ai/models/fal-ai/fast-sdxl-controlnet-canny/image-to-image) | — | images |
| ControlNet SDXL | [`fal-ai/fast-sdxl-controlnet-canny/inpainting`](https://fal.ai/models/fal-ai/fast-sdxl-controlnet-canny/inpainting) | — | images |
| Creative Upscaler | [`fal-ai/creative-upscaler`](https://fal.ai/models/fal-ai/creative-upscaler) | — | image |
| Crystal Upscaler | [`clarityai/crystal-upscaler`](https://fal.ai/models/clarityai/crystal-upscaler) | Clarity AI | images |
| DDColor | [`fal-ai/ddcolor`](https://fal.ai/models/fal-ai/ddcolor) | — | image |
| DocRes | [`fal-ai/docres`](https://fal.ai/models/fal-ai/docres) | — | image |
| DocRes-dewarp | [`fal-ai/docres/dewarp`](https://fal.ai/models/fal-ai/docres/dewarp) | — | image |
| DRCT-Super-Resolution | [`fal-ai/drct-super-resolution`](https://fal.ai/models/fal-ai/drct-super-resolution) | — | image |
| DreamOmni2 | [`fal-ai/dreamomni2/edit`](https://fal.ai/models/fal-ai/dreamomni2/edit) | — | image |
| DWPose Pose Prediction | [`fal-ai/dwpose`](https://fal.ai/models/fal-ai/dwpose) | — | image |
| Embed Product | [`bria/embed-product`](https://fal.ai/models/bria/embed-product) | Bria AI | json |
| Emu 3.5 Image | [`fal-ai/emu-3.5-image/edit-image`](https://fal.ai/models/fal-ai/emu-3.5-image/edit-image) | — | images |
| EVF-SAM2 Segmentation | [`fal-ai/evf-sam`](https://fal.ai/models/fal-ai/evf-sam) | — | image |
| Expression Change | [`fal-ai/image-apps-v2/expression-change`](https://fal.ai/models/fal-ai/image-apps-v2/expression-change) | — | images |
| Extract Object | [`bria/extract-object`](https://fal.ai/models/bria/extract-object) | Bria AI | image |
| Face Retoucher | [`fal-ai/retoucher`](https://fal.ai/models/fal-ai/retoucher) | — | image |
| FASHN Virtual Try-On V1.5 | [`fal-ai/fashn/tryon/v1.5`](https://fal.ai/models/fal-ai/fashn/tryon/v1.5) | Fashn | images |
| FASHN Virtual Try-On V1.6 | [`fal-ai/fashn/tryon/v1.6`](https://fal.ai/models/fal-ai/fashn/tryon/v1.6) | Fashn | images |
| Ffmpeg Api | [`fal-ai/ffmpeg-api/extract-frame`](https://fal.ai/models/fal-ai/ffmpeg-api/extract-frame) | — | images |
| Fibo Edit | [`bria/fibo-edit/edit`](https://fal.ai/models/bria/fibo-edit/edit) | Bria AI | images |
| Fibo Edit [Add Object by Text] | [`bria/fibo-edit/add_object_by_text`](https://fal.ai/models/bria/fibo-edit/add_object_by_text) | Bria AI | images |
| Fibo Edit [Blend] | [`bria/fibo-edit/blend`](https://fal.ai/models/bria/fibo-edit/blend) | Bria AI | images |
| Fibo Edit [Colorize] | [`bria/fibo-edit/colorize`](https://fal.ai/models/bria/fibo-edit/colorize) | Bria AI | images |
| Fibo Edit [Erase by Text] | [`bria/fibo-edit/erase_by_text`](https://fal.ai/models/bria/fibo-edit/erase_by_text) | Bria AI | images |
| Fibo Edit [Relight] | [`bria/fibo-edit/relight`](https://fal.ai/models/bria/fibo-edit/relight) | Bria AI | images |
| Fibo Edit [Replace Object by Text] | [`bria/fibo-edit/replace_object_by_text`](https://fal.ai/models/bria/fibo-edit/replace_object_by_text) | Bria AI | images |
| Fibo Edit [Reseason] | [`bria/fibo-edit/reseason`](https://fal.ai/models/bria/fibo-edit/reseason) | Bria AI | images |
| Fibo Edit [Restore] | [`bria/fibo-edit/restore`](https://fal.ai/models/bria/fibo-edit/restore) | Bria AI | images |
| Fibo Edit [Restyle] | [`bria/fibo-edit/restyle`](https://fal.ai/models/bria/fibo-edit/restyle) | Bria AI | images |
| Fibo Edit [Rewrite Text] | [`bria/fibo-edit/rewrite_text`](https://fal.ai/models/bria/fibo-edit/rewrite_text) | Bria AI | images |
| Fibo Edit [Sketch to Image] | [`bria/fibo-edit/sketch_to_colored_image`](https://fal.ai/models/bria/fibo-edit/sketch_to_colored_image) | Bria AI | images |
| FILM | [`fal-ai/film`](https://fal.ai/models/fal-ai/film) | — | images |
| Finegrain Eraser | [`fal-ai/finegrain-eraser`](https://fal.ai/models/fal-ai/finegrain-eraser) | Finegrain | image |
| Finegrain Eraser Bbox | [`fal-ai/finegrain-eraser/bbox`](https://fal.ai/models/fal-ai/finegrain-eraser/bbox) | Finegrain | image |
| Finegrain Eraser Mask | [`fal-ai/finegrain-eraser/mask`](https://fal.ai/models/fal-ai/finegrain-eraser/mask) | Finegrain | image |
| Firered Image Edit | [`fal-ai/firered-image-edit`](https://fal.ai/models/fal-ai/firered-image-edit) | — | images |
| Firered Image Edit V1.1 | [`fal-ai/firered-image-edit-v1.1`](https://fal.ai/models/fal-ai/firered-image-edit-v1.1) | FireRed | images |
| Florence-2 Large | [`fal-ai/florence-2-large/caption-to-phrase-grounding`](https://fal.ai/models/fal-ai/florence-2-large/caption-to-phrase-grounding) | — | image |
| Florence-2 Large | [`fal-ai/florence-2-large/dense-region-caption`](https://fal.ai/models/fal-ai/florence-2-large/dense-region-caption) | — | image |
| Florence-2 Large | [`fal-ai/florence-2-large/object-detection`](https://fal.ai/models/fal-ai/florence-2-large/object-detection) | — | image |
| Florence-2 Large | [`fal-ai/florence-2-large/ocr-with-region`](https://fal.ai/models/fal-ai/florence-2-large/ocr-with-region) | — | image |
| Florence-2 Large | [`fal-ai/florence-2-large/open-vocabulary-detection`](https://fal.ai/models/fal-ai/florence-2-large/open-vocabulary-detection) | — | image |
| Florence-2 Large | [`fal-ai/florence-2-large/referring-expression-segmentation`](https://fal.ai/models/fal-ai/florence-2-large/referring-expression-segmentation) | — | image |
| Florence-2 Large | [`fal-ai/florence-2-large/region-proposal`](https://fal.ai/models/fal-ai/florence-2-large/region-proposal) | — | image |
| Florence-2 Large | [`fal-ai/florence-2-large/region-to-segmentation`](https://fal.ai/models/fal-ai/florence-2-large/region-to-segmentation) | — | image |
| Flow-Edit | [`fal-ai/flowedit`](https://fal.ai/models/fal-ai/flowedit) | — | image |
| Flux 2 [klein] Realtime | [`fal-ai/flux-2/klein/realtime`](https://fal.ai/models/fal-ai/flux-2/klein/realtime) | Black Forest Labs | images |
| FLUX 2 Edit | [`fal-ai/flux-2/edit`](https://fal.ai/models/fal-ai/flux-2/edit) | Black Forest Labs | images |
| FLUX 2 Flash Edit | [`fal-ai/flux-2/flash/edit`](https://fal.ai/models/fal-ai/flux-2/flash/edit) | Black Forest Labs | images |
| Flux 2 Flex | [`fal-ai/flux-2-flex/edit`](https://fal.ai/models/fal-ai/flux-2-flex/edit) | Black Forest Labs | images |
| FLUX 2 Lora Edit | [`fal-ai/flux-2/lora/edit`](https://fal.ai/models/fal-ai/flux-2/lora/edit) | Black Forest Labs | images |
| Flux 2 Lora Gallery | [`fal-ai/flux-2-lora-gallery/add-background`](https://fal.ai/models/fal-ai/flux-2-lora-gallery/add-background) | Black Forest Labs | images |
| Flux 2 Lora Gallery | [`fal-ai/flux-2-lora-gallery/apartment-staging`](https://fal.ai/models/fal-ai/flux-2-lora-gallery/apartment-staging) | Black Forest Labs | images |
| Flux 2 Lora Gallery | [`fal-ai/flux-2-lora-gallery/face-to-full-portrait`](https://fal.ai/models/fal-ai/flux-2-lora-gallery/face-to-full-portrait) | Black Forest Labs | images |
| Flux 2 Lora Gallery | [`fal-ai/flux-2-lora-gallery/multiple-angles`](https://fal.ai/models/fal-ai/flux-2-lora-gallery/multiple-angles) | Black Forest Labs | images |
| Flux 2 Lora Gallery | [`fal-ai/flux-2-lora-gallery/virtual-tryon`](https://fal.ai/models/fal-ai/flux-2-lora-gallery/virtual-tryon) | Black Forest Labs | images |
| Flux 2 Max | [`fal-ai/flux-2-max/edit`](https://fal.ai/models/fal-ai/flux-2-max/edit) | Black Forest Labs | images |
| FLUX 2 Pro Edit | [`fal-ai/flux-2-pro/edit`](https://fal.ai/models/fal-ai/flux-2-pro/edit) | Black Forest Labs | images |
| FLUX 2 Pro Outpaint | [`fal-ai/flux-2-pro/outpaint`](https://fal.ai/models/fal-ai/flux-2-pro/outpaint) | Black Forest Labs | images |
| FLUX 2 Turbo Edit | [`fal-ai/flux-2/turbo/edit`](https://fal.ai/models/fal-ai/flux-2/turbo/edit) | Black Forest Labs | images |
| Flux Kontext Lora | [`fal-ai/flux-kontext-lora`](https://fal.ai/models/fal-ai/flux-kontext-lora) | Black Forest Labs | images |
| Flux Kontext Lora | [`fal-ai/flux-kontext-lora/inpaint`](https://fal.ai/models/fal-ai/flux-kontext-lora/inpaint) | Black Forest Labs | images |
| Flux Pro Erase | [`fal-ai/flux-pro/v1/erase`](https://fal.ai/models/fal-ai/flux-pro/v1/erase) | Black Forest Labs | images |
| FLUX Virtual Try-On | [`fal-ai/flux-pro/v1/vto`](https://fal.ai/models/fal-ai/flux-pro/v1/vto) | Black Forest Labs | images |
| Flux Vision Upscaler | [`fal-ai/flux-vision-upscaler`](https://fal.ai/models/fal-ai/flux-vision-upscaler) | Black Forest Labs | image |
| FLUX.1 [dev] | [`fal-ai/flux-1/dev/image-to-image`](https://fal.ai/models/fal-ai/flux-1/dev/image-to-image) | Black Forest Labs | images |
| FLUX.1 [dev] | [`fal-ai/flux/dev/image-to-image`](https://fal.ai/models/fal-ai/flux/dev/image-to-image) | Black Forest Labs | images |
| FLUX.1 [dev] Canny with LoRAs | [`fal-ai/flux-lora-canny`](https://fal.ai/models/fal-ai/flux-lora-canny) | Black Forest Labs | images |
| FLUX.1 [dev] Control LoRA Canny | [`fal-ai/flux-control-lora-canny/image-to-image`](https://fal.ai/models/fal-ai/flux-control-lora-canny/image-to-image) | Black Forest Labs | images |
| FLUX.1 [dev] Control LoRA Depth | [`fal-ai/flux-control-lora-depth/image-to-image`](https://fal.ai/models/fal-ai/flux-control-lora-depth/image-to-image) | Black Forest Labs | images |
| FLUX.1 [dev] Depth with LoRAs | [`fal-ai/flux-lora-depth`](https://fal.ai/models/fal-ai/flux-lora-depth) | Black Forest Labs | images |
| FLUX.1 [dev] Fill with LoRAs | [`fal-ai/flux-lora-fill`](https://fal.ai/models/fal-ai/flux-lora-fill) | Black Forest Labs | images |
| FLUX.1 [dev] Redux | [`fal-ai/flux-1/dev/redux`](https://fal.ai/models/fal-ai/flux-1/dev/redux) | Black Forest Labs | images |
| FLUX.1 [dev] Redux | [`fal-ai/flux/dev/redux`](https://fal.ai/models/fal-ai/flux/dev/redux) | Black Forest Labs | images |
| FLUX.1 [dev] with Controlnets and Loras | [`fal-ai/flux-general/differential-diffusion`](https://fal.ai/models/fal-ai/flux-general/differential-diffusion) | Black Forest Labs | images |
| FLUX.1 [dev] with Controlnets and Loras | [`fal-ai/flux-general/image-to-image`](https://fal.ai/models/fal-ai/flux-general/image-to-image) | Black Forest Labs | images |
| FLUX.1 [dev] with Controlnets and Loras | [`fal-ai/flux-general/inpainting`](https://fal.ai/models/fal-ai/flux-general/inpainting) | Black Forest Labs | images |
| FLUX.1 [dev] with Controlnets and Loras | [`fal-ai/flux-general/rf-inversion`](https://fal.ai/models/fal-ai/flux-general/rf-inversion) | Black Forest Labs | images |
| FLUX.1 [dev] with LoRAs | [`fal-ai/flux-lora/image-to-image`](https://fal.ai/models/fal-ai/flux-lora/image-to-image) | Black Forest Labs | images |
| FLUX.1 [pro] Fill | [`fal-ai/flux-pro/v1/fill`](https://fal.ai/models/fal-ai/flux-pro/v1/fill) | Black Forest Labs | images |
| FLUX.1 [pro] Fill Fine-tuned | [`fal-ai/flux-pro/v1/fill-finetuned`](https://fal.ai/models/fal-ai/flux-pro/v1/fill-finetuned) | Black Forest Labs | images |
| FLUX.1 [schnell] Redux | [`fal-ai/flux-1/schnell/redux`](https://fal.ai/models/fal-ai/flux-1/schnell/redux) | Black Forest Labs | images |
| FLUX.1 [schnell] Redux | [`fal-ai/flux/schnell/redux`](https://fal.ai/models/fal-ai/flux/schnell/redux) | Black Forest Labs | images |
| FLUX.1 Kontext [dev] | [`fal-ai/flux-kontext/dev`](https://fal.ai/models/fal-ai/flux-kontext/dev) | Black Forest Labs | images |
| FLUX.1 Kontext [max] | [`fal-ai/flux-pro/kontext/max`](https://fal.ai/models/fal-ai/flux-pro/kontext/max) | Black Forest Labs | images |
| FLUX.1 Kontext [max] | [`fal-ai/flux-pro/kontext/max/multi`](https://fal.ai/models/fal-ai/flux-pro/kontext/max/multi) | Black Forest Labs | images |
| FLUX.1 Kontext [pro] | [`fal-ai/flux-pro/kontext`](https://fal.ai/models/fal-ai/flux-pro/kontext) | Black Forest Labs | images |
| FLUX.1 Kontext [pro] | [`fal-ai/flux-pro/kontext/multi`](https://fal.ai/models/fal-ai/flux-pro/kontext/multi) | Black Forest Labs | images |
| FLUX.1 Krea [dev] | [`fal-ai/flux-1/krea/image-to-image`](https://fal.ai/models/fal-ai/flux-1/krea/image-to-image) | Black Forest Labs | images |
| FLUX.1 Krea [dev] | [`fal-ai/flux/krea/image-to-image`](https://fal.ai/models/fal-ai/flux/krea/image-to-image) | Black Forest Labs | images |
| FLUX.1 Krea [dev] Inpainting with LoRAs | [`fal-ai/flux-krea-lora/inpainting`](https://fal.ai/models/fal-ai/flux-krea-lora/inpainting) | Black Forest Labs | images |
| FLUX.1 Krea [dev] Redux | [`fal-ai/flux-1/krea/redux`](https://fal.ai/models/fal-ai/flux-1/krea/redux) | Black Forest Labs | images |
| FLUX.1 Krea [dev] Redux | [`fal-ai/flux/krea/redux`](https://fal.ai/models/fal-ai/flux/krea/redux) | Black Forest Labs | images |
| FLUX.1 Krea [dev] with LoRAs | [`fal-ai/flux-krea-lora/image-to-image`](https://fal.ai/models/fal-ai/flux-krea-lora/image-to-image) | Black Forest Labs | images |
| FLUX.1 SRPO [dev] | [`fal-ai/flux-1/srpo/image-to-image`](https://fal.ai/models/fal-ai/flux-1/srpo/image-to-image) | Black Forest Labs | images |
| FLUX.1 SRPO [dev] | [`fal-ai/flux/srpo/image-to-image`](https://fal.ai/models/fal-ai/flux/srpo/image-to-image) | Black Forest Labs | images |
| FLUX.2 [klein] 4B | [`fal-ai/flux-2/klein/4b/edit`](https://fal.ai/models/fal-ai/flux-2/klein/4b/edit) | Black Forest Labs | images |
| FLUX.2 [klein] 4B Base | [`fal-ai/flux-2/klein/4b/base/edit`](https://fal.ai/models/fal-ai/flux-2/klein/4b/base/edit) | Black Forest Labs | images |
| FLUX.2 [klein] 4B Base LoRA | [`fal-ai/flux-2/klein/4b/base/edit/lora`](https://fal.ai/models/fal-ai/flux-2/klein/4b/base/edit/lora) | Black Forest Labs | images |
| FLUX.2 [klein] 4B LoRA | [`fal-ai/flux-2/klein/4b/edit/lora`](https://fal.ai/models/fal-ai/flux-2/klein/4b/edit/lora) | Black Forest Labs | images |
| FLUX.2 [klein] 9B | [`fal-ai/flux-2/klein/9b/edit`](https://fal.ai/models/fal-ai/flux-2/klein/9b/edit) | Black Forest Labs | images |
| FLUX.2 [klein] 9B Base | [`fal-ai/flux-2/klein/9b/base/edit`](https://fal.ai/models/fal-ai/flux-2/klein/9b/base/edit) | Black Forest Labs | images |
| FLUX.2 [klein] 9B Base LoRA | [`fal-ai/flux-2/klein/9b/base/edit/lora`](https://fal.ai/models/fal-ai/flux-2/klein/9b/base/edit/lora) | Black Forest Labs | images |
| FLUX.2 [klein] 9B LoRA | [`fal-ai/flux-2/klein/9b/edit/lora`](https://fal.ai/models/fal-ai/flux-2/klein/9b/edit/lora) | Black Forest Labs | images |
| FLUX1.1 [pro] Redux | [`fal-ai/flux-pro/v1.1/redux`](https://fal.ai/models/fal-ai/flux-pro/v1.1/redux) | Black Forest Labs | images |
| FLUX1.1 [pro] ultra Redux | [`fal-ai/flux-pro/v1.1-ultra/redux`](https://fal.ai/models/fal-ai/flux-pro/v1.1-ultra/redux) | Black Forest Labs | images |
| Gemini 2.5 Flash Image | [`fal-ai/gemini-25-flash-image/edit`](https://fal.ai/models/fal-ai/gemini-25-flash-image/edit) | Google | images |
| Gemini 3 Pro Image Preview | [`fal-ai/gemini-3-pro-image-preview/edit`](https://fal.ai/models/fal-ai/gemini-3-pro-image-preview/edit) | Google | images |
| Gemini 3.1 Flash Image Preview | [`fal-ai/gemini-3.1-flash-image-preview/edit`](https://fal.ai/models/fal-ai/gemini-3.1-flash-image-preview/edit) | Google | images |
| Genfill | [`bria/genfill/v2`](https://fal.ai/models/bria/genfill/v2) | — | images |
| Ghiblify Images | [`fal-ai/ghiblify`](https://fal.ai/models/fal-ai/ghiblify) | — | image |
| Glm Image | [`fal-ai/glm-image/image-to-image`](https://fal.ai/models/fal-ai/glm-image/image-to-image) | — | images |
| GPT Image 1 Mini | [`fal-ai/gpt-image-1-mini/edit`](https://fal.ai/models/fal-ai/gpt-image-1-mini/edit) | OpenAI | images |
| GPT Image 2 API | [`openai/gpt-image-2/edit`](https://fal.ai/models/openai/gpt-image-2/edit) | OpenAI | images |
| GPT-Image 1.5 | [`fal-ai/gpt-image-1.5/edit`](https://fal.ai/models/fal-ai/gpt-image-1.5/edit) | OpenAI | images |
| gpt-image-1 | [`fal-ai/gpt-image-1/edit-image`](https://fal.ai/models/fal-ai/gpt-image-1/edit-image) | OpenAI | images |
| Grok Imagine Image | [`xai/grok-imagine-image/edit`](https://fal.ai/models/xai/grok-imagine-image/edit) | xAI | images |
| Grok Imagine Image Editing Quality | [`xai/grok-imagine-image/quality/edit`](https://fal.ai/models/xai/grok-imagine-image/quality/edit) | xAI | images |
| Hair Change | [`fal-ai/image-apps-v2/hair-change`](https://fal.ai/models/fal-ai/image-apps-v2/hair-change) | — | images |
| Headshot Generator | [`fal-ai/image-apps-v2/headshot-photo`](https://fal.ai/models/fal-ai/image-apps-v2/headshot-photo) | — | images |
| Hidream I1 Full | [`fal-ai/hidream-i1-full/image-to-image`](https://fal.ai/models/fal-ai/hidream-i1-full/image-to-image) | Hidream | images |
| Hidream O1 Image | [`fal-ai/hidream-o1-image/dev/edit`](https://fal.ai/models/fal-ai/hidream-o1-image/dev/edit) | Hidream | images |
| Hidream O1 Image | [`fal-ai/hidream-o1-image/edit`](https://fal.ai/models/fal-ai/hidream-o1-image/edit) | Hidream | images |
| Hunyuan Image | [`fal-ai/hunyuan-image/v3/instruct/edit`](https://fal.ai/models/fal-ai/hunyuan-image/v3/instruct/edit) | Tencent | images |
| Hunyuan World | [`fal-ai/hunyuan_world`](https://fal.ai/models/fal-ai/hunyuan_world) | Tencent | image |
| Hy Wu Edit | [`fal-ai/hy-wu-edit`](https://fal.ai/models/fal-ai/hy-wu-edit) | — | images |
| IC-Light-v2 for Image Relighting | [`fal-ai/iclight-v2`](https://fal.ai/models/fal-ai/iclight-v2) | — | images |
| Ideogram | [`fal-ai/ideogram/v3/layerize-text`](https://fal.ai/models/fal-ai/ideogram/v3/layerize-text) | Ideogram | image |
| Ideogram | [`fal-ai/ideogram/v3/reframe`](https://fal.ai/models/fal-ai/ideogram/v3/reframe) | Ideogram | images |
| Ideogram | [`fal-ai/ideogram/v3/remix`](https://fal.ai/models/fal-ai/ideogram/v3/remix) | Ideogram | images |
| Ideogram Remove Background | [`fal-ai/ideogram/remove-background`](https://fal.ai/models/fal-ai/ideogram/remove-background) | Ideogram | image |
| Ideogram Replace Background | [`fal-ai/ideogram/v3/replace-background`](https://fal.ai/models/fal-ai/ideogram/v3/replace-background) | Ideogram | images |
| Ideogram Upscale | [`fal-ai/ideogram/upscale`](https://fal.ai/models/fal-ai/ideogram/upscale) | Ideogram | images |
| Ideogram V2 Edit | [`fal-ai/ideogram/v2/edit`](https://fal.ai/models/fal-ai/ideogram/v2/edit) | Ideogram | images |
| Ideogram V2 Remix | [`fal-ai/ideogram/v2/remix`](https://fal.ai/models/fal-ai/ideogram/v2/remix) | Ideogram | images |
| Ideogram V2 Turbo Edit | [`fal-ai/ideogram/v2/turbo/edit`](https://fal.ai/models/fal-ai/ideogram/v2/turbo/edit) | Ideogram | images |
| Ideogram V2 Turbo Remix | [`fal-ai/ideogram/v2/turbo/remix`](https://fal.ai/models/fal-ai/ideogram/v2/turbo/remix) | Ideogram | images |
| Ideogram V2A Remix | [`fal-ai/ideogram/v2a/remix`](https://fal.ai/models/fal-ai/ideogram/v2a/remix) | Ideogram | images |
| Ideogram V2A Turbo Remix | [`fal-ai/ideogram/v2a/turbo/remix`](https://fal.ai/models/fal-ai/ideogram/v2a/turbo/remix) | Ideogram | images |
| Ideogram V3 Character | [`fal-ai/ideogram/character`](https://fal.ai/models/fal-ai/ideogram/character) | Ideogram | images |
| Ideogram V3 Character Edit | [`fal-ai/ideogram/character/edit`](https://fal.ai/models/fal-ai/ideogram/character/edit) | Ideogram | images |
| Ideogram V3 Character Remix | [`fal-ai/ideogram/character/remix`](https://fal.ai/models/fal-ai/ideogram/character/remix) | Ideogram | images |
| Ideogram V3 Edit | [`fal-ai/ideogram/v3/edit`](https://fal.ai/models/fal-ai/ideogram/v3/edit) | Ideogram | images |
| Ideogram V4.0q Image to Image | [`ideogram/v4/image-to-image`](https://fal.ai/models/ideogram/v4/image-to-image) | Ideogram | images |
| Ideogram V4.0q Image to Image LoRA | [`ideogram/v4/image-to-image/lora`](https://fal.ai/models/ideogram/v4/image-to-image/lora) | Ideogram | images |
| Ideogram V4.0q Tiling | [`ideogram/v4/tiling`](https://fal.ai/models/ideogram/v4/tiling) | Ideogram | images |
| Ideogram V4.0q Tiling LoRA | [`ideogram/v4/tiling/lora`](https://fal.ai/models/ideogram/v4/tiling/lora) | Ideogram | images |
| Image Editing | [`fal-ai/image-editing/baby-version`](https://fal.ai/models/fal-ai/image-editing/baby-version) | — | images |
| Image Editing Age Progression | [`fal-ai/image-editing/age-progression`](https://fal.ai/models/fal-ai/image-editing/age-progression) | — | images |
| Image Editing Background Change | [`fal-ai/image-editing/background-change`](https://fal.ai/models/fal-ai/image-editing/background-change) | — | images |
| Image Editing Broccoli Haircut | [`fal-ai/image-editing/broccoli-haircut`](https://fal.ai/models/fal-ai/image-editing/broccoli-haircut) | — | images |
| Image Editing Cartoonify | [`fal-ai/image-editing/cartoonify`](https://fal.ai/models/fal-ai/image-editing/cartoonify) | — | images |
| Image Editing Color Correction | [`fal-ai/image-editing/color-correction`](https://fal.ai/models/fal-ai/image-editing/color-correction) | — | images |
| Image Editing Expression Change | [`fal-ai/image-editing/expression-change`](https://fal.ai/models/fal-ai/image-editing/expression-change) | — | images |
| Image Editing Face Enhancement | [`fal-ai/image-editing/face-enhancement`](https://fal.ai/models/fal-ai/image-editing/face-enhancement) | — | images |
| Image Editing Hair Change | [`fal-ai/image-editing/hair-change`](https://fal.ai/models/fal-ai/image-editing/hair-change) | — | images |
| Image Editing Object Removal | [`fal-ai/image-editing/object-removal`](https://fal.ai/models/fal-ai/image-editing/object-removal) | — | images |
| Image Editing Photo Restoration | [`fal-ai/image-editing/photo-restoration`](https://fal.ai/models/fal-ai/image-editing/photo-restoration) | — | images |
| Image Editing Plushie Style | [`fal-ai/image-editing/plushie-style`](https://fal.ai/models/fal-ai/image-editing/plushie-style) | — | images |
| Image Editing Professional Photo | [`fal-ai/image-editing/professional-photo`](https://fal.ai/models/fal-ai/image-editing/professional-photo) | — | images |
| Image Editing Realism | [`fal-ai/image-editing/realism`](https://fal.ai/models/fal-ai/image-editing/realism) | — | images |
| Image Editing Reframe | [`fal-ai/image-editing/reframe`](https://fal.ai/models/fal-ai/image-editing/reframe) | — | images |
| Image Editing Retouch | [`fal-ai/image-editing/retouch`](https://fal.ai/models/fal-ai/image-editing/retouch) | — | images |
| Image Editing Scene Composition | [`fal-ai/image-editing/scene-composition`](https://fal.ai/models/fal-ai/image-editing/scene-composition) | — | images |
| Image Editing Style Transfer | [`fal-ai/image-editing/style-transfer`](https://fal.ai/models/fal-ai/image-editing/style-transfer) | — | images |
| Image Editing Text Removal | [`fal-ai/image-editing/text-removal`](https://fal.ai/models/fal-ai/image-editing/text-removal) | — | images |
| Image Editing Time Of Day | [`fal-ai/image-editing/time-of-day`](https://fal.ai/models/fal-ai/image-editing/time-of-day) | — | images |
| Image Editing Weather Effect | [`fal-ai/image-editing/weather-effect`](https://fal.ai/models/fal-ai/image-editing/weather-effect) | — | images |
| Image Editing Wojak Style | [`fal-ai/image-editing/wojak-style`](https://fal.ai/models/fal-ai/image-editing/wojak-style) | — | images |
| Image Editing Youtube Thumbnails | [`fal-ai/image-editing/youtube-thumbnails`](https://fal.ai/models/fal-ai/image-editing/youtube-thumbnails) | — | images |
| Image Outpaint | [`fal-ai/image-apps-v2/outpaint`](https://fal.ai/models/fal-ai/image-apps-v2/outpaint) | — | images |
| Image Preprocessors | [`fal-ai/image-preprocessors/depth-anything/v2`](https://fal.ai/models/fal-ai/image-preprocessors/depth-anything/v2) | — | image |
| Image Preprocessors | [`fal-ai/image-preprocessors/hed`](https://fal.ai/models/fal-ai/image-preprocessors/hed) | — | image |
| Image Preprocessors | [`fal-ai/image-preprocessors/lineart`](https://fal.ai/models/fal-ai/image-preprocessors/lineart) | — | image |
| Image Preprocessors | [`fal-ai/image-preprocessors/midas`](https://fal.ai/models/fal-ai/image-preprocessors/midas) | — | json |
| Image Preprocessors | [`fal-ai/image-preprocessors/mlsd`](https://fal.ai/models/fal-ai/image-preprocessors/mlsd) | — | image |
| Image Preprocessors | [`fal-ai/image-preprocessors/pidi`](https://fal.ai/models/fal-ai/image-preprocessors/pidi) | — | image |
| Image Preprocessors | [`fal-ai/image-preprocessors/sam`](https://fal.ai/models/fal-ai/image-preprocessors/sam) | — | image |
| Image Preprocessors | [`fal-ai/image-preprocessors/scribble`](https://fal.ai/models/fal-ai/image-preprocessors/scribble) | — | image |
| Image Preprocessors | [`fal-ai/image-preprocessors/teed`](https://fal.ai/models/fal-ai/image-preprocessors/teed) | — | image |
| Image Preprocessors | [`fal-ai/image-preprocessors/zoe`](https://fal.ai/models/fal-ai/image-preprocessors/zoe) | — | image |
| Image2Pixel | [`fal-ai/image2pixel`](https://fal.ai/models/fal-ai/image2pixel) | — | images |
| Image2svg | [`fal-ai/image2svg`](https://fal.ai/models/fal-ai/image2svg) | — | images |
| Imagineart 2.0 Edit Preview | [`imagineart/imagineart-2.0-edit-preview/image-to-image`](https://fal.ai/models/imagineart/imagineart-2.0-edit-preview/image-to-image) | — | images |
| Inpainting sdxl and sd | [`fal-ai/inpaint`](https://fal.ai/models/fal-ai/inpaint) | — | image |
| Instant Character | [`fal-ai/instant-character`](https://fal.ai/models/fal-ai/instant-character) | — | images |
| Invisible Watermark | [`fal-ai/invisible-watermark`](https://fal.ai/models/fal-ai/invisible-watermark) | — | image |
| IP Adapter Face ID | [`fal-ai/ip-adapter-face-id`](https://fal.ai/models/fal-ai/ip-adapter-face-id) | — | image |
| Joyai Image Edit | [`fal-ai/joyai-image-edit`](https://fal.ai/models/fal-ai/joyai-image-edit) | — | images |
| Juggernaut Flux Base | [`rundiffusion-fal/juggernaut-flux/base/image-to-image`](https://fal.ai/models/rundiffusion-fal/juggernaut-flux/base/image-to-image) | Rundiffusion | images |
| Juggernaut Flux Lora | [`rundiffusion-fal/juggernaut-flux-lora/inpainting`](https://fal.ai/models/rundiffusion-fal/juggernaut-flux-lora/inpainting) | Rundiffusion | images |
| Juggernaut Flux Pro | [`rundiffusion-fal/juggernaut-flux/pro/image-to-image`](https://fal.ai/models/rundiffusion-fal/juggernaut-flux/pro/image-to-image) | Rundiffusion | images |
| Kling Image | [`fal-ai/kling-image/o3/image-to-image`](https://fal.ai/models/fal-ai/kling-image/o3/image-to-image) | Kling | images |
| Kling Image | [`fal-ai/kling-image/v3/image-to-image`](https://fal.ai/models/fal-ai/kling-image/v3/image-to-image) | Kling | images |
| Kling Kolors Virtual TryOn v1.5 | [`fal-ai/kling/v1-5/kolors-virtual-try-on`](https://fal.ai/models/fal-ai/kling/v1-5/kolors-virtual-try-on) | Kling | image |
| Kling O1 Image | [`fal-ai/kling-image/o1`](https://fal.ai/models/fal-ai/kling-image/o1) | Kling | images |
| Kolors Image to Image | [`fal-ai/kolors/image-to-image`](https://fal.ai/models/fal-ai/kolors/image-to-image) | — | images |
| Latent Consistency Models (v1.5/XL) | [`fal-ai/fast-lcm-diffusion/image-to-image`](https://fal.ai/models/fal-ai/fast-lcm-diffusion/image-to-image) | — | images |
| Latent Consistency Models (v1.5/XL) | [`fal-ai/fast-lcm-diffusion/inpainting`](https://fal.ai/models/fal-ai/fast-lcm-diffusion/inpainting) | — | images |
| Leffa Pose Transfer | [`fal-ai/leffa/pose-transfer`](https://fal.ai/models/fal-ai/leffa/pose-transfer) | — | image |
| Leffa Virtual TryOn | [`fal-ai/leffa/virtual-tryon`](https://fal.ai/models/fal-ai/leffa/virtual-tryon) | — | image |
| Live Portrait | [`fal-ai/live-portrait/image`](https://fal.ai/models/fal-ai/live-portrait/image) | — | image |
| Longcat Image | [`fal-ai/longcat-image/edit`](https://fal.ai/models/fal-ai/longcat-image/edit) | — | images |
| Luma Photon | [`fal-ai/luma-photon/flash/modify`](https://fal.ai/models/fal-ai/luma-photon/flash/modify) | Luma AI | images |
| Luma Photon | [`fal-ai/luma-photon/modify`](https://fal.ai/models/fal-ai/luma-photon/modify) | Luma AI | images |
| Luma Photon Flash Reframe | [`fal-ai/luma-photon/flash/reframe`](https://fal.ai/models/fal-ai/luma-photon/flash/reframe) | Luma AI | images |
| Luma Photon Reframe | [`fal-ai/luma-photon/reframe`](https://fal.ai/models/fal-ai/luma-photon/reframe) | Luma AI | images |
| Luma Uni-1 Edit | [`luma/agent/uni-1/v1/edit`](https://fal.ai/models/luma/agent/uni-1/v1/edit) | Luma AI | images |
| Luma Uni-1 Edit Max | [`luma/agent/uni-1/v1/max/edit`](https://fal.ai/models/luma/agent/uni-1/v1/max/edit) | Luma AI | images |
| Mai Image 2.5 | [`microsoft/mai-image-2.5/edit`](https://fal.ai/models/microsoft/mai-image-2.5/edit) | Microsoft | images |
| Makeup Changer | [`fal-ai/image-apps-v2/makeup-application`](https://fal.ai/models/fal-ai/image-apps-v2/makeup-application) | — | images |
| Marigold Depth Estimation | [`fal-ai/imageutils/marigold-depth`](https://fal.ai/models/fal-ai/imageutils/marigold-depth) | — | image |
| Midas Depth Estimation | [`fal-ai/imageutils/depth`](https://fal.ai/models/fal-ai/imageutils/depth) | — | image |
| Minimax Image Subject Reference | [`fal-ai/minimax/image-01/subject-reference`](https://fal.ai/models/fal-ai/minimax/image-01/subject-reference) | Minimax | images |
| Moondream3 Preview [Segment] | [`fal-ai/moondream3-preview/segment`](https://fal.ai/models/fal-ai/moondream3-preview/segment) | Moondream | image |
| MoonDreamNext Detection | [`fal-ai/moondream-next/detection`](https://fal.ai/models/fal-ai/moondream-next/detection) | Moondream | image |
| NAFNet-deblur | [`fal-ai/nafnet/deblur`](https://fal.ai/models/fal-ai/nafnet/deblur) | — | image |
| NAFNet-denoise | [`fal-ai/nafnet/denoise`](https://fal.ai/models/fal-ai/nafnet/denoise) | — | image |
| Nano Banana | [`fal-ai/nano-banana/edit`](https://fal.ai/models/fal-ai/nano-banana/edit) | Google | images |
| Nano Banana 2 | [`fal-ai/nano-banana-2/edit`](https://fal.ai/models/fal-ai/nano-banana-2/edit) | Google | images |
| Nano Banana Lite Edit | [`google/nano-banana-lite/edit`](https://fal.ai/models/google/nano-banana-lite/edit) | Google | images |
| Nano Banana Pro | [`fal-ai/nano-banana-pro/edit`](https://fal.ai/models/fal-ai/nano-banana-pro/edit) | Google | images |
| Object Removal | [`fal-ai/image-apps-v2/object-removal`](https://fal.ai/models/fal-ai/image-apps-v2/object-removal) | — | images |
| Object Removal | [`fal-ai/object-removal`](https://fal.ai/models/fal-ai/object-removal) | — | images |
| Object Removal | [`fal-ai/object-removal/bbox`](https://fal.ai/models/fal-ai/object-removal/bbox) | — | images |
| Object Removal | [`fal-ai/object-removal/mask`](https://fal.ai/models/fal-ai/object-removal/mask) | — | images |
| Omni Zero | [`fal-ai/omni-zero`](https://fal.ai/models/fal-ai/omni-zero) | — | image |
| Optimized Latent Consistency (SDv1.5) | [`fal-ai/lcm-sd15-i2i`](https://fal.ai/models/fal-ai/lcm-sd15-i2i) | — | images |
| PASD | [`fal-ai/pasd`](https://fal.ai/models/fal-ai/pasd) | — | images |
| PATINA | [`fal-ai/patina`](https://fal.ai/models/fal-ai/patina) | Fal | images |
| PATINA | [`fal-ai/patina/material/extract`](https://fal.ai/models/fal-ai/patina/material/extract) | Fal | images |
| Perspective Change | [`fal-ai/image-apps-v2/perspective`](https://fal.ai/models/fal-ai/image-apps-v2/perspective) | — | images |
| Phota | [`fal-ai/phota/edit`](https://fal.ai/models/fal-ai/phota/edit) | Phota | images |
| Phota Enhance | [`fal-ai/phota/enhance`](https://fal.ai/models/fal-ai/phota/enhance) | Phota | images |
| Photo Restoration | [`fal-ai/image-apps-v2/photo-restoration`](https://fal.ai/models/fal-ai/image-apps-v2/photo-restoration) | — | images |
| Photography Effects | [`fal-ai/image-apps-v2/photography-effects`](https://fal.ai/models/fal-ai/image-apps-v2/photography-effects) | — | images |
| PhotoMaker | [`fal-ai/photomaker`](https://fal.ai/models/fal-ai/photomaker) | — | images |
| Pixelcut Background Remover | [`pixelcut/background-removal`](https://fal.ai/models/pixelcut/background-removal) | Pixelcut | image |
| Playground v2.5 | [`fal-ai/playground-v25/image-to-image`](https://fal.ai/models/fal-ai/playground-v25/image-to-image) | — | images |
| Playground v2.5 | [`fal-ai/playground-v25/inpainting`](https://fal.ai/models/fal-ai/playground-v25/inpainting) | — | images |
| Portrait Enhance | [`fal-ai/image-apps-v2/portrait-enhance`](https://fal.ai/models/fal-ai/image-apps-v2/portrait-enhance) | — | images |
| Post Processing | [`fal-ai/post-processing`](https://fal.ai/models/fal-ai/post-processing) | — | images |
| Post Processing Blur | [`fal-ai/post-processing/blur`](https://fal.ai/models/fal-ai/post-processing/blur) | — | images |
| Post Processing Chromatic Aberration | [`fal-ai/post-processing/chromatic-aberration`](https://fal.ai/models/fal-ai/post-processing/chromatic-aberration) | — | images |
| Post Processing Color Correction | [`fal-ai/post-processing/color-correction`](https://fal.ai/models/fal-ai/post-processing/color-correction) | — | images |
| Post Processing Color Tint | [`fal-ai/post-processing/color-tint`](https://fal.ai/models/fal-ai/post-processing/color-tint) | — | images |
| Post Processing Desaturate | [`fal-ai/post-processing/desaturate`](https://fal.ai/models/fal-ai/post-processing/desaturate) | — | images |
| Post Processing Dissolve | [`fal-ai/post-processing/dissolve`](https://fal.ai/models/fal-ai/post-processing/dissolve) | — | images |
| Post Processing Dodge Burn | [`fal-ai/post-processing/dodge-burn`](https://fal.ai/models/fal-ai/post-processing/dodge-burn) | — | images |
| Post Processing Grain | [`fal-ai/post-processing/grain`](https://fal.ai/models/fal-ai/post-processing/grain) | — | images |
| Post Processing Parabolize | [`fal-ai/post-processing/parabolize`](https://fal.ai/models/fal-ai/post-processing/parabolize) | — | images |
| Post Processing Sharpen | [`fal-ai/post-processing/sharpen`](https://fal.ai/models/fal-ai/post-processing/sharpen) | — | images |
| Post Processing Solarize | [`fal-ai/post-processing/solarize`](https://fal.ai/models/fal-ai/post-processing/solarize) | — | images |
| Post Processing Vignette | [`fal-ai/post-processing/vignette`](https://fal.ai/models/fal-ai/post-processing/vignette) | — | images |
| Product Holding | [`fal-ai/image-apps-v2/product-holding`](https://fal.ai/models/fal-ai/image-apps-v2/product-holding) | — | images |
| Product Photography | [`fal-ai/image-apps-v2/product-photography`](https://fal.ai/models/fal-ai/image-apps-v2/product-photography) | — | images |
| PuLID | [`fal-ai/pulid`](https://fal.ai/models/fal-ai/pulid) | — | images |
| PuLID Flux | [`fal-ai/flux-pulid`](https://fal.ai/models/fal-ai/flux-pulid) | Black Forest Labs | images |
| Qwen Image | [`fal-ai/qwen-image/image-to-image`](https://fal.ai/models/fal-ai/qwen-image/image-to-image) | Alibaba | images |
| Qwen Image 2 | [`fal-ai/qwen-image-2/edit`](https://fal.ai/models/fal-ai/qwen-image-2/edit) | Alibaba | images |
| Qwen Image 2 | [`fal-ai/qwen-image-2/pro/edit`](https://fal.ai/models/fal-ai/qwen-image-2/pro/edit) | Alibaba | images |
| Qwen Image Edit | [`fal-ai/qwen-image-edit`](https://fal.ai/models/fal-ai/qwen-image-edit) | Alibaba | images |
| Qwen Image Edit | [`fal-ai/qwen-image-edit/image-to-image`](https://fal.ai/models/fal-ai/qwen-image-edit/image-to-image) | Alibaba | images |
| Qwen Image Edit | [`fal-ai/qwen-image-edit/inpaint`](https://fal.ai/models/fal-ai/qwen-image-edit/inpaint) | Alibaba | images |
| Qwen Image Edit 2509 | [`fal-ai/qwen-image-edit-2509`](https://fal.ai/models/fal-ai/qwen-image-edit-2509) | Alibaba | images |
| Qwen Image Edit 2509 Lora | [`fal-ai/qwen-image-edit-2509-lora`](https://fal.ai/models/fal-ai/qwen-image-edit-2509-lora) | Alibaba | images |
| Qwen Image Edit 2509 Lora Gallery | [`fal-ai/qwen-image-edit-2509-lora-gallery/add-background`](https://fal.ai/models/fal-ai/qwen-image-edit-2509-lora-gallery/add-background) | Alibaba | images |
| Qwen Image Edit 2509 Lora Gallery | [`fal-ai/qwen-image-edit-2509-lora-gallery/face-to-full-portrait`](https://fal.ai/models/fal-ai/qwen-image-edit-2509-lora-gallery/face-to-full-portrait) | Alibaba | images |
| Qwen Image Edit 2509 Lora Gallery | [`fal-ai/qwen-image-edit-2509-lora-gallery/group-photo`](https://fal.ai/models/fal-ai/qwen-image-edit-2509-lora-gallery/group-photo) | Alibaba | images |
| Qwen Image Edit 2509 Lora Gallery | [`fal-ai/qwen-image-edit-2509-lora-gallery/integrate-product`](https://fal.ai/models/fal-ai/qwen-image-edit-2509-lora-gallery/integrate-product) | Alibaba | images |
| Qwen Image Edit 2509 Lora Gallery | [`fal-ai/qwen-image-edit-2509-lora-gallery/lighting-restoration`](https://fal.ai/models/fal-ai/qwen-image-edit-2509-lora-gallery/lighting-restoration) | Alibaba | images |
| Qwen Image Edit 2509 Lora Gallery | [`fal-ai/qwen-image-edit-2509-lora-gallery/multiple-angles`](https://fal.ai/models/fal-ai/qwen-image-edit-2509-lora-gallery/multiple-angles) | Alibaba | images |
| Qwen Image Edit 2509 Lora Gallery | [`fal-ai/qwen-image-edit-2509-lora-gallery/next-scene`](https://fal.ai/models/fal-ai/qwen-image-edit-2509-lora-gallery/next-scene) | Alibaba | images |
| Qwen Image Edit 2509 Lora Gallery | [`fal-ai/qwen-image-edit-2509-lora-gallery/remove-element`](https://fal.ai/models/fal-ai/qwen-image-edit-2509-lora-gallery/remove-element) | Alibaba | images |
| Qwen Image Edit 2509 Lora Gallery | [`fal-ai/qwen-image-edit-2509-lora-gallery/remove-lighting`](https://fal.ai/models/fal-ai/qwen-image-edit-2509-lora-gallery/remove-lighting) | Alibaba | images |
| Qwen Image Edit 2509 Lora Gallery | [`fal-ai/qwen-image-edit-2509-lora-gallery/shirt-design`](https://fal.ai/models/fal-ai/qwen-image-edit-2509-lora-gallery/shirt-design) | Alibaba | images |
| Qwen Image Edit 2511 | [`fal-ai/qwen-image-edit-2511`](https://fal.ai/models/fal-ai/qwen-image-edit-2511) | Alibaba | images |
| Qwen Image Edit 2511 | [`fal-ai/qwen-image-edit-2511/lora`](https://fal.ai/models/fal-ai/qwen-image-edit-2511/lora) | Alibaba | images |
| Qwen Image Edit 2511 Multiple Angles | [`fal-ai/qwen-image-edit-2511-multiple-angles`](https://fal.ai/models/fal-ai/qwen-image-edit-2511-multiple-angles) | Alibaba | images |
| Qwen Image Edit Lora | [`fal-ai/qwen-image-edit-lora`](https://fal.ai/models/fal-ai/qwen-image-edit-lora) | Alibaba | images |
| Qwen Image Edit Plus | [`fal-ai/qwen-image-edit-plus`](https://fal.ai/models/fal-ai/qwen-image-edit-plus) | Alibaba | images |
| Qwen Image Edit Plus Lora | [`fal-ai/qwen-image-edit-plus-lora`](https://fal.ai/models/fal-ai/qwen-image-edit-plus-lora) | Alibaba | images |
| Qwen Image Edit Plus Lora Gallery | [`fal-ai/qwen-image-edit-plus-lora-gallery/add-background`](https://fal.ai/models/fal-ai/qwen-image-edit-plus-lora-gallery/add-background) | Alibaba | images |
| Qwen Image Edit Plus Lora Gallery | [`fal-ai/qwen-image-edit-plus-lora-gallery/face-to-full-portrait`](https://fal.ai/models/fal-ai/qwen-image-edit-plus-lora-gallery/face-to-full-portrait) | Alibaba | images |
| Qwen Image Edit Plus Lora Gallery | [`fal-ai/qwen-image-edit-plus-lora-gallery/group-photo`](https://fal.ai/models/fal-ai/qwen-image-edit-plus-lora-gallery/group-photo) | Alibaba | images |
| Qwen Image Edit Plus Lora Gallery | [`fal-ai/qwen-image-edit-plus-lora-gallery/integrate-product`](https://fal.ai/models/fal-ai/qwen-image-edit-plus-lora-gallery/integrate-product) | Alibaba | images |
| Qwen Image Edit Plus Lora Gallery | [`fal-ai/qwen-image-edit-plus-lora-gallery/lighting-restoration`](https://fal.ai/models/fal-ai/qwen-image-edit-plus-lora-gallery/lighting-restoration) | Alibaba | images |
| Qwen Image Edit Plus Lora Gallery | [`fal-ai/qwen-image-edit-plus-lora-gallery/multiple-angles`](https://fal.ai/models/fal-ai/qwen-image-edit-plus-lora-gallery/multiple-angles) | Alibaba | images |
| Qwen Image Edit Plus Lora Gallery | [`fal-ai/qwen-image-edit-plus-lora-gallery/next-scene`](https://fal.ai/models/fal-ai/qwen-image-edit-plus-lora-gallery/next-scene) | Alibaba | images |
| Qwen Image Edit Plus Lora Gallery | [`fal-ai/qwen-image-edit-plus-lora-gallery/remove-element`](https://fal.ai/models/fal-ai/qwen-image-edit-plus-lora-gallery/remove-element) | Alibaba | images |
| Qwen Image Edit Plus Lora Gallery | [`fal-ai/qwen-image-edit-plus-lora-gallery/remove-lighting`](https://fal.ai/models/fal-ai/qwen-image-edit-plus-lora-gallery/remove-lighting) | Alibaba | images |
| Qwen Image Edit Plus Lora Gallery | [`fal-ai/qwen-image-edit-plus-lora-gallery/shirt-design`](https://fal.ai/models/fal-ai/qwen-image-edit-plus-lora-gallery/shirt-design) | Alibaba | images |
| Qwen Image Layered | [`fal-ai/qwen-image-layered`](https://fal.ai/models/fal-ai/qwen-image-layered) | Alibaba | images |
| Qwen Image Layered | [`fal-ai/qwen-image-layered/lora`](https://fal.ai/models/fal-ai/qwen-image-layered/lora) | Alibaba | images |
| Qwen Image Max | [`fal-ai/qwen-image-max/edit`](https://fal.ai/models/fal-ai/qwen-image-max/edit) | Alibaba | images |
| Recraft | [`fal-ai/recraft/vectorize`](https://fal.ai/models/fal-ai/recraft/vectorize) | Recraft | image |
| Recraft Creative Upscale | [`fal-ai/recraft/upscale/creative`](https://fal.ai/models/fal-ai/recraft/upscale/creative) | Recraft | image |
| Recraft Crisp Upscale | [`fal-ai/recraft/upscale/crisp`](https://fal.ai/models/fal-ai/recraft/upscale/crisp) | Recraft | image |
| Recraft V3 | [`fal-ai/recraft/v3/image-to-image`](https://fal.ai/models/fal-ai/recraft/v3/image-to-image) | Recraft | images |
| Relighting | [`fal-ai/image-apps-v2/relighting`](https://fal.ai/models/fal-ai/image-apps-v2/relighting) | — | images |
| Rembg Enhance (Remove Background Enhance) | [`smoretalk-ai/rembg-enhance`](https://fal.ai/models/smoretalk-ai/rembg-enhance) | Smoretalk AI | image |
| Remove Background | [`fal-ai/imageutils/rembg`](https://fal.ai/models/fal-ai/imageutils/rembg) | — | image |
| Replace Background | [`bria/replace-background`](https://fal.ai/models/bria/replace-background) | Bria AI | json |
| RIFE | [`fal-ai/rife`](https://fal.ai/models/fal-ai/rife) | — | images |
| Sam 3 | [`fal-ai/sam-3/image-rle`](https://fal.ai/models/fal-ai/sam-3/image-rle) | — | json |
| Sam 3 1 | [`fal-ai/sam-3-1/image`](https://fal.ai/models/fal-ai/sam-3-1/image) | — | image |
| Sam 3 1 | [`fal-ai/sam-3-1/image-rle`](https://fal.ai/models/fal-ai/sam-3-1/image-rle) | — | json |
| SDXL ControlNet Union | [`fal-ai/sdxl-controlnet-union/image-to-image`](https://fal.ai/models/fal-ai/sdxl-controlnet-union/image-to-image) | — | images |
| SDXL ControlNet Union | [`fal-ai/sdxl-controlnet-union/inpainting`](https://fal.ai/models/fal-ai/sdxl-controlnet-union/inpainting) | — | images |
| SeedVR2 | [`fal-ai/seedvr/upscale/image`](https://fal.ai/models/fal-ai/seedvr/upscale/image) | SeedVR | image |
| SeedVR2 | [`fal-ai/seedvr/upscale/image/seamless`](https://fal.ai/models/fal-ai/seedvr/upscale/image/seamless) | SeedVR | image |
| Segment Anything Model 2 | [`fal-ai/sam2/auto-segment`](https://fal.ai/models/fal-ai/sam2/auto-segment) | — | json |
| Segment Anything Model 2 | [`fal-ai/sam2/image`](https://fal.ai/models/fal-ai/sam2/image) | — | image |
| Segment Anything Model 3 | [`fal-ai/sam-3/image`](https://fal.ai/models/fal-ai/sam-3/image) | — | image |
| Smart Resize | [`fal-ai/smart-resize`](https://fal.ai/models/fal-ai/smart-resize) | — | images |
| Stable Diffusion V3 | [`fal-ai/stable-diffusion-v3-medium/image-to-image`](https://fal.ai/models/fal-ai/stable-diffusion-v3-medium/image-to-image) | Stability AI | images |
| Stable Diffusion with LoRAs | [`fal-ai/lora/image-to-image`](https://fal.ai/models/fal-ai/lora/image-to-image) | — | images |
| Stable Diffusion with LoRAs | [`fal-ai/lora/inpaint`](https://fal.ai/models/fal-ai/lora/inpaint) | — | images |
| Stable Diffusion XL | [`fal-ai/fast-sdxl/image-to-image`](https://fal.ai/models/fal-ai/fast-sdxl/image-to-image) | — | images |
| Stable Diffusion XL | [`fal-ai/fast-sdxl/inpainting`](https://fal.ai/models/fal-ai/fast-sdxl/inpainting) | — | images |
| Stable Diffusion XL Lightning | [`fal-ai/fast-lightning-sdxl/image-to-image`](https://fal.ai/models/fal-ai/fast-lightning-sdxl/image-to-image) | — | images |
| Stable Diffusion XL Lightning | [`fal-ai/fast-lightning-sdxl/inpainting`](https://fal.ai/models/fal-ai/fast-lightning-sdxl/inpainting) | — | images |
| Stepx Edit2 | [`fal-ai/stepx-edit2`](https://fal.ai/models/fal-ai/stepx-edit2) | — | images |
| Style Transfer | [`fal-ai/image-apps-v2/style-transfer`](https://fal.ai/models/fal-ai/image-apps-v2/style-transfer) | — | images |
| Telestyle V2 Style Transfer | [`fal-ai/telestyle-v2`](https://fal.ai/models/fal-ai/telestyle-v2) | — | images |
| Texture Transform | [`fal-ai/image-apps-v2/texture-transform`](https://fal.ai/models/fal-ai/image-apps-v2/texture-transform) | — | images |
| Topaz | [`fal-ai/topaz/upscale/image`](https://fal.ai/models/fal-ai/topaz/upscale/image) | Topaz Labs | image |
| try-on | [`fal-ai/cat-vton`](https://fal.ai/models/fal-ai/cat-vton) | — | image |
| Uno | [`fal-ai/uno`](https://fal.ai/models/fal-ai/uno) | — | images |
| Upscale | [`bria/upscale/creative`](https://fal.ai/models/bria/upscale/creative) | Bria AI | image |
| Upscale Images | [`fal-ai/esrgan`](https://fal.ai/models/fal-ai/esrgan) | — | image |
| Uso | [`fal-ai/uso`](https://fal.ai/models/fal-ai/uso) | — | images |
| Vecglypher | [`fal-ai/vecglypher/image-to-svg`](https://fal.ai/models/fal-ai/vecglypher/image-to-svg) | — | image |
| Vidu | [`fal-ai/vidu/q2/reference-to-image`](https://fal.ai/models/fal-ai/vidu/q2/reference-to-image) | Vidu | image |
| Vidu | [`fal-ai/vidu/reference-to-image`](https://fal.ai/models/fal-ai/vidu/reference-to-image) | Vidu | image |
| Virtual Try-on | [`fal-ai/image-apps-v2/virtual-try-on`](https://fal.ai/models/fal-ai/image-apps-v2/virtual-try-on) | — | images |
| Wan | [`fal-ai/wan/v2.2-a14b/image-to-image`](https://fal.ai/models/fal-ai/wan/v2.2-a14b/image-to-image) | Alibaba | image |
| Wan | [`fal-ai/wan/v2.7/edit`](https://fal.ai/models/fal-ai/wan/v2.7/edit) | Alibaba | images |
| Wan | [`fal-ai/wan/v2.7/pro/edit`](https://fal.ai/models/fal-ai/wan/v2.7/pro/edit) | Alibaba | images |
| Wan 2.5 Image to Image | [`fal-ai/wan-25-preview/image-to-image`](https://fal.ai/models/fal-ai/wan-25-preview/image-to-image) | Alibaba | images |
| Wan v2.6 Image to Image | [`wan/v2.6/image-to-image`](https://fal.ai/models/wan/v2.6/image-to-image) | Alibaba | images |
| Workflow Utilities Extract Nth Frame | [`fal-ai/workflow-utilities/extract-nth-frame`](https://fal.ai/models/fal-ai/workflow-utilities/extract-nth-frame) | — | images |
| Z Image Turbo Controlnet | [`fal-ai/z-image/turbo/controlnet`](https://fal.ai/models/fal-ai/z-image/turbo/controlnet) | Alibaba | images |
| Z Image Turbo Controlnet Lora | [`fal-ai/z-image/turbo/controlnet/lora`](https://fal.ai/models/fal-ai/z-image/turbo/controlnet/lora) | Alibaba | images |
| Z Image Turbo Image To Image | [`fal-ai/z-image/turbo/image-to-image`](https://fal.ai/models/fal-ai/z-image/turbo/image-to-image) | Alibaba | images |
| Z Image Turbo Image To Image Lora | [`fal-ai/z-image/turbo/image-to-image/lora`](https://fal.ai/models/fal-ai/z-image/turbo/image-to-image/lora) | Alibaba | images |
| Z Image Turbo Inpaint | [`fal-ai/z-image/turbo/inpaint`](https://fal.ai/models/fal-ai/z-image/turbo/inpaint) | Alibaba | images |
| Z Image Turbo Inpaint Lora | [`fal-ai/z-image/turbo/inpaint/lora`](https://fal.ai/models/fal-ai/z-image/turbo/inpaint/lora) | Alibaba | images |

</details>

<details>
<summary><strong>image-to-video</strong> — 193 models</summary>

| Model | Endpoint | Lab | Output |
| --- | --- | --- | --- |
| AI Avatar Multi | [`fal-ai/ai-avatar/multi`](https://fal.ai/models/fal-ai/ai-avatar/multi) | — | video |
| AI Avatar Multi Text | [`fal-ai/ai-avatar/multi-text`](https://fal.ai/models/fal-ai/ai-avatar/multi-text) | — | video |
| AI Avatar Single Text | [`fal-ai/ai-avatar/single-text`](https://fal.ai/models/fal-ai/ai-avatar/single-text) | — | video |
| AMT Frame Interpolation | [`fal-ai/amt-interpolation/frame-interpolation`](https://fal.ai/models/fal-ai/amt-interpolation/frame-interpolation) | — | video |
| Bernini-R Reference to Video | [`fal-ai/bernini-r/reference-to-video`](https://fal.ai/models/fal-ai/bernini-r/reference-to-video) | Bytedance | video |
| Bytedance Omnihuman V1.5 | [`fal-ai/bytedance/omnihuman/v1.5`](https://fal.ai/models/fal-ai/bytedance/omnihuman/v1.5) | Bytedance | video |
| Bytedance Seedance V1 Pro Fast Image To Video | [`fal-ai/bytedance/seedance/v1/pro/fast/image-to-video`](https://fal.ai/models/fal-ai/bytedance/seedance/v1/pro/fast/image-to-video) | Bytedance | video |
| Bytedance Seedance V1.5 Pro Image To Video | [`fal-ai/bytedance/seedance/v1.5/pro/image-to-video`](https://fal.ai/models/fal-ai/bytedance/seedance/v1.5/pro/image-to-video) | Bytedance | video |
| CogVideoX-5B | [`fal-ai/cogvideox-5b/image-to-video`](https://fal.ai/models/fal-ai/cogvideox-5b/image-to-video) | — | video |
| Cosmos 3 Super Image to Video | [`nvidia/cosmos-3-super/image-to-video`](https://fal.ai/models/nvidia/cosmos-3-super/image-to-video) | NVIDIA | video |
| Cosmos Predict 2.5 2B | [`fal-ai/cosmos-predict-2.5/image-to-video`](https://fal.ai/models/fal-ai/cosmos-predict-2.5/image-to-video) | NVIDIA | video |
| Creatify Aurora | [`fal-ai/creatify/aurora`](https://fal.ai/models/fal-ai/creatify/aurora) | Creatify AI | video |
| Davinci Magihuman | [`fal-ai/davinci-magihuman`](https://fal.ai/models/fal-ai/davinci-magihuman) | — | video |
| Fabric 1.0 | [`veed/fabric-1.0`](https://fal.ai/models/veed/fabric-1.0) | Veed | video |
| Fabric 1.0 Fast | [`veed/fabric-1.0/fast`](https://fal.ai/models/veed/fabric-1.0/fast) | Veed | video |
| Ffmpeg Api Images to Video | [`fal-ai/ffmpeg-api/images-to-video`](https://fal.ai/models/fal-ai/ffmpeg-api/images-to-video) | — | video |
| Flashhead | [`fal-ai/flashhead`](https://fal.ai/models/fal-ai/flashhead) | Soul AI Lab | video |
| Framepack | [`fal-ai/framepack`](https://fal.ai/models/fal-ai/framepack) | — | video |
| Framepack | [`fal-ai/framepack/flf2v`](https://fal.ai/models/fal-ai/framepack/flf2v) | — | video |
| Framepack F1 | [`fal-ai/framepack/f1`](https://fal.ai/models/fal-ai/framepack/f1) | — | video |
| Gemini Omni Flash | [`google/gemini-omni-flash/image-to-video`](https://fal.ai/models/google/gemini-omni-flash/image-to-video) | Google | video |
| Gemini Omni Flash | [`google/gemini-omni-flash/reference-to-video`](https://fal.ai/models/google/gemini-omni-flash/reference-to-video) | Google | video |
| Grok Imagine Reference to Video | [`xai/grok-imagine-video/reference-to-video`](https://fal.ai/models/xai/grok-imagine-video/reference-to-video) | xAI | video |
| Grok Imagine Video | [`xai/grok-imagine-video/image-to-video`](https://fal.ai/models/xai/grok-imagine-video/image-to-video) | xAI | video |
| Grok Imagine Video 1.5 | [`xai/grok-imagine-video/v1.5/image-to-video`](https://fal.ai/models/xai/grok-imagine-video/v1.5/image-to-video) | xAI | video |
| Happy Horse | [`alibaba/happy-horse/image-to-video`](https://fal.ai/models/alibaba/happy-horse/image-to-video) | Alibaba | video |
| Happy Horse | [`alibaba/happy-horse/reference-to-video`](https://fal.ai/models/alibaba/happy-horse/reference-to-video) | Alibaba | video |
| Happy Horse 1.1 Image to Video | [`alibaba/happy-horse/v1.1/image-to-video`](https://fal.ai/models/alibaba/happy-horse/v1.1/image-to-video) | Alibaba | video |
| Happy Horse 1.1 Reference to Video | [`alibaba/happy-horse/v1.1/reference-to-video`](https://fal.ai/models/alibaba/happy-horse/v1.1/reference-to-video) | Alibaba | video |
| Heygen | [`fal-ai/heygen/avatar4/image-to-video`](https://fal.ai/models/fal-ai/heygen/avatar4/image-to-video) | Heygen | video |
| High Quality Stable Video Diffusion | [`fal-ai/stable-video`](https://fal.ai/models/fal-ai/stable-video) | Stability AI | video |
| Hunyuan Video Image-to-Video Inference | [`fal-ai/hunyuan-video-image-to-video`](https://fal.ai/models/fal-ai/hunyuan-video-image-to-video) | Tencent | video |
| Hunyuan Video Image-to-Video LoRA Inference | [`fal-ai/hunyuan-video-img2vid-lora`](https://fal.ai/models/fal-ai/hunyuan-video-img2vid-lora) | Tencent | video |
| Hunyuan Video V1.5 | [`fal-ai/hunyuan-video-v1.5/image-to-video`](https://fal.ai/models/fal-ai/hunyuan-video-v1.5/image-to-video) | Tencent | video |
| Kandinsky5 Pro | [`fal-ai/kandinsky5-pro/image-to-video`](https://fal.ai/models/fal-ai/kandinsky5-pro/image-to-video) | Kandinsky | video |
| Kling 1.0 | [`fal-ai/kling-video/v1/standard/image-to-video`](https://fal.ai/models/fal-ai/kling-video/v1/standard/image-to-video) | Kling | video |
| Kling 1.5 | [`fal-ai/kling-video/v1.5/pro/image-to-video`](https://fal.ai/models/fal-ai/kling-video/v1.5/pro/image-to-video) | Kling | video |
| Kling 1.6 | [`fal-ai/kling-video/v1.6/pro/image-to-video`](https://fal.ai/models/fal-ai/kling-video/v1.6/pro/image-to-video) | Kling | video |
| Kling 1.6 | [`fal-ai/kling-video/v1.6/standard/image-to-video`](https://fal.ai/models/fal-ai/kling-video/v1.6/standard/image-to-video) | Kling | video |
| Kling 1.6 Elements | [`fal-ai/kling-video/v1.6/pro/elements`](https://fal.ai/models/fal-ai/kling-video/v1.6/pro/elements) | Kling | video |
| Kling 1.6 Elements | [`fal-ai/kling-video/v1.6/standard/elements`](https://fal.ai/models/fal-ai/kling-video/v1.6/standard/elements) | Kling | video |
| Kling 2.0 Master | [`fal-ai/kling-video/v2/master/image-to-video`](https://fal.ai/models/fal-ai/kling-video/v2/master/image-to-video) | Kling | video |
| Kling 2.1 (pro) | [`fal-ai/kling-video/v2.1/pro/image-to-video`](https://fal.ai/models/fal-ai/kling-video/v2.1/pro/image-to-video) | Kling | video |
| Kling 2.1 (standard) | [`fal-ai/kling-video/v2.1/standard/image-to-video`](https://fal.ai/models/fal-ai/kling-video/v2.1/standard/image-to-video) | Kling | video |
| Kling 2.1 Master | [`fal-ai/kling-video/v2.1/master/image-to-video`](https://fal.ai/models/fal-ai/kling-video/v2.1/master/image-to-video) | Kling | video |
| Kling AI Avatar | [`fal-ai/kling-video/v1/standard/ai-avatar`](https://fal.ai/models/fal-ai/kling-video/v1/standard/ai-avatar) | Kling | video |
| Kling AI Avatar Pro | [`fal-ai/kling-video/v1/pro/ai-avatar`](https://fal.ai/models/fal-ai/kling-video/v1/pro/ai-avatar) | Kling | video |
| Kling AI Avatar v2 Pro | [`fal-ai/kling-video/ai-avatar/v2/pro`](https://fal.ai/models/fal-ai/kling-video/ai-avatar/v2/pro) | Kling | video |
| Kling AI Avatar v2 Standard | [`fal-ai/kling-video/ai-avatar/v2/standard`](https://fal.ai/models/fal-ai/kling-video/ai-avatar/v2/standard) | Kling | video |
| Kling O1 First Frame Last Frame to Video [Pro] | [`fal-ai/kling-video/o1/image-to-video`](https://fal.ai/models/fal-ai/kling-video/o1/image-to-video) | Kling | video |
| Kling O1 First Frame Last Frame to Video [Standard] | [`fal-ai/kling-video/o1/standard/image-to-video`](https://fal.ai/models/fal-ai/kling-video/o1/standard/image-to-video) | Kling | video |
| Kling O1 Reference Image to Video [Pro] | [`fal-ai/kling-video/o1/reference-to-video`](https://fal.ai/models/fal-ai/kling-video/o1/reference-to-video) | Kling | video |
| Kling O1 Reference Image to Video [Standard] | [`fal-ai/kling-video/o1/standard/reference-to-video`](https://fal.ai/models/fal-ai/kling-video/o1/standard/reference-to-video) | Kling | video |
| Kling O3 Image to Video [Pro] | [`fal-ai/kling-video/o3/pro/image-to-video`](https://fal.ai/models/fal-ai/kling-video/o3/pro/image-to-video) | Kling | video |
| Kling O3 Image to Video [Pro] | [`fal-ai/kling-video/o3/standard/image-to-video`](https://fal.ai/models/fal-ai/kling-video/o3/standard/image-to-video) | Kling | video |
| Kling O3 Reference to Video [Pro] | [`fal-ai/kling-video/o3/pro/reference-to-video`](https://fal.ai/models/fal-ai/kling-video/o3/pro/reference-to-video) | Kling | video |
| Kling O3 Reference to Video [Standard] | [`fal-ai/kling-video/o3/standard/reference-to-video`](https://fal.ai/models/fal-ai/kling-video/o3/standard/reference-to-video) | Kling | video |
| Kling Video | [`fal-ai/kling-video/o3/4k/image-to-video`](https://fal.ai/models/fal-ai/kling-video/o3/4k/image-to-video) | Kling | video |
| Kling Video | [`fal-ai/kling-video/o3/4k/reference-to-video`](https://fal.ai/models/fal-ai/kling-video/o3/4k/reference-to-video) | Kling | video |
| Kling Video | [`fal-ai/kling-video/v2.5-turbo/pro/image-to-video`](https://fal.ai/models/fal-ai/kling-video/v2.5-turbo/pro/image-to-video) | Kling | video |
| Kling Video | [`fal-ai/kling-video/v2.5-turbo/standard/image-to-video`](https://fal.ai/models/fal-ai/kling-video/v2.5-turbo/standard/image-to-video) | Kling | video |
| Kling Video | [`fal-ai/kling-video/v3/4k/image-to-video`](https://fal.ai/models/fal-ai/kling-video/v3/4k/image-to-video) | Kling | video |
| Kling Video v2.6 Image to Video | [`fal-ai/kling-video/v2.6/pro/image-to-video`](https://fal.ai/models/fal-ai/kling-video/v2.6/pro/image-to-video) | Kling | video |
| Kling Video v3 Image to Video [Pro] | [`fal-ai/kling-video/v3/pro/image-to-video`](https://fal.ai/models/fal-ai/kling-video/v3/pro/image-to-video) | Kling | video |
| Kling Video v3 Image to Video [Standard] | [`fal-ai/kling-video/v3/standard/image-to-video`](https://fal.ai/models/fal-ai/kling-video/v3/standard/image-to-video) | Kling | video |
| Kling Video V3 Standard Turbo Image to Video | [`fal-ai/kling-video/v3/turbo/standard/image-to-video`](https://fal.ai/models/fal-ai/kling-video/v3/turbo/standard/image-to-video) | Kling | video |
| Kling Video V3 Turbo Pro Image to Video | [`fal-ai/kling-video/v3/turbo/pro/image-to-video`](https://fal.ai/models/fal-ai/kling-video/v3/turbo/pro/image-to-video) | Kling | video |
| Live Portrait | [`fal-ai/live-portrait`](https://fal.ai/models/fal-ai/live-portrait) | — | video |
| LongCat Video | [`fal-ai/longcat-video/image-to-video/480p`](https://fal.ai/models/fal-ai/longcat-video/image-to-video/480p) | — | video |
| LongCat Video | [`fal-ai/longcat-video/image-to-video/720p`](https://fal.ai/models/fal-ai/longcat-video/image-to-video/720p) | — | video |
| LongCat Video Distilled | [`fal-ai/longcat-video/distilled/image-to-video/480p`](https://fal.ai/models/fal-ai/longcat-video/distilled/image-to-video/480p) | — | video |
| LongCat Video Distilled | [`fal-ai/longcat-video/distilled/image-to-video/720p`](https://fal.ai/models/fal-ai/longcat-video/distilled/image-to-video/720p) | — | video |
| Ltx 2.3 Quality | [`fal-ai/ltx-2.3-quality/image-to-video`](https://fal.ai/models/fal-ai/ltx-2.3-quality/image-to-video) | Lightricks | video |
| Ltx 2.3 Quality | [`fal-ai/ltx-2.3-quality/image-to-video/lora`](https://fal.ai/models/fal-ai/ltx-2.3-quality/image-to-video/lora) | Lightricks | video |
| Ltx 2.3 Quality | [`fal-ai/ltx-2.3-quality/ingredient`](https://fal.ai/models/fal-ai/ltx-2.3-quality/ingredient) | Lightricks | video |
| LTX 2.3 Video Fast | [`fal-ai/ltx-2.3/image-to-video/fast`](https://fal.ai/models/fal-ai/ltx-2.3/image-to-video/fast) | Lightricks | video |
| LTX 2.3 Video Pro | [`fal-ai/ltx-2.3/image-to-video`](https://fal.ai/models/fal-ai/ltx-2.3/image-to-video) | Lightricks | video |
| LTX Video (preview) | [`fal-ai/ltx-video/image-to-video`](https://fal.ai/models/fal-ai/ltx-video/image-to-video) | Lightricks | video |
| LTX Video 2.0 Fast | [`fal-ai/ltx-2/image-to-video/fast`](https://fal.ai/models/fal-ai/ltx-2/image-to-video/fast) | Lightricks | video |
| LTX Video 2.0 Pro | [`fal-ai/ltx-2/image-to-video`](https://fal.ai/models/fal-ai/ltx-2/image-to-video) | Lightricks | video |
| LTX Video-0.9.7 13B Distilled | [`fal-ai/ltx-video-13b-distilled/image-to-video`](https://fal.ai/models/fal-ai/ltx-video-13b-distilled/image-to-video) | Lightricks | video |
| LTX-2 19B | [`fal-ai/ltx-2-19b/image-to-video`](https://fal.ai/models/fal-ai/ltx-2-19b/image-to-video) | Lightricks | video |
| LTX-2 19B | [`fal-ai/ltx-2-19b/image-to-video/lora`](https://fal.ai/models/fal-ai/ltx-2-19b/image-to-video/lora) | Lightricks | video |
| LTX-2 19B Distilled | [`fal-ai/ltx-2-19b/distilled/image-to-video`](https://fal.ai/models/fal-ai/ltx-2-19b/distilled/image-to-video) | Lightricks | video |
| LTX-2 19B Distilled | [`fal-ai/ltx-2-19b/distilled/image-to-video/lora`](https://fal.ai/models/fal-ai/ltx-2-19b/distilled/image-to-video/lora) | Lightricks | video |
| LTX-2.3 22B | [`fal-ai/ltx-2.3-22b/image-to-video`](https://fal.ai/models/fal-ai/ltx-2.3-22b/image-to-video) | Lightricks | video |
| LTX-2.3 22B | [`fal-ai/ltx-2.3-22b/image-to-video/lora`](https://fal.ai/models/fal-ai/ltx-2.3-22b/image-to-video/lora) | Lightricks | video |
| LTX-2.3 22B Distilled | [`fal-ai/ltx-2.3-22b/distilled/image-to-video`](https://fal.ai/models/fal-ai/ltx-2.3-22b/distilled/image-to-video) | Lightricks | video |
| LTX-2.3 22B Distilled | [`fal-ai/ltx-2.3-22b/distilled/image-to-video/lora`](https://fal.ai/models/fal-ai/ltx-2.3-22b/distilled/image-to-video/lora) | Lightricks | video |
| LTX-Video 13B 0.9.8 Distilled | [`fal-ai/ltxv-13b-098-distilled/image-to-video`](https://fal.ai/models/fal-ai/ltxv-13b-098-distilled/image-to-video) | Lightricks | video |
| Luma Ray 2 (Image to Video) | [`fal-ai/luma-dream-machine/ray-2/image-to-video`](https://fal.ai/models/fal-ai/luma-dream-machine/ray-2/image-to-video) | Luma AI | video |
| Luma Ray 2 Flash (Image to Video) | [`fal-ai/luma-dream-machine/ray-2-flash/image-to-video`](https://fal.ai/models/fal-ai/luma-dream-machine/ray-2-flash/image-to-video) | Luma AI | video |
| Luma Ray 3.2 Image to Video | [`luma/agent/ray/v3.2/image-to-video`](https://fal.ai/models/luma/agent/ray/v3.2/image-to-video) | Luma AI | video |
| Lynx | [`bytedance/lynx`](https://fal.ai/models/bytedance/lynx) | Bytedance | video |
| MAGI-1 (Distilled) | [`fal-ai/magi-distilled/image-to-video`](https://fal.ai/models/fal-ai/magi-distilled/image-to-video) | — | video |
| Marey Realism V1.5 | [`moonvalley/marey/i2v`](https://fal.ai/models/moonvalley/marey/i2v) | Moonvalley | video |
| Minimax | [`fal-ai/minimax/hailuo-02-fast/image-to-video`](https://fal.ai/models/fal-ai/minimax/hailuo-02-fast/image-to-video) | Minimax | video |
| MiniMax (Hailuo AI) Video 01 | [`fal-ai/minimax/video-01-live/image-to-video`](https://fal.ai/models/fal-ai/minimax/video-01-live/image-to-video) | Minimax | video |
| MiniMax (Hailuo AI) Video 01 | [`fal-ai/minimax/video-01/image-to-video`](https://fal.ai/models/fal-ai/minimax/video-01/image-to-video) | Minimax | video |
| MiniMax (Hailuo AI) Video 01 Director - Image to Video | [`fal-ai/minimax/video-01-director/image-to-video`](https://fal.ai/models/fal-ai/minimax/video-01-director/image-to-video) | Minimax | video |
| MiniMax (Hailuo AI) Video 01 Subject Reference | [`fal-ai/minimax/video-01-subject-reference`](https://fal.ai/models/fal-ai/minimax/video-01-subject-reference) | Minimax | video |
| MiniMax Hailuo 02 [Pro] (Image to Video) | [`fal-ai/minimax/hailuo-02/pro/image-to-video`](https://fal.ai/models/fal-ai/minimax/hailuo-02/pro/image-to-video) | Minimax | video |
| MiniMax Hailuo 02 [Standard] (Image to Video) | [`fal-ai/minimax/hailuo-02/standard/image-to-video`](https://fal.ai/models/fal-ai/minimax/hailuo-02/standard/image-to-video) | Minimax | video |
| MiniMax Hailuo 2.3 [Pro] (Image to Video) | [`fal-ai/minimax/hailuo-2.3/pro/image-to-video`](https://fal.ai/models/fal-ai/minimax/hailuo-2.3/pro/image-to-video) | Minimax | video |
| MiniMax Hailuo 2.3 [Standard] (Image to Video) | [`fal-ai/minimax/hailuo-2.3/standard/image-to-video`](https://fal.ai/models/fal-ai/minimax/hailuo-2.3/standard/image-to-video) | Minimax | video |
| MiniMax Hailuo 2.3 Fast [Pro] (Image to Video) | [`fal-ai/minimax/hailuo-2.3-fast/pro/image-to-video`](https://fal.ai/models/fal-ai/minimax/hailuo-2.3-fast/pro/image-to-video) | Minimax | video |
| MiniMax Hailuo 2.3 Fast [Standard] (Image to Video) | [`fal-ai/minimax/hailuo-2.3-fast/standard/image-to-video`](https://fal.ai/models/fal-ai/minimax/hailuo-2.3-fast/standard/image-to-video) | Minimax | video |
| MuseTalk | [`fal-ai/musetalk`](https://fal.ai/models/fal-ai/musetalk) | — | video |
| OmniHuman | [`fal-ai/bytedance/omnihuman`](https://fal.ai/models/fal-ai/bytedance/omnihuman) | Bytedance | video |
| Ovi | [`fal-ai/ovi/image-to-video`](https://fal.ai/models/fal-ai/ovi/image-to-video) | — | video |
| Pika | [`fal-ai/pika/v2.2/pikaframes`](https://fal.ai/models/fal-ai/pika/v2.2/pikaframes) | Pika | video |
| Pika Effects (v1.5) | [`fal-ai/pika/v1.5/pikaffects`](https://fal.ai/models/fal-ai/pika/v1.5/pikaffects) | Pika | video |
| Pika Image to Video (v2.1) | [`fal-ai/pika/v2.1/image-to-video`](https://fal.ai/models/fal-ai/pika/v2.1/image-to-video) | Pika | video |
| Pika Image to Video (v2.2) | [`fal-ai/pika/v2.2/image-to-video`](https://fal.ai/models/fal-ai/pika/v2.2/image-to-video) | Pika | video |
| Pika Image to Video Turbo (v2) | [`fal-ai/pika/v2/turbo/image-to-video`](https://fal.ai/models/fal-ai/pika/v2/turbo/image-to-video) | Pika | video |
| Pika Scenes (v2.2) | [`fal-ai/pika/v2.2/pikascenes`](https://fal.ai/models/fal-ai/pika/v2.2/pikascenes) | Pika | video |
| PixVerse C1 Image To Video | [`fal-ai/pixverse/c1/image-to-video`](https://fal.ai/models/fal-ai/pixverse/c1/image-to-video) | Pixverse | video |
| PixVerse C1 Reference To Video | [`fal-ai/pixverse/c1/reference-to-video`](https://fal.ai/models/fal-ai/pixverse/c1/reference-to-video) | Pixverse | video |
| PixVerse C1 Transition | [`fal-ai/pixverse/c1/transition`](https://fal.ai/models/fal-ai/pixverse/c1/transition) | Pixverse | video |
| PixVerse Swap | [`fal-ai/pixverse/swap`](https://fal.ai/models/fal-ai/pixverse/swap) | Pixverse | video |
| PixVerse V3.5 Effects | [`fal-ai/pixverse/v3.5/effects`](https://fal.ai/models/fal-ai/pixverse/v3.5/effects) | Pixverse | video |
| PixVerse V3.5 Image To Video | [`fal-ai/pixverse/v3.5/image-to-video`](https://fal.ai/models/fal-ai/pixverse/v3.5/image-to-video) | Pixverse | video |
| PixVerse V3.5 Image To Video Fast | [`fal-ai/pixverse/v3.5/image-to-video/fast`](https://fal.ai/models/fal-ai/pixverse/v3.5/image-to-video/fast) | Pixverse | video |
| PixVerse V3.5 Transition | [`fal-ai/pixverse/v3.5/transition`](https://fal.ai/models/fal-ai/pixverse/v3.5/transition) | Pixverse | video |
| PixVerse V4 Effects | [`fal-ai/pixverse/v4/effects`](https://fal.ai/models/fal-ai/pixverse/v4/effects) | Pixverse | video |
| PixVerse V4 Image To Video | [`fal-ai/pixverse/v4/image-to-video`](https://fal.ai/models/fal-ai/pixverse/v4/image-to-video) | Pixverse | video |
| PixVerse V4 Image To Video Fast | [`fal-ai/pixverse/v4/image-to-video/fast`](https://fal.ai/models/fal-ai/pixverse/v4/image-to-video/fast) | Pixverse | video |
| PixVerse V4.5 Effects | [`fal-ai/pixverse/v4.5/effects`](https://fal.ai/models/fal-ai/pixverse/v4.5/effects) | Pixverse | video |
| PixVerse V4.5 Image To Video | [`fal-ai/pixverse/v4.5/image-to-video`](https://fal.ai/models/fal-ai/pixverse/v4.5/image-to-video) | Pixverse | video |
| PixVerse V4.5 Image To Video Fast | [`fal-ai/pixverse/v4.5/image-to-video/fast`](https://fal.ai/models/fal-ai/pixverse/v4.5/image-to-video/fast) | Pixverse | video |
| PixVerse V4.5 Transition | [`fal-ai/pixverse/v4.5/transition`](https://fal.ai/models/fal-ai/pixverse/v4.5/transition) | Pixverse | video |
| PixVerse V5 Effects | [`fal-ai/pixverse/v5/effects`](https://fal.ai/models/fal-ai/pixverse/v5/effects) | Pixverse | video |
| PixVerse V5 Image To Video | [`fal-ai/pixverse/v5/image-to-video`](https://fal.ai/models/fal-ai/pixverse/v5/image-to-video) | Pixverse | video |
| PixVerse V5 Transition | [`fal-ai/pixverse/v5/transition`](https://fal.ai/models/fal-ai/pixverse/v5/transition) | Pixverse | video |
| PixVerse V5.5 Effects | [`fal-ai/pixverse/v5.5/effects`](https://fal.ai/models/fal-ai/pixverse/v5.5/effects) | Pixverse | video |
| PixVerse V5.5 Image To Video | [`fal-ai/pixverse/v5.5/image-to-video`](https://fal.ai/models/fal-ai/pixverse/v5.5/image-to-video) | Pixverse | video |
| PixVerse V5.5 Transition | [`fal-ai/pixverse/v5.5/transition`](https://fal.ai/models/fal-ai/pixverse/v5.5/transition) | Pixverse | video |
| PixVerse V5.6 Image To Video | [`fal-ai/pixverse/v5.6/image-to-video`](https://fal.ai/models/fal-ai/pixverse/v5.6/image-to-video) | Pixverse | video |
| PixVerse V5.6 Transition | [`fal-ai/pixverse/v5.6/transition`](https://fal.ai/models/fal-ai/pixverse/v5.6/transition) | Pixverse | video |
| PixVerse V6 Image To Video | [`fal-ai/pixverse/v6/image-to-video`](https://fal.ai/models/fal-ai/pixverse/v6/image-to-video) | Pixverse | video |
| PixVerse V6 Transition | [`fal-ai/pixverse/v6/transition`](https://fal.ai/models/fal-ai/pixverse/v6/transition) | Pixverse | video |
| Sad Talker | [`fal-ai/sadtalker`](https://fal.ai/models/fal-ai/sadtalker) | — | video |
| Sad Talker | [`fal-ai/sadtalker/reference`](https://fal.ai/models/fal-ai/sadtalker/reference) | — | video |
| Seedance 1.0 Pro | [`fal-ai/bytedance/seedance/v1/pro/image-to-video`](https://fal.ai/models/fal-ai/bytedance/seedance/v1/pro/image-to-video) | Bytedance | video |
| Seedance 2 Image to Video | [`bytedance/seedance-2.0/image-to-video`](https://fal.ai/models/bytedance/seedance-2.0/image-to-video) | Bytedance | video |
| Seedance 2 Reference to Video | [`bytedance/seedance-2.0/reference-to-video`](https://fal.ai/models/bytedance/seedance-2.0/reference-to-video) | Bytedance | video |
| Seedance 2.0 Fast Image to Video | [`bytedance/seedance-2.0/fast/image-to-video`](https://fal.ai/models/bytedance/seedance-2.0/fast/image-to-video) | Bytedance | video |
| Seedance 2.0 Fast Reference to Video | [`bytedance/seedance-2.0/fast/reference-to-video`](https://fal.ai/models/bytedance/seedance-2.0/fast/reference-to-video) | Bytedance | video |
| Seedance 2.0 Mini | [`bytedance/seedance-2.0/mini/reference-to-video`](https://fal.ai/models/bytedance/seedance-2.0/mini/reference-to-video) | Bytedance | video |
| Seedance 2.0 Mini Image to Video | [`bytedance/seedance-2.0/mini/image-to-video`](https://fal.ai/models/bytedance/seedance-2.0/mini/image-to-video) | Bytedance | video |
| Sora 2 | [`fal-ai/sora-2/characters`](https://fal.ai/models/fal-ai/sora-2/characters) | OpenAI | text |
| Sora 2 | [`fal-ai/sora-2/image-to-video`](https://fal.ai/models/fal-ai/sora-2/image-to-video) | OpenAI | video |
| Sora 2 | [`fal-ai/sora-2/image-to-video/pro`](https://fal.ai/models/fal-ai/sora-2/image-to-video/pro) | OpenAI | video |
| Stable Video Diffusion Turbo | [`fal-ai/fast-svd-lcm`](https://fal.ai/models/fal-ai/fast-svd-lcm) | — | video |
| sync-3 Avatar Image to Video | [`fal-ai/sync-lipsync/v3/image-to-video`](https://fal.ai/models/fal-ai/sync-lipsync/v3/image-to-video) | Syncso | video |
| V2.6 | [`wan/v2.6/image-to-video/flash`](https://fal.ai/models/wan/v2.6/image-to-video/flash) | Alibaba | video |
| Veo 2 (Image to Video) | [`fal-ai/veo2/image-to-video`](https://fal.ai/models/fal-ai/veo2/image-to-video) | Google | video |
| Veo 3 Fast [Image to Video] | [`fal-ai/veo3/fast/image-to-video`](https://fal.ai/models/fal-ai/veo3/fast/image-to-video) | Google | video |
| Veo 3.1 | [`fal-ai/veo3.1/first-last-frame-to-video`](https://fal.ai/models/fal-ai/veo3.1/first-last-frame-to-video) | Google | video |
| Veo 3.1 | [`fal-ai/veo3.1/image-to-video`](https://fal.ai/models/fal-ai/veo3.1/image-to-video) | Google | video |
| Veo 3.1 | [`fal-ai/veo3.1/reference-to-video`](https://fal.ai/models/fal-ai/veo3.1/reference-to-video) | Google | video |
| Veo 3.1 Fast | [`fal-ai/veo3.1/fast/first-last-frame-to-video`](https://fal.ai/models/fal-ai/veo3.1/fast/first-last-frame-to-video) | Google | video |
| Veo 3.1 Fast | [`fal-ai/veo3.1/fast/image-to-video`](https://fal.ai/models/fal-ai/veo3.1/fast/image-to-video) | Google | video |
| Veo 3.1 Fast | [`fal-ai/veo3.1/fast/reference-to-video`](https://fal.ai/models/fal-ai/veo3.1/fast/reference-to-video) | — | video |
| Veo3 | [`fal-ai/veo3/image-to-video`](https://fal.ai/models/fal-ai/veo3/image-to-video) | Google | video |
| Veo3.1 Lite FLF | [`fal-ai/veo3.1/lite/first-last-frame-to-video`](https://fal.ai/models/fal-ai/veo3.1/lite/first-last-frame-to-video) | Google | video |
| Veo3.1 Lite Image to Video | [`fal-ai/veo3.1/lite/image-to-video`](https://fal.ai/models/fal-ai/veo3.1/lite/image-to-video) | Google | video |
| Vidu | [`fal-ai/vidu/q1/reference-to-video`](https://fal.ai/models/fal-ai/vidu/q1/reference-to-video) | Vidu | video |
| Vidu | [`fal-ai/vidu/q2/image-to-video/pro`](https://fal.ai/models/fal-ai/vidu/q2/image-to-video/pro) | Vidu | video |
| Vidu | [`fal-ai/vidu/q2/image-to-video/turbo`](https://fal.ai/models/fal-ai/vidu/q2/image-to-video/turbo) | Vidu | video |
| Vidu | [`fal-ai/vidu/q2/reference-to-video/pro`](https://fal.ai/models/fal-ai/vidu/q2/reference-to-video/pro) | Vidu | video |
| Vidu | [`fal-ai/vidu/q3/image-to-video`](https://fal.ai/models/fal-ai/vidu/q3/image-to-video) | Vidu | video |
| Vidu | [`fal-ai/vidu/q3/image-to-video/turbo`](https://fal.ai/models/fal-ai/vidu/q3/image-to-video/turbo) | Vidu | video |
| Vidu | [`fal-ai/vidu/q3/reference-to-video/mix`](https://fal.ai/models/fal-ai/vidu/q3/reference-to-video/mix) | Vidu | video |
| Vidu Image to Video | [`fal-ai/vidu/image-to-video`](https://fal.ai/models/fal-ai/vidu/image-to-video) | Vidu | video |
| Vidu Image to Video | [`fal-ai/vidu/q1/image-to-video`](https://fal.ai/models/fal-ai/vidu/q1/image-to-video) | Vidu | video |
| Vidu Reference to Video | [`fal-ai/vidu/reference-to-video`](https://fal.ai/models/fal-ai/vidu/reference-to-video) | Vidu | video |
| Vidu Start End to Video | [`fal-ai/vidu/q1/start-end-to-video`](https://fal.ai/models/fal-ai/vidu/q1/start-end-to-video) | Vidu | video |
| Vidu Start-End to Video | [`fal-ai/vidu/start-end-to-video`](https://fal.ai/models/fal-ai/vidu/start-end-to-video) | Vidu | video |
| Vidu Template to Video | [`fal-ai/vidu/template-to-video`](https://fal.ai/models/fal-ai/vidu/template-to-video) | Vidu | video |
| Wan | [`fal-ai/wan/v2.2-a14b/image-to-video/turbo`](https://fal.ai/models/fal-ai/wan/v2.2-a14b/image-to-video/turbo) | Alibaba | video |
| Wan | [`fal-ai/wan/v2.7/image-to-video`](https://fal.ai/models/fal-ai/wan/v2.7/image-to-video) | Alibaba | video |
| Wan 2.5 Image to Video | [`fal-ai/wan-25-preview/image-to-video`](https://fal.ai/models/fal-ai/wan-25-preview/image-to-video) | Alibaba | video |
| Wan 2.7 Reference to Video | [`fal-ai/wan/v2.7/reference-to-video`](https://fal.ai/models/fal-ai/wan/v2.7/reference-to-video) | Alibaba | video |
| Wan Effects | [`fal-ai/wan-effects`](https://fal.ai/models/fal-ai/wan-effects) | Alibaba | video |
| Wan v2.2 5B | [`fal-ai/wan/v2.2-5b/image-to-video`](https://fal.ai/models/fal-ai/wan/v2.2-5b/image-to-video) | Alibaba | video |
| Wan v2.2 A14B | [`fal-ai/wan/v2.2-a14b/image-to-video`](https://fal.ai/models/fal-ai/wan/v2.2-a14b/image-to-video) | Alibaba | video |
| Wan v2.2 A14B Image-to-Video A14B with LoRAs | [`fal-ai/wan/v2.2-a14b/image-to-video/lora`](https://fal.ai/models/fal-ai/wan/v2.2-a14b/image-to-video/lora) | Alibaba | video |
| Wan v2.6 Image to Video | [`wan/v2.6/image-to-video`](https://fal.ai/models/wan/v2.6/image-to-video) | Alibaba | video |
| Wan-2.1 First-Last-Frame-to-Video | [`fal-ai/wan-flf2v`](https://fal.ai/models/fal-ai/wan-flf2v) | Alibaba | video |
| Wan-2.1 Image-to-Video | [`fal-ai/wan-i2v`](https://fal.ai/models/fal-ai/wan-i2v) | Alibaba | video |
| Wan-2.1 Image-to-Video with LoRAs | [`fal-ai/wan-i2v-lora`](https://fal.ai/models/fal-ai/wan-i2v-lora) | Alibaba | video |
| Wan-2.1 Pro Image-to-Video | [`fal-ai/wan-pro/image-to-video`](https://fal.ai/models/fal-ai/wan-pro/image-to-video) | Alibaba | video |

</details>

<details>
<summary><strong>text-to-image</strong> — 189 models</summary>

| Model | Endpoint | Lab | Output |
| --- | --- | --- | --- |
| AuraFlow | [`fal-ai/aura-flow`](https://fal.ai/models/fal-ai/aura-flow) | — | images |
| Bagel | [`fal-ai/bagel`](https://fal.ai/models/fal-ai/bagel) | — | images |
| Bitdance | [`fal-ai/bitdance`](https://fal.ai/models/fal-ai/bitdance) | — | images |
| Boogu Image | [`fal-ai/boogu-image`](https://fal.ai/models/fal-ai/boogu-image) | — | images |
| Bria Text-to-Image Base | [`fal-ai/bria/text-to-image/base`](https://fal.ai/models/fal-ai/bria/text-to-image/base) | Bria AI | images |
| Bria Text-to-Image Fast | [`fal-ai/bria/text-to-image/fast`](https://fal.ai/models/fal-ai/bria/text-to-image/fast) | Bria AI | images |
| Bria Text-to-Image HD | [`fal-ai/bria/text-to-image/hd`](https://fal.ai/models/fal-ai/bria/text-to-image/hd) | Bria AI | images |
| Bytedance Dreamina V3.1 Text To Image | [`fal-ai/bytedance/dreamina/v3.1/text-to-image`](https://fal.ai/models/fal-ai/bytedance/dreamina/v3.1/text-to-image) | Bytedance | images |
| Bytedance Seedream V4 Text To Image | [`fal-ai/bytedance/seedream/v4/text-to-image`](https://fal.ai/models/fal-ai/bytedance/seedream/v4/text-to-image) | Bytedance | images |
| Bytedance Seedream V4.5 Text To Image | [`fal-ai/bytedance/seedream/v4.5/text-to-image`](https://fal.ai/models/fal-ai/bytedance/seedream/v4.5/text-to-image) | Bytedance | images |
| Bytedance Seedream V5 Lite Text To Image | [`fal-ai/bytedance/seedream/v5/lite/text-to-image`](https://fal.ai/models/fal-ai/bytedance/seedream/v5/lite/text-to-image) | Bytedance | images |
| CogView | [`fal-ai/cogview4`](https://fal.ai/models/fal-ai/cogview4) | — | images |
| ControlNet SDXL | [`fal-ai/fast-sdxl-controlnet-canny`](https://fal.ai/models/fal-ai/fast-sdxl-controlnet-canny) | — | images |
| Cosmos 3 Super | [`nvidia/cosmos-3-super/text-to-image`](https://fal.ai/models/nvidia/cosmos-3-super/text-to-image) | NVIDIA | images |
| DeepSeek Janus-Pro | [`fal-ai/janus`](https://fal.ai/models/fal-ai/janus) | — | images |
| Dreamshaper | [`fal-ai/dreamshaper`](https://fal.ai/models/fal-ai/dreamshaper) | — | images |
| Emu 3.5 Image | [`fal-ai/emu-3.5-image/text-to-image`](https://fal.ai/models/fal-ai/emu-3.5-image/text-to-image) | — | images |
| Ernie Image | [`fal-ai/ernie-image`](https://fal.ai/models/fal-ai/ernie-image) | Baidu | images |
| Ernie Image Lora | [`fal-ai/ernie-image/lora`](https://fal.ai/models/fal-ai/ernie-image/lora) | Baidu | images |
| Ernie Image Lora Turbo | [`fal-ai/ernie-image/lora/turbo`](https://fal.ai/models/fal-ai/ernie-image/lora/turbo) | Baidu | images |
| Ernie Image Turbo | [`fal-ai/ernie-image/turbo`](https://fal.ai/models/fal-ai/ernie-image/turbo) | Baidu | images |
| Fibo | [`bria/fibo/generate`](https://fal.ai/models/bria/fibo/generate) | Bria AI | images |
| Fibo Bbq Preview | [`bria/fibo-bbq-preview/generate`](https://fal.ai/models/bria/fibo-bbq-preview/generate) | Bria AI | images |
| Fibo Lite | [`bria/fibo-lite/generate`](https://fal.ai/models/bria/fibo-lite/generate) | Bria AI | images |
| FLUX 2 | [`fal-ai/flux-2`](https://fal.ai/models/fal-ai/flux-2) | Black Forest Labs | images |
| FLUX 2 Flash | [`fal-ai/flux-2/flash`](https://fal.ai/models/fal-ai/flux-2/flash) | Black Forest Labs | images |
| Flux 2 Flex | [`fal-ai/flux-2-flex`](https://fal.ai/models/fal-ai/flux-2-flex) | Black Forest Labs | images |
| FLUX 2 Lora | [`fal-ai/flux-2/lora`](https://fal.ai/models/fal-ai/flux-2/lora) | Black Forest Labs | images |
| Flux 2 Lora Gallery | [`fal-ai/flux-2-lora-gallery/ballpoint-pen-sketch`](https://fal.ai/models/fal-ai/flux-2-lora-gallery/ballpoint-pen-sketch) | Black Forest Labs | images |
| Flux 2 Lora Gallery | [`fal-ai/flux-2-lora-gallery/digital-comic-art`](https://fal.ai/models/fal-ai/flux-2-lora-gallery/digital-comic-art) | Black Forest Labs | images |
| Flux 2 Lora Gallery | [`fal-ai/flux-2-lora-gallery/satellite-view-style`](https://fal.ai/models/fal-ai/flux-2-lora-gallery/satellite-view-style) | Black Forest Labs | images |
| Flux 2 Lora Gallery | [`fal-ai/flux-2-lora-gallery/sepia-vintage`](https://fal.ai/models/fal-ai/flux-2-lora-gallery/sepia-vintage) | Black Forest Labs | images |
| FLUX 2 Lora Gallery Hdr Style | [`fal-ai/flux-2-lora-gallery/hdr-style`](https://fal.ai/models/fal-ai/flux-2-lora-gallery/hdr-style) | Black Forest Labs | images |
| FLUX 2 Lora Gallery Realism | [`fal-ai/flux-2-lora-gallery/realism`](https://fal.ai/models/fal-ai/flux-2-lora-gallery/realism) | Black Forest Labs | images |
| Flux 2 Max | [`fal-ai/flux-2-max`](https://fal.ai/models/fal-ai/flux-2-max) | Black Forest Labs | images |
| Flux 2 Pro | [`fal-ai/flux-2-pro`](https://fal.ai/models/fal-ai/flux-2-pro) | Black Forest Labs | images |
| FLUX 2 Turbo | [`fal-ai/flux-2/turbo`](https://fal.ai/models/fal-ai/flux-2/turbo) | Black Forest Labs | images |
| Flux Kontext Lora | [`fal-ai/flux-kontext-lora/text-to-image`](https://fal.ai/models/fal-ai/flux-kontext-lora/text-to-image) | Black Forest Labs | images |
| Flux Krea Lora | [`fal-ai/flux-krea-lora/stream`](https://fal.ai/models/fal-ai/flux-krea-lora/stream) | Black Forest Labs | images |
| Flux Lora | [`fal-ai/flux-lora/stream`](https://fal.ai/models/fal-ai/flux-lora/stream) | Black Forest Labs | images |
| FLUX.1 [dev] | [`fal-ai/flux-1/dev`](https://fal.ai/models/fal-ai/flux-1/dev) | Black Forest Labs | images |
| FLUX.1 [dev] | [`fal-ai/flux/dev`](https://fal.ai/models/fal-ai/flux/dev) | Black Forest Labs | images |
| FLUX.1 [dev] Control LoRA Canny | [`fal-ai/flux-control-lora-canny`](https://fal.ai/models/fal-ai/flux-control-lora-canny) | Black Forest Labs | images |
| FLUX.1 [dev] Control LoRA Depth | [`fal-ai/flux-control-lora-depth`](https://fal.ai/models/fal-ai/flux-control-lora-depth) | Black Forest Labs | images |
| FLUX.1 [dev] Inpainting with LoRAs | [`fal-ai/flux-lora/inpainting`](https://fal.ai/models/fal-ai/flux-lora/inpainting) | Black Forest Labs | images |
| FLUX.1 [dev] with Controlnets and Loras | [`fal-ai/flux-general`](https://fal.ai/models/fal-ai/flux-general) | Black Forest Labs | images |
| FLUX.1 [dev] with LoRAs | [`fal-ai/flux-lora`](https://fal.ai/models/fal-ai/flux-lora) | Black Forest Labs | images |
| FLUX.1 [schnell] | [`fal-ai/flux-1/schnell`](https://fal.ai/models/fal-ai/flux-1/schnell) | Black Forest Labs | images |
| FLUX.1 [schnell] | [`fal-ai/flux/schnell`](https://fal.ai/models/fal-ai/flux/schnell) | Black Forest Labs | images |
| FLUX.1 Kontext [max] | [`fal-ai/flux-pro/kontext/max/text-to-image`](https://fal.ai/models/fal-ai/flux-pro/kontext/max/text-to-image) | Black Forest Labs | images |
| FLUX.1 Kontext [pro] | [`fal-ai/flux-pro/kontext/text-to-image`](https://fal.ai/models/fal-ai/flux-pro/kontext/text-to-image) | Black Forest Labs | images |
| FLUX.1 Krea [dev] | [`fal-ai/flux-1/krea`](https://fal.ai/models/fal-ai/flux-1/krea) | Black Forest Labs | images |
| FLUX.1 Krea [dev] | [`fal-ai/flux/krea`](https://fal.ai/models/fal-ai/flux/krea) | Black Forest Labs | images |
| FLUX.1 Krea [dev] with LoRAs | [`fal-ai/flux-krea-lora`](https://fal.ai/models/fal-ai/flux-krea-lora) | Black Forest Labs | images |
| FLUX.1 SRPO [dev] | [`fal-ai/flux-1/srpo`](https://fal.ai/models/fal-ai/flux-1/srpo) | Black Forest Labs | images |
| FLUX.1 SRPO [dev] | [`fal-ai/flux/srpo`](https://fal.ai/models/fal-ai/flux/srpo) | Black Forest Labs | images |
| FLUX.1 Subject | [`fal-ai/flux-subject`](https://fal.ai/models/fal-ai/flux-subject) | Black Forest Labs | images |
| FLUX.2 [klein] 4B | [`fal-ai/flux-2/klein/4b`](https://fal.ai/models/fal-ai/flux-2/klein/4b) | Black Forest Labs | images |
| FLUX.2 [klein] 4B Base | [`fal-ai/flux-2/klein/4b/base`](https://fal.ai/models/fal-ai/flux-2/klein/4b/base) | Black Forest Labs | images |
| FLUX.2 [klein] 4B Base LoRA | [`fal-ai/flux-2/klein/4b/base/lora`](https://fal.ai/models/fal-ai/flux-2/klein/4b/base/lora) | Black Forest Labs | images |
| FLUX.2 [klein] 4B LoRA | [`fal-ai/flux-2/klein/4b/lora`](https://fal.ai/models/fal-ai/flux-2/klein/4b/lora) | Black Forest Labs | images |
| FLUX.2 [klein] 9B | [`fal-ai/flux-2/klein/9b`](https://fal.ai/models/fal-ai/flux-2/klein/9b) | Black Forest Labs | images |
| FLUX.2 [klein] 9B Base | [`fal-ai/flux-2/klein/9b/base`](https://fal.ai/models/fal-ai/flux-2/klein/9b/base) | Black Forest Labs | images |
| FLUX.2 [klein] 9B Base LoRA | [`fal-ai/flux-2/klein/9b/base/lora`](https://fal.ai/models/fal-ai/flux-2/klein/9b/base/lora) | Black Forest Labs | images |
| FLUX.2 [klein] 9B LoRA | [`fal-ai/flux-2/klein/9b/lora`](https://fal.ai/models/fal-ai/flux-2/klein/9b/lora) | Black Forest Labs | images |
| FLUX1.1 [pro] | [`fal-ai/flux-pro/v1.1`](https://fal.ai/models/fal-ai/flux-pro/v1.1) | Black Forest Labs | images |
| FLUX1.1 [pro] ultra | [`fal-ai/flux-pro/v1.1-ultra`](https://fal.ai/models/fal-ai/flux-pro/v1.1-ultra) | Black Forest Labs | images |
| FLUX1.1 [pro] ultra Fine-tuned | [`fal-ai/flux-pro/v1.1-ultra-finetuned`](https://fal.ai/models/fal-ai/flux-pro/v1.1-ultra-finetuned) | Black Forest Labs | images |
| Fooocus | [`fal-ai/fast-fooocus-sdxl/image-to-image`](https://fal.ai/models/fal-ai/fast-fooocus-sdxl/image-to-image) | — | images |
| Fooocus | [`fal-ai/fooocus`](https://fal.ai/models/fal-ai/fooocus) | — | images |
| Fooocus Image Prompt | [`fal-ai/fooocus/image-prompt`](https://fal.ai/models/fal-ai/fooocus/image-prompt) | — | images |
| Fooocus Inpainting | [`fal-ai/fooocus/inpaint`](https://fal.ai/models/fal-ai/fooocus/inpaint) | — | images |
| Fooocus Upscale or Vary | [`fal-ai/fooocus/upscale-or-vary`](https://fal.ai/models/fal-ai/fooocus/upscale-or-vary) | — | images |
| Gemini 2.5 Flash Image | [`fal-ai/gemini-25-flash-image`](https://fal.ai/models/fal-ai/gemini-25-flash-image) | Google | images |
| Gemini 3 Pro Image Preview | [`fal-ai/gemini-3-pro-image-preview`](https://fal.ai/models/fal-ai/gemini-3-pro-image-preview) | Google | images |
| Gemini 3.1 Flash Image Preview | [`fal-ai/gemini-3.1-flash-image-preview`](https://fal.ai/models/fal-ai/gemini-3.1-flash-image-preview) | Google | images |
| GLM Image | [`fal-ai/glm-image`](https://fal.ai/models/fal-ai/glm-image) | — | images |
| GPT Image 1 Mini | [`fal-ai/gpt-image-1-mini`](https://fal.ai/models/fal-ai/gpt-image-1-mini) | OpenAI | images |
| GPT Image 2 API | [`openai/gpt-image-2`](https://fal.ai/models/openai/gpt-image-2) | OpenAI | images |
| GPT-Image 1.5 | [`fal-ai/gpt-image-1.5`](https://fal.ai/models/fal-ai/gpt-image-1.5) | OpenAI | images |
| gpt-image-1 | [`fal-ai/gpt-image-1/text-to-image`](https://fal.ai/models/fal-ai/gpt-image-1/text-to-image) | OpenAI | images |
| Grok Imagine Image | [`xai/grok-imagine-image`](https://fal.ai/models/xai/grok-imagine-image) | xAI | images |
| Grok Imagine Image | [`xai/grok-imagine-image/quality/text-to-image`](https://fal.ai/models/xai/grok-imagine-image/quality/text-to-image) | xAI | images |
| Hidream I1 Dev | [`fal-ai/hidream-i1-dev`](https://fal.ai/models/fal-ai/hidream-i1-dev) | Hidream | images |
| Hidream I1 Fast | [`fal-ai/hidream-i1-fast`](https://fal.ai/models/fal-ai/hidream-i1-fast) | Hidream | images |
| Hidream I1 Full | [`fal-ai/hidream-i1-full`](https://fal.ai/models/fal-ai/hidream-i1-full) | Hidream | images |
| Hidream O1 Image | [`fal-ai/hidream-o1-image`](https://fal.ai/models/fal-ai/hidream-o1-image) | Hidream | images |
| Hidream O1 Image | [`fal-ai/hidream-o1-image/dev`](https://fal.ai/models/fal-ai/hidream-o1-image/dev) | — | images |
| Hunyuan Image | [`fal-ai/hunyuan-image/v2.1/text-to-image`](https://fal.ai/models/fal-ai/hunyuan-image/v2.1/text-to-image) | Tencent | images |
| Hunyuan Image | [`fal-ai/hunyuan-image/v3/text-to-image`](https://fal.ai/models/fal-ai/hunyuan-image/v3/text-to-image) | Tencent | images |
| Hunyuan Image 3.0 Instruct | [`fal-ai/hunyuan-image/v3/instruct/text-to-image`](https://fal.ai/models/fal-ai/hunyuan-image/v3/instruct/text-to-image) | Tencent | images |
| Ideogram | [`fal-ai/ideogram/custom-models/generate`](https://fal.ai/models/fal-ai/ideogram/custom-models/generate) | Ideogram | images |
| Ideogram Text to Image | [`fal-ai/ideogram/v3`](https://fal.ai/models/fal-ai/ideogram/v3) | Ideogram | images |
| Ideogram Transparent | [`fal-ai/ideogram/v3/generate-transparent`](https://fal.ai/models/fal-ai/ideogram/v3/generate-transparent) | Ideogram | images |
| Ideogram V2 | [`fal-ai/ideogram/v2`](https://fal.ai/models/fal-ai/ideogram/v2) | Ideogram | images |
| Ideogram V2 Turbo | [`fal-ai/ideogram/v2/turbo`](https://fal.ai/models/fal-ai/ideogram/v2/turbo) | Ideogram | images |
| Ideogram V2A | [`fal-ai/ideogram/v2a`](https://fal.ai/models/fal-ai/ideogram/v2a) | Ideogram | images |
| Ideogram V2A Turbo | [`fal-ai/ideogram/v2a/turbo`](https://fal.ai/models/fal-ai/ideogram/v2a/turbo) | Ideogram | images |
| Ideogram V4.0 Text to Image | [`ideogram/v4`](https://fal.ai/models/ideogram/v4) | Ideogram | images |
| Ideogram V4.0q Text to Image (LoRA) | [`ideogram/v4/lora`](https://fal.ai/models/ideogram/v4/lora) | Ideogram | images |
| Illusion Diffusion | [`fal-ai/illusion-diffusion`](https://fal.ai/models/fal-ai/illusion-diffusion) | — | image |
| Imagineart 1.5 Preview | [`imagineart/imagineart-1.5-preview/text-to-image`](https://fal.ai/models/imagineart/imagineart-1.5-preview/text-to-image) | ImagineArt | images |
| ImagineArt 1.5 Pro Preview | [`imagineart/imagineart-1.5-pro-preview/text-to-image`](https://fal.ai/models/imagineart/imagineart-1.5-pro-preview/text-to-image) | ImagineArt | images |
| Imagineart 2.0 Preview | [`imagineart/imagineart-2.0-preview/text-to-image`](https://fal.ai/models/imagineart/imagineart-2.0-preview/text-to-image) | ImagineArt | images |
| Juggernaut Flux Base | [`rundiffusion-fal/juggernaut-flux/base`](https://fal.ai/models/rundiffusion-fal/juggernaut-flux/base) | Rundiffusion | images |
| Juggernaut Flux Base LoRA | [`rundiffusion-fal/juggernaut-flux-lora`](https://fal.ai/models/rundiffusion-fal/juggernaut-flux-lora) | Rundiffusion | images |
| Juggernaut Flux Lightning | [`rundiffusion-fal/juggernaut-flux/lightning`](https://fal.ai/models/rundiffusion-fal/juggernaut-flux/lightning) | Rundiffusion | images |
| Juggernaut Flux Pro | [`rundiffusion-fal/juggernaut-flux/pro`](https://fal.ai/models/rundiffusion-fal/juggernaut-flux/pro) | Rundiffusion | images |
| Kling Image | [`fal-ai/kling-image/o3/text-to-image`](https://fal.ai/models/fal-ai/kling-image/o3/text-to-image) | Kling | images |
| Kling Image | [`fal-ai/kling-image/v3/text-to-image`](https://fal.ai/models/fal-ai/kling-image/v3/text-to-image) | Kling | images |
| Kolors | [`fal-ai/kolors`](https://fal.ai/models/fal-ai/kolors) | — | images |
| Krea 2 Large | [`krea/v2/large/text-to-image`](https://fal.ai/models/krea/v2/large/text-to-image) | Krea | images |
| Krea 2 Medium | [`krea/v2/medium/text-to-image`](https://fal.ai/models/krea/v2/medium/text-to-image) | Krea | images |
| Krea 2 Medium Text to Image Turbo | [`krea/v2/medium/turbo/text-to-image`](https://fal.ai/models/krea/v2/medium/turbo/text-to-image) | Krea | images |
| Krea 2 Text to Image Turbo LoRA | [`fal-ai/krea-2/turbo/lora`](https://fal.ai/models/fal-ai/krea-2/turbo/lora) | Krea 2 | images |
| Krea 2 Turbo | [`fal-ai/krea-2/turbo`](https://fal.ai/models/fal-ai/krea-2/turbo) | Krea | images |
| Latent Consistency Models (v1.5/XL) | [`fal-ai/fast-lcm-diffusion`](https://fal.ai/models/fal-ai/fast-lcm-diffusion) | — | images |
| Longcat Image | [`fal-ai/longcat-image`](https://fal.ai/models/fal-ai/longcat-image) | — | images |
| Luma Photon | [`fal-ai/luma-photon`](https://fal.ai/models/fal-ai/luma-photon) | Luma AI | images |
| Luma Photon Flash | [`fal-ai/luma-photon/flash`](https://fal.ai/models/fal-ai/luma-photon/flash) | Luma AI | images |
| Luma Uni-1 Text to Image | [`luma/agent/uni-1/v1/text-to-image`](https://fal.ai/models/luma/agent/uni-1/v1/text-to-image) | Luma AI | images |
| Luma Uni-1 Text to Image Max | [`luma/agent/uni-1/v1/max`](https://fal.ai/models/luma/agent/uni-1/v1/max) | Luma AI | images |
| Lumina Image 2 | [`fal-ai/lumina-image/v2`](https://fal.ai/models/fal-ai/lumina-image/v2) | — | images |
| Mai Image 2.5 Text to Image | [`microsoft/mai-image-2.5`](https://fal.ai/models/microsoft/mai-image-2.5) | Microsoft | images |
| MiniMax (Hailuo AI) Text to Image | [`fal-ai/minimax/image-01`](https://fal.ai/models/fal-ai/minimax/image-01) | Minimax | images |
| Nano Banana | [`fal-ai/nano-banana`](https://fal.ai/models/fal-ai/nano-banana) | Google | images |
| Nano Banana 2 | [`fal-ai/nano-banana-2`](https://fal.ai/models/fal-ai/nano-banana-2) | Google | images |
| Nano Banana 2 Lite | [`google/nano-banana-2-lite`](https://fal.ai/models/google/nano-banana-2-lite) | Google | images |
| Nano Banana Lite | [`google/nano-banana-lite`](https://fal.ai/models/google/nano-banana-lite) | Google | images |
| Nano Banana Pro | [`fal-ai/nano-banana-pro`](https://fal.ai/models/fal-ai/nano-banana-pro) | Google | images |
| Nucleus Image | [`fal-ai/nucleus-image`](https://fal.ai/models/fal-ai/nucleus-image) | — | images |
| OmniGen v1 | [`fal-ai/omnigen-v1`](https://fal.ai/models/fal-ai/omnigen-v1) | — | images |
| Omnigen V2 | [`fal-ai/omnigen-v2`](https://fal.ai/models/fal-ai/omnigen-v2) | — | images |
| Ovis Image | [`fal-ai/ovis-image`](https://fal.ai/models/fal-ai/ovis-image) | — | images |
| PATINA | [`fal-ai/patina/material`](https://fal.ai/models/fal-ai/patina/material) | Fal | images |
| Phota Text to Image | [`fal-ai/phota`](https://fal.ai/models/fal-ai/phota) | — | images |
| PixArt-Σ | [`fal-ai/pixart-sigma`](https://fal.ai/models/fal-ai/pixart-sigma) | — | images |
| Playground v2.5 | [`fal-ai/playground-v25`](https://fal.ai/models/fal-ai/playground-v25) | — | images |
| Pony V7 | [`fal-ai/pony-v7`](https://fal.ai/models/fal-ai/pony-v7) | — | images |
| Qwen Image | [`fal-ai/qwen-image`](https://fal.ai/models/fal-ai/qwen-image) | Alibaba | images |
| Qwen Image 2 | [`fal-ai/qwen-image-2/pro/text-to-image`](https://fal.ai/models/fal-ai/qwen-image-2/pro/text-to-image) | Alibaba | images |
| Qwen Image 2 | [`fal-ai/qwen-image-2/text-to-image`](https://fal.ai/models/fal-ai/qwen-image-2/text-to-image) | Alibaba | images |
| Qwen Image 2512 | [`fal-ai/qwen-image-2512`](https://fal.ai/models/fal-ai/qwen-image-2512) | Alibaba | images |
| Qwen Image 2512 | [`fal-ai/qwen-image-2512/lora`](https://fal.ai/models/fal-ai/qwen-image-2512/lora) | Alibaba | images |
| Qwen Image Max | [`fal-ai/qwen-image-max/text-to-image`](https://fal.ai/models/fal-ai/qwen-image-max/text-to-image) | Alibaba | images |
| Realistic Vision | [`fal-ai/realistic-vision`](https://fal.ai/models/fal-ai/realistic-vision) | — | images |
| Recraft 20b | [`fal-ai/recraft-20b`](https://fal.ai/models/fal-ai/recraft-20b) | Recraft | images |
| Recraft V3 | [`fal-ai/recraft/v3/text-to-image`](https://fal.ai/models/fal-ai/recraft/v3/text-to-image) | Recraft | images |
| Recraft V4 | [`fal-ai/recraft/v4/text-to-image`](https://fal.ai/models/fal-ai/recraft/v4/text-to-image) | Recraft | images |
| Recraft V4 (Vector) | [`fal-ai/recraft/v4/text-to-vector`](https://fal.ai/models/fal-ai/recraft/v4/text-to-vector) | Recraft | images |
| Recraft V4 Pro | [`fal-ai/recraft/v4/pro/text-to-image`](https://fal.ai/models/fal-ai/recraft/v4/pro/text-to-image) | Recraft | images |
| Recraft V4 Pro (Vector) | [`fal-ai/recraft/v4/pro/text-to-vector`](https://fal.ai/models/fal-ai/recraft/v4/pro/text-to-vector) | Recraft | images |
| Recraft V4.1 Text to Image | [`fal-ai/recraft/v4.1/text-to-image`](https://fal.ai/models/fal-ai/recraft/v4.1/text-to-image) | Recraft | images |
| Recraft V4.1 Text to Image Pro | [`fal-ai/recraft/v4.1/pro/text-to-image`](https://fal.ai/models/fal-ai/recraft/v4.1/pro/text-to-image) | Recraft | images |
| Recraft V4.1 Text to Image Utility | [`fal-ai/recraft/v4.1/utility/text-to-image`](https://fal.ai/models/fal-ai/recraft/v4.1/utility/text-to-image) | Recraft | images |
| Recraft V4.1 Text to Vector | [`fal-ai/recraft/v4.1/text-to-vector`](https://fal.ai/models/fal-ai/recraft/v4.1/text-to-vector) | Recraft | images |
| Recraft V4.1 Text to Vector Pro | [`fal-ai/recraft/v4.1/pro/text-to-vector`](https://fal.ai/models/fal-ai/recraft/v4.1/pro/text-to-vector) | Recraft | images |
| Recraft V4.1 Utility Text to Image | [`fal-ai/recraft/v4.1/utility/pro/text-to-image`](https://fal.ai/models/fal-ai/recraft/v4.1/utility/pro/text-to-image) | Recraft | images |
| Rundiffusion Photo Flux | [`rundiffusion-fal/rundiffusion-photo-flux`](https://fal.ai/models/rundiffusion-fal/rundiffusion-photo-flux) | Rundiffusion | images |
| Sana | [`fal-ai/sana`](https://fal.ai/models/fal-ai/sana) | — | images |
| Sana Sprint | [`fal-ai/sana/sprint`](https://fal.ai/models/fal-ai/sana/sprint) | — | images |
| Sana v1.5 1.6B | [`fal-ai/sana/v1.5/1.6b`](https://fal.ai/models/fal-ai/sana/v1.5/1.6b) | — | images |
| Sana v1.5 4.8B | [`fal-ai/sana/v1.5/4.8b`](https://fal.ai/models/fal-ai/sana/v1.5/4.8b) | — | images |
| SDXL ControlNet Union | [`fal-ai/sdxl-controlnet-union`](https://fal.ai/models/fal-ai/sdxl-controlnet-union) | — | images |
| Sensenova U1 Infographic | [`fal-ai/sensenova-u1-infographic`](https://fal.ai/models/fal-ai/sensenova-u1-infographic) | — | images |
| SoteDiffusion | [`fal-ai/stable-cascade/sote-diffusion`](https://fal.ai/models/fal-ai/stable-cascade/sote-diffusion) | Stability AI | images |
| Stable Cascade | [`fal-ai/stable-cascade`](https://fal.ai/models/fal-ai/stable-cascade) | Stability AI | images |
| Stable Diffusion 3.5 Large | [`fal-ai/stable-diffusion-v35-large`](https://fal.ai/models/fal-ai/stable-diffusion-v35-large) | Stability AI | images |
| Stable Diffusion 3.5 Medium | [`fal-ai/stable-diffusion-v35-medium`](https://fal.ai/models/fal-ai/stable-diffusion-v35-medium) | Stability AI | images |
| Stable Diffusion v1.5 | [`fal-ai/stable-diffusion-v15`](https://fal.ai/models/fal-ai/stable-diffusion-v15) | Stability AI | images |
| Stable Diffusion V3 | [`fal-ai/stable-diffusion-v3-medium`](https://fal.ai/models/fal-ai/stable-diffusion-v3-medium) | Stability AI | images |
| Stable Diffusion with LoRAs | [`fal-ai/lora`](https://fal.ai/models/fal-ai/lora) | — | images |
| Stable Diffusion XL | [`fal-ai/fast-sdxl`](https://fal.ai/models/fal-ai/fast-sdxl) | — | images |
| Stable Diffusion XL Lightning | [`fal-ai/fast-lightning-sdxl`](https://fal.ai/models/fal-ai/fast-lightning-sdxl) | — | images |
| Vecglypher | [`fal-ai/vecglypher`](https://fal.ai/models/fal-ai/vecglypher) | — | image |
| Vidu | [`fal-ai/vidu/q2/text-to-image`](https://fal.ai/models/fal-ai/vidu/q2/text-to-image) | Vidu | image |
| Wan | [`fal-ai/wan/v2.2-5b/text-to-image`](https://fal.ai/models/fal-ai/wan/v2.2-5b/text-to-image) | Alibaba | image |
| Wan | [`fal-ai/wan/v2.2-a14b/text-to-image`](https://fal.ai/models/fal-ai/wan/v2.2-a14b/text-to-image) | Alibaba | image |
| Wan | [`fal-ai/wan/v2.7/pro/text-to-image`](https://fal.ai/models/fal-ai/wan/v2.7/pro/text-to-image) | Alibaba | images |
| Wan | [`fal-ai/wan/v2.7/text-to-image`](https://fal.ai/models/fal-ai/wan/v2.7/text-to-image) | Alibaba | images |
| Wan 2.5 Text to Image | [`fal-ai/wan-25-preview/text-to-image`](https://fal.ai/models/fal-ai/wan-25-preview/text-to-image) | Alibaba | images |
| Wan v2.2 A14B Text-to-Image A14B with LoRAs | [`fal-ai/wan/v2.2-a14b/text-to-image/lora`](https://fal.ai/models/fal-ai/wan/v2.2-a14b/text-to-image/lora) | Alibaba | image |
| Wan v2.6 Text to Image | [`wan/v2.6/text-to-image`](https://fal.ai/models/wan/v2.6/text-to-image) | Alibaba | images |
| Z Image Base | [`fal-ai/z-image/base`](https://fal.ai/models/fal-ai/z-image/base) | Alibaba | images |
| Z Image Base Lora | [`fal-ai/z-image/base/lora`](https://fal.ai/models/fal-ai/z-image/base/lora) | Alibaba | images |
| Z Image Turbo | [`fal-ai/z-image/turbo`](https://fal.ai/models/fal-ai/z-image/turbo) | Alibaba | images |
| Z Image Turbo Lora | [`fal-ai/z-image/turbo/lora`](https://fal.ai/models/fal-ai/z-image/turbo/lora) | Alibaba | images |
| Z-Image Turbo Seamless Tiling | [`fal-ai/z-image/turbo/tiling`](https://fal.ai/models/fal-ai/z-image/turbo/tiling) | Alibaba | images |
| Z-Image Turbo Seamless Tiling Lora | [`fal-ai/z-image/turbo/tiling/lora`](https://fal.ai/models/fal-ai/z-image/turbo/tiling/lora) | Alibaba | images |

</details>

<details>
<summary><strong>video-to-video</strong> — 181 models</summary>

| Model | Endpoint | Lab | Output |
| --- | --- | --- | --- |
| AMT Interpolation | [`fal-ai/amt-interpolation`](https://fal.ai/models/fal-ai/amt-interpolation) | — | video |
| AnimateDiff | [`fal-ai/fast-animatediff/video-to-video`](https://fal.ai/models/fal-ai/fast-animatediff/video-to-video) | — | video |
| AnimateDiff Turbo | [`fal-ai/fast-animatediff/turbo/video-to-video`](https://fal.ai/models/fal-ai/fast-animatediff/turbo/video-to-video) | — | video |
| Auto-Captioner | [`fal-ai/auto-caption`](https://fal.ai/models/fal-ai/auto-caption) | — | text |
| Ben-Video-Bg-Rm | [`fal-ai/ben/v2/video`](https://fal.ai/models/fal-ai/ben/v2/video) | — | video |
| Bernini-R Edit Video | [`fal-ai/bernini-r/edit-video`](https://fal.ai/models/fal-ai/bernini-r/edit-video) | Bytedance | video |
| Bernini-R Reference Edit Video | [`fal-ai/bernini-r/reference-edit-video`](https://fal.ai/models/fal-ai/bernini-r/reference-edit-video) | Bytedance | video |
| Birefnet | [`fal-ai/birefnet/v2/video`](https://fal.ai/models/fal-ai/birefnet/v2/video) | — | video |
| Bria Video Eraser | [`bria/bria_video_eraser/erase/keypoints`](https://fal.ai/models/bria/bria_video_eraser/erase/keypoints) | Bria AI | json |
| Bria Video Eraser | [`bria/bria_video_eraser/erase/prompt`](https://fal.ai/models/bria/bria_video_eraser/erase/prompt) | Bria AI | json |
| Bria Video Eraser Erase Mask | [`bria/bria_video_eraser/erase/mask`](https://fal.ai/models/bria/bria_video_eraser/erase/mask) | Bria AI | json |
| Bria's VRMBG 3.0 | [`bria/video/background-removal/v3`](https://fal.ai/models/bria/video/background-removal/v3) | — | video |
| Bria's VRMBG 3.0 Realtime | [`bria/video/background-removal/realtime`](https://fal.ai/models/bria/video/background-removal/realtime) | Bria AI | json |
| Bytedance Dreamactor V2 | [`fal-ai/bytedance/dreamactor/v2`](https://fal.ai/models/fal-ai/bytedance/dreamactor/v2) | Bytedance | video |
| Bytedance Upscaler Upscale Video | [`fal-ai/bytedance-upscaler/upscale/video`](https://fal.ai/models/fal-ai/bytedance-upscaler/upscale/video) | Bytedance | video |
| CogVideoX-5B | [`fal-ai/cogvideox-5b/video-to-video`](https://fal.ai/models/fal-ai/cogvideox-5b/video-to-video) | — | video |
| Controlfoley | [`fal-ai/controlfoley`](https://fal.ai/models/fal-ai/controlfoley) | — | video |
| Cosmos Predict 2.5 2B | [`fal-ai/cosmos-predict-2.5/video-to-video`](https://fal.ai/models/fal-ai/cosmos-predict-2.5/video-to-video) | — | video |
| Crystal Upscaler [Video] | [`clarityai/crystal-video-upscaler`](https://fal.ai/models/clarityai/crystal-video-upscaler) | Clarity AI | video |
| Depth Anything Video | [`fal-ai/depth-anything-video`](https://fal.ai/models/fal-ai/depth-anything-video) | — | video |
| DWPose Pose Prediction | [`fal-ai/dwpose/video`](https://fal.ai/models/fal-ai/dwpose/video) | — | video |
| Editto | [`fal-ai/editto`](https://fal.ai/models/fal-ai/editto) | — | video |
| Ffmpeg Api | [`fal-ai/ffmpeg-api/merge-videos`](https://fal.ai/models/fal-ai/ffmpeg-api/merge-videos) | — | video |
| FFmpeg API Compose | [`fal-ai/ffmpeg-api/compose`](https://fal.ai/models/fal-ai/ffmpeg-api/compose) | — | text |
| Ffmpeg Api Merge Audio-Video | [`fal-ai/ffmpeg-api/merge-audio-video`](https://fal.ai/models/fal-ai/ffmpeg-api/merge-audio-video) | — | video |
| FILM | [`fal-ai/film/video`](https://fal.ai/models/fal-ai/film/video) | — | video |
| Flashvsr | [`fal-ai/flashvsr/upscale/video`](https://fal.ai/models/fal-ai/flashvsr/upscale/video) | — | video |
| Gemini Omni Flash | [`google/gemini-omni-flash/edit`](https://fal.ai/models/google/gemini-omni-flash/edit) | Google | video |
| Grok Imagine Extend Video | [`xai/grok-imagine-video/extend-video`](https://fal.ai/models/xai/grok-imagine-video/extend-video) | xAI | video |
| Grok Imagine Video | [`xai/grok-imagine-video/edit-video`](https://fal.ai/models/xai/grok-imagine-video/edit-video) | xAI | video |
| Happy Horse Video Edit | [`alibaba/happy-horse/video-edit`](https://fal.ai/models/alibaba/happy-horse/video-edit) | Alibaba | video |
| Heygen | [`fal-ai/heygen/v2/translate/precision`](https://fal.ai/models/fal-ai/heygen/v2/translate/precision) | Heygen | video |
| Heygen | [`fal-ai/heygen/v2/translate/speed`](https://fal.ai/models/fal-ai/heygen/v2/translate/speed) | Heygen | video |
| Heygen Lipsync - Precision | [`fal-ai/heygen/v3/lipsync/precision`](https://fal.ai/models/fal-ai/heygen/v3/lipsync/precision) | Heygen | video |
| Heygen Lipsync - Speed | [`fal-ai/heygen/v3/lipsync/speed`](https://fal.ai/models/fal-ai/heygen/v3/lipsync/speed) | Heygen | video |
| Hunyuan Video (Video-to-Video) | [`fal-ai/hunyuan-video/video-to-video`](https://fal.ai/models/fal-ai/hunyuan-video/video-to-video) | Tencent | video |
| Hunyuan Video Foley | [`fal-ai/hunyuan-video-foley`](https://fal.ai/models/fal-ai/hunyuan-video-foley) | Tencent | video |
| Infinitalk | [`fal-ai/infinitalk`](https://fal.ai/models/fal-ai/infinitalk) | — | video |
| Infinitalk | [`fal-ai/infinitalk/video-to-video`](https://fal.ai/models/fal-ai/infinitalk/video-to-video) | — | video |
| Kling O1 Edit Video [Pro] | [`fal-ai/kling-video/o1/video-to-video/edit`](https://fal.ai/models/fal-ai/kling-video/o1/video-to-video/edit) | Kling | video |
| Kling O1 Edit Video [Standard] | [`fal-ai/kling-video/o1/standard/video-to-video/edit`](https://fal.ai/models/fal-ai/kling-video/o1/standard/video-to-video/edit) | Kling | video |
| Kling O1 Reference Video to Video [Pro] | [`fal-ai/kling-video/o1/video-to-video/reference`](https://fal.ai/models/fal-ai/kling-video/o1/video-to-video/reference) | Kling | video |
| Kling O1 Reference Video to Video [Standard] | [`fal-ai/kling-video/o1/standard/video-to-video/reference`](https://fal.ai/models/fal-ai/kling-video/o1/standard/video-to-video/reference) | Kling | video |
| Kling O3 Edit Video [Pro] | [`fal-ai/kling-video/o3/pro/video-to-video/edit`](https://fal.ai/models/fal-ai/kling-video/o3/pro/video-to-video/edit) | Kling | video |
| Kling O3 Edit Video [Standard] | [`fal-ai/kling-video/o3/standard/video-to-video/edit`](https://fal.ai/models/fal-ai/kling-video/o3/standard/video-to-video/edit) | Kling | video |
| Kling O3 Reference Video to Video [Pro] | [`fal-ai/kling-video/o3/pro/video-to-video/reference`](https://fal.ai/models/fal-ai/kling-video/o3/pro/video-to-video/reference) | Kling | video |
| Kling O3 Reference Video to Video [Standard] | [`fal-ai/kling-video/o3/standard/video-to-video/reference`](https://fal.ai/models/fal-ai/kling-video/o3/standard/video-to-video/reference) | Kling | video |
| Kling Video | [`fal-ai/kling-video/v3/pro/motion-control`](https://fal.ai/models/fal-ai/kling-video/v3/pro/motion-control) | Kling | video |
| Kling Video | [`fal-ai/kling-video/v3/standard/motion-control`](https://fal.ai/models/fal-ai/kling-video/v3/standard/motion-control) | Kling | video |
| Kling Video v2.6 Motion Control [Pro] | [`fal-ai/kling-video/v2.6/pro/motion-control`](https://fal.ai/models/fal-ai/kling-video/v2.6/pro/motion-control) | Kling | video |
| Kling Video v2.6 Motion Control [Standard] | [`fal-ai/kling-video/v2.6/standard/motion-control`](https://fal.ai/models/fal-ai/kling-video/v2.6/standard/motion-control) | Kling | video |
| Krea Wan 14B | [`fal-ai/krea-wan-14b/video-to-video`](https://fal.ai/models/fal-ai/krea-wan-14b/video-to-video) | — | video |
| LatentSync | [`fal-ai/latentsync`](https://fal.ai/models/fal-ai/latentsync) | — | video |
| Lightx | [`fal-ai/lightx/recamera`](https://fal.ai/models/fal-ai/lightx/recamera) | — | video |
| Lightx | [`fal-ai/lightx/relight`](https://fal.ai/models/fal-ai/lightx/relight) | — | video |
| Lipsync | [`veed/lipsync`](https://fal.ai/models/veed/lipsync) | Veed | video |
| LTX 2.3 22B | [`fal-ai/ltx-2.3-22b/reference-video-to-video/lora`](https://fal.ai/models/fal-ai/ltx-2.3-22b/reference-video-to-video/lora) | Lightricks | video |
| Ltx 2.3 Quality | [`fal-ai/ltx-2.3-quality/colorization`](https://fal.ai/models/fal-ai/ltx-2.3-quality/colorization) | Lightricks | video |
| Ltx 2.3 Quality | [`fal-ai/ltx-2.3-quality/cross-eyed`](https://fal.ai/models/fal-ai/ltx-2.3-quality/cross-eyed) | Lightricks | video |
| Ltx 2.3 Quality | [`fal-ai/ltx-2.3-quality/day-to-night`](https://fal.ai/models/fal-ai/ltx-2.3-quality/day-to-night) | Lightricks | video |
| Ltx 2.3 Quality | [`fal-ai/ltx-2.3-quality/deblur`](https://fal.ai/models/fal-ai/ltx-2.3-quality/deblur) | Lightricks | video |
| Ltx 2.3 Quality | [`fal-ai/ltx-2.3-quality/decompression`](https://fal.ai/models/fal-ai/ltx-2.3-quality/decompression) | Lightricks | video |
| Ltx 2.3 Quality | [`fal-ai/ltx-2.3-quality/hdr`](https://fal.ai/models/fal-ai/ltx-2.3-quality/hdr) | Lightricks | json |
| Ltx 2.3 Quality | [`fal-ai/ltx-2.3-quality/hdr/lora`](https://fal.ai/models/fal-ai/ltx-2.3-quality/hdr/lora) | Lightricks | json |
| Ltx 2.3 Quality | [`fal-ai/ltx-2.3-quality/inpaint`](https://fal.ai/models/fal-ai/ltx-2.3-quality/inpaint) | — | video |
| Ltx 2.3 Quality | [`fal-ai/ltx-2.3-quality/inpaint/lora`](https://fal.ai/models/fal-ai/ltx-2.3-quality/inpaint/lora) | — | video |
| Ltx 2.3 Quality | [`fal-ai/ltx-2.3-quality/instant-shave`](https://fal.ai/models/fal-ai/ltx-2.3-quality/instant-shave) | Lightricks | video |
| Ltx 2.3 Quality | [`fal-ai/ltx-2.3-quality/outpaint`](https://fal.ai/models/fal-ai/ltx-2.3-quality/outpaint) | Lightricks | video |
| Ltx 2.3 Quality | [`fal-ai/ltx-2.3-quality/outpaint/lora`](https://fal.ai/models/fal-ai/ltx-2.3-quality/outpaint/lora) | Lightricks | video |
| Ltx 2.3 Quality | [`fal-ai/ltx-2.3-quality/reference-video-to-video`](https://fal.ai/models/fal-ai/ltx-2.3-quality/reference-video-to-video) | Lightricks | video |
| Ltx 2.3 Quality | [`fal-ai/ltx-2.3-quality/reference-video-to-video/lora`](https://fal.ai/models/fal-ai/ltx-2.3-quality/reference-video-to-video/lora) | Lightricks | video |
| Ltx 2.3 Quality | [`fal-ai/ltx-2.3-quality/render-to-real`](https://fal.ai/models/fal-ai/ltx-2.3-quality/render-to-real) | Lightricks | video |
| Ltx 2.3 Quality | [`fal-ai/ltx-2.3-quality/water-simulation`](https://fal.ai/models/fal-ai/ltx-2.3-quality/water-simulation) | Lightricks | video |
| LTX Video 2.0 Pro | [`fal-ai/ltx-2/extend-video`](https://fal.ai/models/fal-ai/ltx-2/extend-video) | Lightricks | video |
| LTX Video 2.0 Retake | [`fal-ai/ltx-2/retake-video`](https://fal.ai/models/fal-ai/ltx-2/retake-video) | Lightricks | video |
| LTX Video 2.3 Pro | [`fal-ai/ltx-2.3/extend-video`](https://fal.ai/models/fal-ai/ltx-2.3/extend-video) | Lightricks | video |
| LTX Video 2.3 Pro | [`fal-ai/ltx-2.3/retake-video`](https://fal.ai/models/fal-ai/ltx-2.3/retake-video) | Lightricks | video |
| LTX Video-0.9.5 | [`fal-ai/ltx-video-v095/extend`](https://fal.ai/models/fal-ai/ltx-video-v095/extend) | Lightricks | video |
| LTX Video-0.9.5 | [`fal-ai/ltx-video-v095/multiconditioning`](https://fal.ai/models/fal-ai/ltx-video-v095/multiconditioning) | Lightricks | video |
| LTX Video-0.9.7 13B Distilled | [`fal-ai/ltx-video-13b-distilled/extend`](https://fal.ai/models/fal-ai/ltx-video-13b-distilled/extend) | Lightricks | video |
| LTX Video-0.9.7 13B Distilled | [`fal-ai/ltx-video-13b-distilled/multiconditioning`](https://fal.ai/models/fal-ai/ltx-video-13b-distilled/multiconditioning) | Lightricks | video |
| LTX-2 19B | [`fal-ai/ltx-2-19b/extend-video`](https://fal.ai/models/fal-ai/ltx-2-19b/extend-video) | Lightricks | video |
| LTX-2 19B | [`fal-ai/ltx-2-19b/extend-video/lora`](https://fal.ai/models/fal-ai/ltx-2-19b/extend-video/lora) | Lightricks | video |
| LTX-2 19B | [`fal-ai/ltx-2-19b/video-to-video`](https://fal.ai/models/fal-ai/ltx-2-19b/video-to-video) | Lightricks | video |
| LTX-2 19B | [`fal-ai/ltx-2-19b/video-to-video/lora`](https://fal.ai/models/fal-ai/ltx-2-19b/video-to-video/lora) | Lightricks | video |
| LTX-2 19B Distilled | [`fal-ai/ltx-2-19b/distilled/extend-video`](https://fal.ai/models/fal-ai/ltx-2-19b/distilled/extend-video) | Lightricks | video |
| LTX-2 19B Distilled | [`fal-ai/ltx-2-19b/distilled/extend-video/lora`](https://fal.ai/models/fal-ai/ltx-2-19b/distilled/extend-video/lora) | Lightricks | video |
| LTX-2 19B Distilled | [`fal-ai/ltx-2-19b/distilled/video-to-video`](https://fal.ai/models/fal-ai/ltx-2-19b/distilled/video-to-video) | Lightricks | video |
| LTX-2 19B Distilled | [`fal-ai/ltx-2-19b/distilled/video-to-video/lora`](https://fal.ai/models/fal-ai/ltx-2-19b/distilled/video-to-video/lora) | Lightricks | video |
| LTX-2.3 22B | [`fal-ai/ltx-2.3-22b/extend-video`](https://fal.ai/models/fal-ai/ltx-2.3-22b/extend-video) | Lightricks | video |
| LTX-2.3 22B | [`fal-ai/ltx-2.3-22b/extend-video/lora`](https://fal.ai/models/fal-ai/ltx-2.3-22b/extend-video/lora) | Lightricks | video |
| LTX-2.3 22B | [`fal-ai/ltx-2.3-22b/reference-video-to-video`](https://fal.ai/models/fal-ai/ltx-2.3-22b/reference-video-to-video) | Lightricks | video |
| LTX-2.3 22B | [`fal-ai/ltx-2.3-22b/video-to-video`](https://fal.ai/models/fal-ai/ltx-2.3-22b/video-to-video) | Lightricks | video |
| LTX-2.3 22B | [`fal-ai/ltx-2.3-22b/video-to-video/lora`](https://fal.ai/models/fal-ai/ltx-2.3-22b/video-to-video/lora) | Lightricks | video |
| LTX-2.3 22B Distilled | [`fal-ai/ltx-2.3-22b/distilled/reference-video-to-video`](https://fal.ai/models/fal-ai/ltx-2.3-22b/distilled/reference-video-to-video) | Lightricks | video |
| LTX-2.3 22B Distilled | [`fal-ai/ltx-2.3-22b/distilled/reference-video-to-video/lora`](https://fal.ai/models/fal-ai/ltx-2.3-22b/distilled/reference-video-to-video/lora) | Lightricks | video |
| LTX-2.3 22B Distilled | [`fal-ai/ltx-2.3-22b/distilled/video-to-video`](https://fal.ai/models/fal-ai/ltx-2.3-22b/distilled/video-to-video) | Lightricks | video |
| LTX-2.3 22B Distilled | [`fal-ai/ltx-2.3-22b/distilled/video-to-video/lora`](https://fal.ai/models/fal-ai/ltx-2.3-22b/distilled/video-to-video/lora) | Lightricks | video |
| LTX-Video 13B 0.9.8 Distilled | [`fal-ai/ltxv-13b-098-distilled/extend`](https://fal.ai/models/fal-ai/ltxv-13b-098-distilled/extend) | Lightricks | video |
| LTX-Video 13B 0.9.8 Distilled | [`fal-ai/ltxv-13b-098-distilled/multiconditioning`](https://fal.ai/models/fal-ai/ltxv-13b-098-distilled/multiconditioning) | Lightricks | video |
| Lucy 2.1 VTON Realtime | [`decart/lucy2-vton/realtime`](https://fal.ai/models/decart/lucy2-vton/realtime) | Decart | json |
| Lucy Edit [Pro] | [`decart/lucy-edit/pro`](https://fal.ai/models/decart/lucy-edit/pro) | Decart | video |
| Lucy Restyle | [`decart/lucy-restyle`](https://fal.ai/models/decart/lucy-restyle) | Decart | video |
| Luma Ray 2 Flash Modify | [`fal-ai/luma-dream-machine/ray-2-flash/modify`](https://fal.ai/models/fal-ai/luma-dream-machine/ray-2-flash/modify) | Luma AI | video |
| Luma Ray 2 Flash Reframe | [`fal-ai/luma-dream-machine/ray-2-flash/reframe`](https://fal.ai/models/fal-ai/luma-dream-machine/ray-2-flash/reframe) | Luma AI | video |
| Luma Ray 2 Modify | [`fal-ai/luma-dream-machine/ray-2/modify`](https://fal.ai/models/fal-ai/luma-dream-machine/ray-2/modify) | Luma AI | video |
| Luma Ray 2 Reframe | [`fal-ai/luma-dream-machine/ray-2/reframe`](https://fal.ai/models/fal-ai/luma-dream-machine/ray-2/reframe) | Luma AI | video |
| Luma Ray 3.2 Reframe | [`luma/agent/ray/v3.2/reframe`](https://fal.ai/models/luma/agent/ray/v3.2/reframe) | Luma AI | video |
| Luma Ray 3.2 Video to Video | [`luma/agent/ray/v3.2/video-to-video`](https://fal.ai/models/luma/agent/ray/v3.2/video-to-video) | Luma AI | video |
| MAGI-1 (Distilled) | [`fal-ai/magi-distilled/extend-video`](https://fal.ai/models/fal-ai/magi-distilled/extend-video) | — | video |
| Marey Realism V1.5 | [`moonvalley/marey/motion-transfer`](https://fal.ai/models/moonvalley/marey/motion-transfer) | Moonvalley | video |
| Marey Realism V1.5 | [`moonvalley/marey/pose-transfer`](https://fal.ai/models/moonvalley/marey/pose-transfer) | Moonvalley | video |
| Mirelo SFX | [`mirelo-ai/sfx-v1/video-to-video`](https://fal.ai/models/mirelo-ai/sfx-v1/video-to-video) | Mirelo AI | video |
| Mirelo SFX V1.5 | [`mirelo-ai/sfx-v1.5/video-to-video`](https://fal.ai/models/mirelo-ai/sfx-v1.5/video-to-video) | Mirelo AI | video |
| Mirelo SFX1.6 | [`mirelo-ai/sfx1.6/video-to-video`](https://fal.ai/models/mirelo-ai/sfx1.6/video-to-video) | Mirelo AI | video |
| MMAudio V2 | [`fal-ai/mmaudio-v2`](https://fal.ai/models/fal-ai/mmaudio-v2) | — | video |
| One To All Animation | [`fal-ai/one-to-all-animation/1.3b`](https://fal.ai/models/fal-ai/one-to-all-animation/1.3b) | — | video |
| One To All Animation | [`fal-ai/one-to-all-animation/14b`](https://fal.ai/models/fal-ai/one-to-all-animation/14b) | — | video |
| Pikadditions (v2) | [`fal-ai/pika/v2/pikadditions`](https://fal.ai/models/fal-ai/pika/v2/pikadditions) | Pika | video |
| Pixelcut Video Background Removal | [`pixelcut/video-background-removal`](https://fal.ai/models/pixelcut/video-background-removal) | Pixelcut | video |
| PixVerse Extend | [`fal-ai/pixverse/extend`](https://fal.ai/models/fal-ai/pixverse/extend) | Pixverse | video |
| PixVerse Extend Fast | [`fal-ai/pixverse/extend/fast`](https://fal.ai/models/fal-ai/pixverse/extend/fast) | Pixverse | video |
| PixVerse Lipsync | [`fal-ai/pixverse/lipsync`](https://fal.ai/models/fal-ai/pixverse/lipsync) | Pixverse | video |
| PixVerse Sound Effects | [`fal-ai/pixverse/sound-effects`](https://fal.ai/models/fal-ai/pixverse/sound-effects) | Pixverse | video |
| PixVerse V6 Extend | [`fal-ai/pixverse/v6/extend`](https://fal.ai/models/fal-ai/pixverse/v6/extend) | Pixverse | video |
| RIFE | [`fal-ai/rife/video`](https://fal.ai/models/fal-ai/rife/video) | — | video |
| Sam 3 | [`fal-ai/sam-3/video`](https://fal.ai/models/fal-ai/sam-3/video) | — | video |
| Sam 3 | [`fal-ai/sam-3/video-rle`](https://fal.ai/models/fal-ai/sam-3/video-rle) | — | video |
| Sam 3 1 | [`fal-ai/sam-3-1/video`](https://fal.ai/models/fal-ai/sam-3-1/video) | — | video |
| Sam 3 1 | [`fal-ai/sam-3-1/video-rle`](https://fal.ai/models/fal-ai/sam-3-1/video-rle) | — | video |
| Scail 2 | [`fal-ai/scail-2`](https://fal.ai/models/fal-ai/scail-2) | z AI | video |
| SeedVR2 | [`fal-ai/seedvr/upscale/video`](https://fal.ai/models/fal-ai/seedvr/upscale/video) | SeedVR | video |
| Segment Anything Model 2 | [`fal-ai/sam2/video`](https://fal.ai/models/fal-ai/sam2/video) | — | video |
| Sora 2 | [`fal-ai/sora-2/video-to-video/remix`](https://fal.ai/models/fal-ai/sora-2/video-to-video/remix) | OpenAI | video |
| Subtitles | [`veed/subtitles`](https://fal.ai/models/veed/subtitles) | Veed | video |
| Sync Lipsync | [`fal-ai/sync-lipsync/v2/pro`](https://fal.ai/models/fal-ai/sync-lipsync/v2/pro) | Syncso | video |
| Sync Lipsync 2.0 | [`fal-ai/sync-lipsync/v2`](https://fal.ai/models/fal-ai/sync-lipsync/v2) | Syncso | video |
| Sync React-1 | [`fal-ai/sync-lipsync/react-1`](https://fal.ai/models/fal-ai/sync-lipsync/react-1) | Syncso | video |
| sync-3 Lipsync | [`fal-ai/sync-lipsync/v3`](https://fal.ai/models/fal-ai/sync-lipsync/v3) | Syncso | video |
| sync.so -- lipsync 1.9.0-beta | [`fal-ai/sync-lipsync`](https://fal.ai/models/fal-ai/sync-lipsync) | Syncso | video |
| ThinkSound | [`fal-ai/thinksound`](https://fal.ai/models/fal-ai/thinksound) | — | video |
| ThinkSound | [`fal-ai/thinksound/audio`](https://fal.ai/models/fal-ai/thinksound/audio) | — | audio |
| Topaz Video Upscale | [`fal-ai/topaz/upscale/video`](https://fal.ai/models/fal-ai/topaz/upscale/video) | Topaz Labs | video |
| V2.6 | [`wan/v2.6/reference-to-video/flash`](https://fal.ai/models/wan/v2.6/reference-to-video/flash) | Alibaba | video |
| Veo 3.1 | [`fal-ai/veo3.1/extend-video`](https://fal.ai/models/fal-ai/veo3.1/extend-video) | Google | video |
| Veo 3.1 Fast | [`fal-ai/veo3.1/fast/extend-video`](https://fal.ai/models/fal-ai/veo3.1/fast/extend-video) | Google | video |
| Video | [`bria/video/background-removal`](https://fal.ai/models/bria/video/background-removal) | Bria AI | video |
| Video | [`bria/video/erase/keypoints`](https://fal.ai/models/bria/video/erase/keypoints) | Bria AI | json |
| Video | [`bria/video/erase/mask`](https://fal.ai/models/bria/video/erase/mask) | Bria AI | json |
| Video | [`bria/video/erase/prompt`](https://fal.ai/models/bria/video/erase/prompt) | Bria AI | json |
| Video | [`bria/video/increase-resolution`](https://fal.ai/models/bria/video/increase-resolution) | Bria AI | video |
| Video Background Removal | [`veed/video-background-removal`](https://fal.ai/models/veed/video-background-removal) | Veed | video |
| Video Background Removal | [`veed/video-background-removal/fast`](https://fal.ai/models/veed/video-background-removal/fast) | Veed | video |
| Video Background Removal | [`veed/video-background-removal/green-screen`](https://fal.ai/models/veed/video-background-removal/green-screen) | Veed | video |
| Video Sound Effects Generator | [`cassetteai/video-sound-effects-generator`](https://fal.ai/models/cassetteai/video-sound-effects-generator) | Cassette AI | video |
| Video Upscaler | [`fal-ai/video-upscaler`](https://fal.ai/models/fal-ai/video-upscaler) | — | video |
| Vidu | [`fal-ai/vidu/q2/video-extension/pro`](https://fal.ai/models/fal-ai/vidu/q2/video-extension/pro) | Vidu | video |
| Void Video Inpainting | [`fal-ai/void-video-inpainting`](https://fal.ai/models/fal-ai/void-video-inpainting) | — | video |
| Wan | [`fal-ai/wan/v2.2-a14b/video-to-video`](https://fal.ai/models/fal-ai/wan/v2.2-a14b/video-to-video) | Alibaba | video |
| Wan | [`fal-ai/wan/v2.7/edit-video`](https://fal.ai/models/fal-ai/wan/v2.7/edit-video) | Alibaba | video |
| Wan 2.1 VACE Long Reframe | [`fal-ai/wan-vace-apps/long-reframe`](https://fal.ai/models/fal-ai/wan-vace-apps/long-reframe) | Alibaba | video |
| Wan 2.2 VACE Fun A14B | [`fal-ai/wan-22-vace-fun-a14b/depth`](https://fal.ai/models/fal-ai/wan-22-vace-fun-a14b/depth) | Alibaba | video |
| Wan 2.2 VACE Fun A14B | [`fal-ai/wan-22-vace-fun-a14b/inpainting`](https://fal.ai/models/fal-ai/wan-22-vace-fun-a14b/inpainting) | Alibaba | video |
| Wan 2.2 VACE Fun A14B | [`fal-ai/wan-22-vace-fun-a14b/outpainting`](https://fal.ai/models/fal-ai/wan-22-vace-fun-a14b/outpainting) | Alibaba | video |
| Wan 2.2 VACE Fun A14B | [`fal-ai/wan-22-vace-fun-a14b/reframe`](https://fal.ai/models/fal-ai/wan-22-vace-fun-a14b/reframe) | Alibaba | video |
| Wan Motion | [`fal-ai/wan-motion`](https://fal.ai/models/fal-ai/wan-motion) | — | video |
| Wan v2.6 Reference to Video | [`wan/v2.6/reference-to-video`](https://fal.ai/models/wan/v2.6/reference-to-video) | Alibaba | video |
| Wan VACE 14B | [`fal-ai/wan-vace-14b`](https://fal.ai/models/fal-ai/wan-vace-14b) | Alibaba | video |
| Wan VACE 14B | [`fal-ai/wan-vace-14b/depth`](https://fal.ai/models/fal-ai/wan-vace-14b/depth) | Alibaba | video |
| Wan VACE 14B | [`fal-ai/wan-vace-14b/inpainting`](https://fal.ai/models/fal-ai/wan-vace-14b/inpainting) | Alibaba | video |
| Wan VACE 14B | [`fal-ai/wan-vace-14b/outpainting`](https://fal.ai/models/fal-ai/wan-vace-14b/outpainting) | Alibaba | video |
| Wan VACE 14B | [`fal-ai/wan-vace-14b/pose`](https://fal.ai/models/fal-ai/wan-vace-14b/pose) | Alibaba | video |
| Wan VACE 14B | [`fal-ai/wan-vace-14b/reframe`](https://fal.ai/models/fal-ai/wan-vace-14b/reframe) | Alibaba | video |
| Wan VACE Video Edit | [`fal-ai/wan-vace-apps/video-edit`](https://fal.ai/models/fal-ai/wan-vace-apps/video-edit) | Alibaba | video |
| Wan-2.2 Animate Move | [`fal-ai/wan/v2.2-14b/animate/move`](https://fal.ai/models/fal-ai/wan/v2.2-14b/animate/move) | Alibaba | video |
| Wan-2.2 Animate Replace | [`fal-ai/wan/v2.2-14b/animate/replace`](https://fal.ai/models/fal-ai/wan/v2.2-14b/animate/replace) | Alibaba | video |
| Workflow Utilities Auto Subtitle | [`fal-ai/workflow-utilities/auto-subtitle`](https://fal.ai/models/fal-ai/workflow-utilities/auto-subtitle) | — | video |
| Workflow Utilities Blend Video | [`fal-ai/workflow-utilities/blend-video`](https://fal.ai/models/fal-ai/workflow-utilities/blend-video) | — | video |
| Workflow Utilities Reverse Video | [`fal-ai/workflow-utilities/reverse-video`](https://fal.ai/models/fal-ai/workflow-utilities/reverse-video) | — | video |
| Workflow Utilities Scale Video | [`fal-ai/workflow-utilities/scale-video`](https://fal.ai/models/fal-ai/workflow-utilities/scale-video) | — | video |
| Workflow Utilities Trim Video | [`fal-ai/workflow-utilities/trim-video`](https://fal.ai/models/fal-ai/workflow-utilities/trim-video) | — | video |

</details>

<details>
<summary><strong>text-to-video</strong> — 130 models</summary>

| Model | Endpoint | Lab | Output |
| --- | --- | --- | --- |
| AnimateDiff | [`fal-ai/fast-animatediff/text-to-video`](https://fal.ai/models/fal-ai/fast-animatediff/text-to-video) | — | video |
| AnimateDiff Turbo | [`fal-ai/fast-animatediff/turbo/text-to-video`](https://fal.ai/models/fal-ai/fast-animatediff/turbo/text-to-video) | — | video |
| Avatars | [`veed/avatars/text-to-video`](https://fal.ai/models/veed/avatars/text-to-video) | Veed | video |
| Avatars Text to Video | [`argil/avatars/text-to-video`](https://fal.ai/models/argil/avatars/text-to-video) | Argil AI | video |
| Bernini-R Text to Video | [`fal-ai/bernini-r/text-to-video`](https://fal.ai/models/fal-ai/bernini-r/text-to-video) | Bytedance | video |
| Bytedance Seedance V1 Pro Fast Text To Video | [`fal-ai/bytedance/seedance/v1/pro/fast/text-to-video`](https://fal.ai/models/fal-ai/bytedance/seedance/v1/pro/fast/text-to-video) | Bytedance | video |
| Bytedance Seedance V1.5 Pro Text To Video | [`fal-ai/bytedance/seedance/v1.5/pro/text-to-video`](https://fal.ai/models/fal-ai/bytedance/seedance/v1.5/pro/text-to-video) | Bytedance | video |
| CogVideoX-5B | [`fal-ai/cogvideox-5b`](https://fal.ai/models/fal-ai/cogvideox-5b) | — | video |
| Cosmos Predict 2.5 2B | [`fal-ai/cosmos-predict-2.5/text-to-video`](https://fal.ai/models/fal-ai/cosmos-predict-2.5/text-to-video) | NVIDIA | video |
| Cosmos Predict 2.5 2B Distilled | [`fal-ai/cosmos-predict-2.5/distilled/text-to-video`](https://fal.ai/models/fal-ai/cosmos-predict-2.5/distilled/text-to-video) | NVIDIA | video |
| Fabric 1.0 | [`veed/fabric-1.0/text`](https://fal.ai/models/veed/fabric-1.0/text) | Veed | video |
| Gemini Omni Flash | [`google/gemini-omni-flash`](https://fal.ai/models/google/gemini-omni-flash) | Google | video |
| Grok Imagine Video | [`xai/grok-imagine-video/text-to-video`](https://fal.ai/models/xai/grok-imagine-video/text-to-video) | xAI | video |
| Happy Horse | [`alibaba/happy-horse/text-to-video`](https://fal.ai/models/alibaba/happy-horse/text-to-video) | Alibaba | video |
| Happy Horse 1.1 Text to Video | [`alibaba/happy-horse/v1.1/text-to-video`](https://fal.ai/models/alibaba/happy-horse/v1.1/text-to-video) | Alibaba | video |
| Heygen | [`fal-ai/heygen/avatar3/digital-twin`](https://fal.ai/models/fal-ai/heygen/avatar3/digital-twin) | Heygen | video |
| Heygen | [`fal-ai/heygen/avatar4/digital-twin`](https://fal.ai/models/fal-ai/heygen/avatar4/digital-twin) | Heygen | video |
| Heygen | [`fal-ai/heygen/v2/video-agent`](https://fal.ai/models/fal-ai/heygen/v2/video-agent) | Heygen | video |
| Heygen v5 Digital Twin | [`fal-ai/heygen/avatar5/digital-twin`](https://fal.ai/models/fal-ai/heygen/avatar5/digital-twin) | Heygen | video |
| Heygen Video Agent | [`fal-ai/heygen/v3/video-agent`](https://fal.ai/models/fal-ai/heygen/v3/video-agent) | Heygen | video |
| Hunyuan Video | [`fal-ai/hunyuan-video`](https://fal.ai/models/fal-ai/hunyuan-video) | Tencent | video |
| Hunyuan Video V1.5 | [`fal-ai/hunyuan-video-v1.5/text-to-video`](https://fal.ai/models/fal-ai/hunyuan-video-v1.5/text-to-video) | Tencent | video |
| Infinitalk | [`fal-ai/infinitalk/single-text`](https://fal.ai/models/fal-ai/infinitalk/single-text) | — | video |
| Infinity Star | [`fal-ai/infinity-star/text-to-video`](https://fal.ai/models/fal-ai/infinity-star/text-to-video) | — | video |
| Kandinsky5 | [`fal-ai/kandinsky5/text-to-video`](https://fal.ai/models/fal-ai/kandinsky5/text-to-video) | Kandinsky | video |
| Kandinsky5 | [`fal-ai/kandinsky5/text-to-video/distill`](https://fal.ai/models/fal-ai/kandinsky5/text-to-video/distill) | Kandinsky | video |
| Kandinsky5 Pro | [`fal-ai/kandinsky5-pro/text-to-video`](https://fal.ai/models/fal-ai/kandinsky5-pro/text-to-video) | Kandinsky | video |
| Kling 1.0 | [`fal-ai/kling-video/v1/standard/effects`](https://fal.ai/models/fal-ai/kling-video/v1/standard/effects) | Kling | video |
| Kling 1.0 | [`fal-ai/kling-video/v1/standard/text-to-video`](https://fal.ai/models/fal-ai/kling-video/v1/standard/text-to-video) | Kling | video |
| Kling 1.5 | [`fal-ai/kling-video/v1.5/pro/effects`](https://fal.ai/models/fal-ai/kling-video/v1.5/pro/effects) | Kling | video |
| Kling 1.5 | [`fal-ai/kling-video/v1.5/pro/text-to-video`](https://fal.ai/models/fal-ai/kling-video/v1.5/pro/text-to-video) | Kling | video |
| Kling 1.6 | [`fal-ai/kling-video/v1.6/pro/effects`](https://fal.ai/models/fal-ai/kling-video/v1.6/pro/effects) | Kling | video |
| Kling 1.6 | [`fal-ai/kling-video/v1.6/pro/text-to-video`](https://fal.ai/models/fal-ai/kling-video/v1.6/pro/text-to-video) | Kling | video |
| Kling 1.6 | [`fal-ai/kling-video/v1.6/standard/effects`](https://fal.ai/models/fal-ai/kling-video/v1.6/standard/effects) | Kling | video |
| Kling 1.6 | [`fal-ai/kling-video/v1.6/standard/text-to-video`](https://fal.ai/models/fal-ai/kling-video/v1.6/standard/text-to-video) | Kling | video |
| Kling 2.0 Master | [`fal-ai/kling-video/v2/master/text-to-video`](https://fal.ai/models/fal-ai/kling-video/v2/master/text-to-video) | Kling | video |
| Kling 2.1 Master | [`fal-ai/kling-video/v2.1/master/text-to-video`](https://fal.ai/models/fal-ai/kling-video/v2.1/master/text-to-video) | Kling | video |
| Kling LipSync Audio-to-Video | [`fal-ai/kling-video/lipsync/audio-to-video`](https://fal.ai/models/fal-ai/kling-video/lipsync/audio-to-video) | Kling | video |
| Kling LipSync Text-to-Video | [`fal-ai/kling-video/lipsync/text-to-video`](https://fal.ai/models/fal-ai/kling-video/lipsync/text-to-video) | Kling | video |
| Kling O3 Text to Video [Pro] | [`fal-ai/kling-video/o3/pro/text-to-video`](https://fal.ai/models/fal-ai/kling-video/o3/pro/text-to-video) | Kling | video |
| Kling O3 Text to Video [Standard] | [`fal-ai/kling-video/o3/standard/text-to-video`](https://fal.ai/models/fal-ai/kling-video/o3/standard/text-to-video) | Kling | video |
| Kling v2.5 Text to Video | [`fal-ai/kling-video/v2.5-turbo/pro/text-to-video`](https://fal.ai/models/fal-ai/kling-video/v2.5-turbo/pro/text-to-video) | Kling | video |
| Kling Video | [`fal-ai/kling-video/o3/4k/text-to-video`](https://fal.ai/models/fal-ai/kling-video/o3/4k/text-to-video) | Kling | video |
| Kling Video v2.6 Text to Video | [`fal-ai/kling-video/v2.6/pro/text-to-video`](https://fal.ai/models/fal-ai/kling-video/v2.6/pro/text-to-video) | Kling | video |
| Kling Video V3 Standard Turbo Text to Video | [`fal-ai/kling-video/v3/turbo/standard/text-to-video`](https://fal.ai/models/fal-ai/kling-video/v3/turbo/standard/text-to-video) | Kling | video |
| Kling Video V3 Text to Video 4K | [`fal-ai/kling-video/v3/4k/text-to-video`](https://fal.ai/models/fal-ai/kling-video/v3/4k/text-to-video) | Kling | video |
| Kling Video v3 Text to Video [Pro] | [`fal-ai/kling-video/v3/pro/text-to-video`](https://fal.ai/models/fal-ai/kling-video/v3/pro/text-to-video) | Kling | video |
| Kling Video v3 Text to Video [Standard] | [`fal-ai/kling-video/v3/standard/text-to-video`](https://fal.ai/models/fal-ai/kling-video/v3/standard/text-to-video) | Kling | video |
| Kling Video V3 Turbo Pro Text to Video | [`fal-ai/kling-video/v3/turbo/pro/text-to-video`](https://fal.ai/models/fal-ai/kling-video/v3/turbo/pro/text-to-video) | Kling | video |
| Krea Wan 14b- Text to Video | [`fal-ai/krea-wan-14b/text-to-video`](https://fal.ai/models/fal-ai/krea-wan-14b/text-to-video) | — | video |
| LongCat Video | [`fal-ai/longcat-video/text-to-video/480p`](https://fal.ai/models/fal-ai/longcat-video/text-to-video/480p) | — | video |
| LongCat Video | [`fal-ai/longcat-video/text-to-video/720p`](https://fal.ai/models/fal-ai/longcat-video/text-to-video/720p) | — | video |
| LongCat Video Distilled | [`fal-ai/longcat-video/distilled/text-to-video/480p`](https://fal.ai/models/fal-ai/longcat-video/distilled/text-to-video/480p) | — | video |
| LongCat Video Distilled | [`fal-ai/longcat-video/distilled/text-to-video/720p`](https://fal.ai/models/fal-ai/longcat-video/distilled/text-to-video/720p) | — | video |
| Ltx 2.3 Quality | [`fal-ai/ltx-2.3-quality/text-to-video`](https://fal.ai/models/fal-ai/ltx-2.3-quality/text-to-video) | Lightricks | video |
| Ltx 2.3 Quality | [`fal-ai/ltx-2.3-quality/text-to-video/lora`](https://fal.ai/models/fal-ai/ltx-2.3-quality/text-to-video/lora) | Lightricks | video |
| LTX 2.3 Video Fast | [`fal-ai/ltx-2.3/text-to-video/fast`](https://fal.ai/models/fal-ai/ltx-2.3/text-to-video/fast) | Lightricks | video |
| LTX Video (preview) | [`fal-ai/ltx-video`](https://fal.ai/models/fal-ai/ltx-video) | Lightricks | video |
| LTX Video 2.0 Fast | [`fal-ai/ltx-2/text-to-video/fast`](https://fal.ai/models/fal-ai/ltx-2/text-to-video/fast) | Lightricks | video |
| LTX Video 2.0 Pro | [`fal-ai/ltx-2/text-to-video`](https://fal.ai/models/fal-ai/ltx-2/text-to-video) | Lightricks | video |
| LTX Video 2.3 Pro | [`fal-ai/ltx-2.3/text-to-video`](https://fal.ai/models/fal-ai/ltx-2.3/text-to-video) | Lightricks | video |
| LTX Video-0.9.5 | [`fal-ai/ltx-video-v095`](https://fal.ai/models/fal-ai/ltx-video-v095) | Lightricks | video |
| LTX Video-0.9.7 13B Distilled | [`fal-ai/ltx-video-13b-distilled`](https://fal.ai/models/fal-ai/ltx-video-13b-distilled) | Lightricks | video |
| LTX-2 19B | [`fal-ai/ltx-2-19b/text-to-video`](https://fal.ai/models/fal-ai/ltx-2-19b/text-to-video) | Lightricks | video |
| LTX-2 19B | [`fal-ai/ltx-2-19b/text-to-video/lora`](https://fal.ai/models/fal-ai/ltx-2-19b/text-to-video/lora) | Lightricks | video |
| LTX-2 19B Distilled | [`fal-ai/ltx-2-19b/distilled/text-to-video`](https://fal.ai/models/fal-ai/ltx-2-19b/distilled/text-to-video) | Lightricks | video |
| LTX-2 19B Distilled | [`fal-ai/ltx-2-19b/distilled/text-to-video/lora`](https://fal.ai/models/fal-ai/ltx-2-19b/distilled/text-to-video/lora) | Lightricks | video |
| LTX-2.3 22B | [`fal-ai/ltx-2.3-22b/text-to-video`](https://fal.ai/models/fal-ai/ltx-2.3-22b/text-to-video) | Lightricks | video |
| LTX-2.3 22B | [`fal-ai/ltx-2.3-22b/text-to-video/lora`](https://fal.ai/models/fal-ai/ltx-2.3-22b/text-to-video/lora) | Lightricks | video |
| LTX-2.3 22B Distilled | [`fal-ai/ltx-2.3-22b/distilled/text-to-video`](https://fal.ai/models/fal-ai/ltx-2.3-22b/distilled/text-to-video) | Lightricks | video |
| LTX-2.3 22B Distilled | [`fal-ai/ltx-2.3-22b/distilled/text-to-video/lora`](https://fal.ai/models/fal-ai/ltx-2.3-22b/distilled/text-to-video/lora) | Lightricks | video |
| LTX-Video 13B 0.9.8 Distilled | [`fal-ai/ltxv-13b-098-distilled`](https://fal.ai/models/fal-ai/ltxv-13b-098-distilled) | Lightricks | video |
| Luma Ray 2 | [`fal-ai/luma-dream-machine/ray-2`](https://fal.ai/models/fal-ai/luma-dream-machine/ray-2) | Luma AI | video |
| Luma Ray 2 Flash | [`fal-ai/luma-dream-machine/ray-2-flash`](https://fal.ai/models/fal-ai/luma-dream-machine/ray-2-flash) | Luma AI | video |
| Luma Ray 3.2 Text to Video | [`luma/agent/ray/v3.2/text-to-video`](https://fal.ai/models/luma/agent/ray/v3.2/text-to-video) | Luma AI | video |
| MAGI-1 (Distilled) | [`fal-ai/magi-distilled`](https://fal.ai/models/fal-ai/magi-distilled) | — | video |
| Marey Realism V1.5 | [`moonvalley/marey/t2v`](https://fal.ai/models/moonvalley/marey/t2v) | Moonvalley | video |
| MiniMax (Hailuo AI) Video 01 | [`fal-ai/minimax/video-01`](https://fal.ai/models/fal-ai/minimax/video-01) | Minimax | video |
| MiniMax (Hailuo AI) Video 01 Director | [`fal-ai/minimax/video-01-director`](https://fal.ai/models/fal-ai/minimax/video-01-director) | Minimax | video |
| MiniMax (Hailuo AI) Video 01 Live | [`fal-ai/minimax/video-01-live`](https://fal.ai/models/fal-ai/minimax/video-01-live) | Minimax | video |
| MiniMax Hailuo 02 [Pro] (Text to Video) | [`fal-ai/minimax/hailuo-02/pro/text-to-video`](https://fal.ai/models/fal-ai/minimax/hailuo-02/pro/text-to-video) | Minimax | video |
| MiniMax Hailuo 02 [Standard] (Text to Video) | [`fal-ai/minimax/hailuo-02/standard/text-to-video`](https://fal.ai/models/fal-ai/minimax/hailuo-02/standard/text-to-video) | Minimax | video |
| MiniMax Hailuo 2.3 [Pro] (Text to Video) | [`fal-ai/minimax/hailuo-2.3/pro/text-to-video`](https://fal.ai/models/fal-ai/minimax/hailuo-2.3/pro/text-to-video) | Minimax | video |
| MiniMax Hailuo 2.3 [Standard] (Text to Video) | [`fal-ai/minimax/hailuo-2.3/standard/text-to-video`](https://fal.ai/models/fal-ai/minimax/hailuo-2.3/standard/text-to-video) | Minimax | video |
| Ovi Text to Video | [`fal-ai/ovi`](https://fal.ai/models/fal-ai/ovi) | — | video |
| Pika Text to Video (v2.1) | [`fal-ai/pika/v2.1/text-to-video`](https://fal.ai/models/fal-ai/pika/v2.1/text-to-video) | Pika | video |
| Pika Text to Video (v2.2) | [`fal-ai/pika/v2.2/text-to-video`](https://fal.ai/models/fal-ai/pika/v2.2/text-to-video) | Pika | video |
| Pika Text to Video Turbo (v2) | [`fal-ai/pika/v2/turbo/text-to-video`](https://fal.ai/models/fal-ai/pika/v2/turbo/text-to-video) | Pika | video |
| PixVerse C1 Text To Video | [`fal-ai/pixverse/c1/text-to-video`](https://fal.ai/models/fal-ai/pixverse/c1/text-to-video) | Pixverse | video |
| PixVerse V3.5 Text To Video | [`fal-ai/pixverse/v3.5/text-to-video`](https://fal.ai/models/fal-ai/pixverse/v3.5/text-to-video) | Pixverse | video |
| PixVerse V3.5 Text To Video Fast | [`fal-ai/pixverse/v3.5/text-to-video/fast`](https://fal.ai/models/fal-ai/pixverse/v3.5/text-to-video/fast) | Pixverse | video |
| PixVerse V4 Text To Video | [`fal-ai/pixverse/v4/text-to-video`](https://fal.ai/models/fal-ai/pixverse/v4/text-to-video) | Pixverse | video |
| PixVerse V4 Text To Video Fast | [`fal-ai/pixverse/v4/text-to-video/fast`](https://fal.ai/models/fal-ai/pixverse/v4/text-to-video/fast) | Pixverse | video |
| PixVerse V4.5 Text To Video | [`fal-ai/pixverse/v4.5/text-to-video`](https://fal.ai/models/fal-ai/pixverse/v4.5/text-to-video) | Pixverse | video |
| PixVerse V4.5 Text To Video Fast | [`fal-ai/pixverse/v4.5/text-to-video/fast`](https://fal.ai/models/fal-ai/pixverse/v4.5/text-to-video/fast) | Pixverse | video |
| PixVerse V5 Text To Video | [`fal-ai/pixverse/v5/text-to-video`](https://fal.ai/models/fal-ai/pixverse/v5/text-to-video) | Pixverse | video |
| PixVerse V5.5 Text To Video | [`fal-ai/pixverse/v5.5/text-to-video`](https://fal.ai/models/fal-ai/pixverse/v5.5/text-to-video) | Pixverse | video |
| PixVerse V5.6 Text To Video | [`fal-ai/pixverse/v5.6/text-to-video`](https://fal.ai/models/fal-ai/pixverse/v5.6/text-to-video) | Pixverse | video |
| PixVerse V6 Text To Video | [`fal-ai/pixverse/v6/text-to-video`](https://fal.ai/models/fal-ai/pixverse/v6/text-to-video) | Pixverse | video |
| Seedance 1.0 Pro | [`fal-ai/bytedance/seedance/v1/pro/text-to-video`](https://fal.ai/models/fal-ai/bytedance/seedance/v1/pro/text-to-video) | Bytedance | video |
| Seedance 2.0 Fast Text to Video | [`bytedance/seedance-2.0/fast/text-to-video`](https://fal.ai/models/bytedance/seedance-2.0/fast/text-to-video) | Bytedance | video |
| Seedance 2.0 Mini Text to Video | [`bytedance/seedance-2.0/mini/text-to-video`](https://fal.ai/models/bytedance/seedance-2.0/mini/text-to-video) | Bytedance | video |
| Seedance 2.0 Text to Video API | [`bytedance/seedance-2.0/text-to-video`](https://fal.ai/models/bytedance/seedance-2.0/text-to-video) | Bytedance | video |
| Sora 2 | [`fal-ai/sora-2/text-to-video`](https://fal.ai/models/fal-ai/sora-2/text-to-video) | OpenAI | video |
| Sora 2 | [`fal-ai/sora-2/text-to-video/pro`](https://fal.ai/models/fal-ai/sora-2/text-to-video/pro) | OpenAI | video |
| Stable Video Diffusion | [`fal-ai/fast-svd/text-to-video`](https://fal.ai/models/fal-ai/fast-svd/text-to-video) | — | video |
| Stable Video Diffusion Turbo | [`fal-ai/fast-svd-lcm/text-to-video`](https://fal.ai/models/fal-ai/fast-svd-lcm/text-to-video) | — | video |
| T2V Turbo - Video Crafter | [`fal-ai/t2v-turbo`](https://fal.ai/models/fal-ai/t2v-turbo) | — | video |
| Veo 2 | [`fal-ai/veo2`](https://fal.ai/models/fal-ai/veo2) | Google | video |
| Veo 3 | [`fal-ai/veo3`](https://fal.ai/models/fal-ai/veo3) | Google | video |
| Veo 3 Fast | [`fal-ai/veo3/fast`](https://fal.ai/models/fal-ai/veo3/fast) | Google | video |
| Veo 3.1 | [`fal-ai/veo3.1`](https://fal.ai/models/fal-ai/veo3.1) | Google | video |
| Veo 3.1 Fast | [`fal-ai/veo3.1/fast`](https://fal.ai/models/fal-ai/veo3.1/fast) | Google | video |
| Veo3.1 Lite Text to Video | [`fal-ai/veo3.1/lite`](https://fal.ai/models/fal-ai/veo3.1/lite) | Google | video |
| Vidu | [`fal-ai/vidu/q2/text-to-video`](https://fal.ai/models/fal-ai/vidu/q2/text-to-video) | Vidu | video |
| Vidu | [`fal-ai/vidu/q3/text-to-video`](https://fal.ai/models/fal-ai/vidu/q3/text-to-video) | Vidu | video |
| Vidu | [`fal-ai/vidu/q3/text-to-video/turbo`](https://fal.ai/models/fal-ai/vidu/q3/text-to-video/turbo) | Vidu | video |
| Vidu Text to Video | [`fal-ai/vidu/q1/text-to-video`](https://fal.ai/models/fal-ai/vidu/q1/text-to-video) | Vidu | video |
| Wan | [`fal-ai/wan/v2.2-5b/text-to-video/distill`](https://fal.ai/models/fal-ai/wan/v2.2-5b/text-to-video/distill) | Alibaba | video |
| Wan | [`fal-ai/wan/v2.2-5b/text-to-video/fast-wan`](https://fal.ai/models/fal-ai/wan/v2.2-5b/text-to-video/fast-wan) | Alibaba | video |
| Wan | [`fal-ai/wan/v2.2-a14b/text-to-video/turbo`](https://fal.ai/models/fal-ai/wan/v2.2-a14b/text-to-video/turbo) | Alibaba | video |
| Wan 2.5 Text to Video | [`fal-ai/wan-25-preview/text-to-video`](https://fal.ai/models/fal-ai/wan-25-preview/text-to-video) | Alibaba | video |
| Wan Text to Video | [`fal-ai/wan/v2.7/text-to-video`](https://fal.ai/models/fal-ai/wan/v2.7/text-to-video) | Alibaba | video |
| Wan v2.2 5B | [`fal-ai/wan/v2.2-5b/text-to-video`](https://fal.ai/models/fal-ai/wan/v2.2-5b/text-to-video) | Alibaba | video |
| Wan v2.6 Text to Video | [`wan/v2.6/text-to-video`](https://fal.ai/models/wan/v2.6/text-to-video) | Alibaba | video |
| Wan-2.1 Pro Text-to-Video | [`fal-ai/wan-pro/text-to-video`](https://fal.ai/models/fal-ai/wan-pro/text-to-video) | Alibaba | video |
| Wan-2.1 Text-to-Video | [`fal-ai/wan-t2v`](https://fal.ai/models/fal-ai/wan-t2v) | Alibaba | video |
| Wan-2.1 Text-to-Video with LoRAs | [`fal-ai/wan-t2v-lora`](https://fal.ai/models/fal-ai/wan-t2v-lora) | Alibaba | video |
| Wan-2.2 Text-to-Video A14B | [`fal-ai/wan/v2.2-a14b/text-to-video`](https://fal.ai/models/fal-ai/wan/v2.2-a14b/text-to-video) | Alibaba | video |
| Wan-2.2 Text-to-Video A14B with LoRAs | [`fal-ai/wan/v2.2-a14b/text-to-video/lora`](https://fal.ai/models/fal-ai/wan/v2.2-a14b/text-to-video/lora) | Alibaba | video |

</details>

<details>
<summary><strong>training</strong> — 52 models</summary>

| Model | Endpoint | Lab | Output |
| --- | --- | --- | --- |
| ERNIE-Image Trainer | [`fal-ai/ernie-image-trainer`](https://fal.ai/models/fal-ai/ernie-image-trainer) | Baidu | json |
| FLUX 2 [klein] 9b Base Trainer | [`fal-ai/flux-2-klein-9b-base-trainer`](https://fal.ai/models/fal-ai/flux-2-klein-9b-base-trainer) | Black Forest Labs | json |
| Flux 2 Klein 9B Base Trainer | [`fal-ai/flux-2-klein-9b-base-trainer/edit`](https://fal.ai/models/fal-ai/flux-2-klein-9b-base-trainer/edit) | Black Forest Labs | json |
| FLUX 2 Trainer | [`fal-ai/flux-2-trainer`](https://fal.ai/models/fal-ai/flux-2-trainer) | Black Forest Labs | json |
| FLUX 2 Trainer Edit | [`fal-ai/flux-2-trainer/edit`](https://fal.ai/models/fal-ai/flux-2-trainer/edit) | Black Forest Labs | json |
| FLUX 2 Trainer V2 | [`fal-ai/flux-2-trainer-v2`](https://fal.ai/models/fal-ai/flux-2-trainer-v2) | Black Forest Labs | json |
| FLUX 2 Trainer V2 Edit | [`fal-ai/flux-2-trainer-v2/edit`](https://fal.ai/models/fal-ai/flux-2-trainer-v2/edit) | Black Forest Labs | json |
| Flux Kontext Trainer | [`fal-ai/flux-kontext-trainer`](https://fal.ai/models/fal-ai/flux-kontext-trainer) | Black Forest Labs | json |
| Ideogram | [`fal-ai/ideogram/custom-models`](https://fal.ai/models/fal-ai/ideogram/custom-models) | Ideogram | text |
| Ideogram V4.0q LoRA Trainer | [`ideogram/v4/trainer`](https://fal.ai/models/ideogram/v4/trainer) | Ideogram | json |
| Krea 2 Trainer | [`fal-ai/krea-2-trainer`](https://fal.ai/models/fal-ai/krea-2-trainer) | Krea | json |
| LTX 2.3 Trainer (V2) - Audio Inpainting | [`fal-ai/ltx23-trainer-v2/audio-inpaint`](https://fal.ai/models/fal-ai/ltx23-trainer-v2/audio-inpaint) | Lightricks | video |
| LTX 2.3 Trainer (V2) - Audio+Video Reference IC-LoRA | [`fal-ai/ltx23-trainer-v2/ic-lora/av2av`](https://fal.ai/models/fal-ai/ltx23-trainer-v2/ic-lora/av2av) | Lightricks | video |
| LTX 2.3 Trainer (V2) - Audio+Video Reference Transformation | [`fal-ai/ltx23-trainer-v2/av2av`](https://fal.ai/models/fal-ai/ltx23-trainer-v2/av2av) | Lightricks | video |
| LTX 2.3 Trainer (V2) - Audio-to-Audio | [`fal-ai/ltx23-trainer-v2/a2a`](https://fal.ai/models/fal-ai/ltx23-trainer-v2/a2a) | Lightricks | video |
| LTX 2.3 Trainer (V2) - Audio-to-Audio IC-LoRA | [`fal-ai/ltx23-trainer-v2/ic-lora/a2a`](https://fal.ai/models/fal-ai/ltx23-trainer-v2/ic-lora/a2a) | Lightricks | video |
| LTX 2.3 Trainer (V2) - Audio-to-Video | [`fal-ai/ltx23-trainer-v2/a2v`](https://fal.ai/models/fal-ai/ltx23-trainer-v2/a2v) | Lightricks | video |
| LTX 2.3 Trainer (V2) - Backward Audio Extension | [`fal-ai/ltx23-trainer-v2/audio-extend-suffix`](https://fal.ai/models/fal-ai/ltx23-trainer-v2/audio-extend-suffix) | Lightricks | video |
| LTX 2.3 Trainer (V2) - Backward Video Extension | [`fal-ai/ltx23-trainer-v2/extend-suffix`](https://fal.ai/models/fal-ai/ltx23-trainer-v2/extend-suffix) | Lightricks | video |
| LTX 2.3 Trainer (V2) - Forward Audio Extension | [`fal-ai/ltx23-trainer-v2/audio-extend-prefix`](https://fal.ai/models/fal-ai/ltx23-trainer-v2/audio-extend-prefix) | Lightricks | video |
| LTX 2.3 Trainer (V2) - Forward Video Extension | [`fal-ai/ltx23-trainer-v2/extend-prefix`](https://fal.ai/models/fal-ai/ltx23-trainer-v2/extend-prefix) | Lightricks | video |
| LTX 2.3 Trainer (V2) - Image-to-Video | [`fal-ai/ltx23-trainer-v2/i2v`](https://fal.ai/models/fal-ai/ltx23-trainer-v2/i2v) | Lightricks | video |
| LTX 2.3 Trainer (V2) - Keyframe Interpolation | [`fal-ai/ltx23-trainer-v2/interpolate`](https://fal.ai/models/fal-ai/ltx23-trainer-v2/interpolate) | Lightricks | video |
| LTX 2.3 Trainer (V2) - Masked Audio+Video IC-LoRA | [`fal-ai/ltx23-trainer-v2/ic-lora/av2av-masked`](https://fal.ai/models/fal-ai/ltx23-trainer-v2/ic-lora/av2av-masked) | Lightricks | video |
| LTX 2.3 Trainer (V2) - Masked Audio+Video Transformation | [`fal-ai/ltx23-trainer-v2/av2av-masked`](https://fal.ai/models/fal-ai/ltx23-trainer-v2/av2av-masked) | Lightricks | video |
| LTX 2.3 Trainer (V2) - Masked Video-to-Video | [`fal-ai/ltx23-trainer-v2/v2v-masked`](https://fal.ai/models/fal-ai/ltx23-trainer-v2/v2v-masked) | Lightricks | video |
| LTX 2.3 Trainer (V2) - Masked Video-to-Video IC-LoRA | [`fal-ai/ltx23-trainer-v2/ic-lora/v2v-masked`](https://fal.ai/models/fal-ai/ltx23-trainer-v2/ic-lora/v2v-masked) | Lightricks | video |
| LTX 2.3 Trainer (V2) - Spatial Outpainting | [`fal-ai/ltx23-trainer-v2/outpaint`](https://fal.ai/models/fal-ai/ltx23-trainer-v2/outpaint) | Lightricks | video |
| LTX 2.3 Trainer (V2) - Text-to-Audio | [`fal-ai/ltx23-trainer-v2/t2a`](https://fal.ai/models/fal-ai/ltx23-trainer-v2/t2a) | Lightricks | video |
| LTX 2.3 Trainer (V2) - Text-to-Video | [`fal-ai/ltx23-trainer-v2/t2v`](https://fal.ai/models/fal-ai/ltx23-trainer-v2/t2v) | Lightricks | video |
| LTX 2.3 Trainer (V2) - Video Inpainting | [`fal-ai/ltx23-trainer-v2/inpaint`](https://fal.ai/models/fal-ai/ltx23-trainer-v2/inpaint) | Lightricks | video |
| LTX 2.3 Trainer (V2) - Video-to-Audio | [`fal-ai/ltx23-trainer-v2/v2a`](https://fal.ai/models/fal-ai/ltx23-trainer-v2/v2a) | Lightricks | video |
| LTX 2.3 Trainer (V2) - Video-to-Video | [`fal-ai/ltx23-trainer-v2/v2v`](https://fal.ai/models/fal-ai/ltx23-trainer-v2/v2v) | Lightricks | video |
| LTX 2.3 Trainer (V2) - Video-to-Video IC-LoRA | [`fal-ai/ltx23-trainer-v2/ic-lora/v2v`](https://fal.ai/models/fal-ai/ltx23-trainer-v2/ic-lora/v2v) | Lightricks | video |
| LTX-2.3 22B Video to Video Trainer | [`fal-ai/ltx23-v2v-trainer`](https://fal.ai/models/fal-ai/ltx23-v2v-trainer) | Lightricks | video |
| LTX-2.3 22B Video Trainer | [`fal-ai/ltx23-video-trainer`](https://fal.ai/models/fal-ai/ltx23-video-trainer) | Lightricks | video |
| Phota Create Profile | [`fal-ai/phota/create-profile`](https://fal.ai/models/fal-ai/phota/create-profile) | Phota | text |
| Qwen Image 2512 Trainer | [`fal-ai/qwen-image-2512-trainer`](https://fal.ai/models/fal-ai/qwen-image-2512-trainer) | Alibaba | json |
| Qwen Image Edit 2509 Trainer | [`fal-ai/qwen-image-edit-2509-trainer`](https://fal.ai/models/fal-ai/qwen-image-edit-2509-trainer) | Alibaba | json |
| Recraft V3 Create Style | [`fal-ai/recraft/v3/create-style`](https://fal.ai/models/fal-ai/recraft/v3/create-style) | Recraft | text |
| Stable Audio 3 Trainer | [`fal-ai/stable-audio-3-trainer`](https://fal.ai/models/fal-ai/stable-audio-3-trainer) | Stability AI | json |
| Train Flux LoRA | [`fal-ai/flux-lora-fast-training`](https://fal.ai/models/fal-ai/flux-lora-fast-training) | Black Forest Labs | json |
| Train Flux LoRAs For Portraits | [`fal-ai/flux-lora-portrait-trainer`](https://fal.ai/models/fal-ai/flux-lora-portrait-trainer) | Black Forest Labs | json |
| Turbo Flux Trainer | [`fal-ai/turbo-flux-trainer`](https://fal.ai/models/fal-ai/turbo-flux-trainer) | Black Forest Labs | json |
| Wan 2.2 14B Image Trainer | [`fal-ai/wan-22-image-trainer`](https://fal.ai/models/fal-ai/wan-22-image-trainer) | Alibaba | json |
| Wan-2.1 LoRA Trainer | [`fal-ai/wan-trainer/i2v-720p`](https://fal.ai/models/fal-ai/wan-trainer/i2v-720p) | Alibaba | json |
| Wan-2.1 LoRA Trainer | [`fal-ai/wan-trainer/t2v`](https://fal.ai/models/fal-ai/wan-trainer/t2v) | Alibaba | json |
| Wan-2.1 LoRA Trainer | [`fal-ai/wan-trainer/t2v-14b`](https://fal.ai/models/fal-ai/wan-trainer/t2v-14b) | Alibaba | json |
| Wan-2.2 LoRA Trainer | [`fal-ai/wan-22-trainer/i2v-a14b`](https://fal.ai/models/fal-ai/wan-22-trainer/i2v-a14b) | Alibaba | json |
| Wan-2.2 LoRA Trainer | [`fal-ai/wan-22-trainer/t2v-a14b`](https://fal.ai/models/fal-ai/wan-22-trainer/t2v-a14b) | Alibaba | json |
| Z Image Trainer | [`fal-ai/z-image-trainer`](https://fal.ai/models/fal-ai/z-image-trainer) | Alibaba | json |
| Z Image Turbo Trainer V2 | [`fal-ai/z-image-turbo-trainer-v2`](https://fal.ai/models/fal-ai/z-image-turbo-trainer-v2) | Alibaba | json |

</details>

<details>
<summary><strong>text-to-audio</strong> — 45 models</summary>

| Model | Endpoint | Lab | Output |
| --- | --- | --- | --- |
| ACE Step | [`fal-ai/ace-step`](https://fal.ai/models/fal-ai/ace-step) | — | audio |
| ACE Step Prompt To Audio | [`fal-ai/ace-step/prompt-to-audio`](https://fal.ai/models/fal-ai/ace-step/prompt-to-audio) | — | audio |
| CSM-1B | [`fal-ai/csm-1b`](https://fal.ai/models/fal-ai/csm-1b) | — | audio |
| DiffRhythm: Lyrics to Song | [`fal-ai/diffrhythm`](https://fal.ai/models/fal-ai/diffrhythm) | — | audio |
| Elevenlabs | [`fal-ai/elevenlabs/text-to-dialogue/eleven-v3`](https://fal.ai/models/fal-ai/elevenlabs/text-to-dialogue/eleven-v3) | ElevenLabs | audio |
| Elevenlabs Music | [`fal-ai/elevenlabs/music`](https://fal.ai/models/fal-ai/elevenlabs/music) | ElevenLabs | audio |
| Elevenlabs Sound Effects V2 | [`fal-ai/elevenlabs/sound-effects/v2`](https://fal.ai/models/fal-ai/elevenlabs/sound-effects/v2) | ElevenLabs | audio |
| Elevenlabs Tts Eleven V3 | [`fal-ai/elevenlabs/tts/eleven-v3`](https://fal.ai/models/fal-ai/elevenlabs/tts/eleven-v3) | ElevenLabs | audio |
| ElevenLabs TTS Multilingual v2 | [`fal-ai/elevenlabs/tts/multilingual-v2`](https://fal.ai/models/fal-ai/elevenlabs/tts/multilingual-v2) | ElevenLabs | audio |
| F5 TTS | [`fal-ai/f5-tts`](https://fal.ai/models/fal-ai/f5-tts) | — | json |
| Gemini TTS | [`fal-ai/gemini-tts`](https://fal.ai/models/fal-ai/gemini-tts) | Google | audio |
| Kokoro TTS | [`fal-ai/kokoro/american-english`](https://fal.ai/models/fal-ai/kokoro/american-english) | — | audio |
| Kokoro TTS (Brazilian Portuguese) | [`fal-ai/kokoro/brazilian-portuguese`](https://fal.ai/models/fal-ai/kokoro/brazilian-portuguese) | — | audio |
| Kokoro TTS (British English) | [`fal-ai/kokoro/british-english`](https://fal.ai/models/fal-ai/kokoro/british-english) | — | audio |
| Kokoro TTS (French) | [`fal-ai/kokoro/french`](https://fal.ai/models/fal-ai/kokoro/french) | — | audio |
| Kokoro TTS (Hindi) | [`fal-ai/kokoro/hindi`](https://fal.ai/models/fal-ai/kokoro/hindi) | — | audio |
| Kokoro TTS (Italian) | [`fal-ai/kokoro/italian`](https://fal.ai/models/fal-ai/kokoro/italian) | — | audio |
| Kokoro TTS (Japanese) | [`fal-ai/kokoro/japanese`](https://fal.ai/models/fal-ai/kokoro/japanese) | — | audio |
| Kokoro TTS (Mandarin Chinese) | [`fal-ai/kokoro/mandarin-chinese`](https://fal.ai/models/fal-ai/kokoro/mandarin-chinese) | — | audio |
| Kokoro TTS (Spanish) | [`fal-ai/kokoro/spanish`](https://fal.ai/models/fal-ai/kokoro/spanish) | — | audio |
| Ltx 2.3 Quality | [`fal-ai/ltx-2.3-quality/text-to-audio`](https://fal.ai/models/fal-ai/ltx-2.3-quality/text-to-audio) | Lightricks | audio |
| Ltx 2.3 Quality | [`fal-ai/ltx-2.3-quality/text-to-audio/lora`](https://fal.ai/models/fal-ai/ltx-2.3-quality/text-to-audio/lora) | Lightricks | audio |
| Lyria 3 Pro | [`fal-ai/lyria3/pro`](https://fal.ai/models/fal-ai/lyria3/pro) | Google | audio |
| Lyria2 | [`fal-ai/lyria2`](https://fal.ai/models/fal-ai/lyria2) | Google | audio |
| Lyria3 | [`fal-ai/lyria3`](https://fal.ai/models/fal-ai/lyria3) | Google | audio |
| MiniMax (Hailuo AI) Music | [`fal-ai/minimax-music`](https://fal.ai/models/fal-ai/minimax-music) | Minimax | audio |
| MiniMax (Hailuo AI) Music v1.5 | [`fal-ai/minimax-music/v1.5`](https://fal.ai/models/fal-ai/minimax-music/v1.5) | Minimax | audio |
| Minimax Music | [`fal-ai/minimax-music/v2`](https://fal.ai/models/fal-ai/minimax-music/v2) | Minimax | audio |
| Minimax Music 2.5 | [`fal-ai/minimax-music/v2.5`](https://fal.ai/models/fal-ai/minimax-music/v2.5) | Minimax | audio |
| Minimax Music 2.6 | [`fal-ai/minimax-music/v2.6`](https://fal.ai/models/fal-ai/minimax-music/v2.6) | Minimax | audio |
| Mirelo SFX1.6 | [`mirelo-ai/sfx1.6/text-to-audio`](https://fal.ai/models/mirelo-ai/sfx1.6/text-to-audio) | Mirelo AI | audio |
| MMAudio V2 Text to Audio | [`fal-ai/mmaudio-v2/text-to-audio`](https://fal.ai/models/fal-ai/mmaudio-v2/text-to-audio) | — | audio |
| music generator | [`cassetteai/music-generator`](https://fal.ai/models/cassetteai/music-generator) | Cassette AI | json |
| Seed Audio 1.0 | [`bytedance/seed-audio-1.0`](https://fal.ai/models/bytedance/seed-audio-1.0) | Bytedance | audio |
| Sonilo V1.1 Text to Music | [`sonilo/v1.1/text-to-music`](https://fal.ai/models/sonilo/v1.1/text-to-music) | Sonilo | audio |
| Sound Effects Generator | [`cassetteai/sound-effects-generator`](https://fal.ai/models/cassetteai/sound-effects-generator) | Cassette AI | json |
| Stable Audio 2.5 | [`fal-ai/stable-audio-25/text-to-audio`](https://fal.ai/models/fal-ai/stable-audio-25/text-to-audio) | Stability AI | audio |
| Stable Audio 3 | [`fal-ai/stable-audio-3/medium/text-to-audio`](https://fal.ai/models/fal-ai/stable-audio-3/medium/text-to-audio) | Stability AI | audio |
| Stable Audio 3 | [`fal-ai/stable-audio-3/small/music/base/text-to-audio`](https://fal.ai/models/fal-ai/stable-audio-3/small/music/base/text-to-audio) | Stability AI | audio |
| Stable Audio 3 Medium Base Text to Audio | [`fal-ai/stable-audio-3/medium/base/text-to-audio`](https://fal.ai/models/fal-ai/stable-audio-3/medium/base/text-to-audio) | Stability AI | audio |
| Stable Audio 3 Small Music Text to Audio | [`fal-ai/stable-audio-3/small/music/text-to-audio`](https://fal.ai/models/fal-ai/stable-audio-3/small/music/text-to-audio) | Stability AI | audio |
| Stable Audio 3 Small SFX Base Text to Audio | [`fal-ai/stable-audio-3/small/sfx/base/text-to-audio`](https://fal.ai/models/fal-ai/stable-audio-3/small/sfx/base/text-to-audio) | — | audio |
| Stable Audio 3 Small SFX Text to Audio | [`fal-ai/stable-audio-3/small/sfx/text-to-audio`](https://fal.ai/models/fal-ai/stable-audio-3/small/sfx/text-to-audio) | Stability AI | audio |
| Stable Audio Open | [`fal-ai/stable-audio`](https://fal.ai/models/fal-ai/stable-audio) | Stability AI | json |
| Zonos-Audio-Clone | [`fal-ai/zonos`](https://fal.ai/models/fal-ai/zonos) | — | audio |

</details>

<details>
<summary><strong>audio-to-audio</strong> — 43 models</summary>

| Model | Endpoint | Lab | Output |
| --- | --- | --- | --- |
| ACE Step Audio Inpaint | [`fal-ai/ace-step/audio-inpaint`](https://fal.ai/models/fal-ai/ace-step/audio-inpaint) | — | audio |
| ACE Step Audio Outpaint | [`fal-ai/ace-step/audio-outpaint`](https://fal.ai/models/fal-ai/ace-step/audio-outpaint) | — | audio |
| ACE Step Audio To Audio | [`fal-ai/ace-step/audio-to-audio`](https://fal.ai/models/fal-ai/ace-step/audio-to-audio) | — | audio |
| Audio Understanding | [`fal-ai/audio-understanding`](https://fal.ai/models/fal-ai/audio-understanding) | — | text |
| DeepFilterNet 3 | [`fal-ai/deepfilternet3`](https://fal.ai/models/fal-ai/deepfilternet3) | — | json |
| Demucs | [`fal-ai/demucs`](https://fal.ai/models/fal-ai/demucs) | — | json |
| Dia Tts | [`fal-ai/dia-tts/voice-clone`](https://fal.ai/models/fal-ai/dia-tts/voice-clone) | — | audio |
| ElevenLabs Audio Isolation | [`fal-ai/elevenlabs/audio-isolation`](https://fal.ai/models/fal-ai/elevenlabs/audio-isolation) | ElevenLabs | audio |
| ElevenLabs Voice Changer | [`fal-ai/elevenlabs/voice-changer`](https://fal.ai/models/fal-ai/elevenlabs/voice-changer) | ElevenLabs | audio |
| FFmpeg API [Merge Audios] | [`fal-ai/ffmpeg-api/merge-audios`](https://fal.ai/models/fal-ai/ffmpeg-api/merge-audios) | — | audio |
| Kling Video Create Voice | [`fal-ai/kling-video/create-voice`](https://fal.ai/models/fal-ai/kling-video/create-voice) | Kling | text |
| Mirelo SFX1.6 | [`mirelo-ai/sfx1.6/extend-audio`](https://fal.ai/models/mirelo-ai/sfx1.6/extend-audio) | Mirelo AI | audio |
| Mirelo SFX1.6 | [`mirelo-ai/sfx1.6/inpaint-audio`](https://fal.ai/models/mirelo-ai/sfx1.6/inpaint-audio) | Mirelo AI | audio |
| Personaplex | [`fal-ai/personaplex`](https://fal.ai/models/fal-ai/personaplex) | — | audio |
| Personaplex | [`fal-ai/personaplex/realtime`](https://fal.ai/models/fal-ai/personaplex/realtime) | — | audio |
| Qwen 3 TTS - Clone Voice [0.6B] | [`fal-ai/qwen-3-tts/clone-voice/0.6b`](https://fal.ai/models/fal-ai/qwen-3-tts/clone-voice/0.6b) | Alibaba | json |
| Qwen 3 TTS - Clone Voice [1.7B] | [`fal-ai/qwen-3-tts/clone-voice/1.7b`](https://fal.ai/models/fal-ai/qwen-3-tts/clone-voice/1.7b) | Alibaba | json |
| Sam Audio | [`fal-ai/sam-audio/separate`](https://fal.ai/models/fal-ai/sam-audio/separate) | — | json |
| Sam Audio | [`fal-ai/sam-audio/span-separate`](https://fal.ai/models/fal-ai/sam-audio/span-separate) | — | json |
| Stable Audio 2.5 | [`fal-ai/stable-audio-25/audio-to-audio`](https://fal.ai/models/fal-ai/stable-audio-25/audio-to-audio) | Stability AI | audio |
| Stable Audio 25 | [`fal-ai/stable-audio-25/inpaint`](https://fal.ai/models/fal-ai/stable-audio-25/inpaint) | Stability AI | audio |
| Stable Audio 3 | [`fal-ai/stable-audio-3/small/music/audio-to-audio`](https://fal.ai/models/fal-ai/stable-audio-3/small/music/audio-to-audio) | Stability AI | audio |
| Stable Audio 3 | [`fal-ai/stable-audio-3/small/sfx/audio-to-audio`](https://fal.ai/models/fal-ai/stable-audio-3/small/sfx/audio-to-audio) | Stability AI | audio |
| Stable Audio 3 Medium Audio Inpainting | [`fal-ai/stable-audio-3/medium/audio-inpainting`](https://fal.ai/models/fal-ai/stable-audio-3/medium/audio-inpainting) | Stability AI | audio |
| Stable Audio 3 Medium Audio Outpainting | [`fal-ai/stable-audio-3/medium/audio-outpainting`](https://fal.ai/models/fal-ai/stable-audio-3/medium/audio-outpainting) | Stability AI | audio |
| Stable Audio 3 Medium Audio to Audio | [`fal-ai/stable-audio-3/medium/audio-to-audio`](https://fal.ai/models/fal-ai/stable-audio-3/medium/audio-to-audio) | Stability AI | audio |
| Stable Audio 3 Medium Base Audio Inpainting | [`fal-ai/stable-audio-3/medium/base/audio-inpainting`](https://fal.ai/models/fal-ai/stable-audio-3/medium/base/audio-inpainting) | Stability AI | audio |
| Stable Audio 3 Medium Base Audio Outpainting | [`fal-ai/stable-audio-3/medium/base/audio-outpainting`](https://fal.ai/models/fal-ai/stable-audio-3/medium/base/audio-outpainting) | Stability AI | audio |
| Stable Audio 3 Medium Base Audio to Audio | [`fal-ai/stable-audio-3/medium/base/audio-to-audio`](https://fal.ai/models/fal-ai/stable-audio-3/medium/base/audio-to-audio) | Stability AI | audio |
| Stable Audio 3 Small Music Audio Inpainting | [`fal-ai/stable-audio-3/small/music/audio-inpainting`](https://fal.ai/models/fal-ai/stable-audio-3/small/music/audio-inpainting) | Stability AI | audio |
| Stable Audio 3 Small Music Audio Outpainting | [`fal-ai/stable-audio-3/small/music/audio-outpainting`](https://fal.ai/models/fal-ai/stable-audio-3/small/music/audio-outpainting) | Stability AI | audio |
| Stable Audio 3 Small Music Base Audio Inpainting | [`fal-ai/stable-audio-3/small/music/base/audio-inpainting`](https://fal.ai/models/fal-ai/stable-audio-3/small/music/base/audio-inpainting) | Stability AI | audio |
| Stable Audio 3 Small Music Base Audio Outpainting | [`fal-ai/stable-audio-3/small/music/base/audio-outpainting`](https://fal.ai/models/fal-ai/stable-audio-3/small/music/base/audio-outpainting) | Stability AI | audio |
| Stable Audio 3 Small Music Base Audio to Audio | [`fal-ai/stable-audio-3/small/music/base/audio-to-audio`](https://fal.ai/models/fal-ai/stable-audio-3/small/music/base/audio-to-audio) | Stability AI | audio |
| Stable Audio 3 Small SFX Audio Inpainting | [`fal-ai/stable-audio-3/small/sfx/audio-inpainting`](https://fal.ai/models/fal-ai/stable-audio-3/small/sfx/audio-inpainting) | Stability AI | audio |
| Stable Audio 3 Small SFX Audio Outpainting | [`fal-ai/stable-audio-3/small/sfx/audio-outpainting`](https://fal.ai/models/fal-ai/stable-audio-3/small/sfx/audio-outpainting) | Stability AI | audio |
| Stable Audio 3 Small SFX Base Audio Inpainting | [`fal-ai/stable-audio-3/small/sfx/base/audio-inpainting`](https://fal.ai/models/fal-ai/stable-audio-3/small/sfx/base/audio-inpainting) | Stability AI | audio |
| Stable Audio 3 Small SFX Base Audio Outpainting | [`fal-ai/stable-audio-3/small/sfx/base/audio-outpainting`](https://fal.ai/models/fal-ai/stable-audio-3/small/sfx/base/audio-outpainting) | Stability AI | audio |
| Stable Audio 3 Small SFX Base Audio to Audio | [`fal-ai/stable-audio-3/small/sfx/base/audio-to-audio`](https://fal.ai/models/fal-ai/stable-audio-3/small/sfx/base/audio-to-audio) | Stability AI | audio |
| Tada | [`fal-ai/tada/3b/text-to-speech`](https://fal.ai/models/fal-ai/tada/3b/text-to-speech) | — | audio |
| Tada TTS 1B | [`fal-ai/tada/1b/text-to-speech`](https://fal.ai/models/fal-ai/tada/1b/text-to-speech) | Hume AI | audio |
| Workflow Utilities Audio Compressor | [`fal-ai/workflow-utilities/audio-compressor`](https://fal.ai/models/fal-ai/workflow-utilities/audio-compressor) | — | audio |
| Workflow Utilities Impulse Response | [`fal-ai/workflow-utilities/impulse-response`](https://fal.ai/models/fal-ai/workflow-utilities/impulse-response) | — | audio |

</details>

<details>
<summary><strong>image-to-3d</strong> — 36 models</summary>

| Model | Endpoint | Lab | Output |
| --- | --- | --- | --- |
| Hunyuan 3D Pro Image to 3D | [`fal-ai/hunyuan-3d/v3.1/pro/image-to-3d`](https://fal.ai/models/fal-ai/hunyuan-3d/v3.1/pro/image-to-3d) | Tencent | file |
| Hunyuan 3D Rapid Image to 3D | [`fal-ai/hunyuan-3d/v3.1/rapid/image-to-3d`](https://fal.ai/models/fal-ai/hunyuan-3d/v3.1/rapid/image-to-3d) | Tencent | file |
| Hunyuan World | [`fal-ai/hunyuan_world/image-to-world`](https://fal.ai/models/fal-ai/hunyuan_world/image-to-world) | Tencent | json |
| Hunyuan3D | [`fal-ai/hunyuan3d/v2`](https://fal.ai/models/fal-ai/hunyuan3d/v2) | Tencent | file |
| Hunyuan3D | [`fal-ai/hunyuan3d/v2/mini`](https://fal.ai/models/fal-ai/hunyuan3d/v2/mini) | Tencent | file |
| Hunyuan3D | [`fal-ai/hunyuan3d/v2/mini/turbo`](https://fal.ai/models/fal-ai/hunyuan3d/v2/mini/turbo) | Tencent | file |
| Hunyuan3D | [`fal-ai/hunyuan3d/v2/multi-view`](https://fal.ai/models/fal-ai/hunyuan3d/v2/multi-view) | Tencent | file |
| Hunyuan3D | [`fal-ai/hunyuan3d/v2/multi-view/turbo`](https://fal.ai/models/fal-ai/hunyuan3d/v2/multi-view/turbo) | Tencent | file |
| Hunyuan3D | [`fal-ai/hunyuan3d/v2/turbo`](https://fal.ai/models/fal-ai/hunyuan3d/v2/turbo) | Tencent | file |
| Hunyuan3d V3 | [`fal-ai/hunyuan3d-v3/image-to-3d`](https://fal.ai/models/fal-ai/hunyuan3d-v3/image-to-3d) | Tencent | file |
| Hunyuan3d V3 | [`fal-ai/hunyuan3d-v3/sketch-to-3d`](https://fal.ai/models/fal-ai/hunyuan3d-v3/sketch-to-3d) | Tencent | file |
| Hyper3d | [`fal-ai/hyper3d/rodin/v2`](https://fal.ai/models/fal-ai/hyper3d/rodin/v2) | Hyper3D | file |
| Hyper3D - Rodin V2.5 - Image to 3D | [`fal-ai/hyper3d/rodin/v2.5`](https://fal.ai/models/fal-ai/hyper3d/rodin/v2.5) | Hyper3D | file |
| Hyper3D - Rodin V2.5 - Image to 3D - Fast | [`fal-ai/hyper3d/rodin/v2.5/fast`](https://fal.ai/models/fal-ai/hyper3d/rodin/v2.5/fast) | Hyper3D | file |
| Hyper3D Rodin | [`fal-ai/hyper3d/rodin`](https://fal.ai/models/fal-ai/hyper3d/rodin) | Hyper3D | file |
| Meshy 5 Multi | [`fal-ai/meshy/v5/multi-image-to-3d`](https://fal.ai/models/fal-ai/meshy/v5/multi-image-to-3d) | Meshy | file |
| Meshy 6 | [`fal-ai/meshy/v6/image-to-3d`](https://fal.ai/models/fal-ai/meshy/v6/image-to-3d) | Meshy | file |
| Meshy 6 - Multi Image To 3D | [`fal-ai/meshy/v6/multi-image-to-3d`](https://fal.ai/models/fal-ai/meshy/v6/multi-image-to-3d) | Meshy | file |
| Meshy 6 Preview | [`fal-ai/meshy/v6-preview/image-to-3d`](https://fal.ai/models/fal-ai/meshy/v6-preview/image-to-3d) | Meshy | file |
| Pixal3d | [`fal-ai/pixal3d`](https://fal.ai/models/fal-ai/pixal3d) | — | file |
| ReconViaGen 0.5 | [`fal-ai/reconviagen-0.5`](https://fal.ai/models/fal-ai/reconviagen-0.5) | — | file |
| Sam 3 | [`fal-ai/sam-3/3d-body`](https://fal.ai/models/fal-ai/sam-3/3d-body) | — | file |
| Sam 3 | [`fal-ai/sam-3/3d-objects`](https://fal.ai/models/fal-ai/sam-3/3d-objects) | — | file |
| Trellis | [`fal-ai/trellis`](https://fal.ai/models/fal-ai/trellis) | Trellis | file |
| Trellis | [`fal-ai/trellis/multi`](https://fal.ai/models/fal-ai/trellis/multi) | Trellis | file |
| Trellis 2 | [`fal-ai/trellis-2`](https://fal.ai/models/fal-ai/trellis-2) | Trellis | file |
| Trellis 2 | [`fal-ai/trellis-2/retexture`](https://fal.ai/models/fal-ai/trellis-2/retexture) | Trellis | file |
| TRELLIS.2 LoRA Inference | [`fal-ai/trellis-2-lora`](https://fal.ai/models/fal-ai/trellis-2-lora) | Trellis | file |
| TRELLIS.2 Trainer | [`fal-ai/trellis-2-lora-trainer`](https://fal.ai/models/fal-ai/trellis-2-lora-trainer) | Trellis | json |
| Tripo H3.1 Image to 3D | [`tripo3d/h3.1/image-to-3d`](https://fal.ai/models/tripo3d/h3.1/image-to-3d) | Tripo3D | file |
| Tripo H3.1 Multiview to 3D | [`tripo3d/h3.1/multiview-to-3d`](https://fal.ai/models/tripo3d/h3.1/multiview-to-3d) | Tripo3D | file |
| Tripo P1 Image to 3D | [`tripo3d/p1/image-to-3d`](https://fal.ai/models/tripo3d/p1/image-to-3d) | Tripo3D | file |
| Tripo3D | [`tripo3d/tripo/v2.5/image-to-3d`](https://fal.ai/models/tripo3d/tripo/v2.5/image-to-3d) | Tripo3D | file |
| Tripo3D | [`tripo3d/tripo/v2.5/multiview-to-3d`](https://fal.ai/models/tripo3d/tripo/v2.5/multiview-to-3d) | Tripo3D | file |
| Triposplat | [`tripo3d/triposplat`](https://fal.ai/models/tripo3d/triposplat) | Tripo3D | file |
| TripoSR | [`fal-ai/triposr`](https://fal.ai/models/fal-ai/triposr) | — | file |

</details>

<details>
<summary><strong>text-to-speech</strong> — 33 models</summary>

| Model | Endpoint | Lab | Output |
| --- | --- | --- | --- |
| Async Text to Speech Pro V1.0 | [`async/tts-pro/v1.0`](https://fal.ai/models/async/tts-pro/v1.0) | Async | audio |
| Bytedance Seed Speech Text to Speech | [`fal-ai/bytedance/seed-speech/tts/v2`](https://fal.ai/models/fal-ai/bytedance/seed-speech/tts/v2) | Bytedance | audio |
| Chatterbox | [`fal-ai/chatterbox/text-to-speech`](https://fal.ai/models/fal-ai/chatterbox/text-to-speech) | Resemble AI | audio |
| Chatterbox | [`fal-ai/chatterbox/text-to-speech/multilingual`](https://fal.ai/models/fal-ai/chatterbox/text-to-speech/multilingual) | Resemble AI | audio |
| Chatterboxhd | [`resemble-ai/chatterboxhd/text-to-speech`](https://fal.ai/models/resemble-ai/chatterboxhd/text-to-speech) | Resemble AI | audio |
| Dia | [`fal-ai/dia-tts`](https://fal.ai/models/fal-ai/dia-tts) | — | audio |
| ElevenLabs TTS Turbo v2.5 | [`fal-ai/elevenlabs/tts/turbo-v2.5`](https://fal.ai/models/fal-ai/elevenlabs/tts/turbo-v2.5) | ElevenLabs | audio |
| Gemini 3.1 Flash Tts | [`fal-ai/gemini-3.1-flash-tts`](https://fal.ai/models/fal-ai/gemini-3.1-flash-tts) | Google | audio |
| Index TTS 2.0 | [`fal-ai/index-tts-2/text-to-speech`](https://fal.ai/models/fal-ai/index-tts-2/text-to-speech) | — | audio |
| Inworld TTS-1.5 Max | [`fal-ai/inworld-tts`](https://fal.ai/models/fal-ai/inworld-tts) | Inworld | audio |
| Kling TTS | [`fal-ai/kling-video/v1/tts`](https://fal.ai/models/fal-ai/kling-video/v1/tts) | Kling | audio |
| Maya | [`fal-ai/maya/batch`](https://fal.ai/models/fal-ai/maya/batch) | — | audio |
| Maya | [`fal-ai/maya/stream`](https://fal.ai/models/fal-ai/maya/stream) | — | json |
| Maya1 | [`fal-ai/maya`](https://fal.ai/models/fal-ai/maya) | — | audio |
| Minimax | [`fal-ai/minimax/preview/speech-2.5-hd`](https://fal.ai/models/fal-ai/minimax/preview/speech-2.5-hd) | Minimax | audio |
| Minimax | [`fal-ai/minimax/preview/speech-2.5-turbo`](https://fal.ai/models/fal-ai/minimax/preview/speech-2.5-turbo) | Minimax | audio |
| MiniMax Speech 2.6 [HD] | [`fal-ai/minimax/speech-2.6-hd`](https://fal.ai/models/fal-ai/minimax/speech-2.6-hd) | Minimax | audio |
| MiniMax Speech 2.6 [Turbo] | [`fal-ai/minimax/speech-2.6-turbo`](https://fal.ai/models/fal-ai/minimax/speech-2.6-turbo) | Minimax | audio |
| MiniMax Speech 2.8 [HD] | [`fal-ai/minimax/speech-2.8-hd`](https://fal.ai/models/fal-ai/minimax/speech-2.8-hd) | Minimax | audio |
| MiniMax Speech 2.8 [Turbo] | [`fal-ai/minimax/speech-2.8-turbo`](https://fal.ai/models/fal-ai/minimax/speech-2.8-turbo) | Minimax | audio |
| MiniMax Speech-02 HD | [`fal-ai/minimax/speech-02-hd`](https://fal.ai/models/fal-ai/minimax/speech-02-hd) | Minimax | audio |
| MiniMax Speech-02 Turbo | [`fal-ai/minimax/speech-02-turbo`](https://fal.ai/models/fal-ai/minimax/speech-02-turbo) | Minimax | audio |
| MiniMax Voice Cloning | [`fal-ai/minimax/voice-clone`](https://fal.ai/models/fal-ai/minimax/voice-clone) | Minimax | audio |
| MiniMax Voice Design | [`fal-ai/minimax/voice-design`](https://fal.ai/models/fal-ai/minimax/voice-design) | Minimax | audio |
| Orpheus TTS | [`fal-ai/orpheus-tts`](https://fal.ai/models/fal-ai/orpheus-tts) | — | audio |
| Qwen 3 TTS - Text to Speech [0.6B] | [`fal-ai/qwen-3-tts/text-to-speech/0.6b`](https://fal.ai/models/fal-ai/qwen-3-tts/text-to-speech/0.6b) | Alibaba | audio |
| Qwen 3 TTS - Text to Speech [1.7B] | [`fal-ai/qwen-3-tts/text-to-speech/1.7b`](https://fal.ai/models/fal-ai/qwen-3-tts/text-to-speech/1.7b) | Alibaba | audio |
| Qwen 3 TTS - Voice Design [1.7B] | [`fal-ai/qwen-3-tts/voice-design/1.7b`](https://fal.ai/models/fal-ai/qwen-3-tts/voice-design/1.7b) | Alibaba | audio |
| Vibevoice | [`fal-ai/vibevoice/0.5b`](https://fal.ai/models/fal-ai/vibevoice/0.5b) | — | audio |
| VibeVoice 1.5B | [`fal-ai/vibevoice`](https://fal.ai/models/fal-ai/vibevoice) | — | audio |
| VibeVoice 7B | [`fal-ai/vibevoice/7b`](https://fal.ai/models/fal-ai/vibevoice/7b) | — | audio |
| xAI Text to Speech | [`xai/tts/v1`](https://fal.ai/models/xai/tts/v1) | xAI | audio |
| Zonos2 Text to Speech | [`fal-ai/zonos2`](https://fal.ai/models/fal-ai/zonos2) | Zyphra | audio |

</details>

<details>
<summary><strong>vision</strong> — 33 models</summary>

| Model | Endpoint | Lab | Output |
| --- | --- | --- | --- |
| Florence 2 Large Caption | [`fal-ai/florence-2-large/caption`](https://fal.ai/models/fal-ai/florence-2-large/caption) | — | text |
| Florence 2 Large Detailed Caption | [`fal-ai/florence-2-large/detailed-caption`](https://fal.ai/models/fal-ai/florence-2-large/detailed-caption) | — | text |
| Florence 2 Large OCR | [`fal-ai/florence-2-large/ocr`](https://fal.ai/models/fal-ai/florence-2-large/ocr) | — | text |
| Florence 2 Large Region To Category | [`fal-ai/florence-2-large/region-to-category`](https://fal.ai/models/fal-ai/florence-2-large/region-to-category) | — | text |
| Florence-2 Large | [`fal-ai/florence-2-large/more-detailed-caption`](https://fal.ai/models/fal-ai/florence-2-large/more-detailed-caption) | — | text |
| Florence-2 Large | [`fal-ai/florence-2-large/region-to-description`](https://fal.ai/models/fal-ai/florence-2-large/region-to-description) | — | text |
| GOT OCR 2.0 | [`fal-ai/got-ocr/v2`](https://fal.ai/models/fal-ai/got-ocr/v2) | — | json |
| Isaac 0.1 | [`perceptron/isaac-01`](https://fal.ai/models/perceptron/isaac-01) | Perceptron | json |
| LLaVA v1.6 34B | [`fal-ai/llava-next`](https://fal.ai/models/fal-ai/llava-next) | — | json |
| Marlin | [`fal-ai/marlin`](https://fal.ai/models/fal-ai/marlin) | — | json |
| Marlin Find | [`fal-ai/marlin/find`](https://fal.ai/models/fal-ai/marlin/find) | — | json |
| Moondream | [`fal-ai/moondream/batched`](https://fal.ai/models/fal-ai/moondream/batched) | — | json |
| Moondream 3 Preview [Query] | [`fal-ai/moondream3-preview/query`](https://fal.ai/models/fal-ai/moondream3-preview/query) | Moondream | json |
| Moondream2 | [`fal-ai/moondream2`](https://fal.ai/models/fal-ai/moondream2) | Moondream | text |
| Moondream2 | [`fal-ai/moondream2/object-detection`](https://fal.ai/models/fal-ai/moondream2/object-detection) | Moondream | image |
| Moondream2 | [`fal-ai/moondream2/point-object-detection`](https://fal.ai/models/fal-ai/moondream2/point-object-detection) | Moondream | image |
| Moondream2 | [`fal-ai/moondream2/visual-query`](https://fal.ai/models/fal-ai/moondream2/visual-query) | Moondream | text |
| Moondream3 Preview [Caption] | [`fal-ai/moondream3-preview/caption`](https://fal.ai/models/fal-ai/moondream3-preview/caption) | Moondream | json |
| Moondream3 Preview [Detect] | [`fal-ai/moondream3-preview/detect`](https://fal.ai/models/fal-ai/moondream3-preview/detect) | Moondream | image |
| Moondream3 Preview [Point] | [`fal-ai/moondream3-preview/point`](https://fal.ai/models/fal-ai/moondream3-preview/point) | Moondream | image |
| MoonDreamNext | [`fal-ai/moondream-next`](https://fal.ai/models/fal-ai/moondream-next) | Moondream | text |
| MoonDreamNext Batch | [`fal-ai/moondream-next/batch`](https://fal.ai/models/fal-ai/moondream-next/batch) | Moondream | json |
| Nemotron Diffusion Vlm | [`fal-ai/nemotron-diffusion-vlm`](https://fal.ai/models/fal-ai/nemotron-diffusion-vlm) | NVIDIA | json |
| NSFW Checker | [`fal-ai/x-ailab/nsfw`](https://fal.ai/models/fal-ai/x-ailab/nsfw) | — | json |
| NSFW Filter | [`fal-ai/imageutils/nsfw`](https://fal.ai/models/fal-ai/imageutils/nsfw) | — | json |
| OpenRouter [Vision] | [`openrouter/router/vision`](https://fal.ai/models/openrouter/router/vision) | Openrouter | json |
| Sa2VA 4B Image | [`fal-ai/sa2va/4b/image`](https://fal.ai/models/fal-ai/sa2va/4b/image) | — | json |
| Sa2VA 4B Video | [`fal-ai/sa2va/4b/video`](https://fal.ai/models/fal-ai/sa2va/4b/video) | — | json |
| Sa2VA 8B Image | [`fal-ai/sa2va/8b/image`](https://fal.ai/models/fal-ai/sa2va/8b/image) | — | json |
| Sa2VA 8B Video | [`fal-ai/sa2va/8b/video`](https://fal.ai/models/fal-ai/sa2va/8b/video) | — | json |
| Sam 3 | [`fal-ai/sam-3/image/embed`](https://fal.ai/models/fal-ai/sam-3/image/embed) | — | text |
| Scene Finder | [`fal-ai/scene-finder`](https://fal.ai/models/fal-ai/scene-finder) | Fal | images |
| Video Understanding | [`fal-ai/video-understanding`](https://fal.ai/models/fal-ai/video-understanding) | — | text |

</details>

<details>
<summary><strong>audio-to-video</strong> — 20 models</summary>

| Model | Endpoint | Lab | Output |
| --- | --- | --- | --- |
| Avatars | [`veed/avatars/audio-to-video`](https://fal.ai/models/veed/avatars/audio-to-video) | Veed | video |
| Avatars Audio to Video | [`argil/avatars/audio-to-video`](https://fal.ai/models/argil/avatars/audio-to-video) | Argil AI | video |
| EchoMimic V3 | [`fal-ai/echomimic-v3`](https://fal.ai/models/fal-ai/echomimic-v3) | — | video |
| ElevenLabs Dubbing | [`fal-ai/elevenlabs/dubbing`](https://fal.ai/models/fal-ai/elevenlabs/dubbing) | ElevenLabs | video |
| Flashtalk | [`fal-ai/flashtalk`](https://fal.ai/models/fal-ai/flashtalk) | Soul AI Lab | video |
| Longcat Single Avatar | [`fal-ai/longcat-single-avatar/audio-to-video`](https://fal.ai/models/fal-ai/longcat-single-avatar/audio-to-video) | — | video |
| Longcat Single Avatar | [`fal-ai/longcat-single-avatar/image-audio-to-video`](https://fal.ai/models/fal-ai/longcat-single-avatar/image-audio-to-video) | — | video |
| LTX 2.0 Video Pro | [`fal-ai/ltx-2/audio-to-video`](https://fal.ai/models/fal-ai/ltx-2/audio-to-video) | Lightricks | video |
| Ltx 2.3 Quality | [`fal-ai/ltx-2.3-quality/audio-to-video`](https://fal.ai/models/fal-ai/ltx-2.3-quality/audio-to-video) | Lightricks | video |
| Ltx 2.3 Quality | [`fal-ai/ltx-2.3-quality/audio-to-video/lora`](https://fal.ai/models/fal-ai/ltx-2.3-quality/audio-to-video/lora) | Lightricks | video |
| LTX 2.3 Video Pro | [`fal-ai/ltx-2.3/audio-to-video`](https://fal.ai/models/fal-ai/ltx-2.3/audio-to-video) | Lightricks | video |
| LTX-2 19B | [`fal-ai/ltx-2-19b/audio-to-video`](https://fal.ai/models/fal-ai/ltx-2-19b/audio-to-video) | Lightricks | video |
| LTX-2 19B | [`fal-ai/ltx-2-19b/audio-to-video/lora`](https://fal.ai/models/fal-ai/ltx-2-19b/audio-to-video/lora) | Lightricks | video |
| LTX-2 19B Distilled | [`fal-ai/ltx-2-19b/distilled/audio-to-video`](https://fal.ai/models/fal-ai/ltx-2-19b/distilled/audio-to-video) | Lightricks | video |
| LTX-2 19B Distilled | [`fal-ai/ltx-2-19b/distilled/audio-to-video/lora`](https://fal.ai/models/fal-ai/ltx-2-19b/distilled/audio-to-video/lora) | Lightricks | video |
| LTX-2.3 22B | [`fal-ai/ltx-2.3-22b/audio-to-video`](https://fal.ai/models/fal-ai/ltx-2.3-22b/audio-to-video) | Lightricks | video |
| LTX-2.3 22B | [`fal-ai/ltx-2.3-22b/audio-to-video/lora`](https://fal.ai/models/fal-ai/ltx-2.3-22b/audio-to-video/lora) | Lightricks | video |
| LTX-2.3 22B Distilled | [`fal-ai/ltx-2.3-22b/distilled/audio-to-video`](https://fal.ai/models/fal-ai/ltx-2.3-22b/distilled/audio-to-video) | Lightricks | video |
| LTX-2.3 22B Distilled | [`fal-ai/ltx-2.3-22b/distilled/audio-to-video/lora`](https://fal.ai/models/fal-ai/ltx-2.3-22b/distilled/audio-to-video/lora) | Lightricks | video |
| Wan-2.2 Speech-to-Video 14B | [`fal-ai/wan/v2.2-14b/speech-to-video`](https://fal.ai/models/fal-ai/wan/v2.2-14b/speech-to-video) | Alibaba | video |

</details>

<details>
<summary><strong>text-to-3d</strong> — 11 models</summary>

| Model | Endpoint | Lab | Output |
| --- | --- | --- | --- |
| Hunyuan 3d | [`fal-ai/hunyuan-3d/v3.1/rapid/text-to-3d`](https://fal.ai/models/fal-ai/hunyuan-3d/v3.1/rapid/text-to-3d) | Tencent | file |
| Hunyuan 3D Pro Text to 3D | [`fal-ai/hunyuan-3d/v3.1/pro/text-to-3d`](https://fal.ai/models/fal-ai/hunyuan-3d/v3.1/pro/text-to-3d) | Tencent | file |
| Hunyuan Motion [0.46B] | [`fal-ai/hunyuan-motion/fast`](https://fal.ai/models/fal-ai/hunyuan-motion/fast) | Tencent | json |
| Hunyuan Motion [1B] | [`fal-ai/hunyuan-motion`](https://fal.ai/models/fal-ai/hunyuan-motion) | Tencent | json |
| Hunyuan3d V3 | [`fal-ai/hunyuan3d-v3/text-to-3d`](https://fal.ai/models/fal-ai/hunyuan3d-v3/text-to-3d) | Tencent | file |
| Hyper3D - Rodin V2.5 - Text to 3D | [`fal-ai/hyper3d/rodin/v2.5/text-to-3d`](https://fal.ai/models/fal-ai/hyper3d/rodin/v2.5/text-to-3d) | Hyper3D | file |
| Hyper3D - Rodin V2.5 - Text to 3D - Fast | [`fal-ai/hyper3d/rodin/v2.5/text-to-3d/fast`](https://fal.ai/models/fal-ai/hyper3d/rodin/v2.5/text-to-3d/fast) | Hyper3D | file |
| Meshy 6 | [`fal-ai/meshy/v6/text-to-3d`](https://fal.ai/models/fal-ai/meshy/v6/text-to-3d) | Meshy | file |
| Meshy 6 Preview | [`fal-ai/meshy/v6-preview/text-to-3d`](https://fal.ai/models/fal-ai/meshy/v6-preview/text-to-3d) | Meshy | file |
| Tripo H3.1 Text to 3D | [`tripo3d/h3.1/text-to-3d`](https://fal.ai/models/tripo3d/h3.1/text-to-3d) | Tripo3D | file |
| Tripo P1 Text to 3D | [`tripo3d/p1/text-to-3d`](https://fal.ai/models/tripo3d/p1/text-to-3d) | Tripo3D | file |

</details>

<details>
<summary><strong>speech-to-text</strong> — 10 models</summary>

| Model | Endpoint | Lab | Output |
| --- | --- | --- | --- |
| Cohere Transcribe | [`fal-ai/cohere-transcribe`](https://fal.ai/models/fal-ai/cohere-transcribe) | Cohere | json |
| ElevenLabs Speech to Text | [`fal-ai/elevenlabs/speech-to-text`](https://fal.ai/models/fal-ai/elevenlabs/speech-to-text) | ElevenLabs | json |
| ElevenLabs Speech to Text - Scribe V2 | [`fal-ai/elevenlabs/speech-to-text/scribe-v2`](https://fal.ai/models/fal-ai/elevenlabs/speech-to-text/scribe-v2) | ElevenLabs | json |
| Nemotron Asr Multilingual | [`nvidia/nemotron-asr-multilingual/asr`](https://fal.ai/models/nvidia/nemotron-asr-multilingual/asr) | NVIDIA | json |
| Pipecat's Smart Turn model | [`fal-ai/smart-turn`](https://fal.ai/models/fal-ai/smart-turn) | — | json |
| Speech-to-Text | [`fal-ai/speech-to-text`](https://fal.ai/models/fal-ai/speech-to-text) | — | json |
| Speech-To-text | [`fal-ai/speech-to-text/stream`](https://fal.ai/models/fal-ai/speech-to-text/stream) | — | json |
| Speech-to-Text | [`fal-ai/speech-to-text/turbo`](https://fal.ai/models/fal-ai/speech-to-text/turbo) | — | json |
| Speech-to-Text | [`fal-ai/speech-to-text/turbo/stream`](https://fal.ai/models/fal-ai/speech-to-text/turbo/stream) | — | json |
| Wizper (Whisper v3 -- fal.ai edition) | [`fal-ai/wizper`](https://fal.ai/models/fal-ai/wizper) | — | json |

</details>

<details>
<summary><strong>3d-to-3d</strong> — 7 models</summary>

| Model | Endpoint | Lab | Output |
| --- | --- | --- | --- |
| Hunyuan 3D Part Splitter | [`fal-ai/hunyuan-3d/v3.1/part`](https://fal.ai/models/fal-ai/hunyuan-3d/v3.1/part) | Tencent | json |
| Hunyuan 3D Smart Topology | [`fal-ai/hunyuan-3d/v3.1/smart-topology`](https://fal.ai/models/fal-ai/hunyuan-3d/v3.1/smart-topology) | Tencent | file |
| Meshy 5 Remesh | [`fal-ai/meshy/v5/remesh`](https://fal.ai/models/fal-ai/meshy/v5/remesh) | Meshy | file |
| Meshy 5 Retexture | [`fal-ai/meshy/v5/retexture`](https://fal.ai/models/fal-ai/meshy/v5/retexture) | Meshy | file |
| Meshy Rigging | [`fal-ai/meshy/rigging`](https://fal.ai/models/fal-ai/meshy/rigging) | Meshy | json |
| Meshy Rigging Multi Animation | [`fal-ai/meshy/rigging/multi-animation`](https://fal.ai/models/fal-ai/meshy/rigging/multi-animation) | Meshy | json |
| Sam 3 | [`fal-ai/sam-3/3d-align`](https://fal.ai/models/fal-ai/sam-3/3d-align) | — | file |

</details>

<details>
<summary><strong>json</strong> — 6 models</summary>

| Model | Endpoint | Lab | Output |
| --- | --- | --- | --- |
| Ffmpeg Api | [`fal-ai/ffmpeg-api/loudnorm`](https://fal.ai/models/fal-ai/ffmpeg-api/loudnorm) | — | audio |
| FFmpeg API Metadata | [`fal-ai/ffmpeg-api/metadata`](https://fal.ai/models/fal-ai/ffmpeg-api/metadata) | — | json |
| FFmpeg API Waveform | [`fal-ai/ffmpeg-api/waveform`](https://fal.ai/models/fal-ai/ffmpeg-api/waveform) | — | json |
| Omnilottie | [`fal-ai/omnilottie`](https://fal.ai/models/fal-ai/omnilottie) | — | json |
| Omnilottie | [`fal-ai/omnilottie/image-to-lottie`](https://fal.ai/models/fal-ai/omnilottie/image-to-lottie) | — | json |
| Omnilottie | [`fal-ai/omnilottie/video-to-lottie`](https://fal.ai/models/fal-ai/omnilottie/video-to-lottie) | — | json |

</details>

<details>
<summary><strong>llm</strong> — 5 models</summary>

| Model | Endpoint | Lab | Output |
| --- | --- | --- | --- |
| Bytedance Seed V2 Mini | [`fal-ai/bytedance/seed/v2/mini`](https://fal.ai/models/fal-ai/bytedance/seed/v2/mini) | Bytedance | json |
| Nemotron 3 Nano Omni | [`nvidia/nemotron-3-nano-omni`](https://fal.ai/models/nvidia/nemotron-3-nano-omni) | NVIDIA | json |
| OpenRouter | [`openrouter/router`](https://fal.ai/models/openrouter/router) | Openrouter | json |
| OpenRouter [Enterprise] | [`openrouter/router/enterprise`](https://fal.ai/models/openrouter/router/enterprise) | Openrouter | json |
| Video Prompt Generator | [`fal-ai/video-prompt-generator`](https://fal.ai/models/fal-ai/video-prompt-generator) | — | text |

</details>

<details>
<summary><strong>video-to-audio</strong> — 5 models</summary>

| Model | Endpoint | Lab | Output |
| --- | --- | --- | --- |
| Kling Video | [`fal-ai/kling-video/video-to-audio`](https://fal.ai/models/fal-ai/kling-video/video-to-audio) | Kling | video |
| Mirelo SFX | [`mirelo-ai/sfx-v1/video-to-audio`](https://fal.ai/models/mirelo-ai/sfx-v1/video-to-audio) | Mirelo AI | audio |
| Mirelo SFX V1.5 | [`mirelo-ai/sfx-v1.5/video-to-audio`](https://fal.ai/models/mirelo-ai/sfx-v1.5/video-to-audio) | Mirelo AI | audio |
| Sam Audio | [`fal-ai/sam-audio/visual-separate`](https://fal.ai/models/fal-ai/sam-audio/visual-separate) | — | json |
| V1.1 | [`sonilo/v1.1/video-to-music`](https://fal.ai/models/sonilo/v1.1/video-to-music) | Sonilo | audio |

</details>

<details>
<summary><strong>text-to-json</strong> — 4 models</summary>

| Model | Endpoint | Lab | Output |
| --- | --- | --- | --- |
| Fibo | [`bria/fibo/generate/structured_prompt`](https://fal.ai/models/bria/fibo/generate/structured_prompt) | Bria AI | json |
| Fibo Edit [Structured Instruction] | [`bria/fibo-edit/edit/structured_instruction`](https://fal.ai/models/bria/fibo-edit/edit/structured_instruction) | Bria AI | json |
| Fibo Lite | [`bria/fibo-lite/generate/structured_prompt`](https://fal.ai/models/bria/fibo-lite/generate/structured_prompt) | Bria AI | json |
| Fibo Lite | [`bria/fibo-lite/generate/structured_prompt/lite`](https://fal.ai/models/bria/fibo-lite/generate/structured_prompt/lite) | Bria AI | json |

</details>

<details>
<summary><strong>video-to-text</strong> — 3 models</summary>

| Model | Endpoint | Lab | Output |
| --- | --- | --- | --- |
| Nemotron 3 Nano Omni | [`nvidia/nemotron-3-nano-omni/video`](https://fal.ai/models/nvidia/nemotron-3-nano-omni/video) | NVIDIA | json |
| OpenRouter [Video] | [`openrouter/router/video`](https://fal.ai/models/openrouter/router/video) | Openrouter | json |
| OpenRouter [Video][Enterprise] | [`openrouter/router/video/enterprise`](https://fal.ai/models/openrouter/router/video/enterprise) | Openrouter | json |

</details>

<details>
<summary><strong>audio-to-text</strong> — 2 models</summary>

| Model | Endpoint | Lab | Output |
| --- | --- | --- | --- |
| Nemotron 3 Nano Omni | [`nvidia/nemotron-3-nano-omni/audio`](https://fal.ai/models/nvidia/nemotron-3-nano-omni/audio) | NVIDIA | json |
| Silero VAD | [`fal-ai/silero-vad`](https://fal.ai/models/fal-ai/silero-vad) | — | json |

</details>

<details>
<summary><strong>speech-to-speech</strong> — 2 models</summary>

| Model | Endpoint | Lab | Output |
| --- | --- | --- | --- |
| Chatterbox | [`fal-ai/chatterbox/speech-to-speech`](https://fal.ai/models/fal-ai/chatterbox/speech-to-speech) | Resemble AI | audio |
| Chatterboxhd | [`resemble-ai/chatterboxhd/speech-to-speech`](https://fal.ai/models/resemble-ai/chatterboxhd/speech-to-speech) | Resemble AI | audio |

</details>

<details>
<summary><strong>unknown</strong> — 2 models</summary>

| Model | Endpoint | Lab | Output |
| --- | --- | --- | --- |
| OpenRouter [Audio] | [`openrouter/router/audio`](https://fal.ai/models/openrouter/router/audio) | Openrouter | json |
| Workflow Utilities Interleave Video | [`fal-ai/workflow-utilities/interleave-video`](https://fal.ai/models/fal-ai/workflow-utilities/interleave-video) | — | video |

</details>

<details>
<summary><strong>image-to-json</strong> — 1 model</summary>

| Model | Endpoint | Lab | Output |
| --- | --- | --- | --- |
| Bagel | [`fal-ai/bagel/understand`](https://fal.ai/models/fal-ai/bagel/understand) | — | json |

</details>

<details>
<summary><strong>image-to-text</strong> — 1 model</summary>

| Model | Endpoint | Lab | Output |
| --- | --- | --- | --- |
| Nemotron 3 Nano Omni | [`nvidia/nemotron-3-nano-omni/vision`](https://fal.ai/models/nvidia/nemotron-3-nano-omni/vision) | NVIDIA | json |

</details>

<details>
<summary><strong>workflow</strong> — 1 model</summary>

| Model | Endpoint | Lab | Output |
| --- | --- | --- | --- |
| Workflow Utilities Pick Image By Index | [`fal-ai/workflow-utilities/pick-image-by-index`](https://fal.ai/models/fal-ai/workflow-utilities/pick-image-by-index) | — | images |

</details>

<!-- END GENERATED MODEL LIST -->

## Registry Maintenance

The auto-generated nodes come from the committed snapshot at `data/fal_registry.json`. To refresh it from fal's platform API:

```bash
python scripts/build_registry.py --out data/fal_registry.json --since-days 0
```

Then regenerate the model list in this README:

```bash
python scripts/build_readme.py
```

A weekly GitHub Action runs the refresh automatically and opens a PR with the updated registry, so the catalog stays current without manual work.

## Troubleshooting

If you encounter any errors during installation or usage, try the following:

1. Ensure you have the latest version of ComfyUI installed
2. Update this custom node package:
   ```
   cd custom_nodes/ComfyUI-fal-API
   git pull
   pip install -r requirements.txt
   ```
3. If you're using ComfyUI Windows Portable, you may need to install fal-client manually:
   ```
   ComfyUI_windows_portable>.\python_embeded\python.exe -m pip install fal-client
   ```
4. **Dynamic nodes not appearing?** Check the ComfyUI console for a line like `Registered N dynamic fal nodes` at startup. If it says the nodes are disabled, remove `enabled = false` from the `[dynamic_nodes]` section of your `config.ini` (and check the `categories` filter isn't excluding what you're looking for). Any registry loading error is also printed there.
5. **`VIDEO` output is `None` or video sockets are missing?** Update ComfyUI — native `VIDEO`/`AUDIO` types require a recent ComfyUI version.
6. **API calls failing?** As of 2.0, failed fal requests raise visible errors that include fal's actual error message (validation issues, content policy, quota). Read the error text in ComfyUI — it usually tells you exactly which parameter to fix.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## Support

If you encounter any issues or have questions, please open an issue on the [GitHub repository](https://github.com/gokayfem/ComfyUI-fal-API/issues).

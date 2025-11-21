# ComfyUI-fal-API

Custom nodes for using Flux models with  fal API in ComfyUI with only one API Key for all.

## Table of Contents

- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Available Nodes](#available-nodes)
  - [Image Generation](#image-generation)
  - [Video Generation](#video-generation)
  - [Language Models (LLMs)](#language-models-llms)
  - [Vision Language Models (VLMs)](#vision-language-models-vlms)
- [Troubleshooting](#troubleshooting)
- [License](#license)

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

2. Open the `config.ini` file inside `custom_nodes/ComfyUI-fal-API`

3. Replace `<your_fal_api_key_here>` with your actual fal API key:
   ```ini
   [API]
   FAL_KEY = your_actual_api_key
   ```

4. Alternatively, you can set the FAL_KEY environment variable:
   ```bash
   export FAL_KEY=your_actual_api_key
   ```

## Usage

After installation and configuration, restart ComfyUI. The new nodes will be available in the node browser under the "FAL" category.

## Available Nodes

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
- **SeedEdit 3.0 (fal)**: Use SeedEdit 3.0 to edit images
- **Seedream 4.0 Edit (fal)**: Use Seedream 4.0 to edit images
- **Nano Banana Text-to-Image (fal)**: Use Nano Banana to generate images
- **Nano Banana Edit (fal)**: Use Nano Banana to edit images
- **Reve Text-to-Image (fal)**: Use Reve's image model to generate images
- **Dreamina v3.1 Text-to-Image (fal)**: Use Dreamina v3.1 to generate images

### Video Generation

- **Infinity Star Text-to-Video (fal)**: Generate videos using Infinity Star and text prompts
- **Kling Video Generation (fal)**: Generate videos using the Kling model
- **Kling Pro v1.0 Video Generation (fal)**: Original version of Kling Pro for video generation
- **Kling Pro v1.6 Video Generation (fal)**: Latest version of Kling Pro with improved quality
- **Kling Master v2.0 Video Generation (fal)**: Advanced video generation with Kling Master
- **Kling Pro 2.1 Video Generation (fal)**: Video Generation with Kling Pro with First Frame Last Frame support
- **Kling v2.5 Turbo Pro Image-to-Video (fal)**: Video Generation with Kling Turbo with First Frame Last Frame support
- **Krea Wan 14b Video-to-Video (fal)**: Video-to-Video generation using Krea Wan 14b model
- **Runway Gen3 Image-to-Video (fal)**: Convert images to videos using Runway Gen3
- **Luma Dream Machine (fal)**: Create videos with Luma Dream Machine
- **MiniMax Video Generation (fal)**: Generate videos using MiniMax model
- **MiniMax Text-to-Video (fal)**: Create videos from text prompts using MiniMax
- **MiniMax Subject Reference (fal)**: Generate videos with subject reference using MiniMax
- **Pixverse Swap (fal)**: Swap a person, object, or background in a video using Pixverse Swap.
- **Google Veo2 Image-to-Video (fal)**: Convert images to videos using Google's Veo2 model
- **Veo 3.1 First-Last-Frame-to-Video (fal)**: First Frame - Last Frame (Optional) video generation using the full VEO 3.1 model
- **Veo 3.1 Fast First-Last-Frame-to-Video (fal)**: First Frame - Last Frame (Optional) video generation using the fast VEO 3.1 model
- **Wan Pro Image-to-Video (fal)**: High-quality video generation with Wan Pro model
- **Wan 2.5 Preview Image-to-Video (fal)**: Image-to-video generation with the latest Wan 2.5 preview model
- **Wan VACE Video Edit (fal)**: Video + Reference Images to video generation with Wan VACE.
- **Wan 2.2 14b Animate: Replace Character (fal)**: Animate video content by replacing the foreground character with a new or augmented character using Wan 2.2 14b.
- **Wan 2.2 14b Animate: Move Character (fal)**: Animate video content by moving the foreground character within the scene using Wan 2.2 14b.
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

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

If you encounter any issues or have questions, please open an issue on the [GitHub repository](https://github.com/gokayfem/ComfyUI-fal-API/issues).

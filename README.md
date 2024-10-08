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
   git clone https://github.com/gokayfem/ComfyUI-FLUX-fal-API.git
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Configuration

1. Get your fal API key from [fal.ai](https://fal.ai/dashboard/keys)

2. Open the `config.ini` file in the root directory of this project

3. Replace `<your_fal_api_key_here>` with your actual fal API key:
   ```ini
   [API]
   FAL_KEY = your_actual_api_key
   ```

## Usage

After installation and configuration, restart ComfyUI. The new nodes will be available in the node browser under the "FAL" category.

## Available Nodes

### Image Generation

- **Flux Pro (fal)**: Generate high-quality images using the Flux Pro model
- **Flux Dev (fal)**: Use the development version of Flux for image generation
- **Flux Schnell (fal)**: Fast image generation with Flux Schnell
- **Flux Pro 1.1 (fal)**: Latest version of Flux Pro for image generation
- **Flux General (fal)**: ControlNets, Ipadapters, Loras for Flux Dev

### Video Generation

- **Kling Video Generation (fal)**: Generate videos using the Kling model
- **Kling Pro Video Generation (fal)**: Advanced video generation with Kling Pro
- **Runway Gen3 Image-to-Video (fal)**: Convert images to videos using Runway Gen3
- **Luma Dream Machine (fal)**: Create videos with Luma Dream Machine
- **Load Video from URL**: Load and process videos from a given URL

### Language Models (LLMs)

- **LLM (fal)**: Large Language Model for text generation and processing
  - Available models:
    - google/gemini-flash-1.5-8b
    - anthropic/claude-3.5-sonnet
    - anthropic/claude-3-haiku
    - google/gemini-pro-1.5
    - google/gemini-flash-1.5
    - meta-llama/llama-3.2-1b-instruct
    - meta-llama/llama-3.2-3b-instruct
    - meta-llama/llama-3.1-8b-instruct
    - meta-llama/llama-3.1-70b-instruct
    - openai/gpt-4o-mini
    - openai/gpt-4o

### Vision Language Models (VLMs)

- **VLM (fal)**: Vision Language Model for image understanding and text generation
  - Available models:
    - google/gemini-flash-1.5-8b
    - anthropic/claude-3.5-sonnet
    - anthropic/claude-3-haiku
    - google/gemini-pro-1.5
    - google/gemini-flash-1.5
    - openai/gpt-4o
  - Supports various tasks such as image captioning, visual question answering, and more

## Troubleshooting

If you encounter any errors during installation or usage, try the following:

1. Ensure you have the latest version of ComfyUI installed
2. Update this custom node package:
   ```
   cd custom_nodes/ComfyUI-FLUX-fal-API
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

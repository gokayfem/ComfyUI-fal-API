from .fal_utils import ApiHandler, FalConfig, ImageUtils

# Initialize FalConfig
fal_config = FalConfig()


class VLMNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "User prompt sent to the model.",
                    },
                ),
                "model": (
                    [
                        "google/gemini-2.5-flash",
                        "anthropic/claude-sonnet-4.5",
                        "openai/gpt-4o",
                        "qwen/qwen3-vl-235b-a22b-instruct",
                        "x-ai/grok-4-fast",
                        "Custom",
                    ],
                    {
                        "default": "google/gemini-2.5-flash",
                        "tooltip": "Vision model to use. Select 'Custom' to type any OpenRouter model id in custom_model_name.",
                    },
                ),
                "system_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Optional system prompt to steer the model's behavior.",
                    },
                ),
                "image": (
                    "IMAGE",
                    {
                        "tooltip": "Image(s) for the model to analyze. Batches are sent as multiple images.",
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.1,
                        "tooltip": "Sampling temperature. Lower is more deterministic.",
                    },
                ),
                "reasoning": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Request reasoning from the model.",
                    },
                ),
            },
            "optional": {
                "max_tokens": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 100000,
                        "tooltip": "Maximum output tokens. 0 uses the model default.",
                    },
                ),
                "custom_model_name": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "OpenRouter model id used when model is set to 'Custom'.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_text"
    CATEGORY = "FAL/VLM"

    def generate_text(self, prompt, model, system_prompt, image, temperature, reasoning, max_tokens=0, custom_model_name=""):
        try:
            # Handle custom model selection
            if model == "Custom":
                if not custom_model_name or custom_model_name.strip() == "":
                    return ApiHandler.handle_text_generation_error(
                        "Custom", "Custom model name is required when 'Custom' is selected"
                    )
                model = custom_model_name.strip()

            # Upload single image or batch and collect URLs
            image_urls = ImageUtils.prepare_images(image)
            if not image_urls:
                return ApiHandler.handle_text_generation_error(
                    model, "Failed to upload image(s)"
                )

            arguments = {
                "model": model,
                "prompt": prompt,
                "system_prompt": system_prompt,
                "image_urls": image_urls,
                "temperature": temperature,
                "reasoning": reasoning,
                "stream": False,
            }

            # Only include max_tokens if it's greater than 0
            if max_tokens > 0:
                arguments["max_tokens"] = max_tokens

            result = ApiHandler.submit_and_get_result(
                "openrouter/router/vision", arguments
            )
            return (result["output"],)
        except Exception as e:
            return ApiHandler.handle_text_generation_error(model, e)


# Node class mappings
NODE_CLASS_MAPPINGS = {
    "VLM_fal": VLMNode,
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "VLM_fal": "VLM (fal)",
}

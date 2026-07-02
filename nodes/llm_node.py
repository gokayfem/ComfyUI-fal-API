from .fal_utils import ApiHandler, FalConfig

# Initialize FalConfig
fal_config = FalConfig()


class LLMNode:
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
                        "openai/gpt-4.1",
                        "openai/gpt-oss-120b",
                        "meta-llama/llama-4-maverick",
                        "Custom",
                    ],
                    {
                        "default": "google/gemini-2.5-flash",
                        "tooltip": "Model to use. Select 'Custom' to type any OpenRouter model id in custom_model_name.",
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
                        "tooltip": "Request the model's reasoning trace (returned on the 'reasoning' output).",
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

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("output", "reasoning",)
    FUNCTION = "generate_text"
    CATEGORY = "FAL/LLM"

    def generate_text(self, prompt, model, system_prompt, temperature, reasoning, max_tokens=0, custom_model_name=""):
        try:
            # Handle custom model selection
            if model == "Custom":
                if not custom_model_name or custom_model_name.strip() == "":
                    # Raises a clear FalApiError
                    ApiHandler.handle_text_generation_error(
                        "Custom", "Custom model name is required when 'Custom' is selected"
                    )
                model = custom_model_name.strip()

            arguments = {
                "model": model,
                "prompt": prompt,
                "system_prompt": system_prompt,
                "temperature": temperature,
                "reasoning": reasoning,
                "stream": False,
            }

            # Only include max_tokens if it's greater than 0
            if max_tokens > 0:
                arguments["max_tokens"] = max_tokens

            result = ApiHandler.submit_and_get_result("openrouter/router", arguments)

            # Extract output and reasoning
            output_text = result.get("output", "")
            reasoning_text = result.get("reasoning", "")

            return (output_text, reasoning_text)
        except Exception as e:
            # Raises a clear FalApiError (passes an existing FalApiError through
            # unchanged, so the custom-model validation error is not re-wrapped)
            return ApiHandler.handle_text_generation_error(model, e)


# Node class mappings
NODE_CLASS_MAPPINGS = {
    "LLM_fal": LLMNode,
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "LLM_fal": "LLM (fal)",
}

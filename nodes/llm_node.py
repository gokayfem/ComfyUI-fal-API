from .fal_utils import ApiHandler, FalConfig

# Initialize FalConfig
fal_config = FalConfig()


class LLMNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "model": (
                    [
                        "google/gemini-2.5-flash",
                        "anthropic/claude-sonnet-4.5",
                        "openai/gpt-4.1",
                        "openai/gpt-oss-120b",
                        "meta-llama/llama-4-maverick",
                        "Custom",
                    ],
                    {"default": "google/gemini-2.5-flash"},
                ),
                "system_prompt": ("STRING", {"default": "", "multiline": True}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "reasoning": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "max_tokens": ("INT", {"default": 0, "min": 0, "max": 100000}),
                "custom_model_name": ("STRING", {"default": "", "multiline": False}),
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
                    error_result = ApiHandler.handle_text_generation_error(
                        "Custom", "Custom model name is required when 'Custom' is selected"
                    )
                    return (error_result[0], "")
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
            error_result = ApiHandler.handle_text_generation_error(model, str(e))
            return (error_result[0], "")


# Node class mappings
NODE_CLASS_MAPPINGS = {
    "LLM_fal": LLMNode,
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "LLM_fal": "LLM (fal)",
}

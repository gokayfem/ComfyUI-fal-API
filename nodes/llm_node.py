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
                        "google/gemini-flash-1.5-8b",
                        "anthropic/claude-3.5-sonnet",
                        "anthropic/claude-3-haiku",
                        "google/gemini-pro-1.5",
                        "google/gemini-flash-1.5",
                        "meta-llama/llama-3.2-1b-instruct",
                        "meta-llama/llama-3.2-3b-instruct",
                        "meta-llama/llama-3.1-8b-instruct",
                        "meta-llama/llama-3.1-70b-instruct",
                        "openai/gpt-4o-mini",
                        "openai/gpt-4o",
                    ],
                    {"default": "google/gemini-flash-1.5-8b"},
                ),
                "system_prompt": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_text"
    CATEGORY = "FAL/LLM"

    def generate_text(self, prompt, model, system_prompt):
        try:
            arguments = {
                "model": model,
                "prompt": prompt,
                "system_prompt": system_prompt,
            }

            result = ApiHandler.submit_and_get_result("fal-ai/any-llm", arguments)
            return (result["output"],)
        except Exception as e:
            return ApiHandler.handle_text_generation_error(model, str(e))


# Node class mappings
NODE_CLASS_MAPPINGS = {
    "LLM_fal": LLMNode,
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "LLM_fal": "LLM (fal)",
}

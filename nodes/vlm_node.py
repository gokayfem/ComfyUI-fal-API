from .fal_utils import ApiHandler, FalConfig, ImageUtils

# Initialize FalConfig
fal_config = FalConfig()


class VLMNode:
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
                        "openai/gpt-4o",
                    ],
                    {"default": "google/gemini-flash-1.5-8b"},
                ),
                "system_prompt": ("STRING", {"default": "", "multiline": True}),
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_text"
    CATEGORY = "FAL/VLM"

    def generate_text(self, prompt, model, system_prompt, image):
        try:
            # Upload the image using ImageUtils
            image_url = ImageUtils.upload_image(image)
            if not image_url:
                return ApiHandler.handle_text_generation_error(
                    model, "Failed to upload image"
                )

            arguments = {
                "model": model,
                "prompt": prompt,
                "system_prompt": system_prompt,
                "image_url": image_url,
            }

            result = ApiHandler.submit_and_get_result(
                "fal-ai/any-llm/vision", arguments
            )
            return (result["output"],)
        except Exception as e:
            return ApiHandler.handle_text_generation_error(model, str(e))


# Node class mappings
NODE_CLASS_MAPPINGS = {
    "VLM_fal": VLMNode,
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "VLM_fal": "VLM (fal)",
}

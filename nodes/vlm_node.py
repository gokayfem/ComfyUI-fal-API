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
                        "google/gemini-2.5-flash",
                        "anthropic/claude-sonnet-4.5",
                        "openai/gpt-4o",
                        "qwen/qwen3-vl-235b-a22b-instruct",
                        "x-ai/grok-4-fast",
                        "Custom",
                    ],
                    {"default": "google/gemini-2.5-flash"},
                ),
                "system_prompt": ("STRING", {"default": "", "multiline": True}),
                "image": ("IMAGE",),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "reasoning": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "max_tokens": ("INT", {"default": 0, "min": 0, "max": 100000}),
                "custom_model_name": ("STRING", {"default": "", "multiline": False}),
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

            # Handle multiple images - image can be a batch
            image_urls = []

            # Check if image is a batch (4D tensor) or single image (3D tensor)
            if len(image.shape) == 4:
                # Batch of images
                for i in range(image.shape[0]):
                    single_image = image[i:i+1]
                    image_url = ImageUtils.upload_image(single_image)
                    if not image_url:
                        return ApiHandler.handle_text_generation_error(
                            model, f"Failed to upload image {i+1}"
                        )
                    image_urls.append(image_url)
            else:
                # Single image
                image_url = ImageUtils.upload_image(image)
                if not image_url:
                    return ApiHandler.handle_text_generation_error(
                        model, "Failed to upload image"
                    )
                image_urls.append(image_url)

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
            return ApiHandler.handle_text_generation_error(model, str(e))


# Node class mappings
NODE_CLASS_MAPPINGS = {
    "VLM_fal": VLMNode,
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "VLM_fal": "VLM (fal)",
}

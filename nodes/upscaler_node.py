from .fal_utils import ApiHandler, FalConfig, ImageUtils, ResultProcessor

# Initialize FalConfig
fal_config = FalConfig()


class UpscalerNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "upscale_factor": (
                    "FLOAT",
                    {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.5},
                ),
                "negative_prompt": (
                    "STRING",
                    {
                        "default": "(worst quality, low quality, normal quality:2)",
                        "multiline": True,
                    },
                ),
                "creativity": (
                    "FLOAT",
                    {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "resemblance": (
                    "FLOAT",
                    {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "guidance_scale": (
                    "FLOAT",
                    {"default": 4.0, "min": 1.0, "max": 20.0, "step": 0.5},
                ),
                "num_inference_steps": ("INT", {"default": 18, "min": 1, "max": 100}),
                "enable_safety_checker": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "seed": ("INT", {"default": -1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_upscaled_image"
    CATEGORY = "FAL/Image"

    def generate_upscaled_image(
        self,
        image,
        upscale_factor,
        negative_prompt,
        creativity,
        resemblance,
        guidance_scale,
        num_inference_steps,
        enable_safety_checker,
        seed=-1,
    ):
        try:
            # Upload the image using ImageUtils
            image_url = ImageUtils.upload_image(image)
            if not image_url:
                return ApiHandler.handle_image_generation_error(
                    "clarity-upscaler", "Failed to upload image for upscaling"
                )

            arguments = {
                "image_url": image_url,
                "prompt": "masterpiece, best quality, highres",
                "upscale_factor": upscale_factor,
                "negative_prompt": negative_prompt,
                "creativity": creativity,
                "resemblance": resemblance,
                "guidance_scale": guidance_scale,
                "num_inference_steps": num_inference_steps,
                "enable_safety_checker": enable_safety_checker,
            }

            if seed != -1:
                arguments["seed"] = seed

            result = ApiHandler.submit_and_get_result(
                "fal-ai/clarity-upscaler", arguments
            )
            return ResultProcessor.process_image_result(result)
        except Exception as e:
            return ApiHandler.handle_image_generation_error("clarity-upscaler", str(e))


# Node class mappings
NODE_CLASS_MAPPINGS = {
    "Upscaler_fal": UpscalerNode,
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "Upscaler_fal": "Clarity Upscaler (fal)",
}

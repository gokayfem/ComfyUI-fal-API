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


class SeedvrUpscalerNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "upscale_factor": (
                    "FLOAT",
                    {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.5},
                ),
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
        seed=-1,
    ):
        try:
            # Upload the image using ImageUtils
            image_url = ImageUtils.upload_image(image)
            if not image_url:
                return ApiHandler.handle_image_generation_error(
                    "seedvr-upscaler", "Failed to upload image for upscaling"
                )

            arguments = {
                "image_url": image_url,
                "upscale_factor": upscale_factor,
            }

            if seed != -1:
                arguments["seed"] = seed

            result = ApiHandler.submit_and_get_result(
                "fal-ai/seedvr/upscale/image", arguments
            )
            return ResultProcessor.process_single_image_result(result)
        except Exception as e:
            return ApiHandler.handle_image_generation_error("seedvr-upscaler", str(e))

class BriaVideoIncreaseResolutionNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
              
                "upscale_factor": (
                    "INT",
                    {"default": 2, "min": 2, "max": 4, "step": 2},
                ),
            },
            "optional": {
                "video": ("VIDEO",),
                "input_video_url": ("STRING", {"default": ""}),
                "output_container_and_codec": ("STRING", {"default": "mp4_h264", "options": ["mp4_h264", "mp4_h265","mov_h265","mov_proresks", "webm_vp9","mkv_h265","mkv_h265", "mkv_vp9", "gif"]}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_url",)
    FUNCTION = "generate_upscaled_video"
    CATEGORY = "FAL/Image"

    def generate_upscaled_video(
        self,
        video =None,
        input_video_url=None,
        upscale_factor=2,
        output_container_and_codec="mp4_h264"
    ):
        try:
            
            video_url = input_video_url
            if video is not None:
                video_url = ImageUtils.upload_file(video.get_stream_source())
            if not video_url or video_url=="":
                return ApiHandler.handle_video_generation_error(
                    "bria-video-increase-resolution", "Failed to upload video for upscaling. No video URL provided, or Video provided."
                )
            arguments={
        "video_url": video_url,
        "desired_increase": str(upscale_factor),
        "output_container_and_codec": output_container_and_codec
    }
            print(arguments)
            result = ApiHandler.submit_and_get_result(
                "bria/video/increase-resolution", arguments
            )
            return (result["video"]["url"],)
        except Exception as e:
            return ApiHandler.handle_video_generation_error(
                "bria/video/increase-resolution", str(e)
            )


# Node class mappings
NODE_CLASS_MAPPINGS = {
    "Upscaler_fal": UpscalerNode,
    "Seedvr_Upscaler_fal": SeedvrUpscalerNode,
    "Bria_Video_Increase_Resolution_fal": BriaVideoIncreaseResolutionNode,
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "Upscaler_fal": "Clarity Upscaler (fal)",
    "Seedvr_Upscaler_fal": "Seedvr Upscaler (fal)",
    "Bria_Video_Increase_Resolution_fal": "Bria Video Increase Resolution (fal)",
}

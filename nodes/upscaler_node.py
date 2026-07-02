from .fal_utils import ApiHandler, FalConfig, ImageUtils, ResultProcessor

# Initialize FalConfig
fal_config = FalConfig()


class UpscalerNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image to upscale."}),
                "upscale_factor": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "min": 1.0,
                        "max": 4.0,
                        "step": 0.5,
                        "tooltip": "How much to enlarge the image (1x-4x).",
                    },
                ),
                "negative_prompt": (
                    "STRING",
                    {
                        "default": "(worst quality, low quality, normal quality:2)",
                        "multiline": True,
                        "tooltip": "Concepts to avoid during the creative upscale.",
                    },
                ),
                "creativity": (
                    "FLOAT",
                    {
                        "default": 0.35,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Higher values allow the model to invent more detail.",
                    },
                ),
                "resemblance": (
                    "FLOAT",
                    {
                        "default": 0.6,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Higher values keep the result closer to the input image.",
                    },
                ),
                "guidance_scale": (
                    "FLOAT",
                    {
                        "default": 4.0,
                        "min": 1.0,
                        "max": 20.0,
                        "step": 0.5,
                        "tooltip": "Classifier-free guidance scale for the diffusion pass.",
                    },
                ),
                "num_inference_steps": (
                    "INT",
                    {
                        "default": 18,
                        "min": 1,
                        "max": 100,
                        "tooltip": "Number of diffusion steps; more steps is slower but can add detail.",
                    },
                ),
                "enable_safety_checker": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Filter potentially unsafe output images.",
                    },
                ),
            },
            "optional": {
                "seed": (
                    "INT",
                    {
                        "default": -1,
                        "tooltip": "Random seed for reproducibility. -1 uses a random seed.",
                    },
                ),
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
            # Upload the image using ImageUtils (raises on failure)
            image_url = ImageUtils.upload_image(image)

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
            return ApiHandler.handle_image_generation_error("clarity-upscaler", e)


class SeedvrUpscalerNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image to upscale."}),
                "upscale_factor": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "min": 1.0,
                        "max": 4.0,
                        "step": 0.5,
                        "tooltip": "How much to enlarge the image (1x-4x).",
                    },
                ),
            },
            "optional": {
                "seed": (
                    "INT",
                    {
                        "default": -1,
                        "tooltip": "Random seed for reproducibility. -1 uses a random seed.",
                    },
                ),
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
            # Upload the image using ImageUtils (raises on failure)
            image_url = ImageUtils.upload_image(image)

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
            return ApiHandler.handle_image_generation_error("seedvr-upscaler", e)


class SeedvrUpscaleVideoNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "upscale_factor": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "min": 0.00,
                        "max": 5.0,
                        "step": 0.01,
                        "tooltip": "Upscaling factor applied when upscale_mode is 'factor'.",
                    },
                ),
            },
            "optional": {
                "video": (
                    "VIDEO",
                    {
                        "tooltip": "Video to upscale. Takes precedence over input_video_url when connected.",
                    },
                ),
                "input_video_url": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "URL of the video to upscale. Used when no VIDEO input is connected.",
                    },
                ),
                "upscale_mode": (
                    ["factor", "target"],
                    {
                        "default": "factor",
                        "tooltip": "'factor' scales by upscale_factor; 'target' scales to target_resolution.",
                    },
                ),
                "target_resolution": (
                    ["720p", "1080p", "1440p", "2160p"],
                    {
                        "default": "1080p",
                        "tooltip": "Output resolution used when upscale_mode is 'target'.",
                    },
                ),
                "noise_scale": (
                    "FLOAT",
                    {
                        "default": 0.1,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Amount of noise conditioning; higher can hallucinate more detail.",
                    },
                ),
                "output_quality": (
                    ["low", "medium", "high", "maximum"],
                    {
                        "default": "high",
                        "tooltip": "Encoding quality of the output video.",
                    },
                ),
                "output_write_mode": (
                    ["fast", "balanced", "small"],
                    {
                        "default": "balanced",
                        "tooltip": "Encoder speed/size trade-off for writing the output file.",
                    },
                ),
                "output_format": (
                    [
                        "X264 (.mp4)",
                        "VP9 (.webm)",
                        "PRORES444 (.mov)",
                        "GIF (.gif)",
                    ],
                    {
                        "default": "X264 (.mp4)",
                        "tooltip": "Container and codec of the output video.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_url",)
    FUNCTION = "generate_upscaled_video"
    CATEGORY = "FAL/VideoUpscaling"

    def generate_upscaled_video(
        self,
        upscale_factor=2.0,
        video=None,
        input_video_url=None,
        upscale_mode="factor",
        target_resolution="1080p",
        noise_scale=0.1,
        output_format="X264 (.mp4)",
        output_quality="high",
        output_write_mode="balanced",
    ):
        try:
            video_url = input_video_url
            if video is not None:
                video_url = ImageUtils.upload_file(video.get_stream_source())
            if not video_url:
                return ApiHandler.handle_video_generation_error(
                    "seedvr-upscale-video",
                    "No video provided. Connect a VIDEO input or set input_video_url.",
                )

            # The API enum is "PRORES4444 (.mov)"; the dropdown historically
            # exposes "PRORES444 (.mov)", so translate at the argument level.
            api_output_format = (
                "PRORES4444 (.mov)"
                if output_format == "PRORES444 (.mov)"
                else output_format
            )

            arguments = {
                "video_url": video_url,
                "upscale_mode": upscale_mode,
                "upscale_factor": upscale_factor,
                "target_resolution": target_resolution,
                "noise_scale": noise_scale,
                "output_format": api_output_format,
                "output_quality": output_quality,
                "output_write_mode": output_write_mode,
            }

            result = ApiHandler.submit_and_get_result(
                "fal-ai/seedvr/upscale/video", arguments
            )
            return (result["video"]["url"],)
        except Exception as e:
            return ApiHandler.handle_video_generation_error(
                "seedvr-upscale-video", e
            )


class BriaVideoIncreaseResolutionNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "upscale_factor": (
                    "INT",
                    {
                        "default": 2,
                        "min": 2,
                        "max": 4,
                        "step": 2,
                        "tooltip": "Resolution increase factor. The API accepts 2 or 4.",
                    },
                ),
            },
            "optional": {
                "video": (
                    "VIDEO",
                    {
                        "tooltip": "Video to upscale. Takes precedence over input_video_url when connected.",
                    },
                ),
                "input_video_url": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "URL of the video to upscale. Used when no VIDEO input is connected.",
                    },
                ),
                "output_container_and_codec": (
                    [
                        "mp4_h264",
                        "mp4_h265",
                        "mov_h265",
                        "mov_proresks",
                        "webm_vp9",
                        "mkv_h265",
                        "mkv_vp9",
                        "gif",
                    ],
                    {
                        "default": "mp4_h264",
                        "tooltip": "Container and codec of the output video.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_url",)
    FUNCTION = "generate_upscaled_video"
    CATEGORY = "FAL/VideoUpscaling"

    def generate_upscaled_video(
        self,
        video=None,
        input_video_url=None,
        upscale_factor=2,
        output_container_and_codec="mp4_h264",
    ):
        try:
            video_url = input_video_url
            if video is not None:
                video_url = ImageUtils.upload_file(video.get_stream_source())
            if not video_url:
                return ApiHandler.handle_video_generation_error(
                    "bria-video-increase-resolution",
                    "No video provided. Connect a VIDEO input or set input_video_url.",
                )

            arguments = {
                "video_url": video_url,
                "desired_increase": str(upscale_factor),
                "output_container_and_codec": output_container_and_codec,
            }

            result = ApiHandler.submit_and_get_result(
                "bria/video/increase-resolution", arguments
            )
            return (result["video"]["url"],)
        except Exception as e:
            return ApiHandler.handle_video_generation_error(
                "bria-video-increase-resolution", e
            )


class TopazUpscaleVideoNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "upscale_factor": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "min": 1.0,
                        "max": 5.0,
                        "step": 0.1,
                        "tooltip": "How much to enlarge the video (1x-5x).",
                    },
                ),
            },
            "optional": {
                "video": (
                    "VIDEO",
                    {
                        "tooltip": "Video to upscale. Takes precedence over input_video_url when connected.",
                    },
                ),
                "input_video_url": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "URL of the video to upscale. Used when no VIDEO input is connected.",
                    },
                ),
                "use_fps": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Enable frame interpolation to target_fps.",
                    },
                ),
                "target_fps": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 60,
                        "tooltip": "Target output frame rate. Only used when use_fps is enabled and value is non-zero.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_url",)
    FUNCTION = "generate_upscaled_video"
    CATEGORY = "FAL/VideoUpscaling"

    def generate_upscaled_video(
        self,
        video=None,
        input_video_url=None,
        upscale_factor=2.0,
        use_fps=False,
        target_fps=0,
    ):
        try:
            video_url = input_video_url
            if video is not None:
                video_url = ImageUtils.upload_file(video.get_stream_source())
            if not video_url:
                return ApiHandler.handle_video_generation_error(
                    "fal-ai/topaz/upscale/video",
                    "No video provided. Connect a VIDEO input or set input_video_url.",
                )

            arguments = {
                "video_url": video_url,
                "upscale_factor": upscale_factor,
            }
            if target_fps != 0 and use_fps:
                arguments["target_fps"] = target_fps

            result = ApiHandler.submit_and_get_result(
                "fal-ai/topaz/upscale/video", arguments
            )
            return (result["video"]["url"],)
        except Exception as e:
            return ApiHandler.handle_video_generation_error(
                "fal-ai/topaz/upscale/video", e
            )


# Node class mappings
NODE_CLASS_MAPPINGS = {
    "Upscaler_fal": UpscalerNode,
    "Seedvr_Upscaler_fal": SeedvrUpscalerNode,
    "Seedvr_Upscale_Video_fal": SeedvrUpscaleVideoNode,
    "Bria_Video_Increase_Resolution_fal": BriaVideoIncreaseResolutionNode,
    "Topaz_Upscale_Video_fal": TopazUpscaleVideoNode,
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "Upscaler_fal": "Clarity Upscaler (fal)",
    "Seedvr_Upscaler_fal": "Seedvr Upscaler (fal)",
    "Seedvr_Upscale_Video_fal": "Seedvr Upscale Video (fal)",
    "Bria_Video_Increase_Resolution_fal": "Bria Video Increase Resolution (fal)",
    "Topaz_Upscale_Video_fal": "Topaz Upscale Video (fal)",
}

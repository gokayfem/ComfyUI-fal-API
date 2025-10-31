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

class SeedvrUpscaleVideoNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "upscale_factor": (
                    "FLOAT",
                    {"default": 2.0, "min": 0.00, "max": 5.0, "step": 0.01},
                ),
            },
            "optional": {
                "video": ("VIDEO",),
                "input_video_url": ("STRING", {"default": ""}),
                "upscale_mode": (["factor", "target_resolution"], {"default": "factor"}),
                "target_resolution": (["720p", "1080p", "1440p", "2160p"], {"default": "1080p"}),
                "noise_scale": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.05}),
                "output_quality": (["low", "medium", "high", "maximum"], {"default": "high"}),
                "output_write_mode": (["fast", "balanced", "small"], {"default": "balanced" }),
                "output_format": (["X264 (.mp4)", "VP9 (.webm)","PRORES444 (.mov)","GIF (.gif)"], {"default": "X264 (.mp4)"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_url",)
    FUNCTION = "generate_upscaled_video"
    CATEGORY = "FAL/Image"

    def generate_upscaled_video(
        self,
        upscale_factor= 2.0,
        video =None,
        input_video_url=None,
        upscale_mode="factor",
        target_resolution="1080p",
        noise_scale=0.1,
        output_format="X264 (.mp4)",
        output_quality="high",
        output_write_mode="balanced",
    ):
        # try:
            
        video_url = input_video_url
        if video is not None:
            video_url = ImageUtils.upload_file(video.get_stream_source())
        if not video_url or video_url=="":
            return ApiHandler.handle_video_generation_error(
                "bria-video-increase-resolution", "Failed to upload video for upscaling. No video URL provided, or Video provided."
            )
        arguments={
"video_url": video_url,
"upscale_mode": upscale_mode,
"upscale_factor": upscale_factor,
"target_resolution": target_resolution  ,
"noise_scale": noise_scale,
"output_format": output_format,
"output_quality": output_quality,
"output_write_mode": output_write_mode
}
        print(arguments)
        result = ApiHandler.submit_and_get_result(
            "fal-ai/seedvr/upscale/video", arguments
        )
        print(result)
        return (result["video"]["url"],)
        # except Exception as e:
        #     return ApiHandler.handle_video_generation_error(
        #         "fal-ai/seedvr/upscale/video", str(e)
        #     )

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
                "output_container_and_codec": (["mp4_h264", "mp4_h265","mov_h265","mov_proresks", "webm_vp9","mkv_h265","mkv_h265", "mkv_vp9", "gif"], {"default": "mp4_h264"}),
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
            result = ApiHandler.submit_and_get_result(
                "bria/video/increase-resolution", arguments
            )
            return (result["video"]["url"],)
        except Exception as e:
            return ApiHandler.handle_video_generation_error(
                "bria/video/increase-resolution", str(e)
            )


class TopazUpscaleVideoNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
              
                "upscale_factor": (
                    "FLOAT",
                    {"default": 2.0, "min": 1.0, "max": 5.0, "step": 0.1},
                ),
            },
            "optional": {
                "video": ("VIDEO",),
                "input_video_url": ("STRING", {"default": ""}),
                "use_fps": ("BOOLEAN", {"default": False}),
                "target_fps": ("INT", {"default": 0, "min": 0, "max": 60}),
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
        upscale_factor=2.0,
        use_fps=False,
        target_fps= 0
    ):
        try:
            video_url = input_video_url
            if video is not None:
                video_url = ImageUtils.upload_file(video.get_stream_source())
            if not video_url or video_url=="":
                return ApiHandler.handle_video_generation_error(
                    "fal-ai/topaz/upscale/video", "Failed to upload video for upscaling. No video URL provided, or Video provided."
                )
            arguments={
        "video_url": video_url,
        "desired_increase": str(upscale_factor)
    }
            if target_fps != 0 and use_fps:
                arguments["target_fps"] = target_fps
     
            result = ApiHandler.submit_and_get_result(
                "fal-ai/topaz/upscale/video", arguments
            )
            return (result["video"]["url"],)
        except Exception as e:
            return ApiHandler.handle_video_generation_error(
                "fal-ai/topaz/upscale/video", str(e)
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

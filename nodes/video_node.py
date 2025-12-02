import asyncio
import os
import tempfile

import cv2
import numpy as np
import requests
import torch
from fal_client import AsyncClient

from .fal_utils import ApiHandler, FalConfig, ImageUtils

# Initialize FalConfig
fal_config = FalConfig()

class MiniMaxNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_video"
    CATEGORY = "FAL/VideoGeneration"

    def generate_video(self, prompt, image):
        try:
            image_url = ImageUtils.upload_image(image)
            if not image_url:
                return ApiHandler.handle_video_generation_error(
                    "minimax/video-01-live", "Failed to upload image"
                )

            arguments = {
                "prompt": prompt,
                "image_url": image_url,
            }

            result = ApiHandler.submit_and_get_result(
                "fal-ai/minimax/video-01-live/image-to-video", arguments
            )
            video_url = result["video"]["url"]
            return (video_url,)
        except Exception as e:
            return ApiHandler.handle_video_generation_error(
                "minimax/video-01-live", str(e)
            )


class MiniMaxTextToVideoNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_video"
    CATEGORY = "FAL/VideoGeneration"

    def generate_video(self, prompt):
        try:
            arguments = {
                "prompt": prompt,
            }

            result = ApiHandler.submit_and_get_result("fal-ai/minimax-video", arguments)
            video_url = result["video"]["url"]
            return (video_url,)
        except Exception as e:
            return ApiHandler.handle_video_generation_error("minimax-video", str(e))


class MiniMaxSubjectReferenceNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "subject_reference_image": ("IMAGE",),
                "prompt_optimizer": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_video"
    CATEGORY = "FAL/VideoGeneration"

    def generate_video(self, prompt, subject_reference_image, prompt_optimizer):
        try:
            image_url = ImageUtils.upload_image(subject_reference_image)
            if not image_url:
                return ApiHandler.handle_video_generation_error(
                    "minimax/video-01-subject-reference",
                    "Failed to upload subject reference image",
                )

            arguments = {
                "prompt": prompt,
                "subject_reference_image_url": image_url,
                "prompt_optimizer": prompt_optimizer,
            }

            result = ApiHandler.submit_and_get_result(
                "fal-ai/minimax/video-01-subject-reference", arguments
            )
            video_url = result["video"]["url"]
            return (video_url,)
        except Exception as e:
            return ApiHandler.handle_video_generation_error(
                "minimax/video-01-subject-reference", str(e)
            )


class KlingNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "duration": (["5", "10"], {"default": "5"}),
                "aspect_ratio": (["16:9", "9:16", "1:1"], {"default": "16:9"}),
            },
            "optional": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_video"
    CATEGORY = "FAL/VideoGeneration"

    def generate_video(self, prompt, duration, aspect_ratio, image=None):
        arguments = {
            "prompt": prompt,
            "duration": duration,
            "aspect_ratio": aspect_ratio,
        }

        try:
            if image is not None:
                image_url = ImageUtils.upload_image(image)
                if image_url:
                    arguments["image_url"] = image_url
                    result = ApiHandler.submit_and_get_result(
                        "fal-ai/kling-video/v1/standard/image-to-video", arguments
                    )
                else:
                    return ApiHandler.handle_video_generation_error(
                        "kling-video/v1/standard", "Failed to upload image"
                    )
            else:
                result = ApiHandler.submit_and_get_result(
                    "fal-ai/kling-video/v1/standard/text-to-video", arguments
                )

            video_url = result["video"]["url"]
            return (video_url,)
        except Exception as e:
            return ApiHandler.handle_video_generation_error(
                "kling-video/v1/standard", str(e)
            )


class KlingPro10Node:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "duration": (["5", "10"], {"default": "5"}),
                "aspect_ratio": (["16:9", "9:16", "1:1"], {"default": "16:9"}),
            },
            "optional": {
                "image": ("IMAGE",),
                "tail_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_video"
    CATEGORY = "FAL/VideoGeneration"

    def generate_video(
        self, prompt, duration, aspect_ratio, image=None, tail_image=None
    ):
        arguments = {
            "prompt": prompt,
            "duration": duration,
            "aspect_ratio": aspect_ratio,
        }

        try:
            if image is not None:
                image_url = ImageUtils.upload_image(image)
                if image_url:
                    arguments["image_url"] = image_url

                    # Handle tail image if provided
                    if tail_image is not None:
                        tail_image_url = ImageUtils.upload_image(tail_image)
                        if tail_image_url:
                            arguments["tail_image_url"] = tail_image_url
                        else:
                            return ApiHandler.handle_video_generation_error(
                                "kling-video/v1/pro", "Failed to upload tail image"
                            )

                    result = ApiHandler.submit_and_get_result(
                        "fal-ai/kling-video/v1/pro/image-to-video", arguments
                    )
                else:
                    return ApiHandler.handle_video_generation_error(
                        "kling-video/v1/pro", "Failed to upload image"
                    )
            else:
                result = ApiHandler.submit_and_get_result(
                    "fal-ai/kling-video/v1/pro/text-to-video", arguments
                )

            video_url = result["video"]["url"]
            return (video_url,)
        except Exception as e:
            return ApiHandler.handle_video_generation_error(
                "kling-video/v1/pro", str(e)
            )


class KlingPro16Node:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "duration": (["5", "10"], {"default": "5"}),
                "aspect_ratio": (["16:9", "9:16", "1:1"], {"default": "16:9"}),
            },
            "optional": {
                "image": ("IMAGE",),
                "tail_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_video"
    CATEGORY = "FAL/VideoGeneration"

    def generate_video(
        self, prompt, duration, aspect_ratio, image=None, tail_image=None
    ):
        arguments = {
            "prompt": prompt,
            "duration": duration,
            "aspect_ratio": aspect_ratio,
        }

        try:
            if image is not None:
                image_url = ImageUtils.upload_image(image)
                if image_url:
                    arguments["image_url"] = image_url

                    # Handle tail image if provided
                    if tail_image is not None:
                        tail_image_url = ImageUtils.upload_image(tail_image)
                        if tail_image_url:
                            arguments["tail_image_url"] = tail_image_url
                        else:
                            return ApiHandler.handle_video_generation_error(
                                "kling-video/v1.6/pro", "Failed to upload tail image"
                            )

                    result = ApiHandler.submit_and_get_result(
                        "fal-ai/kling-video/v1.6/pro/image-to-video", arguments
                    )
                else:
                    return ApiHandler.handle_video_generation_error(
                        "kling-video/v1.6/pro", "Failed to upload image"
                    )
            else:
                result = ApiHandler.submit_and_get_result(
                    "fal-ai/kling-video/v1.6/pro/text-to-video", arguments
                )

            video_url = result["video"]["url"]
            return (video_url,)
        except Exception as e:
            return ApiHandler.handle_video_generation_error(
                "kling-video/v1.6/pro", str(e)
            )


class KlingMasterNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "duration": (["5", "10"], {"default": "5"}),
                "aspect_ratio": (["16:9", "9:16", "1:1"], {"default": "16:9"}),
            },
            "optional": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_video"
    CATEGORY = "FAL/VideoGeneration"

    def generate_video(self, prompt, duration, aspect_ratio, image=None):
        arguments = {
            "prompt": prompt,
            "duration": duration,
            "aspect_ratio": aspect_ratio,
        }

        try:
            if image is not None:
                image_url = ImageUtils.upload_image(image)
                if image_url:
                    arguments["image_url"] = image_url
                    result = ApiHandler.submit_and_get_result(
                        "fal-ai/kling-video/v2/master/image-to-video", arguments
                    )
                else:
                    return ApiHandler.handle_video_generation_error(
                        "kling-video/v2/master", "Failed to upload image"
                    )
            else:
                result = ApiHandler.submit_and_get_result(
                    "fal-ai/kling-video/v2/master/text-to-video", arguments
                )

            video_url = result["video"]["url"]
            return (video_url,)
        except Exception as e:
            return ApiHandler.handle_video_generation_error(
                "kling-video/v2/master", str(e)
            )
        
class KlingOmniImageToVideoNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "start_image": ("IMAGE",),
            },
            "optional": {
                "end_image": ("IMAGE",),
                "duration": (["5", "10"], {"default": "5"}),
                "variations": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
            },
        }

    RETURN_TYPES = ("STRING",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "generate_video"
    CATEGORY = "FAL/VideoGeneration"

    def generate_video(self, prompt, start_image, end_image=None, duration="5", variations=1):
        try:
            start_image_url = ImageUtils.upload_image(start_image)
            if not start_image_url:
                return ApiHandler.handle_video_generation_error(
                    "kling-video/o1/image-to-video", "Failed to upload start image"
                )

            arguments = {
                "prompt": prompt,
                "start_image_url": start_image_url,
                "duration": duration,
            }

            if end_image is not None:
                end_image_url = ImageUtils.upload_image(end_image)
                if end_image_url:
                    arguments["end_image_url"] = end_image_url
                else:
                    return ApiHandler.handle_video_generation_error(
                        "kling-video/o1/image-to-video", "Failed to upload end image"
                    )

            results = ApiHandler.submit_multiple_and_get_results(
                "fal-ai/kling-video/o1/image-to-video", arguments, variations
            )
            return ([r["video"]["url"] for r in results],)
        except Exception as e:
            return ApiHandler.handle_video_generation_error(
                "kling-video/o1/image-to-video", str(e)
            )

class KlingOmniReferenceToVideoNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "reference_images": ("IMAGE", {"default": None, "multiple": True}),
                "element_1_frontal_image": ("IMAGE",),
                "element_1_reference_images": ("IMAGE", {"default": None, "multiple": True}),
                "element_2_frontal_image": ("IMAGE",),
                "element_2_reference_images": ("IMAGE", {"default": None, "multiple": True}),
                "element_3_frontal_image": ("IMAGE",),
                "element_3_reference_images": ("IMAGE", {"default": None, "multiple": True}),
                "element_4_frontal_image": ("IMAGE",),
                "element_4_reference_images": ("IMAGE", {"default": None, "multiple": True}),
                "element_5_frontal_image": ("IMAGE",),
                "element_5_reference_images": ("IMAGE", {"default": None, "multiple": True}),
                "element_6_frontal_image": ("IMAGE",),
                "element_6_reference_images": ("IMAGE", {"default": None, "multiple": True}),
                "element_7_frontal_image": ("IMAGE",),
                "element_7_reference_images": ("IMAGE", {"default": None, "multiple": True}),
                "duration": (["5", "10"], {"default": "5"}),
                "aspect_ratio": (["16:9", "9:16", "1:1"], {"default": "16:9"}),
                "variations": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
            },
        }

    RETURN_TYPES = ("STRING",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "generate_video"
    CATEGORY = "FAL/VideoGeneration"

    def generate_video(
        self,
        prompt,
        reference_images=None,
        element_1_frontal_image=None,
        element_1_reference_images=None,
        element_2_frontal_image=None,
        element_2_reference_images=None,
        element_3_frontal_image=None,
        element_3_reference_images=None,
        element_4_frontal_image=None,
        element_4_reference_images=None,
        element_5_frontal_image=None,
        element_5_reference_images=None,
        element_6_frontal_image=None,
        element_6_reference_images=None,
        element_7_frontal_image=None,
        element_7_reference_images=None,
        duration="5",
        aspect_ratio="16:9",
        variations=1
    ):
        try:
            arguments = {
                "prompt": prompt,
                "duration": duration,
                "aspect_ratio": aspect_ratio,
            }

            # Handle reference images
            if reference_images is not None:
                ref_image_urls = ImageUtils.prepare_images(reference_images)
                if ref_image_urls:
                    arguments["image_urls"] = ref_image_urls

            # Build elements array
            elements = []

            # Process each element (up to 7)
            for i in range(1, 8):
                frontal_img = locals().get(f"element_{i}_frontal_image")
                ref_imgs = locals().get(f"element_{i}_reference_images")

                if frontal_img is not None:
                    element = {}

                    # Upload frontal image
                    frontal_url = ImageUtils.upload_image(frontal_img)
                    if frontal_url:
                        element["frontal_image_url"] = frontal_url

                    # Upload reference images if provided
                    if ref_imgs is not None:
                        ref_urls = ImageUtils.prepare_images(ref_imgs)
                        if ref_urls:
                            element["reference_image_urls"] = ref_urls

                    elements.append(element)

            if elements:
                arguments["elements"] = elements

            results = ApiHandler.submit_multiple_and_get_results(
                "fal-ai/kling-video/o1/reference-to-video", arguments, variations
            )
            return ([r["video"]["url"] for r in results],)
        except Exception as e:
            return ApiHandler.handle_video_generation_error(
                "kling-video/o1/reference-to-video", str(e)
            )

class KlingOmniVideoToVideoEditNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "video": ("VIDEO",),
            },
            "optional": {
                "keep_audio": ("BOOLEAN", {"default": False}),
                "reference_images": ("IMAGE", {"default": None, "multiple": True}),
                "element_1_frontal_image": ("IMAGE",),
                "element_1_reference_images": ("IMAGE", {"default": None, "multiple": True}),
                "element_2_frontal_image": ("IMAGE",),
                "element_2_reference_images": ("IMAGE", {"default": None, "multiple": True}),
                "element_3_frontal_image": ("IMAGE",),
                "element_3_reference_images": ("IMAGE", {"default": None, "multiple": True}),
                "element_4_frontal_image": ("IMAGE",),
                "element_4_reference_images": ("IMAGE", {"default": None, "multiple": True}),
                "variations": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_url",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "edit_video"
    CATEGORY = "FAL/VideoGeneration"

    def edit_video(
        self,
        prompt,
        video,
        keep_audio=False,
        reference_images=None,
        element_1_frontal_image=None,
        element_1_reference_images=None,
        element_2_frontal_image=None,
        element_2_reference_images=None,
        element_3_frontal_image=None,
        element_3_reference_images=None,
        element_4_frontal_image=None,
        element_4_reference_images=None,
        variations=1
    ):
        try:
            video_url = ImageUtils.upload_file(video.get_stream_source())
            if not video_url:
                return ApiHandler.handle_video_generation_error(
                    "kling-video/o1/video-to-video/edit", "Failed to upload video"
                )

            arguments = {
                "prompt": prompt,
                "video_url": video_url,
                "keep_audio": keep_audio,
            }

            # Handle reference images
            if reference_images is not None:
                ref_image_urls = ImageUtils.prepare_images(reference_images)
                if ref_image_urls:
                    arguments["image_urls"] = ref_image_urls

            # Build elements array
            elements = []

            # Process each element (up to 4 for video-to-video/edit)
            for i in range(1, 5):
                frontal_img = locals().get(f"element_{i}_frontal_image")
                ref_imgs = locals().get(f"element_{i}_reference_images")

                if frontal_img is not None:
                    element = {}

                    # Upload frontal image
                    frontal_url = ImageUtils.upload_image(frontal_img)
                    if frontal_url:
                        element["frontal_image_url"] = frontal_url

                    # Upload reference images if provided
                    if ref_imgs is not None:
                        ref_urls = ImageUtils.prepare_images(ref_imgs)
                        if ref_urls:
                            element["reference_image_urls"] = ref_urls

                    elements.append(element)

            if elements:
                arguments["elements"] = elements

            results = ApiHandler.submit_multiple_and_get_results(
                "fal-ai/kling-video/o1/video-to-video/edit", arguments, variations
            )
            return ([r["video"]["url"] for r in results],)
        except Exception as e:
            return ApiHandler.handle_video_generation_error(
                "kling-video/o1/video-to-video/edit", str(e)
            )

class KlingOmniVideoToVideoReferenceNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "video": ("VIDEO",),
            },
            "optional": {
                "keep_audio": ("BOOLEAN", {"default": False}),
                "reference_images": ("IMAGE", {"default": None, "multiple": True}),
                "element_1_frontal_image": ("IMAGE",),
                "element_1_reference_images": ("IMAGE", {"default": None, "multiple": True}),
                "element_2_frontal_image": ("IMAGE",),
                "element_2_reference_images": ("IMAGE", {"default": None, "multiple": True}),
                "element_3_frontal_image": ("IMAGE",),
                "element_3_reference_images": ("IMAGE", {"default": None, "multiple": True}),
                "element_4_frontal_image": ("IMAGE",),
                "element_4_reference_images": ("IMAGE", {"default": None, "multiple": True}),
                "aspect_ratio": (["auto", "16:9", "9:16", "1:1"], {"default": "auto"}),
                "duration": (["5", "10"], {"default": "5"}),
                "variations": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_url",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "generate_video"
    CATEGORY = "FAL/VideoGeneration"

    def generate_video(
        self,
        prompt,
        video,
        keep_audio=False,
        reference_images=None,
        element_1_frontal_image=None,
        element_1_reference_images=None,
        element_2_frontal_image=None,
        element_2_reference_images=None,
        element_3_frontal_image=None,
        element_3_reference_images=None,
        element_4_frontal_image=None,
        element_4_reference_images=None,
        aspect_ratio="auto",
        duration="5",
        variations=1
    ):
        try:
            video_url = ImageUtils.upload_file(video.get_stream_source())
            if not video_url:
                return ApiHandler.handle_video_generation_error(
                    "kling-video/o1/video-to-video/reference", "Failed to upload video"
                )

            arguments = {
                "prompt": prompt,
                "video_url": video_url,
                "keep_audio": keep_audio,
                "aspect_ratio": aspect_ratio,
                "duration": duration,
            }

            # Handle reference images
            if reference_images is not None:
                ref_image_urls = ImageUtils.prepare_images(reference_images)
                if ref_image_urls:
                    arguments["image_urls"] = ref_image_urls

            # Build elements array
            elements = []

            # Process each element (up to 4 for video-to-video/reference)
            for i in range(1, 5):
                frontal_img = locals().get(f"element_{i}_frontal_image")
                ref_imgs = locals().get(f"element_{i}_reference_images")

                if frontal_img is not None:
                    element = {}

                    # Upload frontal image
                    frontal_url = ImageUtils.upload_image(frontal_img)
                    if frontal_url:
                        element["frontal_image_url"] = frontal_url

                    # Upload reference images if provided
                    if ref_imgs is not None:
                        ref_urls = ImageUtils.prepare_images(ref_imgs)
                        if ref_urls:
                            element["reference_image_urls"] = ref_urls

                    elements.append(element)

            if elements:
                arguments["elements"] = elements

            results = ApiHandler.submit_multiple_and_get_results(
                "fal-ai/kling-video/o1/video-to-video/reference", arguments, variations
            )
            return ([r["video"]["url"] for r in results],)
        except Exception as e:
            return ApiHandler.handle_video_generation_error(
                "kling-video/o1/video-to-video/reference", str(e)
            )

class RunwayGen3Node:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "image": ("IMAGE",),
                "duration": (["5", "10"], {"default": "5"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_video"
    CATEGORY = "FAL/VideoGeneration"

    def generate_video(self, prompt, image, duration):
        try:
            image_url = ImageUtils.upload_image(image)
            if not image_url:
                return ApiHandler.handle_video_generation_error(
                    "runway-gen3", "Failed to upload image"
                )

            arguments = {
                "prompt": prompt,
                "image_url": image_url,
                "duration": duration,
            }

            result = ApiHandler.submit_and_get_result(
                "fal-ai/runway-gen3/turbo/image-to-video", arguments
            )
            video_url = result["video"]["url"]
            return (video_url,)
        except Exception as e:
            return ApiHandler.handle_video_generation_error("runway-gen3", str(e))


class LumaDreamMachineNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "mode": (
                    ["text-to-video", "image-to-video"],
                    {"default": "text-to-video"},
                ),
                "aspect_ratio": (
                    ["16:9", "9:16", "4:3", "3:4", "21:9", "9:21"],
                    {"default": "16:9"},
                ),
            },
            "optional": {
                "image": ("IMAGE",),
                "end_image": ("IMAGE",),
                "loop": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_video"
    CATEGORY = "FAL/VideoGeneration"

    def generate_video(
        self, prompt, mode, aspect_ratio, image=None, end_image=None, loop=False
    ):
        arguments = {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "loop": loop,
        }

        try:
            if mode == "image-to-video":
                if image is None:
                    return ApiHandler.handle_video_generation_error(
                        "luma-dream-machine",
                        "Image is required for image-to-video mode",
                    )
                image_url = ImageUtils.upload_image(image)
                if not image_url:
                    return ApiHandler.handle_video_generation_error(
                        "luma-dream-machine", "Failed to upload image"
                    )
                arguments["image_url"] = image_url

                if end_image is not None:
                    end_image_url = ImageUtils.upload_image(end_image)
                    if end_image_url:
                        arguments["end_image_url"] = end_image_url
                    else:
                        return ApiHandler.handle_video_generation_error(
                            "luma-dream-machine", "Failed to upload end image"
                        )

                endpoint = "fal-ai/luma-dream-machine/ray-2/image-to-video"
            else:
                endpoint = "fal-ai/luma-dream-machine/ray-2"

            result = ApiHandler.submit_and_get_result(endpoint, arguments)
            video_url = result["video"]["url"]
            return (video_url,)
        except Exception as e:
            return ApiHandler.handle_video_generation_error(
                "luma-dream-machine", str(e)
            )


class Veo2ImageToVideoNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "image": ("IMAGE",),
                "aspect_ratio": (
                    ["auto", "auto_prefer_portrait", "16:9", "9:16"],
                    {"default": "auto"},
                ),
                "duration": (["5s", "6s", "7s", "8s"], {"default": "5s"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_video"
    CATEGORY = "FAL/VideoGeneration"

    def generate_video(self, prompt, image, aspect_ratio, duration):
        try:
            image_url = ImageUtils.upload_image(image)
            if not image_url:
                return ApiHandler.handle_video_generation_error(
                    "veo2", "Failed to upload image"
                )

            arguments = {
                "prompt": prompt,
                "image_url": image_url,
                "aspect_ratio": aspect_ratio,
                "duration": duration,
            }

            result = ApiHandler.submit_and_get_result(
                "fal-ai/veo2/image-to-video", arguments
            )
            video_url = result["video"]["url"]
            return (video_url,)
        except Exception as e:
            return ApiHandler.handle_video_generation_error("veo2", str(e))


class WanProNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "image": ("IMAGE",),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "enable_safety_checker": ("BOOLEAN", {"default": True}),
                "variations": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
            },
        }

    RETURN_TYPES = ("STRING",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "generate_video"
    CATEGORY = "FAL/VideoGeneration"

    def generate_video(self, prompt, image, seed=0, enable_safety_checker=True, variations=1):
        try:
            image_url = ImageUtils.upload_image(image)
            if not image_url:
                return ApiHandler.handle_video_generation_error(
                    "wan-pro", "Failed to upload image"
                )

            arguments = {
                "prompt": prompt,
                "image_url": image_url,
                "enable_safety_checker": enable_safety_checker,
            }

            # Only add seed if it's not 0 (default)
            if seed != 0:
                arguments["seed"] = seed

            results = ApiHandler.submit_multiple_and_get_results(
                "fal-ai/wan-pro/image-to-video", arguments, variations
            )

            return ([r["video"]["url"] for r in results],)
        except Exception as e:
            return ApiHandler.handle_video_generation_error("wan-pro", str(e))

class Wan25Node:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "image": ("IMAGE",),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "resolution": (
                    ["480p", "720p", "1080p"],
                    {"default": "1080p"}),
                "duration": (
                    ["5", "10"],
                    {"default": "5"}),
                "negative_prompt": ("STRING", {"default": "low resolution, error, worst quality, low quality, defects", "multiline": True}),
                "enable_prompt_expansion": ("BOOLEAN", {"default": True}),
                "variations": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
            },
        }

    RETURN_TYPES = ("STRING",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "generate_video"
    CATEGORY = "FAL/VideoGeneration"

    def generate_video(
        self,
        prompt,
        image,
        seed=0,
        resolution="1080p",
        duration="5",
        negative_prompt="low resolution, error, worst quality, low quality, defects",
        enable_prompt_expansion=True,
        variations=1,
    ):
        try:
            image_url = ImageUtils.upload_image(image)
            if not image_url:
                return ApiHandler.handle_video_generation_error(
                    "wan-25", "Failed to upload image"
                )

            arguments = {
                "prompt": prompt,
                "image_url": image_url,
                "resolution": resolution,
                "duration": duration,
                "negative_prompt": negative_prompt,
                "enable_prompt_expansion": enable_prompt_expansion,
            }

            # include seed if non-default
            if seed != 0:
                arguments["seed"] = seed


            results = ApiHandler.submit_multiple_and_get_results(
                "fal-ai/wan-25-preview/image-to-video", arguments, variations
            )

            return ([r["video"]["url"] for r in results],)

        except Exception as e:
            return ApiHandler.handle_video_generation_error("wan-25", str(e))
        
        
class WanVACEVideoEditNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "video": ("VIDEO", {"default": None}),
                "input_video_url": ("STRING", {"default": ""}),
                
                "images": ("IMAGE", {"default": None, "multiple": True}),
                "video_type": (["auto", "general", "human"], {"default": "auto"}),
                "resolution": (
                    ["auto", "240p", "360p", "480p", "580p", "720p", "1080p"],
                    {"default": "auto"}
                ),
                "acceleration": (["regular", "low", "none", "default"], {"default": "regular"}),
                "enable_auto_downsample": ("BOOLEAN", {"default": True}),
                "aspect_ratio": (["auto", "16:9", "9:16", "1:1"], {"default": "auto"}),
                "auto_downsample_min_fps": ("INT", {"default": 15, "min": 1, "max": 60}),
                "enable_safety_checker": ("BOOLEAN", {"default": True}),
                "variations": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_url",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "edit_video"
    CATEGORY = "FAL/VideoGeneration"

    def edit_video(
        self,
        prompt,
        video=None,
        input_video_url="",
        images=None,
        video_type="auto",
        resolution="auto",
        acceleration="regular",
        enable_auto_downsample=True,
        aspect_ratio="auto",
        auto_downsample_min_fps=15,
        enable_safety_checker=True,
        variations=1,
    ):
        try:
            if video is None and input_video_url is "":
                return ApiHandler.handle_video_generation_error(
                    "wan-vace", "Video or Video Frames input is required."
                )
            if video is None and input_video_url is not "":
                video_url = input_video_url
            else:
                video_url = ImageUtils.upload_file(video.get_stream_source())
            if not video_url:
                return ApiHandler.handle_video_generation_error(
                    "wan-vace", "Failed to upload video"
                )
            arguments = {
                "prompt": prompt,
                "video_url": video_url,
                "video_type": video_type,
                "resolution": resolution,
                "acceleration": acceleration,
                "enable_auto_downsample": enable_auto_downsample,
                "aspect_ratio": aspect_ratio,
                "auto_downsample_min_fps": auto_downsample_min_fps,
                "enable_safety_checker": enable_safety_checker,
            }

            image_urls = []

            if images is not None:
                import torch

                if isinstance(images, torch.Tensor):
                    if images.ndim == 4 and images.shape[0] > 1:
                        for i in range(images.shape[0]):
                            single_img = images[i:i+1]
                            img_url = ImageUtils.upload_image(single_img)
                            if img_url:
                                image_urls.append(img_url)
                    else:
                        img_url = ImageUtils.upload_image(images)
                        if img_url:
                            image_urls.append(img_url)

                elif isinstance(images, (list, tuple)):
                    for img in images:
                        img_url = ImageUtils.upload_image(img)
                        if img_url:
                            image_urls.append(img_url)

            if image_urls:
                arguments["image_urls"] = image_urls

            results = ApiHandler.submit_multiple_and_get_results(
                "fal-ai/wan-vace-apps/video-edit",
                arguments,
                variations
            )

            return ([r["video"]["url"] for r in results],)

        except Exception as e:
            return ApiHandler.handle_video_generation_error("wan-vace", str(e))


class Wan2214bAnimateReplaceNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"default": None}),

            },
            "optional": {
                "video": ("VIDEO", {"default": None}),
                "input_video_url": ("STRING", {"default": ""}),
                "resolution": (
                    ["auto", "240p", "360p", "480p", "580p", "720p", "1080p"],
                    {"default": "auto"}
                ),
                "num_inference_steps": ("INT", {"default": 20, "min": 1}),
                "shift": ("INT", {"default": 5}),
                "video_quality": (["low", "medium", "high"], {"default": "high"}),
                "video_write_mode": (["balanced", "fast", "high_quality"], {"default": "balanced"}),
                "seed": ("INT", {"default": 24}),
                "enable_safety_checker": ("BOOLEAN", {"default": True}),
                "enable_output_safety_checker": ("BOOLEAN", {"default": False}),
                "variations": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_url",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "edit_video"
    CATEGORY = "FAL/VideoGeneration"

    def edit_video(
        self,
        video=None,
        input_video_url="",
        image=None,
        resolution="auto",
        num_inference_steps=20,
        shift=5,
        video_quality="high",
        video_write_mode="balanced",
        seed=24,
        enable_safety_checker=True,
        enable_output_safety_checker=False,
        variations=1,
    ):
        try:
            if video is None and input_video_url is "":
                return ApiHandler.handle_video_generation_error(
                    "wan-22animatereplace", "Video or Video Frames input is required."
                )
            if video is None and input_video_url is not "":
                video_url = input_video_url
            else:
                video_url = ImageUtils.upload_file(video.get_stream_source())
            if not video_url:
                return ApiHandler.handle_video_generation_error(
                    "wan-22animatereplace", "Failed to upload video"
                )

          
            image_url = ImageUtils.upload_image(image)
            if not image_url:
                return ApiHandler.handle_video_generation_error(
                    "wan-22animatereplace", "Failed to upload image"
                )

            arguments={
        "video_url": video_url,
        "image_url": image_url,
        "resolution": resolution,
        "num_inference_steps": num_inference_steps,
        "enable_safety_checker": enable_safety_checker,
        "enable_output_safety_checker": enable_output_safety_checker,
        "shift": shift,
        "video_quality": video_quality,
        "video_write_mode": video_write_mode,
        "seed": seed
    }

            results = ApiHandler.submit_multiple_and_get_results(
                "fal-ai/wan/v2.2-14b/animate/replace",
                arguments,
                variations
            )

            return ([r["video"]["url"] for r in results],)

        except Exception as e:
            return ApiHandler.handle_video_generation_error("wan-22animatereplace", str(e))




class Wan2214bAnimateMoveNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"default": None}),

            },
            "optional": {
                "video": ("VIDEO", {"default": None}),
                "input_video_url": ("STRING", {"default": ""}),
                "resolution": (
                    ["480p", "580p", "720p"],
                    {"default": "480p"}
                ),
                "num_inference_steps": ("INT", {"default": 20, "min": 1}),
                "shift": ("INT", {"default": 5}),
                "video_quality": (["low", "medium", "high"], {"default": "high"}),
                "video_write_mode": (["balanced", "fast", "high_quality"], {"default": "balanced"}),
                "seed": ("INT", {"default": 24}),
                "enable_safety_checker": ("BOOLEAN", {"default": True}),
                "enable_output_safety_checker": ("BOOLEAN", {"default": False}),
                "variations": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_url",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "edit_video"
    CATEGORY = "FAL/VideoGeneration"

    def edit_video(
        self,
        video=None,
        input_video_url="",
        image=None,
        resolution="auto",
        num_inference_steps=20,
        shift=5,
        video_quality="high",
        video_write_mode="balanced",
        seed=24,
        enable_safety_checker=True,
        enable_output_safety_checker=False,
        variations=1,
    ):
        try:
            if video is None and input_video_url is "":
                return ApiHandler.handle_video_generation_error(
                    "wan-22animatemove", "Video or Video Frames input is required."
                )
            if video is None and input_video_url is not "":
                video_url = input_video_url
            else:
                video_url = ImageUtils.upload_file(video.get_stream_source())
            if not video_url:
                return ApiHandler.handle_video_generation_error(
                    "wan-22animatemove", "Failed to upload video"
                )

          
            image_url = ImageUtils.upload_image(image)
            if not image_url:
                return ApiHandler.handle_video_generation_error(
                    "wan-22animatemove", "Failed to upload image"
                )

            arguments={
        "video_url": video_url,
        "image_url": image_url,
        "resolution": resolution,
        "num_inference_steps": num_inference_steps,
        "enable_safety_checker": enable_safety_checker,
        "enable_output_safety_checker": enable_output_safety_checker,
        "shift": shift,
        "video_quality": video_quality,
        "video_write_mode": video_write_mode,
        "seed": seed
    }
            results = ApiHandler.submit_multiple_and_get_results(
                "fal-ai/wan/v2.2-14b/animate/move",
                arguments,
                variations
            )

            return ([r["video"]["url"] for r in results],)

        except Exception as e:
            return ApiHandler.handle_video_generation_error("wan-22animatemove", str(e))



class Wan22VACEFun14bNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": ""}),
                "video": ("VIDEO", {"default": None}),
            },
            "optional": {
                "task": (["depth", "pose"], {"default": "depth"}),
                "preprocess": ("BOOLEAN", {"default": True}),
                "ref_images": ("IMAGE", {"default": None, "multiple": True}),
                "first_frame": ("IMAGE", {"default": None}),
                "last_frame": ("IMAGE", {"default": None}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "seed": ("INT", {"default": -1, "min": 0, "max": 2147483647}),
                "resolution": (
                    ["480p", "580p", "720p"],
                    {"default": "480p"}
                ),
                "aspect_ratio": (["auto", "16:9", "9:16", "1:1"], {"default": "auto"}),
                "num_inference_steps": ("INT", {"default": 30, "min": 1}),
                "guidance_scale": ("FLOAT", {"default": 5, "min": 0.0, "max": 10}),
                "sampler": (["unipc", "dpm++", "euler"], {"default": "unipc"}),
                "match_input_num_frames": ("BOOLEAN", {"default": False}),
                "num_frames": ("INT", {"default": 81, "min": 17, "max": 241}),
                "match_input_frames_per_second": ("BOOLEAN", {"default": False}),
                "frames_per_second": ("INT", {"default": 16, "min": 5, "max": 30}),
                "shift": ("INT", {"default": 5}),
                "acceleration": (["none", "low", "regular"], {"default": "regular"}),
                "video_quality": (["low", "medium", "high"], {"default": "high"}),
                "video_write_mode": (["balanced", "fast", "small"], {"default": "balanced"}),
                "return_frames_zip": ("BOOLEAN", {"default": False}),
                "num_interpolated_frames": ("INT", {"default": 0, "min": 0, "max": 5}),
                "temporal_downsample_factor": ("INT", {"default": 0, "min": 0, "max": 5}),
                "enable_auto_downsample": ("BOOLEAN", {"default": False}),
                "auto_downsample_min_fps": ("INT", {"default": 15, "min": 1, "max": 60}),
                "interpolator_model": (["rife", "film"], {"default": "film"}),
                "enable_safety_checker": ("BOOLEAN", {"default": False}),
                "enable_output_safety_checker": ("BOOLEAN", {"default": False}),
                "variations": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("video_url", "frames_zip_url",)
    OUTPUT_IS_LIST = (True, True,)
    FUNCTION = "edit_video"
    CATEGORY = "FAL/VideoGeneration"

    def edit_video(
        self,
        prompt="",
        video=None,
        task="depth",
        preprocess=True,
        ref_images=None,
        first_frame=None,
        last_frame=None,
        negative_prompt="",
        seed=-1,
        resolution="480p",
        aspect_ratio="auto",
        num_inference_steps=30,
        guidance_scale=5,
        sampler="unipc",
        match_input_num_frames=False,
        num_frames=81,
        match_input_frames_per_second=False,
        frames_per_second=16,
        shift=5,
        acceleration="regular",
        video_quality="high",
        video_write_mode="balanced",
        return_frames_zip=False,
        num_interpolated_frames=0,
        temporal_downsample_factor=0,
        enable_auto_downsample=False,
        auto_downsample_min_fps=15,
        interpolator_model="film",
        enable_safety_checker=False,
        enable_output_safety_checker=False,
        variations=1,
    ):
        try:
            if video is None:
                return ApiHandler.handle_video_generation_error(
                    "wan-22-vace-fun-a14b", "Video input is required."
                )

            video_url = ImageUtils.upload_file(video.get_stream_source())
            if not video_url:
                return ApiHandler.handle_video_generation_error(
                    "wan-22-vace-fun-a14b", "Failed to upload video"
                )

            # Build arguments
            arguments = {
                "prompt": prompt,
                "video_url": video_url,
                "preprocess": preprocess,
                "negative_prompt": negative_prompt,
                "resolution": resolution,
                "aspect_ratio": aspect_ratio,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "sampler": sampler,
                "num_frames": num_frames,
                "frames_per_second": frames_per_second,
                "shift": shift,
                "acceleration": acceleration,
                "video_quality": video_quality,
                "video_write_mode": video_write_mode,
                "return_frames_zip": return_frames_zip,
                "num_interpolated_frames": num_interpolated_frames,
                "temporal_downsample_factor": temporal_downsample_factor,
                "enable_auto_downsample": enable_auto_downsample,
                "auto_downsample_min_fps": auto_downsample_min_fps,
                "interpolator_model": interpolator_model,
                "enable_safety_checker": enable_safety_checker,
                "enable_output_safety_checker": enable_output_safety_checker,
            }

            # Set seed
            if seed != -1:
                arguments["seed"] = seed

            # Handle optional match_input settings
            if match_input_num_frames and video is not None:
                try:
                    arguments["num_frames"] = len(list(video.get_stream()))
                except:
                    pass
            if match_input_frames_per_second and video is not None:
                try:
                    arguments["frames_per_second"] = video.get_fps()
                except:
                    pass

            # Upload reference images if provided
            if ref_images is not None:
                ref_image_urls = ImageUtils.prepare_images(ref_images)
                if ref_image_urls:
                    arguments["ref_image_urls"] = ref_image_urls

            # Upload first frame if provided
            if first_frame is not None:
                first_frame_url = ImageUtils.upload_image(first_frame)
                if first_frame_url:
                    arguments["first_frame_image_url"] = first_frame_url

            # Upload last frame if provided
            if last_frame is not None:
                last_frame_url = ImageUtils.upload_image(last_frame)
                if last_frame_url:
                    arguments["last_frame_image_url"] = last_frame_url

            # Submit to API with task-specific endpoint
            results = ApiHandler.submit_multiple_and_get_results(
                f"fal-ai/wan-22-vace-fun-a14b/{task}",
                arguments,
                variations
            )

            # Return list of outputs
            video_url = [r["video"]["url"] for r in results]
            frames_zip_url = [r.get("frames_zip", {}).get("url", "") for r in results] if return_frames_zip else [""] * len(results)

            return (video_url, frames_zip_url)

        except Exception as e:
            return ApiHandler.handle_video_generation_error("wan-22-vace-fun-a14b", str(e))


# =============================================================================
# HYPER CUSTOM DY ENDPOINTS
# =============================================================================
# These are specialized, highly customizable DY endpoints with extensive
# parameter sets for advanced video generation and manipulation tasks.
# =============================================================================

class DYWanFun22Node:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "architecture": (["vace", "control"], {"default": "vace"}),
                "control_video": ("VIDEO", {"default": None}),
                "ref_image": ("IMAGE", {"default": None}),
            },
            "optional": {
                "turbo_mode": ("BOOLEAN", {"default": True}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "image_size": (["custom", "square_hd", "square", "portrait_4_3", "portrait_16_9", "landscape_4_3", "landscape_16_9"], {"default": "custom"}),
                "custom_width": ("INT", {"default": 1280, "min": 0, "max": 8192, "step": 8}),
                "custom_height": ("INT", {"default": 720, "min": 0, "max": 8192, "step": 8}),
                "num_frames": ("INT", {"default": 81, "min": 1, "max": 1000}),
                "frames_per_second": ("INT", {"default": 16, "min": 5, "max": 30}),
                "num_inference_steps": ("INT", {"default": 4, "min": 1, "max": 100}),
                "guidance_scale": ("FLOAT", {"default": 1, "min": 0.0, "max": 10.0}),
                "seed": ("INT", {"default": -1, "min": 0, "max": 2147483647}),
                "sampler": (["uni_pc", "dpmpp_2m", "dpmpp_2m_sde", "euler", "euler_ancestral"], {"default": "uni_pc"}),
                "shift": ("INT", {"default": 5, "min": 0, "max": 10}),
                "variations": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                # Control strengths
                "vace_mask_video": ("VIDEO", {"default": None}),
                "preprocess_all_maps": ("BOOLEAN", {"default": True}),
                "strength_vace": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0}),
                "pose_strength": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0}),
                "pose_video": ("VIDEO", {"default": None, "multiple": True}),
                "depth_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0}),
                "depth_video": ("VIDEO", {"default": None, "multiple": True}),
                "normal_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0}),
                "normal_video": ("VIDEO", {"default": None, "multiple": True}),
                "canny_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0}),
                "canny_video": ("VIDEO", {"default": None, "multiple": True}),
                # Advanced settings
                "num_interpolated_frames": ("INT", {"default": 0, "min": 0, "max": 5}),
                "temporal_downsample_factor": ("INT", {"default": 0, "min": 0, "max": 5}),
                "enable_auto_downsample": ("BOOLEAN", {"default": False}),
                "auto_downsample_min_fps": ("INT", {"default": 8, "min": 0, "max": 60}),
                "return_frames_zip": ("BOOLEAN", {"default": False}),
                "lora_path_1": ("STRING", {"default": ""}),
                "lora_strength_1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "lora_transformer_1": (["high", "low", "both"], {"default": "high"}),
                "lora_path_2": ("STRING", {"default": ""}),
                "lora_strength_2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "lora_transformer_2": (["high", "low", "both"], {"default": "high"}),
                "lora_path_3": ("STRING", {"default": ""}),
                "lora_strength_3": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "lora_transformer_3": (["high", "low", "both"], {"default": "high"}),
                "lora_path_4": ("STRING", {"default": ""}),
                "lora_strength_4": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "lora_transformer_4": (["high", "low", "both"], {"default": "high"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("video_url", "frames_zip_url",)
    OUTPUT_IS_LIST = (True, True,)
    FUNCTION = "generate_video"
    CATEGORY = "FAL/VideoGeneration/DY"

    def generate_video(
        self,
        architecture="vace",
        control_video=None,
        ref_image=None,
        turbo_mode=True,
        prompt="",
        negative_prompt="",
        image_size="custom",
        custom_width=1280,
        custom_height=720,
        num_frames=81,
        frames_per_second=16,
        num_inference_steps=4,
        guidance_scale=1.0,
        seed=-1,
        sampler="uni_pc",
        shift=5,
        variations=1,
        vace_mask_video=None,
        preprocess_all_maps=True,
        strength_vace=1.0,
        pose_strength=0.6,
        pose_video=None,
        depth_strength=0.0,
        depth_video=None,
        normal_strength=0.0,
        normal_video=None,
        canny_strength=0.0,
        canny_video=None,
        num_interpolated_frames=0,
        temporal_downsample_factor=0,
        enable_auto_downsample=False,
        auto_downsample_min_fps=8,
        return_frames_zip=False,
        lora_path_1="",
        lora_strength_1=1.0,
        lora_transformer_1="high",
        lora_path_2="",
        lora_strength_2=1.0,
        lora_transformer_2="high",
        lora_path_3="",
        lora_strength_3=1.0,
        lora_transformer_3="high",
        lora_path_4="",
        lora_strength_4=1.0,
        lora_transformer_4="high",
    ):
        try:
            if ref_image is None:
                return ApiHandler.handle_video_generation_error(
                    "dy-wan-fun-22", "Reference image is required."
                )

            # Upload reference image
            ref_image_url = ImageUtils.upload_image(ref_image)
            if not ref_image_url:
                return ApiHandler.handle_video_generation_error(
                    "dy-wan-fun-22", "Failed to upload reference image"
                )

            # Build arguments
            arguments = {
                "ref_image_url": ref_image_url,
                "architecture": architecture,
                "turbo_mode": turbo_mode,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "num_frames": num_frames,
                "frames_per_second": frames_per_second,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "sampler": sampler,
                "shift": shift,
                "preprocess_all_maps": preprocess_all_maps,
                "return_frames_zip": return_frames_zip,
                "pose_strength": pose_strength,
                "depth_strength": depth_strength,
                "normal_strength": normal_strength,
                "canny_strength": canny_strength
            }

            # Set seed
            if seed != -1:
                arguments["seed"] = seed

            # Handle image_size - use custom dimensions if provided, otherwise use aspect_ratio preset
            if image_size == "custom":
                arguments["image_size"] = {"width": custom_width, "height": custom_height}
            else:
                arguments["image_size"] = image_size

            # Upload control video if provided
            if control_video is not None:
                control_video_url = ImageUtils.upload_file(control_video.get_stream_source())
                if control_video_url:
                    arguments["control_video_url"] = control_video_url

            # Handle VACE strength
            if architecture == "vace":
                arguments["strength_vace"] = strength_vace

            # Handle VACE mask video
            if vace_mask_video is not None:
                vace_mask_video_url = ImageUtils.upload_file(vace_mask_video.get_stream_source())
                if vace_mask_video_url:
                    arguments["vace_mask_video_url"] = vace_mask_video_url

            # Upload and add pose video/strength if provided
            if pose_video is not None:
                pose_video_url = ImageUtils.upload_file(pose_video.get_stream_source())
                if pose_video_url:
                    arguments["pose_video_url"] = pose_video_url

            # Upload and add depth video/strength if provided
            if depth_video is not None:
                depth_video_url = ImageUtils.upload_file(depth_video.get_stream_source())
                if depth_video_url:
                    arguments["depth_video_url"] = depth_video_url

            # Upload and add normal video/strength if provided
            if normal_video is not None:
                normal_video_url = ImageUtils.upload_file(normal_video.get_stream_source())
                if normal_video_url:
                    arguments["normal_video_url"] = normal_video_url

            # Upload and add canny video/strength if provided
            if canny_video is not None:
                canny_video_url = ImageUtils.upload_file(canny_video.get_stream_source())
                if canny_video_url:
                    arguments["canny_video_url"] = canny_video_url

            # Add advanced interpolation settings if non-default
            if num_interpolated_frames > 0:
                arguments["num_interpolated_frames"] = num_interpolated_frames
            if temporal_downsample_factor > 0:
                arguments["temporal_downsample_factor"] = temporal_downsample_factor
            if enable_auto_downsample:
                arguments["enable_auto_downsample"] = enable_auto_downsample
                arguments["auto_downsample_min_fps"] = auto_downsample_min_fps

            # Add LoRAs if provided
            loras = []
            if lora_path_1:
                loras.append({"url": lora_path_1, "strength": lora_strength_1, "transformer": lora_transformer_1})
            if lora_path_2:
                loras.append({"url": lora_path_2, "strength": lora_strength_2, "transformer": lora_transformer_2})
            if lora_path_3:
                loras.append({"url": lora_path_3, "strength": lora_strength_3, "transformer": lora_transformer_3})
            if lora_path_4:
                loras.append({"url": lora_path_4, "strength": lora_strength_4, "transformer": lora_transformer_4})
            if loras:
                arguments["loras"] = loras

            # Submit to API
            results = ApiHandler.submit_multiple_and_get_results(
                "fal-ai/dy-wan-fun-22",
                arguments,
                variations
            )

            # Return list of outputs
            video_url = [r["video"]["url"] for r in results]
            frames_zip_url = [r.get("frames_zip", {}).get("url", "") for r in results] if return_frames_zip else [""] * len(results)

            return (video_url, frames_zip_url)

        except Exception as e:
            return ApiHandler.handle_video_generation_error("dy-wan-fun-22", str(e))



class DYWanUpscalerNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
            },
            "optional": {
                "video": ("VIDEO", {"default": None}),
                "video_url": ("STRING", {"default": ""}),
                "prompt": ("STRING", {"default": "cinematic composition, realistic high-quality photo, RAW photo, masterpiece, photorealistic, 8k", "multiline": True}),
                "negative_prompt": ("STRING", {"default": "oversaturated, overexposed, static, blurry details", "multiline": True}),
                "strength": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 1.0, "step": 0.01}),
                "guidance_scale": ("FLOAT", {"default": 3.5, "min": 1.0, "max": 20.0, "step": 0.1}),
                "num_inference_steps": ("INT", {"default": 10, "min": 1, "max": 50}),
                "seed": ("INT", {"default": -1, "min": 0, "max": 2147483647}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 120}),
                "image_size": (["custom", "landscape_16_9", "landscape_4_3", "portrait_16_9", "portrait_4_3", "square", "square_hd"], {"default": "custom"}),
                "custom_width": ("INT", {"default": 1920, "min": 0, "max": 8192, "step": 8}),
                "custom_height": ("INT", {"default": 1080, "min": 0, "max": 8192, "step": 8}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_url",)
    FUNCTION = "upscale_video"
    CATEGORY = "FAL/VideoGeneration/DY"

    def upscale_video(
        self,
        video=None,
        video_url="",
        prompt="cinematic composition, realistic high-quality photo, RAW photo, masterpiece, photorealistic, 8k",
        negative_prompt="oversaturated, overexposed, static, blurry details",
        strength=0.02,
        guidance_scale=3.5,
        num_inference_steps=10,
        seed=-1,
        fps=24,
        image_size="custom",
        custom_width=1920,
        custom_height=1080
    ):
        try:
            if video is None and video_url == "":
                return ApiHandler.handle_video_generation_error(
                    "dy-wan-upscaler", "Video input is required."
                )

            # Upload video
            if video:
                video_url = ImageUtils.upload_file(video.get_stream_source())
                if not video_url:
                    return ApiHandler.handle_video_generation_error(
                        "dy-wan-upscaler", "Failed to upload video"
                    )

            # Build arguments
            arguments = {
                "video_url": video_url,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "strength": strength,
                "guidance_scale": guidance_scale,
                "num_inference_steps": num_inference_steps,
                "fps": fps,
            }

            # Set seed
            if seed != -1:
                arguments["seed"] = seed

            # Handle image_size - use custom dimensions if provided, otherwise use preset
            if image_size == "custom":
                arguments["image_size"] = {"width": custom_width, "height": custom_height}
            else:
                arguments["image_size"] = image_size

            # Submit to API
            result = ApiHandler.submit_and_get_result(
                "fal-ai/dy-wan-upscaler",
                arguments,
            )

            video_url = result["video"]["url"]
            return (video_url,)

        except Exception as e:
            return ApiHandler.handle_video_generation_error("dy-wan-upscaler", str(e))


# =============================================================================
# END HYPER CUSTOM DY ENDPOINTS
# =============================================================================


class PixverseSwapNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"default": None}),
                "mode": (["person", "object", "background"], {"default": "person"}),
                "keyframe_id": ("INT", {"default": 1}),
                "quality": (
                    ["360p", "540p", "720p"],
                    {"default": "720p"}
                ),
                "original_sound_switch": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "video": ("VIDEO", {"default": None}),
                "input_video_url": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_url",)
    FUNCTION = "edit_video"
    CATEGORY = "FAL/VideoGeneration"

    def edit_video(
        self,
        video=None,
        input_video_url="",
        image=None,
        keyframe_id=1,
        quality="720p",
        original_sound_switch=True,
        mode="person"):
        try:
            if video is None and input_video_url is "":
                return ApiHandler.handle_video_generation_error(
                    "pixverse-swap", "Video or Video Frames input is required."
                )
            if video is None and input_video_url is not "":
                video_url = input_video_url
            else:
                video_url = ImageUtils.upload_file(video.get_stream_source())
            if not video_url:
                return ApiHandler.handle_video_generation_error(
                    "pixverse-swap", "Failed to upload video"
                )

          
            image_url = ImageUtils.upload_image(image)
            if not image_url:
                return ApiHandler.handle_video_generation_error(
                    "pixverse-swap", "Failed to upload image"
                )

            arguments={
        "video_url": video_url,
        "image_url": image_url,
        "keyframe_id": keyframe_id,
        "quality": quality,
        "original_sound_switch": original_sound_switch,
        "mode": mode
    }
            result = ApiHandler.submit_and_get_result(
                "fal-ai/pixverse/swap",
                arguments,
            )

            return (result["video"]["url"],)

        except Exception as e:
            return ApiHandler.handle_video_generation_error("pixverse-swap", str(e))




class KreaWan14bVideoToVideoNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "enable_prompt_expansion": ("BOOLEAN", {"default": True}),
                "strength": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 3.0}),
            },
            "optional": {
                "video": ("VIDEO", {"default": None}),
                "input_video_url": ("STRING", {"default": ""}),
                "seed": ("INT", {"default": 42}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_url",)
    FUNCTION = "edit_video"
    CATEGORY = "FAL/VideoGeneration"

    def edit_video(
        self,
        prompt="",
        strength=0.85,
        video=None,
        input_video_url="",
        seed=24,
        enable_prompt_expansion=True,
    ):
        try:
            if video is None and input_video_url is "":
                return ApiHandler.handle_video_generation_error(
                    "krea-wan-14b", "Video or Video URL input is required."
                )
            if video is None and input_video_url is not "":
                video_url = input_video_url
            else:
                video_url = ImageUtils.upload_file(video.get_stream_source())
            if not video_url:
                return ApiHandler.handle_video_generation_error(
                    "krea-wan-14b", "Failed to upload video"
                )
            arguments={
                    "prompt": prompt,
                    "strength": strength,
                    "enable_prompt_expansion": enable_prompt_expansion,
                    "video_url": video_url,
                    "seed": seed
                }
            result = ApiHandler.submit_and_get_result(
                "fal-ai/krea-wan-14b/video-to-video",
                arguments,
            )

            return (result["video"]["url"],)

        except Exception as e:
            return ApiHandler.handle_video_generation_error("krea-wan-14b", str(e))




class InfinityStarTextToVideoNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "negative_prompt": ("STRING", {"default": "low quality.", "multiline": True}),
                "aspect_ratio": (
                    ["16:9",  "9:6","1:1"],
                    {"default": "16:9"}
                ),
                "guidance_scale": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 10}),
            },
            "optional": {
                "tau_video": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 2}),
                "enhance_prompt": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 42}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_url",)
    FUNCTION = "edit_video"
    CATEGORY = "FAL/VideoGeneration"

    def edit_video(
        self,
        prompt="",
        negative_prompt="low quality.",
        enhance_prompt=True,
        seed=24,
        aspect_ratio="16:9",
        guidance_scale=7.5,
        tau_video=0.4,
    ):
        try:
            arguments={
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "aspect_ratio": aspect_ratio,
                    "guidance_scale": guidance_scale,
                    "enhance_prompt": enhance_prompt,
                    "tau_video": tau_video,
                    "seed": seed
                }
            result = ApiHandler.submit_and_get_result(
                "fal-ai/infinity-star/text-to-video",
                arguments,
            )

            return (result["video"]["url"],)

        except Exception as e:
            return ApiHandler.handle_video_generation_error("infinity-star-text-to-video", str(e))



class CombinedVideoGenerationNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "image": ("IMAGE",),
                "kling_duration": (["5", "10"], {"default": "5"}),
                "kling_luma_aspect_ratio": (
                    ["16:9", "9:16", "1:1"],
                    {"default": "16:9"},
                ),
                "luma_loop": ("BOOLEAN", {"default": False}),
                "veo2_aspect_ratio": (
                    ["auto", "auto_prefer_portrait", "16:9", "9:16"],
                    {"default": "auto"},
                ),
                "veo2_duration": (["5s", "6s", "7s", "8s"], {"default": "5s"}),
                "enable_klingpro": ("BOOLEAN", {"default": True}),
                "enable_klingmaster": ("BOOLEAN", {"default": True}),
                "enable_minimax": ("BOOLEAN", {"default": True}),
                "enable_luma": ("BOOLEAN", {"default": True}),
                "enable_veo2": ("BOOLEAN", {"default": True}),
                "enable_wanpro": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "klingpro_v1.6_video",
        "klingmaster_v2.0_video",
        "minimax_video",
        "luma_video",
        "veo2_video",
        "wanpro_video",
    )
    FUNCTION = "generate_videos"
    CATEGORY = "FAL/VideoGeneration"

    async def generate_klingpro_video(
        self, client, prompt, image_url, kling_duration, kling_luma_aspect_ratio
    ):
        try:
            arguments = {
                "prompt": prompt,
                "image_url": image_url,
                "duration": kling_duration,
                "aspect_ratio": kling_luma_aspect_ratio,
            }
            handler = await client.submit(
                "fal-ai/kling-video/v1.6/pro/image-to-video", arguments=arguments
            )
            while True:
                result = await handler.get()
                if "video" in result and "url" in result["video"]:
                    return result["video"]["url"]
                elif result.get("status") == "FAILED":
                    raise Exception("Video generation failed")
                await asyncio.sleep(1)
        except Exception as e:
            print(f"Error generating KlingPro video: {str(e)}")
            return "Error: Unable to generate KlingPro video."

    async def generate_klingmaster_video(
        self, client, prompt, image_url, kling_duration, kling_luma_aspect_ratio
    ):
        try:
            arguments = {
                "prompt": prompt,
                "image_url": image_url,
                "duration": kling_duration,
                "aspect_ratio": kling_luma_aspect_ratio,
            }
            handler = await client.submit(
                "fal-ai/kling-video/v2/master/image-to-video", arguments=arguments
            )
            while True:
                result = await handler.get()
                if "video" in result and "url" in result["video"]:
                    return result["video"]["url"]
                elif result.get("status") == "FAILED":
                    raise Exception("Video generation failed")
                await asyncio.sleep(1)
        except Exception as e:
            print(f"Error generating KlingMaster video: {str(e)}")
            return "Error: Unable to generate KlingMaster video."

    async def generate_minimax_video(self, client, prompt, image_url):
        try:
            arguments = {
                "prompt": prompt,
                "image_url": image_url,
            }
            handler = await client.submit(
                "fal-ai/minimax/video-01-live/image-to-video", arguments=arguments
            )
            while True:
                result = await handler.get()
                if "video" in result and "url" in result["video"]:
                    return result["video"]["url"]
                elif result.get("status") == "FAILED":
                    raise Exception("Video generation failed")
                await asyncio.sleep(1)
        except Exception as e:
            print(f"Error generating MiniMax video: {str(e)}")
            return "Error: Unable to generate MiniMax video."

    async def generate_luma_video(
        self, client, prompt, image_url, kling_luma_aspect_ratio, luma_loop
    ):
        try:
            arguments = {
                "prompt": prompt,
                "image_url": image_url,
                "aspect_ratio": kling_luma_aspect_ratio,
                "loop": luma_loop,
            }
            handler = await client.submit(
                "fal-ai/luma-dream-machine/ray-2/image-to-video", arguments=arguments
            )
            while True:
                result = await handler.get()
                if "video" in result and "url" in result["video"]:
                    return result["video"]["url"]
                elif result.get("status") == "FAILED":
                    raise Exception("Video generation failed")
                await asyncio.sleep(1)
        except Exception as e:
            print(f"Error generating Luma video: {str(e)}")
            return "Error: Unable to generate Luma video."

    async def generate_veo2_video(
        self, client, prompt, image_url, aspect_ratio, duration
    ):
        try:
            arguments = {
                "prompt": prompt,
                "image_url": image_url,
                "aspect_ratio": aspect_ratio,
                "duration": duration,
            }
            handler = await client.submit(
                "fal-ai/veo2/image-to-video", arguments=arguments
            )
            while True:
                result = await handler.get()
                if "video" in result and "url" in result["video"]:
                    return result["video"]["url"]
                elif result.get("status") == "FAILED":
                    raise Exception("Video generation failed")
                await asyncio.sleep(1)
        except Exception as e:
            print(f"Error generating Veo2 video: {str(e)}")
            return "Error: Unable to generate Veo2 video."

    async def generate_wanpro_video(self, client, prompt, image_url):
        try:
            arguments = {
                "prompt": prompt,
                "image_url": image_url,
                "enable_safety_checker": True,
                "seed": None,  # Let the API choose a random seed
            }

            handler = await client.submit(
                "fal-ai/wan-pro/image-to-video", arguments=arguments
            )
            while True:
                result = await handler.get()
                if "video" in result and "url" in result["video"]:
                    return result["video"]["url"]
                elif result.get("status") == "FAILED":
                    raise Exception("Video generation failed")
                await asyncio.sleep(1)
        except Exception as e:
            print(f"Error generating Wan Pro video: {str(e)}")
            return "Error: Unable to generate Wan Pro video."

    async def generate_all_videos(
        self,
        prompt,
        image_url,
        kling_duration,
        kling_luma_aspect_ratio,
        luma_loop,
        veo2_aspect_ratio,
        veo2_duration,
        enable_klingpro,
        enable_klingmaster,
        enable_minimax,
        enable_luma,
        enable_veo2,
        enable_wanpro,
    ):
        try:
            tasks = []
            results = [None] * 6  # Initialize results list with None values

            # Create async client with the same key as the sync client
            client = AsyncClient(key=fal_config.get_key())

            # Add tasks based on enabled services
            if enable_klingpro:
                tasks.append(
                    self.generate_klingpro_video(
                        client,
                        prompt,
                        image_url,
                        kling_duration,
                        kling_luma_aspect_ratio,
                    )
                )
            else:
                tasks.append(None)

            if enable_klingmaster:
                tasks.append(
                    self.generate_klingmaster_video(
                        client,
                        prompt,
                        image_url,
                        kling_duration,
                        kling_luma_aspect_ratio,
                    )
                )
            else:
                tasks.append(None)

            if enable_minimax:
                tasks.append(self.generate_minimax_video(client, prompt, image_url))
            else:
                tasks.append(None)

            if enable_luma:
                tasks.append(
                    self.generate_luma_video(
                        client, prompt, image_url, kling_luma_aspect_ratio, luma_loop
                    )
                )
            else:
                tasks.append(None)

            if enable_veo2:
                tasks.append(
                    self.generate_veo2_video(
                        client, prompt, image_url, veo2_aspect_ratio, veo2_duration
                    )
                )
            else:
                tasks.append(None)

            if enable_wanpro:
                tasks.append(self.generate_wanpro_video(client, prompt, image_url))
            else:
                tasks.append(None)

            # Filter out None tasks and execute them
            valid_tasks = [task for task in tasks if task is not None]
            if valid_tasks:
                completed_results = await asyncio.gather(*valid_tasks)

                # Place results in their correct positions
                result_index = 0
                for i, task in enumerate(tasks):
                    if task is not None:
                        results[i] = completed_results[result_index]
                        result_index += 1
                    else:
                        results[i] = "Service disabled"

            return results
        except Exception as e:
            print(f"Error in generate_all_videos: {str(e)}")
            return ["Error: Unable to generate videos."] * 6

    def generate_videos(
        self,
        prompt,
        image,
        kling_duration,
        kling_luma_aspect_ratio,
        luma_loop,
        veo2_aspect_ratio,
        veo2_duration,
        enable_klingpro,
        enable_klingmaster,
        enable_minimax,
        enable_luma,
        enable_veo2,
        enable_wanpro,
    ):
        try:
            # Upload image once to be used by all services
            image_url = ImageUtils.upload_image(image)
            if not image_url:
                return ("Error: Unable to upload image.",) * 6

            # Create event loop for async operations
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Run all video generations concurrently
            results = loop.run_until_complete(
                self.generate_all_videos(
                    prompt,
                    image_url,
                    kling_duration,
                    kling_luma_aspect_ratio,
                    luma_loop,
                    veo2_aspect_ratio,
                    veo2_duration,
                    enable_klingpro,
                    enable_klingmaster,
                    enable_minimax,
                    enable_luma,
                    enable_veo2,
                    enable_wanpro,
                )
            )
            loop.close()

            return tuple(results)
        except Exception as e:
            print(f"Error in combined video generation: {str(e)}")
            return ("Error: Unable to generate videos.",) * 6


class VideoUpscalerNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_url": ("STRING", {"default": ""}),
                "scale": (
                    "FLOAT",
                    {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.5},
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "upscale_video"
    CATEGORY = "FAL/VideoGeneration"

    def upscale_video(self, video_url, scale):
        try:
            arguments = {"video_url": video_url, "scale": scale}

            result = ApiHandler.submit_and_get_result(
                "fal-ai/video-upscaler", arguments
            )
            video_url = result["video"]["url"]
            return (video_url,)
        except Exception as e:
            return ApiHandler.handle_video_generation_error("video-upscaler", str(e))

class UploadVideoNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_url",)
    FUNCTION = "upload_video"
    CATEGORY = "video"

    def upload_video(self, video):
        video_url = ImageUtils.upload_file(video.get_stream_source())
        return (video_url,)
    
class UploadFileNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_url",)
    FUNCTION = "upload_file"
    CATEGORY = "video"

    def upload_file(self, path):
        file_url = ImageUtils.upload_file(path)
        return (file_url,)

class LoadVideoURL:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {"default": "https://example.com/video.mp4"}),
                "force_rate": ("INT", {"default": 0, "min": 0, "max": 60, "step": 1}),
                "force_size": (
                    [
                        "Disabled",
                        "Custom Height",
                        "Custom Width",
                        "Custom",
                        "256x?",
                        "?x256",
                        "256x256",
                        "512x?",
                        "?x512",
                        "512x512",
                    ],
                ),
                "custom_width": (
                    "INT",
                    {"default": 512, "min": 0, "max": 8192, "step": 8},
                ),
                "custom_height": (
                    "INT",
                    {"default": 512, "min": 0, "max": 8192, "step": 8},
                ),
                "frame_load_cap": (
                    "INT",
                    {"default": 0, "min": 0, "max": 1000000, "step": 1},
                ),
                "skip_first_frames": (
                    "INT",
                    {"default": 0, "min": 0, "max": 1000000, "step": 1},
                ),
                "select_every_nth": (
                    "INT",
                    {"default": 1, "min": 1, "max": 1000000, "step": 1},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "VHS_VIDEOINFO")
    RETURN_NAMES = ("frames", "frame_count", "video_info")
    FUNCTION = "load_video_from_url"
    CATEGORY = "video"

    def load_video_from_url(
        self,
        url,
        force_rate,
        force_size,
        custom_width,
        custom_height,
        frame_load_cap,
        skip_first_frames,
        select_every_nth,
    ):
        # Download the video to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            response = requests.get(url, stream=True)
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)
            temp_file_path = temp_file.name

        # Load the video using OpenCV
        cap = cv2.VideoCapture(temp_file_path)

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps

        # Calculate target size
        if force_size != "Disabled":
            if force_size == "Custom Width":
                new_height = int(height * (custom_width / width))
                new_width = custom_width
            elif force_size == "Custom Height":
                new_width = int(width * (custom_height / height))
                new_height = custom_height
            elif force_size == "Custom":
                new_width, new_height = custom_width, custom_height
            else:
                target_width, target_height = map(
                    int, force_size.replace("?", "0").split("x")
                )
                if target_width == 0:
                    new_width = int(width * (target_height / height))
                    new_height = target_height
                else:
                    new_height = int(height * (target_width / width))
                    new_width = target_width
        else:
            new_width, new_height = width, height

        frames = []
        frame_count = 0

        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            if i < skip_first_frames:
                continue

            if (i - skip_first_frames) % select_every_nth != 0:
                continue

            if force_size != "Disabled":
                frame = cv2.resize(frame, (new_width, new_height))

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(frame).float() / 255.0
            frames.append(frame)

            frame_count += 1

            if frame_load_cap > 0 and frame_count >= frame_load_cap:
                break

        cap.release()
        os.unlink(temp_file_path)

        frames = torch.stack(frames)

        video_info = {
            "source_fps": fps,
            "source_frame_count": total_frames,
            "source_duration": duration,
            "source_width": width,
            "source_height": height,
            "loaded_fps": fps if force_rate == 0 else force_rate,
            "loaded_frame_count": frame_count,
            "loaded_duration": frame_count / (fps if force_rate == 0 else force_rate),
            "loaded_width": new_width,
            "loaded_height": new_height,
        }

        return (frames, frame_count, video_info)


class SeedanceImageToVideoNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "image": ("IMAGE",),
                "resolution": (["480p", "720p"], {"default": "720p"}),
                "duration": (["5", "10"], {"default": "5"}),
                "camera_fixed": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_video"
    CATEGORY = "FAL/VideoGeneration"

    def generate_video(self, prompt, image, resolution, duration, camera_fixed, seed=-1):
        try:
            image_url = ImageUtils.upload_image(image)
            if not image_url:
                return ApiHandler.handle_video_generation_error(
                    "fal-ai/bytedance/seedance/v1/lite/image-to-video",
                    "Failed to upload image",
                )

            arguments = {
                "prompt": prompt,
                "image_url": image_url,
                "resolution": resolution,
                "duration": duration,
                "camera_fixed": camera_fixed,
            }

            # Only add seed if it's not -1 (random)
            if seed != -1:
                arguments["seed"] = seed

            result = ApiHandler.submit_and_get_result(
                "fal-ai/bytedance/seedance/v1/lite/image-to-video", arguments
            )
            video_url = result["video"]["url"]
            return (video_url,)
        except Exception as e:
            return ApiHandler.handle_video_generation_error(
                "fal-ai/bytedance/seedance/v1/lite/image-to-video", str(e)
            )


class SeedanceTextToVideoNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "aspect_ratio": (["16:9", "4:3", "1:1", "9:21"], {"default": "16:9"}),
                "resolution": (["480p", "720p"], {"default": "720p"}),
                "duration": (["5", "10"], {"default": "5"}),
                "camera_fixed": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_video"
    CATEGORY = "FAL/VideoGeneration"

    def generate_video(self, prompt, aspect_ratio, resolution, duration, camera_fixed, seed=-1):
        try:
            arguments = {
                "prompt": prompt,
                "aspect_ratio": aspect_ratio,
                "resolution": resolution,
                "duration": duration,
                "camera_fixed": camera_fixed,
            }

            # Only add seed if it's not -1 (random)
            if seed != -1:
                arguments["seed"] = seed

            result = ApiHandler.submit_and_get_result(
                "fal-ai/bytedance/seedance/v1/lite/text-to-video", arguments
            )
            video_url = result["video"]["url"]
            return (video_url,)
        except Exception as e:
            return ApiHandler.handle_video_generation_error(
                "fal-ai/bytedance/seedance/v1/lite/text-to-video", str(e)
            )


class SeedanceProImageToVideoNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "image": ("IMAGE",),
                "duration": (["2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12",], {"default": "5"}),
            },
            "optional": {
                "end_image": ("IMAGE",),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "cfg_scale": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1}),
                "variations": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1})
            },
        }

    RETURN_TYPES = ("STRING",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "generate_video"
    CATEGORY = "FAL/VideoGeneration"

    def generate_video(self, prompt, image, duration, end_image=None, negative_prompt="", cfg_scale=0.5, variations=1):
        try:
            image_url = ImageUtils.upload_image(image)
            if not image_url:
                return ApiHandler.handle_video_generation_error(
                    "fal-ai/bytedance/seedance/v1/pro/image-to-video",
                    "Failed to upload image",
                )

            arguments = {
                "prompt": prompt,
                "image_url": image_url,
                "duration": duration,
                "negative_prompt": negative_prompt,
                "cfg_scale": cfg_scale
            }

            # Handle optional End image
            if end_image is not None:
                end_image_url = ImageUtils.upload_image(end_image)
                if end_image_url:
                    arguments["end_image_url"] = end_image_url
                else:
                    return ApiHandler.handle_video_generation_error(
                        "seedance/v1/pro/image-to-video", "Failed to upload end image"
                    )

            results = ApiHandler.submit_multiple_and_get_results(
                "fal-ai/bytedance/seedance/v1/pro/image-to-video", arguments, variations
            )

            # Return list of video URLs
            return ([r["video"]["url"] for r in results],)
        
        except Exception as e:
            return ApiHandler.handle_video_generation_error(
                "fal-ai/bytedance/seedance/v1/pro/image-to-video", str(e)
            )


class Veo3Node:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "aspect_ratio": (["16:9", "9:16", "1:1"], {"default": "16:9"}),
                "duration": (["8s"], {"default": "8s"}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "enhance_prompt": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "generate_audio": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_video"
    CATEGORY = "FAL/VideoGeneration"

    def generate_video(
        self,
        prompt,
        aspect_ratio,
        duration,
        negative_prompt="",
        enhance_prompt=True,
        seed=-1,
        generate_audio=True,
    ):
        arguments = {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "duration": duration,
            "negative_prompt": negative_prompt,
            "enhance_prompt": enhance_prompt,
            "generate_audio": generate_audio,
        }

        if seed != -1:
            arguments["seed"] = seed

        try:
            result = ApiHandler.submit_and_get_result("fal-ai/veo3", arguments)
            video_url = result["video"]["url"]
            return (video_url,)
        except Exception as e:
            return ApiHandler.handle_video_generation_error("veo3", str(e))


class FalKling21ProImageToVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "image": ("IMAGE",),
                "duration": (["5", "10"], {"default": "5"}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"default": "blur, distort, and low quality", "multiline": True}),
                "cfg_scale": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1}),
                "tail_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_video"
    CATEGORY = "FAL/VideoGeneration"

    def generate_video(self, prompt, image, duration, negative_prompt="blur, distort, and low quality", cfg_scale=0.5, tail_image=None):
        try:
            image_url = ImageUtils.upload_image(image)
            if not image_url:
                return ApiHandler.handle_video_generation_error(
                    "kling-video/v2.1/pro", "Failed to upload image"
                )

            arguments = {
                "prompt": prompt,
                "image_url": image_url,
                "duration": duration,
                "negative_prompt": negative_prompt,
                "cfg_scale": cfg_scale,
            }

            # Handle optional tail image
            if tail_image is not None:
                tail_image_url = ImageUtils.upload_image(tail_image)
                if tail_image_url:
                    arguments["tail_image_url"] = tail_image_url
                else:
                    return ApiHandler.handle_video_generation_error(
                        "kling-video/v2.1/pro", "Failed to upload tail image"
                    )

            result = ApiHandler.submit_and_get_result(
                "fal-ai/kling-video/v2.1/pro/image-to-video", arguments
            )
            video_url = result["video"]["url"]
            return (video_url,)
        except Exception as e:
            return ApiHandler.handle_video_generation_error(
                "kling-video/v2.1/pro", str(e)
            )


class FalKling25TurboProImageToVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "image": ("IMAGE",),
                "duration": (["5", "10"], {"default": "5"}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"default": "blur, distort, and low quality", "multiline": True}),
                "cfg_scale": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1}),
                "tail_image": ("IMAGE",),
                "variations": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1})
            },
        }

    RETURN_TYPES = ("STRING",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "generate_video"
    CATEGORY = "FAL/VideoGeneration"

    def generate_video(self, prompt, image, duration, negative_prompt="blur, distort, and low quality", cfg_scale=0.5, tail_image=None, variations=1):
        try:
            image_url = ImageUtils.upload_image(image)
            if not image_url:
                return ApiHandler.handle_video_generation_error(
                    "kling-video/v2.5-turbo/pro", "Failed to upload image"
                )

            arguments = {
                "prompt": prompt,
                "image_url": image_url,
                "duration": duration,
                "negative_prompt": negative_prompt,
                "cfg_scale": cfg_scale,
            }

            # Handle optional tail image
            if tail_image is not None:
                tail_image_url = ImageUtils.upload_image(tail_image)
                if tail_image_url:
                    arguments["tail_image_url"] = tail_image_url
                else:
                    return ApiHandler.handle_video_generation_error(
                        "kling-video/v2.5-turbo/pro", "Failed to upload tail image"
                    )

            results = ApiHandler.submit_multiple_and_get_results(
                "fal-ai/kling-video/v2.5-turbo/pro/image-to-video", arguments, variations
            )

            # Return list of video URLs
            return ([r["video"]["url"] for r in results],)
            
        except Exception as e:
            return ApiHandler.handle_video_generation_error(
                "kling-video/v2.5-turbo/pro", str(e)
            )


class FalSora2ProImageToVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "image": ("IMAGE",),
            },
            "optional": {
                "resolution": (["auto", "720p", "1080p"], {"default": "auto"}),
                "aspect_ratio": (["auto", "9:16", "16:9"], {"default": "auto"}),
                "duration": ([4, 8, 12], {"default": 4}),
                "delete_video": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_video"
    CATEGORY = "FAL/VideoGeneration"

    def generate_video(self, prompt, image, resolution="auto", aspect_ratio="auto", duration=4, delete_video=True):
        try:
            image_url = ImageUtils.upload_image(image)
            if not image_url:
                return ApiHandler.handle_video_generation_error(
                    "sora-2/pro", "Failed to upload image"
                )

            arguments = {
                "prompt": prompt,
                "image_url": image_url,
                "resolution": resolution,
                "aspect_ratio": aspect_ratio,
                "duration": duration,
                "delete_video": delete_video,
            }

            result = ApiHandler.submit_and_get_result(
                "fal-ai/sora-2/image-to-video/pro", arguments
            )
            video_url = result["video"]["url"]
            return (video_url,)
        except Exception as e:
            return ApiHandler.handle_video_generation_error(
                "sora-2/pro", str(e)
            )


class FalVeo31FirstLastFrameToVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "first_frame": ("IMAGE",),
            },
            "optional": {
                "last_frame": ("IMAGE",),
                "duration": (["4s", "6s", "8s"], {"default": "8s"}),
                "aspect_ratio": (["auto", "9:16", "16:9", "1:1"], {"default": "auto"}),
                "resolution": (["720p", "1080p"], {"default": "720p"}),
                "generate_audio": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_video"
    CATEGORY = "FAL/VideoGeneration"

    def generate_video(self, prompt, first_frame, last_frame=None, duration="8s", aspect_ratio="auto", resolution="720p", generate_audio=True):
        try:
            first_frame_url = ImageUtils.upload_image(first_frame)
            if not first_frame_url:
                return ApiHandler.handle_video_generation_error(
                    "veo3.1", "Failed to upload first frame"
                )

            # Conditional routing based on whether last_frame is provided
            if last_frame is None:
                # Use image-to-video endpoint (first frame only)
                endpoint = "fal-ai/veo3.1/image-to-video"
                arguments = {
                    "prompt": prompt,
                    "image_url": first_frame_url,
                    "duration": duration,
                    "aspect_ratio": aspect_ratio,
                    "resolution": resolution,
                    "generate_audio": generate_audio,
                }
            else:
                # Use first-last-frame-to-video endpoint (both frames)
                endpoint = "fal-ai/veo3.1/first-last-frame-to-video"
                last_frame_url = ImageUtils.upload_image(last_frame)
                if not last_frame_url:
                    return ApiHandler.handle_video_generation_error(
                        "veo3.1", "Failed to upload last frame"
                    )
                arguments = {
                    "prompt": prompt,
                    "first_frame_url": first_frame_url,
                    "last_frame_url": last_frame_url,
                    "duration": duration,
                    "aspect_ratio": aspect_ratio,
                    "resolution": resolution,
                    "generate_audio": generate_audio,
                }

            result = ApiHandler.submit_and_get_result(endpoint, arguments)
            video_url = result["video"]["url"]
            return (video_url,)
        except Exception as e:
            return ApiHandler.handle_video_generation_error(
                "veo3.1", str(e)
            )


class FalVeo31FastFirstLastFrameToVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "first_frame": ("IMAGE",),
            },
            "optional": {
                "last_frame": ("IMAGE",),
                "duration": (["4s", "6s", "8s"], {"default": "8s"}),
                "aspect_ratio": (["auto", "9:16", "16:9", "1:1"], {"default": "auto"}),
                "resolution": (["720p", "1080p"], {"default": "720p"}),
                "generate_audio": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_video"
    CATEGORY = "FAL/VideoGeneration"

    def generate_video(self, prompt, first_frame, last_frame=None, duration="8s", aspect_ratio="auto", resolution="720p", generate_audio=True):
        try:
            first_frame_url = ImageUtils.upload_image(first_frame)
            if not first_frame_url:
                return ApiHandler.handle_video_generation_error(
                    "veo3.1/fast", "Failed to upload first frame"
                )

            # Conditional routing based on whether last_frame is provided
            if last_frame is None:
                # Use image-to-video endpoint (first frame only)
                endpoint = "fal-ai/veo3.1/fast/image-to-video"
                arguments = {
                    "prompt": prompt,
                    "image_url": first_frame_url,
                    "duration": duration,
                    "aspect_ratio": aspect_ratio,
                    "resolution": resolution,
                    "generate_audio": generate_audio,
                }
            else:
                # Use first-last-frame-to-video endpoint (both frames)
                endpoint = "fal-ai/veo3.1/fast/first-last-frame-to-video"
                last_frame_url = ImageUtils.upload_image(last_frame)
                if not last_frame_url:
                    return ApiHandler.handle_video_generation_error(
                        "veo3.1/fast", "Failed to upload last frame"
                    )
                arguments = {
                    "prompt": prompt,
                    "first_frame_url": first_frame_url,
                    "last_frame_url": last_frame_url,
                    "duration": duration,
                    "aspect_ratio": aspect_ratio,
                    "resolution": resolution,
                    "generate_audio": generate_audio,
                }

            result = ApiHandler.submit_and_get_result(endpoint, arguments)
            video_url = result["video"]["url"]
            return (video_url,)
        except Exception as e:
            return ApiHandler.handle_video_generation_error(
                "veo3.1/fast", str(e)
            )


# Update Node class mappings
NODE_CLASS_MAPPINGS = {
    "InfinityStarTextToVideo_fal": InfinityStarTextToVideoNode,
    "Kling_fal": KlingNode,
    "KlingPro10_fal": KlingPro10Node,
    "KlingPro16_fal": KlingPro16Node,
    "KlingMaster_fal": KlingMasterNode,
    "KlingOmniImageToVideo_fal": KlingOmniImageToVideoNode,
    "KlingOmniReferenceToVideo_fal": KlingOmniReferenceToVideoNode,
    "KlingOmniVideoToVideoEdit_fal": KlingOmniVideoToVideoEditNode,
    "KlingOmniVideoToVideoReference_fal": KlingOmniVideoToVideoReferenceNode,
    "Krea_Wan14b_VideoToVideo_fal": KreaWan14bVideoToVideoNode,
    "RunwayGen3_fal": RunwayGen3Node,
    "LumaDreamMachine_fal": LumaDreamMachineNode,
    "LoadVideoURL": LoadVideoURL,
    "UploadVideo_fal": UploadVideoNode,
    "UploadFile_fal": UploadFileNode,
    "MiniMax_fal": MiniMaxNode,
    "MiniMaxTextToVideo_fal": MiniMaxTextToVideoNode,
    "MiniMaxSubjectReference_fal": MiniMaxSubjectReferenceNode,
    "PixverseSwapNode_fal": PixverseSwapNode,
    "VideoUpscaler_fal": VideoUpscalerNode,
    "CombinedVideoGeneration_fal": CombinedVideoGenerationNode,
    "Veo2ImageToVideo_fal": Veo2ImageToVideoNode,
    "WanPro_fal": WanProNode,
    "Wan25_preview_fal": Wan25Node,
    "WanVACEVideoEdit_fal": WanVACEVideoEditNode,
    "Wan2214b_animate_replace_character_fal": Wan2214bAnimateReplaceNode,
    "Wan2214b_animate_move_character_fal": Wan2214bAnimateMoveNode,
    "Wan22VACEFun14b_fal": Wan22VACEFun14bNode,
    "DYWanFun22_fal": DYWanFun22Node,
    "DYWanUpscaler_fal": DYWanUpscalerNode,
    "SeedanceImageToVideo_fal": SeedanceImageToVideoNode,
    "SeedanceProImageToVideo_fal": SeedanceProImageToVideoNode,
    "SeedanceTextToVideo_fal": SeedanceTextToVideoNode,
    "Veo3_fal": Veo3Node,
    "Kling21Pro_fal": FalKling21ProImageToVideo,
    "Kling25TurboPro_fal": FalKling25TurboProImageToVideo,
    "Sora2Pro_fal": FalSora2ProImageToVideo,
    "Veo31_fal": FalVeo31FirstLastFrameToVideo,
    "Veo31Fast_fal": FalVeo31FastFirstLastFrameToVideo,
}

# Update Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "InfinityStarTextToVideo_fal": "Infinity Star Text-to-Video (fal)",
    "Kling_fal": "Kling Video Generation (fal)",
    "KlingPro10_fal": "Kling Pro v1.0 Video Generation (fal)",
    "KlingPro16_fal": "Kling Pro v1.6 Video Generation (fal)",
    "KlingMaster_fal": "Kling Master v2.0 Video Generation (fal)",
    "KlingOmniImageToVideo_fal": "Kling Omni Image-to-Video (fal)",
    "KlingOmniReferenceToVideo_fal": "Kling Omni Reference-to-Video (fal)",
    "KlingOmniVideoToVideoEdit_fal": "Kling Omni Video-to-Video Edit (fal)",
    "KlingOmniVideoToVideoReference_fal": "Kling Omni Video-to-Video Reference (fal)",
    "Krea_Wan14b_VideoToVideo_fal": "Krea Wan 14b Video-to-Video (fal)",
    "RunwayGen3_fal": "Runway Gen3 Image-to-Video (fal)",
    "LumaDreamMachine_fal": "Luma Dream Machine (fal)",
    "LoadVideoURL": "Load Video from URL",
    "UploadVideo_fal": "Upload Video (fal)",
    "UploadFile_fal": "Upload File (fal)",
    "MiniMax_fal": "MiniMax Video Generation (fal)",
    "MiniMaxTextToVideo_fal": "MiniMax Text-to-Video (fal)",
    "MiniMaxSubjectReference_fal": "MiniMax Subject Reference (fal)",
    "PixverseSwapNode_fal": "Pixverse Swap (fal)",
    "VideoUpscaler_fal": "Video Upscaler (fal)",
    "CombinedVideoGeneration_fal": "Combined Video Generation (fal)",
    "Veo2ImageToVideo_fal": "Google Veo2 Image-to-Video (fal)",
    "WanPro_fal": "Wan Pro Image-to-Video (fal)",
    "SeedanceImageToVideo_fal": "Seedance Image-to-Video (fal)",
    "SeedanceProImageToVideo_fal": "Seedance Pro Image-to-Video (fal)",
    "SeedanceTextToVideo_fal": "Seedance Text-to-Video (fal)",
    "Veo3_fal": "Veo3 Video Generation (fal)",
    "Wan25_preview_fal": "Wan 2.5 Preview Image-to-Video (fal)",
    "WanVACEVideoEdit_fal": "Wan VACE Video Edit (fal)",
    "Wan2214b_animate_replace_character_fal": "Wan 2.2 14b Animate: Replace Character (fal)",
    "Wan2214b_animate_move_character_fal": "Wan 2.2 14b Animate: Move Character (fal)",
    "Wan22VACEFun14b_fal": "Wan 2.2 VACE Fun 14b Video-to-Video (fal)",
    "DYWanFun22_fal": "DY Wan Fun 22 Video Generation (fal)",
    "DYWanUpscaler_fal": "DY Wan Upscaler (fal)",
    "Kling21Pro_fal": "Kling v2.1 Pro Image-to-Video (fal)",
    "Kling25TurboPro_fal": "Kling v2.5 Turbo Pro Image-to-Video (fal)",
    "Sora2Pro_fal": "Sora 2 Pro Image-to-Video (fal)",
    "Veo31_fal": "Veo 3.1 First-Last-Frame-to-Video (fal)",
    "Veo31Fast_fal": "Veo 3.1 Fast First-Last-Frame-to-Video (fal)",
}

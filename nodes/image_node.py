from .fal_utils import ApiHandler, ImageUtils, ResultProcessor


# Remove all the configuration code since it's now handled by FalConfig
def upload_image(image):
    """Upload image tensor to FAL and return URL."""
    return ImageUtils.upload_image(image)


class Sana:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "image_size": (
                    [
                        "square_hd",
                        "square",
                        "portrait_4_3",
                        "portrait_16_9",
                        "landscape_4_3",
                        "landscape_16_9",
                        "custom",
                    ],
                    {"default": "square_hd"},
                ),
                "width": (
                    "INT",
                    {"default": 3840, "min": 512, "max": 4096, "step": 16},
                ),
                "height": (
                    "INT",
                    {"default": 2160, "min": 512, "max": 4096, "step": 16},
                ),
                "num_inference_steps": ("INT", {"default": 18, "min": 1, "max": 50}),
                "guidance_scale": (
                    "FLOAT",
                    {"default": 5.0, "min": 1.0, "max": 20.0, "step": 0.1},
                ),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "seed": ("INT", {"default": -1}),
                "enable_safety_checker": ("BOOLEAN", {"default": True}),
                "output_format": (["png", "jpeg"], {"default": "png"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "FAL/Image"

    def generate_image(
        self,
        prompt,
        image_size,
        width,
        height,
        num_inference_steps,
        guidance_scale,
        num_images,
        negative_prompt="",
        seed=-1,
        enable_safety_checker=True,
        output_format="png",
    ):
        arguments = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "num_images": num_images,
            "enable_safety_checker": enable_safety_checker,
            "output_format": output_format,
        }

        if image_size == "custom":
            arguments["image_size"] = {"width": width, "height": height}
        else:
            arguments["image_size"] = image_size

        if seed != -1:
            arguments["seed"] = seed

        try:
            result = ApiHandler.submit_and_get_result("fal-ai/sana", arguments)
            return ResultProcessor.process_image_result(result)
        except Exception as e:
            return ApiHandler.handle_image_generation_error("Sana", e)


class Recraft:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "image_size": (
                    [
                        "square_hd",
                        "square",
                        "portrait_4_3",
                        "portrait_16_9",
                        "landscape_4_3",
                        "landscape_16_9",
                        "custom",
                    ],
                    {"default": "square_hd"},
                ),
                "width": ("INT", {"default": 512, "min": 512, "max": 2048, "step": 16}),
                "height": (
                    "INT",
                    {"default": 512, "min": 512, "max": 2048, "step": 16},
                ),
                "style": (
                    [
                        "any",
                        "realistic_image",
                        "digital_illustration",
                        "vector_illustration",
                        "realistic_image/b_and_w",
                        "realistic_image/hard_flash",
                        "realistic_image/hdr",
                        "realistic_image/natural_light",
                        "realistic_image/studio_portrait",
                        "realistic_image/enterprise",
                        "realistic_image/motion_blur",
                        "digital_illustration/pixel_art",
                        "digital_illustration/hand_drawn",
                        "digital_illustration/grain",
                        "digital_illustration/infantile_sketch",
                        "digital_illustration/2d_art_poster",
                        "digital_illustration/handmade_3d",
                        "digital_illustration/hand_drawn_outline",
                        "digital_illustration/engraving_color",
                        "digital_illustration/2d_art_poster_2",
                        "vector_illustration/engraving",
                        "vector_illustration/line_art",
                        "vector_illustration/line_circuit",
                        "vector_illustration/linocut",
                    ],
                    {"default": "realistic_image"},
                ),
            },
            "optional": {
                "style_id": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "FAL/Image"

    def generate_image(self, prompt, image_size, width, height, style, style_id=""):
        arguments = {
            "prompt": prompt,
            "style": style,
        }

        if image_size == "custom":
            arguments["image_size"] = {"width": width, "height": height}
        else:
            arguments["image_size"] = image_size

        if style_id:
            arguments["style_id"] = style_id

        try:
            result = ApiHandler.submit_and_get_result("fal-ai/recraft-v3", arguments)
            return ResultProcessor.process_image_result(result)
        except Exception as e:
            return ApiHandler.handle_image_generation_error("Recraft", e)


class HidreamFull:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "image_size": (
                    [
                        "square_hd",
                        "square",
                        "portrait_4_3",
                        "portrait_16_9",
                        "landscape_4_3",
                        "landscape_16_9",
                        "custom",
                    ],
                    {"default": "landscape_4_3"},
                ),
                "width": (
                    "INT",
                    {"default": 1024, "min": 512, "max": 1440, "step": 32},
                ),
                "height": (
                    "INT",
                    {"default": 768, "min": 512, "max": 1440, "step": 32},
                ),
                "num_inference_steps": ("INT", {"default": 28, "min": 1, "max": 100}),
                "guidance_scale": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 20.0}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 10}),
                "safety_tolerance": (["1", "2", "3", "4", "5", "6"], {"default": "2"}),
            },
            "optional": {
                "seed": ("INT", {"default": -1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "FAL/Image"

    def generate_image(
        self,
        prompt,
        image_size,
        width,
        height,
        num_inference_steps,
        guidance_scale,
        num_images,
        safety_tolerance,
        seed=-1,
    ):
        arguments = {
            "prompt": prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "num_images": num_images,
            "safety_tolerance": safety_tolerance,
        }
        if image_size == "custom":
            arguments["image_size"] = {"width": width, "height": height}
        else:
            arguments["image_size"] = image_size
        if seed != -1:
            arguments["seed"] = seed

        try:
            result = ApiHandler.submit_and_get_result(
                "fal-ai/hidream-i1-full", arguments
            )
            return ResultProcessor.process_image_result(result)
        except Exception as e:
            return ApiHandler.handle_image_generation_error("Hidream Full", e)


class Ideogramv3:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "image_size": (
                    [
                        "square_hd",
                        "square",
                        "portrait_4_3",
                        "portrait_16_9",
                        "landscape_4_3",
                        "landscape_16_9",
                        "custom",
                    ],
                    {"default": "landscape_4_3"},
                ),
                "width": (
                    "INT",
                    {"default": 1024, "min": 512, "max": 1440, "step": 32},
                ),
                "height": (
                    "INT",
                    {"default": 768, "min": 512, "max": 1440, "step": 32},
                ),
                "num_inference_steps": ("INT", {"default": 28, "min": 1, "max": 100}),
                "guidance_scale": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 20.0}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 10}),
                "safety_tolerance": (["1", "2", "3", "4", "5", "6"], {"default": "2"}),
            },
            "optional": {
                "seed": ("INT", {"default": -1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "FAL/Image"

    def generate_image(
        self,
        prompt,
        image_size,
        width,
        height,
        num_inference_steps,
        guidance_scale,
        num_images,
        safety_tolerance,
        seed=-1,
    ):
        arguments = {
            "prompt": prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "num_images": num_images,
            "safety_tolerance": safety_tolerance,
        }
        if image_size == "custom":
            arguments["image_size"] = {"width": width, "height": height}
        else:
            arguments["image_size"] = image_size
        if seed != -1:
            arguments["seed"] = seed

        try:
            result = ApiHandler.submit_and_get_result("fal-ai/ideogram/v3", arguments)
            return ResultProcessor.process_image_result(result)
        except Exception as e:
            return ApiHandler.handle_image_generation_error("Ideogramv3", e)


class FluxPro:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "image_size": (
                    [
                        "square_hd",
                        "square",
                        "portrait_4_3",
                        "portrait_16_9",
                        "landscape_4_3",
                        "landscape_16_9",
                        "custom",
                    ],
                    {"default": "landscape_4_3"},
                ),
                "width": (
                    "INT",
                    {"default": 1024, "min": 512, "max": 1440, "step": 32},
                ),
                "height": (
                    "INT",
                    {"default": 768, "min": 512, "max": 1440, "step": 32},
                ),
                "num_inference_steps": ("INT", {"default": 28, "min": 1, "max": 100}),
                "guidance_scale": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 20.0}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 10}),
                "safety_tolerance": (["1", "2", "3", "4", "5", "6"], {"default": "2"}),
            },
            "optional": {
                "seed": ("INT", {"default": -1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "FAL/Image"

    def generate_image(
        self,
        prompt,
        image_size,
        width,
        height,
        num_inference_steps,
        guidance_scale,
        num_images,
        safety_tolerance,
        seed=-1,
    ):
        arguments = {
            "prompt": prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "num_images": num_images,
            "safety_tolerance": safety_tolerance,
        }
        if image_size == "custom":
            arguments["image_size"] = {"width": width, "height": height}
        else:
            arguments["image_size"] = image_size
        if seed != -1:
            arguments["seed"] = seed

        try:
            result = ApiHandler.submit_and_get_result("fal-ai/flux-pro", arguments)
            return ResultProcessor.process_image_result(result)
        except Exception as e:
            return ApiHandler.handle_image_generation_error("FluxPro", e)


class FluxDev:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "image_size": (
                    [
                        "square_hd",
                        "square",
                        "portrait_4_3",
                        "portrait_16_9",
                        "landscape_4_3",
                        "landscape_16_9",
                        "custom",
                    ],
                    {"default": "landscape_4_3"},
                ),
                "width": (
                    "INT",
                    {"default": 1024, "min": 512, "max": 1536, "step": 16},
                ),
                "height": (
                    "INT",
                    {"default": 768, "min": 512, "max": 1536, "step": 16},
                ),
                "num_inference_steps": ("INT", {"default": 28, "min": 1, "max": 100}),
                "guidance_scale": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 20.0}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 10}),
                "enable_safety_checker": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "seed": ("INT", {"default": -1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "FAL/Image"

    def generate_image(
        self,
        prompt,
        image_size,
        width,
        height,
        num_inference_steps,
        guidance_scale,
        num_images,
        enable_safety_checker,
        seed=-1,
    ):
        arguments = {
            "prompt": prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "num_images": num_images,
            "enable_safety_checker": enable_safety_checker,
        }
        if image_size == "custom":
            arguments["image_size"] = {"width": width, "height": height}
        else:
            arguments["image_size"] = image_size
        if seed != -1:
            arguments["seed"] = seed

        try:
            result = ApiHandler.submit_and_get_result("fal-ai/flux/dev", arguments)
            return ResultProcessor.process_image_result(result)
        except Exception as e:
            return ApiHandler.handle_image_generation_error("FluxDev", e)


class FluxSchnell:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "image_size": (
                    [
                        "square_hd",
                        "square",
                        "portrait_4_3",
                        "portrait_16_9",
                        "landscape_4_3",
                        "landscape_16_9",
                        "custom",
                    ],
                    {"default": "landscape_4_3"},
                ),
                "width": (
                    "INT",
                    {"default": 1024, "min": 512, "max": 1536, "step": 32},
                ),
                "height": (
                    "INT",
                    {"default": 768, "min": 512, "max": 1536, "step": 32},
                ),
                "num_inference_steps": ("INT", {"default": 4, "min": 1, "max": 100}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 10}),
                "enable_safety_checker": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "seed": ("INT", {"default": -1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "FAL/Image"

    def generate_image(
        self,
        prompt,
        image_size,
        width,
        height,
        num_inference_steps,
        num_images,
        enable_safety_checker,
        seed=-1,
    ):
        arguments = {
            "prompt": prompt,
            "num_inference_steps": num_inference_steps,
            "num_images": num_images,
            "enable_safety_checker": enable_safety_checker,
        }
        if image_size == "custom":
            arguments["image_size"] = {"width": width, "height": height}
        else:
            arguments["image_size"] = image_size
        if seed != -1:
            arguments["seed"] = seed

        try:
            result = ApiHandler.submit_and_get_result("fal-ai/flux/schnell", arguments)
            return ResultProcessor.process_image_result(result)
        except Exception as e:
            return ApiHandler.handle_image_generation_error("FluxSchnell", e)


class FluxPro11:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "image_size": (
                    [
                        "square_hd",
                        "square",
                        "portrait_4_3",
                        "portrait_16_9",
                        "landscape_4_3",
                        "landscape_16_9",
                        "custom",
                    ],
                    {"default": "landscape_4_3"},
                ),
                "width": (
                    "INT",
                    {"default": 1024, "min": 512, "max": 1440, "step": 32},
                ),
                "height": (
                    "INT",
                    {"default": 768, "min": 512, "max": 1440, "step": 32},
                ),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 10}),
                "safety_tolerance": (["1", "2", "3", "4", "5", "6"], {"default": "2"}),
            },
            "optional": {
                "seed": ("INT", {"default": -1}),
                "sync_mode": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "FAL/Image"

    def generate_image(
        self,
        prompt,
        image_size,
        width,
        height,
        num_images,
        safety_tolerance,
        seed=-1,
        sync_mode=False,
    ):
        arguments = {
            "prompt": prompt,
            "num_images": num_images,
            "safety_tolerance": safety_tolerance,
            "sync_mode": sync_mode,
        }
        if image_size == "custom":
            arguments["image_size"] = {"width": width, "height": height}
        else:
            arguments["image_size"] = image_size
        if seed != -1:
            arguments["seed"] = seed

        try:
            result = ApiHandler.submit_and_get_result("fal-ai/flux-pro/v1.1", arguments)
            return ResultProcessor.process_image_result(result)
        except Exception as e:
            return ApiHandler.handle_image_generation_error("FluxPro 1.1", e)


class FluxPro1Fill:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 10}),
                "safety_tolerance": (["1", "2", "3", "4", "5", "6"], {"default": "2"}),
                "output_format": (["png", "jpeg"], {"default": "png"}),
                
            },
            "optional": {
                "image": ("IMAGE", {"default": None}),
                "mask_image": ("IMAGE", {"default": None}),
                "seed": ("INT", {"default": -1}),
                "sync_mode": ("BOOLEAN", {"default": False}),
                "enhance_prompt": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "FAL/Image"

    def generate_image(
        self,
        prompt,
        image = None,
        mask_image = None,
        output_format = "png",
        num_images = 1,
        safety_tolerance = "2",
        seed=0,
        sync_mode=False,
        enhance_prompt=True,
    ):
        if image is None or mask_image is None:
            return ApiHandler.handle_image_generation_error(
                "FluxPro 1/FILL", "Both image and mask_image inputs are required."
            )
        image_url = ImageUtils.upload_image(image)
        mask_url = ImageUtils.upload_image(mask_image)
        arguments = {
            "prompt": prompt,
            "num_images": num_images,
            "safety_tolerance": safety_tolerance,
            "sync_mode": sync_mode,
        }
        arguments={
            "prompt": prompt,
            "num_images": num_images,
            "sync_mode": sync_mode,
            "output_format": output_format,
            "safety_tolerance": safety_tolerance,
            "image_url": image_url,
            "mask_url": mask_url,
            "enhance_prompt": enhance_prompt
        }
        if seed != 0:
            arguments["seed"] = seed

        try:
            result = ApiHandler.submit_and_get_result("fal-ai/flux-pro/v1/fill", arguments)
            return ResultProcessor.process_image_result(result)
        except Exception as e:
            return ApiHandler.handle_image_generation_error("FluxPro 1/FILL", e)


class FluxUltra:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "aspect_ratio": (
                    ["21:9", "16:9", "4:3", "1:1", "3:4", "9:16", "9:21"],
                    {"default": "16:9"},
                ),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 1}),
                "safety_tolerance": (["1", "2", "3", "4", "5", "6"], {"default": "2"}),
                "enable_safety_checker": ("BOOLEAN", {"default": True}),
                "raw": ("BOOLEAN", {"default": False}),
                "sync_mode": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "seed": ("INT", {"default": -1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "FAL/Image"

    def generate_image(
        self,
        prompt,
        aspect_ratio,
        num_images,
        safety_tolerance,
        enable_safety_checker,
        raw,
        sync_mode,
        seed=-1,
    ):
        arguments = {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "num_images": num_images,
            "safety_tolerance": safety_tolerance,
            "enable_safety_checker": enable_safety_checker,
            "raw": raw,
            "sync_mode": sync_mode,
        }
        if seed != -1:
            arguments["seed"] = seed

        try:
            result = ApiHandler.submit_and_get_result(
                "fal-ai/flux-pro/v1.1-ultra", arguments
            )
            return ResultProcessor.process_image_result(result)
        except Exception as e:
            return ApiHandler.handle_image_generation_error("FluxUltra", e)


class FluxLora:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "image_size": (
                    [
                        "square_hd",
                        "square",
                        "portrait_4_3",
                        "portrait_16_9",
                        "landscape_4_3",
                        "landscape_16_9",
                        "custom",
                    ],
                    {"default": "landscape_4_3"},
                ),
                "width": (
                    "INT",
                    {"default": 1024, "min": 512, "max": 1536, "step": 16},
                ),
                "height": (
                    "INT",
                    {"default": 768, "min": 512, "max": 1536, "step": 16},
                ),
                "num_inference_steps": ("INT", {"default": 28, "min": 1, "max": 50}),
                "guidance_scale": (
                    "FLOAT",
                    {"default": 3.0, "min": 0.0, "max": 20.0, "step": 0.1},
                ),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
                "enable_safety_checker": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "seed": ("INT", {"default": -1}),
                "lora_path_1": ("STRING", {"default": ""}),
                "lora_scale_1": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05},
                ),
                "lora_path_2": ("STRING", {"default": ""}),
                "lora_scale_2": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "FAL/Image"

    def generate_image(
        self,
        prompt,
        image_size,
        width,
        height,
        num_inference_steps,
        guidance_scale,
        num_images,
        enable_safety_checker,
        seed=-1,
        lora_path_1="",
        lora_scale_1=1.0,
        lora_path_2="",
        lora_scale_2=1.0,
    ):
        arguments = {
            "prompt": prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "num_images": num_images,
            "enable_safety_checker": enable_safety_checker,
        }
        if image_size == "custom":
            arguments["image_size"] = {"width": width, "height": height}
        else:
            arguments["image_size"] = image_size
        if seed != -1:
            arguments["seed"] = seed

        # Add LoRAs
        loras = []
        if lora_path_1:
            loras.append({"path": lora_path_1, "scale": lora_scale_1})
        if lora_path_2:
            loras.append({"path": lora_path_2, "scale": lora_scale_2})
        if loras:
            arguments["loras"] = loras

        try:
            result = ApiHandler.submit_and_get_result("fal-ai/flux-lora", arguments)
            return ResultProcessor.process_image_result(result)
        except Exception as e:
            return ApiHandler.handle_image_generation_error("FluxLora", e)


class FluxGeneral:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "image_size": (
                    [
                        "square_hd",
                        "square",
                        "portrait_4_3",
                        "portrait_16_9",
                        "landscape_4_3",
                        "landscape_16_9",
                        "custom",
                    ],
                    {"default": "landscape_4_3"},
                ),
                "width": (
                    "INT",
                    {"default": 1024, "min": 512, "max": 1536, "step": 16},
                ),
                "height": (
                    "INT",
                    {"default": 768, "min": 512, "max": 1536, "step": 16},
                ),
                "num_inference_steps": ("INT", {"default": 28, "min": 1, "max": 50}),
                "guidance_scale": (
                    "FLOAT",
                    {"default": 3.0, "min": 0.0, "max": 20.0, "step": 0.1},
                ),
                "real_cfg_scale": (
                    "FLOAT",
                    {"default": 3.3, "min": 0.0, "max": 5.0, "step": 0.1},
                ),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
                "enable_safety_checker": ("BOOLEAN", {"default": False}),
                "use_real_cfg": ("BOOLEAN", {"default": False}),
                "sync_mode": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "seed": ("INT", {"default": -1}),
                "ip_adapter_scale": (
                    "FLOAT",
                    {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.1},
                ),
                "controlnet_conditioning_scale": (
                    "FLOAT",
                    {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.1},
                ),
                "ip_adapters": (
                    ["None", "XLabs-AI/flux-ip-adapter"],
                    {"default": "None"},
                ),
                "controlnets": (
                    [
                        "None",
                        "XLabs-AI/flux-controlnet-depth-v3",
                        "Shakker-Labs/FLUX.1-dev-ControlNet-Depth",
                        "jasperai/Flux.1-dev-Controlnet-Depth",
                        "jasperai/Flux.1-dev-Controlnet-Surface-Normals",
                        "XLabs-AI/flux-controlnet-canny-v3",
                        "InstantX/FLUX.1-dev-Controlnet-Canny",
                        "jasperai/Flux.1-dev-Controlnet-Upscaler",
                        "promeai/FLUX.1-controlnet-lineart-promeai",
                    ],
                    {"default": "None"},
                ),
                "controlnet_unions": (
                    [
                        "None",
                        "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro",
                        "InstantX/FLUX.1-dev-Controlnet-Union",
                    ],
                    {"default": "None"},
                ),
                "controlnet_union_control_mode": (
                    ["canny", "tile", "depth", "blur", "pose", "gray", "low_quality"],
                    {"default": "canny"},
                ),
                "control_image": ("IMAGE",),
                "control_mask": ("MASK",),
                "ip_adapter_image": ("IMAGE",),
                "ip_adapter_mask": ("MASK",),
                "lora_path_1": ("STRING", {"default": ""}),
                "lora_scale_1": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05},
                ),
                "lora_path_2": ("STRING", {"default": ""}),
                "lora_scale_2": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "FAL/Image"

    def generate_image(
        self,
        prompt,
        image_size,
        width,
        height,
        num_inference_steps,
        guidance_scale,
        real_cfg_scale,
        num_images,
        enable_safety_checker,
        use_real_cfg,
        sync_mode,
        seed=-1,
        lora_path_1="",
        lora_scale_1=1.0,
        lora_path_2="",
        lora_scale_2=1.0,
        ip_adapter_scale=0.6,
        controlnet_conditioning_scale=0.6,
        controlnet_union_control_mode="canny",
        ip_adapters="None",
        controlnets="None",
        controlnet_unions="None",
        control_image=None,
        control_mask=None,
        ip_adapter_image=None,
        ip_adapter_mask=None,
    ):
        arguments = {
            "prompt": prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "real_cfg_scale": real_cfg_scale,
            "num_images": num_images,
            "enable_safety_checker": enable_safety_checker,
            "use_real_cfg": use_real_cfg,
            "sync_mode": sync_mode,
        }
        if image_size == "custom":
            arguments["image_size"] = {"width": width, "height": height}
        else:
            arguments["image_size"] = image_size
        if seed != -1:
            arguments["seed"] = seed

        # Add ip_adapters if selected
        if ip_adapters != "None":
            arguments["ip_adapters"] = [
                {
                    "path": ip_adapters,
                    "image_encoder_path": "openai/clip-vit-large-patch14",
                    "scale": ip_adapter_scale,
                }
            ]

        # Controlnet mapping
        controlnet_mapping = {
            "XLabs-AI/flux-controlnet-depth-v3": "https://huggingface.co/XLabs-AI/flux-controlnet-depth-v3/resolve/main/flux-depth-controlnet-v3.safetensors",
            "Shakker-Labs/FLUX.1-dev-ControlNet-Depth": "https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Depth/resolve/main/diffusion_pytorch_model.safetensors",
            "jasperai/Flux.1-dev-Controlnet-Depth": "https://huggingface.co/jasperai/Flux.1-dev-Controlnet-Depth/resolve/main/diffusion_pytorch_model.safetensors",
            "jasperai/Flux.1-dev-Controlnet-Surface-Normals": "https://huggingface.co/jasperai/Flux.1-dev-Controlnet-Surface-Normals/resolve/main/diffusion_pytorch_model.safetensors",
            "XLabs-AI/flux-controlnet-canny-v3": "https://huggingface.co/XLabs-AI/flux-controlnet-canny-v3/resolve/main/flux-canny-controlnet-v3.safetensors",
            "InstantX/FLUX.1-dev-Controlnet-Canny": "https://huggingface.co/InstantX/FLUX.1-dev-Controlnet-Canny/resolve/main/diffusion_pytorch_model.safetensors",
            "jasperai/Flux.1-dev-Controlnet-Upscaler": "https://huggingface.co/jasperai/Flux.1-dev-Controlnet-Upscaler/resolve/main/diffusion_pytorch_model.safetensors",
            "promeai/FLUX.1-controlnet-lineart-promeai": "https://huggingface.co/promeai/FLUX.1-controlnet-lineart-promeai/resolve/main/diffusion_pytorch_model.safetensors",
        }

        # Add controlnets if selected
        if controlnets != "None":
            controlnet_path = controlnet_mapping.get(controlnets, controlnets)
            arguments["controlnets"] = [
                {
                    "path": controlnet_path,
                    "conditioning_scale": controlnet_conditioning_scale,
                }
            ]

        # Add controlnet_unions if selected
        if controlnet_unions != "None":
            arguments["controlnet_unions"] = [
                {
                    "path": controlnet_unions,
                    "controls": [
                        {
                            "control_mode": controlnet_union_control_mode,
                        }
                    ],
                }
            ]

        # Handle controlnets
        if controlnets != "None" and control_image is not None:
            control_image_url = ImageUtils.upload_image(control_image)
            if control_image_url:
                controlnet_path = controlnet_mapping.get(controlnets, controlnets)
                arguments["controlnets"] = [
                    {
                        "path": controlnet_path,
                        "conditioning_scale": controlnet_conditioning_scale,
                        "control_image_url": control_image_url,
                    }
                ]
                if control_mask is not None:
                    mask_image = ImageUtils.mask_to_image(control_mask)
                    mask_image_url = ImageUtils.upload_image(mask_image)
                    if mask_image_url:
                        arguments["controlnets"][0]["mask_image_url"] = mask_image_url

        # Handle controlnet_unions
        if controlnet_unions != "None" and control_image is not None:
            control_image_url = ImageUtils.upload_image(control_image)
            if control_image_url:
                arguments["controlnet_unions"] = [
                    {
                        "path": controlnet_unions,
                        "controls": [
                            {
                                "control_mode": controlnet_union_control_mode,
                                "control_image_url": control_image_url,
                            }
                        ],
                    }
                ]
                if control_mask is not None:
                    mask_image = ImageUtils.mask_to_image(control_mask)
                    mask_image_url = ImageUtils.upload_image(mask_image)
                    if mask_image_url:
                        arguments["controlnet_unions"][0]["controls"][0][
                            "mask_image_url"
                        ] = mask_image_url

        # Handle ip_adapters
        if ip_adapters != "None" and ip_adapter_image is not None:
            ip_adapter_image_url = ImageUtils.upload_image(ip_adapter_image)
            if ip_adapter_image_url:
                ip_adapter_path = (
                    "https://huggingface.co/XLabs-AI/flux-ip-adapter/resolve/main/flux-ip-adapter.safetensors?download=true"
                    if ip_adapters == "XLabs-AI/flux-ip-adapter"
                    else ip_adapters
                )
                arguments["ip_adapters"] = [
                    {
                        "path": ip_adapter_path,
                        "image_encoder_path": "openai/clip-vit-large-patch14",
                        "image_url": ip_adapter_image_url,
                        "scale": ip_adapter_scale,
                    }
                ]
                if ip_adapter_mask is not None:
                    mask_image = ImageUtils.mask_to_image(ip_adapter_mask)
                    mask_image_url = ImageUtils.upload_image(mask_image)
                    if mask_image_url:
                        arguments["ip_adapters"][0]["mask_image_url"] = mask_image_url

        # Add LoRAs if provided
        loras = []
        if lora_path_1:
            loras.append({"path": lora_path_1, "scale": lora_scale_1})
        if lora_path_2:
            loras.append({"path": lora_path_2, "scale": lora_scale_2})
        if loras:
            arguments["loras"] = loras

        try:
            result = ApiHandler.submit_and_get_result("fal-ai/flux-general", arguments)
            return ResultProcessor.process_image_result(result)
        except Exception as e:
            return ApiHandler.handle_image_generation_error("FluxGeneral", e)


class FluxProKontext:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "image": ("IMAGE",),
            },
            "optional": {
                "aspect_ratio": (
                    [
                        None,
                        "21:9",
                        "16:9",
                        "4:3",
                        "3:2",
                        "1:1",
                        "2:3",
                        "3:4",
                        "9:16",
                        "9:21",
                    ],
                    {"default": None},
                ),
                "max_quality": ("BOOLEAN", {"default": False}),
                "guidance_scale": (
                    "FLOAT",
                    {"default": 3.5, "min": 1.0, "max": 20.0, "step": 0.1},
                ),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
                "safety_tolerance": (["1", "2", "3", "4", "5", "6"], {"default": "2"}),
                "output_format": (["jpeg", "png"], {"default": "jpeg"}),
                "sync_mode": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "FAL/Image"

    def generate_image(
        self,
        prompt,
        image,
        aspect_ratio="1:1",
        max_quality=False,
        guidance_scale=3.5,
        num_images=1,
        safety_tolerance="2",
        output_format="jpeg",
        sync_mode=False,
        seed=0,
    ):
        # Upload the input image to get URL
        image_url = ImageUtils.upload_image(image)
        if not image_url:
            model_name = "Flux Pro Kontext Max" if max_quality else "Flux Pro Kontext"
            print(f"Error: Failed to upload image for {model_name}")
            return ResultProcessor.create_blank_image()

        # Dynamic endpoint selection based on max_quality toggle
        endpoint = (
            "fal-ai/flux-pro/kontext/max" if max_quality else "fal-ai/flux-pro/kontext"
        )

        arguments = {
            "prompt": prompt,
            "image_url": image_url,
            "aspect_ratio": aspect_ratio,
            "guidance_scale": guidance_scale,
            "num_images": num_images,
            "safety_tolerance": safety_tolerance,
            "output_format": output_format,
            "sync_mode": sync_mode,
        }

        if seed > 0:
            arguments["seed"] = seed

        try:
            result = ApiHandler.submit_and_get_result(endpoint, arguments)
            return ResultProcessor.process_image_result(result)
        except Exception as e:
            model_name = "Flux Pro Kontext Max" if max_quality else "Flux Pro Kontext"
            return ApiHandler.handle_image_generation_error(model_name, e)


class FluxProKontextMulti:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
            },
            "optional": {
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "aspect_ratio": (
                    [
                        None,
                        "21:9",
                        "16:9",
                        "4:3",
                        "3:2",
                        "1:1",
                        "2:3",
                        "3:4",
                        "9:16",
                        "9:21",
                    ],
                    {"default": None},
                ),
                "max_quality": ("BOOLEAN", {"default": False}),
                "guidance_scale": (
                    "FLOAT",
                    {"default": 3.5, "min": 1.0, "max": 20.0, "step": 0.1},
                ),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
                "safety_tolerance": (["1", "2", "3", "4", "5", "6"], {"default": "2"}),
                "output_format": (["jpeg", "png"], {"default": "jpeg"}),
                "sync_mode": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "FAL/Image"

    def generate_image(
        self,
        prompt,
        image_1,
        image_2,
        image_3=None,
        image_4=None,
        aspect_ratio="1:1",
        max_quality=False,
        guidance_scale=3.5,
        num_images=1,
        safety_tolerance="2",
        output_format="jpeg",
        sync_mode=False,
        seed=0,
    ):
        # Upload all provided images
        image_urls = []

        for i, img in enumerate([image_1, image_2, image_3, image_4], 1):
            if img is not None:
                url = ImageUtils.upload_image(img)
                if url:
                    image_urls.append(url)
                else:
                    model_name = (
                        "Flux Pro Kontext Max Multi"
                        if max_quality
                        else "Flux Pro Kontext Multi"
                    )
                    print(f"Error: Failed to upload image {i} for {model_name}")
                    return ResultProcessor.create_blank_image()

        if len(image_urls) < 2:
            model_name = (
                "Flux Pro Kontext Max Multi"
                if max_quality
                else "Flux Pro Kontext Multi"
            )
            print(f"Error: At least 2 images required for {model_name}")
            return ResultProcessor.create_blank_image()

        # Dynamic endpoint selection based on max_quality toggle
        endpoint = (
            "fal-ai/flux-pro/kontext/max/multi"
            if max_quality
            else "fal-ai/flux-pro/kontext/multi"
        )

        arguments = {
            "prompt": prompt,
            "image_urls": image_urls,
            "aspect_ratio": aspect_ratio,
            "guidance_scale": guidance_scale,
            "num_images": num_images,
            "safety_tolerance": safety_tolerance,
            "output_format": output_format,
            "sync_mode": sync_mode,
        }

        if seed > 0:
            arguments["seed"] = seed

        try:
            result = ApiHandler.submit_and_get_result(endpoint, arguments)
            return ResultProcessor.process_image_result(result)
        except Exception as e:
            model_name = (
                "Flux Pro Kontext Max Multi"
                if max_quality
                else "Flux Pro Kontext Multi"
            )
            return ApiHandler.handle_image_generation_error(model_name, e)


class FluxProKontextTextToImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "aspect_ratio": (
                    ["21:9", "16:9", "4:3", "3:2", "1:1", "2:3", "3:4", "9:16", "9:21"],
                    {"default": "1:1"},
                ),
                "max_quality": ("BOOLEAN", {"default": False}),
                "guidance_scale": (
                    "FLOAT",
                    {"default": 3.5, "min": 1.0, "max": 20.0, "step": 0.1},
                ),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
                "safety_tolerance": (["1", "2", "3", "4", "5", "6"], {"default": "2"}),
                "output_format": (["jpeg", "png"], {"default": "jpeg"}),
                "sync_mode": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "FAL/Image"

    def generate_image(
        self,
        prompt,
        aspect_ratio="1:1",
        max_quality=False,
        guidance_scale=3.5,
        num_images=1,
        safety_tolerance="2",
        output_format="jpeg",
        sync_mode=False,
        seed=0,
    ):
        # Dynamic endpoint selection based on max_quality toggle
        endpoint = (
            "fal-ai/flux-pro/kontext/max/text-to-image"
            if max_quality
            else "fal-ai/flux-pro/kontext/text-to-image"
        )

        arguments = {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "guidance_scale": guidance_scale,
            "num_images": num_images,
            "safety_tolerance": safety_tolerance,
            "output_format": output_format,
            "sync_mode": sync_mode,
        }

        if seed > 0:
            arguments["seed"] = seed

        try:
            result = ApiHandler.submit_and_get_result(endpoint, arguments)
            return ResultProcessor.process_image_result(result)
        except Exception as e:
            model_name = (
                "Flux Pro Kontext Max Text-to-Image"
                if max_quality
                else "Flux Pro Kontext Text-to-Image"
            )
            return ApiHandler.handle_image_generation_error(model_name, e)


class Imagen4PreviewNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "FAL/Image"

    def generate_image(
        self,
        prompt,
    ):
        arguments = {
            "prompt": prompt,
        }

        try:
            result = ApiHandler.submit_and_get_result(
                "fal-ai/imagen4/preview", arguments
            )
            return ResultProcessor.process_image_result(result)
        except Exception as e:
            return ApiHandler.handle_image_generation_error("Imagen4 Preview", e)


class QwenImageEdit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "image": ("IMAGE",),
                "image_size": (
                    [
                        "square_hd",
                        "square",
                        "portrait_4_3",
                        "portrait_16_9",
                        "landscape_4_3",
                        "landscape_16_9",
                        "custom",
                    ],
                    {"default": "square_hd"},
                ),
                "width": (
                    "INT",
                    {"default": 512, "min": 128, "max": 2048, "step": 8},
                ),
                "height": (
                    "INT",
                    {"default": 512, "min": 128, "max": 2048, "step": 8},
                ),
                "num_inference_steps": ("INT", {"default": 30, "min": 1, "max": 50}),
                "guidance_scale": (
                    "FLOAT",
                    {"default": 4.0, "min": 1.0, "max": 20.0, "step": 0.1},
                ),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
                "enable_safety_checker": ("BOOLEAN", {"default": True}),
                "output_format": (["png", "jpeg"], {"default": "png"}),
                "acceleration": (["none", "regular", "high"], {"default": "none"}),
                "sync_mode": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "seed": ("INT", {"default": -1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "edit_image"
    CATEGORY = "FAL/Image"

    def edit_image(
        self,
        prompt,
        image,
        image_size,
        width,
        height,
        num_inference_steps,
        guidance_scale,
        num_images,
        enable_safety_checker,
        output_format,
        acceleration,
        sync_mode,
        negative_prompt="",
        seed=-1,
    ):
        model_name = "Qwen Image Edit"
        image_url = ImageUtils.upload_image(image)
        if not image_url:
            print(f"Error: Failed to upload image for {model_name}")
            return ResultProcessor.create_blank_image()

        arguments = {
            "prompt": prompt,
            "image_url": image_url,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "num_images": num_images,
            "enable_safety_checker": enable_safety_checker,
            "output_format": output_format,
            "acceleration": acceleration,
            "sync_mode": sync_mode,
        }

        if image_size == "custom":
            arguments["image_size"] = {"width": width, "height": height}
        else:
            arguments["image_size"] = image_size

        if negative_prompt:
            arguments["negative_prompt"] = negative_prompt

        if seed != -1:
            arguments["seed"] = seed

        try:
            result = ApiHandler.submit_and_get_result("fal-ai/qwen-image-edit", arguments)
            return ResultProcessor.process_image_result(result)
        except Exception as e:
            return ApiHandler.handle_image_generation_error(model_name, e)


class QwenImageEditPlusLoRA:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "images": ("IMAGE", {"default": None, "multiple": True})
            },
            "optional": {
                "image_size": (
                    [
                        "custom",
                        "square_hd",
                        "square",
                        "portrait_4_3",
                        "portrait_16_9",
                        "landscape_4_3",
                        "landscape_16_9",
                    ],
                    {"default": "landscape_16_9"},
                ),
                "custom_width": ("INT", {"default": 1920, "min": 128, "max": 2048, "step": 8}),
                "custom_height": ("INT", {"default": 1080, "min": 128, "max": 2048, "step": 8}),
                "num_inference_steps": ("INT", {"default": 28, "min": 2, "max": 50}),
                "guidance_scale": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "seed": ("INT", {"default": -1}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
                "enable_safety_checker": ("BOOLEAN", {"default": True}),
                "output_format": (["png", "jpeg"], {"default": "png"}),
                "lora_path_1": ("STRING", {"default": ""}),
                "lora_scale_1": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05},
                ),
                "lora_path_2": ("STRING", {"default": ""}),
                "lora_scale_2": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05},
                ),
                "lora_path_3": ("STRING", {"default": ""}),
                "lora_scale_3": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05},
                ),
                "lora_path_4": ("STRING", {"default": ""}),
                "lora_scale_4": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "edit_image"
    CATEGORY = "FAL/Image"

    def edit_image(
        self,
        prompt,
        images=None,
        image_size="square_hd",
        custom_width=1024,
        custom_height=1024,
        num_inference_steps=28,
        guidance_scale=4.0,
        negative_prompt="",
        seed=-1,
        num_images=1,
        enable_safety_checker=True,
        output_format="png",
        lora_path_1="",
        lora_scale_1=1.0,
        lora_path_2="",
        lora_scale_2=1.0,
        lora_path_3="",
        lora_scale_3=1.0,
        lora_path_4="",
        lora_scale_4=1.0,
    ):
        model_name = "Qwen Image Edit Plus LoRA"

        # Handle multiple images
        image_urls = ImageUtils.prepare_images(images)
        if not image_urls:
            print(f"Error: No images provided for {model_name}")
            return ResultProcessor.create_blank_image()

        arguments = {
            "prompt": prompt,
            "image_urls": image_urls,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "num_images": num_images,
            "enable_safety_checker": enable_safety_checker,
            "output_format": output_format,
        }

        if image_size == "custom":
            arguments["image_size"] = {"width": custom_width, "height": custom_height}
        else:
            arguments["image_size"] = image_size

        if negative_prompt:
            arguments["negative_prompt"] = negative_prompt

        if seed != -1:
            arguments["seed"] = seed

        # Add LoRAs if provided (maximum 4)
        loras = []
        if lora_path_1:
            loras.append({"path": lora_path_1, "scale": lora_scale_1})
        if lora_path_2:
            loras.append({"path": lora_path_2, "scale": lora_scale_2})
        if lora_path_3:
            loras.append({"path": lora_path_3, "scale": lora_scale_3})
        if lora_path_4:
            loras.append({"path": lora_path_4, "scale": lora_scale_4})
        if loras:
            arguments["loras"] = loras

        try:
            result = ApiHandler.submit_and_get_result(
                "fal-ai/qwen-image-edit-plus-lora", arguments
            )
            return ResultProcessor.process_image_result(result)
        except Exception as e:
            return ApiHandler.handle_image_generation_error(model_name, e)


# https://fal.ai/api/openapi/queue/openapi.json?endpoint_id=fal-ai/bytedance/seededit/v3/edit-image
class SeedEditV3:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "image": ("IMAGE",),
            },
            "optional": {
                "guidance_scale": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1},
                ),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2**32 - 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "FAL/Image"

    def generate_image(
        self,
        prompt,
        image,
        guidance_scale=0.5,
        seed=-1,
    ):
        model_name = "SeedEdit 3.0"
        image_url = ImageUtils.upload_image(image)
        if not image_url:
            print(f"Error: Failed to upload image for {model_name}")
            return ResultProcessor.create_blank_image()

        endpoint = "fal-ai/bytedance/seededit/v3/edit-image"
        arguments = {
            "prompt": prompt,
            "image_url": image_url,
            "guidance_scale": guidance_scale,
        }
        if seed != -1:
            arguments["seed"] = seed

        try:
            result = ApiHandler.submit_and_get_result(endpoint, arguments)
            return ResultProcessor.process_single_image_result(result)
        except Exception as e:
            return ApiHandler.handle_image_generation_error(model_name, e)


# https://fal.ai/api/openapi/queue/openapi.json?endpoint_id=fal-ai/bytedance/seedream/v4/edit
class SeedreamV4Edit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "image_1": ("IMAGE",),
                "image_size": (
                    [
                        "square_hd",
                        "square",
                        "portrait_4_3",
                        "portrait_16_9",
                        "landscape_4_3",
                        "landscape_16_9",
                        "custom",
                    ],
                    {"default": "square_hd"},
                ),
                "width": ("INT", {"default": 3840, "min": 1024, "max": 4096}),
                "height": ("INT", {"default": 2160, "min": 1024, "max": 4096}),
            },
            "optional": {
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
                "image_6": ("IMAGE",),
                "image_7": ("IMAGE",),
                "image_8": ("IMAGE",),
                "image_9": ("IMAGE",),
                "image_10": ("IMAGE",),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 6}),
                "max_images": ("INT", {"default": 1, "min": 1, "max": 6}),
                "enable_safety_checker": ("BOOLEAN", {"default": True}),
                "sync_mode": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2**32 - 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "FAL/Image"

    def generate_image(
        self,
        prompt,
        image_1,
        image_size,
        width,
        height,
        num_images,
        max_images,
        image_2=None,
        image_3=None,
        image_4=None,
        image_5=None,
        image_6=None,
        image_7=None,
        image_8=None,
        image_9=None,
        image_10=None,
        enable_safety_checker=True,
        seed=-1,
        sync_mode=False,
    ):
        model_name = "Seedream 4.0 Edit"

        image_urls = []
        for i, img in enumerate(
            [
                image_1,
                image_2,
                image_3,
                image_4,
                image_5,
                image_6,
                image_7,
                image_8,
                image_9,
                image_10,
            ],
            1,
        ):
            if img is not None:
                url = ImageUtils.upload_image(img)
                if url:
                    image_urls.append(url)
                else:
                    print(f"Error: Failed to upload image {i} for {model_name}")
                    return ResultProcessor.create_blank_image()

        # the total number of images (image inputs + image outputs) must not exceed 15
        max_total_images = 15
        potential_total = len(image_urls) + (num_images * max_images)
        if potential_total > max_total_images:
            print(
                f"Error: Total images (inputs + outputs) must be <= {max_total_images}. "
                f"inputs={len(image_urls)}, num_images={num_images}, max_images={max_images}"
            )
            return ResultProcessor.create_blank_image()

        endpoint = "fal-ai/bytedance/seedream/v4/edit"
        arguments = {
            "prompt": prompt,
            "image_urls": image_urls,
            "num_images": num_images,
            "max_images": max_images,
            "enable_safety_checker": enable_safety_checker,
            "sync_mode": sync_mode,
        }
        if image_size == "custom":
            arguments["image_size"] = {"width": width, "height": height}
        else:
            arguments["image_size"] = image_size
        if seed != -1:
            arguments["seed"] = seed

        try:
            result = ApiHandler.submit_and_get_result(endpoint, arguments)
            return ResultProcessor.process_image_result(result)
        except Exception as e:
            return ApiHandler.handle_image_generation_error(model_name, e)



class NanoBananaTextToImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "aspect_ratio": (
                    ["21:9", "1:1", "4:3", "3:2", "2:3", "5:4", "4:5", "3:4", "16:9", "9:16"],
                    {"default": "1:1"},
                ),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
                "output_format": (["jpeg", "png"], {"default": "png"}),
                "sync_mode": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "FAL/Image"

    def generate_image(
        self,
        prompt,
        aspect_ratio="1:1",
        num_images=1,
        output_format="png",
        sync_mode=False,
    ):
        arguments = {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "num_images": num_images,
            "output_format": output_format,
            "sync_mode": sync_mode,
        }

        try:
            result = ApiHandler.submit_and_get_result("fal-ai/nano-banana", arguments)
            return ResultProcessor.process_image_result(result)
        except Exception as e:
            return ApiHandler.handle_image_generation_error("Nano Banana Text-to-Image", e)


class NanoBananaEdit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "image_1": ("IMAGE",),
            },
            "optional": {
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "images": ("IMAGE", {"default": None, "multiple": True}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
                "output_format": (["jpeg", "png"], {"default": "jpeg"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "FAL/Image"

    def generate_image(
        self,
        prompt,
        image_1,
        image_2=None,
        image_3=None,
        image_4=None,
        images=None,
        num_images=1,
        output_format="jpeg",
    ):
        # Upload all provided images
        singleImages = ImageUtils.prepare_images([image_1, image_2, image_3, image_4])
        batchImages = ImageUtils.prepare_images(images)
        image_urls = singleImages + batchImages
        

        arguments = {
            "prompt": prompt,
            "image_urls": image_urls,
            "num_images": num_images,
            "output_format": output_format,
        }

        try:
            result = ApiHandler.submit_and_get_result("fal-ai/nano-banana/edit", arguments)
            return ResultProcessor.process_image_result(result)
        except Exception as e:
            return ApiHandler.handle_image_generation_error("Nano Banana Edit", e)


class ReveTextToImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "aspect_ratio": (
                    ["16:9", "4:3", "3:2", "1:1", "2:3", "3:4", "9:16"],
                    {"default": "3:2"},
                ),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
                "output_format": (["jpeg", "png"], {"default": "png"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "FAL/Image"

    def generate_image(
        self,
        prompt,
        aspect_ratio="1:1",
        num_images=1,
        output_format="png",
    ):
        arguments = {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "num_images": num_images,
            "output_format": output_format,
        }

        try:
            result = ApiHandler.submit_and_get_result("fal-ai/reve/text-to-image", arguments)
            return ResultProcessor.process_image_result(result)
        except Exception as e:
            return ApiHandler.handle_image_generation_error("Reve Text-to-Image", e)


class Dreamina31TextToImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "image_size": (
                    [
                        "square_hd",
                        "square",
                        "portrait_4_3",
                        "portrait_16_9",
                        "landscape_4_3",
                        "landscape_16_9",
                        "custom",
                    ],
                    {"default": "square_hd"},
                ),
            },
            "optional": {
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
                "sync_mode": ("BOOLEAN", {"default": False}),
                "output_format": (["jpeg", "png"], {"default": "png"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2**32 - 1}),
                "enhance_prompt": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "FAL/Image"

    def generate_image(
        self,
        prompt,
        image_size,
        num_images=1,
        sync_mode=False,
        output_format="png",
        seed=-1,
        enhance_prompt=False,
    ):
        arguments = {
            "prompt": prompt,
            "image_size": image_size,
            "num_images": num_images,
            "sync_mode": sync_mode,
            "output_format": output_format,
            "enhance_prompt": enhance_prompt,
        }

        if seed > 0:
            arguments["seed"] = seed

        try:
            result = ApiHandler.submit_and_get_result("fal-ai/bytedance/dreamina/v3.1/text-to-image", arguments)
            return ResultProcessor.process_image_result(result)
        except Exception as e:
            return ApiHandler.handle_image_generation_error("Dreamina v3.1 Text-to-Image", e)


# Node class mappings
NODE_CLASS_MAPPINGS = {
    "Ideogramv3_fal": Ideogramv3,
    "Hidreamfull_fal": HidreamFull,
    "FluxPro_fal": FluxPro,
    "FluxDev_fal": FluxDev,
    "FluxSchnell_fal": FluxSchnell,
    "FluxPro11_fal": FluxPro11,
    "FluxPro1Fill_fal": FluxPro1Fill,
    "FluxUltra_fal": FluxUltra,
    "FluxGeneral_fal": FluxGeneral,
    "FluxLora_fal": FluxLora,
    "Recraft_fal": Recraft,
    "Sana_fal": Sana,
    "FluxProKontext_fal": FluxProKontext,
    "FluxProKontextMulti_fal": FluxProKontextMulti,
    "FluxProKontextTextToImage_fal": FluxProKontextTextToImage,
    "Imagen4Preview_fal": Imagen4PreviewNode,
    "QwenImageEdit_fal": QwenImageEdit,
    "QwenImageEditPlusLoRA_fal": QwenImageEditPlusLoRA,
    "SeedEditV3_fal": SeedEditV3,
    "SeedreamV4Edit_fal": SeedreamV4Edit,
    "NanoBananaTextToImage_fal": NanoBananaTextToImage,
    "NanoBananaEdit_fal": NanoBananaEdit,
    "ReveTextToImage_fal": ReveTextToImage,
    "Dreamina31TextToImage_fal": Dreamina31TextToImage,
}


# Display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "Ideogramv3_fal": "Ideogramv3 (fal)",
    "Hidreamfull_fal": "HidreamFull (fal)",
    "FluxPro_fal": "Flux Pro (fal)",
    "FluxDev_fal": "Flux Dev (fal)",
    "FluxSchnell_fal": "Flux Schnell (fal)",
    "FluxPro11_fal": "Flux Pro 1.1 (fal)",
    "FluxPro1Fill_fal": "Flux Pro 1 Fill (fal)",
    "FluxUltra_fal": "Flux Ultra (fal)",
    "FluxGeneral_fal": "Flux General (fal)",
    "FluxLora_fal": "Flux LoRA (fal)",
    "Recraft_fal": "Recraft V3 (fal)",
    "Sana_fal": "Sana (fal)",
    "FluxProKontext_fal": "Flux Pro Kontext (fal)",
    "FluxProKontextMulti_fal": "Flux Pro Kontext Multi (fal)",
    "FluxProKontextTextToImage_fal": "Flux Pro Kontext Text-to-Image (fal)",
    "Imagen4Preview_fal": "Imagen4 Preview (fal)",
    "QwenImageEdit_fal": "Qwen Image Edit (fal)",
    "QwenImageEditPlusLoRA_fal": "Qwen Image Edit Plus LoRA (fal)",
    "SeedEditV3_fal": "SeedEdit 3.0 (fal)",
    "SeedreamV4Edit_fal": "Seedream 4.0 Edit (fal)",
    "NanoBananaTextToImage_fal": "Nano Banana Text-to-Image (fal)",
    "NanoBananaEdit_fal": "Nano Banana Edit (fal)",
    "ReveTextToImage_fal": "Reve Text-to-Image (fal)",
    "Dreamina31TextToImage_fal": "Dreamina v3.1 Text-to-Image (fal)",
}

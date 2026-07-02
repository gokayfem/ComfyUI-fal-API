import os
import tempfile
import zipfile

import torch
from PIL import Image

from .fal_utils import ApiHandler, FalConfig, ImageUtils

# Initialize FalConfig
fal_config = FalConfig()


def create_zip_from_images(images):
    """Create a zip file from a list of images."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as temp_zip:
            with zipfile.ZipFile(temp_zip, "w") as zf:
                for idx, img_tensor in enumerate(images):
                    # Convert tensor to PIL Image
                    if isinstance(img_tensor, torch.Tensor):
                        # Convert to numpy and scale to 0-255 range
                        img_np = (img_tensor.cpu().numpy() * 255).astype("uint8")
                        # Handle different tensor formats
                        if img_np.shape[0] == 3:  # If in format (C, H, W)
                            img_np = img_np.transpose(1, 2, 0)
                        img = Image.fromarray(img_np)
                    else:
                        img = img_tensor

                    # Save image to temporary file
                    with tempfile.NamedTemporaryFile(
                        suffix=".png", delete=False
                    ) as temp_img:
                        img.save(temp_img, format="PNG")
                        temp_img_path = temp_img.name

                    # Add to zip file
                    zf.write(temp_img_path, f"image_{idx}.png")
                    os.unlink(temp_img_path)

            # Upload the zip through the shared utility (raises on failure)
            return ImageUtils.upload_file(temp_zip.name)
    except Exception as e:
        return ApiHandler.handle_text_generation_error(
            "flux-lora-fast-training", f"Failed to create zip file: {str(e)}"
        )


class FluxLoraTrainerNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": (
                    "IMAGE",
                    {"tooltip": "Training images. Ignored when images_zip_url is set."},
                ),
                "steps": (
                    "INT",
                    {
                        "default": 1000,
                        "min": 100,
                        "max": 10000,
                        "step": 100,
                        "tooltip": "Number of training steps.",
                    },
                ),
                "create_masks": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Automatically create segmentation masks for subject training.",
                    },
                ),
                "is_style": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Enable for style LoRAs instead of subject LoRAs.",
                    },
                ),
            },
            "optional": {
                "trigger_word": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Token used to invoke the trained concept in prompts.",
                    },
                ),
                "images_zip_url": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "URL of a pre-uploaded zip of training images. Overrides the IMAGE input.",
                    },
                ),
                "is_input_format_already_preprocessed": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Set when the archive already contains preprocessed data (images + captions).",
                    },
                ),
                "data_archive_format": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Archive format hint (e.g. 'zip') when it cannot be inferred from the URL.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lora_file_url",)
    FUNCTION = "train_lora"
    CATEGORY = "FAL/Training"

    def train_lora(
        self,
        images,
        steps,
        create_masks,
        is_style,
        trigger_word="",
        images_zip_url="",
        is_input_format_already_preprocessed=False,
        data_archive_format="",
    ):
        try:
            # Use provided zip URL if available, otherwise create and upload zip file
            images_url = (
                images_zip_url if images_zip_url else create_zip_from_images(images)
            )
            if not images_url:
                return ApiHandler.handle_text_generation_error(
                    "flux-lora-fast-training", "Failed to upload images"
                )

            # Prepare arguments for the API
            arguments = {
                "images_data_url": images_url,
                "steps": steps,
                "create_masks": create_masks,
                "is_style": is_style,
                "is_input_format_already_preprocessed": is_input_format_already_preprocessed,
            }

            if trigger_word:
                arguments["trigger_word"] = trigger_word

            if data_archive_format:
                arguments["data_archive_format"] = data_archive_format

            # Submit training job
            result = ApiHandler.submit_and_get_result(
                "fal-ai/flux-lora-fast-training", arguments
            )
            lora_url = result["diffusers_lora_file"]["url"]
            return (lora_url,)

        except Exception as e:
            return ApiHandler.handle_text_generation_error(
                "flux-lora-fast-training", e
            )


class HunyuanVideoLoraTrainerNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": (
                    "IMAGE",
                    {"tooltip": "Training images. Ignored when images_zip_url is set."},
                ),
                "steps": (
                    "INT",
                    {
                        "default": 1000,
                        "min": 100,
                        "max": 10000,
                        "step": 100,
                        "tooltip": "Number of training steps.",
                    },
                ),
            },
            "optional": {
                "trigger_word": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Token used to invoke the trained concept in prompts.",
                    },
                ),
                "learning_rate": (
                    "FLOAT",
                    {
                        "default": 0.0001,
                        "min": 0.00001,
                        "max": 0.01,
                        "tooltip": "Training learning rate.",
                    },
                ),
                "do_caption": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Automatically caption the training images.",
                    },
                ),
                "images_zip_url": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "URL of a pre-uploaded zip of training images. Overrides the IMAGE input.",
                    },
                ),
                "data_archive_format": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Archive format hint (e.g. 'zip') when it cannot be inferred from the URL.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lora_file_url",)
    FUNCTION = "train_lora"
    CATEGORY = "FAL/Training"

    def train_lora(
        self,
        images,
        steps,
        trigger_word="",
        learning_rate=0.0001,
        do_caption=True,
        images_zip_url="",
        data_archive_format="",
    ):
        try:
            # Use provided zip URL if available, otherwise create and upload zip file
            images_url = (
                images_zip_url if images_zip_url else create_zip_from_images(images)
            )
            if not images_url:
                return ApiHandler.handle_text_generation_error(
                    "hunyuan-video-lora-training", "Failed to upload images"
                )

            # Prepare arguments for the API
            arguments = {
                "images_data_url": images_url,
                "steps": steps,
                "learning_rate": learning_rate,
                "do_caption": do_caption,
            }

            if trigger_word:
                arguments["trigger_word"] = trigger_word

            if data_archive_format:
                arguments["data_archive_format"] = data_archive_format

            # Submit training job
            result = ApiHandler.submit_and_get_result(
                "fal-ai/hunyuan-video-lora-training", arguments
            )
            lora_url = result["diffusers_lora_file"]["url"]
            return (lora_url,)

        except Exception as e:
            return ApiHandler.handle_text_generation_error(
                "hunyuan-video-lora-training", e
            )


class WanLoraTrainerNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "training_data_url": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "URL of the training data archive (images/videos with optional captions).",
                    },
                ),
                "number_of_steps": (
                    "INT",
                    {
                        "default": 400,
                        "min": 5,
                        "max": 10000,
                        "step": 1,
                        "tooltip": "Number of training steps.",
                    },
                ),
                "learning_rate": (
                    "FLOAT",
                    {
                        "default": 0.0002,
                        "min": 0.00001,
                        "max": 0.01,
                        "tooltip": "Training learning rate.",
                    },
                ),
            },
            "optional": {
                "trigger_phrase": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Phrase used to invoke the trained concept in prompts.",
                    },
                ),
                "auto_scale_input": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Automatically rescale input media to the training resolution.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lora_file_url",)
    FUNCTION = "train_lora"
    CATEGORY = "FAL/Training"

    def train_lora(
        self,
        training_data_url,
        number_of_steps,
        learning_rate,
        trigger_phrase="",
        auto_scale_input=True,
    ):
        try:
            if not training_data_url:
                return ApiHandler.handle_text_generation_error(
                    "wan-trainer", "No training data URL provided"
                )

            # Prepare arguments for the API
            arguments = {
                "training_data_url": training_data_url,
                "number_of_steps": number_of_steps,
                "learning_rate": learning_rate,
                "auto_scale_input": auto_scale_input,
            }

            if trigger_phrase:
                arguments["trigger_phrase"] = trigger_phrase

            # Submit training job
            result = ApiHandler.submit_and_get_result("fal-ai/wan-trainer", arguments)
            lora_url = result["lora_file"]["url"]
            return (lora_url,)

        except Exception as e:
            return ApiHandler.handle_text_generation_error("wan-trainer", e)


class LtxVideoTrainerNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "training_data_url": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "URL of the training data archive (videos/images with optional captions).",
                    },
                ),
                "rank": (
                    ["8", "16", "32", "64", "128"],
                    {
                        "default": "128",
                        "tooltip": "LoRA rank. Higher rank captures more detail but produces larger files.",
                    },
                ),
                "number_of_steps": (
                    "INT",
                    {
                        "default": 1000,
                        "min": 100,
                        "max": 10000,
                        "step": 1,
                        "tooltip": "Number of training steps.",
                    },
                ),
                "number_of_frames": (
                    "INT",
                    {
                        "default": 81,
                        "min": 1,
                        "max": 1000,
                        "tooltip": "Frames per training sample.",
                    },
                ),
                "frame_rate": (
                    "INT",
                    {
                        "default": 25,
                        "min": 1,
                        "max": 60,
                        "tooltip": "Frame rate used for training samples.",
                    },
                ),
                "resolution": (
                    ["low", "medium", "high"],
                    {"default": "medium", "tooltip": "Training resolution."},
                ),
                "aspect_ratio": (
                    ["16:9", "1:1", "9:16"],
                    {"default": "1:1", "tooltip": "Training aspect ratio."},
                ),
                "learning_rate": (
                    "FLOAT",
                    {
                        "default": 0.0002,
                        "min": 0.00001,
                        "max": 0.01,
                        "tooltip": "Training learning rate.",
                    },
                ),
            },
            "optional": {
                "trigger_phrase": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Phrase used to invoke the trained concept in prompts.",
                    },
                ),
                "auto_scale_input": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Automatically rescale input media to the training resolution.",
                    },
                ),
                "split_input_into_scenes": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Split long input videos into individual scenes before training.",
                    },
                ),
                "split_input_duration_threshold": (
                    "FLOAT",
                    {
                        "default": 30.0,
                        "min": 1.0,
                        "max": 300.0,
                        "tooltip": "Videos longer than this many seconds are split into scenes.",
                    },
                ),
                "validation_negative_prompt": (
                    "STRING",
                    {
                        "default": "blurry, low quality, bad quality, out of focus",
                        "tooltip": "Negative prompt used for validation renders during training.",
                    },
                ),
                "validation_number_of_frames": (
                    "INT",
                    {
                        "default": 81,
                        "min": 1,
                        "max": 1000,
                        "tooltip": "Frames per validation render.",
                    },
                ),
                "validation_resolution": (
                    ["low", "medium", "high"],
                    {"default": "high", "tooltip": "Resolution of validation renders."},
                ),
                "validation_aspect_ratio": (
                    ["16:9", "1:1", "9:16"],
                    {"default": "1:1", "tooltip": "Aspect ratio of validation renders."},
                ),
                "validation_reverse": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Also render reversed validation videos.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lora_file_url",)
    FUNCTION = "train_lora"
    CATEGORY = "FAL/Training"

    def train_lora(
        self,
        training_data_url,
        rank,
        number_of_steps,
        number_of_frames,
        frame_rate,
        resolution,
        aspect_ratio,
        learning_rate,
        trigger_phrase="",
        auto_scale_input=False,
        split_input_into_scenes=True,
        split_input_duration_threshold=30.0,
        validation_negative_prompt="blurry, low quality, bad quality, out of focus",
        validation_number_of_frames=81,
        validation_resolution="high",
        validation_aspect_ratio="1:1",
        validation_reverse=False,
    ):
        try:
            if not training_data_url:
                return ApiHandler.handle_text_generation_error(
                    "ltx-video-trainer", "No training data URL provided"
                )

            # Prepare arguments for the API
            arguments = {
                "training_data_url": training_data_url,
                "rank": int(rank),
                "number_of_steps": number_of_steps,
                "number_of_frames": number_of_frames,
                "frame_rate": frame_rate,
                "resolution": resolution,
                "aspect_ratio": aspect_ratio,
                "learning_rate": learning_rate,
                "auto_scale_input": auto_scale_input,
                "split_input_into_scenes": split_input_into_scenes,
                "split_input_duration_threshold": split_input_duration_threshold,
                "validation_negative_prompt": validation_negative_prompt,
                "validation_number_of_frames": validation_number_of_frames,
                "validation_resolution": validation_resolution,
                "validation_aspect_ratio": validation_aspect_ratio,
                "validation_reverse": validation_reverse,
            }

            if trigger_phrase:
                arguments["trigger_phrase"] = trigger_phrase

            # Submit training job
            result = ApiHandler.submit_and_get_result(
                "fal-ai/ltx-video-trainer", arguments
            )
            lora_url = result["lora_file"]["url"]
            return (lora_url,)

        except Exception as e:
            return ApiHandler.handle_text_generation_error("ltx-video-trainer", e)


# Node class mappings
NODE_CLASS_MAPPINGS = {
    "FluxLoraTrainer_fal": FluxLoraTrainerNode,
    "HunyuanVideoLoraTrainer_fal": HunyuanVideoLoraTrainerNode,
    "WanLoraTrainer_fal": WanLoraTrainerNode,
    "LtxVideoTrainer_fal": LtxVideoTrainerNode,
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxLoraTrainer_fal": "Flux LoRA Trainer (fal)",
    "HunyuanVideoLoraTrainer_fal": "Hunyuan Video LoRA Trainer (fal)",
    "WanLoraTrainer_fal": "WAN LoRA Trainer (fal)",
    "LtxVideoTrainer_fal": "LTX Video LoRA Trainer (fal)",
}

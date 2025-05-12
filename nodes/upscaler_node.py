import os
import configparser
import tempfile
import requests
from PIL import Image
import io
import numpy as np
import torch
from fal_client.client import SyncClient

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
config_path = os.path.join(parent_dir, "config.ini")

config = configparser.ConfigParser()
config.read(config_path)

try:
    if os.environ.get("FAL_KEY"):
        fal_key = os.environ["FAL_KEY"]
    else:
        fal_key = config['API']['FAL_KEY']
        os.environ["FAL_KEY"] = fal_key

except KeyError:
    print("Error: FAL_KEY not found in config.ini or environment variables")

# Create the client with API key
fal_client = SyncClient(key=fal_key)

def upload_image(image):
    try:
        if isinstance(image, torch.Tensor):
            image_np = image.cpu().numpy()
        else:
            image_np = np.array(image)

        if image_np.ndim == 4:
            image_np = image_np.squeeze(0)
        if image_np.ndim == 2:
            image_np = np.stack([image_np] * 3, axis=-1)
        elif image_np.shape[0] == 3:
            image_np = np.transpose(image_np, (1, 2, 0))

        if image_np.dtype == np.float32 or image_np.dtype == np.float64:
            image_np = (image_np * 255).astype(np.uint8)

        pil_image = Image.fromarray(image_np)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            pil_image.save(temp_file, format="PNG")
            temp_file_path = temp_file.name

        image_url = fal_client.upload_file(temp_file_path)
        return image_url
    except Exception as e:
        print(f"Error uploading image: {str(e)}")
        return None
    finally:
        if 'temp_file_path' in locals():
            os.unlink(temp_file_path)

class UpscalerNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "upscale_factor": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.5}),
                "negative_prompt": ("STRING", {"default": "(worst quality, low quality, normal quality:2)", "multiline": True}),
                "creativity": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05}),
                "resemblance": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05}),
                "guidance_scale": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 20.0, "step": 0.5}),
                "num_inference_steps": ("INT", {"default": 18, "min": 1, "max": 100}),
                "enable_safety_checker": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "seed": ("INT", {"default": -1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_upscaled_image"
    CATEGORY = "FAL/Image"

    def generate_upscaled_image(self, image, upscale_factor, negative_prompt, creativity, resemblance, guidance_scale, num_inference_steps, enable_safety_checker, seed=-1):
        image_url = upload_image(image)
        if not image_url:
            print("Failed to upload image for upscaling.")
            return self.create_blank_image()

        arguments = {
            "image_url": image_url,
            "prompt": "masterpiece, best quality, highres",
            "upscale_factor": upscale_factor,
            "negative_prompt": negative_prompt,
            "creativity": creativity,
            "resemblance": resemblance,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "enable_safety_checker": enable_safety_checker
        }

        if seed != -1:
            arguments["seed"] = seed

        try:
            handler = fal_client.submit("fal-ai/clarity-upscaler", arguments=arguments)
            result = handler.get()
            return self.process_result(result)
        except Exception as e:
            print(f"Error generating upscaled image: {str(e)}")
            return self.create_blank_image()

    def process_result(self, result):
        try:
            img_url = result["image"]["url"]
            img_response = requests.get(img_url)
            img = Image.open(io.BytesIO(img_response.content))
            img_array = np.array(img).astype(np.float32) / 255.0

            # Stack the images along a new first dimension
            stacked_images = np.stack([img_array], axis=0)
            
            # Convert to PyTorch tensor
            img_tensor = torch.from_numpy(stacked_images)
            return (img_tensor,)
        except Exception as e:
            print(f"Error processing result: {str(e)}")
            return self.create_blank_image()

    def create_blank_image(self):   
        blank_img = Image.new('RGB', (512, 512), color='black')
        img_array = np.array(blank_img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array)[None,]
        return (img_tensor,)

# Node class mappings
NODE_CLASS_MAPPINGS = {
    "Upscaler_fal": UpscalerNode,
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "Upscaler_fal": "Clarity Upscaler (fal)",
}

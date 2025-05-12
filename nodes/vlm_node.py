import os
import configparser
from fal_client.client import SyncClient
import torch
from PIL import Image
import tempfile
import numpy as np

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

class VLMNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "model": (["google/gemini-flash-1.5-8b", "anthropic/claude-3.5-sonnet", "anthropic/claude-3-haiku", 
                           "google/gemini-pro-1.5", "google/gemini-flash-1.5", "openai/gpt-4o"], 
                          {"default": "google/gemini-flash-1.5-8b"}),
                "system_prompt": ("STRING", {"default": "", "multiline": True}),
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_text"
    CATEGORY = "FAL/VLM"

    def generate_text(self, prompt, model, system_prompt, image):
        try:
            # Convert the image tensor to a numpy array
            if isinstance(image, torch.Tensor):
                image_np = image.cpu().numpy()
            else:
                image_np = np.array(image)

            # Ensure the image is in the correct format (H, W, C)
            if image_np.ndim == 4:
                image_np = image_np.squeeze(0)  # Remove batch dimension if present
            if image_np.ndim == 2:
                image_np = np.stack([image_np] * 3, axis=-1)  # Convert grayscale to RGB
            elif image_np.shape[0] == 3:
                image_np = np.transpose(image_np, (1, 2, 0))  # Change from (C, H, W) to (H, W, C)

            # Normalize the image data to 0-255 range
            if image_np.dtype == np.float32 or image_np.dtype == np.float64:
                image_np = (image_np * 255).astype(np.uint8)

            # Convert to PIL Image
            pil_image = Image.fromarray(image_np)

            # Save the image to a temporary file
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                pil_image.save(temp_file, format="PNG")
                temp_file_path = temp_file.name

            # Upload the temporary file
            image_url = fal_client.upload_file(temp_file_path)

            arguments = {
                "model": model,
                "prompt": prompt,
                "system_prompt": system_prompt,
                "image_url": image_url,
            }

            handler = fal_client.submit("fal-ai/any-llm/vision", arguments=arguments)
            result = handler.get()
            return (result["output"],)
        except Exception as e:
            print(f"Error generating text with VLM: {str(e)}")
            return ("Error: Unable to generate text.",)
        finally:
            # Clean up the temporary file
            if 'temp_file_path' in locals():
                os.unlink(temp_file_path)

# Node class mappings
NODE_CLASS_MAPPINGS = {
    "VLM_fal": VLMNode,
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "VLM_fal": "VLM (fal)",
}

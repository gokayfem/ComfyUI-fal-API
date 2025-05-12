import os
import configparser
from fal_client.client import SyncClient
import tempfile
import zipfile
import torch
from PIL import Image

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

def create_zip_from_images(images):
    """Create a zip file from a list of images."""
    with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_zip:
        with zipfile.ZipFile(temp_zip, 'w') as zf:
            for idx, img_tensor in enumerate(images):
                # Convert tensor to PIL Image
                if isinstance(img_tensor, torch.Tensor):
                    # Convert to numpy and scale to 0-255 range
                    img_np = (img_tensor.cpu().numpy() * 255).astype('uint8')
                    # Handle different tensor formats
                    if img_np.shape[0] == 3:  # If in format (C, H, W)
                        img_np = img_np.transpose(1, 2, 0)
                    img = Image.fromarray(img_np)
                else:
                    img = img_tensor

                # Save image to temporary file
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_img:
                    img.save(temp_img, format='PNG')
                    temp_img_path = temp_img.name
                
                # Add to zip file
                zf.write(temp_img_path, f'image_{idx}.png')
                os.unlink(temp_img_path)
        
        return fal_client.upload_file(temp_zip.name)

class FluxLoraTrainerNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "steps": ("INT", {"default": 1000, "min": 100, "max": 10000, "step": 100}),
                "create_masks": ("BOOLEAN", {"default": True}),
                "is_style": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "trigger_word": ("STRING", {"default": ""}),
                "images_zip_url": ("STRING", {"default": ""}),
                "is_input_format_already_preprocessed": ("BOOLEAN", {"default": False}),
                "data_archive_format": ("STRING", {"default": ""}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lora_file_url",)
    FUNCTION = "train_lora"
    CATEGORY = "FAL/Training"

    def train_lora(self, images, steps, create_masks, is_style, trigger_word="", images_zip_url="", 
                  is_input_format_already_preprocessed=False, data_archive_format=""):
        try:
            # Use provided zip URL if available, otherwise create and upload zip file
            images_url = images_zip_url if images_zip_url else create_zip_from_images(images)
            if not images_url:
                return ("Error: Unable to upload images.", "")

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
            handler = fal_client.submit("fal-ai/flux-lora-fast-training", arguments=arguments)
            result = handler.get()

            lora_url = result["diffusers_lora_file"]["url"]

            return (lora_url, )

        except Exception as e:
            print(f"Error during LoRA training: {str(e)}")
            return ("Error: Training failed.", "")

class HunyuanVideoLoraTrainerNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "steps": ("INT", {"default": 1000, "min": 100, "max": 10000, "step": 100}),
            },
            "optional": {
                "trigger_word": ("STRING", {"default": ""}),
                "learning_rate": ("FLOAT", {"default": 0.0001, "min": 0.00001, "max": 0.01}),
                "do_caption": ("BOOLEAN", {"default": True}),
                "images_zip_url": ("STRING", {"default": ""}),
                "data_archive_format": ("STRING", {"default": ""}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lora_file_url",)
    FUNCTION = "train_lora"
    CATEGORY = "FAL/Training"

    def train_lora(self, images, steps, trigger_word="", learning_rate=0.0001, do_caption=True, 
                  images_zip_url="", data_archive_format=""):
        try:
            # Use provided zip URL if available, otherwise create and upload zip file
            images_url = images_zip_url if images_zip_url else create_zip_from_images(images)
            if not images_url:
                return ("Error: Unable to upload images.", "")

            # Prepare arguments for the API
            arguments = {
                "images_data_url": images_url,
                "steps": steps,
                "learning_rate": learning_rate,
                "do_caption": do_caption
            }
            
            if trigger_word:
                arguments["trigger_word"] = trigger_word
                
            if data_archive_format:
                arguments["data_archive_format"] = data_archive_format

            # Submit training job
            handler = fal_client.submit("fal-ai/hunyuan-video-lora-training", arguments=arguments)
            result = handler.get()

            lora_url = result["diffusers_lora_file"]["url"]

            return (lora_url,)

        except Exception as e:
            print(f"Error during LoRA training: {str(e)}")
            return ("Error: Training failed.", "")

# Node class mappings
NODE_CLASS_MAPPINGS = {
    "FluxLoraTrainer_fal": FluxLoraTrainerNode,
    "HunyuanVideoLoraTrainer_fal": HunyuanVideoLoraTrainerNode,
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxLoraTrainer_fal": "Flux LoRA Trainer (fal)",
    "HunyuanVideoLoraTrainer_fal": "Hunyuan Video LoRA Trainer (fal)",
}

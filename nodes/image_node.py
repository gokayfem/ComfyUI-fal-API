import requests
from PIL import Image
import io
import numpy as np
import torch
import os
import configparser
import tempfile
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
        return image_url
    except Exception as e:
        print(f"Error uploading image: {str(e)}")
        return None
    finally:
        # Clean up the temporary file
        if 'temp_file_path' in locals():
            os.unlink(temp_file_path)

class Sana:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "image_size": (["square_hd", "square", "portrait_4_3", "portrait_16_9", "landscape_4_3", "landscape_16_9", "custom"], {"default": "square_hd"}),
                "width": ("INT", {"default": 3840, "min": 512, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 2160, "min": 512, "max": 4096, "step": 16}),
                "num_inference_steps": ("INT", {"default": 18, "min": 1, "max": 50}),
                "guidance_scale": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 20.0, "step": 0.1}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "seed": ("INT", {"default": -1}),
                "enable_safety_checker": ("BOOLEAN", {"default": True}),
                "output_format": (["png", "jpeg"], {"default": "png"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "FAL/Image"

    def generate_image(self, prompt, image_size, width, height, num_inference_steps, guidance_scale, num_images, negative_prompt="", seed=-1, enable_safety_checker=True, output_format="png"):
        arguments = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "num_images": num_images,
            "enable_safety_checker": enable_safety_checker,
            "output_format": output_format
        }

        if image_size == "custom":
            arguments["image_size"] = {"width": width, "height": height}
        else:
            arguments["image_size"] = image_size

        if seed != -1:
            arguments["seed"] = seed

        try:
            handler = fal_client.submit("fal-ai/sana", arguments=arguments)
            result = handler.get()
            return self.process_result(result)
        except Exception as e:
            print(f"Error generating image with Sana: {str(e)}")
            return self.create_blank_image()

class Recraft:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "image_size": (["square_hd", "square", "portrait_4_3", "portrait_16_9", "landscape_4_3", "landscape_16_9", "custom"], {"default": "square_hd"}),
                "width": ("INT", {"default": 512, "min": 512, "max": 2048, "step": 16}),
                "height": ("INT", {"default": 512, "min": 512, "max": 2048, "step": 16}),
                "style": ([
                    "any", "realistic_image", "digital_illustration", "vector_illustration",
                    "realistic_image/b_and_w", "realistic_image/hard_flash", "realistic_image/hdr",
                    "realistic_image/natural_light", "realistic_image/studio_portrait",
                    "realistic_image/enterprise", "realistic_image/motion_blur",
                    "digital_illustration/pixel_art", "digital_illustration/hand_drawn",
                    "digital_illustration/grain", "digital_illustration/infantile_sketch",
                    "digital_illustration/2d_art_poster", "digital_illustration/handmade_3d",
                    "digital_illustration/hand_drawn_outline", "digital_illustration/engraving_color",
                    "digital_illustration/2d_art_poster_2", "vector_illustration/engraving",
                    "vector_illustration/line_art", "vector_illustration/line_circuit",
                    "vector_illustration/linocut"
                ], {"default": "realistic_image"}),
            },
            "optional": {
                "style_id": ("STRING", {"default": ""}),
            }
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
            handler = fal_client.submit("fal-ai/recraft-v3", arguments=arguments)
            result = handler.get()
            return self.process_result(result)
        except Exception as e:
            print(f"Error generating image with Recraft: {str(e)}")
            return self.create_blank_image()

class FluxPro:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "image_size": (["square_hd", "square", "portrait_4_3", "portrait_16_9", "landscape_4_3", "landscape_16_9", "custom"], {"default": "landscape_4_3"}),
                "width": ("INT", {"default": 1024, "min": 512, "max": 1440, "step": 32}),
                "height": ("INT", {"default": 768, "min": 512, "max": 1440, "step": 32}),
                "num_inference_steps": ("INT", {"default": 28, "min": 1, "max": 100}),
                "guidance_scale": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 20.0}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 10}),
                "safety_tolerance": (["1", "2", "3", "4", "5", "6"], {"default": "2"}),
            },
            "optional": {
                "seed": ("INT", {"default": -1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "FAL/Image"

    def generate_image(self, prompt, image_size, width, height, num_inference_steps, guidance_scale, num_images, safety_tolerance, seed=-1):
        arguments = {
            "prompt": prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "num_images": num_images,
            "safety_tolerance": safety_tolerance
        }
        if image_size == "custom":
            arguments["image_size"] = {"width": width, "height": height}
        else:
            arguments["image_size"] = image_size
        if seed != -1:
            arguments["seed"] = seed

        try:
            handler = fal_client.submit("fal-ai/flux-pro", arguments=arguments)
            result = handler.get()
            return self.process_result(result)
        except Exception as e:
            print(f"Error generating image with FluxPro: {str(e)}")
            return self.create_blank_image()

class FluxDev:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "image_size": (["square_hd", "square", "portrait_4_3", "portrait_16_9", "landscape_4_3", "landscape_16_9", "custom"], {"default": "landscape_4_3"}),
                "width": ("INT", {"default": 1024, "min": 512, "max": 1536, "step": 16}),
                "height": ("INT", {"default": 768, "min": 512, "max": 1536, "step": 16}),
                "num_inference_steps": ("INT", {"default": 28, "min": 1, "max": 100}),
                "guidance_scale": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 20.0}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 10}),
                "enable_safety_checker": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "seed": ("INT", {"default": -1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "FAL/Image"

    def generate_image(self, prompt, image_size, width, height, num_inference_steps, guidance_scale, num_images, enable_safety_checker, seed=-1):
        arguments = {
            "prompt": prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "num_images": num_images,
            "enable_safety_checker": enable_safety_checker
        }
        if image_size == "custom":
            arguments["image_size"] = {"width": width, "height": height}
        else:
            arguments["image_size"] = image_size
        if seed != -1:
            arguments["seed"] = seed

        try:
            handler = fal_client.submit("fal-ai/flux/dev", arguments=arguments)
            result = handler.get()
            return self.process_result(result)
        except Exception as e:
            print(f"Error generating image with FluxDev: {str(e)}")
            return self.create_blank_image()

class FluxSchnell:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "image_size": (["square_hd", "square", "portrait_4_3", "portrait_16_9", "landscape_4_3", "landscape_16_9", "custom"], {"default": "landscape_4_3"}),
                "width": ("INT", {"default": 1024, "min": 512, "max": 1536, "step": 32}),
                "height": ("INT", {"default": 768, "min": 512, "max": 1536, "step": 32}),
                "num_inference_steps": ("INT", {"default": 4, "min": 1, "max": 100}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 10}),
                "enable_safety_checker": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "seed": ("INT", {"default": -1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "FAL/Image"

    def generate_image(self, prompt, image_size, width, height, num_inference_steps, num_images, enable_safety_checker, seed=-1):
        arguments = {
            "prompt": prompt,
            "num_inference_steps": num_inference_steps,
            "num_images": num_images,
            "enable_safety_checker": enable_safety_checker
        }
        if image_size == "custom":
            arguments["image_size"] = {"width": width, "height": height}
        else:
            arguments["image_size"] = image_size
        if seed != -1:
            arguments["seed"] = seed

        try:
            handler = fal_client.submit("fal-ai/flux/schnell", arguments=arguments)
            result = handler.get()
            return self.process_result(result)
        except Exception as e:
            print(f"Error generating image with FluxSchnell: {str(e)}")
            return self.create_blank_image()

class FluxPro11:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "image_size": (["square_hd", "square", "portrait_4_3", "portrait_16_9", "landscape_4_3", "landscape_16_9", "custom"], {"default": "landscape_4_3"}),
                "width": ("INT", {"default": 1024, "min": 512, "max": 1440, "step": 32}),
                "height": ("INT", {"default": 768, "min": 512, "max": 1440, "step": 32}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 10}),
                "safety_tolerance": (["1", "2", "3", "4", "5", "6"], {"default": "2"}),
            },
            "optional": {
                "seed": ("INT", {"default": -1}),
                "sync_mode": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "FAL/Image"

    def generate_image(self, prompt, image_size, width, height, num_images, safety_tolerance, seed=-1, sync_mode=False):
        arguments = {
            "prompt": prompt,
            "num_images": num_images,
            "safety_tolerance": safety_tolerance,
            "sync_mode": sync_mode
        }
        if image_size == "custom":
            arguments["image_size"] = {"width": width, "height": height}
        else:
            arguments["image_size"] = image_size
        if seed != -1:
            arguments["seed"] = seed

        try:
            handler = fal_client.submit("fal-ai/flux-pro/v1.1", arguments=arguments)
            result = handler.get()
            return self.process_result(result)
        except Exception as e:
            print(f"Error generating image with FluxPro 1.1: {str(e)}")
            return self.create_blank_image()

class FluxUltra:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "aspect_ratio": (["21:9", "16:9", "4:3", "1:1", "3:4", "9:16", "9:21"], {"default": "16:9"}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 1}),
                "safety_tolerance": (["1", "2", "3", "4", "5", "6"], {"default": "2"}),
                "enable_safety_checker": ("BOOLEAN", {"default": True}),
                "raw": ("BOOLEAN", {"default": False}),
                "sync_mode": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "seed": ("INT", {"default": -1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "FAL/Image"

    def generate_image(self, prompt, aspect_ratio, num_images, safety_tolerance, enable_safety_checker, raw, sync_mode, seed=-1):
        arguments = {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "num_images": num_images,
            "safety_tolerance": safety_tolerance,
            "enable_safety_checker": enable_safety_checker,
            "raw": raw,
            "sync_mode": sync_mode
        }
        if seed != -1:
            arguments["seed"] = seed

        try:
            handler = fal_client.submit("fal-ai/flux-pro/v1.1-ultra", arguments=arguments)
            result = handler.get()
            return self.process_result(result)
        except Exception as e:
            print(f"Error generating image with FluxUltra: {str(e)}")
            return self.create_blank_image()        

class FluxLora:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "image_size": (["square_hd", "square", "portrait_4_3", "portrait_16_9", "landscape_4_3", "landscape_16_9", "custom"], {"default": "landscape_4_3"}),
                "width": ("INT", {"default": 1024, "min": 512, "max": 1536, "step": 16}),
                "height": ("INT", {"default": 768, "min": 512, "max": 1536, "step": 16}),
                "num_inference_steps": ("INT", {"default": 28, "min": 1, "max": 50}),
                "guidance_scale": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
                "enable_safety_checker": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "seed": ("INT", {"default": -1}),
                "lora_path_1": ("STRING", {"default": ""}),
                "lora_scale_1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "lora_path_2": ("STRING", {"default": ""}),
                "lora_scale_2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "FAL/Image"

    def generate_image(self, prompt, image_size, width, height, num_inference_steps, guidance_scale, num_images, enable_safety_checker, seed=-1, lora_path_1="", lora_scale_1=1.0, lora_path_2="", lora_scale_2=1.0):
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
            handler = fal_client.submit("fal-ai/flux-lora", arguments=arguments)
            result = handler.get()
            return self.process_result(result)
        except Exception as e:
            print(f"Error generating image with FluxLora: {str(e)}")
            return self.create_blank_image()

class FluxGeneral:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "image_size": (["square_hd", "square", "portrait_4_3", "portrait_16_9", "landscape_4_3", "landscape_16_9", "custom"], {"default": "landscape_4_3"}),
                "width": ("INT", {"default": 1024, "min": 512, "max": 1536, "step": 16}),
                "height": ("INT", {"default": 768, "min": 512, "max": 1536, "step": 16}),
                "num_inference_steps": ("INT", {"default": 28, "min": 1, "max": 50}),
                "guidance_scale": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "real_cfg_scale": ("FLOAT", {"default": 3.3, "min": 0.0, "max": 5.0, "step": 0.1}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
                "enable_safety_checker": ("BOOLEAN", {"default": False}),
                "use_real_cfg": ("BOOLEAN", {"default": False}),
                "sync_mode": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "seed": ("INT", {"default": -1}),
                "ip_adapter_scale": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.1}),
                "controlnet_conditioning_scale": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.1}),
                "ip_adapters": (["None", "XLabs-AI/flux-ip-adapter"], {"default": "None"}),
                "controlnets": ([
                    "None",
                    "XLabs-AI/flux-controlnet-depth-v3",
                    "Shakker-Labs/FLUX.1-dev-ControlNet-Depth",
                    "jasperai/Flux.1-dev-Controlnet-Depth",
                    "jasperai/Flux.1-dev-Controlnet-Surface-Normals",
                    "XLabs-AI/flux-controlnet-canny-v3",
                    "InstantX/FLUX.1-dev-Controlnet-Canny",
                    "jasperai/Flux.1-dev-Controlnet-Upscaler",
                    "promeai/FLUX.1-controlnet-lineart-promeai"
                ], {"default": "None"}),
                "controlnet_unions": ([
                    "None",
                    "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro",
                    "InstantX/FLUX.1-dev-Controlnet-Union"
                ], {"default": "None"}),
                "controlnet_union_control_mode": (["canny", "tile", "depth", "blur", "pose", "gray", "low_quality"], {"default": "canny"}),
                "control_image": ("IMAGE",),
                "control_mask": ("MASK",),
                "ip_adapter_image": ("IMAGE",),
                "ip_adapter_mask": ("MASK",),
                "lora_path_1": ("STRING", {"default": ""}),
                "lora_scale_1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "lora_path_2": ("STRING", {"default": ""}),
                "lora_scale_2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "FAL/Image"

    def generate_image(self, prompt, image_size, width, height, num_inference_steps, guidance_scale, real_cfg_scale, num_images, enable_safety_checker, use_real_cfg, sync_mode, seed=-1, lora_path_1="", lora_scale_1=1.0, lora_path_2="", lora_scale_2=1.0, ip_adapter_scale=0.6, controlnet_conditioning_scale=0.6, controlnet_union_control_mode="canny", ip_adapters="None", controlnets="None", controlnet_unions="None", control_image=None, control_mask=None, ip_adapter_image=None, ip_adapter_mask=None):
        arguments = {
            "prompt": prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "real_cfg_scale": real_cfg_scale,
            "num_images": num_images,
            "enable_safety_checker": enable_safety_checker,
            "use_real_cfg": use_real_cfg,
            "sync_mode": sync_mode
        }
        if image_size == "custom":
            arguments["image_size"] = {"width": width, "height": height}
        else:
            arguments["image_size"] = image_size
        if seed != -1:
            arguments["seed"] = seed

        # Add ip_adapters if selected
        if ip_adapters != "None":
            arguments["ip_adapters"] = [{
                "path": ip_adapters,
                "image_encoder_path": "openai/clip-vit-large-patch14",
                "scale": ip_adapter_scale
            }]

        # Controlnet mapping
        controlnet_mapping = {
            "XLabs-AI/flux-controlnet-depth-v3": "https://huggingface.co/XLabs-AI/flux-controlnet-depth-v3/resolve/main/flux-depth-controlnet-v3.safetensors",
            "Shakker-Labs/FLUX.1-dev-ControlNet-Depth": "https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Depth/resolve/main/diffusion_pytorch_model.safetensors",
            "jasperai/Flux.1-dev-Controlnet-Depth": "https://huggingface.co/jasperai/Flux.1-dev-Controlnet-Depth/resolve/main/diffusion_pytorch_model.safetensors",
            "jasperai/Flux.1-dev-Controlnet-Surface-Normals": "https://huggingface.co/jasperai/Flux.1-dev-Controlnet-Surface-Normals/resolve/main/diffusion_pytorch_model.safetensors",
            "XLabs-AI/flux-controlnet-canny-v3": "https://huggingface.co/XLabs-AI/flux-controlnet-canny-v3/resolve/main/flux-canny-controlnet-v3.safetensors",
            "InstantX/FLUX.1-dev-Controlnet-Canny": "https://huggingface.co/InstantX/FLUX.1-dev-Controlnet-Canny/resolve/main/diffusion_pytorch_model.safetensors",
            "jasperai/Flux.1-dev-Controlnet-Upscaler": "https://huggingface.co/jasperai/Flux.1-dev-Controlnet-Upscaler/resolve/main/diffusion_pytorch_model.safetensors",
            "promeai/FLUX.1-controlnet-lineart-promeai": "https://huggingface.co/promeai/FLUX.1-controlnet-lineart-promeai/resolve/main/diffusion_pytorch_model.safetensors"
        }

        # Add controlnets if selected
        if controlnets != "None":
            controlnet_path = controlnet_mapping.get(controlnets, controlnets)
            arguments["controlnets"] = [{
                "path": controlnet_path,
                "conditioning_scale": controlnet_conditioning_scale
            }]

        # Add controlnet_unions if selected
        if controlnet_unions != "None":
            arguments["controlnet_unions"] = [{
                "path": controlnet_unions,
                "controls": [{
                    "control_mode": controlnet_union_control_mode,
                }]
            }]

        # Handle controlnets
        if controlnets != "None" and control_image is not None:
            control_image_url = upload_image(control_image)
            if control_image_url:
                controlnet_path = controlnet_mapping.get(controlnets, controlnets)
                arguments["controlnets"] = [{
                    "path": controlnet_path,
                    "conditioning_scale": controlnet_conditioning_scale,
                    "control_image_url": control_image_url
                }]
                if control_mask is not None:
                    mask_image = self.mask_to_image(control_mask)
                    mask_image_url = upload_image(mask_image)
                    if mask_image_url:
                        arguments["controlnets"][0]["mask_image_url"] = mask_image_url

        # Handle controlnet_unions
        if controlnet_unions != "None" and control_image is not None:
            control_image_url = upload_image(control_image)
            if control_image_url:
                arguments["controlnet_unions"] = [{
                    "path": controlnet_unions,
                    "controls": [{
                        "control_mode": controlnet_union_control_mode,
                        "control_image_url": control_image_url
                    }]
                }]
                if control_mask is not None:
                    mask_image = self.mask_to_image(control_mask)
                    mask_image_url = upload_image(mask_image)
                    if mask_image_url:
                        arguments["controlnet_unions"][0]["controls"][0]["mask_image_url"] = mask_image_url

        # Handle ip_adapters
        if ip_adapters != "None" and ip_adapter_image is not None:
            ip_adapter_image_url = upload_image(ip_adapter_image)
            if ip_adapter_image_url:
                ip_adapter_path = "https://huggingface.co/XLabs-AI/flux-ip-adapter/resolve/main/flux-ip-adapter.safetensors?download=true" if ip_adapters == "XLabs-AI/flux-ip-adapter" else ip_adapters
                arguments["ip_adapters"] = [{
                    "path": ip_adapter_path,
                    "image_encoder_path": "openai/clip-vit-large-patch14",
                    "image_url": ip_adapter_image_url,
                    "scale": ip_adapter_scale
                }]
                if ip_adapter_mask is not None:
                    mask_image = self.mask_to_image(ip_adapter_mask)
                    mask_image_url = upload_image(mask_image)
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
            handler = fal_client.submit("fal-ai/flux-general", arguments=arguments)
            result = handler.get()
            return self.process_result(result)
        except Exception as e:
            print(f"Error generating image with FluxGeneral: {str(e)}")
            return self.create_blank_image()

    def mask_to_image(self, mask):
        result = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
        return result

# Common methods for all classes
def process_result(self, result):
    images = []
    for img_info in result["images"]:
        img_url = img_info["url"]
        img_response = requests.get(img_url)
        img = Image.open(io.BytesIO(img_response.content))
        img_array = np.array(img).astype(np.float32) / 255.0
        images.append(img_array)

    # Stack the images along a new first dimension
    stacked_images = np.stack(images, axis=0)
    
    # Convert to PyTorch tensor
    img_tensor = torch.from_numpy(stacked_images)
    
    return (img_tensor,)

def create_blank_image(self):   
    blank_img = Image.new('RGB', (512, 512), color='black')
    img_array = np.array(blank_img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array)[None,]
    return (img_tensor,)

# Add common methods to all classes
for cls in [FluxPro, FluxDev, FluxSchnell, FluxPro11, FluxUltra, FluxGeneral, FluxLora, Recraft, Sana]:
    cls.process_result = process_result
    cls.create_blank_image = create_blank_image

# Node class mappings
NODE_CLASS_MAPPINGS = {
    "FluxPro_fal": FluxPro,
    "FluxDev_fal": FluxDev,
    "FluxSchnell_fal": FluxSchnell,
    "FluxPro11_fal": FluxPro11,
    "FluxUltra_fal": FluxUltra,
    "FluxGeneral_fal": FluxGeneral,
    "FluxLora_fal": FluxLora,
    "Recraft_fal": Recraft,
    "Sana_fal": Sana,
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxPro_fal": "Flux Pro (fal)",
    "FluxDev_fal": "Flux Dev (fal)",
    "FluxSchnell_fal": "Flux Schnell (fal)",
    "FluxPro11_fal": "Flux Pro 1.1 (fal)",
    "FluxUltra_fal": "Flux Ultra (fal)",
    "FluxGeneral_fal": "Flux General (fal)",
    "FluxLora_fal": "Flux LoRA (fal)",
    "Recraft_fal": "Recraft V3 (fal)",
    "Sana_fal": "Sana (fal)"
}

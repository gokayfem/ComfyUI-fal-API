import os
import configparser
from fal_client.client import SyncClient

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
config_path = os.path.join(parent_dir, "config.ini")

config = configparser.ConfigParser()
config.read(config_path)

try:
    if os.environ.get("FAL_KEY") is not None:
        print("FAL_KEY found in environment variables")
        fal_key = os.environ["FAL_KEY"]
    else:
        print("FAL_KEY not found in environment variables")
        fal_key = config['API']['FAL_KEY']
        print("FAL_KEY found in config.ini")
        os.environ["FAL_KEY"] = fal_key
        print("FAL_KEY set in environment variables")

    # Check if FAL key is the default placeholder
    if fal_key == "<your_fal_api_key_here>":
        print("WARNING: You are using the default FAL API key placeholder!")
        print("Please set your actual FAL API key in either:")
        print("1. The config.ini file under [API] section")
        print("2. Or as an environment variable named FAL_KEY")
        print("Get your API key from: https://fal.ai/dashboard/keys")
except KeyError:
    print("Error: FAL_KEY not found in config.ini or environment variables")

# Create the client with API key
fal_client = SyncClient(key=fal_key)

class LLMNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "model": (["google/gemini-flash-1.5-8b", "anthropic/claude-3.5-sonnet", "anthropic/claude-3-haiku", 
                           "google/gemini-pro-1.5", "google/gemini-flash-1.5", "meta-llama/llama-3.2-1b-instruct", 
                           "meta-llama/llama-3.2-3b-instruct", "meta-llama/llama-3.1-8b-instruct", 
                           "meta-llama/llama-3.1-70b-instruct", "openai/gpt-4o-mini", "openai/gpt-4o"], 
                          {"default": "google/gemini-flash-1.5-8b"}),
                "system_prompt": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_text"
    CATEGORY = "FAL/LLM"

    def generate_text(self, prompt, model, system_prompt):
        arguments = {
            "model": model,
            "prompt": prompt,
            "system_prompt": system_prompt,
        }

        try:
            handler = fal_client.submit("fal-ai/any-llm", arguments=arguments)
            result = handler.get()
            return (result["output"],)
        except Exception as e:
            print(f"Error generating text with LLM: {str(e)}")
            return ("Error: Unable to generate text.",)

# Node class mappings
NODE_CLASS_MAPPINGS = {
    "LLM_fal": LLMNode,
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "LLM_fal": "LLM (fal)",
}

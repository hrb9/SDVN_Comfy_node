import google.generativeai as genai
from nodes import NODE_CLASS_MAPPINGS as ALL_NODE_CLASS_MAPPINGS


class API_chatbot:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "chatbot": (["Gemini 1.5 Flash", "Gemini 1.5 Pro"],),
            "APIkey": ("STRING", {"default": "", "multiline": False, "tooltip": "Chatbot API"}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "The random seed used for creating the noise."}),
            "prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "Chatbot prompt"})
        }
        }

    CATEGORY = "✨ SDVN/API"

    RETURN_TYPES = ("STRING",)
    FUNCTION = "api_chatbot"

    def api_chatbot(self, chatbot, APIkey, seed, prompt):
        model_list = {
            "Gemini 1.5 Flash": "gemini-1.5-flash",
            "Gemini 1.5 Pro": "gemini-1.5-pro"
        }
        if "DPRandomGenerator" in ALL_NODE_CLASS_MAPPINGS:
            cls = ALL_NODE_CLASS_MAPPINGS["DPRandomGenerator"]
            prompt = cls().get_prompt(prompt, seed, 'No')[0]
        model_name = model_list[chatbot]
        if 'gemini' in model_name:
            genai.configure(api_key=APIkey)
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return (response.text.strip(),)


NODE_CLASS_MAPPINGS = {
    "SDVN API chatbot": API_chatbot,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SDVN API chatbot": "💬 API chatbot"
}

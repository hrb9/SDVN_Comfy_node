import google.generativeai as genai
from openai import OpenAI
from nodes import NODE_CLASS_MAPPINGS as ALL_NODE_CLASS_MAPPINGS
from torchvision.transforms import ToPILImage
import torch
from PIL import Image
import numpy as np


def tensor2pil(tensor: torch.Tensor) -> Image.Image:
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)
    if tensor.ndim == 3 and tensor.shape[-1] == 3:
        np_image = (tensor.numpy() * 255).astype(np.uint8)
    else:
        raise ValueError(
            "Tensor phải có shape [H, W, C] hoặc [1, H, W, C] với C = 3 (RGB).")
    pil_image = Image.fromarray(np_image)
    return pil_image


class API_chatbot:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "chatbot": (["Gemini | 1.5 Flash", "Gemini | 1.5 Pro", "OpenAI | GPT 4-o mini", "OpenAI | GPT 4-o", "OpenAI | GPT 3.5 Turbo", "HuggingFace | Meta Llama-3.2"],),
                "preset": (["None", "Python Function"],),
                "APIkey": ("STRING", {"default": "", "multiline": False, "tooltip": "Chatbot API"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "The random seed"}),
                "main_prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "Chatbot prompt"}),
                "sub_prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "Chatbot prompt"})
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "The for gemini model"})
            }
        }

    CATEGORY = "✨ SDVN/API"

    RETURN_TYPES = ("STRING",)
    FUNCTION = "api_chatbot"

    def api_chatbot(self, chatbot, preset, APIkey, seed, main_prompt, sub_prompt, image=None):
        model_list = {
            "Gemini | 1.5 Flash": "gemini-1.5-flash",
            "Gemini | 1.5 Pro": "gemini-1.5-pro",
            "OpenAI | GPT 4-o mini": "gpt-4o-mini",
            "OpenAI | GPT 4-o": "gpt-4o",
            "OpenAI | GPT 3.5 Turbo": "gpt-3.5-turbo-0125",
            "HuggingFace | Meta Llama-3.2": "meta-llama/Llama-3.2-3B-Instruct"
        }
        preset_prompt = {
            "None": [],
            "Python Function": [
                {"role": "user", "content": "Tôi sẽ yêu cầu một hàm def python với nhiệm vụ bất kỳ, hãy cho tôi câu trả lời là hàm python đó,viết thật đơn giản, và không cần bất kỳ hướng dẫn nào khác, các import đặt trong hàm. Đối với yêu cầu đầu vào hoặc đầu ra là hình ảnh, hãy nhớ ảnh ở dạng tensor"},
                {"role": "assistant", "content": "Đồng ý! Hãy đưa ra yêu cầu của bạn."}
            ]
        }
        if "DPRandomGenerator" in ALL_NODE_CLASS_MAPPINGS:
            cls = ALL_NODE_CLASS_MAPPINGS["DPRandomGenerator"]
            main_prompt = cls().get_prompt(main_prompt, seed, 'No')[0]
            sub_prompt = cls().get_prompt(sub_prompt, seed, 'No')[0]
        prompt = f"{main_prompt}.{sub_prompt}"
        model_name = model_list[chatbot]
        if 'Gemini' in chatbot:
            genai.configure(api_key=APIkey)
            model = genai.GenerativeModel(model_name)
            prompt += preset_prompt[preset][0]["content"] if preset != "None" else ""
            if image == None:
                response = model.generate_content(prompt)
            else:
                image = tensor2pil(image)
                response = model.generate_content([prompt, image])
            answer = response.text
        if "HuggingFace" in chatbot:
            answer = ""
            client = OpenAI(
                base_url="https://api-inference.huggingface.co/v1/", api_key=APIkey)
            messages = [
                {"role": "user", "content": prompt}
            ]
            messages = preset_prompt[preset] + messages
            stream = client.chat.completions.create(
                model=model_name,
                messages=messages,
                stream=True
            )
            for chunk in stream:
                answer += chunk.choices[0].delta.content
        if "OpenAI" in chatbot:
            answer = ""
            client = OpenAI(
                api_key=APIkey)
            messages = [
                {"role": "user", "content": prompt}
            ]
            messages = preset_prompt[preset] + messages
            stream = client.chat.completions.create(
                model=model_name,
                messages=messages,
                stream=True
            )
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    answer += chunk.choices[0].delta.content
        return (answer.strip(),)


NODE_CLASS_MAPPINGS = {
    "SDVN API chatbot": API_chatbot,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SDVN API chatbot": "💬 API chatbot"
}

import google.generativeai as genai
from openai import OpenAI
from nodes import NODE_CLASS_MAPPINGS as ALL_NODE_CLASS_MAPPINGS
from torchvision.transforms import ToPILImage
import torch
import re
from PIL import Image
import numpy as np
import io
import base64


class AnyType(str):
    """A special class that is always equal in not equal comparisons. Credit to pythongosssss"""

    def __eq__(self, _) -> bool:
        return True

    def __ne__(self, __value: object) -> bool:
        return False


any = AnyType("*")


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


def encode_image(image_tensor):
    image = tensor2pil(image_tensor)
    with io.BytesIO() as image_buffer:
        image.save(image_buffer, format="PNG")
        image_buffer.seek(0)
        encoded_image = base64.b64encode(image_buffer.read()).decode('utf-8')

    return encoded_image


class run_python_code:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input": (any,),
                "function": ("STRING", {"default": """
def function(input):
    output = input.strip()                               
    return output                       
                """, "multiline": True, })
            }
        }

    CATEGORY = "✨ SDVN/Dev"

    RETURN_TYPES = (any,)
    RETURN_NAMES = ("output",)
    FUNCTION = "python_function"

    def python_function(self, input, function):
        pattern = r"def.*?return[^\n]*"
        match = re.search(pattern, function, re.DOTALL)
        function = match.group(0) if match else ""
        pattern = r"def\s+(\w+)\s*\("
        matches = re.findall(pattern, function)[0]
        local_context = {}
        exec(function, {}, local_context)
        function = local_context[matches]
        output = function(input)
        return (output,)


class API_chatbot:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "chatbot": (["Gemini | 1.5 Flash", "Gemini | 1.5 Pro", "OpenAI | GPT 4-o mini", "OpenAI | GPT 4-o", "OpenAI | GPT 3.5 Turbo", "HuggingFace | Meta Llama-3.2"],),
                "preset": (["None", "Python Function | vi"],),
                "APIkey": ("STRING", {"default": "", "multiline": False, "tooltip": """
Get API Gemini: https://aistudio.google.com/app/apikey
Get API OpenAI: https://platform.openai.com/settings/organization/api-keys
Get API HugggingFace: https://huggingface.co/settings/tokens
                                      """}),
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
            "HuggingFace | Meta Llama-3.2": "meta-llama/Llama-3.2-3B-Instruct",
            "HuggingFace | Qwen 2.5-72B": "Qwen/Qwen2.5-72B-Instruct"
        }
        preset_prompt = {
            "None": [],
            "Python Function | vi": [
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
            if image != None:
                image = encode_image(image)
                prompt = [{"type": "text", "text": prompt, }, {
                    "type": "image_url", "image_url": {"url":  f"data:image/jpeg;base64,{image}"}, },]
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
            if image != None:
                answer = answer.split('return True')[-1]
        return (answer.strip(),)


class API_DALLE:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "OpenAI_API": ("STRING", {"default": "", "multiline": False, "tooltip": "Get API: https://platform.openai.com/settings/organization/api-keys"}),
                "width": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 64, "display": "slider", "lazy": True}),
                "height": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 64, "display": "slider", "lazy": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "The random seed"}),
                "prompt": ("STRING", {"default": "", "multiline": True, "placeholder": "Image size ['256x256', '512x512', '1024x1024', '1024x1792', '1792x1024']", "tooltip": "Image size ['256x256', '512x512', '1024x1024', '1024x1792', '1792x1024']"}),
            }
        }

    CATEGORY = "✨ SDVN/API"

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "api_dalle"

    def api_dalle(self, OpenAI_API, width, height, seed, prompt):
        client = OpenAI(
            api_key=OpenAI_API
        )
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size=f"{width}x{height}",
            quality="standard",
            n=1,
        )
        cls = ALL_NODE_CLASS_MAPPINGS["SDVN Load Image Url"]
        image_url = response.data[0].url
        print(image_url)
        image = cls().load_image_url(image_url)[0]
        return (image,)


NODE_CLASS_MAPPINGS = {
    "SDVN Run Python Code": run_python_code,
    "SDVN API chatbot": API_chatbot,
    "SDVN DALL-E Generate Image": API_DALLE,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SDVN Run Python Code": "👨🏻‍💻 Run Python Code",
    "SDVN API chatbot": "💬 API chatbot",
    "SDVN DALL-E Generate Image": "🎨 DALL-E Generate Image",
}

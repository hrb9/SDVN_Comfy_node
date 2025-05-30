from nodes import NODE_CLASS_MAPPINGS as ALL_NODE
from google import genai
from openai import OpenAI
import io, base64, torch, numpy as np, re, os, json
from googletrans import LANGUAGES
from PIL import Image, ImageOps
from gradio_client import Client, handle_file
from google.genai import types
from io import BytesIO

def i2tensor(i) -> torch.Tensor:
    i = ImageOps.exif_transpose(i)
    image = i.convert("RGB")
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image)[None,]
    return image 

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

def pil_to_bytesio(image, filename="image.png"):
    image = tensor2pil(image)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    buffer.name = filename
    return buffer

def mask_bytesio(mask, filename="mask_alpha.png"):
    mask_img = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
    mask_img = tensor2pil(mask_img)
    mask = ImageOps.invert(mask_img.convert("L"))
    mask_rgba = mask.convert("RGBA")
    mask_rgba.putalpha(mask)
    buffer = BytesIO()
    mask_rgba.save(buffer, format="PNG")
    buffer.seek(0)
    buffer.name = filename  
    return buffer

def lang_list():
    lang_list = ["None"]
    for i in LANGUAGES.items():
        lang_list += [i[1]]
    return lang_list

class AnyType(str):
    """A special class that is always equal in not equal comparisons. Credit to pythongosssss"""

    def __eq__(self, _) -> bool:
        return True

    def __ne__(self, __value: object) -> bool:
        return False


any = AnyType("*")

def api_check():
    api_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"API_key.json")
    if os.path.exists(api_file):
        with open(api_file, 'r', encoding='utf-8') as f:
            api_list = json.load(f)
        return api_list
    else:
        return None

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
                "function": ("STRING", {"default": """
def function(input):
    output = input.strip()                               
    return output                       
                """, "multiline": True, })
            },
            "optional": {
                "input": (any,),
                "input2": (any,),
                "input3": (any,),
            }
        }

    CATEGORY = "📂 SDVN/👨🏻‍💻 Dev"
    OUTPUT_IS_LIST = (True,)
    RETURN_TYPES = (any,)
    RETURN_NAMES = ("output",)
    FUNCTION = "python_function"

    def python_function(self, function, input=None, input2=None, input3=None):
        check_list = [input, input2, input3]
        b = 3
        new_list = []
        for i in check_list:
            if i == None:
                b -= 1
            else:
                new_list += [i]

        pattern = r"def.*?return[^\n]*"
        match = re.search(pattern, function, re.DOTALL)
        function = match.group(0) if match else ""
        pattern = r"def\s+(\w+)\s*\("
        matches = re.findall(pattern, function)[0]
        local_context = {}
        exec(function, {}, local_context)
        function = local_context[matches]
        if b == 3:
            output = function(new_list[0], new_list[1], new_list[2])
        elif b == 2:
            output = function(new_list[0], new_list[1])
        elif b == 1:
            output = function(new_list[0])
        elif b == 0:
            output = function()
        if not isinstance(output, list):
            output = [output]
        return ([*output],)

model_list = {
    "Gemini | 2.0 Flash (Img support)" : "gemini-2.0-flash",
    "Gemini | 2.0 Flash Lite (Img support)": "gemini-2.0-flash-lite",
    "Gemini | 2.5 Pro Preview (Img support)": "gemini-2.5-pro-preview-05-06",
    "Gemini | 2.5 Flash Preview (Img support)": "gemini-2.5-flash-preview-04-17",
    "OpenAI | GPT o4-mini (Img support)": "o4-mini",
    "OpenAI | GPT 4-o mini (Img support)": "gpt-4o-mini",
    "OpenAI | GPT o3 (Img support)": "o3",
    "OpenAI | GPT o1 (Img support)": "o1",
    "OpenAI | GPT o1-pro (Img support)": "o1-pro",
    "OpenAI | GPT 4-o (Img support)": "gpt-4o",
    "OpenAI | GPT 4.1 (Img support)": "gpt-4.1",
    "OpenAI | GPT o1-mini": "o1-mini",
    "OpenAI | GPT o3-mini": "o3-mini",
    "OpenAI | GPT 3.5 Turbo": "gpt-3.5-turbo-0125",
    "Deepseek | R1": "deepseek-chat",
    "HuggingFace | DeepSeek R1 32B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "HuggingFace | DeepSeek R1 1.5B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "HuggingFace | Meta Llama-3.2": "meta-llama/Llama-3.2-3B-Instruct",
    "HuggingFace | Qwen 2.5-72B": "Qwen/Qwen2.5-72B-Instruct"
}

preset_prompt = {
    "None": [],
    "Python Function": [
        {"role": "user", "content": "I will ask for a def python function with any task, give me the answer that python function, write simply, and don't need any other instructions, the imports are placed in the function. For input or output requirements of an image, remember the image is in tensor form"},
        {"role": "assistant", "content": "Agree! Please submit your request."}
    ],
    "Prompt Generate": [
        {"role": "user", "content": "Send the description on demand, limit 100 words, only send me the answer" }
    ]
}

class API_chatbot:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "chatbot": (list(model_list),),
                "preset": (list(preset_prompt),),
                "APIkey": ("STRING", {"default": "", "multiline": False, "tooltip": """
Get API Gemini: https://aistudio.google.com/app/apikey
Get API OpenAI: https://platform.openai.com/settings/organization/api-keys
Get API HugggingFace: https://huggingface.co/settings/tokens
                                      """}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "The random seed"}),
                "main_prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "Chatbot prompt"}),
                "sub_prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "Chatbot prompt"}),
                "translate": (lang_list(),),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "The for gemini model"})
            }
        }

    CATEGORY = "📂 SDVN/💬 API"

    RETURN_TYPES = ("STRING",)
    FUNCTION = "api_chatbot"

    def api_chatbot(self, chatbot, preset, APIkey, seed, main_prompt, sub_prompt, translate, image=None):
        if APIkey == "":
            api_list = api_check()
            if api_check() != None:
                if "Gemini" in chatbot:
                    APIkey =  api_list["Gemini"]
                if "HuggingFace" in chatbot:
                    APIkey =  api_list["HuggingFace"]
                if "OpenAI" in chatbot:
                    APIkey =  api_list["OpenAI"]
                if "Deepseek" in chatbot:
                    APIkey =  api_list["Deepseek"]

        if "DPRandomGenerator" in ALL_NODE:
            cls = ALL_NODE["DPRandomGenerator"]
            main_prompt = cls().get_prompt(main_prompt, seed, 'No')[0]
            sub_prompt = cls().get_prompt(sub_prompt, seed, 'No')[0]
        main_prompt = ALL_NODE["SDVN Translate"]().ggtranslate(main_prompt,translate)[0]
        sub_prompt = ALL_NODE["SDVN Translate"]().ggtranslate(sub_prompt,translate)[0]
        prompt = f"{main_prompt}.{sub_prompt}"
        model_name = model_list[chatbot]
        if 'Gemini' in chatbot:
            prompt += preset_prompt[preset][0]["content"] if preset != "None" else ""
            client = genai.Client(api_key=APIkey)
            if image == None:
                response = client.models.generate_content(
                    model=model_name,
                    contents=[prompt])
            else:
                image = tensor2pil(image)
                response = client.models.generate_content(
                    model=model_name,
                    contents=[prompt, image])
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
                temperature=0.5,
                max_tokens=2048,
                top_p=0.7,
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
        if "Deepseek" in chatbot:
            client = OpenAI(api_key=APIkey, base_url="https://api.deepseek.com")
            response = client.chat.completions.create(
                model = model_name,
                messages=[
                    {"role": "system", "content": {preset_prompt[preset][0]['content']} if preset != "None" else "You are a helpful assistant"},
                    {"role": "user", "content": prompt},
                ],
                stream=False
                )
            answer = response.choices[0].message.content
        return (answer.strip(),)


class API_DALLE:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "OpenAI_API": ("STRING", {"default": "", "multiline": False, "tooltip": "Get API: https://platform.openai.com/settings/organization/api-keys"}),
                "size": (['1024x1024', '1024x1792', '1792x1024'],{"default": '1024x1024'}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "The random seed"}),
                "prompt": ("STRING", {"default": "", "multiline": True, "placeholder": "Get API: https://platform.openai.com/settings/organization/api-keys"}),
                "quality": (["standard","hd"], {"default": "standard",}),
                "translate": (lang_list(),),
            }
        }

    CATEGORY = "📂 SDVN/💬 API"

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "api_dalle"

    def api_dalle(self, OpenAI_API, size, seed, prompt, quality, translate):
        if OpenAI_API == "":
            api_list = api_check()
            OpenAI_API =  api_list["OpenAI"]
        if "DPRandomGenerator" in ALL_NODE:
            cls = ALL_NODE["DPRandomGenerator"]
            prompt = cls().get_prompt(prompt, seed, 'No')[0]
        prompt = ALL_NODE["SDVN Translate"]().ggtranslate(prompt,translate)[0]

        client = OpenAI(
            api_key=OpenAI_API
        )
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size=size,
            quality=quality,
        )
        image_url = response.data[0].url
        image = ALL_NODE["SDVN Load Image Url"]().load_image_url(image_url)["result"][0]
        return (image,)

class API_DALLE_2:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "OpenAI_API": ("STRING", {"default": "", "multiline": False, "tooltip": "Get API: https://platform.openai.com/settings/organization/api-keys"}),
                "size": (['auto','256x256', '512x512', '1024x1024'],{"default": "1024x1024"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "The random seed"}),
                "prompt": ("STRING", {"default": "", "multiline": True, "placeholder": "Get API: https://platform.openai.com/settings/organization/api-keys"}),
                "n": ("INT", {"default": 1, "min": 1, "max": 4}),
                "translate": (lang_list(),),
            },
            "optional": {
                "image": ("IMAGE",),
                "mask": ("MASK",)
            }
        }

    CATEGORY = "📂 SDVN/💬 API"

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "api_dalle"
    OUTPUT_IS_LIST = (True,)

    def api_dalle(self, OpenAI_API, size, seed, prompt, n, translate, image = None, mask = None):
        if OpenAI_API == "":
            api_list = api_check()
            OpenAI_API =  api_list["OpenAI"]
        if "DPRandomGenerator" in ALL_NODE:
            cls = ALL_NODE["DPRandomGenerator"]
            prompt = cls().get_prompt(prompt, seed, 'No')[0]
        prompt = ALL_NODE["SDVN Translate"]().ggtranslate(prompt,translate)[0]

        client = OpenAI(
            api_key=OpenAI_API
        )
        if image != None and mask != None:
            response = client.images.edit(
                model="dall-e-2",
                prompt=prompt,
                image = pil_to_bytesio(image),
                n = n,
                mask = mask_bytesio(mask),
                size=size,
            )
        else:
            response = client.images.generate(
                model="dall-e-2",
                prompt=prompt,
                size=size,
                n=n,
            )
        images = []
        for i in range(n):
            image_url = response.data[i].url
            image = ALL_NODE["SDVN Load Image Url"]().load_image_url(image_url)["result"][0]
            images.append(image)
        return (images,)
    
class API_GPT_image:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "OpenAI_API": ("STRING", {"default": "", "multiline": False, "tooltip": "Get API: https://platform.openai.com/settings/organization/api-keys"}),
                "size": (["auto",'1024x1024', '1536x1024', '1024x1536'],{"default": "auto"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "The random seed"}),
                "prompt": ("STRING", {"default": "", "multiline": True, "placeholder": "Get API: https://platform.openai.com/settings/organization/api-keys"}),
                "quality": (["auto","low","medium","high"], {"default": "medium",}),
                "background": (["opaque","transparent"], {"default": "opaque",}),
                "n": ("INT", {"default": 1, "min": 1, "max": 4}),
                "translate": (lang_list(),),
            },
            "optional": {
                "image": ("IMAGE",),
                "mask": ("MASK",)
            }
        }

    CATEGORY = "📂 SDVN/💬 API"

    RETURN_TYPES = ("IMAGE",)
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "API_GPT_image"

    def API_GPT_image(self, OpenAI_API, size, seed, prompt, quality, background, n, translate, image = None, mask = None):
        OpenAI_API = OpenAI_API[0]
        size = size[0]
        seed = seed[0]
        prompt = prompt[0]
        quality = quality[0]
        background = background[0]
        n = n[0]

        translate = translate[0]
        if OpenAI_API == "":
            api_list = api_check()
            OpenAI_API =  api_list["OpenAI"]
        if "DPRandomGenerator" in ALL_NODE:
            prompt = ALL_NODE["DPRandomGenerator"]().get_prompt(prompt, seed, 'No')[0]
        prompt = ALL_NODE["SDVN Translate"]().ggtranslate(prompt,translate)[0]
            
        client = OpenAI(
            api_key=OpenAI_API
        )
        if image == None:
            result = client.images.generate(
                model="gpt-image-1",
                prompt=prompt,
                size = size,
                quality = quality,
                background = background,
                moderation = "low",
                n = n
            )
        elif mask == None:
            result = client.images.edit(
                model="gpt-image-1",
                prompt=prompt,
                size = size,
                quality = quality,
                image = [pil_to_bytesio(img) for img in image],
                n = n,
            )
        else:
            result = client.images.edit(
                model="gpt-image-1",
                prompt=prompt,
                size = size,
                quality = quality,
                image = pil_to_bytesio(image[0]),
                n = n,
                mask = mask_bytesio(mask[0]),
            )
        images = []
        for i in range(n):
            image_base64 = result.data[i].b64_json
            image_bytes = base64.b64decode(image_base64)
            image_pil = Image.open(BytesIO(image_bytes))
            image_ten = i2tensor(image_pil)
            images.append(image_ten)
        return (images,)
    
class Gemini_Flash2_Image:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "Gemini_API": ("STRING", {"default": "", "multiline": False, "tooltip": "Get API: https://aistudio.google.com/apikey"}),
                "max_size_input": ("INT", {"default":0,"min":0,"max":2048,"step":64}),
                "prompt": ("STRING", {"default": "", "multiline": True, "placeholder": "Prompt"}),
                "translate": (lang_list(),{"default":"english"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "The random seed"}),
            },
            "optional": {
                "image": ("IMAGE",)
            }
        }
    INPUT_IS_LIST = True
    CATEGORY = "📂 SDVN/💬 API"

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "api_imagen"

    def api_imagen(self, Gemini_API, max_size_input, seed, prompt, translate, image = None):
        Gemini_API, max_size_input, seed, prompt, translate = [Gemini_API[0], max_size_input[0], seed[0], prompt[0], translate[0]]
  
        if Gemini_API == "":
            api_list = api_check()
            Gemini_API =  api_list["Gemini"]
        if "DPRandomGenerator" in ALL_NODE:
            cls = ALL_NODE["DPRandomGenerator"]
            prompt = cls().get_prompt(prompt, seed, 'No')[0]
        prompt = ALL_NODE["SDVN Translate"]().ggtranslate(prompt,translate)[0]
        client = genai.Client(api_key=Gemini_API)
        if image != None:
            if max_size_input != 0:
                list_img = [ALL_NODE["SDVN Upscale Image"]().upscale("Maxsize", max_size_input, max_size_input, 1, "None", i)[0] for i in image]
            list_img = [tensor2pil(i) for i in image]
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp-image-generation",
            contents=[prompt, *list_img] if image != None else prompt,
            config=types.GenerateContentConfig(
            response_modalities=['Text', 'Image']
            )
        )
        for part in response.candidates[0].content.parts:
            if part.text is not None:
                print(part.text)
            elif part.inline_data is not None:
                image = Image.open(BytesIO(part.inline_data.data))              
        image = i2tensor(image)
        return (image,)

class API_Imagen:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "Gemini_API": ("STRING", {"default": "", "multiline": False, "tooltip": "Get API: https://aistudio.google.com/apikey"}),
                "aspect_ratio": (['1:1', '3:4', '4:3', '9:16', '16:9'],{"default": "1:1"}),
                "person_gen": ("BOOLEAN", {"default": True},),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "The random seed"}),
                "prompt": ("STRING", {"default": "", "multiline": True, "placeholder": "Prompt"}),
                "translate": (lang_list(),),
            }
        }

    CATEGORY = "📂 SDVN/💬 API"

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "api_imagen"

    def api_imagen(self, Gemini_API, aspect_ratio, person_gen, seed, prompt, translate):
        if Gemini_API == "":
            api_list = api_check()
            Gemini_API =  api_list["Gemini"]
        if "DPRandomGenerator" in ALL_NODE:
            cls = ALL_NODE["DPRandomGenerator"]
            prompt = cls().get_prompt(prompt, seed, 'No')[0]
        prompt = ALL_NODE["SDVN Translate"]().ggtranslate(prompt,translate)[0]
        client = genai.Client(api_key=Gemini_API)
        response = client.models.generate_images(
            model='imagen-4.0-generate-preview-05-20',
            prompt=prompt,
            config=types.GenerateImagesConfig(
                number_of_images= 1,
                aspect_ratio = aspect_ratio,
                output_mime_type = 'image/jpeg',
                person_generation = 'ALLOW_ADULT' if person_gen else 'DONT_ALLOW',
            ),
        )
        for generated_image in response.generated_images:
            image = Image.open(BytesIO(generated_image.image.image_bytes))
        image = i2tensor(image)
        return (image,)
        
class ic_light_v2:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (["v2", "v2_vary"],{"default":"v2"}),
                "bg_source": (['None', 'Left Light', 'Right Light', 'Top Light', 'Bottom Light'],{"default":"None"}),
                "prompt": ("STRING",{"default":"","multiline": True}),
                "translate": (lang_list(),),
                "n_prompt": ("STRING",{"default":"","multiline": False}),
                "hf_token": ("STRING",{"default":"","multiline": False}),
                "image_size": ("INT", {"default":1024,"min":512,"max":2048}),
                "steps": ("INT", {"default":25,"min":1,"max":50}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, }),
            }
        }

    CATEGORY = "📂 SDVN/💬 API"
    RETURN_TYPES = ("IMAGE","IMAGE")
    RETURN_NAMES = ("image","grey_img")
    FUNCTION = "ic_light_v2"

    def ic_light_v2(s, image, mode, bg_source, prompt, translate, n_prompt, hf_token, image_size, steps, seed):

        if "DPRandomGenerator" in ALL_NODE:
            cls = ALL_NODE["DPRandomGenerator"]
            prompt = cls().get_prompt(prompt, seed, 'No')[0]
            n_prompt = cls().get_prompt(n_prompt, seed, 'No')[0]
        prompt = ALL_NODE["SDVN Translate"]().ggtranslate(prompt,translate)[0]
        n_prompt = ALL_NODE["SDVN Translate"]().ggtranslate(n_prompt,translate)[0]
        if hf_token == "":
            api_list = api_check()
            if api_check() != None:
                hf_token =  api_list["HuggingFace"]

        samples = image.movedim(-1, 1)
        w = samples.shape[3]
        h = samples.shape[2]
        width = image_size
        height = image_size
        if width/height < w/h:
            height = round(h * width / w)
        else:
            width = round(w * height / h)
        image = tensor2pil(image)

        input_path = "/tmp/ic_light.jpg"
        space_path = "lllyasviel/iclight-v2" if mode == "v2" else "lllyasviel/iclight-v2-vary"

        if not os.path.isdir("/tmp"):
            os.mkdir("/tmp")
        image.save(input_path, format="JPEG")
        if hf_token == "":
            client = Client(space_path)
        else:
            client = Client(space_path, hf_token = hf_token)
        if mode == "v2":
            result = client.predict(
                    input_fg = handle_file(input_path),
                    bg_source = bg_source,
                    prompt = prompt,
                    image_width = width,
                    image_height = height,
                    num_samples = 1,
                    seed = seed,
                    steps = steps,
                    n_prompt = n_prompt,
                    cfg=1,
                    gs=5,
                    rs=1,
                    init_denoise=0.999,
                    api_name="/process"
            )
        else:
            result = client.predict(
                    input_fg = handle_file(input_path),
                    bg_source = bg_source,
                    prompt = prompt,
                    image_width = width,
                    image_height = height,
                    num_samples = 1,
                    seed = seed,
                    steps = steps,
                    n_prompt = n_prompt,
                    cfg=2,
                    gs=5,
                    enable_hr_fix=True,
                    hr_downscale=0.5,
                    lowres_denoise=0.8,
                    highres_denoise=0.99,
                    api_name="/process"
            )

        img_path = result[0][0]['image']
        img_grey_path = result[1]
        img = ALL_NODE["SDVN Load Image Url"]().load_image_url(img_path)["result"][0]
        img_grey = ALL_NODE["SDVN Load Image Url"]().load_image_url(img_grey_path)["result"][0]
        return (img,img_grey,)

class joy_caption:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "caption_type": (["Descriptive", "Descriptive (Informal)", "Training Prompt", "MidJourney", "Booru tag list", "Booru-like tag list", "Art Critic", "Product Listing", "Social Media Post"],),
                "caption_length": (["any", "very short", "short", "medium-length", "long", "very long"] + [str(i) for i in range(20, 261, 10)],),
                "extra_options": ([
                    "None",
					"If there is a person/character in the image you must refer to them as {name}.",
					"Do NOT include information about people/characters that cannot be changed (like ethnicity, gender, etc), but do still include changeable attributes (like hair style).",
					"Include information about lighting.",
					"Include information about camera angle.",
					"Include information about whether there is a watermark or not.",
					"Include information about whether there are JPEG artifacts or not.",
					"If it is a photo you MUST include information about what camera was likely used and details such as aperture, shutter speed, ISO, etc.",
					"Do NOT include anything sexual; keep it PG.",
					"Do NOT mention the image's resolution.",
					"You MUST include information about the subjective aesthetic quality of the image from low to very high.",
					"Include information on the image's composition style, such as leading lines, rule of thirds, or symmetry.",
					"Do NOT mention any text that is in the image.",
					"Specify the depth of field and whether the background is in focus or blurred.",
					"If applicable, mention the likely use of artificial or natural lighting sources.",
					"Do NOT use any ambiguous language.",
					"Include whether the image is sfw, suggestive, or nsfw.",
					"ONLY describe the most important elements of the image."
				],),
                "name_input": ("STRING",{"default":"","multiline": False}),
                "custom_prompt": ("STRING",{"default":"","multiline": True}),
                "translate": (lang_list(),),
                "hf_token": ("STRING",{"default":"","multiline": False}),
            }
        }

    CATEGORY = "📂 SDVN/💬 API"
    RETURN_TYPES = ("STRING","STRING",)
    RETURN_NAMES = ("prompt","caption",)
    FUNCTION = "joy_caption"

    def joy_caption(s, image, caption_type, caption_length, extra_options, name_input, custom_prompt, translate, hf_token):
        if custom_prompt != "":
            custom_prompt = ALL_NODE["SDVN Translate"]().ggtranslate(custom_prompt,translate)[0]
        if hf_token == "":
            api_list = api_check()
            if api_check() != None:
                hf_token =  api_list["HuggingFace"]
        extra_options = "" if extra_options == "None" else extra_options

        image = tensor2pil(image)
        input_path = "/tmp/joy_caption.jpg"
        if not os.path.isdir("/tmp"):
            os.mkdir("/tmp")
        image.save(input_path, format="JPEG")

        space_path = "fancyfeast/joy-caption-alpha-two"
        if hf_token == "":
            client = Client(space_path)
        else:
            client = Client(space_path, hf_token = hf_token)
        result = client.predict(
                input_image = handle_file(input_path),
                caption_type = caption_type,
                caption_length = caption_length,
                extra_options = [extra_options],
                name_input = name_input,
                custom_prompt = custom_prompt,
                api_name="/stream_chat"
        )
        return result
    
NODE_CLASS_MAPPINGS = {
    "SDVN Run Python Code": run_python_code,
    "SDVN API chatbot": API_chatbot,
    "SDVN DALL-E Generate Image": API_DALLE,
    "SDVN Dall-E Generate Image 2": API_DALLE_2,
    "SDVN GPT Image": API_GPT_image, 
    "SDVN IC-Light v2": ic_light_v2,
    "SDVN Joy Caption": joy_caption,
    "SDVN Google Imagen": API_Imagen,
    "SDVN Gemini Flash 2 Image": Gemini_Flash2_Image,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SDVN Run Python Code": "👨🏻‍💻 Run Python Code",
    "SDVN API chatbot": "💬 Chatbot",
    "SDVN DALL-E Generate Image": "🎨 DALL-E 3",
    "SDVN IC-Light v2": "✨ IC-Light v2",
    "SDVN Joy Caption": "✨ Joy Caption",
    "SDVN Google Imagen": "🎨 Google Imagen",
    "SDVN Gemini Flash 2 Image": "🎨 Gemini Flash 2 Image",
    "SDVN GPT Image": "🎨 GPT Image",
    "SDVN Dall-E Generate Image 2": "🎨 DALL-E 2",
}

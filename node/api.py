from nodes import NODE_CLASS_MAPPINGS as ALL_NODE
import google.generativeai as genai
from openai import OpenAI
import io, base64, torch, numpy as np, re, os, json
from googletrans import LANGUAGES
from PIL import Image, ImageOps
from gradio_client import Client, handle_file

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
    "Gemini | 1.5 Flash": "gemini-1.5-flash",
    "Gemini | 1.5 Pro": "gemini-1.5-pro",
    "OpenAI | GPT 4-o mini": "gpt-4o-mini",
    "OpenAI | GPT 4-o": "gpt-4o",
    "OpenAI | GPT 3.5 Turbo": "gpt-3.5-turbo-0125",
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

def dic2list(dic):
    l = []
    for i in dic:
        l += [i]
    return l

class API_chatbot:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "chatbot": (dic2list(model_list),),
                "preset": (dic2list(preset_prompt),),
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

        if "DPRandomGenerator" in ALL_NODE:
            cls = ALL_NODE["DPRandomGenerator"]
            main_prompt = cls().get_prompt(main_prompt, seed, 'No')[0]
            sub_prompt = cls().get_prompt(sub_prompt, seed, 'No')[0]
        main_prompt = ALL_NODE["SDVN Translate"]().ggtranslate(main_prompt,translate)[0]
        sub_prompt = ALL_NODE["SDVN Translate"]().ggtranslate(sub_prompt,translate)[0]
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
        return (answer.strip(),)


class API_DALLE:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "OpenAI_API": ("STRING", {"default": "", "multiline": False, "tooltip": "Get API: https://platform.openai.com/settings/organization/api-keys"}),
                "size": (['1024x1024', '1024x1792', '1792x1024'],{"default": "1024x1024"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "The random seed"}),
                "prompt": ("STRING", {"default": "", "multiline": True, "placeholder": "Get API: https://platform.openai.com/settings/organization/api-keys"}),
                "translate": (lang_list(),),
            }
        }

    CATEGORY = "📂 SDVN/💬 API"

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "api_dalle"

    def api_dalle(self, OpenAI_API, size, seed, prompt,translate):
        if OpenAI_API == "":
            api_list = api_check()
            OpenAI_API =  api_list["OpenAI"]
        if "DPRandomGenerator" in ALL_NODE:
            cls = ALL_NODE["DPRandomGenerator"]
            prompt = cls().get_prompt(prompt, seed, 'No')[0]
        prompt = ALL_NODE["SDVN Translate"]().ggtranslate(prompt,translate)[0]

        width, height = size.split("x")
        client = OpenAI(
            api_key=OpenAI_API
        )
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size=f"{int(width)}x{int(height)}",
            quality="standard",
            n=1,
        )
        cls = ALL_NODE["SDVN Load Image Url"]
        image_url = response.data[0].url
        print(image_url)
        image = cls().load_image_url(image_url)["result"][0]
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
    "SDVN IC-Light v2": ic_light_v2,
    "SDVN Joy Caption": joy_caption, 
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SDVN Run Python Code": "👨🏻‍💻 Run Python Code",
    "SDVN API chatbot": "💬 Chatbot",
    "SDVN DALL-E Generate Image": "🎨 DALL-E",
    "SDVN IC-Light v2": "✨ IC-Light v2",
    "SDVN Joy Caption": "✨ Joy Caption", 
}

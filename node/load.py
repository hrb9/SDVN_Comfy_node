
import requests, math, json, os, re, sys, torch, hashlib, subprocess, numpy as np, csv
import folder_paths, comfy.sd, comfy.utils
from PIL import Image, ImageOps
from googletrans import LANGUAGES
from nodes import NODE_CLASS_MAPPINGS as ALL_NODE
from comfy.cldm.control_types import UNION_CONTROLNET_TYPES
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))
    
def style_list():
    file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),"styles.csv")
    with open(file_path, mode="r", encoding="utf-8") as file:
        reader = csv.reader(file)
        data_list = [row for row in reader]
    my_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),"my_styles.csv")
    if os.path.exists(my_path):
            with open(my_path, mode="r", encoding="utf-8") as file:
                reader = csv.reader(file)
                my_data_list = [row for row in reader]
            data_list = my_data_list + data_list
    card_list = []
    for i in data_list:
        card_list += [i[0]]
    return (card_list,data_list)
    
def lang_list():
    lang_list = ["None"]
    for i in LANGUAGES.items():
        lang_list += [i[1]]
    return lang_list

def none2list(folderlist):
    list = ["None"]
    list += folderlist
    return list


def i2tensor(i) -> torch.Tensor:
    i = ImageOps.exif_transpose(i)
    image = i.convert("RGB")
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image)[None,]
    return image


def run_gallery_dl(url):
    command = ['gallery-dl', '-G', url]
    try:
        result = subprocess.run(command, check=True,
                                text=True, capture_output=True)
        result = result.stdout.strip()
        if 'http' not in result:
            result = 'https://raw.githubusercontent.com/StableDiffusionVN/SDVN_Comfy_node/refs/heads/main/preview/eror.jpg'
        if '\n' in result:
            result = result.split('\n')[0]
    except:
        print('Cannot find image link')
        result = 'https://raw.githubusercontent.com/StableDiffusionVN/SDVN_Comfy_node/refs/heads/main/preview/eror.jpg'
    return result

def civit_downlink(link):
    command = ['wget', link, '-O', 'model.html']
    subprocess.run(command, check=True, text=True, capture_output=True)
    try:
        # Mở tệp và đọc nội dung
        with open('model.html', 'r', encoding='utf-8') as file:
            html_content = file.read()
        pattern = r'"modelVersionId":(\d+),'
        model_id = re.findall(pattern, html_content)
        if model_id:
            api_link = f'https://civitai.com/api/download/models/{model_id[0]}'
            print(f'Download model id_link: {api_link}')
            return api_link
        else:
            return "Không tìm thấy đoạn nội dung phù hợp."
    except requests.RequestException as e:
        return f"Lỗi khi tải trang: {e}"


def check_link(link):
    if 'huggingface.co' in link:
        if 'blob' in link:
            link = link.replace('blob', 'resolve')
            return link
        else:
            return link
    if 'civitai.com' in link:
        if 'civitai.com/models' in link:
            return civit_downlink(link)
        else:
            return link


def token(link):
    if "civitai" in link:
        token = f'?token=8c7337ac0c39fe4133ae19a3d65b806f'
    else:
        token = ""
    return token

def download_model(url, name, type):
    url = url.replace("&", "\&").split("?")[0]
    url = check_link(url)
    checkpoint_path = os.path.join(folder_paths.models_dir, type)
    command = ['aria2c', '-c', '-x', '16', '-s', '16',
               '-k', '1M', f'{url}{token(url)}', '-d', checkpoint_path, '-o', name]
    subprocess.run(command, check=True, text=True, capture_output=True)


class LoadImage:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        exclude_folders = ["clipspace", "folder_to_exclude2"]
        file_list = []

        for root, dirs, files in os.walk(input_dir):
            # Exclude specific folders
            dirs[:] = [d for d in dirs if d not in exclude_folders]

            for file in files:
                file_path = os.path.relpath(
                    os.path.join(root, file), start=input_dir)
                # so the filename is processed correctly in widgets.js
                file_path = file_path.replace("\\", "/")
                img_type = file_path.split('.')[-1].lower()
                if img_type in ["jpeg", "webp", "png", "jpg", "bmp"]:
                    file_list.append(file_path)

        return {
            "required": {
                "Load_url": ("BOOLEAN", {"default": True},),
                "Url": ("STRING", {"default": "", "multiline": False},),
                "image": (sorted(file_list), {"image_upload": True})
            }
        }

    CATEGORY = "📂 SDVN"

    RETURN_TYPES = ("IMAGE", "MASK", "STRING",)
    RETURN_NAMES = ("image","mask", "img_path",)
    FUNCTION = "load_image"

    def load_image(self, Url, Load_url, image):
        image_path = folder_paths.get_annotated_filepath(image)
        if Url != '' and Load_url:
            if 'http' in Url:
                Url = run_gallery_dl(Url)
                i = Image.open(requests.get(Url, stream=True).raw)
                image_path = ''
            else:
                image_path = Url
                i = Image.open(Url)
        else:
            print(image_path)
            i = Image.open(image_path)
        ii = ImageOps.exif_transpose(i)
        if 'A' in ii.getbands():
            mask = np.array(ii.getchannel('A')).astype(np.float32) / 255.0
            mask = 1. - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
        return (i2tensor(i), mask.unsqueeze(0), image_path)

    @classmethod
    def IS_CHANGED(self, Url, Load_url, image=None):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(self, Url, Load_url, image=None):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)

        return True

class LoadImageFolder:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "folder_path": ("STRING", {"default": "", "multiline": False},),
                "index": ("INT",{"default":0,"min":0}),
                "auto_index": ("BOOLEAN", {"default": False, "label_on": "loop", "label_off": "off"},),
            }
        }

    CATEGORY = "📂 SDVN"

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("image", "list_img",)
    FUNCTION = "load_image"

    def list_img_by_path(s,file_path):
        list_img = []

        if os.path.isfile(file_path):
            file_path = os.path.dirname(file_path)

        for file in os.listdir(file_path):
            file_full_path = os.path.join(file_path, file)
            if os.path.isdir(file_full_path):
                list_img.extend(s.list_img_by_path(file_full_path))
            elif os.path.isfile(file_full_path):
                type_name = file.split('.')[-1].lower()
                if type_name in ["jpeg", "webp", "png", "jpg", "bmp"]:
                    list_img.append(file_full_path)
        return list_img

    def load_image(self, folder_path, index, auto_index):
        list_img = self.list_img_by_path(folder_path)
        index = index%len(list_img) if auto_index else index
        image_path = list_img[index]
        str_list_img = "\n".join(list_img)
        i = Image.open(image_path)
        return (i2tensor(i), str_list_img)
    
class LoadImageUrl:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "Url": ("STRING", {"default": "", "multiline": False},)
        }
        }

    CATEGORY = "📂 SDVN"

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load_image_url"

    def load_image_url(self, Url):  
        if 'http' in Url:
            Url = run_gallery_dl(Url)
            image = Image.open(requests.get(Url, stream=True).raw)
        else:
            image = Image.open(Url)
        image = i2tensor(image)
        results = ALL_NODE["PreviewImage"]().save_images(image)
        results["result"] = (image,)
        return results

class CheckpointLoaderDownload:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "Download": ("BOOLEAN", {"default": True},),
                "Download_url": ("STRING", {"default": "", "multiline": False},),
                "Ckpt_url_name": ("STRING", {"default": "model.safetensors", "multiline": False},),
            },
            "optional": {
                "Ckpt_name": (none2list(folder_paths.get_filename_list("checkpoints")), {"tooltip": "The name of the checkpoint (model) to load."})
            }
        }
    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "STRING")
    RETURN_NAMES = ("model", "clip", "vae", "ckpt_path")
    OUTPUT_TOOLTIPS = ("The model used for denoising latents.",
                       "The CLIP model used for encoding text prompts.",
                       "The VAE model used for encoding and decoding images to and from latent space.")
    FUNCTION = "load_checkpoint"

    CATEGORY = "📂 SDVN"
    DESCRIPTION = "Loads a diffusion model checkpoint, diffusion models are used to denoise latents."

    def load_checkpoint(self, Download, Download_url, Ckpt_url_name, Ckpt_name=None):
        if Download and Download_url != "":
            download_model(Download_url, Ckpt_url_name, "checkpoints")
            Ckpt_name = Ckpt_url_name
        results = ALL_NODE["CheckpointLoaderSimple"]().load_checkpoint(Ckpt_name)
        path = folder_paths.get_full_path_or_raise("checkpoints", Ckpt_name)
        index = 0
        if not Download or Download_url == '':
            name = Ckpt_name.split("/")[-1].rsplit(".", 1)[0]
            for i in ["jpg","jpeg","png"]:
                if os.path.exists(os.path.join(os.path.dirname(path),f"{name}.{i}")):
                    i_cover = os.path.join(os.path.dirname(path),f"{name}.{i}")
                    index += 1
        if index > 0:
            i = Image.open(i_cover)
            i = i2tensor(i)
            ui = ALL_NODE["PreviewImage"]().save_images(i)["ui"]
            return {"ui":ui, "result":(results[0],results[1],results[2],path)}
        else:
            return (results[0],results[1],results[2],path)
        
class LoraLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "Download": ("BOOLEAN", {"default": True},),
                "Download_url": ("STRING", {"default": "", "multiline": False},),
                "Lora_url_name": ("STRING", {"default": "model.safetensors", "multiline": False},),
                "lora_name": (none2list(folder_paths.get_filename_list("loras")), {"default": "None", "tooltip": "The name of the LoRA."}),
            },
            "optional": {
                "model": ("MODEL", {"tooltip": "The diffusion model the LoRA will be applied to."}),
                "clip": ("CLIP", {"default": None, "tooltip": "The CLIP model the LoRA will be applied to."}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to modify the diffusion model. This value can be negative."}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to modify the CLIP model. This value can be negative."}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("model", "clip", "lora_path")
    FUNCTION = "load_lora"
    CATEGORY = "📂 SDVN"

    def load_lora(self, Download, Download_url, Lora_url_name, lora_name, model = None, clip = None, strength_model=1, strength_clip=1):
        if not Download or Download_url == '':
            if lora_name == "None":
                return (model, clip)
        if Download and Download_url != '':
            download_model(Download_url, Lora_url_name, "loras")
            lora_name = Lora_url_name
        path = folder_paths.get_full_path_or_raise("loras", lora_name)
        index = 0
        if not Download or Download_url == '':
            name = lora_name.split("/")[-1].rsplit(".", 1)[0]
            for i in ["jpg","jpeg","png"]:
                if os.path.exists(os.path.join(os.path.dirname(path),f"{name}.{i}")):
                    i_cover = os.path.join(os.path.dirname(path),f"{name}.{i}")
                    index += 1

        if model == None or clip == None:
            results = (None, None,)
        else:
            results = ALL_NODE["LoraLoader"]().load_lora(model, clip, lora_name, strength_model, strength_clip)
        if index > 0:
            i = Image.open(i_cover)
            i = i2tensor(i)
            ui = ALL_NODE["PreviewImage"]().save_images(i)["ui"]
            return {"ui":ui, "result":(results[0],results[1],path)}
        else:
            return (results[0],results[1],path)

class CLIPTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "The text to be encoded."}),
                "negative": ("STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "The text to be encoded."}),
                "style": (none2list(style_list()[0]),{"default": "None"}),
                "translate": (lang_list(),),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "The random seed"}),
                "clip": ("CLIP", {"tooltip": "The CLIP model used for encoding the text."})
            }
        }
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "STRING")
    RETURN_NAMES = ("positive", "negative", "prompt")
    OUTPUT_TOOLTIPS = (
        "A conditioning containing the embedded text used to guide the diffusion model.",)
    FUNCTION = "encode"

    CATEGORY = "📂 SDVN"
    DESCRIPTION = "Encodes a text prompt using a CLIP model into an embedding that can be used to guide the diffusion model towards generating specific images."

    def encode(self, clip, positive, negative, style, translate, seed):
        if style != "None":
            positive += f"{positive}, {style_list()[1][style_list()[0].index(style)][1]}"
            negative += f"{negative}, {style_list()[1][style_list()[0].index(style)][2]}" if len(style_list()[1][style_list()[0].index(style)]) > 2 else ""
        if "DPRandomGenerator" in ALL_NODE:
            cls = ALL_NODE["DPRandomGenerator"]
            positive = cls().get_prompt(positive, seed, 'No')[0]
            negative = cls().get_prompt(negative, seed, 'No')[0]
        positive = ALL_NODE["SDVN Translate"]().ggtranslate(positive,translate)[0]
        negative = ALL_NODE["SDVN Translate"]().ggtranslate(negative,translate)[0]
        prompt =f"""
Positive: {positive}

Negative: {negative}
        """
        token_p = clip.tokenize(positive)
        token_n = clip.tokenize(negative)
        return (clip.encode_from_tokens_scheduled(token_p), clip.encode_from_tokens_scheduled(token_n), prompt)

class StyleLoad:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("STRING", {"multiline": True, "dynamicPrompts": True, "placeholder": "Adding style tags: Edit the file my_styles.csv.example to my_styles.csv, then add the style you desire"}),
                "negative": ("STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "The text to be encoded."}),
                "style": (none2list(style_list()[0]),{"default": "None"}),
                "style2": (none2list(style_list()[0]),{"default": "None"}),
                "style3": (none2list(style_list()[0]),{"default": "None"}),
                "style4": (none2list(style_list()[0]),{"default": "None"}),
                "style5": (none2list(style_list()[0]),{"default": "None"}),
                "style6": (none2list(style_list()[0]),{"default": "None"}),
                "translate": (lang_list(),),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "The random seed"}),
            }
        }
    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("positive", "negative",)
    FUNCTION = "loadstyle"

    CATEGORY = "📂 SDVN"
    DESCRIPTION = "Encodes a text prompt using a CLIP model into an embedding that can be used to guide the diffusion model towards generating specific images."

    def loadstyle(self, positive, negative, translate, seed, **kargs):
        print(kargs)
        for i in kargs:
            if kargs[i] != "None":
                positive = f"{positive}, {style_list()[1][style_list()[0].index(kargs[i])][1]}"
                negative = f"{negative}, {style_list()[1][style_list()[0].index(kargs[i])][2]}" if len(style_list()[1][style_list()[0].index(kargs[i])]) > 2 else ""
        if "DPRandomGenerator" in ALL_NODE:
            cls = ALL_NODE["DPRandomGenerator"]
            positive = cls().get_prompt(positive, seed, 'No')[0]
            negative = cls().get_prompt(negative, seed, 'No')[0]
        positive = ALL_NODE["SDVN Translate"]().ggtranslate(positive,translate)[0]
        negative = ALL_NODE["SDVN Translate"]().ggtranslate(negative,translate)[0]
        return (positive,negative,)
    
ModelType_list = {
    "SD 1.5": [7.0, "euler_ancestral", "normal"],
    "SDXL": [9.0, "dpmpp_2m_sde", "karras"],
    "Flux": [1.0, "euler", "simple"],
    "SD 1.5 Hyper": [1.0, "euler_ancestral", "sgm_uniform"],
    "SDXL Hyper": [1.0, "euler_ancestral", "sgm_uniform"],
    "SDXL Lightning": [1.0, "dpmpp_2m_sde", "sgm_uniform"],
}

StepsType_list = {
    "Denoise": 20,
    "Lightning 8steps": 8,
    "Hyper 8steps": 8,
    "Lightning 4steps": 8,
    "Hyper 4steps": 8,
    "Flux dev turbo (hyper 8steps)": 8,
    "Flux schnell": 4,
}


class Easy_KSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model used for denoising the input latent."}),
                "positive": ("CONDITIONING", {"tooltip": "The conditioning describing the attributes you want to include in the image."}),
                "ModelType": (none2list(list(ModelType_list)),),
                "StepsType": (none2list(list(StepsType_list)),),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The amount of denoising applied, lower values will maintain the structure of the initial image allowing for image to image sampling."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "tooltip": "The number of steps used in the denoising process."}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01, "tooltip": "The Classifier-Free Guidance scale balances creativity and adherence to the prompt. Higher values result in images more closely matching the prompt however too high values will negatively impact quality."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "The algorithm used when sampling, this can affect the quality, speed, and style of the generated output."}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": "The scheduler controls how noise is gradually removed to form the image."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "The random seed used for creating the noise."}),
                "Tiled": ("BOOLEAN", {"default": False},),
            },
            "optional": {
                "negative": ("CONDITIONING", {"tooltip": "The conditioning describing the attributes you want to exclude from the image."}),
                "latent_image": ("LATENT", {"tooltip": "The latent image to denoise."}),
                "vae": ("VAE", {"tooltip": "The VAE model used for decoding the latent."}),
                "tile_width": ("INT", {"default": 1024, "min": 512, "max": 4096, "step": 64, }),
                "tile_height": ("INT", {"default": 1024, "min": 512, "max": 4096, "step": 64, }),
                "FluxGuidance":  ("FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("LATENT", "IMAGE",)
    OUTPUT_TOOLTIPS = ("The denoised latent.",)
    FUNCTION = "sample"

    CATEGORY = "📂 SDVN"
    DESCRIPTION = "Uses the provided model, positive and negative conditioning to denoise the latent image."

    def sample(self, model, positive, ModelType, StepsType, sampler_name, scheduler, seed, Tiled=False, tile_width=None, tile_height=None, steps=20, cfg=7, denoise=1.0, negative=None, latent_image=None, vae=None, FluxGuidance = 3.5):
        if ModelType != 'None':
            cfg, sampler_name, scheduler = ModelType_list[ModelType]
        StepsType_list["Denoise"] = steps
        if FluxGuidance != 3.5:
            positive = ALL_NODE["FluxGuidance"]().append(positive,FluxGuidance)[0]
        if negative == None:
            cls_zero_negative = ALL_NODE["ConditioningZeroOut"]
            negative = cls_zero_negative().zero_out(positive)[0]
        if tile_width == None or tile_height == None:
            tile_width = tile_height = 1024
        if latent_image == None:
            cls_emply = ALL_NODE["EmptyLatentImage"]
            latent_image = cls_emply().generate(tile_width, tile_height, 1)[0]
            tile_width = int(math.ceil(tile_width/2))
            tile_height = int(math.ceil(tile_width/2))
        if Tiled == True:
            if "TiledDiffusion" in ALL_NODE:
                cls_tiled = ALL_NODE["TiledDiffusion"]
                model = cls_tiled().apply(model, "Mixture of Diffusers",
                                          tile_width, tile_height, 96, 4)[0]
            else:
                print(
                    'Not install TiledDiffusion node (https://github.com/shiimizu/ComfyUI-TiledDiffusion)')
        if StepsType != 'None':
            steps = int(math.ceil(StepsType_list[StepsType]*denoise))
        cls = ALL_NODE["KSampler"]
        samples = cls().sample(model, seed, steps, cfg, sampler_name,
                               scheduler, positive, negative, latent_image, denoise)[0]
        if vae != None:
            cls_decode = ALL_NODE["VAEDecode"]
            images = cls_decode().decode(vae, samples)[0]
        else:
            images = None
        return (samples, images,)


class UpscaleImage:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "mode": (["Maxsize", "Resize", "Scale"], ),
            "model_name": (none2list(folder_paths.get_filename_list("upscale_models")), {"default": "None", }),
            "scale": ("FLOAT", {"default": 1, "min": 0, "max": 10, "step": 0.01, }),
            "width": ("INT", {"default": 1024, "min": 0, "max": 4096, "step": 1, }),
            "height": ("INT", {"default": 1024, "min": 0, "max": 4096, "step": 1, }),
            "image": ("IMAGE",),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"

    CATEGORY = "📂 SDVN/🏞️ Image"

    def upscale(self, mode, width, height, scale, model_name, image):
        if width == 0 and height == 0:
            s = image
        else:
            samples = image.movedim(-1, 1)
            w = samples.shape[3]
            h = samples.shape[2]
            if mode == 'Maxsize':
                if width/height < w/h:
                    height = round(h * width / w)
                else:
                    width = round(w * height / h)
            if mode == 'Scale':
                width = round(w * scale)
                height = round(h * scale)
            if width == 0:
                width = max(1, round(w * height / h))
            elif height == 0:
                height = max(1, round(h * width / w))
            if model_name != "None":
                upscale_model = ALL_NODE["UpscaleModelLoader"](
                ).load_model(model_name)[0]
                image = ALL_NODE["ImageUpscaleWithModel"]().upscale(
                    upscale_model, image)[0]
            samples = image.movedim(-1, 1)
            s = comfy.utils.common_upscale(
                samples, width, height, "nearest-exact", "disabled")
            s = s.movedim(1, -1)
        return (s,)


class UpscaleLatentImage:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "mode": (["Maxsize", "Resize", "Scale"], ),
            "model_name": (none2list(folder_paths.get_filename_list("upscale_models")), {"default": "None", }),
            "scale": ("FLOAT", {"default": 2, "min": 0, "max": 10, "step": 0.01, }),
            "width": ("INT", {"default": 1024, "min": 0, "max": 4096, "step": 1, }),
            "height": ("INT", {"default": 1024, "min": 0, "max": 4096, "step": 1, }),
            "latent": ("LATENT",),
            "vae": ("VAE",),
        }}

    RETURN_TYPES = ("LATENT", "VAE",)
    FUNCTION = "upscale_latent"

    CATEGORY = "📂 SDVN/🏞️ Image"

    def upscale_latent(self, mode, width, height, scale, model_name, latent, vae):
        image = ALL_NODE["VAEDecode"]().decode(vae, latent)[0]
        s = UpscaleImage().upscale(mode, width, height,
                                   scale, model_name, image)[0]
        l = ALL_NODE["VAEEncode"]().encode(vae, s)[0]
        return (l, vae,)


def preprocessor_list():
    preprocessor_list = ["None","InvertImage"]
    AIO_NOT_SUPPORTED = ["InpaintPreprocessor",
                         "MeshGraphormer+ImpactDetector-DepthMapPreprocessor", "DiffusionEdge_Preprocessor"]
    AIO_NOT_SUPPORTED += ["SavePoseKpsAsJsonFile", "FacialPartColoringFromPoseKps",
                          "UpperBodyTrackingFromPoseKps", "RenderPeopleKps", "RenderAnimalKps"]
    AIO_NOT_SUPPORTED += ["Unimatch_OptFlowPreprocessor", "MaskOptFlow"]
    for k in ALL_NODE:
        if "Preprocessor" in k and "Inspire" not in k:
            if k not in AIO_NOT_SUPPORTED:
                preprocessor_list += [k]
    return preprocessor_list


class AutoControlNetApply:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"positive": ("CONDITIONING", ),
                             "negative": ("CONDITIONING", ),
                             "image": ("IMAGE", ),
                             "control_net": (none2list(folder_paths.get_filename_list("controlnet")),),
                             "preprocessor": (preprocessor_list(),),
                             "union_type": (["None","auto"] + list(UNION_CONTROLNET_TYPES.keys()),),
                             "resolution": ("INT", {"default": 512, "min": 512, "max": 2048, "step": 1}),
                             "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                             "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                             "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001})
                             },
                "optional": {"vae": ("VAE", ),
                             }
                }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "IMAGE")
    RETURN_NAMES = ("positive", "negative", "image")
    FUNCTION = "apply_controlnet"

    CATEGORY = "📂 SDVN"

    def apply_controlnet(self, positive, negative, control_net, preprocessor, union_type, resolution, image, strength, start_percent, end_percent, vae=None, extra_concat=[]):
        if control_net == "None":
            return (positive, negative, image)
        if preprocessor == "InvertImage":
            image = ALL_NODE["ImageInvert"]().invert(image)[0]
        elif preprocessor != "None":
            if "AIO_Preprocessor" in ALL_NODE:
                r = ALL_NODE["AIO_Preprocessor"]().execute(preprocessor, image, resolution)
                if "result" in r:
                    image = r["result"][0]
                else:
                    image = r[0]
            else:
                print(
                    "You have not installed it yet Controlnet Aux (https://github.com/Fannovel16/comfyui_controlnet_aux)")
        control_net = ALL_NODE["ControlNetLoader"]().load_controlnet(control_net)[0]
        if union_type != "None":
            control_net = ALL_NODE["SetUnionControlNetType"]().set_controlnet_type(control_net,union_type)[0]
        p, n = ALL_NODE["ControlNetApplyAdvanced"]().apply_controlnet(
            positive, negative, control_net, image, strength, start_percent, end_percent, vae)
        results = ALL_NODE["PreviewImage"]().save_images(image)
        results["result"] = (p, n, image)
        return results

class Inpaint:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"SetLatentNoiseMask":("BOOLEAN",),
                             "pixels": ("IMAGE", ),
                             "vae": ("VAE", ),},
                "optional": {"mask": ("MASK", ),
                             "positive": ("CONDITIONING", ),
                             "negative": ("CONDITIONING", ),}
                             }

    RETURN_TYPES = ("CONDITIONING","CONDITIONING","LATENT",)
    RETURN_NAMES = ("positive", "negative","latent",)
    FUNCTION = "encode"

    CATEGORY = "📂 SDVN"

    def encode(self, SetLatentNoiseMask, pixels, vae, mask = None, positive = None, negative = None):
        if SetLatentNoiseMask or mask == None:
            r = ALL_NODE["VAEEncode"]().encode(vae,pixels)[0]
            if mask != None:
                r = ALL_NODE["SetLatentNoiseMask"]().set_mask(r, mask)[0]
        elif positive == None or negative == None:
            r = ALL_NODE["VAEEncodeForInpaint"]().encode(vae, pixels, mask)[0]
        else:
            r = ALL_NODE["InpaintModelConditioning"]().encode(positive, negative, pixels, vae, mask)
            positive = r[0]
            negative = r[1]
            r = r[2]
        return (positive,negative,r,)
    
class ApplyStyleModel:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"positive": ("CONDITIONING", ),
                             "image": ("IMAGE",),
                             "style_model": (folder_paths.get_filename_list("style_models"), ),
                             "clip_vision_model": (folder_paths.get_filename_list("clip_vision"), ),
                             "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}),
                             "downsampling":([1,2,3,4,5,6], {"default": 1}),
                             
                             },
                "hidden":  {"strength_type": (["multiply"], ),}
                             }
    
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("positive",)
    FUNCTION = "applystyle"

    CATEGORY = "📂 SDVN"

    def applystyle(self, positive, image, style_model, clip_vision_model, strength, downsampling, strength_type = "multiply"):
        clip_vision_model = ALL_NODE["CLIPVisionLoader"]().load_clip(clip_vision_model)[0]
        clip_vision_encode = ALL_NODE["CLIPVisionEncode"]().encode(clip_vision_model, image, "center")[0]
        style_model = ALL_NODE["StyleModelLoader"]().load_style_model(style_model)[0]
        mode="area" if downsampling==3 else "bicubic"
        cond = style_model.get_cond(clip_vision_encode).flatten(start_dim=0, end_dim=1).unsqueeze(dim=0)
        if downsampling>1:
            (b,t,h)=cond.shape
            m = int(np.sqrt(t))
            cond=torch.nn.functional.interpolate(cond.view(b, m, m, h).transpose(1,-1), size=(m//downsampling, m//downsampling), mode=mode)
            cond=cond.transpose(1,-1).reshape(b,-1,h)
        if strength_type == "multiply":
            cond *= strength
        c = []
        for t in positive:
            n = [torch.cat((t[0], cond), dim=1), t[1].copy()]
            c.append(n)
        return (c, )

class CheckpointDownload:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "Download_url": ("STRING", {"default": "", "multiline": False},),
                "Ckpt_url_name": ("STRING", {"default": "model.safetensors", "multiline": False},),
            }
        }
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    OUTPUT_TOOLTIPS = ("The model used for denoising latents.",
                       "The CLIP model used for encoding text prompts.",
                       "The VAE model used for encoding and decoding images to and from latent space.")
    FUNCTION = "checkpoint_download"

    CATEGORY = "📂 SDVN/📥 Download"

    def checkpoint_download(self, Download_url, Ckpt_url_name):
        download_model(Download_url, Ckpt_url_name, "checkpoints")
        return ALL_NODE["CheckpointLoaderSimple"]().load_checkpoint(Ckpt_url_name)

class LoraDownload:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model the LoRA will be applied to."}),
                "clip": ("CLIP", {"default": None, "tooltip": "The CLIP model the LoRA will be applied to."}),
                "Download_url": ("STRING", {"default": "", "multiline": False},),
                "Lora_url_name": ("STRING", {"default": "model.safetensors", "multiline": False},),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to modify the diffusion model. This value can be negative."}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to modify the CLIP model. This value can be negative."}),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    OUTPUT_TOOLTIPS = ("The modified diffusion model.",
                       "The modified CLIP model.")
    FUNCTION = "load_lora"

    CATEGORY = "📂 SDVN/📥 Download"
    DESCRIPTION = "LoRAs are used to modify diffusion and CLIP models, altering the way in which latents are denoised such as applying styles. Multiple LoRA nodes can be linked together."

    def load_lora(self, model, clip, Download_url, Lora_url_name, strength_model, strength_clip):
        download_model(Download_url, Lora_url_name, "loras")
        return ALL_NODE["LoraLoader"]().load_lora(model, clip, Lora_url_name, strength_model, strength_clip)

class CLIPVisionDownload:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
                    "Download_url": ("STRING", {"default": "", "multiline": False},),
                    "Url_name": ("STRING", {"default": "model.safetensors", "multiline": False},)
                             }}
    RETURN_TYPES = ("CLIP_VISION",)
    FUNCTION = "download"

    CATEGORY = "📂 SDVN/📥 Download"

    def download(self, Download_url, Url_name):
        download_model(Download_url, Url_name, "clip_vision")
        return ALL_NODE["CLIPVisionLoader"]().load_clip(Url_name)

class UpscaleModelDownload:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
                    "Download_url": ("STRING", {"default": "", "multiline": False},),
                    "Url_name": ("STRING", {"default": "model.safetensors", "multiline": False},)
                             }}
    RETURN_TYPES = ("UPSCALE_MODEL",)
    FUNCTION = "download"

    CATEGORY = "📂 SDVN/📥 Download"

    def download(self, Download_url, Url_name):
        download_model(Download_url, Url_name, "upscale_models")
        return ALL_NODE["UpscaleModelLoader"]().load_model(Url_name)

class VAEDownload:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
                    "Download_url": ("STRING", {"default": "", "multiline": False},),
                    "Url_name": ("STRING", {"default": "model.safetensors", "multiline": False},)
                             }}
    RETURN_TYPES = ("VAE",)
    FUNCTION = "download"

    CATEGORY = "📂 SDVN/📥 Download"

    def download(self, Download_url, Url_name):
        download_model(Download_url, Url_name, "vae")
        return ALL_NODE["VAELoader"]().load_vae(Url_name)

class ControlNetDownload:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
                    "Download_url": ("STRING", {"default": "", "multiline": False},),
                    "Url_name": ("STRING", {"default": "model.safetensors", "multiline": False},)
                             }}
    RETURN_TYPES = ("CONTROL_NET",)
    FUNCTION = "download"

    CATEGORY = "📂 SDVN/📥 Download"

    def download(self, Download_url, Url_name):
        download_model(Download_url, Url_name, "controlnet")
        return ALL_NODE["ControlNetLoader"]().load_controlnet(Url_name)

class UNETDownload:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
                    "Download_url": ("STRING", {"default": "", "multiline": False},),
                    "Url_name": ("STRING", {"default": "model.safetensors", "multiline": False}),
                    "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],)
                             }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "download"

    CATEGORY = "📂 SDVN/📥 Download"

    def download(self, Download_url, Url_name, weight_dtype):
        download_model(Download_url, Url_name, "diffusion_models")
        return ALL_NODE["UNETLoader"]().load_unet(Url_name,weight_dtype)

class CLIPDownload:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
                    "Download_url": ("STRING", {"default": "", "multiline": False},),
                    "Url_name": ("STRING", {"default": "model.safetensors", "multiline": False}),
                    "type": (["stable_diffusion", "stable_cascade", "sd3", "stable_audio", "mochi", "ltxv"],)
                             }}
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "download"

    CATEGORY = "📂 SDVN/📥 Download"

    def download(self, Download_url, Url_name, type):
        download_model(Download_url, Url_name, "text_encoders")
        return ALL_NODE["CLIPLoader"]().load_clip(Url_name,type)

class StyleModelDownload:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
                    "Download_url": ("STRING", {"default": "", "multiline": False},),
                    "Url_name": ("STRING", {"default": "model.safetensors", "multiline": False},)
                             }}
    RETURN_TYPES = ("STYLE_MODEL",)
    FUNCTION = "download"

    CATEGORY = "📂 SDVN/📥 Download"

    def download(self, Download_url, Url_name):
        download_model(Download_url, Url_name, "style_models")
        return ALL_NODE["StyleModelLoader"]().load_style_model(Url_name)
    
                
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "SDVN Load Checkpoint": CheckpointLoaderDownload,
    "SDVN Load Lora": LoraLoader,
    "SDVN Load Image": LoadImage,
    "SDVN Load Image Folder": LoadImageFolder,
    "SDVN Load Image Url": LoadImageUrl,
    "SDVN CLIP Text Encode": CLIPTextEncode,
    "SDVN Controlnet Apply": AutoControlNetApply,
    "SDVN Inpaint": Inpaint,
    "SDVN Apply Style Model": ApplyStyleModel,
    "SDVN KSampler": Easy_KSampler,
    "SDVN Styles":StyleLoad, 
    "SDVN Upscale Image": UpscaleImage,
    "SDVN UPscale Latent": UpscaleLatentImage,
    "SDVN Checkpoint Download": CheckpointDownload,
    "SDVN Lora Download": LoraDownload,
    "SDVN CLIPVision Download":CLIPVisionDownload,
    "SDVN UpscaleModel Download":UpscaleModelDownload,
    "SDVN VAE Download":VAEDownload,
    "SDVN ControlNet Download":ControlNetDownload,
    "SDVN UNET Download":UNETDownload,
    "SDVN CLIP Download":CLIPDownload,
    "SDVN StyleModel Download":StyleModelDownload,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "SDVN Load Checkpoint": "📀 Load Checkpoint",
    "SDVN Load Lora": "🎨 Load Lora",
    "SDVN Load Image": "🏞️ Load Image",
    "SDVN Load Image Folder": "🏞️ Load Image Folder",
    "SDVN Load Image Url": "📥 Load Image Url",
    "SDVN CLIP Text Encode": "🔡 CLIP Text Encode",
    "SDVN KSampler": "⌛️ KSampler",
    "SDVN Controlnet Apply": "🎚️ Controlnet Apply",
    "SDVN Inpaint": "👨‍🎨 Inpaint",
    "SDVN Apply Style Model": "🌈 Apply Style Model",
    "SDVN Styles":"🗂️ Prompt Styles",
    "SDVN Upscale Image": "↗️ Upscale Image",
    "SDVN UPscale Latent": "↗️ Upscale Latent",
    "SDVN Checkpoint Download": "📥 Checkpoint Download",
    "SDVN Lora Download": "📥 Lora Download",
    "SDVN CLIPVision Download":"📥 CLIPVision Download",
    "SDVN UpscaleModel Download":"📥 UpscaleModel Download",
    "SDVN VAE Download":"📥 VAE Download",
    "SDVN ControlNet Download":"📥 ControlNet Download",
    "SDVN UNET Download":"📥 UNET Download",
    "SDVN CLIP Download":"📥 CLIP Download",
    "SDVN StyleModel Download":"📥  StyleModel Download",
}

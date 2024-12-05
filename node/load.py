import comfy.sd
import requests
import math
import os
import re
import sys
from PIL import Image, ImageOps
import torch
import subprocess
import numpy as np
import folder_paths
import comfy.utils
import hashlib
from PIL.PngImagePlugin import PngInfo
from nodes import NODE_CLASS_MAPPINGS as ALL_NODE_CLASS_MAPPINGS
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "comfy"))


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
    result = subprocess.run(command, check=True,
                            text=True, capture_output=True)
    return result.stdout.strip()


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
    if type == "ckpt":
        checkpoint_path = os.path.join(folder_paths.models_dir, "checkpoints")
    if type == "lora":
        checkpoint_path = os.path.join(folder_paths.models_dir, "loras")
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
                file_list.append(file_path)

        return {
            "required": {
                "Load_url": ("BOOLEAN", {"default": True},),
                "Url": ("STRING", {"default": "", "multiline": False},),
                "image": (sorted(file_list), {"image_upload": True})
            }
        }

    CATEGORY = "✨ SDVN"

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"

    def load_image(self, Url, Load_url, image=None):
        if Url != '' and Load_url:
            if 'pinterest.com' in Url:
                Url = run_gallery_dl(Url)
            i = Image.open(requests.get(Url, stream=True).raw)
        else:
            image_path = folder_paths.get_annotated_filepath(image)
            i = Image.open(image_path)
        ii = ImageOps.exif_transpose(i)
        if 'A' in ii.getbands():
            mask = np.array(ii.getchannel('A')).astype(np.float32) / 255.0
            mask = 1. - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
        return (i2tensor(i), mask.unsqueeze(0))

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


class LoadImageUrl:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "Url": ("STRING", {"default": "", "multiline": False},)
        }
        }

    CATEGORY = "✨ SDVN"

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load_image_url"

    def load_image_url(self, Url):
        if 'pinterest.com' in Url:
            Url = run_gallery_dl(Url)
        image = Image.open(requests.get(Url, stream=True).raw)
        return (i2tensor(image),)


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
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    OUTPUT_TOOLTIPS = ("The model used for denoising latents.",
                       "The CLIP model used for encoding text prompts.",
                       "The VAE model used for encoding and decoding images to and from latent space.")
    FUNCTION = "load_checkpoint"

    CATEGORY = "✨ SDVN"
    DESCRIPTION = "Loads a diffusion model checkpoint, diffusion models are used to denoise latents."

    def load_checkpoint(self, Download, Download_url, Ckpt_url_name, Ckpt_name=None):
        if Download and Download_url != "":
            download_model(Download_url, Ckpt_url_name, "ckpt")
            ckpt_path = folder_paths.get_full_path_or_raise(
                "checkpoints", Ckpt_url_name)
        else:
            ckpt_path = folder_paths.get_full_path_or_raise(
                "checkpoints", Ckpt_name)
        out = comfy.sd.load_checkpoint_guess_config(
            ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
        return out[:3]


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

    CATEGORY = "✨ SDVN"
    DESCRIPTION = "Loads a diffusion model checkpoint, diffusion models are used to denoise latents."

    def checkpoint_download(self, Download_url, Ckpt_url_name):
        download_model(Download_url, Ckpt_url_name, "ckpt")
        ckpt_path = folder_paths.get_full_path_or_raise(
            "checkpoints", Ckpt_url_name)
        out = comfy.sd.load_checkpoint_guess_config(
            ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
        return out[:3]


class LoraDownload:
    def __init__(self):
        self.loaded_lora = None

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

    CATEGORY = "✨ SDVN"
    DESCRIPTION = "LoRAs are used to modify diffusion and CLIP models, altering the way in which latents are denoised such as applying styles. Multiple LoRA nodes can be linked together."

    def load_lora(self, model, clip, Download_url, Lora_url_name, strength_model, strength_clip):
        download_model(Download_url, Lora_url_name, "lora")
        lora_path = folder_paths.get_full_path_or_raise(
            "loras", Lora_url_name)

        if strength_model == 0 and strength_clip == 0:
            return (model, clip)

        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                temp = self.loaded_lora
                self.loaded_lora = None
                del temp

        if lora is None:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)

        model_lora, clip_lora = comfy.sd.load_lora_for_models(
            model, clip, lora, strength_model, strength_clip)
        return (model_lora, clip_lora)


class LoraLoader:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model the LoRA will be applied to."}),
                "clip": ("CLIP", {"default": None, "tooltip": "The CLIP model the LoRA will be applied to."}),
                "Download": ("BOOLEAN", {"default": True},),
                "Download_url": ("STRING", {"default": "", "multiline": False},),
                "Lora_url_name": ("STRING", {"default": "model.safetensors", "multiline": False},),
                "lora_name": (none2list(folder_paths.get_filename_list("loras")), {"default": "None", "tooltip": "The name of the LoRA."}),
            },
            "optional": {
                "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to modify the diffusion model. This value can be negative."}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to modify the CLIP model. This value can be negative."}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    OUTPUT_TOOLTIPS = ("The modified diffusion model.",
                       "The modified CLIP model.")
    FUNCTION = "load_lora"

    CATEGORY = "✨ SDVN"
    DESCRIPTION = "LoRAs are used to modify diffusion and CLIP models, altering the way in which latents are denoised such as applying styles. Multiple LoRA nodes can be linked together."

    def load_lora(self, model, clip, Download, Download_url, Lora_url_name, lora_name, strength_model=1, strength_clip=1):
        if not Download and Download_url == '' and lora_name == "None":
            return (model, clip)
        if Download and Download_url != '':
            download_model(Download_url, Lora_url_name, "lora")
            lora_path = folder_paths.get_full_path_or_raise(
                "loras", Lora_url_name)
        else:
            lora_path = folder_paths.get_full_path_or_raise(
                "loras", lora_name)
        if strength_model == 0 and strength_clip == 0:
            return (model, clip)
        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                temp = self.loaded_lora
                self.loaded_lora = None
                del temp

        if lora is None:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)

        model_lora, clip_lora = comfy.sd.load_lora_for_models(
            model, clip, lora, strength_model, strength_clip)
        return (model_lora, clip_lora)


class CLIPTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "The text to be encoded."}),
                "negative": ("STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "The text to be encoded."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "The random seed"}),
                "clip": ("CLIP", {"tooltip": "The CLIP model used for encoding the text."})
            }
        }
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING",)
    RETURN_NAMES = ("positive", "negative",)
    OUTPUT_TOOLTIPS = (
        "A conditioning containing the embedded text used to guide the diffusion model.",)
    FUNCTION = "encode"

    CATEGORY = "✨ SDVN"
    DESCRIPTION = "Encodes a text prompt using a CLIP model into an embedding that can be used to guide the diffusion model towards generating specific images."

    def encode(self, clip, positive, negative, seed):
        if "DPRandomGenerator" in ALL_NODE_CLASS_MAPPINGS:
            cls = ALL_NODE_CLASS_MAPPINGS["DPRandomGenerator"]
            positive = cls().get_prompt(positive, seed, 'No')[0]
            negative = cls().get_prompt(negative, seed, 'No')[0]
        token_p = clip.tokenize(positive)
        token_n = clip.tokenize(negative)
        return (clip.encode_from_tokens_scheduled(token_p), clip.encode_from_tokens_scheduled(token_n), )


def dic2list(dic):
    l = []
    for i in dic:
        l += [i]
    return l


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
                "ModelType": (none2list(dic2list(ModelType_list)),),
                "StepsType": (none2list(dic2list(StepsType_list)),),
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
            }
        }

    RETURN_TYPES = ("LATENT", "IMAGE",)
    OUTPUT_TOOLTIPS = ("The denoised latent.",)
    FUNCTION = "sample"

    CATEGORY = "✨ SDVN"
    DESCRIPTION = "Uses the provided model, positive and negative conditioning to denoise the latent image."

    def sample(self, model, positive, ModelType, StepsType, sampler_name, scheduler, seed, Tiled=False, tile_width=None, tile_height=None, steps=20, cfg=7, denoise=1.0, negative=None, latent_image=None, vae=None):
        if ModelType != 'None':
            cfg, sampler_name, scheduler = ModelType_list[ModelType]
        StepsType_list["Denoise"] = steps
        if negative == None:
            cls_zero_negative = ALL_NODE_CLASS_MAPPINGS["ConditioningZeroOut"]
            negative = cls_zero_negative().zero_out(positive)[0]
        if tile_width == None or tile_height == None:
            tile_width = tile_height = 1024
        if latent_image == None:
            cls_emply = ALL_NODE_CLASS_MAPPINGS["EmptyLatentImage"]
            latent_image = cls_emply().generate(tile_width, tile_height, 1)[0]
            tile_width = int(math.ceil(tile_width/2))
            tile_height = int(math.ceil(tile_width/2))
        if Tiled == True:
            if "TiledDiffusion" in ALL_NODE_CLASS_MAPPINGS:
                cls_tiled = ALL_NODE_CLASS_MAPPINGS["TiledDiffusion"]
                model = cls_tiled().apply(model, "Mixture of Diffusers",
                                          tile_width, tile_height, 96, 4)[0]
            else:
                print(
                    'Not install TiledDiffusion node (https://github.com/shiimizu/ComfyUI-TiledDiffusion)')
        if StepsType != 'None':
            steps = int(math.ceil(StepsType_list[StepsType]*denoise))
        cls = ALL_NODE_CLASS_MAPPINGS["KSampler"]
        samples = cls().sample(model, seed, steps, cfg, sampler_name,
                               scheduler, positive, negative, latent_image, denoise)[0]
        if vae != None:
            cls_decode = ALL_NODE_CLASS_MAPPINGS["VAEDecode"]
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

    CATEGORY = "✨ SDVN/Image"

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
                upscale_model = ALL_NODE_CLASS_MAPPINGS["UpscaleModelLoader"](
                ).load_model(model_name)[0]
                image = ALL_NODE_CLASS_MAPPINGS["ImageUpscaleWithModel"]().upscale(
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

    CATEGORY = "✨ SDVN/Image"

    def upscale_latent(self, mode, width, height, scale, model_name, latent, vae):
        image = ALL_NODE_CLASS_MAPPINGS["VAEDecode"]().decode(vae, latent)[0]
        s = UpscaleImage().upscale(mode, width, height,
                                   scale, model_name, image)[0]
        l = ALL_NODE_CLASS_MAPPINGS["VAEEncode"]().encode(vae, s)[0]
        return (l, vae,)


def preprocessor_list():
    preprocessor_list = ["None"]
    AIO_NOT_SUPPORTED = ["InpaintPreprocessor",
                         "MeshGraphormer+ImpactDetector-DepthMapPreprocessor", "DiffusionEdge_Preprocessor"]
    AIO_NOT_SUPPORTED += ["SavePoseKpsAsJsonFile", "FacialPartColoringFromPoseKps",
                          "UpperBodyTrackingFromPoseKps", "RenderPeopleKps", "RenderAnimalKps"]
    AIO_NOT_SUPPORTED += ["Unimatch_OptFlowPreprocessor", "MaskOptFlow"]
    for k in ALL_NODE_CLASS_MAPPINGS:
        if "Preprocessor" in k:
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

    CATEGORY = "✨ SDVN"

    def apply_controlnet(self, positive, negative, control_net, preprocessor, resolution, image, strength, start_percent, end_percent, vae=None, extra_concat=[]):
        if control_net == "None":
            return (positive, negative, image)
        if preprocessor != "None":
            if "AIO_Preprocessor" in ALL_NODE_CLASS_MAPPINGS:
                image = ALL_NODE_CLASS_MAPPINGS["AIO_Preprocessor"]().execute(
                    preprocessor, image, resolution)[0]
            else:
                print(
                    "You have not installed it yet Controlnet Aux (https://github.com/Fannovel16/comfyui_controlnet_aux)")
        control_net = ALL_NODE_CLASS_MAPPINGS["ControlNetLoader"](
        ).load_controlnet(control_net)[0]
        p, n = ALL_NODE_CLASS_MAPPINGS["ControlNetApplyAdvanced"]().apply_controlnet(
            positive, negative, control_net, image, strength, start_percent, end_percent, vae)
        return (p, n, image)


# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "SDVN Load Checkpoint": CheckpointLoaderDownload,
    "SDVN Load Lora": LoraLoader,
    "SDVN Load Image": LoadImage,
    "SDVN Load Image Url": LoadImageUrl,
    "SDVN Checkpoint Download": CheckpointDownload,
    "SDVN Lora Download": LoraDownload,
    "SDVN CLIP Text Encode": CLIPTextEncode,
    "SDVN KSampler": Easy_KSampler,
    "SDVN Upscale Image": UpscaleImage,
    "SDVN UPscale Latent": UpscaleLatentImage,
    "SDVN Controlnet Apply": AutoControlNetApply,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "SDVN Load Checkpoint": "Load Checkpoint",
    "SDVN Load Lora": "Load Lora",
    "SDVN Load Image": "Load Image",
    "SDVN Load Image Url": "Load Image Url",
    "SDVN Checkpoint Download": "Download Checkpoint",
    "SDVN Lora Download": "Download Lora",
    "SDVN CLIP Text Encode": "CLIP Text Encode",
    "SDVN KSampler": "KSampler",
    "SDVN Upscale Image": "↗️ Upscale Image",
    "SDVN UPscale Latent": "↗️ Upscale Latent",
    "SDVN Controlnet Apply": "Controlnet Apply"
}

import comfy.sd
import requests
import os
import re
import sys
from PIL import Image, ImageOps
import torch
import subprocess
import numpy as np
import folder_paths
import hashlib
from PIL.PngImagePlugin import PngInfo
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "comfy"))


def pil2tensor(images: Image.Image | list[Image.Image]) -> torch.Tensor:
    """Converts a PIL Image or a list of PIL Images to a tensor."""

    def single_pil2tensor(image: Image.Image) -> torch.Tensor:
        np_image = np.array(image).astype(np.float32) / 255.0
        if np_image.ndim == 2:  # Grayscale
            return torch.from_numpy(np_image).unsqueeze(0)  # (1, H, W)
        else:  # RGB or RGBA
            return torch.from_numpy(np_image).unsqueeze(0)  # (1, H, W, C)

    if isinstance(images, Image.Image):
        return single_pil2tensor(images)
    else:
        return torch.cat([single_pil2tensor(img) for img in images], dim=0)


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

        return {"required": {
            "Load_url": ("BOOLEAN", {"default": True},),
            "Url": ("STRING", {"default": "", "multiline": False},),
            "image": (sorted(file_list), {"image_upload": True})

        }
        }

    CATEGORY = "SDVN"

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"

    def load_image(self, image, Url, Load_url):
        if Url != '' and Load_url:
            if 'pinterest.com' in Url:
                Url = run_gallery_dl(Url)
            image = Image.open(requests.get(Url, stream=True).raw)
            image = ImageOps.exif_transpose(image)
            return (pil2tensor(image),)

        else:
            image_path = folder_paths.get_annotated_filepath(image)
            i = Image.open(image_path)
            i = ImageOps.exif_transpose(i)
            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
            return (image, mask.unsqueeze(0))

    @classmethod
    def IS_CHANGED(s, image, Url, Load_url):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image, Url, Load_url):
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

    CATEGORY = "SDVN"

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load_image_url"

    def load_image_url(self, Url):
        if 'pinterest.com' in Url:
            Url = run_gallery_dl(Url)
        image = Image.open(requests.get(Url, stream=True).raw)
        image = ImageOps.exif_transpose(image)
        return (pil2tensor(image),)


class CheckpointLoaderDownload:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "Download": ("BOOLEAN", {"default": True},),
                "Download_url": ("STRING", {"default": "", "multiline": False},),
                "Ckpt_url_name": ("STRING", {"default": "model.safetensors", "multiline": False},),
                "Ckpt_name": (folder_paths.get_filename_list("checkpoints"), {"tooltip": "The name of the checkpoint (model) to load."})
            }
        }
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    OUTPUT_TOOLTIPS = ("The model used for denoising latents.",
                       "The CLIP model used for encoding text prompts.",
                       "The VAE model used for encoding and decoding images to and from latent space.")
    FUNCTION = "load_checkpoint"

    CATEGORY = "SDVN"
    DESCRIPTION = "Loads a diffusion model checkpoint, diffusion models are used to denoise latents."

    def load_checkpoint(self, Download, Download_url, Ckpt_url_name, Ckpt_name):
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
                "lora_name": (folder_paths.get_filename_list("loras"), {"tooltip": "The name of the LoRA."}),
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

    def load_lora(self, model, clip, Download, Download_url, Lora_url_name, lora_name, strength_model, strength_clip):
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


# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "SDVN Load Checkpoint": CheckpointLoaderDownload,
    "SDVN Load Lora": LoraLoader,
    "SDVN Load Image": LoadImage,
    "SDVN Load Image Url": LoadImageUrl,
    "SDVN Checkpoint Download": CheckpointDownload,
    "SDVN Lora Download": LoraDownload,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "SDVN Load Checkpoint": "Load Checkpoint",
    "SDVN Load Lora": "Load Lora",
    "SDVN Load Image": "Load Image",
    "SDVN Load Image Url": "Load Image Url",
    "SDVN Checkpoint Download": "Download Checkpoint",
    "SDVN Lora Download": "Download Lora",
}

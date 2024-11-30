import comfy.sd
import requests
import os, re
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


def download(url, name):
    input_dir = folder_paths.get_input_directory()
    command = ['wget', url, '-O', f'{input_dir}/{name}']
    subprocess.run(command, check=True, text=True, capture_output=True)


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


def download_model(url, name):
    url = url.replace("&", "\&").split("?")[0]
    url = check_link(url)
    checkpoint_path = os.path.join(folder_paths.models_dir, "checkpoints")
    command = ['aria2c', '-c', '-x', '16', '-s', '16',
               '-k', '1M', f'{url}{token(url)}', '-d', checkpoint_path, '-o', name]
    subprocess.run(command, check=True, text=True, capture_output=True)


class download_img:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Image_url": ("STRING", {"default": "", "multiline": False},),
                "Name": ("STRING", {"default": "image.jpg", "multiline": False},)
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load"
    CATEGORY = "SDVN"

    def load(self, Image_url, Name):
        # get the image from the url
        if 'pinterest.com' in Image_url:
            Image_url = run_gallery_dl(Image_url)
        download(Image_url, Name)
        image_path = f'{folder_paths.get_input_directory()}/{Name}'
        image = Image.open(image_path)
        image = ImageOps.exif_transpose(image)
        return (pil2tensor(image),)


class load_image_url:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Image_url": ("STRING", {"default": "", "multiline": True},),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load"
    CATEGORY = "SDVN"

    def load(self, Image_url):
        # get the image from the url
        if 'pinterest.com' in Image_url:
            Image_url = run_gallery_dl(Image_url)
        image = Image.open(requests.get(Image_url, stream=True).raw)
        image = ImageOps.exif_transpose(image)
        return (pil2tensor(image),)


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


class CheckpointLoaderDownload:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "Url": ("STRING", {"default": "", "multiline": False},),
                "ckpt_name": ("STRING", {"default": "model.safetensors", "multiline": False},)
            }
        }
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    OUTPUT_TOOLTIPS = ("The model used for denoising latents.",
                       "The CLIP model used for encoding text prompts.",
                       "The VAE model used for encoding and decoding images to and from latent space.")
    FUNCTION = "load_checkpoint"

    CATEGORY = "SDVN"
    DESCRIPTION = "Loads a diffusion model checkpoint, diffusion models are used to denoise latents."

    def load_checkpoint(self, Url, ckpt_name):
        download_model(Url, ckpt_name)
        ckpt_path = folder_paths.get_full_path_or_raise(
            "checkpoints", ckpt_name)
        out = comfy.sd.load_checkpoint_guess_config(
            ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
        return out[:3]


# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "SDVN Load Image": LoadImage,
    "SDVN Load Image Url": load_image_url,
    "SDVN Load Image Down": download_img,
    "SDVN Checkpoint Down": CheckpointLoaderDownload,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "SDVN Load Image": "Load Image",
    "SDVN Load Image Url": "Load Image Url",
    "SDVN Load Image Down": "Load Image Down",
    "SDVN Checkpoint Down": "Checkpoint Download"
}

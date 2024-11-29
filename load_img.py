import requests
import os
from PIL import Image, ImageOps
import torch
import subprocess
import numpy as np
import folder_paths
import hashlib
from PIL.PngImagePlugin import PngInfo


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


class load_image_url:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Image_url": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True
                    },
                ),
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


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "Load Image Url": load_image_url,
    "Load Image (SubFolder)": LoadImage
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Load Image Url": "SDVN Load Image Url",
    "Load Image (SubFolder)": "SDVN Load Image"
}

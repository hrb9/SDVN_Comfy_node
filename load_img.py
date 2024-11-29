import requests
from PIL import Image, ImageOps
import torch
import subprocess
import numpy as np


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


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "Load Image Url": load_image_url
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Load Image Url": "SDVN Load Image"
}

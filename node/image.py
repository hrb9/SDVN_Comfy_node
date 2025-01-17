from nodes import NODE_CLASS_MAPPINGS as ALL_NODE
import torch, numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import platform, math, folder_paths, os, subprocess
from gradio_client import Client, handle_file

os_name = platform.system()

def create_image_with_text(text, image_size=(1200, 100), font_size=40, align = "left"):
    image = Image.new('RGB', image_size, color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    font_size = round(font_size*(image_size[1]/100))
    try:
        if os_name == "Darwin":
            font = ImageFont.truetype("Tahoma.ttf", font_size)
        elif os_name == "Linux":
            font = ImageFont.truetype("LiberationMono-Regular.ttf", font_size)
        else:
            font = ImageFont.truetype("arialbd.ttf", font_size)
    except IOError:
        font = ImageFont.load_default() 

    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    if align == "left":
        text_x = 50
    elif align == "center":
        text_x = (image_size[0] - text_width) / 2
    elif align == "right":
        text_x = (image_size[0] - text_width) - 50
    text_y = (image_size[1] - text_height) / 2

    draw.text((text_x, text_y), text, font=font, fill=(0, 0, 0))
    
    return image

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

class img_list_repeat:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "image6": ("IMAGE",),
                "image7": ("IMAGE",),
                "image8": ("IMAGE",),
                "image9": ("IMAGE",),
                "image10": ("IMAGE",),
            }
        }
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True, )
    CATEGORY = "📂 SDVN/🏞️ Image"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "list_img"

    def list_img(s, image1 = None, image2 = None, image3 = None, image4 = None, image5 = None, image6 = None, image7 = None, image8 = None, image9 = None, image10 = None):
        r = []
        for i in [image1, image2, image3, image4, image5, image6, image7, image8, image9, image10]:
            if i != None:
                if isinstance(i, list):
                    r += [*i]
                else:
                    r += [i]
        return (r,)

class img_repeat:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "repeat": ("INT", {"default":1,"min":1}),
                "image": ("IMAGE",),
            }
        }
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True, )
    CATEGORY = "📂 SDVN/🏞️ Image"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "list_img"

    def list_img(s, repeat, image):
        r = []
        for _ in range(repeat[0]):
            r += [*image]
        return (r,)

class load_img_from_list:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "index": ("INT", {"default":0,"min":0}),
                "image": ("IMAGE",),
            }
        }
    INPUT_IS_LIST = True
    # OUTPUT_IS_LIST = (True, )
    CATEGORY = "📂 SDVN/🏞️ Image"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "load_from_list"

    def load_from_list(s, index, image):
        return (image[index[0]],)
       
class image_layout:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mode": (["row","column","auto"],),
                "max_size": ("INT",{"default":1024,"min":0}),
                "label": ("STRING",),
                "font_size": ("INT",{"default":40,"min":0}),
                "align": (["left","center","right"],),
            },
            "optional": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "image6": ("IMAGE",),
            }
        }
    INPUT_IS_LIST = True
    CATEGORY = "📂 SDVN/🏞️ Image"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "layout"

    def layout(self, mode, max_size, label, align, font_size, image1 = None, image2 = None, image3 = None, image4 = None, image5 = None, image6 = None):
        list_img = []
        full_img = []
        mode = mode[0]
        max_size = max_size[0]
        align = align[0]
        font_size = font_size[0]
        for i in [image1, image2, image3, image4, image5, image6]:
            if i != None:
                if isinstance(i, list):
                    full_img += [*i]
                else:
                    full_img += [i]
        if mode != "auto":
            for i in full_img:
                samples = i.movedim(-1, 1)
                w = samples.shape[3]
                h = samples.shape[2]
                if mode == "row":
                    w = round(w * max_size / h)
                    h = max_size
                elif mode == "column":
                    h = round(h * max_size / w)
                    w = max_size
                i = ALL_NODE["ImageScale"]().upscale(i, "nearest-exact", w, h, "disabled")[0]
                list_img.append(i)
            if mode == "row":
                if len(label) == 1:
                    label = label[0].split(',')
                if len(label) > 1:
                    new_list = []
                    for index in range(len(list_img)):
                        try:
                            r_label = label[index].strip()
                        except:
                            r_label = " "
                        new_list += [self.layout(["row"], [max_size], [r_label], [align], [font_size], [list_img[index]])[0]]
                    list_img = new_list
                list_img = [tensor.squeeze(0) for tensor in list_img]
                img_layout = torch.cat(list_img, dim=1)
            elif mode == "column":
                list_img = [tensor.squeeze(0) for tensor in list_img]
                img_layout = torch.cat(list_img, dim=0)
            r = img_layout.unsqueeze(0)
        else:
            c = math.ceil(math.sqrt(len(full_img)))
            if len(full_img) % c <= c/2 and len(full_img) % c != 0:
                c = c + 1
            new_list = [full_img[i:i + c] for i in range(0, len(full_img), c)]
            for i in new_list:
                list_img += [self.layout(["row"], [max_size], [""], ["left"], [font_size], i)[0]]
            if len(list_img) >1:
                r = self.layout(["column"], [max_size], [""], ["left"], [font_size], list_img)[0]
            else:
                r = list_img[0]
        if ( mode != "row") or ( len(label) == 1 and mode == "row" and ',' not in label[0]):
            label = label[0]
            if label != "":
                samples = r.movedim(-1, 1)
                w = samples.shape[3]
                h = samples.shape[2]
                img_label = create_image_with_text(label, image_size=(w, round(50 * (h / 512))), font_size = font_size, align = align)
                img_label = i2tensor(img_label)
                list_img = [r, img_label]
                list_img = [tensor.squeeze(0) for tensor in list_img]
                img_layout = torch.cat(list_img, dim=0)
                r = img_layout.unsqueeze(0)
        return (r,)

class img_scraper:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "url": ("STRING", {"default":"","multiline": False}),
                "custom_save_folder": ("STRING",),
                "save_folder": (["input","output"],),
                "cookies": (["chrome","firefox"],),
            }
        }
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (True, )
    CATEGORY = "📂 SDVN/👨🏻‍💻 Dev"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "img_scraper"

    def img_scraper(s, url, custom_save_folder, save_folder, cookies):
        url = url.split('?')[0]
        if custom_save_folder == "":
            main_folder = folder_paths.get_input_directory() if save_folder == "input" else folder_paths.get_output_directory()
        else:
            main_folder = custom_save_folder
        sub_folder = url.split('//')[-1].split('/')[0].split('www.')[-1]
        folder = os.path.join(main_folder,sub_folder)
        command = ["gallery-dl", "--cookies-from-browser", cookies, "--directory", folder, url]
        subprocess.run(command, check=True,text=True, capture_output=True)
        result = ALL_NODE["SDVN Load Image Folder"]().load_image(folder, -1, False)[0]
        return (result,)
    
class ic_light_v2:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "bg_source": (['None', 'Left Light', 'Right Light', 'Top Light', 'Bottom Light'],{"default":"None"}),
                "prompt": ("STRING",{"default":"","multiline": True}),
                "n_prompt": ("STRING",{"default":"","multiline": False}),
                "image_size": ("INT", {"default":1024,"min":512,"max":2048}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, }),
                "steps": ("INT", {"default":25,"min":1,"max":50}),
            }
        }

    CATEGORY = "📂 SDVN/👨🏻‍💻 Dev"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "ic_light_v2"

    def ic_light_v2(s, image, bg_source, prompt, n_prompt, image_size, seed, steps):
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
        image.save("/tmp/ic_light.jpg", format="JPEG")
        client = Client("lllyasviel/iclight-v2")
        result = client.predict(
                input_fg = handle_file('/tmp/ic_light.jpg'),
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
        img_path = result[0][0]['image']
        img = ALL_NODE["SDVN Load Image Url"]().load_image_url(img_path)["result"][0]
        return (img,)
    
NODE_CLASS_MAPPINGS = {
    "SDVM Image List Repeat": img_list_repeat,
    "SDVN Image Repeat": img_repeat,
    "SDVN Image Layout": image_layout,
    "SDVN Load Image From List": load_img_from_list,
    "SDVN Image Scraper": img_scraper,
    "SDVN IC Light v2": ic_light_v2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SDVN Image Layout": "🪄 Image Layout",
    "SDVM Image List Repeat": "🔄 Image List",
    "SDVN Image Repeat": "🔄 Image Repeat",
    "SDVN Load Image From List": "📁 Image From List",
    "SDVN Image Scraper": "⏬️ Image Scraper",
    "SDVN IC Light v2": "✨ IC Light v2",
}

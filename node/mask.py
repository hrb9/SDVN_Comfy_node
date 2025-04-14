from nodes import NODE_CLASS_MAPPINGS as ALL_NODE
import folder_paths
import os, numpy as np, torch
from PIL import Image
from ultralytics import YOLO
from torch.hub import download_url_to_file
import torch.nn.functional as FF
from rembg import remove

yolo_model_list = ["face_yolov8n-seg2_60.pt", "face_yolov8m-seg_60.pt",
                    "skin_yolov8n-seg_800.pt", "skin_yolov8n-seg_400.pt", "skin_yolov8m-seg_400.pt",
                    "yolov8_butterfly_custom.pt", "yolo-human-parse-epoch-125.pt", "yolo-human-parse-v2.pt",
                    "hair_yolov8n-seg_60.pt", "flowers_seg_yolov8model.pt", "facial_features_yolo8x-seg.pt", "Anime-yolov8-seg.pt",
                    "yolov8x-seg.pt", "yolov8s-seg.pt", "yolov8n-seg.pt", "yolov8m-seg.pt", "yolov8l-seg.pt",
                    "yolo11s-seg.pt", "yolo11n-seg.pt", "yolo11m-seg.pt", "yolo11l-seg.pt", "yolo11x-seg.pt",
                    ]

base_url = "https://huggingface.co/StableDiffusionVN/yolo/resolve/main/"

def yolo_segment(model, image, threshold, classes):
    image_tensor = image
    image_np = image_tensor.cpu().numpy() 
    image = Image.fromarray(
        (image_np.squeeze(0) * 255).astype(np.uint8)
    )
    results = model(image, classes=classes, conf=threshold)

    im_array = results[0].plot()  
    im = Image.fromarray(im_array[..., ::-1])  

    image_tensor_out = torch.tensor(
        np.array(im).astype(np.float32) / 255.0
    )  
    image_tensor_out = torch.unsqueeze(image_tensor_out, 0)

    res_mask=[]

    for result in results:
        masks = result.masks.data
        res_mask.append(torch.sum(masks, dim=0))
    return (image_tensor_out, res_mask)

labelName = {
    0: "person / Hair",  
    1: "bicycle / Face",  
    2: "car / Neck", 
    3: "motorcycle / Arm",  
    4: "airplane / Hand", 
    5: "bus / Back",  
    6: "train / Leg", 
    7: "truck / Foot",  
    8: "boat / Outfit", 
    9: "traffic light / Person",  
    10: "fire hydrant / Phone",  
    11: "stop sign", 
    12: "parking meter", 
    13: "bench", 
    14: "bird",  
    15: "cat",  
    16: "dog",  
    17: "horse",  
    18: "sheep",  
    19: "cow",  
    20: "elephant", 
    21: "bear",  
    22: "zebra",  
    23: "giraffe",  
    24: "backpack",  
    25: "umbrella",  
    26: "handbag",  
    27: "tie",  
    28: "suitcase",  
    29: "frisbee",  
    30: "skis",  
    31: "snowboard",  
    32: "sports ball",  
    33: "kite",  
    34: "baseball bat",  
    35: "baseball glove",  
    36: "skateboard",  
    37: "surfboard", 
    38: "tennis racket",  
    39: "bottle",  
    40: "wine glass",  
    41: "cup",  
    42: "fork",  
    43: "knife",  
    44: "spoon",  
    45: "bowl",  
    46: "banana",  
    47: "apple",  
    48: "sandwich",  
    49: "orange",  
    50: "broccoli",  
    51: "carrot", 
    52: "hot dog", 
    53: "pizza",  
    54: "donut",  
    55: "cake",  
    56: "chair",  
    57: "couch",  
    58: "potted plant",  
    59: "bed",  
    60: "dining table",  
    61: "toilet",  
    62: "tv", 
    63: "laptop",  
    64: "mouse",  
    65: "remote",  
    66: "keyboard",  
    67: "cell phone",  
    68: "microwave",  
    69: "oven", 
    70: "toaster", 
    71: "sink", 
    72: "refrigerator",  
    73: "book",
    74: "clock", 
    75: "vase",  
    76: "scissors", 
    77: "teddy bear",  
    78: "hair drier",  
    79: "toothbrush",
}

def label_dict(labelName):
    label_dict = {}
    for key, value in labelName.items():
        v = f"{key} - {value}"
        label_dict[v]= key
    return label_dict

class yoloseg:

    yolo_dir = os.path.join(folder_paths.models_dir, "yolo")
    file_list = []
    if os.path.exists(yolo_dir):
        for file in os.listdir(yolo_dir):
            file_full_path = os.path.join(yolo_dir, file)
            if os.path.isfile(file_full_path):
                type_name = file.split('.')[-1].lower()
                if type_name in ["pt"]:
                    file_list.append(file)
    model_list = list(set(yolo_model_list + file_list))
    model_list.sort()
    label_dict = label_dict(labelName)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (s.model_list, {"default": "face_yolov8n-seg2_60.pt"}),
                "image": ("IMAGE",),
                "detect": (["all", "choose", "id"], {"default": "all"}),
                "label": (list(s.label_dict.keys()), {"default": "0 - person / Hair"}),
                "label_id": ("STRING", {"default": "0,1,2"}),
                "threshold": ("FLOAT", {"default": 0.25, "min": 0.01, "max": 1.0, "step": 0.01},),
            },
        }
    
    CATEGORY = "📂 SDVN/🎭 Mask"
    FUNCTION = "yoloseg"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")

    def yoloseg(s, model_name, image, detect, label, label_id, threshold):
        model_folder = s.yolo_dir
        model_path = os.path.join(model_folder, model_name)
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        if not os.path.exists(model_path):
            url = base_url + model_name
            download_url_to_file(url, model_path)
        model = YOLO(model_path)
        if detect == "all":
            classes = []
            for i in range(80):
                classes.append(i)
        elif detect == "choose":
            classes = [s.label_dict[label]]
        elif detect == "id":
            classes = []
            ids = label_id.split(",")
            for id in ids:
                id = int(id.strip())
                if id in s.label_dict.values():
                    classes.append(id)

        res_images = []
        res_masks = []
        for item in image:
            if len(item.shape) == 3:
                item = item.unsqueeze(0)  
            item = item.to(model.device)

            image_out,  masks = yolo_segment(model, item, threshold, classes)

            resized_masks = []
            for mask_tensor in masks:
                resized_mask = torch.nn.functional.interpolate(
                    mask_tensor.unsqueeze(0).unsqueeze(0),
                    size=(image_out.shape[1], image_out.shape[2]),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0).squeeze(0)
                resized_masks.append(resized_mask)
            masks = resized_masks
            
            res_images.append(image_out)
            res_masks.extend(masks)
        yolo_image = torch.cat(res_images, dim=0)
        mask = torch.stack(res_masks, dim=0)
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        mask = mask.squeeze(1)
        invert_mask = (1.0 - mask).to(image.device)
        alpha_image = ALL_NODE["JoinImageWithAlpha"]().join_image_with_alpha(image, invert_mask)[0]
        ui = ALL_NODE["PreviewImage"]().save_images(alpha_image)["ui"]
        return {"ui":ui, "result": (yolo_image, mask)}

class MaskRegions:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",), 
            }
        }

    CATEGORY = "📂 SDVN/🎭 Mask"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("layer_mask",)  
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "separate_regions"

    @staticmethod
    def get_top_left_coords(tensor):
        coords = (tensor > 0).nonzero(as_tuple=False)
        if coords.numel() == 0:
            return (99999, 99999) 
        _, y, x = coords.min(dim=0).values
        return (x.item(), y.item()) 
    
    def separate_regions(s,mask):
        threshold=0.3
        max_iter=100

        device = mask.device
        mask_bin = (mask > threshold).float()  
        mask = mask_bin.clone()
        
        regions = []
        kernel = torch.tensor([[0., 1., 0.],
                            [1., 1., 1.],
                            [0., 1., 0.]], device=device).reshape(1, 1, 3, 3)

        while mask.sum() > 0 and len(regions) < max_iter:
            coords = (mask > 0).nonzero(as_tuple=False)[0]
            y, x = coords[1], coords[2]
            
            seed = torch.zeros_like(mask)
            seed[0, y, x] = 1.0
            
            region = seed.clone()
            prev = torch.zeros_like(region)

            while not torch.equal(region, prev):
                prev = region
                region = FF.conv2d(region.unsqueeze(0), kernel, padding=1)[0]
                region = (region > 0).float() * mask 

            regions.append(region.clone())

            mask = mask * (region == 0).float()
        
        regions_sorted = sorted(regions, key=s.get_top_left_coords)
        
        return (regions_sorted,)

class inpaint_crop:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "crop_size": ([512,768,896,1024,1280], {"default": 768}),
                "padding_blur": ("INT", {"default": 16, "min": 0, "max": 100}),
            },
        }

    CATEGORY = "📂 SDVN/🎭 Mask"
    RETURN_TYPES = ("STITCH", "IMAGE", "MASK")
    RETURN_NAMES = ("stitch", "cropped_image", "cropped_mask")
    FUNCTION = "inpaint"

    def inpaint_crop(self, image, mask, crop_size, padding_blur):
        if "InpaintCrop" not in ALL_NODE:
            raise Exception("Install node InpaintCrop(https://github.com/lquesada/ComfyUI-Inpaint-CropAndStitch)")
        input = ALL_NODE["InpaintCrop"]().inpaint_crop(image, mask, padding_blur, 1.0, True, padding_blur, False, padding_blur, "ranged size", "bicubic", 1024, 1024, 1.00, 0, crop_size - 128, crop_size - 128, crop_size, crop_size, None)
        input[0]["mask"] = mask
        input[0]["crop_size"] = crop_size
        input[0]["padding"] = padding_blur
        return input
    
    def inpaint (s, image, mask, crop_size, padding_blur):
        result = s.inpaint_crop(image, mask, crop_size, padding_blur)
        image = result[1]
        mask = result[2]
        invert_mask = 1.0 - mask
        alpha_image = ALL_NODE["JoinImageWithAlpha"]().join_image_with_alpha(image, invert_mask)[0]
        ui = ALL_NODE["PreviewImage"]().save_images(alpha_image)["ui"]
        return {"ui":ui, "result": result}
    
class LoopInpaintStitch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "stitchs": ("STITCH",),
                "inpainted_images": ("IMAGE",),
            }
        }

    CATEGORY = "📂 SDVN/🎭 Mask"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "inpaint_stitch"
    INPUT_IS_LIST = True

    def inpaint_stitch(self, stitchs, inpainted_images):
        canva = stitchs[0]['original_image']
        index = 0
        for inpainted_image in inpainted_images:
            stitch = stitchs[index]
            print(f'Vòng lặp {index}')
            stitch['original_image'] = canva
            del stitch["mask"]
            del stitch["crop_size"]
            del stitch["padding"]
            image = ALL_NODE["InpaintStitch"]().inpaint_stitch(stitchs[index], inpainted_image,  "bislerp")[0]
            index += 1
            if index < len(inpainted_images):
                canva = ALL_NODE["SDVN Inpaint Crop"]().inpaint_crop(image, stitchs[index]["mask"], stitchs[index]["crop_size"], stitchs[index]["padding"])[0]["original_image"]
        return (image,)

class rmbg:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    CATEGORY = "📂 SDVN/🎭 Mask"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "remove_background"

    def remove_background(s, image):
        """
        Xoá nền ảnh tensor dạng (B, H, W, C), trả về tensor cùng shape đã được xoá nền.
        """
        if image.dim() != 4 or image.shape[-1] != 3:
            raise ValueError("Input phải có shape (B, H, W, 3)")

        result = []
        for img in image:
            img_np = (img.cpu().numpy() * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np)
            img_no_bg = remove(img_pil)
            # img_no_bg = img_no_bg.convert("RGB")
            img_tensor = torch.from_numpy(np.array(img_no_bg).astype(np.float32) / 255.0)
            result.append(img_tensor)
        r_img = torch.stack(result, dim=0)
        ui = ALL_NODE["PreviewImage"]().save_images(r_img)["ui"]
        return {"ui":ui, "result": (r_img, r_img[:, :, :, 3])}
    
NODE_CLASS_MAPPINGS = {
    "SDVN Yolo8 Seg": yoloseg,
    "SDVN Mask Regions": MaskRegions,
    "SDVN Inpaint Crop": inpaint_crop,
    "SDVN Loop Inpaint Stitch": LoopInpaintStitch,
    "SDVN Remove Background": rmbg,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SDVN Yolo8 Seg": "🎭 Yolo Seg Mask",
    "SDVN Mask Regions": "🧩 Mask Regions",
    "SDVN Inpaint Crop": "⚡️ Crop Inpaint",
    "SDVN Loop Inpaint Stitch": "🔄 Loop Inpaint Stitch",
    "SDVN Remove Background": "🧼 Remove Background",
}
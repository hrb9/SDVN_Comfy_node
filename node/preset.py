from nodes import NODE_CLASS_MAPPINGS as ALL_NODE
import folder_paths

class AnyType(str):
    """A special class that is always equal in not equal comparisons. Credit to pythongosssss"""

    def __eq__(self, _) -> bool:
        return True

    def __ne__(self, __value: object) -> bool:
        return False


any = AnyType("*")

class quick_menu:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "Checkpoint":(["None",*folder_paths.get_filename_list("checkpoints")],),
                "Lora":(["None",*folder_paths.get_filename_list("loras")],),
                "Lora2":(["None",*folder_paths.get_filename_list("loras")],),
                "Lora3":(["None",*folder_paths.get_filename_list("loras")],),
                "Lora4":(["None",*folder_paths.get_filename_list("loras")],),
                "Lora5":(["None",*folder_paths.get_filename_list("loras")],),
                "SimpleString": ("STRING", {"default": "", "multiline": False},),
                "SimpleString2": ("STRING", {"default": "", "multiline": False},),
                "String": ("STRING", {"default": "", "multiline": True},),
                "String2": ("STRING", {"default": "", "multiline": True},),
            }}
    
    CATEGORY = "📂 SDVN/👨🏻‍💻 Dev"
    RETURN_TYPES = (any, any, any, any, any, any, any, any, any, any)
    RETURN_NAMES = ("checkpoint name", "lora name", "lora name 2", "lora name 3", "lora name 4", "lora name 5", "simple string", "simple string 2", "string", "string 2")
    FUNCTION = "quick_menu"

    def quick_menu(s, Checkpoint, Lora, Lora2, Lora3, Lora4, Lora5, SimpleString, SimpleString2, String, String2):
        return (Checkpoint, Lora, Lora2, Lora3, Lora4, Lora5, SimpleString, SimpleString2, String, String2,)

class auto_generate:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "Checkpoint":(folder_paths.get_filename_list("checkpoints"),),
                "Lora":(["None",*folder_paths.get_filename_list("loras")],),
                "Lora2":(["None",*folder_paths.get_filename_list("loras")],),
                "Lora3":(["None",*folder_paths.get_filename_list("loras")],),
                "Lora4":(["None",*folder_paths.get_filename_list("loras")],),
                "Lora5":(["None",*folder_paths.get_filename_list("loras")],),
                "Lora_Strength": ("STRING", {"default": "1,1,1,1,1", "multiline": False},),
                "Prompt": ("STRING", {"default": "", "multiline": True},),
                "Active_prompt": ("STRING", {"default": "", "multiline": False},),
                "Image_size": ("STRING", {"default": "1024,1024", "multiline": False},),
                "Steps": ("INT", {"default": 20,},),
                "Denoise": ("FLOAT", {"default": 1,"min":0,"max":1},),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "The random seed"}),
            },
            "optional": {
                "Image": ("IMAGE",),
                "Mask": ("MASK",)
            }
            }

    CATEGORY = "📂 SDVN/✨ Preset"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "auto_generate"

    model_para = {
        "Flux": [1152, "", 0.3],
        "SDXL": [1024, "XL-BasePrompt", 0.3],
        "SDXL Lightning": [1024, "XL-BasePrompt", 0.3],
        "SDXL Hyper": [1024, "XL-BasePrompt", 0.3],
        "SD 1.5": [768, "1.5-BasePrompt", 0.4],
    }
    def auto_generate(s, Checkpoint, Lora, Lora2, Lora3, Lora4, Lora5, Lora_Strength, Prompt, Active_prompt, Image_size, Steps, Denoise, seed, Image = None, Mask = None):
        model, clip, vae = ALL_NODE["CheckpointLoaderSimple"]().load_checkpoint(Checkpoint)
        list_lora = [Lora, Lora2, Lora3, Lora4, Lora5]
        Lora_Strength = ALL_NODE["SDVN Simple Any Input"]().simple_any(Lora_Strength)[0]
        for index in range(list_lora):
            model, clip = ALL_NODE["SDVN Load Lora"]().load_lora(False, "", "", list_lora[index], model, clip,  {Lora_Strength[index] if index <= len(Lora_Strength) else Lora_Strength[-1]}, 1)
        if "flux" in Checkpoint.lower():
            type_model = "Flux"
        elif "xl" in Checkpoint.lower():
            type_model = "SDXL"
        else:
            type_model = "SD 1.5"
        w, h = ALL_NODE["SDVN Simple Any Input"]().simple_any(Image_size)[0]
        Denoise = 1 if Image == None else Denoise
        max_size = s.model_para[type_model][0] / (1.5 * Denoise)
        if w > h:
            n_w = max_size if max_size > w else w
            n_h = h * (max_size/w) if max_size > w else h
        else:
            n_h = max_size if max_size > h else h
            n_w = w * (max_size/h) if max_size > h else w
        if Image == None:
            latent = ALL_NODE["EmptyLatentImage"]().generate(n_w, n_h, 1)[0]
        else:
            image = ALL_NODE["SDVN Upscale Image"]().upscale("Resize", n_w, n_h, 1, "None", Image)[0]
            latent = ALL_NODE["SDVN Inpaint"]().encode(False, image, vae, Mask, None, None)[2]
        Prompt = f"{Active_prompt}, {Prompt}"
        p, n, _ = ALL_NODE["SDVN CLIP Text Encode"]().encode(clip, Prompt, "", s.model_para[type_model][1], "en", seed)
        if type_model == "SDXL":
            for i in list_lora:
                if "lightning" in i.lower():
                    type_model = "SDXL Lightning"
                    break
                if "hyper-sdxl" in i.lower():
                    type_model = "SDXL Hyper" 
        latent_img, img = ALL_NODE["SDVN KSampler"]().sample(model, p, type_model, "Denoise", "euler", "normal", seed, Tiled=False, tile_width=None, tile_height=None, steps=Steps, cfg=7, denoise=1.0, negative=n, latent_image=latent, vae=vae, FluxGuidance = 3.5)
        if w <= n_w:
            return img
        else:
            latent = ALL_NODE["SDVN UPscale Latent"]().upscale("Resize", w, h, 1, "4x-UltraSharp.pth", latent_img, vae)[0]
            img = ALL_NODE["SDVN KSampler"]().sample(model, p, type_model, "Denoise", "euler", "normal", seed, Tiled=True, tile_width=w/2, tile_height=h/2, steps=Steps, cfg=7, denoise=s.model_para[type_model][2], negative=n, latent_image=latent, vae=vae, FluxGuidance = 3.5)
            return img
        
NODE_CLASS_MAPPINGS = {
    "SDVN Quick Menu": quick_menu,
    "SDVN Auto Generate": auto_generate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SDVN Quick Menu": "📋 Quick Menu",
    "SDVN Auto Generate": "💡 Auto Generate"
}
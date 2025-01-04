from nodes import NODE_CLASS_MAPPINGS as ALL_NODE
import folder_paths

class AnyType(str):
    """A special class that is always equal in not equal comparisons. Credit to pythongosssss"""

    def __eq__(self, _) -> bool:
        return True

    def __ne__(self, __value: object) -> bool:
        return False


any = AnyType("*")

def check_type_model(m):
    type_name = m.model.__class__.__name__
    type_name = "SD 1.5" if type_name == "BaseModel" else type_name
    return type_name
    
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

    def quick_menu(s, **kargs):
        r_list = [kargs[i] for i in kargs]
        return tuple(r_list)

class load_model:
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
            }
            }
    CATEGORY = "📂 SDVN/✨ Preset"
    RETURN_TYPES = ("MODEL", "CLIP", "VAE",)
    RETURN_NAMES = ("model", "clip", "vae")
    FUNCTION = "auto_generate"
    def auto_generate(s, Checkpoint, Lora, Lora2, Lora3, Lora4, Lora5, Lora_Strength):
        model, clip, vae = ALL_NODE["CheckpointLoaderSimple"]().load_checkpoint(Checkpoint)
        list_lora = [Lora, Lora2, Lora3, Lora4, Lora5]
        Lora_Strength = ALL_NODE["SDVN Simple Any Input"]().simple_any(Lora_Strength)[0]
        for index in range(len(list_lora)):
            if list_lora[index] != "None":
                try:
                    model, clip, _ = ALL_NODE["SDVN Load Lora"]().load_lora(False, "", "", list_lora[index], model, clip,  Lora_Strength[index] if index + 1 <= len(Lora_Strength) else Lora_Strength[-1], 1)["result"]
                except:
                    model, clip, _ = ALL_NODE["SDVN Load Lora"]().load_lora(False, "", "", list_lora[index], model, clip,  Lora_Strength[index] if index +1 <= len(Lora_Strength) else Lora_Strength[-1], 1)
        return (model, clip, vae,)
    
class join_parameter:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "any1": ("PARAMETER",),
                "any2": ("PARAMETER",),
                "any3": ("PARAMETER",),
                "any4": ("PARAMETER",),
                "any5": ("PARAMETER",),
                "any6": ("PARAMETER",),
                "any7": ("PARAMETER",),
                "any8": ("PARAMETER",),
                "any9": ("PARAMETER",),
                "any10": ("PARAMETER",),
            }
        }

    CATEGORY = "📂 SDVN/✨ Preset"
    RETURN_TYPES = ("PARAMETER",)
    RETURN_NAMES = ("parameter",)
    FUNCTION = "join_parameter"

    def join_parameter(s, any1 = None, any2 = None, any3 = None, any4 = None, any5 = None, any6 = None, any7 = None, any8 = None, any9 = None, any10 = None):
        r = []
        for i in [any1, any2, any3, any4, any5, any6, any7, any8, any9, any10]:
            if i != None:
                if isinstance(i, list):
                    r += [*i]
                else:
                    r += [i]
        return (r,)
class auto_generate:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model":("MODEL",),
                "clip":("CLIP",),
                "vae":("VAE",),
                "Prompt": ("STRING", {"default": "", "multiline": True},),
                "Negative": ("STRING", {"default": "", "multiline": True, "placeholder": "No support Flux model"},),
                "Active_prompt": ("STRING", {"default": "", "multiline": False},),
                "Image_size": ("STRING", {"default": "1024,1024", "multiline": False},),
                "Steps": ("INT", {"default": 20,},),
                "Denoise": ("FLOAT", {"default": 1,"min":0,"max":1},),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "The random seed"}),
            },
            "optional": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "parameter": ("PARAMETER",)
            }
            }
    CATEGORY = "📂 SDVN/✨ Preset"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "auto_generate"
    model_para = {
        "Flux": [1152, "None", 0.3, 1536],
        "SDXL": [1024, "XL-BasePrompt", 0.3, 1536],
        "SDXL Lightning": [1024, "XL-BasePrompt", 0.3, 1536],
        "SDXL Hyper": [1024, "XL-BasePrompt", 0.3, 1536],
        "SD 1.5": [768, "1.5-BasePrompt", 0.4, 1920],
        "None": [768, "1.5-BasePrompt", 0.4, 1920],
    }
    def auto_generate(s, model, clip, vae, Prompt, Negative, Active_prompt, Image_size, Steps, Denoise, seed, image = None, mask = None, parameter = None):
        type_model = check_type_model(model)
        type_model = "None" if type_model not in s.model_para else type_model
        print(f"Type model : {type_model}")
        if type_model == "SDXL" and Steps == 8:
            type_model = "SDXL Lightning"
        size = ALL_NODE["SDVN Simple Any Input"]().simple_any(Image_size)[0]
        if len(size) == 1:
            w = h = size[0]
        else:
            w, h = size[:2]

        if image != None:
            samples = image.movedim(-1, 1)
            i_w = samples.shape[3]
            i_h = samples.shape[2]
            if w/h > i_w/i_h:
                w = int(round(i_w * h / i_h))
            else:
                h = int(round(w * i_h / i_w))

        Denoise = 1 if image == None else Denoise
        max_size = s.model_para[type_model][0] / Denoise
        if w > h:
            n_w = max_size if max_size < w else w
            n_h = h * (max_size/w) if max_size < w else h
        else:
            n_h = max_size if max_size < h else h
            n_w = w * (max_size/h) if max_size < h else w
        n_h = int(round(n_h))
        n_w = int(round(n_w))
        if image == None:
            latent = ALL_NODE["EmptyLatentImage"]().generate(n_w, n_h, 1)[0]
        else:
            image = ALL_NODE["SDVN Upscale Image"]().upscale("Resize", n_w, n_h, 1, "None", image)[0]
            latent = ALL_NODE["SDVN Inpaint"]().encode(True, image, vae, mask, None, None)[2]
        Prompt = f"{Active_prompt}, {Prompt}"
        p, n, _ = ALL_NODE["SDVN CLIP Text Encode"]().encode(clip, Prompt, Negative, s.model_para[type_model][1], "en", seed)
        if parameter != None:
            if not isinstance(parameter, list):
                parameter = [parameter]
            for para in parameter:
                if "controlnet" in para:
                    p, n = ALL_NODE["SDVN Controlnet Apply"]().apply_controlnet(*para["controlnet"], vae=vae, positive = p, negative = n)["result"][:2]
                if "applystyle" in para:
                    p = ALL_NODE["SDVN Apply Style Model"]().applystyle(*para["applystyle"], p)[0]

        tile_size = s.model_para[type_model][3]
        _, img = ALL_NODE["SDVN KSampler"]().sample(model, p, type_model, "Denoise", "euler", "normal", seed, Tiled=True if (n_w > tile_size or n_h > tile_size) and Denoise < 0.5 else False, tile_width=int(round(n_w/2)), tile_height=int(round(n_h/2)), steps=Steps, cfg=7, denoise=Denoise, negative=n, latent_image=latent, vae=vae, FluxGuidance = 3.5)
        if w == n_w:
            return (img,)
        else:
            try:
                upscale_model = folder_paths.get_filename_list("upscale_models")[-1]
            except:
                upscale_model = "None"
            print(f"Upscale by {upscale_model}")
            img = ALL_NODE["SDVN Upscale Image"]().upscale("Resize", w, h, 1, upscale_model, img)[0]
            latent = ALL_NODE["SDVN Inpaint"]().encode(True, img, vae, mask, None, None)[2]
            img = ALL_NODE["SDVN KSampler"]().sample(model, p, type_model, "Denoise", "euler", "normal", seed,  Tiled=True if (n_w > tile_size or n_h > tile_size) else False, tile_width=int(round(w/2)), tile_height=int(round(h/2)), steps=Steps, cfg=7, denoise=s.model_para[type_model][2], negative=n, latent_image=latent, vae=vae, FluxGuidance = 3.5)[1]
            return (img,)
        
                
NODE_CLASS_MAPPINGS = {
    "SDVN Quick Menu": quick_menu,
    "SDVN Auto Generate": auto_generate,
    "SDVN Join Parameter": join_parameter,
    "SDVN Load Model": load_model,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SDVN Quick Menu": "📋 Quick Menu",
    "SDVN Auto Generate": "💡 Auto Generate",
    "SDVN Join Parameter": "🔄 Join Parameter",
    "SDVN Load Model": "💿 Load Model"
}
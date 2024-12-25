from nodes import NODE_CLASS_MAPPINGS as ALL_NODE
from googletrans import Translator, LANGUAGES
import torch, os
import folder_paths

def check_mask(mask_tensor):
    if not isinstance(mask_tensor, torch.Tensor):
        return False
    if mask_tensor.dtype != torch.float32:
        return False
    if mask_tensor.ndim != 3 or mask_tensor.size(0) != 1:
        return False
    if not (0.0 <= mask_tensor.min() and mask_tensor.max() <= 1.0):
        return False
    return True

def check_img(input_tensor):
    if not isinstance(input_tensor, torch.Tensor):
        return False
    if input_tensor.ndim == 4 and input_tensor.size(0) == 1 and input_tensor.size(-1) == 3:
        return True
    return False

def lang_list():
    lang_list = ["None"]
    for i in LANGUAGES.items():
        lang_list += [i[1]]
    return lang_list

class AnyType(str):
    """A special class that is always equal in not equal comparisons. Credit to pythongosssss"""

    def __eq__(self, _) -> bool:
        return True

    def __ne__(self, __value: object) -> bool:
        return False


any = AnyType("*")


class Easy_IPA_weight:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "SDXL": ("BOOLEAN", {"default": False},),
            "Weight": ("STRING", {"default": "0:1,1:1,1,1,4-15:1", "multiline": False, }),
        }
        }

    CATEGORY = "📂 SDVN/💡 Creative"

    RETURN_TYPES = ("STRING",)
    OUTPUT_TOOLTIPS = (
        "Ex: 0-4:1,6:1,1,1 or 0-15:1 or 1,1,1,1,1 or 1:1,5:1,7:1",)
    FUNCTION = "convert_wright"

    def convert_wright(self, SDXL, Weight):
        max_block = 10 if SDXL else 15
        Weight = Weight.split(",")
        index = 0
        convert = []
        for i in range(len(Weight)):
            if ':' not in Weight[i]:
                convert += [f'{str(index)}:{Weight[i]}'] if index <= max_block else []
                index += 1
            elif '-' in Weight[i]:
                ran, num = Weight[i].split(':')
                min, max = ran.split('-')
                index = int(min)
                for j in range(int(min), int(max)+1):
                    convert += [f'{str(index)}:{num}'] if index <= max_block else []
                    index += 1
            else:
                convert += [Weight[i]] if index <= max_block else []
                index += 1
        final_weight = ",".join(convert)
        print(f'Block weight: [{final_weight}]')
        return (final_weight,)

class GGTranslate:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "text": ("STRING", {"default": "", "multiline": True, }),
            "translate": (lang_list(),),
        }}

    CATEGORY = "📂 SDVN/💡 Creative"
    RETURN_TYPES = ("STRING",)
    FUNCTION = "ggtranslate"

    def ggtranslate(self, text, translate):
        if translate == "None" or text == "":
            output = text
        else:
            output = Translator().translate(text, translate, 'auto').text 
            print(f'Translate: "{output}"')
        return (output,)
    
class AnyInput:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "input": ("STRING", {"default": "","placeholder": "Ex: (in1+in2)/in3; in1 in2, in3; or every", "multiline": True, }),
            "output_list": (["None", "keywork", "line"], {"default": "None"}),
            "translate": (lang_list(),),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "The random seed"}),
        },
                "optional": {
                    "in1":(any,),
                    "in2":(any,),
                    "in3":(any,),
                    "in4":(any,),
                }
        }

    # INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True, True, True, True)
    CATEGORY = "📂 SDVN/💡 Creative"
    RETURN_TYPES = ("STRING", "FLOAT", "INT", "BOOLEAN")
    FUNCTION = "any_return"

    def any_return(self, input, output_list, translate, seed, in1 = None, in2 = None, in3 = None, in4 = None):
        in_list = {"in1":in1,"in2":in2,"in3":in3,"in4":in4}
        for i in in_list:
            if in_list[i] !=None and i in input:
                input = input.replace(i,str(in_list[i]))
        if "DPRandomGenerator" in ALL_NODE:
            cls = ALL_NODE["DPRandomGenerator"]
            input = cls().get_prompt(input, seed, 'No')[0]
        input = GGTranslate().ggtranslate(input,translate)[0]
        true_values = ["true",  "1", "yes", "y", "on"]
        input = input.lower()
        if output_list == "None":
            input = [input]
        elif output_list == "keywork":
            input = input.split(',')
        elif output_list == "line":
            input = input.splitlines()
        f = [*input]
        print(f)
        i = [*input]
        b = [*input]
        for x in f:
            try:
                f[f.index(x)] = float(eval(x))
            except:
                f[f.index(x)] = 0.0
        for x in i:
            try:
                i[i.index(x)] = int(eval(x))
            except:
                i[i.index(x)] = 0
        for x in b:
            if x in true_values:
                b[b.index(x)] = True
            else:
                b[b.index(x)] = False
        return (input, f, i, b,)

class SimpleAnyInput:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "input": ("STRING", {"default": "", "multiline": False, }),
        }}

    CATEGORY = "📂 SDVN/💡 Creative"
    RETURN_TYPES = (any,)
    RETURN_NAMES = ("any",)
    FUNCTION = "simple_any"

    def simple_any(s,input):
        try:
            r = int(eval(input))
        except:
            try:
                r = float(eval(input))
            except:
                if input.lower() in ["true",  "yes", "y", "on"]:
                    r = True
                elif input.lower() in ["false",  "no", "n", "off"]:
                    r = False
                else:
                    r = input
        return (r,)

class ImageSize:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "image": ("IMAGE",),
                "latent": ("LATENT",),
                "maxsize": ("INT", {"default": 0, "min": 0, "max": 10240, "tooltip": "0 = noset"}),
            }}
    CATEGORY = "📂 SDVN/💡 Creative"
    RETURN_TYPES = ("INT", "INT",)
    RETURN_NAMES = ("width", "height",)
    FUNCTION = "imagesize"

    def imagesize(s, image=None, latent=None, maxsize = 0):
        if image != None:
            samples = image.movedim(-1, 1)
            w = samples.shape[3]
            h = samples.shape[2]
        elif latent != None:
            w = latent["samples"].shape[-1] * 8
            h = latent["samples"].shape[-2] * 8
        else:
            w = h = 0
        if maxsize > 0:
            if w > h:
                h = int(round(h * ( maxsize / w)))
                w = maxsize
            else:
                w = int(round(w * ( maxsize / h)))
                h = maxsize
        print(f"Image width: {w} | Image height: {h}")
        return (w, h,)


class Seed:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, }),
            }}
    CATEGORY = "📂 SDVN/💡 Creative"
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("seed",)
    FUNCTION = "seed"

    def seed(s, seed=0):
        return (seed,)


class Switch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "true": (any,),
                "false": (any,),
                "target":  ("BOOLEAN", {"default": True},),
            }}
    CATEGORY = "📂 SDVN/💡 Creative"
    RETURN_TYPES = (any,)
    RETURN_NAMES = ("output",)
    FUNCTION = "switch"

    def switch(s, true, false, target):
        if target == True:
            return (true,)
        else:
            return (false,)

class Logic:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_true": (any,),
                "input_false": (any,),
                "a": (any,),
                "b": (any,),
                "logic":  (["a = b", "a != b", "a > b", "a < b", "a >= b", "a <= b", "a in b", "a not in b"],),
            }}
    CATEGORY = "📂 SDVN/💡 Creative"
    RETURN_TYPES = (any,)
    RETURN_NAMES = ("output",)
    FUNCTION = "logic"

    def logic(s, input_true, input_false, a, b, logic):
        if logic == "a = b":
            r = a == b
        elif logic == "a != b":
            r = a != b
        elif logic == "a > b":
            r = a > b
        elif logic == "a < b":
            r = a < b
        elif logic == "a >= b":
            r = a >= b
        elif logic == "a <= b":
            r = a <= b
        elif logic == "a in b":
            r = a in b
        elif logic == " a not in b":
            r = a not in b
        return (input_true if r == True else input_false,)

class Boolean:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "a": (any,),
                "b": (any,),
                "logic":  (["a = b", "a != b", "a > b", "a < b", "a >= b", "a <= b", "a in b", "a not in b"],),
            }}
    CATEGORY = "📂 SDVN/💡 Creative"
    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("boolean",)
    FUNCTION = "logic"

    def logic(s, a, b, logic):
        if logic == "a = b":
            r = a == b
        elif logic == "a != b":
            r = a != b
        elif logic == "a > b":
            r = a > b
        elif logic == "a < b":
            r = a < b
        elif logic == "a >= b":
            r = a >= b
        elif logic == "a <= b":
            r = a <= b
        elif logic == "a in b":
            r = a in b
        elif logic == " a not in b":
            r = a not in b
        return (r,)

class AnyShow:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "any": (any, {"forceInput": True}),
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ()
    FUNCTION = "show"
    OUTPUT_NODE = True

    CATEGORY = "📂 SDVN/💡 Creative"

    def show(self, any):
        if check_img(any[0]):
            results = ALL_NODE["PreviewImage"]().save_images(any[0])
            return results
        elif check_mask(any[0]):
            i = ALL_NODE["MaskToImage"]().mask_to_image(any[0])[0]
            results = ALL_NODE["PreviewImage"]().save_images(i)
            return results
        else:
            return {"ui": {"text": any}}

class Runtest:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "any": (any, {"forceInput": True}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "run"
    OUTPUT_NODE = True

    CATEGORY = "📂 SDVN/💡 Creative"

    def run(self, any):
        return ()

class PipeIn:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "latent": ("LATENT",),
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "any": (any,),
                
            },
        }

    RETURN_TYPES = ("PIPEIN",)
    RETURN_NAMES = ("pipe-in",)
    FUNCTION = "pipein"

    CATEGORY = "📂 SDVN/💡 Creative"

    def pipein(self, model = None, clip = None, positive = None, negative = None, vae = None, latent = None, image = None, mask = None, any = None):
        pipe_in = {"model":model, "clip":clip, "positive":positive, "negative":negative, "vae":vae, "latent":latent, "image":image, "mask":mask, "any":any}
        return (pipe_in,)

class PipeOut:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipe_in": ("PIPEIN",),
                "type": (["model", "clip", "positive", "negative", "vae", "latent", "image", "mask", "any"],{}),
            },
        }

    RETURN_TYPES = (any,)
    RETURN_NAMES = ("pipe-out",)
    FUNCTION = "pipeout"

    CATEGORY = "📂 SDVN/💡 Creative"

    def pipeout(self, pipe_in, type):
        print(type)
        if type == "image":
            print("True")
        out = pipe_in[type]
        return (out,)
    
class PipeOutAll:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipe_in": ("PIPEIN",),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "CONDITIONING", "CONDITIONING", "VAE", "LATENT", "IMAGE", "MASK", any,)
    RETURN_NAMES = ("model", "clip", "positive", "negative", "vae", "latent", "image", "mask", "any")
    FUNCTION = "pipeout"

    CATEGORY = "📂 SDVN/💡 Creative"

    def pipeout(self, pipe_in):
        return (pipe_in["model"],pipe_in["clip"],pipe_in["positive"],pipe_in["negative"],pipe_in["vae"],pipe_in["latent"],pipe_in["image"],pipe_in["mask"],pipe_in["any"],)

def list_txt_path(file_path):
    list_txt = []
    if os.path.isfile(file_path):
        file_path = os.path.dirname(file_path)

    for file in os.listdir(file_path):
        file_full_path = os.path.join(file_path, file)
        if os.path.isdir(file_full_path):
            list_txt.extend(list_txt_path(file_full_path))
        elif os.path.isfile(file_full_path):
            type_name = file.split('.')[-1].lower()
            if type_name in ["txt"]:
                list_txt.append(file_full_path)
    return list_txt

def get_name_file(file_path):
    list_txt = list_txt_path(file_path)
    try:
        for i in list_txt:
            list_txt[list_txt.index(i)] = i.replace(f"{file_path}/","")
        return list_txt
    except:
        return ["None"]

class LoadTextFile:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        return {
            "required": {
                "custom_path": ("STRING",{"default":""}),
                "input_dir": (get_name_file(input_dir),),
                "mode": (["line","keyword","fullfile"],),
                "index": ("INT",{"default":0}),
                "auto_index": ("BOOLEAN", {"default": False, "label_on": "loop", "label_off": "off"},),
            },
            "optional": {
                "string": ("STRING",{"forceInput": True})
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    FUNCTION = "loadtxt"

    CATEGORY = "📂 SDVN/💡 Creative"

    def loadtxt(self, custom_path, input_dir, mode, index, auto_index, string = None):
        if string != None:
            content = string.strip()
        else:
            if custom_path != "":
                path = custom_path
            else:
                path = os.path.join(folder_paths.get_input_directory(), input_dir)
            with open(path, 'r') as file:
                content = file.read().strip()

        if mode == "fullfile":
            resulf = content
        else:
            if mode == "line":
                list_txt = content.splitlines()
            elif mode == "keyword":
                list_txt = content.split(",")
            index = index%len(list_txt) if auto_index else index
            resulf = list_txt[index].strip()
        return (resulf,)

class SaveTextFile:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "content": ("STRING",{"forceInput": True}),
                "save_name": ("STRING",{"default":"filename.txt"}),
                "save_dir": (["input","output"],{"default":"input"}),
                "custom_dir": ("STRING",{"default":""}),
                "mode": (["new_line","new_file","join_string","new_first_line","join_first_string"],),
            },
        }
    RETURN_TYPES = ()
    FUNCTION = "savetxt"
    OUTPUT_NODE = True
    CATEGORY = "📂 SDVN/💡 Creative"

    def savetxt(self, content, save_name, save_dir, custom_dir, mode):
        if content != "":
            if custom_dir != "":
                if os.path.isfile(custom_dir):
                    path = custom_dir.replace(custom_dir.split('.')[-1],'txt')
                else:
                    if not os.path.isdir(custom_dir):
                        os.mkdir(custom_dir)
                    path = os.path.join(custom_dir,save_name)
            else:
                list_dir = {
                    "input": folder_paths.get_input_directory(),
                    "output": folder_paths.get_output_directory()
                }
                path = os.path.join(list_dir[save_dir],save_name)
            if os.path.exists(path):
                with open(path, 'r') as file:
                    old_content = file.read()
            else:
                old_content = ""
            if mode == "new_line":
                new_content = f"{old_content}\n{content}"
            elif mode == "new_file":
                new_content = content
            elif mode == "join_string":
                new_content = f"{old_content}, {content}"
            elif mode == "new_first_line":
                new_content = f"{content}\n{old_content}"
            elif mode == "join_first_string":
                new_content = f"{content}, {old_content}"
            with open(path, 'w') as file:
                file.write(new_content)
        return ()

class any_list_repeat:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "any1": (any,),
                "any2": (any,),
                "any3": (any,),
                "any4": (any,),
                "any5": (any,),
                "any6": (any,),
                "any7": (any,),
                "any8": (any,),
                "any9": (any,),
                "any10": (any,),
            }
        }
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True, )
    CATEGORY = "📂 SDVN/💡 Creative"
    RETURN_TYPES = (any,)
    RETURN_NAMES = ("any",)
    FUNCTION = "list_any"

    def list_any(s, any1 = None, any2 = None, any3 = None, any4 = None, any5 = None, any6 = None, any7 = None, any8 = None, any9 = None, any10 = None):
        r = []
        for i in [any1, any2, any3, any4, any5, any6, any7, any8, any9, any10]:
            if i != None:
                if isinstance(i, list):
                    r += [*i]
                else:
                    r += [i]
        return (r,)

class any_repeat:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "repeat": ("INT", {"default":1,"min":1}),
                "any": (any,),
            }
        }

    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True, )
    CATEGORY = "📂 SDVN/💡 Creative"
    RETURN_TYPES = (any,)
    RETURN_NAMES = ("any",)
    FUNCTION = "list_any"

    def list_any(s, repeat, any):
        r = []
        for _ in range(repeat[0]):
            r += [*any]
        return (r,)

class load_any_from_list:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "index": ("INT", {"default":0,"min":0}),
                "any": (any,),
            }
        }
    INPUT_IS_LIST = True
    CATEGORY = "📂 SDVN/💡 Creative"
    RETURN_TYPES = (any,)
    RETURN_NAMES = ("any",)
    FUNCTION = "load_from_list"

    def load_from_list(s, index, any):
        return (any[index[0]],)

class filter_list:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "start": ("INT", {"default":0,"min":0}),
                "end": ("INT", {"default":0,"min":0}),
                "input": (any,),
            },
            "optional": {
                "boolean": ("BOOLEAN", {"forceInput": True})
            }
        }

    CATEGORY = "📂 SDVN/💡 Creative"
    RETURN_TYPES = (any,)
    RETURN_NAMES = ("any",)
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "filter_list"

    def filter_list(s, start, end, input, boolean = None):
        start = start[0]
        end = end[0]
        if start != 0:
            end = len(input) if end == 0 else end
        if end != 0 and start <= end:
            n_input = []
            for i in range(len(input)):
                if i in range(start-1,end):
                    if boolean != None:
                        try:
                            b = boolean[i]
                        except:
                            b = True
                    else:
                        b = True
                    if b == True:
                        n_input.append(input[i])
            input = [*n_input]
        if boolean != None:
            n_input = []
            for i in range(len(input)):
                try:
                    if boolean[i]:
                        n_input.append(input[i])
                except:
                    None
            input = [*n_input]
        return (input,)
        
NODE_CLASS_MAPPINGS = {
    "SDVN Easy IPAdapter weight": Easy_IPA_weight,
    "SDVN Any Input Type": AnyInput,
    "SDVN Simple Any Input": SimpleAnyInput,
    "SDVN Image Size": ImageSize,
    "SDVN Seed": Seed,
    "SDVN Switch": Switch,
    "SDVN Logic": Logic,
    "SDVN Boolean": Boolean,
    "SDVN Translate": GGTranslate,
    "SDVN Any Show": AnyShow,
    "SDVN Run Test": Runtest,
    "SDVN Pipe In": PipeIn,
    "SDVN Pipe Out": PipeOut,
    "SDVN Pipe Out All": PipeOutAll,
    "SDVN Load Text": LoadTextFile,
    "SDVN Save Text": SaveTextFile,
    "SDVN Any List":any_list_repeat,
    "SDVN Any Repeat":any_repeat,
    "SDVN Any From List":load_any_from_list,
    "SDVN Filter List": filter_list,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SDVN Easy IPAdapter weight": "📊 IPAdapter weight",
    "SDVN Any Input Type": "🔡 Any Input Type",
    "SDVN Image Size": "📐 Image Size",
    "SDVN Seed": "🔢 Seed",
    "SDVN Switch": "🔄 Switch",
    "SDVN Logic": "#️⃣ Logic Switch",
    "SDVN Boolean": "#️⃣ Boolean",
    "SDVN Translate": "🔃 Translate",
    "SDVN Any Show": "🔎 Any show",
    "SDVN Run Test": "⚡️ Run test",
    "SDVN Pipe In": "🪢 Pipe In",
    "SDVN Pipe Out": "🪢 Pipe Out",
    "SDVN Pipe Out All": "🪢 Pipe Out All",
    "SDVN Load Text": "💽 Load Text",
    "SDVN Save Text": "💽 Save Text",
    "SDVN Any List":"🔄 Any List",
    "SDVN Any Repeat":"🔄 Any Repeat",
    "SDVN Any From List":"📁 Any From List",
    "SDVN Filter List": "⚖️ Filter List",
    "SDVN Simple Any Input": "🔡 Simple Any Input",
}

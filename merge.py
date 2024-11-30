import comfy.sd
import comfy.utils
import comfy.model_base
import comfy.model_management
import comfy.model_sampling

import torch
import folder_paths
import json
import os

from comfy.cli_args import args


class ModelMergeBlocks:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model1": ("MODEL",),
                             "model2": ("MODEL",),
                             "input": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                             "middle": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                             "output": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01})
                             }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "merge"

    CATEGORY = "advanced/model_merging"

    def merge(self, model1, model2, **kwargs):
        m = model1.clone()
        kp = model2.get_key_patches("diffusion_model.")
        for i in kwargs:
            kwargs[i] = str(kwargs[i])
            kwargs[i] = kwargs[i].split(',')
        hargs = {}
        for i in kwargs:
            for j in range(len(kwargs[i])):
                if '-' in kwargs[i][j]:
                    ran, num = kwargs[i][j].split(':')
                    min, max = ran.split('-')
                    num = float(num)
                    for a in range(int(min), int(max)+1):
                        hargs[f'{i}.{a}'] = num
                else:
                    kwargs[i][j] = float(kwargs[i][j])
                    hargs[f'{i}.{j}'] = kwargs[i][j]
            hargs[i] = num if '-' in kwargs[i][0] else kwargs[i][j]
        default_ratio = next(iter(hargs.values()))
        for k in kp:
            ratio = default_ratio
            k_unet = k[len("diffusion_model."):]

            last_arg_size = 0
            for arg in hargs:
                if k_unet.startswith(arg) and last_arg_size < len(arg):
                    ratio = hargs[arg]
                    last_arg_size = len(arg)

            m.add_patches({k: kp[k]}, 1.0 - ratio, ratio)
        return (m, )


class ModelMergeSD1(ModelMergeBlocks):
    CATEGORY = "SDVN/Merge"

    @classmethod
    def INPUT_TYPES(s):

        return {"required": {
            "model1": ("MODEL",),
            "model2": ("MODEL",),
            "input_blocks": ("STRING", {"default": "0-6:1,7-11:1"},),
            "middle_block": ("STRING", {"default": "1"},),
            "output_blocks": ("STRING", {"default": "1,1,1,1,1,1,1,1,1,1,1,1"},)
        }
        }


class ModelMergeSDXL(ModelMergeBlocks):
    CATEGORY = "SDVN/Merge"

    @classmethod
    def INPUT_TYPES(s):

        return {"required": {
            "model1": ("MODEL",),
            "model2": ("MODEL",),
            "input_blocks": ("STRING", {"default": "0-4:1,5-8:1"},),
            "middle_block": ("STRING", {"default": "1"},),
            "output_blocks": ("STRING", {"default": "1,1,1,1,1,1,1,1,1"},)
        }
        }


class ModelMergeFlux1(ModelMergeBlocks):
    CATEGORY = "SDVN/Merge"

    @classmethod
    def INPUT_TYPES(s):
        arg_dict = {"model1": ("MODEL",),
                    "model2": ("MODEL",)}

        return {"required": {
            "model1": ("MODEL",),
            "model2": ("MODEL",),
            "double_blocks": ("STRING", {"default": "0-9:1,10-18:1"},),
            "single_blocks": ("STRING", {"default": "0-37:1"},),
        }
        }


# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "SDVN Merge SD1": ModelMergeSD1,
    "SDVN Merge SDXL": ModelMergeSDXL,
    "SDVN Merge Flux": ModelMergeFlux1,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "SDVN Merge SD1": "Merge SD1",
    "SDVN Merge SDXL": "Merge SDXL",
    "SDVN Merge Flux": "Merge Flux"
}

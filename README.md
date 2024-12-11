<div align="center">

# SDVN Comfy Node
**Smart node set, supporting easier and more convenient ways to use ComfyUI**


[![Website][website-shield]][website-url]
[![Dynamic JSON Badge][discord-shield]][discord-url]

[website-shield]: https://img.shields.io/badge/Website-stablediffusion.vn-0075ff
[website-url]: https://stablediffusion.vn/
[discord-shield]: https://img.shields.io/discord/813085864355037235?color=blue&label=Discord&logo=Discord
[discord-url]: https://discord.gg/5SEtApPeyG
![ComfyUI Screenshot](/preview/preview.png)
</div>

___
### [Installing](#Install)
### [Guide](#Guide)
- [✨ Base Node](#BaseNode)
- [🏞️ Image](#Image)
- [📥 Download](#Download)
- [🧬 Merge](#Merge)
- [💡 Creative](#Creative)
- [👨🏻‍💻 API](#API)
### [Example](#Example)

___

# Install

Install with simple commands: 
- `cd <ComfyUI folder path>/custom_nodes`
- `git clone https://github.com/StableDiffusionVN/SDVN_Comfy_node`

Also you should install the following nodes to be able to use all functions:
- [Dynamicprompts](https://github.com/adieyal/comfyui-dynamicprompts)
- [TiledDiffusion](https://github.com/shiimizu/ComfyUI-TiledDiffusion)
- [IPAdapter_plus](https://github.com/cubiq/ComfyUI_IPAdapter_plus)
- [Controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux)
___

# Guide

- **For all nodes with download:** Supports direct download from **civitai** and **huggingface** with model address link and model download link
- **For all dodes load photos with the URL:** Automatically download photos to the url of the image. Can automatically search for the highest quality image with the [Pinterest link](https://www.pinterest.com/) . See also the [support list](https://github.com/mikf/gallery-dl/blob/master/docs/supportedsites.md)
- **For all nodes capable of entering the text:** Support **Google Translate** and [**Dynamic Prompt function**](https://github.com/adieyal/sd-dynamic-prompts/blob/main/docs/SYNTAX.md) (Request installed node [Dynamicprompts](https://github.com/adieyal/comfyui-dynamicprompts))
___
### BaseNode
*A collection of smart nodes that replace basic tasks, helping users build processes smarter and faster*

![Base Nodes](/preview/base_node.png)

**📀 Load checkpoint / 🎨 Load Lora**

 Supports 2 methods of loading checkpoint and downloading checkpoint directly for use.
 - If you leave the Download_url information, checkpoint will be selected according to the Ckpt_name
 - If you enter the checkpoint download url and leave the Download - True option, the checkpoint will be downloaded to checkpoints/loras folder and named Ckpt_url_name

**🏞️ Load Image / 🏞️ Load Image Url**

- Support 2 images download methods from input folders and URL links.
- Support sub-folders in the input folder

**🔡 CLIP Text Encode**

- Simultaneously support both Positive, Negative
- Support Random ability with Dynamic Prompt (Request installed node [Dynamicprompts](https://github.com/adieyal/comfyui-dynamicprompts))
- Support Translate function

**🎚️ Controlnet Apply**

Provide full option to use ControlNet in a single node (Request installed node [Controlnet Aux](https://github.com/Fannovel16/comfyui_controlnet_aux))
- Can choose Controlnet Model, Preprocessor (Automatically detect Aux Preprocessor Aux Preprocessor + Add Invert Image option), Union Type
- Show preview pictures Preprocessor when running

**⏳ Ksampler**

Smart node with many quick options to support flexibly in many different cases, help minimize errors and more flexibility to use.
- Convert 2 options for **negative** and **latent** to optional.
  - Without Negative, an empty clip will be replaced, now the way to connect to Flux will be in accordance with its nature - is not to use Negative
  - Without Latent, an empty Latent image will be created according to the size of Tile Width and Tile Height
- **ModelType:** Automatically adjust **CFG, Sampler name, Scheduler** for different types of models (SD15, SDXL, SDXL lightning, SDXL hyper, Flux ...). Now it is not a headache when it is too much.
- **StepsType:** Automatically adjust Steps according to the model and by Denoise ( Steps = Standing Steps x Denoise). Helps optimize the accurate and fastest process
- **Tiled:** Automatically divide the tiled block by mixture of differenters to minimize GPU when running Ksampler, applied in case of large image size and baby denoise (Request installed node [TiledDiffusion](https://github.com/shiimizu/ComfyUI-TiledDiffusion)). In case there is no latent, size tile = tile_width/2, tile_height/2

**👨‍🎨 Inpaint**

A comprehensive Inpaint support, consolidated from 4 Node Vae Encode, Latent Noise Mask, Vae Encode (For Inpainting), InpaintModelCondinging
- Vae Encode: If Mask = None
- Vae Encode (For Inpainting): If Postive or Negative = None
- Latent Noise Mask: If SetLatentNoiseMask = True
- InpaintModelCondinging: If SetLatentNoiseMask = False, all Image, Vae, Postive, Negative

___

### Image
*Smart node set, support for handling imaging tasks*

![Base Nodes](/preview/image_node.png)

**↗️ Upscale Image**

Smart Resize and Scale image
- Maxsize mode: Automatically calculate and adjust the image size so as not to change the ratio and do not exceed the required size
- Resize mode: Automatically resize the required size
- Scale mode: Calculate the image size according to the *scale index
- Model_name options will use Model Upscale according to the option, helping to keep more details when upscale
  
**↗️ Upscale Latent**

Similar to Upscale Image, but will add Vae Decoder and Vae Encoder to process Latent images, helping the process more neat.

___

### Download

*The set of nodes supports downloading photos of models to the corresponding folder and directly used on Comfyui*
-  *Supports direct download from **civitai** and **huggingface** with model address link and model download link*
 
![Base Nodes](/preview/download_node.png)

___

### Merge
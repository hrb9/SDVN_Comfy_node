<div align="center">

# SDVN Comfy Node
**Smart node set, supporting easier and more convenient ways to use ComfyUI**


[![](https://img.shields.io/badge/Website-stablediffusion.vn-0075ff)](https://stablediffusion.vn) [![](https://img.shields.io/badge/Group-Stable%20Diffusion%20VN-0075ff)](https://www.facebook.com/groups/stablediffusion.vn) [![](https://img.shields.io/discord/813085864355037235?color=blue&label=Discord&logo=Discord)](https://discord.gg/5SEtApPeyG) 

![ComfyUI Screenshot](/preview/preview.png)
</div>

___
[**Installing**](#Install)

[**Guide**](#Guide)
- [✨ Base Node](#BaseNode)
- [🏞️ Image](#Image)
- [📥 Download](#Download)
- [🧬 Merge](#Merge)
- [💡 Creative](#Creative)
- [👨🏻‍💻 API](#API)

[**Example**](#Example)

___
# Todo
- [ ] Smart merge - save Lora, Checkpoint
- [ ] Workflow Example
- [ ] Hướng dẫn tiếng việt
- [x] Install
- [x] Guide

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

- Support 2 images download methods from input folders and URL links / Image Path
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
 
![Download Nodes](/preview/download_node.png)

___

### Merge

*Supports the smart and convenient way to adjust the Weight Model Block Model compared to the original nodes, inspiring more creativity. Refer more information at [SuperMerge](https://github.com/hako-mikan/sd-webui-supermerger), [Lora Block Weight](https://github.com/hako-mikan/sd-webui-lora-block-weight)*

![Merge Nodes](/preview/merge_node.png)

Support 3 types of syntax to adjust for each block
- The non -listed values ​​will take the last block value
- {Block}: {Weight Block}
  - Ex: SD15 has 12 blocks IN from 0-11 
    - `0:1, 1:1, 2:1, 3:1, 4:0, 5:1` <=> `0:1, 1:1, 2:1, 3:1, 4:0, 5:1, 6:1, 7:1, 8:1, 9:1, 10:1, 11:1`
    - `2:0, 3:1` <=> `0:1, 1:1, 2:0, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1, 9:1, 10:1, 11:1`
- {Weight Block}
  - Ex: SDXL has 9 blocks IN from 0-8
    - `0, 0, 0, 0, 1, 1`  <=> `0:0, 1:0, 2:0, 3:0, 4:1, 5:1, 6:1, 7:1, 8:1`
- {Range}: {Weight Block}
  - Ex: Flux has 19 double blocks from 0-18
    - `0-10:0, 11-18:1` <=> `0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:1, 12:1, 13:1, 14:1, 15:1, 16:1, 17:1, 18:1`
- Combining 3 types of syntax
  - Ex: SDXL has 9 blocks OUT from 0-8
    - `0-3:0, 1, 6:1, 0` <=> `0:0, 1:0, 2:0, 3:0, 4:1, 5:0, 6:1, 7:0, 8:0`

[*See more workflow examples*](#Example)

___

### Creative

The node set helps to create the process in a smart way
- 📊 IPAdapter weight: Use the same syntax as the merge
- 🔃 Translate, 🔡 Any Input Type: Support translate and Dynamic prompt
- 🔡 Any Input Type: Support Math, Boolean input value (yes-no, true-false, 1-2)

![Creative Nodes](/preview/creative_node.png)

[*See more workflow examples*](#Example)

___

### API

Support the use of AI models through API
- Support the default API setting through the file: `.../SDVN_Custom_node/API_key.json` (Rename API_key.json.example and fill API)
  - Get Gemini API: https://aistudio.google.com/app/apikey
  - Get HuggingFace API: https://huggingface.co/settings/tokens
  - Get OpenAI API (Chat GPT, Dall-E): https://platform.openai.com/settings/organization/api-keys

![API Nodes](/preview/api_node.png)

**💬 API Chatbot**
- Image: Suport Gemini, ChatGPT
- Preset: Add history and sample statements in cases of each other
- Support translate and Dynamic prompt

**🎨 DALL-E Generate Image**
- Support translate and Dynamic prompt
- Support size: 1024x1024, 1024x1792, 1792x1024
___

# Example

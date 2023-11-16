# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2023(18588220928@163.com), All Rights Reserved.
# ***
# ***    File Author: Dell, Wed 02 Aug 2023 06:43:47 AM CST
# ***
# ************************************************************************************/
#
import random
import gradio as gr
import numpy as np

import torch
import torch.nn.functional as F


from SDXL.model import (
    ImageCreator,
)

from SDXL.util import (
    vram_load,
    vram_unload,
    load_torch_image,
)

from SDXL.canny import (
    detect_edge,
)
from SDXL.tokenizer import (
    create_clip_token_model,
)


import todos
import pdb

# create models
model = ImageCreator() # create_sdxl_base_model(skip_lora=False, skip_vision=True)
clip_token = create_clip_token_model(version="base_1.0")


def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, time_steps, 
    cond_scale, denoise, seed, low_threshold, high_threshold):

    if input_image is not None:
        input_tensor = load_torch_image(input_image)
        input_tensor = F.interpolate(input_tensor, size=(image_resolution, image_resolution), 
            mode="bilinear", align_corners=False)
    else:
        input_tensor = torch.zeros(1, 3, 1024, 1024)

    guide_image = detect_edge(input_tensor, low_threshold, high_threshold)

    if seed == -1:
        seed = random.randint(0, 2147483647)    

    if len(a_prompt) > 0:
        prompt = prompt + "," + a_prompt
    positive_tokens = clip_token.encode(prompt)
    negative_tokens = clip_token.encode(n_prompt)

    with torch.no_grad():
        positive_tensor = model.clip_text(positive_tokens)
        negative_tensor = model.clip_text(negative_tokens)

    vram_load(model.vae_model)
    with torch.no_grad():
        latent_image = model.vae_model.encode(input_tensor) # torch.zeros(1, 3, 1024, 1024))
    vram_unload(model.vae_model)

    latent_image.fill_(0.0)
    # load canny lora model weight if needed ???    
    control_tensor = {}
    control_tensor['lora_weight'] = torch.tensor([1.0])
    control_tensor['lora_guide'] = guide_image

    vram_load(model.unet_model)
    vram_load(model.lora_model)
    with torch.no_grad():
        sample = model.euler_a_forward(positive_tensor, negative_tensor, cond_scale, latent_image, control_tensor,
            time_steps, denoise, seed)
    vram_unload(model.lora_model)
    vram_unload(model.unet_model)

    vram_load(model.vae_model)
    with torch.no_grad():
        latent_output = model.vae_model.decode(sample.cpu())
    vram_unload(model.vae_model)

    x_samples = (latent_output.movedim(1, -1) * 255.0).numpy().astype(np.uint8)

    canny_edge = (guide_image.movedim(1, -1).squeeze(0).numpy() * 255.0).astype(np.uint8)
    return [255 - canny_edge] + [x_samples[0]]


block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## SDXL 1.0 Base Controlnet Model Demo")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source='upload', type="numpy", label='canny')

            prompt = gr.Textbox(label="Prompt", value='bag for children')
            run_button = gr.Button(label="Run")
            with gr.Accordion("Advanced options", open=False):
                num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
                image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=1024, value=1024, step=64)
                denoise = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
                low_threshold = gr.Slider(label="Canny low threshold", minimum=0.0, maximum=1.0, value=0.1, step=0.01)
                high_threshold = gr.Slider(label="Canny high threshold", minimum=0.0, maximum=1.0, value=0.9, step=0.01)
                time_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=30, step=1)
                cond_scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=8.0, step=0.1)
                seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, value=42, step=1)
                # a_prompt = gr.Textbox(label="Added Prompt", value='best quality, extremely detailed')
                a_prompt = gr.Textbox(label="Added Prompt", value='')
                # n_prompt = gr.Textbox(label="Negative Prompt",
                #                       value='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality')
                n_prompt = gr.Textbox(label="Negative Prompt", value='black and white')

        with gr.Column():
            result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery", 
                columns=[2], rows=[2], object_fit="contain", height="auto")

    inputs = [input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, time_steps, 
              cond_scale, denoise, seed, low_threshold, high_threshold]
    run_button.click(fn=process, inputs=inputs, outputs=[result_gallery])

block.launch(server_name='0.0.0.0')

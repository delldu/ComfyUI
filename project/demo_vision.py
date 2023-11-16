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

from SDXL.tokenizer import (
    create_clip_token_model,
)
from SDXL.clip_vision import (
    CLIPVisionEncoder,
)

from SDXL.util import (
    vram_load, 
    vram_unload,
    load_torch_image,
)


import todos
import pdb

# beautiful scenery nature glass bottle landscape, purple galaxy bottle

# create models
model = ImageCreator()
clip_token = create_clip_token_model(version="base_1.0")
clip_vision = CLIPVisionEncoder() # CLIPVisionEncode

def process(prompt, a_prompt, n_prompt, input_image, cond_scale, image_resolution, time_steps, denoise, seed):
    # input_image.shape -- (600, 458, 3), dtype=uint8

    if input_image is not None:
        input_tensor = load_torch_image(input_image)
        input_tensor = F.interpolate(input_tensor, size=(image_resolution, image_resolution), 
            mode="bilinear", align_corners=False)
    else:
        input_tensor = torch.zeros(1, 3, 1024, 1024)

    if seed == -1:
        seed = random.randint(0, 2147483647)

    if len(a_prompt) > 0:
        prompt = prompt + "," + a_prompt
    positive_tokens = clip_token.encode(prompt)
    negative_tokens = clip_token.encode(n_prompt)

    with torch.no_grad():
        positive_tensor = model.clip_text(positive_tokens)
        # positive_tensor is dict:
        #     tensor [text_encoded] size: [1, 77, 2048], min: -809.318359, max: 853.72229, mean: 0.018757
        #     tensor [pool_encoded] size: [1, 1280], min: -4.416891, max: 3.616007, mean: 0.039642
        negative_tensor = model.clip_text(negative_tokens)
        # negative_tensor is dict:
        #     tensor [text_encoded] size: [1, 77, 2048], min: -809.318359, max: 853.72229, mean: 0.018757
        #     tensor [pool_encoded] size: [1, 1280], min: -4.416891, max: 3.616007, mean: 0.039642

    vram_load(model.vae_model)
    with torch.no_grad():
        latent_image = model.vae_model.encode(input_tensor)
        # tensor [latent_image] size: [1, 4, 128, 128], min: -23.92502, max: 10.286383, mean: -3.205484
    vram_unload(model.vae_model)

    positive_tensor['text_encoded'].fill_(0.0)
    positive_tensor['pool_encoded'].fill_(0.0)
    negative_tensor['text_encoded'].fill_(0.0)
    negative_tensor['pool_encoded'].fill_(0.0)
    latent_image.fill_(0.0)

    vram_load(clip_vision)
    with torch.no_grad():
        vision_embeds = clip_vision(load_torch_image(input_image)) # CLIPVisionEncode
    vram_unload(clip_vision)
    positive_tensor['pool_encoded'] = vision_embeds

    control_tensor = {}
    # for k, v in positive_tensor.items():
    #     positive_tensor[k] = v.cuda()
    # for k, v in negative_tensor.items():
    #     negative_tensor[k] = v.cuda()
    # latent_image = latent_image.cuda()

    vram_load(model.unet_model)
    vram_load(model.lora_model)
    with torch.no_grad():
        sample = model.dpm2_a_forward(positive_tensor, negative_tensor, cond_scale, latent_image, control_tensor,
            time_steps, denoise, seed)
    vram_unload(model.unet_model)
    vram_unload(model.lora_model)

    vram_load(model.vae_model)
    with torch.no_grad():
        latent_output = model.vae_model.decode(sample.cpu()) # VAEDecode
    vram_unload(model.vae_model)

    x_samples = (latent_output.movedim(1, -1) * 255.0).numpy().astype(np.uint8)

    return [x_samples[0]]


block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## SDXL 1.0 Base Vision Model Demo")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source='upload', type="numpy", label='Source')

            prompt = gr.Textbox(label="Prompt", value="red bag, clean background, made from cloth")
            run_button = gr.Button(label="Run")
            with gr.Accordion("Advanced options", open=False):
                image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=1024, value=1024, step=64)
                denoise = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
                time_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=30, step=1)
                cond_scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=8.0, step=0.1)
                seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, value=42, step=1)
                # a_prompt = gr.Textbox(label="Added Prompt", value='best quality, extremely detailed')
                a_prompt = gr.Textbox(label="Added Prompt", value='')
                # n_prompt = gr.Textbox(label="Negative Prompt",
                #                       value='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality')
                n_prompt = gr.Textbox(label="Negative Prompt", value='watermark, text')

        with gr.Column():
            result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery", 
                columns=[2], rows=[2], object_fit="contain", height="auto")

    # positive_tensor, negative_tensor, latent_image, cond_scale=7.5, steps=20, denoise=1.0, seed=-1
    inputs = [prompt, a_prompt, n_prompt, input_image, cond_scale, image_resolution, time_steps, denoise, seed]
    run_button.click(fn=process, inputs=inputs, outputs=[result_gallery])

block.launch(server_name='0.0.0.0')

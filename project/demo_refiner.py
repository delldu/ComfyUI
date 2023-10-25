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
import einops

import SDXL
from SDXL.util import (
    load_torch_image,        
)

from SDXL.ksampler import (
    create_sample_model,
)

from SDXL.vae import (
    create_vae_encode_model,
    create_vae_decode_model,
)

from SDXL.clip import (
    create_clip_text_model,
    create_clip_token_model,
)

import todos
import pdb


# create models
sample_mode = create_sample_model()
vae_encode = create_vae_encode_model()
vae_decode = create_vae_decode_model()
clip_token = create_clip_token_model(version="refiner_1.0")
clip_text = create_clip_text_model(version="refiner_1.0")


# inputs = [prompt, a_prompt, n_prompt, input_image, cond_scale, time_steps, denoise, seed]
def process(prompt, a_prompt, n_prompt, input_image, cond_scale, time_steps, denoise, seed):
    # input_image.shape -- (600, 458, 3), dtype=uint8

    positive_tokens = clip_token.encode(prompt + "," + a_prompt)
    negative_tokens = clip_token.encode(n_prompt)

    with torch.no_grad():
        positive_tensor = clip_text(positive_tokens)
        negative_tensor = clip_text(negative_tokens)

        if input_image is not None:
            latent_image = vae_encode(load_torch_image(input_image))
        else:
            latent_image = torch.zeros([1, 4, 128, 128])

    for k, v in positive_tensor.items():
        positive_tensor[k] = v.cuda()
    for k, v in negative_tensor.items():
        negative_tensor[k] = v.cuda()
    latent_image = latent_image.cuda()

    with torch.no_grad():
        sample = sample_mode(positive_tensor, negative_tensor, latent_image, cond_scale, time_steps, denoise, seed)
        latent_output = vae_decode(sample.cpu())

    x_samples = (einops.rearrange(latent_output, 'b c h w -> b h w c') * 255.0).numpy().clip(0, 255).astype(np.uint8)

    return [x_samples[0]]


block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## SDXL Refiner Model (Version 1.0) Demo")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source='upload', type="numpy", label='Source')

            prompt = gr.Textbox(label="Prompt")
            run_button = gr.Button(label="Run")
            with gr.Accordion("Advanced options", open=False):
                denoise = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
                time_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
                cond_scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=8.0, step=0.1)
                seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, value=42, step=1)
                a_prompt = gr.Textbox(label="Added Prompt", value='best quality, extremely detailed')
                n_prompt = gr.Textbox(label="Negative Prompt",
                                      value='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality')
        with gr.Column():
            result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery", 
                columns=[2], rows=[2], object_fit="contain", height="auto")

    # positive_tensor, negative_tensor, latent_image, cond_scale=7.5, steps=20, denoise=1.0, seed=-1
    inputs = [prompt, a_prompt, n_prompt, input_image, cond_scale, time_steps, denoise, seed]
    run_button.click(fn=process, inputs=inputs, outputs=[result_gallery])

block.launch(server_name='0.0.0.0')

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

from SDXL import (
    create_sdxl_base_model,
)

from SDXL.util import (
    load_torch_image,
)

import todos
import pdb

# beautiful scenery nature glass bottle landscape, purple galaxy bottle

# create models
model = create_sdxl_base_model()
sample_mode = model.sample_mode
vae_encode = model.vae_encode
vae_decode = model.vae_decode
clip_token = model.clip_token
clip_text = model.clip_text
clip_vision = model.clip_vision



def process(prompt, a_prompt, n_prompt, input_image, cond_scale, time_steps, denoise, seed):
    # input_image.shape -- (600, 458, 3), dtype=uint8

    if seed == -1:
        seed = random.randint(0, 2147483647)

    if len(a_prompt) > 0:
        prompt = prompt + "," + a_prompt
    positive_tokens = clip_token.encode(prompt)
    negative_tokens = clip_token.encode(n_prompt)

    with torch.no_grad():
        positive_tensor = clip_text(positive_tokens)
        # positive_tensor is dict:
        #     tensor [text_encoded] size: [1, 77, 2048], min: -809.318359, max: 853.72229, mean: 0.018757
        #     tensor [pool_encoded] size: [1, 1280], min: -4.416891, max: 3.616007, mean: 0.039642
        negative_tensor = clip_text(negative_tokens)
        # negative_tensor is dict:
        #     tensor [text_encoded] size: [1, 77, 2048], min: -809.318359, max: 853.72229, mean: 0.018757
        #     tensor [pool_encoded] size: [1, 1280], min: -4.416891, max: 3.616007, mean: 0.039642
        latent_image = vae_encode(torch.zeros(1, 3, 1024, 1024))
        # ==> size: [1, 4, 128, 128], min: -7.33046, max: 9.45929, mean: -0.472631

    positive_tensor['text_encoded'].fill_(0.0)
    positive_tensor['pool_encoded'].fill_(0.0)
    negative_tensor['text_encoded'].fill_(0.0)
    negative_tensor['pool_encoded'].fill_(0.0)
    latent_image.fill_(0.0)

    with torch.no_grad():
        clip_embeds = clip_vision(load_torch_image(input_image)) # CLIPVisionEncode
    positive_tensor['clip_embeds'] = clip_embeds

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
        gr.Markdown("## SDXL 1.0 Vision Model Demo")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source='upload', type="numpy", label='Source')

            prompt = gr.Textbox(label="Prompt", value="red bag, clean background, made from cloth")
            run_button = gr.Button(label="Run")
            with gr.Accordion("Advanced options", open=False):
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
    inputs = [prompt, a_prompt, n_prompt, input_image, cond_scale, time_steps, denoise, seed]
    run_button.click(fn=process, inputs=inputs, outputs=[result_gallery])

block.launch(server_name='0.0.0.0')

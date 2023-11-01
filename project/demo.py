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

import cv2
import SDXL
# from SDXL.util import (
# )

from SDXL import (
    create_sdxl_base_model,
)

from SDXL.util import (
    load_torch_image,
)

import todos
import pdb

# create models
model = create_sdxl_base_model()
sample_mode = model.sample_mode
vae_encode = model.vae_encode
vae_decode = model.vae_decode
clip_token = model.clip_token
clip_text = model.clip_text
clip_vision = model.clip_vision

def ImageResize(image):
    MAX_HW = 1024
    H, W, C = image.shape

    if H > MAX_HW or W > MAX_HW:
        s = min(MAX_H / H, MAX_W / W)
        H, W = int(s * H), int(s * W)

    NH, NW = (H // 8) * 8, (W // 8) * 8
    if NH != H or NW != W:
        return cv2.resize(image, (NW, NH))

    return image

def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y


class CannyDetector:
    def __call__(self, img, low_threshold, high_threshold):
        return cv2.Canny(img, low_threshold, high_threshold)


guide_generator = CannyDetector()
# model, device = SDXL.get_model('v1.0')

def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, time_steps, 
    cond_scale, denoise, seed, low_threshold, high_threshold):

    if input_image is not None:
        input_image = ImageResize(input_image)

    if seed == -1:
        seed = random.randint(0, 2147483647)    

    if len(a_prompt) > 0:
        prompt = prompt + "," + a_prompt
    positive_tokens = clip_token.encode(prompt)
    negative_tokens = clip_token.encode(n_prompt)

    with torch.no_grad():
        positive_tensor = clip_text(positive_tokens)
        negative_tensor = clip_text(negative_tokens)
        latent_image = vae_encode(load_torch_image(input_image)) # torch.zeros(1, 3, 1024, 1024))

    # OK ==> positive_tokens['l'] -- [49406, 3365, 556, 2153, 49407]
    # OK ==> positive_tokens['g'] -- [49406, 3365, 556, 2153, 49407]
    todos.debug.output_var("positive_tensor['text_encoded']", positive_tensor['text_encoded'])
    # BAD ==> tensor [positive_tensor['text_encoded']] size: [1, 77, 2048], min: -66.179535, max: 33.065098, mean: -0.02555
    # OK ==> tensor [positive_tensor['pool_encoded']] size: [1, 1280], min: -3.322949, max: 5.934691, mean: 0.019366

    # text -- 'bag for children\n'
    # tokens['l'] -- [[(49406, 1.0), (3365, 1.0), (556, 1.0), (2153, 1.0), (49407, 1.0)]]
    # tokens['g'] -- [[(49406, 1.0), (3365, 1.0), (556, 1.0), (2153, 1.0), (49407, 1.0)]]
    # tensor [cond] size: [1, 77, 2048], min: -809.318359, max: 853.722839, mean: 0.02496
    # tensor [pooled] size: [1, 1280], min: -3.32294, max: 5.93469, mean: 0.019366


    # -------------------------------------------------------------------------------------------
    # OK ==> negative_tokens['l'] -- [49406, 1449, 537, 1579, 49407]
    # OK ==> negative_tokens['g'] -- [49406, 1449, 537, 1579, 49407]
    todos.debug.output_var("negative_tensor['text_encoded']", negative_tensor['text_encoded'])
    # BAD ==> tensor [negative_tensor['text_encoded']] size: [1, 77, 2048], min: -66.179535, max: 33.065098, mean: -0.021825
    # OK ==> tensor [negative_tensor['pool_encoded']] size: [1, 1280], min: -5.141021, max: 5.103258, mean: -9.5e-05

    # 'black and white\n'
    # tokens['l'] -- [[(49406, 1.0), (1449, 1.0), (537, 1.0), (1579, 1.0), (49407, 1.0)]]
    # tokens['g'] -- [[(49406, 1.0), (1449, 1.0), (537, 1.0), (1579, 1.0), (49407, 1.0)]]
    # tensor [cond] size: [1, 77, 2048], min: -809.318359, max: 853.722839, mean: 0.023284
    # tensor [pooled] size: [1, 1280], min: -5.141018, max: 5.103264, mean: -9.5e-05    

    # pdb.set_trace() # xxxx9999

    if input_image is not None:
        guide_image = HWC3(guide_generator(input_image, low_threshold, high_threshold))
    else:
        guide_image = np.zeros((image_resolution, image_resolution, 3)).astype(np.uint8)
    positive_tensor['lora_guide'] = load_torch_image(guide_image)

    for k, v in positive_tensor.items():
        positive_tensor[k] = v.half().cuda()
    for k, v in negative_tensor.items():
        negative_tensor[k] = v.half().cuda()
    latent_image = latent_image.half().cuda()

    with torch.no_grad():
        sample = sample_mode(positive_tensor, negative_tensor, latent_image, cond_scale, time_steps, denoise, seed)
        latent_output = vae_decode(sample.cpu())

    # latent_output = (latent_output + 1.0)/2.0
    tensor_min = latent_output.min()
    tensor_max = latent_output.max()
    latent_output = (latent_output - tensor_min)/(tensor_max - tensor_min + 1e-5)

    x_samples = (einops.rearrange(latent_output, 'b c h w -> b h w c') * 255.0).numpy().clip(0, 255).astype(np.uint8)


    return [255 - guide_image] + [x_samples[0]]


block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## SDXL Base Model (Version 1.0) Demo")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source='upload', type="numpy", label='canny')

            prompt = gr.Textbox(label="Prompt", value='bag for children')
            run_button = gr.Button(label="Run")
            with gr.Accordion("Advanced options", open=False):
                num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
                image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=1024, value=1024, step=64)
                denoise = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
                low_threshold = gr.Slider(label="Canny low threshold", minimum=1, maximum=255, value=25, step=1)
                high_threshold = gr.Slider(label="Canny high threshold", minimum=1, maximum=255, value=230, step=1)
                time_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
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
